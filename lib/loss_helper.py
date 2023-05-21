# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from utils.nn_distance import nn_distance, huber_loss
from lib.ap_helper import parse_predictions
from lib.loss import SoftmaxRankingLoss, SigmoidRankingLoss, SigmoidFocalClassificationLoss, smoothl1_loss, l1_loss
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8] # put larger weights on positive objectness


# KPS loss
def compute_points_obj_cls_loss_hard_topk(end_points, topk):
    box_label_mask = end_points['box_label_mask']
    seed_inds = end_points['seed_inds'].long()  # B, K
    seed_xyz = end_points['seed_xyz']  # B, K, 3
    seeds_obj_cls_logits = end_points['seeds_obj_cls_logits']  # B, 1, K
    gt_center = end_points['center_label'][:, :, 0:3]  # B, K2, 3
    gt_size = end_points['size_gts'][:, :, 0:3]  # B, K2, 3
    B = gt_center.shape[0]
    K = seed_xyz.shape[1]
    K2 = gt_center.shape[1]

    point_instance_label = end_points['point_instance_label']  # B, num_points
    object_assignment = torch.gather(point_instance_label, 1, seed_inds)  # B, num_seed
    object_assignment[object_assignment < 0] = K2 - 1  # set background points to the last gt bbox
    object_assignment_one_hot = torch.zeros((B, K, K2)).to(seed_xyz.device)
    object_assignment_one_hot.scatter_(2, object_assignment.unsqueeze(-1), 1)  # (B, K, K2)
    delta_xyz = seed_xyz.unsqueeze(2) - gt_center.unsqueeze(1)  # (B, K, K2, 3)
    delta_xyz = delta_xyz / (gt_size.unsqueeze(1) + 1e-6)  # (B, K, K2, 3)
    new_dist = torch.sum(delta_xyz ** 2, dim=-1)
    euclidean_dist1 = torch.sqrt(new_dist + 1e-6)  # BxKxK2
    euclidean_dist1 = euclidean_dist1 * object_assignment_one_hot + 100 * (1 - object_assignment_one_hot)  # BxKxK2
    euclidean_dist1 = euclidean_dist1.transpose(1, 2).contiguous()  # BxK2xK
    topk_inds = torch.topk(euclidean_dist1, topk, largest=False)[1] * box_label_mask[:, :, None] + \
                (box_label_mask[:, :, None] - 1)  # BxK2xtopk
    topk_inds = topk_inds.long()  # BxK2xtopk
    topk_inds = topk_inds.view(B, -1).contiguous()  # B, K2xtopk
    batch_inds = torch.arange(B).unsqueeze(1).repeat(1, K2 * topk).to(seed_xyz.device)
    batch_topk_inds = torch.stack([batch_inds, topk_inds], -1).view(-1, 2).contiguous()

    objectness_label = torch.zeros((B, K + 1), dtype=torch.long).to(seed_xyz.device)
    objectness_label[batch_topk_inds[:, 0], batch_topk_inds[:, 1]] = 1
    objectness_label = objectness_label[:, :K]
    objectness_label_mask = torch.gather(point_instance_label, 1, seed_inds)  # B, num_seed
    objectness_label[objectness_label_mask < 0] = 0

    total_num_points = B * K
    end_points[f'points_hard_topk{topk}_pos_ratio'] = \
        torch.sum(objectness_label.float()) / float(total_num_points)
    end_points[f'points_hard_topk{topk}_neg_ratio'] = 1 - end_points[f'points_hard_topk{topk}_pos_ratio']

    # Compute objectness loss
    criterion = SigmoidFocalClassificationLoss()
    cls_weights = (objectness_label >= 0).float()
    cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
    cls_weights /= torch.clamp(cls_normalizer, min=1.0)
    cls_loss_src = criterion(seeds_obj_cls_logits.view(B, K, 1), objectness_label.unsqueeze(-1), weights=cls_weights)
    objectness_loss = cls_loss_src.sum() / B

    # Compute recall upper bound
    padding_array = torch.arange(0, B).to(point_instance_label.device) * 10000
    padding_array = padding_array.unsqueeze(1)  # B,1
    point_instance_label_mask = (point_instance_label < 0)  # B,num_points
    point_instance_label = point_instance_label + padding_array  # B,num_points
    point_instance_label[point_instance_label_mask] = -1
    num_gt_bboxes = torch.unique(point_instance_label).shape[0] - 1
    seed_instance_label = torch.gather(point_instance_label, 1, seed_inds)  # B,num_seed
    pos_points_instance_label = seed_instance_label * objectness_label + (objectness_label - 1)
    num_query_bboxes = torch.unique(pos_points_instance_label).shape[0] - 1
    if num_gt_bboxes > 0:
        end_points[f'points_hard_topk{topk}_upper_recall_ratio'] = num_query_bboxes / num_gt_bboxes

    return objectness_loss

def compute_objectness_loss_based_on_query_points(end_points, num_decoder_layers):
    """ Compute objectness loss for the proposals.
    """

    if num_decoder_layers > 0:
        prefixes = ['proposal_'] + ['last_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)]
    else:
        prefixes = ['proposal_']  # only proposal

    objectness_loss_sum = 0.0
    for prefix in prefixes:
        # Associate proposal and GT objects
        seed_inds = end_points['seed_inds'].long()  # B,num_seed in [0,num_points-1]
        gt_center = end_points['center_label'][:, :, 0:3]  # B, K2, 3
        
        query_points_sample_inds = end_points['query_points_sample_inds'].long()

        B = seed_inds.shape[0]
        K = query_points_sample_inds.shape[1]
        K2 = gt_center.shape[1]

        seed_obj_gt = torch.gather(end_points['point_obj_mask'], 1, seed_inds)  # B,num_seed
        query_points_obj_gt = torch.gather(seed_obj_gt, 1, query_points_sample_inds)  # B, query_points

        point_instance_label = end_points['point_instance_label']  # B, num_points
        seed_instance_label = torch.gather(point_instance_label, 1, seed_inds)  # B,num_seed
        query_points_instance_label = torch.gather(seed_instance_label, 1, query_points_sample_inds)  # B,query_points

        objectness_mask = torch.ones((B, K)).cuda()

        # Set assignment
        object_assignment = query_points_instance_label  # (B,K) with values in 0,1,...,K2-1
        object_assignment[object_assignment < 0] = K2 - 1  # set background points to the last gt bbox

        end_points[f'{prefix}objectness_label'] = query_points_obj_gt
        end_points[f'{prefix}objectness_mask'] = objectness_mask
        end_points[f'{prefix}object_assignment'] = object_assignment
        total_num_proposal = query_points_obj_gt.shape[0] * query_points_obj_gt.shape[1]
        end_points[f'{prefix}pos_ratio'] = \
            torch.sum(query_points_obj_gt.float().cuda()) / float(total_num_proposal)
        end_points[f'{prefix}neg_ratio'] = \
            torch.sum(objectness_mask.float()) / float(total_num_proposal) - end_points[f'{prefix}pos_ratio']

        # Compute objectness loss
        objectness_scores = end_points[f'{prefix}objectness_scores']
        criterion = SigmoidFocalClassificationLoss()
        cls_weights = objectness_mask.float()
        cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
        cls_weights /= torch.clamp(cls_normalizer, min=1.0)

        cls_loss_src = criterion(objectness_scores.transpose(2, 1).contiguous().view(B, K, 1),
                                 query_points_obj_gt.unsqueeze(-1),
                                 weights=cls_weights)
        objectness_loss = cls_loss_src.sum() / B

        end_points[f'{prefix}objectness_loss'] = objectness_loss
        objectness_loss_sum += objectness_loss
    
    return objectness_loss_sum, end_points

def compute_objectness_loss(data_dict):
    """ Compute objectness loss for the proposals.

    Args:
        data_dict: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """ 
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = data_dict['aggregated_vote_xyz']
    gt_center = data_dict['center_label'][:,:,0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = data_dict['objectness_scores']
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment

def compute_box_and_sem_cls_loss_groupfree(end_points, config, num_decoder_layers,
                                 center_loss_type='smoothl1', center_delta=1.0,
                                 size_loss_type='smoothl1', size_delta=1.0,
                                 heading_loss_type='smoothl1', heading_delta=1.0,
                                 size_cls_agnostic=False):
    """ Compute 3D bounding box and semantic classification loss.
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    if num_decoder_layers > 0:
        prefixes = ['proposal_'] + ['last_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)]
    else:
        prefixes = ['proposal_']  # only proposal
    box_loss_sum = 0.0
    sem_cls_loss_sum = 0.0
    for prefix in prefixes:
        object_assignment = end_points[f'{prefix}object_assignment']
        batch_size = object_assignment.shape[0]
        # Compute center loss
        pred_center = end_points[f'{prefix}center']
        gt_center = end_points['center_label'][:, :, 0:3]

        if center_loss_type == 'smoothl1':
            objectness_label = end_points[f'{prefix}objectness_label'].float()
            object_assignment_expand = object_assignment.unsqueeze(2).repeat(1, 1, 3)
            assigned_gt_center = torch.gather(gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
            center_loss = smoothl1_loss(assigned_gt_center - pred_center, delta=center_delta)  # (B,K)
            center_loss = torch.sum(center_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)
        elif center_loss_type == 'l1':
            objectness_label = end_points[f'{prefix}objectness_label'].float()
            object_assignment_expand = object_assignment.unsqueeze(2).repeat(1, 1, 3)
            assigned_gt_center = torch.gather(gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
            center_loss = l1_loss(assigned_gt_center - pred_center)  # (B,K)
            center_loss = torch.sum(center_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)
        else:
            raise NotImplementedError

        # Compute heading loss
        heading_class_label = torch.gather(end_points['heading_class_label'], 1,
                                           object_assignment)  # select (B,K) from (B,K2)
        criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
        heading_class_loss = criterion_heading_class(end_points[f'{prefix}heading_scores'].transpose(2, 1),
                                                     heading_class_label)  # (B,K)
        heading_class_loss = torch.sum(heading_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        heading_residual_label = torch.gather(end_points['heading_residual_label'], 1,
                                              object_assignment)  # select (B,K) from (B,K2)
        heading_residual_normalized_label = heading_residual_label / (np.pi / num_heading_bin)

        # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
        heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1],
                                                       num_heading_bin).zero_()
        heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1),
                                       1)  # src==1 so it's *one-hot* (B,K,num_heading_bin)
        heading_residual_normalized_error = torch.sum(
            end_points[f'{prefix}heading_residuals_normalized'] * heading_label_one_hot,
            -1) - heading_residual_normalized_label

        if heading_loss_type == 'smoothl1':
            heading_residual_normalized_loss = heading_delta * smoothl1_loss(heading_residual_normalized_error,
                                                                             delta=heading_delta)  # (B,K)
            heading_residual_normalized_loss = torch.sum(
                heading_residual_normalized_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        elif heading_loss_type == 'l1':
            heading_residual_normalized_loss = l1_loss(heading_residual_normalized_error)  # (B,K)
            heading_residual_normalized_loss = torch.sum(
                heading_residual_normalized_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        else:
            raise NotImplementedError

        # Compute size loss
        if size_cls_agnostic:
            pred_size = end_points[f'{prefix}pred_size']
            size_label = torch.gather(
                end_points['size_gts'], 1,
                object_assignment.unsqueeze(-1).repeat(1, 1, 3))  # select (B,K,3) from (B,K2,3)
            size_error = pred_size - size_label
            if size_loss_type == 'smoothl1':
                size_loss = size_delta * smoothl1_loss(size_error,
                                                       delta=size_delta)  # (B,K,3) -> (B,K)
                size_loss = torch.sum(size_loss * objectness_label.unsqueeze(2)) / (
                        torch.sum(objectness_label) + 1e-6)
            elif size_loss_type == 'l1':
                size_loss = l1_loss(size_error)  # (B,K,3) -> (B,K)
                size_loss = torch.sum(size_loss * objectness_label.unsqueeze(2)) / (
                        torch.sum(objectness_label) + 1e-6)
            else:
                raise NotImplementedError
        else:
            size_class_label = torch.gather(end_points['size_class_label'], 1,
                                            object_assignment)  # select (B,K) from (B,K2)
            criterion_size_class = nn.CrossEntropyLoss(reduction='none')
            size_class_loss = criterion_size_class(end_points[f'{prefix}size_scores'].transpose(2, 1),
                                                   size_class_label)  # (B,K)
            size_class_loss = torch.sum(size_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

            size_residual_label = torch.gather(
                end_points['size_residual_label'], 1,
                object_assignment.unsqueeze(-1).repeat(1, 1, 3))  # select (B,K,3) from (B,K2,3)

            size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
            size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1),
                                        1)  # src==1 so it's *one-hot* (B,K,num_size_cluster)
            size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1, 1, 1, 3)  # (B,K,num_size_cluster,3)
            predicted_size_residual_normalized = torch.sum(
                end_points[f'{prefix}size_residuals_normalized'] * size_label_one_hot_tiled,
                2)  # (B,K,3)

            mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(
                0)  # (1,1,num_size_cluster,3)
            mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2)  # (B,K,3)
            size_residual_label_normalized = size_residual_label / mean_size_label  # (B,K,3)

            size_residual_normalized_error = predicted_size_residual_normalized - size_residual_label_normalized

            if size_loss_type == 'smoothl1':
                size_residual_normalized_loss = size_delta * smoothl1_loss(size_residual_normalized_error,
                                                                           delta=size_delta)  # (B,K,3) -> (B,K)
                size_residual_normalized_loss = torch.sum(
                    size_residual_normalized_loss * objectness_label.unsqueeze(2)) / (
                                                        torch.sum(objectness_label) + 1e-6)
            elif size_loss_type == 'l1':
                size_residual_normalized_loss = l1_loss(size_residual_normalized_error)  # (B,K,3) -> (B,K)
                size_residual_normalized_loss = torch.sum(
                    size_residual_normalized_loss * objectness_label.unsqueeze(2)) / (
                                                        torch.sum(objectness_label) + 1e-6)
            else:
                raise NotImplementedError

        # 3.4 Semantic cls loss
        sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment)  # select (B,K) from (B,K2)
        criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
        sem_cls_loss = criterion_sem_cls(end_points[f'{prefix}sem_cls_scores'].transpose(2, 1), sem_cls_label)  # (B,K)
        sem_cls_loss = torch.sum(sem_cls_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        end_points[f'{prefix}center_loss'] = center_loss
        end_points[f'{prefix}heading_cls_loss'] = heading_class_loss
        end_points[f'{prefix}heading_reg_loss'] = heading_residual_normalized_loss
        if size_cls_agnostic:
            end_points[f'{prefix}size_reg_loss'] = size_loss
            box_loss = center_loss + 0.1 * heading_class_loss + heading_residual_normalized_loss + size_loss
        else:
            end_points[f'{prefix}size_cls_loss'] = size_class_loss
            end_points[f'{prefix}size_reg_loss'] = size_residual_normalized_loss
            box_loss = center_loss + 0.1 * heading_class_loss + heading_residual_normalized_loss + 0.1 * size_class_loss + size_residual_normalized_loss
        end_points[f'{prefix}box_loss'] = box_loss
        end_points[f'{prefix}sem_cls_loss'] = sem_cls_loss

        box_loss_sum += box_loss
        sem_cls_loss_sum += sem_cls_loss
    return box_loss_sum, sem_cls_loss_sum, end_points


def compute_reference_loss(data_dict, config, no_reference=False, sent_aug=False):
    """ Compute cluster reference loss

    Args:
        data_dict: dict (read-only)

    Returns:
        ref_loss, lang_loss, cluster_preds, cluster_labels
    """

    # unpack
    # cluster_preds = data_dict["cluster_ref"] # (B, num_proposal)

    # predicted bbox
    # pred_ref = data_dict['cluster_ref'].detach().cpu().numpy() # (B,)
    pred_center = data_dict['center'].detach().cpu().numpy()  # (B,K,3)
    pred_heading_class = torch.argmax(data_dict['heading_scores'], -1)  # B,num_proposal
    pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2,
                                         pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    pred_heading_class = pred_heading_class.detach().cpu().numpy()  # B,num_proposal
    pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal
    pred_size_class = torch.argmax(data_dict['size_scores'], -1)  # B,num_proposal
    pred_size_residual = torch.gather(data_dict['size_residuals'], 2,
                                      pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
                                                                                         3))  # B,num_proposal,1,3
    pred_size_class = pred_size_class.detach().cpu().numpy()
    pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal,3


    gt_box_label_list = data_dict["ref_box_label_list"].cpu().numpy()
    gt_center_list = data_dict['ref_center_label_list'].cpu().numpy()  # (B,3)
    gt_heading_class_list = data_dict['ref_heading_class_label_list'].cpu().numpy()  # B
    gt_heading_residual_list = data_dict['ref_heading_residual_label_list'].cpu().numpy()  # B
    gt_size_class_list = data_dict['ref_size_class_label_list'].cpu().numpy()  # B
    gt_size_residual_list = data_dict['ref_size_residual_label_list'].cpu().numpy()  # B,3
    # convert gt bbox parameters to bbox corners

    batch_size, num_proposals = data_dict['query_features'].shape[0], data_dict['query_features'].shape[2]
    
    batch_size, len_nun_max = gt_center_list.shape[:2]

    lang_num = data_dict["lang_num"]
    max_iou_rate_25 = 0
    max_iou_rate_5 = 0

    if not no_reference:
        cluster_preds = data_dict["cluster_ref"].reshape(batch_size, len_nun_max, num_proposals)
        if 'cluster_ref_spa' in data_dict.keys():
            cluster_preds_vla = data_dict["cluster_ref_vla"].reshape(batch_size, len_nun_max, num_proposals)
            cluster_preds_spa = data_dict["cluster_ref_spa"].reshape(batch_size, len_nun_max, num_proposals)
    else:
        cluster_preds = torch.zeros(batch_size, len_nun_max, num_proposals).cuda()

    # print("cluster_preds",cluster_preds.shape)
    
    loss = 0.
    gt_labels = np.zeros((batch_size, len_nun_max, num_proposals))
    for i in range(batch_size):
        objectness_masks = data_dict['objectness_masks'].squeeze().detach().cpu().numpy()
        if not sent_aug:
            gt_obb_batch = config.param2obb_batch(gt_center_list[i][:, 0:3], gt_heading_class_list[i],
                                                gt_heading_residual_list[i],
                                                gt_size_class_list[i], gt_size_residual_list[i])
            gt_bbox_batch = get_3d_box_batch(gt_obb_batch[:, 3:6], gt_obb_batch[:, 6], gt_obb_batch[:, 0:3])
        labels = np.zeros((len_nun_max, num_proposals))
        for j in range(len_nun_max):
            if j < lang_num[i]:
                # convert the bbox parameters to bbox corners
                pred_obb_batch = config.param2obb_batch(pred_center[i, :, 0:3], pred_heading_class[i],
                                                        pred_heading_residual[i],
                                                        pred_size_class[i], pred_size_residual[i])
                pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])

                if sent_aug:
                    bbox_num = gt_box_label_list[i][j].sum()
                    bbox_idxes = []
                    for idx, flag in enumerate(gt_box_label_list[i][j].tolist()):
                        if flag == 1:
                            bbox_idxes.append(idx)
                    

                    gt_obb_batch = config.param2obb_batch(gt_center_list[i, j][:, 0:3], gt_heading_class_list[i, j],
                                                gt_heading_residual_list[i, j],
                                                gt_size_class_list[i, j], gt_size_residual_list[i, j])
                    gt_bbox_batch = get_3d_box_batch(gt_obb_batch[:, 3:6], gt_obb_batch[:, 6], gt_obb_batch[:, 0:3])

                    for bbox_idx in bbox_idxes:
                        gt_bbox = gt_bbox_batch[bbox_idx]        
                        ious = box3d_iou_batch(pred_bbox_batch, np.tile(gt_bbox, (num_proposals, 1, 1)))
                        if data_dict["istrain"][0] == 1 and not no_reference:
                            ious = ious * objectness_masks[i]
                        
                        ious_ind = ious.argmax()
                        max_ious = ious[ious_ind]
                        
                        if max_ious >= 0.25:
                            labels[j, ious.argmax()] = 1  # treat the bbox with highest iou score as the gt
                            max_iou_rate_25 += 1
                        if max_ious >= 0.5:
                            max_iou_rate_5 += 1
                    
                    if labels[j].sum()>1:
                        labels[j] = labels[j] / labels[j].sum()

                else:
                    ious = box3d_iou_batch(pred_bbox_batch, np.tile(gt_bbox_batch[j], (num_proposals, 1, 1)))
                    if data_dict["istrain"][0] == 1 and not no_reference:
                        ious = ious * objectness_masks[i]

                    ious_ind = ious.argmax()
                    max_ious = ious[ious_ind]
                    
                    if max_ious >= 0.25:
                        labels[j, ious.argmax()] = 1  # treat the bbox with highest iou score as the gt
                        max_iou_rate_25 += 1
                    if max_ious >= 0.5:
                        max_iou_rate_5 += 1

        if sent_aug:
            criterion = SoftmaxRankingLoss()
        else:
            criterion = SoftmaxRankingLoss()
        
        cluster_labels = torch.FloatTensor(labels).cuda()  # B proposals
        gt_labels[i] = labels
        # reference loss
        if 'cluster_ref_spa' in data_dict.keys():
            loss += (criterion(cluster_preds[i, :lang_num[i]], cluster_labels[:lang_num[i]].float().clone()) + \
                    criterion(cluster_preds_vla[i, :lang_num[i]], cluster_labels[:lang_num[i]].float().clone()) + \
                    criterion(cluster_preds_spa[i, :lang_num[i]], cluster_labels[:lang_num[i]].float().clone())) / 3
        else:
            loss += criterion(cluster_preds[i, :lang_num[i]], cluster_labels[:lang_num[i]].float().clone())

    data_dict['max_iou_rate_0.25'] = max_iou_rate_25 / sum(lang_num.cpu().numpy())
    data_dict['max_iou_rate_0.5'] = max_iou_rate_5 / sum(lang_num.cpu().numpy())


    cluster_labels = torch.FloatTensor(gt_labels).cuda()  # B len_nun_max proposals
    loss = loss / batch_size


    return data_dict, loss, cluster_preds, cluster_labels

def compute_lang_classification_loss(data_dict):
    criterion = torch.nn.CrossEntropyLoss()
    object_cat_list = data_dict["object_cat_list"]
    batch_size, len_nun_max = object_cat_list.shape[:2]
    lang_num = data_dict["lang_num"]

    lang_scores = data_dict["lang_scores"].reshape(batch_size, len_nun_max, -1)

    loss = 0.
    for i in range(batch_size):
        num = lang_num[i]
        loss += criterion(lang_scores[i, :num], object_cat_list[i, :num])
    loss = loss / batch_size
    return loss

def get_loss(data_dict, config, detection=True, reference=True, use_lang_classifier=False, num_decoder_layers=6, \
             query_points_obj_topk=5, detection_weight=1., sent_aug=False):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """


    # KPS query points generation loss
    if 'seeds_obj_cls_logits' in data_dict.keys():
        query_points_generation_loss = compute_points_obj_cls_loss_hard_topk(data_dict, query_points_obj_topk)

        data_dict['query_points_generation_loss'] = query_points_generation_loss
    else:
        query_points_generation_loss = 0.0
    

    # Obj loss
    objectness_loss_sum, data_dict = compute_objectness_loss_based_on_query_points(data_dict, num_decoder_layers)

    data_dict['sum_heads_objectness_loss'] = objectness_loss_sum


    # Box loss and sem cls loss
    box_loss_sum, sem_cls_loss_sum, data_dict = compute_box_and_sem_cls_loss_groupfree(
        data_dict, config, num_decoder_layers,
        center_loss_type='smoothl1', center_delta=0.04,
        size_loss_type='smoothl1', size_delta=0.111111111111,
        heading_loss_type='smoothl1', heading_delta=1.0,
        size_cls_agnostic=False)
    data_dict['sum_heads_box_loss'] = box_loss_sum
    data_dict['sum_heads_sem_cls_loss'] = sem_cls_loss_sum
    
    # Detection loss
    detection_loss = 0.8 * query_points_generation_loss + 1.0 / (num_decoder_layers + 1) * (
                0.1 * objectness_loss_sum + 1.0 * box_loss_sum + 0.1 * sem_cls_loss_sum)

    # Language classification loss
    data_dict["lang_loss"] = compute_lang_classification_loss(data_dict)
    
    # Reference loss
    data_dict, ref_loss, _, cluster_labels = compute_reference_loss(data_dict, config, sent_aug=sent_aug)
    data_dict["cluster_labels"] = cluster_labels
    data_dict["ref_loss"] = ref_loss
    loss = detection_weight * detection_loss + 0.1*data_dict["ref_loss"] + 0.1*data_dict["lang_loss"]

    loss *= 10 # amplify

    data_dict['loss'] = loss

     
    return loss, data_dict
