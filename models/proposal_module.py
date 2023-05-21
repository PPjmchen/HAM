""" 
Modified from: https://github.com/facebookresearch/votenet/blob/master/models/proposal_module.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
import lib.pointnet2.pointnet2_utils
from lib.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from models.free_module import FPSModule, PointsObjClsModule, GeneralSamplingModule, ClsAgnosticPredictHead, PredictHead, PositionEmbeddingLearned
from models.transformer import TransformerDecoderLayer
from easydict import EasyDict
from utils.box_util import get_3d_box_batch


class GroupFreeProposalModule(nn.Module):
    def __init__(self, sampling, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, hidden_dim=288, num_decoder_layers=6):
        super().__init__() 

        self.sampling = sampling
        self.num_decoder_layers = num_decoder_layers
        self.num_proposal = num_proposal

        if self.sampling == 'fps':
            self.fps_module = FPSModule(num_proposal)
        elif self.sampling == 'kps':
            self.points_obj_cls = PointsObjClsModule(288)
            self.gsample_module = GeneralSamplingModule()
        else:
            raise NotImplementedError

        # Proposal Module
        self.proposal_head = PredictHead(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, hidden_dim)

         # Transformer Decoder
        # Transformer Decoder Projection
        self.decoder_key_proj = nn.Conv1d(288, 288, kernel_size=1)
        self.decoder_query_proj = nn.Conv1d(288, 288, kernel_size=1)

        # Position Embedding for Self-Attention
        self.decoder_self_posembeds = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder_self_posembeds.append(PositionEmbeddingLearned(6, 288))
        

        # Position Embedding for Cross-Attention
        self.decoder_cross_posembeds = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder_cross_posembeds.append(PositionEmbeddingLearned(3, 288))
        
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    288, nhead=8, dim_feedforward=2048, dropout=0.1, activation='relu',
                    self_posembed=self.decoder_self_posembeds[i],
                    cross_posembed=self.decoder_cross_posembeds[i],
                ))
        
        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.prediction_heads.append(PredictHead(num_class, num_heading_bin, num_size_cluster,
                                                         mean_size_arr, num_proposal, 288))
    def forward(self, xyz, features, data_dict):
        if self.sampling == 'fps':
            xyz, features, sample_inds = self.fps_module(xyz, features) # FPS from 1024 points to 256 points
            cluster_feature = features
            cluster_xyz = xyz
            data_dict['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            data_dict['query_points_feature'] = features  # (batch_size, C, num_proposal)
            data_dict['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal

        elif self.sampling == 'kps':
            points_obj_cls_logits = self.points_obj_cls(features)  # (B, 1, num_seed)
            data_dict['seeds_obj_cls_logits'] = points_obj_cls_logits
            points_obj_cls_scores = torch.sigmoid(points_obj_cls_logits).squeeze(1)
            sample_inds = torch.topk(points_obj_cls_scores, self.num_proposal)[1].int()
            xyz, features, sample_inds = self.gsample_module(xyz, features, sample_inds)
            cluster_feature = features
            cluster_xyz = xyz
            data_dict['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            data_dict['query_points_feature'] = features  # (batch_size, C, num_proposal)
            data_dict['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal
        else:
            raise NotImplementedError
        
        points_xyz = data_dict['fp2_xyz']
        points_features = data_dict["fp2_features"]

        # --------- PROPOSAL GENERATION ---------
        proposal_center, proposal_size = self.proposal_head(cluster_feature,
                                                            base_xyz=cluster_xyz,
                                                            end_points=data_dict,
                                                            prefix='proposal_') 

        base_xyz = proposal_center.detach().clone()
        base_size = proposal_size.detach().clone()

        # Transformer Decoder and Prediction
        if self.num_decoder_layers > 0:
            query = self.decoder_query_proj(cluster_feature)
            key = self.decoder_key_proj(points_features) if self.decoder_key_proj is not None else None
        
        key_pos = points_xyz

        for i in range(self.num_decoder_layers):
            prefix = 'last_' if (i == self.num_decoder_layers - 1) else f'{i}head_'
            query_pos = torch.cat([base_xyz, base_size], -1)

            # Transformer Decoder Layer
            query = self.decoder[i](query, key, query_pos, key_pos)

            # Prediction
            base_xyz, base_size = self.prediction_heads[i](query,
                                                        base_xyz=cluster_xyz,
                                                        end_points=data_dict,
                                                        prefix=prefix)

            base_xyz = base_xyz.detach().clone()
            base_size = base_size.detach().clone()

        data_dict['query_features'] = query
        data_dict['key_features'] = key
        
        objectness_mask = torch.sigmoid(data_dict['last_objectness_scores']).round()
        data_dict['objectness_masks'] = objectness_mask

        data_dict['center'] = data_dict['last_center']
        data_dict['heading_scores'] = data_dict['last_heading_scores']
        data_dict['heading_residuals'] = data_dict['last_heading_residuals']
        data_dict['size_scores'] = data_dict['last_size_scores']
        data_dict['size_residuals'] = data_dict['last_size_residuals']
        
        return data_dict

def decode_scores_classes(output_dict, end_points, num_class):
    pred_logits = output_dict['pred_logits']
    assert pred_logits.shape[-1] == 2+num_class, 'pred_logits.shape wrong'
    objectness_scores = pred_logits[:,:,0:2]  # TODO CHANGE IT; JUST SOFTMAXd
    end_points['objectness_scores'] = objectness_scores
    sem_cls_scores = pred_logits[:,:,2:2+num_class] # Bxnum_proposalx10
    end_points['sem_cls_scores'] = sem_cls_scores
    end_points['objectness_masks'] = objectness_scores.max(2)[1].float().unsqueeze(2)
    return end_points


def decode_dataset_config(data_dict, dataset_config, mean_size_arr):
    if dataset_config is not None:
        # print('decode_dataset_config', flush=True)
        pred_center = data_dict['center'].detach().cpu().numpy()  # (B,K,3)
        pred_heading_class = torch.argmax(data_dict['heading_scores'], -1)  # B,num_proposal
        pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2,
                                             pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
        pred_heading_class = pred_heading_class.detach().cpu().numpy()  # B,num_proposal
        pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal
        pred_size_class = torch.argmax(data_dict['size_scores'], -1)  # B,num_proposal
        pred_size_residual = torch.gather(data_dict['size_residuals'], 2,
                                          pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3))  # B,num_proposal,1,3
        
        size_recover = pred_size_residual + torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)
        pred_size_class_tmp = pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3)
        pred_size = torch.gather(size_recover, 2, pred_size_class_tmp)  # batch_size, num_proposal, 1, 3
        pred_size = pred_size.squeeze_(2)  # batch_size, num_proposal, 3

        data_dict['pred_size'] = pred_size


        pred_size_class = pred_size_class.detach().cpu().numpy()
        pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal,3
        batch_size = pred_center.shape[0]
        pred_obbs, pred_bboxes = [], []
        for i in range(batch_size):
            pred_obb_batch = dataset_config.param2obb_batch(pred_center[i, :, 0:3], pred_heading_class[i],
                                                            pred_heading_residual[i],
                                                            pred_size_class[i], pred_size_residual[i])
            pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])
            pred_obbs.append(torch.from_numpy(pred_obb_batch))
            pred_bboxes.append(torch.from_numpy(pred_bbox_batch))
            # print(pred_obb_batch.shape, pred_bbox_batch.shape)
        data_dict['pred_obbs'] = torch.stack(pred_obbs, dim=0).cuda()
        data_dict['pred_bboxes'] = torch.stack(pred_bboxes, dim=0).cuda()
    return data_dict

def decode_scores(output_dict, end_points,  num_class, num_heading_bin, num_size_cluster, mean_size_arr, center_with_bias=False, quality_channel=False, dataset_config=None):
    end_points = decode_scores_classes(output_dict, end_points, num_class)
    end_points = decode_scores_boxes(output_dict, end_points, num_heading_bin, num_size_cluster, mean_size_arr, center_with_bias, quality_channel)
    end_points = decode_dataset_config(end_points, dataset_config, mean_size_arr)
    return end_points
