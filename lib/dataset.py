'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''

import os
import sys
import time
import h5py
import json
import pickle
import numpy as np
import multiprocessing as mp
import random
from torch.utils import data
from torch.utils.data import Dataset

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.config import CONF
from utils.pc_utils import random_sampling, rotx, roty, rotz
from data.scannet.model_util_scannet import rotate_aligned_boxes, ScannetDatasetConfig, rotate_aligned_boxes_along_axis

# data setting
DC = ScannetDatasetConfig()
MAX_NUM_OBJ = 128
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

# data path
SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, "scannetv2-labels.combined.tsv")
MULTIVIEW_DATA = CONF.MULTIVIEW
GLOVE_PICKLE = os.path.join(CONF.PATH.DATA, "glove.p")

class ScannetReferenceDataset(Dataset):
       
    def __init__(self, scanrefer, scanrefer_new, scanrefer_all_scene, 
        split="train", 
        num_points=40000,
        lang_num_max=32,
        use_height=False, 
        use_color=False, 
        use_normal=False, 
        use_multiview=False, 
        augment=False,
        shuffle=False,
        sent_aug = False,
        sent_len_max = 40):

        self.scanrefer = scanrefer
        self.scanrefer_new = scanrefer_new
        self.scanrefer_new_len = len(scanrefer_new)
        self.scanrefer_all_scene = scanrefer_all_scene # all scene_ids in scanrefer
        self.split = split
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.use_normal = use_normal        
        self.use_multiview = use_multiview
        self.augment = augment
        self.sent_aug = sent_aug
        self.lang_num_max = lang_num_max
        self.sent_len_max = sent_len_max

        # load data
        self._load_data()

        self._language_pack_data()
        self.multiview_data = {}

       
    def __len__(self):
        return self.scanrefer_new_len

    


    def __getitem__(self, idx):
        start = time.time()

        lang_num = len(self.scanrefer_new[idx])
        scene_id = self.scanrefer_new[idx][0]["scene_id"]
        
        object_id_list = self.all_object_id_list[idx]
        object_name_list = self.all_object_name_list[idx]
        ann_id_list = self.all_ann_id_list[idx]
        
        lang_feat_list = self.all_lang_feat_list[idx]
        lang_len_list = self.all_lang_len_list[idx]
        
        # get pc
        mesh_vertices = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_vert.npy") # axis-aligned
        instance_labels = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_ins_label.npy")
        semantic_labels = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_sem_label.npy")
        # instance_bboxes = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_bbox.npy")
        instance_bboxes = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id)+"_aligned_bbox.npy")

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            point_cloud[:,3:6] = (point_cloud[:,3:6]-MEAN_COLOR_RGB)/256.0
            pcl_color = point_cloud[:,3:6]
        
        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals],1)

        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(MULTIVIEW_DATA, "r", libver="latest")

            multiview = self.multiview_data[pid][scene_id]
            point_cloud = np.concatenate([point_cloud, multiview],1)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99) # The height of the floor
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) 
        
        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)        
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices] # same as point_cloud[id][3:6]
        
        # ------------------------------- LABELS ------------------------------    
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))    
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        size_gts = np.zeros((MAX_NUM_OBJ, 3))


        point_obj_mask = np.zeros(self.num_points)
        point_instance_label = np.zeros(self.num_points) - 1

        ref_box_label_list = []
        ref_center_label_list = []
        ref_heading_class_label_list = []
        ref_heading_residual_label_list = []
        ref_size_class_label_list = []
        ref_size_residual_label_list = []

        if self.split != "test":
            num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < MAX_NUM_OBJ else MAX_NUM_OBJ
            target_bboxes_mask[0:num_bbox] = 1
            target_bboxes[0:num_bbox,:] = instance_bboxes[:MAX_NUM_OBJ,0:6]

            point_votes = np.zeros([self.num_points, 3])
            point_votes_mask = np.zeros(self.num_points)

            # ------------------------------- DATA AUGMENTATION ------------------------------        
            if self.augment:
                if random.random() > 0.5:
                    # Flipping along the YZ plane
                    point_cloud[:,0] = -1 * point_cloud[:,0]
                    target_bboxes[:,0] = -1 * target_bboxes[:,0]                
                    
                if random.random() > 0.5:
                    # Flipping along the XZ plane
                    point_cloud[:,1] = -1 * point_cloud[:,1]
                    target_bboxes[:,1] = -1 * target_bboxes[:,1]                                

                # Rotation along X-axis
                rot_angle = (random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat = rotx(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "x")

                # Rotation along Y-axis
                rot_angle = (random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat = roty(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "y")

                # Rotation along up-axis/Z-axis
                rot_angle = (random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
                rot_mat = rotz(rot_angle)
                point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "z")


                scale = np.random.uniform(-0.1, 0.1, (3, 3))
                scale = np.exp(scale)
                # print(scale, '<<< scale', flush=True)
                scale = scale * np.eye(3)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], scale)
                if self.use_height:
                    point_cloud[:, 3] = point_cloud[:, 3] * float(scale[2, 2])
                target_bboxes[:, 0:3] = np.dot(target_bboxes[:, 0:3], scale)
                target_bboxes[:, 3:6] = np.dot(target_bboxes[:, 3:6], scale)
                # Translation
                point_cloud, target_bboxes = self._translate(point_cloud, target_bboxes)

            gt_centers = target_bboxes[:, 0:3]
            gt_centers[instance_bboxes.shape[0]:, :] += 1000.0  # padding centers with a large number

            
            for i_instance in np.unique(instance_labels): 
       
                # find all points belong to that instance
                ind = np.where(instance_labels == i_instance)[0]
                # find the semantic label            
                if semantic_labels[ind[0]] in DC.nyu40ids:
                    x = point_cloud[ind,:3]
                    center = 0.5*(x.min(0) + x.max(0))
                    ilabel = np.argmin(((center - gt_centers) ** 2).sum(-1))
                    point_instance_label[ind] = ilabel
                    point_obj_mask[ind] = 1.0
                    point_votes[ind, :] = center - x
                    point_votes_mask[ind] = 1.0
            point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical

            class_ind = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:num_bbox,-2]]
            # NOTE: set size class as semantic class. Consider use size2class.
            size_classes[0:num_bbox] = class_ind
            size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - DC.mean_size_arr[class_ind,:]
            size_gts[0:instance_bboxes.shape[0], :] = target_bboxes[0:instance_bboxes.shape[0], 3:6]
            # construct the reference target label for each bbox
            
            if self.split == 'train' and self.sent_aug:
                for j in range(self.lang_num_max):
                    ref_box_label = np.zeros(MAX_NUM_OBJ)
                    ref_center_label = np.zeros([MAX_NUM_OBJ, 3])
                    ref_heading_class_label = np.zeros(MAX_NUM_OBJ)
                    ref_heading_residual_label = np.zeros(MAX_NUM_OBJ)
                    ref_size_class_label = np.zeros(MAX_NUM_OBJ)
                    ref_size_residual_label = np.zeros([MAX_NUM_OBJ, 3])

                    for i, gt_id in enumerate(instance_bboxes[:num_bbox, -1]):
                        if isinstance(object_id_list[j], int):
                            if gt_id == object_id_list[j]:
                                ref_box_label[i] = 1
                                ref_center_label[i] = target_bboxes[i, 0:3]
                                ref_heading_class_label[i] = angle_classes[i]
                                ref_heading_residual_label[i] = angle_residuals[i]
                                ref_size_class_label[i] = size_classes[i]
                                ref_size_residual_label[i] = size_residuals[i]
                                
                                ref_box_label_list.append(ref_box_label)
                                ref_center_label_list.append(ref_center_label)
                                ref_heading_class_label_list.append(ref_heading_class_label)
                                ref_heading_residual_label_list.append(ref_heading_residual_label)
                                ref_size_class_label_list.append(ref_size_class_label)
                                ref_size_residual_label_list.append(ref_size_residual_label)
                        
                        elif isinstance(object_id_list[j], str):
                            if str(int(gt_id)) in object_id_list[j].split('-'):
                                ref_box_label[i] = 1
                                ref_center_label[i] = target_bboxes[i, 0:3]
                                ref_heading_class_label[i] = angle_classes[i]
                                ref_heading_residual_label[i] = angle_residuals[i]
                                ref_size_class_label[i] = size_classes[i]
                                ref_size_residual_label[i] = size_residuals[i]

                                if sum(ref_box_label) < len(object_id_list[j].split('-')):
                                    continue
                                elif sum(ref_box_label) == len(object_id_list[j].split('-')):
                                    ref_box_label_list.append(ref_box_label)
                                    ref_center_label_list.append(ref_center_label)
                                    ref_heading_class_label_list.append(ref_heading_class_label)
                                    ref_heading_residual_label_list.append(ref_heading_residual_label)
                                    ref_size_class_label_list.append(ref_size_class_label)
                                    ref_size_residual_label_list.append(ref_size_residual_label)
                                else:
                                    raise NotImplemented
            else:
                for j in range(self.lang_num_max):
                    ref_box_label = np.zeros(MAX_NUM_OBJ)
                    for i, gt_id in enumerate(instance_bboxes[:num_bbox, -1]):
                        if gt_id == object_id_list[j]:
                            ref_box_label[i] = 1
                            ref_center_label = target_bboxes[i, 0:3]
                            ref_heading_class_label = angle_classes[i]
                            ref_heading_residual_label = angle_residuals[i]
                            ref_size_class_label = size_classes[i]
                            ref_size_residual_label = size_residuals[i]
                            
                            ref_box_label_list.append(ref_box_label)
                            ref_center_label_list.append(ref_center_label)
                            ref_heading_class_label_list.append(ref_heading_class_label)
                            ref_heading_residual_label_list.append(ref_heading_residual_label)
                            ref_size_class_label_list.append(ref_size_class_label)
                            ref_size_residual_label_list.append(ref_size_residual_label)

        else:
            num_bbox = 1
            point_votes = np.zeros([self.num_points, 9]) # make 3 votes identical
            point_votes_mask = np.zeros(self.num_points)

        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        try:
            target_bboxes_semcls[0:num_bbox] = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:,-2][0:num_bbox]]
        except KeyError:
            pass
        
        object_cat_list = []
        for i in range(self.lang_num_max):
            object_cat = self.raw2label[object_name_list[i]] if object_name_list[i] in self.raw2label else 17
            object_cat_list.append(object_cat)

        istrain = 0
        if self.split == "train":
            istrain = 1
        

        unk = self.unk


        data_dict = {}
        data_dict["point_clouds"] = point_cloud.astype(np.float32) # point cloud data including features
        
        if self.split == 'test':
            offset = data_dict["point_clouds"].mean(0)[:2]

            data_dict['point_offset'] = offset
            
            data_dict["point_clouds"][:, :2] -= data_dict['point_offset']
            
        
        data_dict["unk"] = unk.astype(np.float32)
        data_dict["istrain"] = istrain
        data_dict["center_label"] = target_bboxes.astype(np.float32)[:,0:3] # (MAX_NUM_OBJ, 3) for GT box center XYZ
        data_dict["heading_class_label"] = angle_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
        data_dict["heading_residual_label"] = angle_residuals.astype(np.float32) # (MAX_NUM_OBJ,)
        data_dict["size_class_label"] = size_classes.astype(np.int64) # (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
        data_dict["size_residual_label"] = size_residuals.astype(np.float32) # (MAX_NUM_OBJ, 3)
        data_dict["num_bbox"] = np.array(num_bbox).astype(np.int64)
        data_dict["size_gts"] = size_gts.astype(np.float32)
        data_dict["sem_cls_label"] = target_bboxes_semcls.astype(np.int64) # (MAX_NUM_OBJ,) semantic class index
        data_dict["box_label_mask"] = target_bboxes_mask.astype(np.float32) # (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
        data_dict["point_obj_mask"] = point_obj_mask.astype(np.int64)
        data_dict['point_instance_label'] = point_instance_label.astype(np.int64)
        data_dict["vote_label"] = point_votes.astype(np.float32)
        data_dict["vote_label_mask"] = point_votes_mask.astype(np.int64)
        data_dict["scan_idx"] = np.array(idx).astype(np.int64)
        data_dict["pcl_color"] = pcl_color

        data_dict["lang_num"] = np.array(lang_num).astype(np.int64)
        data_dict["lang_feat_list"] = np.array(lang_feat_list).astype(np.float32)  # language feature vectors
        data_dict["lang_len_list"] = np.array(lang_len_list).astype(np.int64)  # length of each description
       
        
        data_dict["ref_box_label_list"] = np.array(ref_box_label_list).astype(np.int64)  # 0/1 reference labels for each object bbox
        data_dict["ref_center_label_list"] = np.array(ref_center_label_list).astype(np.float32)
        data_dict["ref_heading_class_label_list"] = np.array(ref_heading_class_label_list).astype(np.int64)
        data_dict["ref_heading_residual_label_list"] = np.array(ref_heading_residual_label_list).astype(np.int64)
        data_dict["ref_size_class_label_list"] = np.array(ref_size_class_label_list).astype(np.int64)
        data_dict["ref_size_residual_label_list"] = np.array(ref_size_residual_label_list).astype(np.float32)
        data_dict["object_cat_list"] = np.array(object_cat_list).astype(np.int64)

        if self.split != 'train':
            data_dict["object_id_list"] = np.array(object_id_list).astype(np.int64)
            data_dict["ann_id_list"] = np.array(ann_id_list).astype(np.int64)
            
            unique_multiple_list = []
            for i in range(self.lang_num_max):
                object_id = object_id_list[i]
                ann_id = ann_id_list[i]
                unique_multiple = self.unique_multiple_lookup[scene_id][str(object_id)][ann_id]
                unique_multiple_list.append(unique_multiple)
            data_dict["unique_multiple_list"] = np.array(unique_multiple_list).astype(np.int64)
        
        return data_dict
    
    def _get_raw2label(self):
        # mapping
        scannet_labels = DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]

        return raw2label

    def _get_unique_multiple_lookup(self):
        all_sem_labels = {}
        cache = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            if scene_id not in all_sem_labels:
                all_sem_labels[scene_id] = []

            if scene_id not in cache:
                cache[scene_id] = {}

            if object_id not in cache[scene_id]:
                cache[scene_id][object_id] = {}
                try:
                    all_sem_labels[scene_id].append(self.raw2label[object_name])
                except KeyError:
                    all_sem_labels[scene_id].append(17)

        # convert to numpy array
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            try:
                sem_label = self.raw2label[object_name]
            except KeyError:
                sem_label = 17

            unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1

            # store
            if scene_id not in unique_multiple_lookup:
                unique_multiple_lookup[scene_id] = {}

            if object_id not in unique_multiple_lookup[scene_id]:
                unique_multiple_lookup[scene_id][object_id] = {}

            if ann_id not in unique_multiple_lookup[scene_id][object_id]:
                unique_multiple_lookup[scene_id][object_id][ann_id] = None

            unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

        return unique_multiple_lookup

    def _tranform_des(self):
        with open(GLOVE_PICKLE, "rb") as f:
            glove = pickle.load(f)

        self.unk = glove['unk']
        
        lang = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            ann_id = data["ann_id"]

            if scene_id not in lang:
                lang[scene_id] = {}

            if object_id not in lang[scene_id]:
                lang[scene_id][object_id] = {}

            if ann_id not in lang[scene_id][object_id]:
                lang[scene_id][object_id][ann_id] = {}

            # tokenize the description
            tokens = data["token"]
            embeddings = np.zeros((self.sent_len_max, 300))
            # tokens = ["sos"] + tokens + ["eos"]
            # embeddings = np.zeros((self.sent_len_max + 2, 300))
            for token_id in range(self.sent_len_max):
                if token_id < len(tokens):
                    token = tokens[token_id]
                    if token in glove:
                        embeddings[token_id] = glove[token]
                    else:
                        embeddings[token_id] = glove["unk"]

            # store
            lang[scene_id][object_id][ann_id] = embeddings

        return lang


    def _load_data(self):
        

        # add scannet data
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))

        # prepare class mapping
        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        # store
        self.raw2nyuid = raw2nyuid
        self.raw2label = self._get_raw2label()

        if self.split != 'train':
            self.unique_multiple_lookup = self._get_unique_multiple_lookup()

        # load language features
        self.lang = self._tranform_des()

    # Inter-Sentence Ensemble
    def _language_pack_data(self):
        
        self.all_object_id_list = []
        self.all_object_name_list = []
        self.all_ann_id_list = []
        self.all_lang_feat_list = []
        self.all_lang_len_list = []

        for idx in range(self.scanrefer_new_len):
            lang_num = len(self.scanrefer_new[idx])
            scene_id = self.scanrefer_new[idx][0]["scene_id"]
            object_id_list = []
            object_name_list = []
            ann_id_list = []
            lang_feat_list = []
            lang_len_list = []


            for i in range(self.lang_num_max):
                if i < lang_num:
                    if '-' in self.scanrefer_new[idx][i]["object_id"]:
                        object_id = str(self.scanrefer_new[idx][i]["object_id"])
                    else:
                        object_id = int(self.scanrefer_new[idx][i]["object_id"])
                    object_name = " ".join(self.scanrefer_new[idx][i]["object_name"].split("_"))
                    ann_id = self.scanrefer_new[idx][i]["ann_id"]

                    lang_feat = self.lang[scene_id][str(object_id)][ann_id]
                    lang_len = len(self.scanrefer_new[idx][i]["token"])
                    lang_len = lang_len if lang_len <= self.sent_len_max else self.sent_len_max
                    

                object_id_list.append(object_id)
                object_name_list.append(object_name)
                ann_id_list.append(ann_id)

                lang_feat_list.append(lang_feat)
                lang_len_list.append(lang_len)

            
            self.all_object_id_list.append(object_id_list)
            self.all_object_name_list.append(object_name_list)
            self.all_ann_id_list.append(ann_id_list)
            self.all_lang_feat_list.append(lang_feat_list)
            self.all_lang_len_list.append(lang_len_list)
    
    def _translate(self, point_set, bbox):
        # unpack
        coords = point_set[:, :3]
        
        # translation factors
        x_factor = random.choice(np.arange(-0.5, 0.501, 0.001))
        y_factor = random.choice(np.arange(-0.5, 0.501, 0.001))
        z_factor = random.choice(np.arange(-0.5, 0.501, 0.001))
        factor = [x_factor, y_factor, z_factor]
        
        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox
