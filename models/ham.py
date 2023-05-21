from defusedxml import NotSupportedError
import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from models.backbone_module import Pointnet2Backbone
from models.proposal_module import GroupFreeProposalModule
from models.lang_module import LangModule
from models.match_module import HAMMatchModule
from models.free_module import PredictHead, ClsAgnosticPredictHead, PositionEmbeddingLearned
from transformer import TransformerDecoderLayer
from data.scannet.model_util_scannet import ScannetDatasetConfig

class HAM(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, 
    num_proposal=256, sampling="kps", use_lang_classifier=True, use_bidir=False, no_reference=False,
    emb_size=300, hidden_size=288, size_cls_agnostic=False, num_decoder_layers=6,
    fuse_with_query=True, fuse_with_key=False, backbone_width=1, 
    multi_window=[4], spa_feedforward_dim=2048, use_color=True, 
    use_normal=True, use_multiview=False, no_height=False, fps_method='D-FPS',
    visualization=False, use_spa=True, sent_len_max=40, ):

        super().__init__()  

        self.hidden_size = hidden_size
        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir 
        self.fuse_with_query = fuse_with_query
        self.fuse_with_key = fuse_with_key
        self.use_spa = use_spa
        self.sent_len_max = sent_len_max
        self.use_color = use_color
        self.use_normal = use_normal
        self.use_multiview = use_multiview
        self.no_height = no_height
        self.no_reference = no_reference
        self.size_cls_agnostic = size_cls_agnostic
        self.num_decoder_layers = num_decoder_layers

        self.fps_method = fps_method


        input_channels = int(not self.no_height) + int(self.use_color) * 3 + int(self.use_normal) * 3 + int(self.use_multiview) * 128
        self.backbone_net = Pointnet2Backbone(input_feature_dim=input_channels, width=backbone_width, fps_method=self.fps_method)

        self.proposal = self.get_proposal()

        self.lang = LangModule(num_class, use_lang_classifier, use_bidir, emb_size, hidden_size, sent_len_max=self.sent_len_max)

        self.multi_window = multi_window
        self.match = self.get_match()

    
    def get_proposal(self, ):

        return GroupFreeProposalModule('kps', self.num_class, self.num_heading_bin, self.num_size_cluster, \
                    self.mean_size_arr, self.num_proposal, self.hidden_size, num_decoder_layers=self.num_decoder_layers)


    def get_match(self,):

        return HAMMatchModule(num_proposals=self.num_proposal, lang_size=(1 + int(self.use_bidir)) * self.hidden_size, 
                hidden_size=self.hidden_size, sent_len_max=self.sent_len_max,
                use_spa=self.use_spa, multi_window=self.multi_window,)


    def forward(self, data_dict):


        # --------- BACKBONE ---------
        data_dict = self.backbone_net(data_dict, use_color=self.use_color, use_normal=self.use_normal)
                
        # --------- PROPOSAL ---------
        points_xyz = xyz = data_dict["fp2_xyz"] # (B, 1024, 3)
        points_features = features = data_dict["fp2_features"] # (B, 288, 1024)
        data_dict["seed_inds"] = data_dict["fp2_inds"]
        data_dict["seed_xyz"] = xyz
        data_dict["seed_features"] = features
        
        data_dict = self.proposal(xyz, features, data_dict)


        # --------- LANGUAGE ENCODING ---------
        data_dict = self.lang(data_dict)

        # --------- MATCHING ---------
        data_dict = self.match(data_dict)
        return data_dict
