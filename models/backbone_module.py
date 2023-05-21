import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule
from lib.pointnet2.pointnet2_utils import gather_operation
class Pointnet2Backbone(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """

    def __init__(self, input_feature_dim=0, width=1, depth=2, fps_method='D-FPS'):
        super().__init__()
        self.depth = depth
        self.width = width
        self.fps_method = fps_method

        self.sa1 = PointnetSAModuleVotes(
            npoint=2048,
            radius=0.2,
            nsample=64,
            mlp=[input_feature_dim] + [64 * width for i in range(depth)] + [128 * width],
            use_xyz=True,
            normalize_xyz=True,
            fps_method='D-FPS'
        )

        self.sa2 = PointnetSAModuleVotes(
            npoint=1024,
            radius=0.4,
            nsample=32,
            mlp=[128 * width] + [128 * width for i in range(depth)] + [256 * width],
            use_xyz=True,
            normalize_xyz=True,
            fps_method=self.fps_method
        )

        self.sa3 = PointnetSAModuleVotes(
            npoint=512,
            radius=0.8,
            nsample=16,
            mlp=[256 * width] + [128 * width for i in range(depth)] + [256 * width],
            use_xyz=True,
            normalize_xyz=True,
            fps_method=self.fps_method
        )

        self.sa4 = PointnetSAModuleVotes(
            npoint=256,
            radius=1.2,
            nsample=16,
            mlp=[256 * width] + [128 * width for i in range(depth)] + [256 * width],
            use_xyz=True,
            normalize_xyz=True,
            fps_method=self.fps_method
        )

        self.fp1 = PointnetFPModule(mlp=[256 * width + 256 * width, 256 * width, 256 * width])
        self.fp2 = PointnetFPModule(mlp=[256 * width + 256 * width, 256 * width, 288])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, data_dict, use_color=False, use_normal=True, no_height=False):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
            Returns
            ----------
            data_dict: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        pointcloud = data_dict["point_clouds"]


        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        if no_height:
            features = None
        xyz, features, fps_inds = self.sa1(xyz, features)
        data_dict['sa1_inds'] = fps_inds
        data_dict['sa1_xyz'] = xyz
        data_dict['sa1_features'] = features

        xyz, features, fps_inds = self.sa2(xyz, features) 
        data_dict['sa2_inds'] = fps_inds
        data_dict['sa2_xyz'] = xyz
        data_dict['sa2_features'] = features

        xyz, features, fps_inds = self.sa3(xyz, features) 
        data_dict['sa3_xyz'] = xyz
        data_dict['sa3_features'] = features

        xyz, features, fps_inds = self.sa4(xyz, features)
        data_dict['sa4_xyz'] = xyz
        data_dict['sa4_features'] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(data_dict['sa3_xyz'], data_dict['sa4_xyz'], data_dict['sa3_features'],
                            data_dict['sa4_features'])
        features = self.fp2(data_dict['sa2_xyz'], data_dict['sa3_xyz'], data_dict['sa2_features'], features)
        data_dict['fp2_features'] = features
        data_dict['fp2_xyz'] = data_dict['sa2_xyz']
        num_seed = data_dict['fp2_xyz'].shape[1]

        
        if self.fps_method == 'D-FPS':
            data_dict['fp2_inds'] = data_dict['sa1_inds'][:, 0:num_seed]  # indices among the entire input point clouds
        else:
            data_dict['fp2_inds'] = data_dict['sa1_inds'][torch.arange(batch_size).unsqueeze(1), data_dict['sa2_inds'].long()]
        
        data_dict['fp2_color'] = gather_operation(data_dict['point_clouds'].permute(0, 2, 1).contiguous(), data_dict['fp2_inds'].contiguous()).permute(0, 2, 1)[:,:,3:6]
        return data_dict

