import torch
import torch.nn as nn
import math
import numpy as np

import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from multi_head_attention import MultiheadAttention
from models.free_module import PositionEmbeddingLearned
from torch.nn.init import xavier_normal_

# Construct a grid cubic(Default: 3*3*3) by given points(B, 3)
def construct_batch_grids(xyz, side_length=3):


    space_cubic_min, space_cubic_max = xyz.min(dim=1)[0].detach().cpu().numpy(), xyz.max(dim=1)[0].detach().cpu().numpy()
    space_index = np.linspace(space_cubic_min, space_cubic_max, side_length+1).transpose(1, 2, 0)
    space_index_x, space_index_y, space_index_z = torch.tensor(space_index[:, 0]), torch.tensor(space_index[:, 1]), torch.tensor(space_index[:, 2])
        
    batch_grid = None
    for i in range(space_index_x.shape[0]):
        grid_x, grid_y, grid_z = torch.meshgrid(space_index_x[i], space_index_y[i], space_index_z[i])
        grid = torch.cat([grid_x[:,:,:, None], grid_y[:,:,:, None], grid_z[:,:,:, None]], dim=-1) 
        if i == 0:
            batch_grid = grid[None, :]
        else:
            batch_grid = torch.cat([batch_grid, grid[None, :]], dim=0)
    
    return batch_grid

def get_point_pos_index(xyz, batch_grid, side_length=3):
    batch_size = xyz.shape[0]
    point_num = xyz.shape[1]
    point_left_bottom = batch_grid[:, :side_length, :side_length, :side_length]-1.1e-6 # (Batch, 3, 3, 3, 3)
    point_right_top = batch_grid[:, -side_length:, -side_length:, -side_length:]+1.1e-6 # (Batch, 3, 3, 3, 3)
    point_pos = (xyz[:, :, None, :] < point_right_top.reshape(batch_size, 1, -1, 3).repeat(1, xyz.shape[1], 1, 1).cuda()) & \
            (xyz[:, :, None, :] > point_left_bottom.reshape(batch_size, 1, -1, 3).repeat(1, xyz.shape[1], 1, 1).cuda())
    _, point_pos_index = point_pos.sum(-1).max(-1) # (Batch, 1024)
    return point_pos_index
class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=116):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i / n_position) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2] * 2 * math.pi)  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2] * 2 * math.pi)  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        if x.size(2) == self.pos_table.size(1):
            return self.pos_table[:, :x.size(2)].clone()
        elif x.size(1) == self.pos_table.size(1):
            return self.pos_table[:, :x.size(1)].clone()
        else:
            raise



class PLACM(nn.Module): # Point-Language Alignment with Context Modulation
    def __init__(self, fuse_with_query=True, fuse_with_key=False, hidden_size=288, feedforward_dim=2048, match=False, sent_len_max=40) -> None:
        super().__init__()

        self_posembed = PositionEmbeddingLearned(6, 288)
        cross_posembed = PositionEmbeddingLearned(3, 288)

        self.fuse_with_query = fuse_with_query
        self.fuse_with_key = fuse_with_key
        if fuse_with_query:
            self.query_lang_pos_encoder = PositionalEncoding(d_hid=6, n_position=sent_len_max)
        else:
            self.query_lang_pos_encoder = None
        
        if fuse_with_key:
            self.key_lang_pos_encoder = PositionalEncoding(d_hid=3, n_position=sent_len_max)
        else:
            self.key_lang_pos_encode = None

        self.fusion_layer_1 = TransformerDecoderLayer(d_model=hidden_size, nhead=8, self_posembed=self_posembed, cross_posembed=cross_posembed, dim_feedforward=feedforward_dim)
        self.fusion_layer_2 = TransformerDecoderLayer(d_model=hidden_size, nhead=8, self_posembed=self_posembed, cross_posembed=cross_posembed, dim_feedforward=feedforward_dim)

    def forward(self, query_proposals, key_points, query_proposals_pos, key_points_pos, lang_features, sentence_feature, objectness_mask):
        proposal_num = query_proposals_pos.shape[1] # (B, 256, 6)
        point_num = key_points_pos.shape[1]
        if self.fuse_with_query:
            query_lang_pos = self.query_lang_pos_encoder(lang_features.permute(0, 2, 1)).repeat(query_proposals_pos.shape[0], 1, 1)
            query_pos = torch.cat([query_proposals_pos, query_lang_pos], dim=1) # (B, 256+sent_len_max, 6)
            query_layer_1 = torch.cat([query_proposals, lang_features], dim=-1) # (B, 288, 256+sent_len_max)
        else:
            query_pos = query_proposals_pos
            query_layer_1 = query_proposals
        
        if self.fuse_with_key:
            key_lang_pos = self.key_lang_pos_encoder(lang_features.permute(0, 2, 1)).repeat(key_points_pos.shape[0], 1, 1)
            key_pos = torch.cat([key_points_pos, key_lang_pos], dim=1)
            key_layer_1 = torch.cat([key_points, lang_features], dim=-1)
        else:
            key_pos = key_points_pos
            key_layer_1 = key_points
        query_layer_1 = self.fusion_layer_1(query_layer_1, key_layer_1, query_pos, key_pos) 

        if self.fuse_with_query:
            query_layer_2 = torch.cat([query_layer_1[:, :, :proposal_num] , torch.mean(query_layer_1[:, :, proposal_num:], dim=-1, keepdim=True)], dim=-1)
            query_pos = query_pos[:, :proposal_num+1]
        else:
            query_layer_2 = query_layer_1

        if self.fuse_with_key:
            key_layer_2 = torch.cat([key_points, sentence_feature.unsqueeze(-1)], dim=-1)
            key_pos = key_pos[:, :point_num+1]
        else:
            key_layer_2 = key_points
        
        query_layer_2 = self.fusion_layer_2(query_layer_2, key_layer_2, query_pos, key_pos)

        query_layer_2 = query_layer_2[:, :, :proposal_num]


        return query_layer_2

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 self_posembed=None, cross_posembed=None):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed

    def with_pos_embed(self, tensor, pos_embed: Optional[Tensor]):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, query, key, query_pos, key_pos, self_attention_mask=None, cross_attention_mask=None, no_self_attention=False, need_weights=False):
        """
        :param query: B C Pq
        :param key: B C Pk
        :param query_pos: B Pq 3/6
        :param key_pos: B Pk 3/6
        :param value_pos: [B Pq 3/6]

        :return:
        """
        # NxCxP to PxNxC
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        if self.cross_posembed is not None:
            key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1)
        else:
            key_pos_embed = None

        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)

        if not no_self_attention:
            q = k = v = self.with_pos_embed(query, query_pos_embed)
            query2 = self.self_attn(q, k, value=v, attn_mask=self_attention_mask)[0]
            query = query + self.dropout1(query2)
            query = self.norm1(query)

        if need_weights:
            query2, attn_weights = self.multihead_attn(query=self.with_pos_embed(query, query_pos_embed),
                                        key=self.with_pos_embed(key, key_pos_embed),
                                        value=self.with_pos_embed(key, key_pos_embed),
                                        attn_mask=cross_attention_mask,
                                        need_weights=need_weights)
            
        else:
            query2 = self.multihead_attn(query=self.with_pos_embed(query, query_pos_embed),
                                        key=self.with_pos_embed(key, key_pos_embed),
                                        value=self.with_pos_embed(key, key_pos_embed),
                                        attn_mask=cross_attention_mask,)[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        # NxCxP to PxNxC
        query = query.permute(1, 2, 0)

        if need_weights:
            return query, attn_weights
        else:
            return query

class MultiWindowsFusion(nn.Module):
    def __init__(self, fuse_with_query=True, fuse_with_key=False, hidden_size=288, multi_window=[3], feedforward_dim=2048, 
                spa_glfirst=True, spa_gllast=True, sent_len_max=40) -> None:
        super().__init__()
        self.multi_window=multi_window
        self.spa_glfirst = spa_glfirst
        self.spa_gllast = spa_gllast

        if spa_glfirst:
            self.fusion_global_first = SingleFusion(fuse_with_query, fuse_with_key, hidden_size, feedforward_dim=feedforward_dim, match=False, sent_len_max=sent_len_max)
        self.fusion_window = torch.nn.ModuleList()
        for window_size in self.multi_window:
            self.fusion_window.append(WindowFusion(fuse_with_query, fuse_with_key, hidden_size, feedforward_dim=feedforward_dim, sent_len_max=sent_len_max))
        
        if spa_gllast:
            self.fusion_global_last = SingleFusion(fuse_with_query, fuse_with_key, hidden_size, feedforward_dim=feedforward_dim, match=False, sent_len_max=sent_len_max)
        
    def forward(self, query_proposals, key_points, query_proposals_pos, key_points_pos, lang_features, sentence_feature, objectness_mask):
        if self.spa_glfirst:
            query_proposals = self.fusion_global_first(query_proposals, key_points, query_proposals_pos, key_points_pos, lang_features, sentence_feature, objectness_mask)
        for i, window_size in enumerate(self.multi_window):
            query_proposals = self.fusion_window[i](query_proposals, key_points, query_proposals_pos, key_points_pos, lang_features, sentence_feature, side_length=window_size)
        
        if self.spa_gllast:
            query_proposals = self.fusion_global_last(query_proposals, key_points, query_proposals_pos, key_points_pos, lang_features, sentence_feature, objectness_mask)
        return query_proposals
        

class WindowFusion(nn.Module):
    def __init__(self, fuse_with_query=True, fuse_with_key=False, hidden_size=288, feedforward_dim=2048, sent_len_max=40) -> None:
        super().__init__()

        self_posembed = PositionEmbeddingLearned(6, 288)
        cross_posembed = PositionEmbeddingLearned(3, 288)

        self.fuse_with_query = fuse_with_query
        self.fuse_with_key = fuse_with_key
        if fuse_with_query:
            self.query_lang_pos_encoder = PositionalEncoding(d_hid=6, n_position=sent_len_max)
        else:
            self.query_lang_pos_encoder = None
        
        if fuse_with_key:
            self.key_lang_pos_encoder = PositionalEncoding(d_hid=3, n_position=sent_len_max)
        else:
            self.key_lang_pos_encode = None

        self.fusion_layer_1 = TransformerDecoderLayer(d_model=hidden_size, nhead=8, self_posembed=self_posembed, cross_posembed=cross_posembed, dim_feedforward=feedforward_dim)
        self.fusion_layer_2 = TransformerDecoderLayer(d_model=hidden_size, nhead=8, self_posembed=self_posembed, cross_posembed=cross_posembed, dim_feedforward=feedforward_dim)
    def forward(self, query_proposals, key_points, query_proposals_pos, key_points_pos, lang_features, sentence_feature, side_length):
        batch_size = key_points.shape[0]
        points_xyz = key_points_pos
        proposal_xyz = query_proposals_pos[:,:,:3]
        
        batch_grid = construct_batch_grids(points_xyz, side_length)
        key_point_pos_index = get_point_pos_index(points_xyz, batch_grid, side_length)
        query_proposal_pos_index = get_point_pos_index(proposal_xyz, batch_grid, side_length)

        self_attention_mask = query_proposal_pos_index[:,:,None] != query_proposal_pos_index[:, None, :]
        self_attention_mask = (self_attention_mask*1).float().masked_fill(self_attention_mask==True, float('-inf'))

        cross_attention_mask = query_proposal_pos_index[:,:,None] != key_point_pos_index[:, None, :]
        cross_attention_mask = (cross_attention_mask*1).float().masked_fill(cross_attention_mask==True, float('-inf'))
        
        proposal_num = query_proposals_pos.shape[1] # (B, 256, 6)
        point_num = key_points_pos.shape[1]
        word_num = lang_features.shape[-1]
        

        if self.fuse_with_query:
            query_lang_pos = self.query_lang_pos_encoder(lang_features).repeat(query_proposals_pos.shape[0], 1, 1)
            query_pos = torch.cat([query_proposals_pos, query_lang_pos], dim=1) # (B, 256+40, 6)
            query_layer_1 = torch.cat([query_proposals, lang_features], dim=-1) # (B, 288, 256+40)
            
            temp_mask = torch.zeros([batch_size, proposal_num+word_num, proposal_num+word_num]).type_as(self_attention_mask)
            temp_mask[:, :proposal_num, :proposal_num] = self_attention_mask
            self_attention_mask_1 = temp_mask
            self_attention_mask_2 = temp_mask[:, :proposal_num+1, :proposal_num+1]

        else:
            query_pos = query_proposals_pos
            query_layer_1 = query_proposals
            self_attention_mask_1 = self_attention_mask
            self_attention_mask_2 = self_attention_mask

        if self.fuse_with_key:
            key_lang_pos = self.key_lang_pos_encoder(lang_features).repeat(key_points_pos.shape[0], 1, 1)
            key_pos = torch.cat([key_points_pos, key_lang_pos], dim=1)
            key_layer_1 = torch.cat([key_points, lang_features], dim=-1)

            temp_mask = torch.zeros([batch_size, proposal_num+(self.fuse_with_query * word_num), point_num+word_num]).type_as(cross_attention_mask)
            temp_mask[:, :proposal_num, :point_num] = cross_attention_mask
            cross_attention_mask_1 = temp_mask
            cross_attention_mask_2 = temp_mask[:, :proposal_num+(self.fuse_with_query * 1), :point_num+1]
        else:
            key_pos = key_points_pos
            key_layer_1 = key_points
            if not self.fuse_with_query:
                raise('NO FUSION!')
            else:
                temp_mask = torch.zeros([batch_size, proposal_num+(self.fuse_with_query * word_num), point_num]).type_as(cross_attention_mask)
                temp_mask[:, :proposal_num, :point_num] = cross_attention_mask
                cross_attention_mask_1 = temp_mask
                cross_attention_mask_2 = temp_mask[:, :proposal_num+(self.fuse_with_query * 1), :point_num]
        
        query_layer_1 = self.fusion_layer_1(query_layer_1, key_layer_1, query_pos, key_pos, self_attention_mask=self_attention_mask_1.detach().clone(), cross_attention_mask=cross_attention_mask_1.detach().clone()) 

        if self.fuse_with_query:
            query_layer_2 = torch.cat([query_layer_1[:, :, :proposal_num] , torch.mean(query_layer_1[:, :, proposal_num:], dim=-1, keepdim=True)], dim=-1)
            query_pos = query_pos[:, :proposal_num+1]
        else:
            query_layer_2 = query_layer_1

        if self.fuse_with_key:
            key_layer_2 = torch.cat([key_points, sentence_feature.unsqueeze(-1)], dim=-1)
            key_pos = key_pos[:, :point_num+1]
        else:
            key_layer_2 = key_points
        
        query_layer_2 = self.fusion_layer_2(query_layer_2, key_layer_2, query_pos, key_pos, self_attention_mask=self_attention_mask_2.detach().clone(), cross_attention_mask=cross_attention_mask_2.detach().clone())

        query_layer_2 = query_layer_2[:, :, :proposal_num]
        torch.cuda.empty_cache()
        return query_layer_2

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
