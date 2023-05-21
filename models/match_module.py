import random
import torch
import torch.nn as nn
from models.transformer import PLACM, MultiWindowsFusion


class HAMMatchModule(nn.Module):
    def __init__(self, num_proposals=256, lang_size=256, hidden_size=128, sent_len_max=40, 
                    use_spa=False, multi_window=[4], ):
        super().__init__()  
        self.fusion_decoder = PLACM(fuse_with_query=False, fuse_with_key=True, hidden_size=hidden_size, sent_len_max=sent_len_max)  
        
        self.use_spa = use_spa

        self.num_proposals = num_proposals

        if self.use_spa:
            self.fusion_decoder_spa = MultiWindowsFusion(fuse_with_query=False, fuse_with_key=True, 
                                                        multi_window=multi_window, feedforward_dim=2048, 
                                                        spa_glfirst=False, spa_gllast=False,
                                                        sent_len_max=sent_len_max)


            match_input_scale = 1
            
            self.match_spa = nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, 1),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Conv1d(hidden_size, hidden_size, 1),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Conv1d(hidden_size, 1, 1)
            )

        self.match_vla = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, 1, 1)
        )

        self.match = nn.Sequential(
            nn.Conv1d(hidden_size*match_input_scale, hidden_size*match_input_scale, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size*match_input_scale),
            nn.Conv1d(hidden_size*match_input_scale, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, 1, 1)
        )
    def forward(self, data_dict):

        len_nun_max = data_dict["lang_feat_list"].shape[1]
        lang_features = data_dict['lang_fea'] # (B, 40, 288)
        sentence_feature = data_dict['lang_emb']

        key = data_dict['key_features']
        key_pos = data_dict['seed_xyz']

        base_xyz = data_dict['last_center']
        base_size = data_dict['last_pred_size']
        base_xyz = base_xyz.detach().clone()
        base_size = base_size.detach().clone()
        query_proposal = data_dict['query_features']
        query_proposal_pos = torch.cat([base_xyz, base_size], -1)
        objectness_scores = data_dict['last_objectness_scores']
         
        batch_size, num_proposal = query_proposal.shape[0], query_proposal.shape[2]
        
        query_proposal = query_proposal[:, None, :, :].repeat(1, len_nun_max, 1, 1).reshape(batch_size*len_nun_max, -1, num_proposal)
        key = key[:, None, :, :].repeat(1, len_nun_max, 1, 1).reshape(batch_size*len_nun_max, -1, key.shape[2])
        query_proposal_pos = query_proposal_pos[:, None, :, :].repeat(1, len_nun_max, 1, 1).reshape(batch_size*len_nun_max, num_proposal, -1)
        key_pos = key_pos[:, None, :, :].repeat(1, len_nun_max, 1, 1).reshape(batch_size*len_nun_max, key.shape[2], -1)
        objectness_scores = objectness_scores[:, None, :, :].repeat(1, len_nun_max, 1, 1).reshape(batch_size*len_nun_max, num_proposal, -1)
        lang_features = lang_features.permute(0, 2, 1)

        
        query_proposal_vla = self.fusion_decoder(query_proposals=query_proposal, key_points=key, query_proposals_pos=query_proposal_pos, key_points_pos=key_pos, 
                                                lang_features=lang_features, sentence_feature=sentence_feature, objectness_mask=objectness_scores)

        
        if self.use_spa:
            query_proposal_spa = self.fusion_decoder_spa(query_proposals=query_proposal, 
                                            key_points=key, 
                                            query_proposals_pos=query_proposal_pos, 
                                            key_points_pos=key_pos, 
                                            lang_features=lang_features, 
                                            sentence_feature=sentence_feature, 
                                            objectness_mask=objectness_scores,
                                            )

            query_proposal_fusion = query_proposal_vla + query_proposal_spa
        else:
            query_proposal_fusion = query_proposal_vla

        objectness_masks = data_dict['objectness_masks']
        objectness_masks = objectness_masks.permute(0, 2, 1).contiguous()
        objectness_masks = objectness_masks[:, None, :, :].repeat(1, len_nun_max, 1, 1).reshape(batch_size*len_nun_max, -1, num_proposal)
        confidence_vla = self.match_vla(query_proposal_vla * objectness_masks).squeeze(1)
        confidence = self.match(query_proposal_fusion * objectness_masks).squeeze(1)

        data_dict['cluster_ref_vla'] = confidence_vla
        data_dict['cluster_ref'] = confidence
        
        if self.use_spa:
            confidence_spa = self.match_spa(query_proposal_spa * objectness_masks).squeeze(1)
            data_dict['cluster_ref_spa'] = confidence_spa
        
        return data_dict
