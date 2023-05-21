import os
import sys
import torch
import random
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.transformer import MultiheadAttention

class LangModule(nn.Module):
    def __init__(self, num_text_classes, use_lang_classifier=True, use_bidir=False, 
        emb_size=300, hidden_size=256, sent_len_max=40):
        super().__init__() 

        self.num_text_classes = num_text_classes
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir

        self.sent_len_max = sent_len_max

        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=self.use_bidir
        )
        lang_size = hidden_size * 2 if self.use_bidir else hidden_size

        # language classifier
        if use_lang_classifier:
            self.lang_cls = nn.Sequential(
                nn.Linear(lang_size, num_text_classes),
                nn.Dropout()
            )

        self.fc = nn.Linear(288, 288)
        self.dropout = nn.Dropout(p=.1)
        self.layer_norm = nn.LayerNorm(288)
        
    def forward(self, data_dict):
        """
        encode the input descriptions
        """
        
        word_embs = data_dict["lang_feat_list"]  # (B, 32, MAX_DES_LEN(126), Dim(300))
        lang_len = data_dict["lang_len_list"] # (B, 32)
        batch_size, len_nun_max, max_des_len = word_embs.shape[:3]

        word_embs = word_embs.reshape(batch_size * len_nun_max, max_des_len, -1) # (B*32, 126, 300)
        lang_len = lang_len.reshape(batch_size * len_nun_max) # (B*32)

        # Random Mask Out 20% Words.
        if data_dict["istrain"][0] == 1:
            for i in range(word_embs.shape[0]):
                len = lang_len[i]
                for j in range(int(len/5)):
                    num = random.randint(0, len-1)
                    word_embs[i, num] = data_dict["unk"][0]

        lang_feat = pack_padded_sequence(word_embs, lang_len.cpu(), batch_first=True, enforce_sorted=False)
        # encode description
        lang_output, lang_last = self.gru(lang_feat)
        cap_emb, cap_len = pad_packed_sequence(lang_output, batch_first=True, total_length=self.sent_len_max)

        if self.use_bidir:
            cap_emb = (cap_emb[:, :, :int(cap_emb.shape[2] / 2)] + cap_emb[:, :, int(cap_emb.shape[2] / 2):]) / 2
        
        b_s, seq_len = cap_emb.shape[:2]
        mask_queries = torch.ones((b_s, seq_len), dtype=torch.int)
        for i in range(b_s):
            mask_queries[i, cap_len[i]:] = 0
        attention_mask = (mask_queries == 0).unsqueeze(1).unsqueeze(1).cuda()  # (b_s, 1, 1, seq_len)
        data_dict["attention_mask"] = attention_mask
        

        lang_fea = F.relu(self.fc(cap_emb))  # batch_size, n, hidden_size
        lang_fea = self.dropout(lang_fea)
        lang_fea = self.layer_norm(lang_fea)
        lang_last = lang_last.permute(1, 0, 2).contiguous().flatten(start_dim=1) # batch_size, hidden_size * num_dir

        # store the encoded language features
        data_dict["lang_fea"] = lang_fea
        data_dict["lang_emb"] = lang_last # B, hidden_size
        
        # classify
        if self.use_lang_classifier:
            data_dict["lang_scores"] = self.lang_cls(data_dict["lang_emb"])

        return data_dict

