# Refs
# paper : https://arxiv.org/abs/1706.03762
# blog: http://nlp.seas.harvard.edu/2018/04/03/attention.html
# code: https://github.com/jadore801120/attention-is-all-you-need-pytorch

import torch
import torch.nn as nn
import numpy as np
from .modules import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    """Multi-head Attention"""
    def __init__(self, n_head, d_model, d_k, d_v, drop_rate=0.1):
        """
        paper setting: n_head = 8, d_k = d_v = d_model / n_head = 64
        Multi-head attention allows the model to jointly attend to information from 
        different representation subspaces at different positions.
        with a single attention head, averaging inhibits this.
        """
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.linear_q = nn.Linear(d_model, n_head*d_k)
        self.linear_k = nn.Linear(d_model, n_head*d_k)
        self.linear_v = nn.Linear(d_model, n_head*d_v)
#         nn.init.normal_(self.linear_q.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
#         nn.init.normal_(self.linear_k.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
#         nn.init.normal_(self.linear_v.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        
        self.linear_o = nn.Linear(n_head*d_v, d_model)
        self.attention = ScaledDotProductAttention(d_k)
        self.drop_out = nn.Dropout(drop_rate)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, q, k, v, mask=None):
        """
        Inputs:
        * q: (B, T_q, d_model)
        * k: (B, T_k, d_model)
        * v: (B, T_v, d_model)
        * mask: (B, T_q, T_k)
        ---------------------
        Outputs:
        * output: (B, T_q, d_model)
        * attn: (n_head, B, T_q, T_k)
        """
        residual = q
        
        n_head, d_model, d_k, d_v = self.n_head, self.d_model, self.d_k, self.d_v
        B, T_q, _ = q.size()
        B, T_k, _ = k.size()
        B, T_v, _ = v.size()
        # through linear layer: 
        # lin_qs : (B, T_q, d_model) --> (B, T_q, n_head * d_k) --> (n_head * B, T_q, d_k)
        # lin_ks : (B, T_k, d_model) --> (B, T_k, n_head * d_k) --> (n_head * B, T_k, d_k)
        # lin_vs : (B, T_v, d_model) --> (B, T_v, n_head * d_v) --> (n_head * B, T_v, d_v)
        
        lin_qs = torch.cat(self.linear_q(q).chunk(n_head, dim=2), dim=0)
        lin_ks = torch.cat(self.linear_k(k).chunk(n_head, dim=2), dim=0)
        lin_vs = torch.cat(self.linear_v(v).chunk(n_head, dim=2), dim=0)
        # same as
        #  lin_qs = self.linear_q(q).view(B, T_q, n_head, d_k)
        #  lin_ks = self.linear_k(k).view(B, T_k, n_head, d_k)
        #  lin_vs = self.linear_v(v).view(B, T_v, n_head, d_v)
        #  lin_qs = lin_qs.permute(2, 0, 1, 3).contiguous().view(-1, T_q, d_k)
        #  lin_ks = lin_ks.permute(2, 0, 1, 3).contiguous().view(-1, T_k, d_k)
        #  lin_vs = lin_vs.permute(2, 0, 1, 3).contiguous().view(-1, T_v, d_v)
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)

        # attention: Scaled Dot-Product Attention
        ## heads: (n_head * B, T_q, d_v)
        ## attn: (n_head * B, T_q, T_k)
        heads, attn = self.attention(q=lin_qs, k=lin_ks, v=lin_vs, mask=mask)
        # concat
        # be aware `heads.view(batch, T_q, n_head*d_k)` is not same as `torch.cat(heads.chunk(n_head, dim=0), dim=-1)`
        # (n_head * B, T_q, d_v) --> (B, T_q, n_head * d_v)
        heads_cat = torch.cat(heads.chunk(n_head, dim=0), dim=-1)
        # same as 
        #  heads = heads.view(n_head, B, T_q, d_v)
        #  heads_cat = heads.permute(1, 2, 0, 3).contiguous().view(B, T_q, -1)
        
        output = self.linear_o(heads_cat)  # (B, T_q, n_head * d_v) --> (B, T_q, d_model)
        output = self.layer_norm(residual + self.drop_out(output))

        return output, attn
    
    
class PositionWiseFFN(nn.Module):
    """Position-wise Feed-Forward Networks"""
    def __init__(self, d_model, d_f, drop_rate=0.1, use_conv=False):
        super(PositionWiseFFN, self).__init__()
        self.use_conv = use_conv
        self.layer_norm = nn.LayerNorm(d_model)
        
        if use_conv:
            self.fc = nn.Sequential(
                nn.Conv1d(d_model, d_f, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(d_f, d_model, kernel_size=1)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(d_model, d_f),
                nn.ReLU(),
                nn.Linear(d_f, d_model)
            )
        self.drop_out = nn.Dropout(drop_rate)
    
    def forward(self, x):
        """
        Inputs:
        x: (B, T, d_model)
        -----------------------
        Ouputs:
        output: (B, T, d_model)
        """
        residual = x
        
        if self.use_conv:
            # (B, T, d_model) --> (B, d_model, T), reshape like (batch, channel, dim)
            x = x.transpose(1, 2)  
            # (B, d_model, T) --> (B, T, d_model)
            output = self.fc(x).transpose(1, 2)  
        else:
            output = self.fc(x)
            
        output = self.layer_norm(residual + self.drop_out(output))
        
        return output
    