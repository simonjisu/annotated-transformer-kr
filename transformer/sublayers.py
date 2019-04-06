# Refs
# paper : https://arxiv.org/abs/1706.03762
# blog: http://nlp.seas.harvard.edu/2018/04/03/attention.html
# code: https://github.com/jadore801120/attention-is-all-you-need-pytorch

import torch
import torch.nn as nn
from .modules import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    """Multi-head Attention"""
    def __init__(self, n_head, d_model, d_k, d_v, drop_rate=0.1, return_attn=True):
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
        self.return_attn = return_attn
        self.linear_q = nn.Linear(d_model, n_head*d_k)
        self.linear_k = nn.Linear(d_model, n_head*d_k)
        self.linear_v = nn.Linear(d_model, n_head*d_v)
        self.linear_o = nn.Linear(n_head*d_v, d_model)
        self.attention = ScaledDotProductAttention(d_k, return_attn=return_attn)
        self.drop_out = nn.Dropout(drop_rate)
        
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
        
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)

        # attention: Scaled Dot-Product Attention
        ## heads: (n_head * B, T_q, d_v)
        ## attn: (n_head * B, T_q, T_k)
        if self.return_attn:
            heads, attn = self.attention(q=lin_qs, k=lin_ks, v=lin_vs, mask=mask)
        else:
            heads = self.attention(q=lin_qs, k=lin_ks, v=lin_vs, mask=mask)
        # concat
        # be aware `heads.view(batch, T_q, n_head*d_k)` is not same as 
        # `torch.cat(heads.chunk(n_head, dim=0), dim=-1)`
        # (n_head * B, T_q, d_v) --> (B, T_q, n_head * d_v)
        heads_cat = torch.cat(heads.chunk(n_head, dim=0), dim=-1)
        # (B, T_q, n_head * d_v) --> (B, T_q, d_model)
        output = self.linear_o(heads_cat)  
        output = self.drop_out(output)
        if self.return_attn:
            return output, attn.view(n_head, B, T_q, T_k).detach()
        return output


class PositionWiseFFN(nn.Module):
    """Position-wise Feed-Forward Networks"""
    def __init__(self, d_model, d_f, drop_rate=0.1, use_conv=False):
        super(PositionWiseFFN, self).__init__()
        self.use_conv = use_conv
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
        if self.use_conv:
            # (B, T, d_model) --> (B, d_model, T), reshape like (batch, channel, dim)
            x = x.transpose(1, 2)  
            # (B, d_model, T) --> (B, T, d_model)
            output = self.fc(x).transpose(1, 2)  
        else:
            output = self.fc(x)
            
        output = self.drop_out(output)
        return output
    