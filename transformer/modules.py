# Refs
# paper : https://arxiv.org/abs/1706.03762
# blog: http://nlp.seas.harvard.edu/2018/04/03/attention.html

import numpy as np
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
#         self.drop_out = nn.Dropout(drop_rate)
        
    def forward(self, q, k, v, mask=None):
        """
        Inputs:
        * q: (B, T_q, d_k)
        * k: (B, T_k, d_k)
        * v: (B, T_v, d_v)  //  T_k = T_v
        -------------------------------
        Outputs:
        * output: (B, T_q, d_v)
        * attn: (B, T_q, T_k)
        """
        assert q.size(2) == k.size(2), "keys & quries must have same dimension"
        assert k.size(1) == v.size(1), "T_k = T_v"
        attn = torch.bmm(q, k.transpose(1, 2))  # (B, T_q, d_k) * (B, d_k, T_k) -> (B, T_q, T_k)
        attn = attn / np.sqrt(self.d_k)
        # why doing this? 
        # for the large values of d_k, the dot products grow large in magnitude, 
        # pushing the softmax function into regions where it has extremely small gradients
        # to counteract this effect, scaled the dot products by 1/sqrt(d_k)
        # to illustrate why the dot products get large,
        # check the function 'check_dotproduct_dist'
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        
        attn = torch.softmax(attn, dim=2)  # (B, T_q, T_k) --> (B, T_q, T_k)
        # fill all nan to zero
        mask_nan = attn != attn
        attn = attn.masked_fill(mask_nan, 0)
        output = torch.bmm(attn, v)  # (B, T_q, T_k) * (B, T_v, d_v) --> (B, T_q, d_v), make sure that T_k == T_v
        
        return output, attn