# Refs
# paper : https://arxiv.org/abs/1706.03762

import torch
import torch.nn as nn
import numpy as np
from .utils import get_padding_mask
from .sublayers import MultiHeadAttention, PositionWiseFFN

# Encode Layer
class Encode_Layer(nn.Module):
    """encode layer"""
    def __init__(self, n_head, d_model, d_k, d_v, d_f, drop_rate=0.1, use_conv=False):
        super(Encode_Layer, self).__init__()
        self.n_head = n_head
        self.selfattn = MultiHeadAttention(n_head, d_model, d_k, d_v, drop_rate=drop_rate)
        self.pwffn = PositionWiseFFN(d_model, d_f, drop_rate=drop_rate, use_conv=use_conv)
        
    def forward(self, enc_input, enc_mask=None):

        """
        Inputs:
        * enc_input: (B, T_e, d_model)
        * enc_mask: (B, T_e, T_e)
        * non_pad_mask: (B, T_e, 1)
        -------------------------------------
        Outputs:
        * enc_output: (B, T_e, d_model)
        * enc_attn: (n_head*B, T_e, T_e)
        """
        # Layer: Multi-Head Attention + Add & Norm
        # encode self-attention
        enc_output, enc_attn = self.selfattn(enc_input, enc_input, enc_input, mask=enc_mask)
        
        # Layer: PositionWiseFFN + Add & Norm
        pw_output = self.pwffn(enc_output)

        return enc_output, enc_attn
    
    
# Decode Layer
class Decode_Layer(nn.Module):
    """decode layer"""
    def __init__(self, n_head, d_model, d_k, d_v, d_f, drop_rate=0.1, use_conv=False):
        super(Decode_Layer, self).__init__()
        self.n_head = n_head
        self.selfattn_masked = MultiHeadAttention(n_head, d_model, d_k, d_v, drop_rate=drop_rate)
        self.dec_enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, drop_rate=drop_rate)
        self.pwffn = PositionWiseFFN(d_model, d_f, drop_rate=drop_rate, use_conv=use_conv)
    
    def forward(self, dec_input, enc_output, dec_self_mask=None, dec_enc_mask=None):
        """
        Inputs:
        * dec_input: (B, T_d, d_model)
        * enc_input: (B, T_e, d_model)
        * dec_self_mask: (B, T_d, T_d)
        * dec_enc_mask: (B, T_d, T_e)
        -------------------------------------
        Outputs:
        * dec_output: (B, T_d, d_model)
        * dec_self_attn: (n_head*B, T_d, T_d)
        * dec_enc_attn: (n_head*B, T_d, T_e)
        """
        # Layer: Multi-Head Attention + Add & Norm
        # decode self-attention
        dec_self_output, dec_self_attn = \
            self.selfattn_masked(dec_input, dec_input, dec_input, mask=dec_self_mask)
        
        # Layer: Multi-Head Attention + Add & Norm
        # decode output(queries) + encode output(keys, values)
        dec_output, dec_enc_attn = \
            self.dec_enc_attn(dec_self_output, enc_output, enc_output, mask=dec_enc_mask)
        
        # Layer: PositionWiseFFN + Add & Norm
        dec_output = self.pwffn(dec_output)

        return dec_output, (dec_self_attn, dec_enc_attn)     
    
    
    
# Position Encoding & Embedding Layers
class PositionalEncoding(nn.Module):
    """Positional Encoding"""
    def __init__(self, n_pos, d_model, pos_pad_idx=0, sos_idx=None, eos_idx=None):
        """
        n_pos = position length, max sequence length + 1
        """
        super(PositionalEncoding, self).__init__()
        self.n_pos = n_pos
        self.d_model = d_model
        self.pe_table = np.array(self.get_pe_table())
        self.pe_table[:, 0::2] = np.sin(self.pe_table[:, 0::2])
        self.pe_table[:, 1::2] = np.cos(self.pe_table[:, 1::2])
        self.pe_table[pos_pad_idx, :] = 0  # embed all pad to 0
        self.pe = nn.Embedding.from_pretrained(torch.FloatTensor(self.pe_table), freeze=True)
        
    def cal_angle(self, pos, hid_idx):
        return pos / (10000 ** ((2*(hid_idx // 2) / self.d_model)) )
    
    def get_pe_table(self):
        return [[self.cal_angle(pos, i) for i in range(self.d_model)] for pos in range(self.n_pos)]         
        
    def forward(self, inputs):
        return self.pe(inputs)
    
    
class Embedding(nn.Module):
    """Custom Embedding Layer"""
    def __init__(self, vocab_len, d_model, padding_idx=1):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_len, d_model, padding_idx=padding_idx)
        
    def forward(self, x):
        # In the embedding layers, authors multiply those weights by `sqrt(d_model)`
        return self.embedding(x) * np.sqrt(self.d_model)
