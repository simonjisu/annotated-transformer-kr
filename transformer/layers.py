# Refs
# paper : https://arxiv.org/abs/1706.03762

import torch
import torch.nn as nn
from utils import get_padding_mask
from sublayers import MultiHeadAttention, PositionWiseFFN

# Encode Layer
class Encode_Layer(nn.Module):
    """encode layer"""
    def __init__(self, n_head, d_model, d_k, d_v, d_f, 
                 drop_rate=0.1, use_conv=False, return_attn=True):
        super(Encode_Layer, self).__init__()
        self.return_attn = return_attn
        self.n_head = n_head
        self.selfattn = MultiHeadAttention(n_head, d_model, d_k, d_v, 
                                           drop_rate=drop_rate, return_attn=return_attn)
        self.pwffn = PositionWiseFFN(d_model, d_f, 
                                     drop_rate=drop_rate, use_conv=use_conv)
        self.norm_selfattn = nn.LayerNorm(d_model)
        self.norm_pwffn = nn.LayerNorm(d_model)
        
    def forward(self, enc_input, enc_mask=None, non_pad_mask=None):
        """
        Inputs:
        * enc_input: (B, T_e, d_model)
        * enc_mask: (B, T_e, T_e)
        * non_pad_mask: (B, T_e, 1)
        -------------------------------------
        Outputs:
        * enc_output: (B, T_e, d_model)
        * enc_attn: (n_head, B, T_e, T_e)
        """
        # Layer: Multi-Head Attention + Add & Norm
        # encode self-attention
        if self.return_attn:
            enc_output, enc_attn = self.selfattn(enc_input, 
                                                 enc_input, 
                                                 enc_input,
                                                 mask=enc_mask)
        else:
            enc_output = self.selfattn(enc_input, 
                                       enc_input, 
                                       enc_input, 
                                       mask=enc_mask)
        enc_output = self.norm_selfattn(enc_input + enc_output)
        enc_output *= non_pad_mask
        
        # Layer: PositionWiseFFN + Add & Norm
        pw_output = self.pwffn(enc_output)
        pw_output *= non_pad_mask
        enc_output = self.norm_pwffn(enc_output + pw_output)
        if self.return_attn:
            # attns: (n_heads, B, T_e, T_e)
            enc_attn = torch.stack([attn*non_pad_mask for attn in enc_attn], dim=0)
            return enc_output, enc_attn
        return enc_output
    
    
# Decode Layer
class Decode_Layer(nn.Module):
    """decode layer"""
    def __init__(self, n_head, d_model, d_k, d_v, d_f, 
                 drop_rate=0.1, use_conv=False, return_attn=True):
        super(Decode_Layer, self).__init__()
        self.return_attn = return_attn
        self.n_head = n_head
        self.selfattn_masked = MultiHeadAttention(n_head, d_model, d_k, d_v, 
                                                  drop_rate=drop_rate, 
                                                  return_attn=return_attn)
        self.dec_enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, 
                                               drop_rate=drop_rate, 
                                               return_attn=return_attn)
        self.pwffn = PositionWiseFFN(d_model, d_f, drop_rate=drop_rate, use_conv=use_conv)
        self.norm_selfattn_masked = nn.LayerNorm(d_model)
        self.norm_dec_enc_attn = nn.LayerNorm(d_model)
        self.norm_pwffn = nn.LayerNorm(d_model)
    
    def forward(self, dec_input, enc_output, dec_self_mask=None, 
                dec_enc_mask=None, non_pad_mask=None):
        """
        Inputs:
        * dec_input: (B, T_d, d_model)
        * enc_input: (B, T_e, d_model)
        * dec_self_mask: (B, T_d, T_d)
        * dec_enc_mask: (B, T_d, T_e)
        * non_pad_mask: (B, T_d, 1)
        -------------------------------------
        Outputs:
        * dec_output: (B, T_d, d_model)
        * dec_self_attn: (n_head, B, T_d, T_d)
        * dec_enc_attn: (n_head, B, T_d, T_e)
        """
        # Layer: Multi-Head Attention + Add & Norm
        # decode self-attention
        if self.return_attn:
            dec_self_output, dec_self_attn = self.selfattn_masked(dec_input, 
                                                                  dec_input, 
                                                                  dec_input, 
                                                                  mask=dec_self_mask)
        else:
            dec_self_output = self.selfattn_masked(dec_input, 
                                                   dec_input, 
                                                   dec_input, 
                                                   mask=dec_self_mask)
        dec_self_output *= non_pad_mask
        dec_self_output = self.norm_selfattn_masked(dec_input + dec_self_output)
        
        
        # Layer: Multi-Head Attention + Add & Norm
        # decode output(queries) + encode output(keys, values)
        if self.return_attn:
            dec_output, dec_enc_attn = self.dec_enc_attn(dec_self_output, 
                                                         enc_output, 
                                                         enc_output, 
                                                         ask=dec_enc_mask)
        else:
             dec_output = self.dec_enc_attn(dec_self_output, 
                                            enc_output, 
                                            enc_output,
                                            mask=dec_enc_mask)
        dec_output *= non_pad_mask
        dec_output = self.norm_dec_enc_attn(dec_self_output + dec_output)
        
        
        # Layer: PositionWiseFFN + Add & Norm
        pw_output = self.pwffn(dec_output)
        pw_output *= non_pad_mask
        dec_output = self.norm_pwffn(dec_output + pw_output)
        
        if self.return_attn: 
            dec_self_attn = torch.stack([attn*non_pad_mask for attn in dec_self_attn], dim=0)
            dec_enc_attn = torch.stack([attn*non_pad_mask for attn in dec_enc_attn], dim=0)
            return dec_output, dec_self_attn, dec_enc_attn
        return dec_output
    
    
# Encoding Layers
class PositionalEncoding(nn.Module):
    """Positional Encoding"""
    def __init__(self, n_pos, d_model, pos_pad_idx=0):
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
    def __init__(self, vocab_len, d_model, pad_idx=1):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_len, d_model, padding_idx=pad_idx)
        
    def forward(self, x):
        # In the embedding layers, authors multiply those weights by `sqrt(d_model)`
        return self.embedding(x) * np.sqrt(self.d_model)