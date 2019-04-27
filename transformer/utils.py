import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from .bleu import compute_bleu

# Used in the model

def get_padding_mask(q, k=None, pad_idx=1, mode='attn'):
    """
    mode: attn
    > `pad_idx` is vocab pad_idx 
    > mask out for pad in attention with queries & keys sequences
    > return shape: (B, T_q, T_k)
    mode: subseq
    > mask out next tokens to preserve 'auto-regressive property'
    > return shape: (B, T_q, T_q)
    """
    B, q_len = q.size()
    B, k_len = k.size() if k is not None else (B, None)
    if mode == 'attn':
        assert k is not None, "must have key sequences"
        padding_mask = k.eq(pad_idx)
        padding_mask = padding_mask.unsqueeze(1).expand(B, q_len, -1)
        row_pad = q.eq(1).unsqueeze(-1).expand(B, q_len, k_len)
        padding_mask = (padding_mask + row_pad).ge(1)
        return padding_mask
    elif mode =='subseq':
        assert k is None, "don't need key sequences"
        subseq_mask = torch.triu(torch.ones((q_len, q_len), device=q.device, dtype=torch.uint8), 
                                 diagonal=1)
        subseq_mask = subseq_mask.unsqueeze(0).expand(B, -1, -1)
        return subseq_mask
    else:
        assert False, "error maksing"
    
    
def get_pos(x, pad_idx=1, sos_idx=None, eos_idx=None, pos_pad_idx=0):
    """
    return postion of tensor function
    pos idx(in the vocabulary):
     - pad = 0
     - sos = 1 if exist 
     - eos = 2 if exist (if only eos exist, eos will be 1)
    start idx will be 1 or 2 if one exist or 3 if all exist
    """
    exists = lambda x: x is not None
    
    def process_sos(pos, idx):
        pos[:, 0] = idx
        return pos
    
    def process_eos(pos, x, eos_idx, pos_eos_idx):
        eos_mask = (x == eos_idx)
        pos = (pos+1).masked_fill(eos_mask, pos_eos_idx) 
        return pos
        
    B, T = x.size()
    pos = torch.arange(1, T+1, device=x.device).expand(B, -1)
    if not exists(sos_idx) and exists(eos_idx):
        pos_eos_idx = 1
        pos = process_eos(pos, x, eos_idx, pos_eos_idx)
    elif exists(sos_idx) and exists(eos_idx):
        pos_eos_idx = 2
        pos = process_eos(pos, x, eos_idx, pos_eos_idx)
        pos = process_sos(pos, 1)
        
    padding_mask = (x == pad_idx)
    pos = pos.masked_fill(padding_mask, pos_pad_idx)
    return pos

# Metrics for PyTorch

def cal_bleu(target, predict):
    """calculate bleu score for pytorch"""
    bleu_score = 0
    ref_lists = [[t] for t in target.tolist()]
    pred_list = predict.tolist()
    for ref, pred in zip(ref_lists, pred_list):
        bleu_score += compute_bleu([ref], [pred])[0]
    return bleu_score


# Inference functions

def check_dotproduct_dist(d_k, sampling_size=1, seq_len=1, threshold=1e-10):
    """
    to check "https://arxiv.org/abs/1706.03762" Paper page 4, annotation 4
    -------------------------------
    To illustrate why the dot products get large, 
    assume that the components of q and k are independent random variables 
    with mean 0 and variance 1.
    Then their dot product has mean 0 and variance d_k
    
    ```
    print("*** notice that the gradient of softmax is y(1-y) ***")
    for d_k in [10, 100, 1000]:
        check_dotproduct_dist(d_k, sampling_size=100000, seq_len=5, threshold=1e-10)
    ```
    
    """
    def cal_grad(attn):
        y = torch.softmax(attn, dim=2)
        return y * (1-y)
    
    q = nn.init.normal_(torch.rand((sampling_size, seq_len, d_k)), mean=0, std=1)
    k = nn.init.normal_(torch.rand((sampling_size, seq_len, d_k)), mean=0, std=1)
    attn = torch.bmm(q, k.transpose(1, 2))
    print(f"size of vector d_k is {d_k}, sampling result, dot product distribution has\n")
    print(f" - mean: {attn.mean().item():.4f}, \n - var: {attn.var().item():.4f}")
    grad = cal_grad(attn)
    g_sum = grad.le(threshold).sum()
    g_percent = g_sum.item()/grad.view(-1).size(0)*100
    print(f"count of gradients that smaller than threshod({threshold}) is {g_sum}, {g_percent:.2f}%")
    
    attn2 = attn / torch.sqrt(torch.as_tensor(d_k).float())
    grad2 = cal_grad(attn2)
    g_sum2 = grad2.le(threshold).sum()
    g_percent2 = g_sum2.item()/grad2.view(-1).size(0)*100
    print(f"after divide by sqrt(d_k), count of gradients that smaller than threshod({threshold}) is {g_sum2}, {g_percent2:.2f}% \n")
    
    
def draw_attentions(n_head, attn, tokens=None, return_figs=False):
    """
    to see `n_head` views of attentions
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for k in range(n_head):
        i = k % 2
        j = k // 2
        axes[i, j].matshow(attn[k].squeeze().numpy(), cmap="binary")
        axes[i, j].set_title(f"head: {k}", loc="center", y=1.25, fontsize=20)
        if tokens is not None:
            axes[i, j].set_xticks(list(range(len(tokens[0]))))
            axes[i, j].set_yticks(list(range(len(tokens[1]))))
            axes[i, j].set_xticklabels(tokens[0], rotation=45)
            axes[i, j].set_yticklabels(tokens[1])
    plt.tight_layout()
    plt.show()
    if return_figs:
        return fig
    