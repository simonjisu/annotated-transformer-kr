import torch
import torch.nn as nn

# Used in the model

def get_padding_mask(q, k=None, pad_idx=1, mode='attn'):
    """
    mode: attn
    > `pad_idx` is vocab pad_idx 
    > mask out for pad in attention with queries & keys sequences
    > return shape: (B, T_q, T_k)
    mode: nonpad
    > `pad_idx` is vocab pad_idx 
    > mask out pad rows in attention
    > return shape: (B, T_q, T_k)
    mode: subseq
    > mask out next tokens to preserve 'auto-regressive property'
    > return shape: (B, T_q, T_q)
    """
    B, q_len = q.size()
    if mode == 'attn':
        assert k is not None, "must have key sequences"
        padding_mask = k.eq(pad_idx)
        padding_mask = padding_mask.unsqueeze(1).expand(B, q_len, -1)
        return padding_mask
    elif mode == 'nonpad':
        # to mask out pad rows
        assert k is None, "don't need key sequences"
        return q.ne(pad_idx).type(torch.float).unsqueeze(-1)
    elif mode =='subseq':
        assert k is None, "don't need key sequences"
        subseq_mask = torch.triu(torch.ones((q_len, q_len), device=q.device, dtype=torch.uint8), 
                                 diagonal=1)
        subseq_mask = subseq_mask.unsqueeze(0).expand(B, -1, -1)
        return subseq_mask
    
    
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
    
    
def draw_attentions(n_head, attn):
    """
    to see `n_head` views of attentions
    """
    fig, axes = plt.subplots(2, 4, dpi=100)
    for k in range(n_head):
        i = k % 2
        j = k // 2
        axes[i, j].matshow(attn[k].squeeze().numpy(), cmap="binary")
        axes[i, j].set_title(f"head: {k}", loc="center", y=1.5)
    plt.tight_layout()
    plt.show()