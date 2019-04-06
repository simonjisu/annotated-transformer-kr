import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_padding_mask

class LabelSmoothing(nn.Module):
    """Label Smoothing"""
    def __init__(self, trg_vocab_size, pad_idx, eps=0.0):
        """
        - eps: smoothing parameters, set it 0.0 to use crossentropy
        
        -------------------------------------------------
        ref: https://arxiv.org/pdf/1512.00567.pdf
        ref: http://nlp.seas.harvard.edu/2018/04/03/attention.html
        #7
        - k : predict class
        - y : target class 
        In Cross Entropy H(q, p) = -\sum q(k|x) log p(k|x)
        replace ground truth distribution (q(k|x) = \delta_{k, y}) to
        q'(k|x) = (1 - eps) q(k|x) + eps * u(k)
        where: q(k|x) = 1 if k == y else 0, u(k) = unifrom distribution to labels (ex. 1/K)
        
        Equal to H(q', p) = H(q, p) + eps/(1-eps) * (D_KL(u || p) + H(u))
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='sum')
        self.pad_idx = pad_idx
        self.eps = eps
        self.trg_vocab_size = trg_vocab_size
        
    def forward(self, x, target):
        """
        Inputs:
        x: (B, T_d, V_target), predict scores
        t: (B, T_d)
        """
        assert x.size(2) == self.trg_vocab_size, \
            'vocab size is not equal x: {}, vocab: {}'.format(x.size(1), self.trg_vocab_size)
        assert target.dim() == 2, 't must be size of (B, T_q)'
        
        # option to not use label smoothing
        if self.eps == 0.0:
            x = x.view(-1, x.size(-1))
            return self.criterion(x, target.view(-1))
        
        non_pad_mask = get_padding_mask(q=target, pad_idx=1, mode='nonpad')

        # label smoothing
        delta_ky = torch.zeros_like(x)
        delta_ky = delta_ky.scatter(dim=2, index=target.unsqueeze(-1), source=1)
        # -2: exclude <s>, </s> token
        # smoothed_dist: B, T_d, vocab_size
        smoothed_dist = (1 - self.eps) * delta_ky + self.eps / (self.trg_vocab_size - 2)
        smoothed_dist *= non_pad_mask
        log_prob = F.log_softmax(x, dim=2)
        loss = (-(smoothed_dist * log_prob)).sum()
        self.smoothed_dist = smoothed_dist
        return loss