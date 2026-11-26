"""Paper-sized Transformer and a kernel-linear/softmax hybrid.

The short-sequence implementation materializes the causal kernel matrix. It
is equivalent to normalized recurrent linear attention, not softmax attention.
It is deliberately not an implementation of any proprietary LLM architecture.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def rope(x):
    length, width = x.shape[-2:]
    inv = 10000.0 ** (-torch.arange(0, width, 2, device=x.device).float() / width)
    angle = torch.arange(length, device=x.device).float()[:, None] * inv[None, :]
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


def linear_attention(q, k, v, blocked=None):
    """phi(q)=ELU(q)+1; dense evaluation of the causal normalized kernel."""
    q, k = F.elu(q.float()) + 1.0, F.elu(k.float()) + 1.0
    score = q @ k.transpose(-1, -2)
    causal = torch.ones(score.shape[-2:], dtype=torch.bool, device=score.device).tril()
    score = score.masked_fill(~causal, 0)
    if blocked is not None:
        rows = torch.arange(score.shape[0], device=score.device)
        score[rows, :, blocked[:, 0], blocked[:, 1]] = 0
    prob = score / score.sum(-1, keepdim=True).clamp_min(1e-6)
    return (prob @ v.float()).to(v.dtype)


class Mixer(nn.Module):
    def __init__(self, width=256, heads=8):
        super().__init__()
        self.width, self.heads = width, heads
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.kind = 'softmax'
        self.conv = None

    def forward(self, x, blocked=None):
        batch, length, _ = x.shape
        qkv = self.qkv(x)
        if self.conv is not None:
            qkv = self.conv(qkv.transpose(1, 2))[..., :length].transpose(1, 2)
        q, k, v = qkv.view(batch, length, 3, self.heads, self.width // self.heads).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        q, k = rope(q), rope(k)
        if self.kind == 'linear':
            out = linear_attention(q, k, v, blocked)
        elif blocked is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            mask = torch.ones(batch, 1, length, length, device=x.device, dtype=torch.bool).tril()
            rows = torch.arange(batch, device=x.device)
            mask[rows, 0, blocked[:, 0], blocked[:, 1]] = False
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(out.transpose(1, 2).reshape(batch, length, self.width))


class Block(nn.Module):
    def __init__(self, width=256, heads=8):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.mixer = Mixer(width, heads)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, x, blocked=None):
        x = x + self.mixer(self.ln1(x), blocked)
        return x + self.mlp(self.ln2(x))


class ToyModel(nn.Module):
    def __init__(self, architecture='transformer', width=256, heads=8, layers=4):
        super().__init__()
        if architecture not in ['transformer', 'hybrid']:
            raise ValueError(architecture)
        self.embedding = nn.Embedding(336, width)
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 336)
        # Initialize additions last: common parameter initializations match
        # exactly across architectures given the same torch seed.
        if architecture == 'hybrid':
            for idx in range(0, layers, 2):
                mixer = self.blocks[idx].mixer
                mixer.kind = 'linear'
                mixer.conv = nn.Conv1d(3 * width, 3 * width, 3, padding=2, groups=3 * width)

    def forward(self, tokens, blocked=None):
        x = self.embedding(tokens)
        for idx, block in enumerate(self.blocks):
            x = block(x, blocked if idx == 0 else None)
        return self.head(self.norm(x))


def weighted_loss(logits, labels, weights):
    loss = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction='none')
    return (loss.view_as(weights) * weights).sum() / weights.sum().clamp_min(1)
