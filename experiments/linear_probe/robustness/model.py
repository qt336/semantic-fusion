"""The plain 4-layer Transformer from the September reconstruction."""
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


class Mixer(nn.Module):
    def __init__(self, width=256, heads=8):
        super().__init__()
        self.width, self.heads = width, heads
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)

    def forward(self, x):
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.heads, self.width // self.heads).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)
        return self.out(out.transpose(1, 2).reshape(batch, length, self.width))


class Block(nn.Module):
    def __init__(self, width=256, heads=8):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.mixer = Mixer(width, heads)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, x):
        x = x + self.mixer(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ToyModel(nn.Module):
    def __init__(self, width=256, heads=8, layers=4):
        super().__init__()
        self.embedding = nn.Embedding(336, width)
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 336)

    def forward(self, tokens):
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))
