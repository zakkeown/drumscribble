"""ANE-optimized self-attention following Apple's ml-ane-transformers pattern."""
import torch
import torch.nn as nn


class ANESelfAttention(nn.Module):
    """Self-attention using Conv2d(1x1) projections for ANE compatibility.

    Operates on (B, C, 1, S) tensors. All Linear replaced with Conv2d.
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        B, C, _, S = x.shape

        qkv = self.qkv(x)  # (B, 3*C, 1, S)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, S)
        q, k, v = qkv.unbind(1)  # each (B, H, D, S)

        # Attention computation
        attn = torch.einsum("bhds,bhdt->bhst", q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum("bhst,bhdt->bhds", attn, v)
        out = out.reshape(B, C, 1, S)
        out = self.proj(out)
        out = self.norm(out)

        return out + residual
