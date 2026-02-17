"""ConvNeXt blocks and backbone for DrumscribbleCNN."""
import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    """Single ConvNeXt block with BatchNorm (ANE-compatible).

    Architecture: DWConv -> BN -> 1x1 Conv (expand) -> GELU -> 1x1 Conv (project) -> Residual
    """

    def __init__(self, dim: int, kernel_size: tuple[int, int] = (1, 7), expand_ratio: int = 4):
        super().__init__()
        padding = (0, kernel_size[1] // 2)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=padding, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, dim * expand_ratio, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * expand_ratio, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pwconv2(self.act(self.pwconv1(self.norm(self.dwconv(x)))))
