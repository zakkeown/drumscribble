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


class ConvNeXtBackbone(nn.Module):
    """4-stage ConvNeXt encoder with frequency-collapsing stem.

    Input: (B, 1, 128, T) mel spectrogram
    Output: (B, 384, 1, T/8) feature map + 3 skip connections
    """

    def __init__(
        self,
        n_mels: int = 128,
        dims: tuple[int, ...] = (64, 128, 256, 384),
        depths: tuple[int, ...] = (5, 5, 5, 5),
        kernels: tuple[tuple[int, int], ...] = ((1, 7), (1, 7), (1, 11), (1, 11)),
    ):
        super().__init__()
        self.stem = nn.Conv2d(1, dims[0], kernel_size=(n_mels, 1), stride=(n_mels, 1))
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dims[i], kernels[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)
            if i < 3:
                downsample = nn.Sequential(
                    nn.BatchNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=(1, 2), stride=(1, 2)),
                )
                self.downsamples.append(downsample)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.stem(x)  # (B, 64, 1, T)
        skips = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < 3:
                skips.append(x)
                x = self.downsamples[i](x)
        return x, skips
