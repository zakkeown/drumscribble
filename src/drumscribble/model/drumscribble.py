"""DrumscribbleCNN: U-Net encoder-decoder with attention bottleneck."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from drumscribble.config import NUM_CLASSES
from drumscribble.model.convnext import ConvNeXtBackbone, ConvNeXtBlock
from drumscribble.model.attention import ANESelfAttention
from drumscribble.model.film import FiLMConditioning


class UNetDecoderBlock(nn.Module):
    """Single U-Net decoder stage: upsample -> concat skip -> fuse -> ConvNeXt block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, kernel: tuple[int, int] = (1, 7)):
        super().__init__()
        self.fuse = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=1)
        self.block = ConvNeXtBlock(out_ch, kernel)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Interpolate to match skip's spatial dims (handles non-power-of-2 sizes)
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.block(x)
        return x


class DrumscribbleCNN(nn.Module):
    """SOTA drum transcription model for CoreML/ANE deployment.

    U-Net encoder-decoder with attention bottleneck:
    - Encoder: ConvNeXt backbone with 3 temporal downsamplings (T -> T/8)
    - Bottleneck: FiLM conditioning + ANE self-attention at T/8
    - Decoder: 3 upsampling stages with skip connections (T/8 -> T)

    Input: (B, 1, 128, T) log-mel spectrogram
    Output: tuple of 3 tensors, each (B, 26, T):
        - onset probabilities
        - velocity estimates [0, 1]
        - offset probabilities
    """

    def __init__(
        self,
        n_mels: int = 128,
        backbone_dims: tuple[int, ...] = (64, 128, 256, 384),
        backbone_depths: tuple[int, ...] = (5, 5, 5, 5),
        num_attn_layers: int = 3,
        num_attn_heads: int = 4,
        mert_dim: int | None = None,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        d = backbone_dims
        hidden = d[-1]  # 384

        # Encoder
        self.backbone = ConvNeXtBackbone(
            n_mels=n_mels, dims=d, depths=backbone_depths,
        )

        # Bottleneck: FiLM + attention at T/8
        self.film = FiLMConditioning(mert_dim, hidden) if mert_dim else None
        self.attention = nn.Sequential(
            *[ANESelfAttention(hidden, num_attn_heads) for _ in range(num_attn_layers)]
        )

        # Decoder: 3 upsampling stages with skip connections
        self.decoder1 = UNetDecoderBlock(d[3], d[2], d[2])  # 384+256->256, T/8->T/4
        self.decoder2 = UNetDecoderBlock(d[2], d[1], d[1])  # 256+128->128, T/4->T/2
        self.decoder3 = UNetDecoderBlock(d[1], d[0], d[0])  # 128+64->64,   T/2->T

        # Output heads (operate on decoder output = dims[0] channels)
        self.head_proj = nn.Sequential(
            nn.BatchNorm2d(d[0]),
            nn.Conv2d(d[0], 128, 1),
            nn.GELU(),
        )
        self.onset_head = nn.Conv2d(128, num_classes, 1)
        self.velocity_head = nn.Conv2d(128, num_classes, 1)
        self.offset_head = nn.Conv2d(128, num_classes, 1)

    def forward(
        self,
        x: torch.Tensor,
        mert_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encoder: (B, 1, 128, T) -> (B, 384, 1, T/8), skips at T, T/2, T/4
        features, skips = self.backbone(x)

        # Bottleneck: optional MERT conditioning + self-attention at T/8
        if self.film is not None:
            features = self.film(features, mert_features)
        features = self.attention(features)

        # Decoder: recover full T resolution via skip connections
        features = self.decoder1(features, skips[2])  # T/8 -> T/4
        features = self.decoder2(features, skips[1])  # T/4 -> T/2
        features = self.decoder3(features, skips[0])  # T/2 -> T

        # Output heads (raw logits — apply sigmoid at inference time)
        h = self.head_proj(features)
        onset = self.onset_head(h).squeeze(2)      # (B, 26, T)
        velocity = self.velocity_head(h).squeeze(2)
        offset = self.offset_head(h).squeeze(2)

        return onset, velocity, offset

    def freeze_bn(self):
        """Freeze all BatchNorm layers (use running stats). For local MPS training."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
