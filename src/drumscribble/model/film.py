"""FiLM (Feature-wise Linear Modulation) for MERT conditioning."""
import torch
import torch.nn as nn


class FiLMConditioning(nn.Module):
    """FiLM: y = gamma * x + beta.

    Projects conditioning features to per-channel gamma and beta.
    Pre-expands to match spatial dims (no broadcasting for ANE).
    """

    def __init__(self, feature_dim: int, target_dim: int):
        super().__init__()
        # Project conditioning to gamma and beta via Conv2d(1x1)
        self.proj = nn.Conv2d(feature_dim, target_dim * 2, kernel_size=1)

    def forward(
        self, x: torch.Tensor, conditioning: torch.Tensor | None
    ) -> torch.Tensor:
        """Apply FiLM conditioning.

        Args:
            x: (B, C, 1, T) feature map.
            conditioning: (B, feature_dim, 1, T) MERT features, or None.

        Returns:
            (B, C, 1, T) modulated features.
        """
        if conditioning is None:
            return x

        # Interpolate conditioning to match x's temporal dimension if needed
        if conditioning.shape[-1] != x.shape[-1]:
            conditioning = torch.nn.functional.interpolate(
                conditioning,
                size=(1, x.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

        params = self.proj(conditioning)  # (B, 2*C, 1, T)
        gamma, beta = params.chunk(2, dim=1)  # each (B, C, 1, T)

        # Element-wise (same shape, no broadcasting needed)
        return gamma * x + beta
