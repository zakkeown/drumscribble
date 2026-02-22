"""Loss functions for DrumscribbleCNN."""

import torch
import torch.nn.functional as F


class DrumscribbleLoss(torch.nn.Module):
    """Combined onset BCE + masked velocity MSE + offset BCE.

    Expects raw logits from the model (no sigmoid applied).
    Uses binary_cross_entropy_with_logits which is numerically stable
    and safe under AMP autocast.
    """

    def __init__(self, velocity_weight: float = 0.5):
        super().__init__()
        self.velocity_weight = velocity_weight

    def forward(
        self,
        onset_logits: torch.Tensor,
        vel_logits: torch.Tensor,
        offset_logits: torch.Tensor,
        onset_target: torch.Tensor,
        vel_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Onset BCE (with logits — numerically stable, autocast-safe)
        onset_loss = F.binary_cross_entropy_with_logits(
            onset_logits, onset_target, reduction="mean"
        )

        # Offset BCE (reuse onset targets)
        offset_loss = F.binary_cross_entropy_with_logits(
            offset_logits, onset_target, reduction="mean"
        )

        # Masked velocity MSE (only where onset_target >= 1.0)
        vel_pred = vel_logits.sigmoid()
        mask = (onset_target >= 1.0).float()
        if mask.sum() > 0:
            vel_loss = ((vel_pred - vel_target) ** 2 * mask).sum() / mask.sum()
        else:
            vel_loss = torch.tensor(0.0, device=onset_logits.device)

        total = onset_loss + offset_loss + self.velocity_weight * vel_loss

        return total, {
            "onset": onset_loss.detach(),
            "velocity": vel_loss.detach(),
            "offset": offset_loss.detach(),
        }
