"""Loss functions for DrumscribbleCNN."""

import torch
import torch.nn.functional as F


class DrumscribbleLoss(torch.nn.Module):
    """Combined onset BCE + masked velocity MSE + offset BCE."""

    def __init__(self, velocity_weight: float = 0.5):
        super().__init__()
        self.velocity_weight = velocity_weight

    def forward(
        self,
        onset_pred: torch.Tensor,
        vel_pred: torch.Tensor,
        offset_pred: torch.Tensor,
        onset_target: torch.Tensor,
        vel_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Clamp predictions for numerical stability (prevents log(0) in BCE)
        onset_pred = onset_pred.clamp(1e-7, 1 - 1e-7)
        offset_pred = offset_pred.clamp(1e-7, 1 - 1e-7)

        # Onset BCE
        onset_loss = F.binary_cross_entropy(onset_pred, onset_target, reduction="mean")

        # Offset BCE (reuse onset targets)
        offset_loss = F.binary_cross_entropy(offset_pred, onset_target, reduction="mean")

        # Masked velocity MSE (only where onset_target >= 1.0)
        mask = (onset_target >= 1.0).float()
        if mask.sum() > 0:
            vel_loss = ((vel_pred - vel_target) ** 2 * mask).sum() / mask.sum()
        else:
            vel_loss = torch.tensor(0.0, device=onset_pred.device)

        total = onset_loss + offset_loss + self.velocity_weight * vel_loss

        return total, {
            "onset": onset_loss.detach(),
            "velocity": vel_loss.detach(),
            "offset": offset_loss.detach(),
        }
