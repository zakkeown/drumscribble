"""Training loop for DrumscribbleCNN."""
import copy
import math
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from drumscribble.model.drumscribble import DrumscribbleCNN
from drumscribble.loss import DrumscribbleLoss


def create_optimizer(
    model: DrumscribbleCNN,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine schedule with linear warmup.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of steps for linear warmup.
        total_steps: Total number of training steps.

    Returns:
        LambdaLR scheduler with linear warmup then cosine decay.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(warmup_steps, 1)
        progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMAModel:
    """Exponential moving average of model parameters.

    Maintains a shadow copy of model parameters that is updated as an
    exponential moving average after each training step. Useful for
    stabilizing training and improving generalization.

    Args:
        model: The model whose parameters to track.
        decay: EMA decay factor. Higher values give more weight to history.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Update EMA parameters with current model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self, model: torch.nn.Module) -> None:
        """Copy EMA parameters to model, backing up originals."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module) -> None:
        """Restore original model parameters from backup."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def train_one_epoch(
    model: DrumscribbleCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: DrumscribbleLoss,
    device: str = "cpu",
    grad_clip: float = 1.0,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    ema: Optional[EMAModel] = None,
) -> float:
    """Train for one epoch.

    Args:
        model: The model to train.
        loader: Training data loader.
        optimizer: Optimizer.
        loss_fn: Loss function.
        device: Device string.
        grad_clip: Max gradient norm for clipping.
        scheduler: Optional LR scheduler (stepped per batch).
        scaler: Optional GradScaler for CUDA AMP.
        ema: Optional EMA model tracker.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    use_amp = scaler is not None

    for mel, onset_target, vel_target in tqdm(loader, desc="Training", leave=False):
        mel = mel.to(device)
        onset_target = onset_target.to(device)
        vel_target = vel_target.to(device)

        if mel.dim() == 3:
            mel = mel.unsqueeze(1)

        if use_amp:
            with torch.amp.autocast("cuda"):
                onset_pred, vel_pred, offset_pred = model(mel)
                loss, _ = loss_fn(
                    onset_pred, vel_pred, offset_pred, onset_target, vel_target
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            onset_pred, vel_pred, offset_pred = model(mel)
            loss, _ = loss_fn(
                onset_pred, vel_pred, offset_pred, onset_target, vel_target
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)
