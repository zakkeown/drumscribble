"""Training loop for DrumscribbleCNN."""
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


def train_one_epoch(
    model: DrumscribbleCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: DrumscribbleLoss,
    device: str = "cpu",
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for mel, onset_target, vel_target in tqdm(loader, desc="Training", leave=False):
        mel = mel.to(device)
        onset_target = onset_target.to(device)
        vel_target = vel_target.to(device)

        if mel.dim() == 3:
            mel = mel.unsqueeze(1)

        onset_pred, vel_pred, offset_pred = model(mel)

        loss, _ = loss_fn(onset_pred, vel_pred, offset_pred, onset_target, vel_target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)
