"""Quick overfitting test: train on 1 batch, verify loss goes to ~0."""
import torch
from drumscribble.model.drumscribble import DrumscribbleCNN
from drumscribble.loss import DrumscribbleLoss


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Tiny model for speed
    model = DrumscribbleCNN(
        backbone_dims=(32, 64, 64, 64),
        backbone_depths=(2, 2, 2, 2),
        num_attn_layers=1,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Synthetic batch: 1 sample, 5s
    mel = torch.randn(1, 1, 128, 312).to(device)
    onset_target = torch.zeros(1, 26, 312).to(device)
    vel_target = torch.zeros(1, 26, 312).to(device)

    # Place a few onsets
    onset_target[0, 0, 50] = 1.0  # kick
    onset_target[0, 5, 100] = 1.0  # snare
    vel_target[0, 0, 50] = 0.8
    vel_target[0, 5, 100] = 0.6

    loss_fn = DrumscribbleLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(200):
        onset_pred, vel_pred, offset_pred = model(mel)
        loss, components = loss_fn(onset_pred, vel_pred, offset_pred, onset_target, vel_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(
                f"Step {step:3d} | loss={loss.item():.4f} "
                f"onset={components['onset'].item():.4f} "
                f"vel={components['velocity'].item():.4f}"
            )

    assert loss.item() < 0.1, f"Failed to overfit: loss={loss.item()}"
    print("\nOverfitting test PASSED")


if __name__ == "__main__":
    main()
