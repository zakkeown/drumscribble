"""Training CLI for DrumscribbleCNN."""
import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from drumscribble.data.egmd import EGMDDataset
from drumscribble.loss import DrumscribbleLoss
from drumscribble.model.drumscribble import DrumscribbleCNN
from drumscribble.train import create_optimizer, train_one_epoch


def main():
    parser = argparse.ArgumentParser(description="Train DrumscribbleCNN")
    parser.add_argument("--config", type=str, default="configs/train/default.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    model = DrumscribbleCNN(
        backbone_dims=tuple(model_cfg["backbone_dims"]),
        backbone_depths=tuple(model_cfg["backbone_depths"]),
        num_attn_layers=model_cfg["num_attn_layers"],
        num_attn_heads=model_cfg["num_attn_heads"],
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    if device == "mps":
        model.freeze_bn()
        print("Frozen BatchNorm for MPS training")

    dataset = EGMDDataset(
        root=Path(data_cfg["egmd_root"]).expanduser(),
        split="train",
        chunk_seconds=train_cfg["chunk_seconds"],
    )
    print(f"Training samples: {len(dataset):,}")

    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=0 if device == "mps" else train_cfg["num_workers"],
        drop_last=True,
    )

    optimizer = create_optimizer(
        model, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    loss_fn = DrumscribbleLoss()

    epochs = args.epochs or train_cfg["epochs"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, loader, optimizer, loss_fn, device=device)
        print(f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch + 1}, ckpt_path)
            print(f"Saved {ckpt_path}")

    final_path = output_dir / "final.pt"
    torch.save({"model": model.state_dict(), "epoch": epochs}, final_path)
    print(f"Training complete. Saved {final_path}")


if __name__ == "__main__":
    main()
