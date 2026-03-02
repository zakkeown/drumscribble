"""Training CLI for DrumscribbleCNN."""
import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from drumscribble.data.augment import SpecAugment
from drumscribble.data.parquet_loader import create_parquet_pipeline
from drumscribble.data.webdataset_loader import create_webdataset_pipeline
from drumscribble.loss import DrumscribbleLoss
from drumscribble.model.drumscribble import DrumscribbleCNN
from drumscribble.train import (
    EMAModel,
    create_optimizer,
    create_scheduler,
    train_one_epoch,
)


class AugmentCollate:
    """Picklable collate function that applies SpecAugment to mel batches."""

    def __init__(self, augment: SpecAugment):
        self.augment = augment

    def __call__(self, batch):
        mels, onsets, vels = zip(*batch)
        mel_batch = torch.stack(mels)
        onset_batch = torch.stack(onsets)
        vel_batch = torch.stack(vels)
        mel_batch = self.augment(mel_batch)
        return mel_batch, onset_batch, vel_batch


def main():
    parser = argparse.ArgumentParser(description="Train DrumscribbleCNN")
    parser.add_argument("--config", type=str, default="configs/train/default.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["parquet", "webdataset"],
        default=None,
        help="Data backend (overrides config, default: parquet)",
    )
    parser.add_argument(
        "--parquet-root",
        type=str,
        default=None,
        help="Root directory for parquet datasets (overrides config)",
    )
    parser.add_argument(
        "--shard-root",
        type=str,
        default=None,
        help="Root directory for feature shards (overrides config)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Dataset names to train on (overrides config)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
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

    # --- Dataset ---
    backend = args.backend or data_cfg.get("backend", "parquet")
    shuffle_buffer = data_cfg.get("shuffle_buffer", 5000)

    if backend == "parquet":
        parquet_root = args.parquet_root or data_cfg.get("parquet_root", "~/Documents/Datasets/hf_cache")
        datasets = args.datasets or data_cfg.get("parquet_datasets", data_cfg["datasets"])
        pipeline = create_parquet_pipeline(
            data_root=parquet_root,
            datasets=datasets,
            split="train",
            shuffle=True,
            shuffle_buffer=shuffle_buffer,
        )
        print(f"Backend: parquet")
        print(f"Data root: {parquet_root}")
    else:
        shard_root = args.shard_root or data_cfg["shard_root"]
        datasets = args.datasets or data_cfg["datasets"]
        pipeline = create_webdataset_pipeline(
            shard_root=shard_root,
            datasets=datasets,
            split="train",
            shuffle=True,
            shuffle_buffer=shuffle_buffer,
        )
        print(f"Backend: webdataset")
        print(f"Shard root: {shard_root}")
    print(f"Training datasets: {', '.join(datasets)}")

    augment = SpecAugment()
    collate_fn = AugmentCollate(augment)
    num_workers = 0 if device == "mps" else train_cfg["num_workers"]

    loader = DataLoader(
        pipeline,
        batch_size=train_cfg["batch_size"],
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # --- Optimizer ---
    optimizer = create_optimizer(
        model, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    loss_fn = DrumscribbleLoss()

    epochs = args.epochs if args.epochs is not None else train_cfg["epochs"]

    # --- Scheduler ---
    # WebDataset IterableDataset doesn't have len(); use config estimate
    estimated_samples = data_cfg.get("estimated_samples", 35000)
    estimated_batches = estimated_samples // train_cfg["batch_size"]
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    warmup_steps = warmup_epochs * estimated_batches
    total_steps = epochs * estimated_batches
    scheduler = create_scheduler(optimizer, warmup_steps, total_steps)

    # --- EMA ---
    ema_decay = train_cfg.get("ema_decay", 0.999)
    ema = EMAModel(model, decay=ema_decay)

    # --- AMP (CUDA only) ---
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    if scaler is not None:
        print("Using CUDA AMP")

    # --- Resume from checkpoint ---
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    for epoch in range(start_epoch, epochs):
        avg_loss = train_one_epoch(
            model,
            loader,
            optimizer,
            loss_fn,
            device=device,
            scheduler=scheduler,
            scaler=scaler,
            ema=ema,
        )
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | lr={lr_now:.2e}"
        )

        if (epoch + 1) % 10 == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
            ckpt_data = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
            }
            if scaler is not None:
                ckpt_data["scaler"] = scaler.state_dict()
            torch.save(ckpt_data, ckpt_path)
            print(f"Saved {ckpt_path}")

    # Save final checkpoint with EMA weights
    ema.apply(model)
    final_path = output_dir / "final.pt"
    torch.save({"model": model.state_dict(), "epoch": epochs}, final_path)
    ema.restore(model)
    print(f"Training complete. Saved {final_path} (EMA weights)")


if __name__ == "__main__":
    main()
