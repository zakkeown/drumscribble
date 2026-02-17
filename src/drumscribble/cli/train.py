"""Training CLI for DrumscribbleCNN."""
import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from drumscribble.data.augment import SpecAugment
from drumscribble.data.egmd import EGMDDataset
from drumscribble.data.multi import MultiDatasetLoader
from drumscribble.data.star import STARDataset
from drumscribble.loss import DrumscribbleLoss
from drumscribble.model.drumscribble import DrumscribbleCNN
from drumscribble.train import (
    EMAModel,
    create_optimizer,
    create_scheduler,
    train_one_epoch,
)


def make_augment_collate(augment: SpecAugment):
    """Create a collate function that applies SpecAugment to mel batches.

    Args:
        augment: SpecAugment transform to apply.

    Returns:
        Collate function for use with DataLoader.
    """

    def collate_fn(batch):
        mels, onsets, vels = zip(*batch)
        mel_batch = torch.stack(mels)
        onset_batch = torch.stack(onsets)
        vel_batch = torch.stack(vels)
        mel_batch = augment(mel_batch)
        return mel_batch, onset_batch, vel_batch

    return collate_fn


def build_dataset(dataset_name: str, data_cfg: dict, train_cfg: dict):
    """Build dataset(s) based on config.

    Args:
        dataset_name: One of "egmd", "star", or "multi".
        data_cfg: Data configuration dict with root paths.
        train_cfg: Training configuration dict.

    Returns:
        Dataset or list of datasets for multi mode.
    """
    chunk_seconds = train_cfg["chunk_seconds"]

    if dataset_name == "egmd":
        return EGMDDataset(
            root=Path(data_cfg["egmd_root"]).expanduser(),
            split="train",
            chunk_seconds=chunk_seconds,
        )
    elif dataset_name == "star":
        return STARDataset(
            root=Path(data_cfg["star_root"]).expanduser(),
            split="training",
            chunk_seconds=chunk_seconds,
        )
    elif dataset_name == "multi":
        egmd = EGMDDataset(
            root=Path(data_cfg["egmd_root"]).expanduser(),
            split="train",
            chunk_seconds=chunk_seconds,
        )
        star = STARDataset(
            root=Path(data_cfg["star_root"]).expanduser(),
            split="training",
            chunk_seconds=chunk_seconds,
        )
        return [egmd, star]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="Train DrumscribbleCNN")
    parser.add_argument("--config", type=str, default="configs/train/default.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["egmd", "star", "multi"],
        help="Dataset to use (overrides config)",
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

    # --- Dataset & DataLoader ---
    dataset_name = args.dataset or train_cfg.get("dataset", "egmd")
    augment = SpecAugment()
    collate_fn = make_augment_collate(augment)
    num_workers = 0 if device == "mps" else train_cfg["num_workers"]

    if dataset_name == "multi":
        datasets = build_dataset("multi", data_cfg, train_cfg)
        weights = train_cfg.get("dataset_weights", [0.5, 0.5])
        multi_loader = MultiDatasetLoader(
            datasets=datasets,
            batch_size=train_cfg["batch_size"],
            weights=weights,
            num_workers=num_workers,
        )
        loader = multi_loader.loader
        # Patch collate into the existing loader by rebuilding
        loader = DataLoader(
            multi_loader.concat,
            batch_size=train_cfg["batch_size"],
            sampler=loader.sampler,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
        )
        total_samples = sum(len(d) for d in datasets)
        print(f"Multi-dataset training: {total_samples:,} total samples")
        for i, ds in enumerate(datasets):
            print(f"  Dataset {i}: {len(ds):,} samples, weight={weights[i]}")
    else:
        dataset = build_dataset(dataset_name, data_cfg, train_cfg)
        print(f"Training samples ({dataset_name}): {len(dataset):,}")
        loader = DataLoader(
            dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
        )

    # --- Optimizer ---
    optimizer = create_optimizer(
        model, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    loss_fn = DrumscribbleLoss()

    epochs = args.epochs or train_cfg["epochs"]

    # --- Scheduler ---
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    warmup_steps = warmup_epochs * len(loader)
    total_steps = epochs * len(loader)
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
