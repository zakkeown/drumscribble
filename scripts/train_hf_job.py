# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "drumscribble @ git+https://github.com/zakkeown/drumscribble.git",
#     "huggingface_hub[hf_xet]",
#     "pyyaml",
# ]
# ///
"""HF Jobs training script for DrumscribbleCNN.

Downloads E-GMD from HF Hub, trains DrumscribbleCNN, and uploads
checkpoints back to HF Hub.

Usage (via hf jobs):
    hf jobs uv run scripts/train_hf_job.py \
        --flavor a10g-large --timeout 24h \
        --secret HF_TOKEN=$HF_TOKEN \
        -- --epochs 100 --batch-size 32
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Train DrumscribbleCNN on HF Jobs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dataset-repo", type=str, default="zkeown/e-gmd-v1")
    parser.add_argument("--output-repo", type=str, default="zkeown/drumscribble-checkpoints")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="HF Hub path to checkpoint (e.g. zkeown/drumscribble-checkpoints/checkpoint_epoch10.pt)")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set. Pass --secret HF_TOKEN=$HF_TOKEN")
        sys.exit(1)

    api = HfApi(token=token)

    # --- Device ---
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        device = "cpu"
        print("WARNING: No GPU detected, training on CPU")
    print(f"Device: {device}")

    # --- Download dataset ---
    print(f"Downloading dataset from {args.dataset_repo}...")
    data_dir = snapshot_download(
        repo_id=args.dataset_repo,
        repo_type="dataset",
        token=token,
        local_dir="/tmp/e-gmd",
    )
    print(f"Dataset downloaded to {data_dir}")

    # --- Create output repo if needed ---
    try:
        api.repo_info(repo_id=args.output_repo, repo_type="model")
    except Exception:
        print(f"Creating output repo {args.output_repo}...")
        api.create_repo(repo_id=args.output_repo, repo_type="model", private=True)

    # --- Build config ---
    config = {
        "model": {
            "n_mels": 128,
            "backbone_dims": [64, 128, 256, 384],
            "backbone_depths": [5, 5, 5, 5],
            "num_attn_layers": 3,
            "num_attn_heads": 4,
        },
        "training": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": 0.05,
            "epochs": args.epochs,
            "grad_clip": 1.0,
            "chunk_seconds": args.chunk_seconds,
            "num_workers": args.num_workers,
            "warmup_epochs": 5,
            "ema_decay": 0.999,
            "dataset": "egmd",
        },
        "data": {
            "egmd_root": data_dir,
        },
    }

    config_path = "/tmp/train_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    print(f"Config written to {config_path}")

    # --- Import training components ---
    from drumscribble.data.augment import SpecAugment
    from drumscribble.data.egmd import EGMDDataset
    from drumscribble.loss import DrumscribbleLoss
    from drumscribble.model.drumscribble import DrumscribbleCNN
    from drumscribble.train import (
        EMAModel,
        create_optimizer,
        create_scheduler,
        train_one_epoch,
    )

    # --- Picklable collate ---
    class AugmentCollate:
        def __init__(self, augment):
            self.augment = augment

        def __call__(self, batch):
            mels, onsets, vels = zip(*batch)
            mel_batch = torch.stack(mels)
            onset_batch = torch.stack(onsets)
            vel_batch = torch.stack(vels)
            mel_batch = self.augment(mel_batch)
            return mel_batch, onset_batch, vel_batch

    # --- Model ---
    model_cfg = config["model"]
    model = DrumscribbleCNN(
        backbone_dims=tuple(model_cfg["backbone_dims"]),
        backbone_depths=tuple(model_cfg["backbone_depths"]),
        num_attn_layers=model_cfg["num_attn_layers"],
        num_attn_heads=model_cfg["num_attn_heads"],
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # --- Dataset & DataLoader ---
    train_cfg = config["training"]
    dataset = EGMDDataset(
        root=Path(data_dir),
        split="train",
        chunk_seconds=train_cfg["chunk_seconds"],
    )
    print(f"Training samples: {len(dataset):,}")

    augment = SpecAugment()
    collate_fn = AugmentCollate(augment)

    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        drop_last=True,
        collate_fn=collate_fn,
    )

    # --- Optimizer ---
    optimizer = create_optimizer(model, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    loss_fn = DrumscribbleLoss()

    epochs = train_cfg["epochs"]
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    warmup_steps = warmup_epochs * len(loader)
    total_steps = epochs * len(loader)
    scheduler = create_scheduler(optimizer, warmup_steps, total_steps)

    ema_decay = train_cfg.get("ema_decay", 0.999)
    ema = EMAModel(model, decay=ema_decay)

    # --- AMP ---
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    if scaler:
        print("Using CUDA AMP")

    # --- Resume ---
    start_epoch = 0
    if args.resume_from:
        print(f"Downloading checkpoint from {args.resume_from}...")
        ckpt_path = api.hf_hub_download(
            repo_id=args.output_repo,
            filename=args.resume_from,
            token=token,
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}")

    # --- Training loop ---
    output_dir = Path("/tmp/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training: epochs {start_epoch+1}-{epochs}, "
          f"batch_size={train_cfg['batch_size']}, lr={train_cfg['lr']}")
    print(f"Steps/epoch: {len(loader)}, total steps: {total_steps}")
    print("=" * 60)

    for epoch in range(start_epoch, epochs):
        avg_loss = train_one_epoch(
            model, loader, optimizer, loss_fn,
            device=device, scheduler=scheduler, scaler=scaler, ema=ema,
        )
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | lr={lr_now:.2e}")

        # Save and upload checkpoint every 10 epochs
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

            # Upload to HF Hub
            api.upload_file(
                path_or_fileobj=str(ckpt_path),
                path_in_repo=f"checkpoint_epoch{epoch+1}.pt",
                repo_id=args.output_repo,
                repo_type="model",
                token=token,
            )
            print(f"Uploaded checkpoint to {args.output_repo}")

    # --- Save final with EMA weights ---
    ema.apply(model)
    final_path = output_dir / "final.pt"
    torch.save({"model": model.state_dict(), "epoch": epochs}, final_path)
    ema.restore(model)

    api.upload_file(
        path_or_fileobj=str(final_path),
        path_in_repo="final.pt",
        repo_id=args.output_repo,
        repo_type="model",
        token=token,
    )
    print(f"\nTraining complete! Final checkpoint (EMA) uploaded to {args.output_repo}")


if __name__ == "__main__":
    main()
