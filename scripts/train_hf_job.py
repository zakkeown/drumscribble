# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "drumscribble @ git+https://github.com/zakkeown/drumscribble.git",
#     "huggingface_hub[hf_xet]",
#     "pyyaml",
# ]
# ///
"""HF Jobs training script for DrumscribbleCNN.

Supports E-GMD, STAR, or multi-dataset (both) training.
Downloads datasets from HF Hub, trains DrumscribbleCNN, and uploads
checkpoints back to HF Hub.

Usage (via hf jobs):
    # E-GMD only
    hf jobs uv run scripts/train_hf_job.py \
        --flavor a10g-large --timeout 24h \
        --secret HF_TOKEN=$HF_TOKEN \
        -- --dataset egmd --epochs 100

    # STAR only
    hf jobs uv run scripts/train_hf_job.py \
        --flavor a10g-large --timeout 24h \
        --secret HF_TOKEN=$HF_TOKEN \
        -- --dataset star --epochs 100

    # Multi-dataset (E-GMD + STAR)
    hf jobs uv run scripts/train_hf_job.py \
        --flavor a10g-large --timeout 48h \
        --secret HF_TOKEN=$HF_TOKEN \
        -- --dataset multi --epochs 100
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from huggingface_hub import HfApi, snapshot_download


def download_dataset(repo_id: str, local_dir: str, token: str) -> str:
    """Download a dataset from HF Hub, return local path."""
    print(f"Downloading {repo_id}...")
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        local_dir=local_dir,
    )
    print(f"  -> {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Train DrumscribbleCNN on HF Jobs")
    parser.add_argument("--dataset", type=str, default="egmd",
                        choices=["egmd", "star", "multi"],
                        help="Dataset to train on")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--egmd-repo", type=str, default="zkeown/e-gmd-v1")
    parser.add_argument("--star-repo", type=str, default="zkeown/star-drums-v1")
    parser.add_argument("--output-repo", type=str, default="zkeown/drumscribble-checkpoints")
    parser.add_argument("--dataset-weights", type=float, nargs=2, default=[0.5, 0.5],
                        help="Weights for multi-dataset mode [egmd, star]")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Checkpoint filename in output repo (e.g. checkpoint_epoch10.pt)")
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

    # --- Download dataset(s) ---
    egmd_dir = None
    star_dir = None

    if args.dataset in ("egmd", "multi"):
        egmd_dir = download_dataset(args.egmd_repo, "/tmp/e-gmd", token)

    if args.dataset in ("star", "multi"):
        star_dir = download_dataset(args.star_repo, "/tmp/star-drums", token)

    # --- Create output repo if needed ---
    try:
        api.repo_info(repo_id=args.output_repo, repo_type="model")
    except Exception:
        print(f"Creating output repo {args.output_repo}...")
        api.create_repo(repo_id=args.output_repo, repo_type="model", private=True)

    # --- Import training components ---
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
    model = DrumscribbleCNN(
        backbone_dims=(64, 128, 256, 384),
        backbone_depths=(5, 5, 5, 5),
        num_attn_layers=3,
        num_attn_heads=4,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # --- Dataset & DataLoader ---
    chunk_seconds = args.chunk_seconds
    augment = SpecAugment()
    collate_fn = AugmentCollate(augment)

    if args.dataset == "egmd":
        dataset = EGMDDataset(root=Path(egmd_dir), split="train", chunk_seconds=chunk_seconds)
        print(f"E-GMD training samples: {len(dataset):,}")
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn,
        )
    elif args.dataset == "star":
        dataset = STARDataset(root=Path(star_dir), split="training", chunk_seconds=chunk_seconds)
        print(f"STAR training samples: {len(dataset):,}")
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn,
        )
    elif args.dataset == "multi":
        egmd_ds = EGMDDataset(root=Path(egmd_dir), split="train", chunk_seconds=chunk_seconds)
        star_ds = STARDataset(root=Path(star_dir), split="training", chunk_seconds=chunk_seconds)
        print(f"E-GMD samples: {len(egmd_ds):,}, STAR samples: {len(star_ds):,}")
        print(f"Weights: E-GMD={args.dataset_weights[0]}, STAR={args.dataset_weights[1]}")

        multi_loader = MultiDatasetLoader(
            datasets=[egmd_ds, star_ds],
            batch_size=args.batch_size,
            weights=args.dataset_weights,
            num_workers=args.num_workers,
        )
        # Rebuild with collate_fn
        loader = DataLoader(
            multi_loader.concat,
            batch_size=args.batch_size,
            sampler=multi_loader.loader.sampler,
            num_workers=args.num_workers,
            drop_last=True,
            collate_fn=collate_fn,
        )
        total = len(egmd_ds) + len(star_ds)
        print(f"Multi-dataset total: {total:,} samples")

    # --- Optimizer ---
    optimizer = create_optimizer(model, lr=args.lr, weight_decay=0.05)
    loss_fn = DrumscribbleLoss()

    warmup_epochs = 5
    warmup_steps = warmup_epochs * len(loader)
    total_steps = args.epochs * len(loader)
    scheduler = create_scheduler(optimizer, warmup_steps, total_steps)

    ema = EMAModel(model, decay=0.999)

    # --- AMP ---
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    if scaler:
        print("Using CUDA AMP")

    # --- Resume ---
    start_epoch = 0
    if args.resume_from:
        print(f"Downloading checkpoint {args.resume_from}...")
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

    print(f"\nStarting training: {args.dataset} | epochs {start_epoch+1}-{args.epochs} | "
          f"batch_size={args.batch_size} | lr={args.lr}")
    print(f"Steps/epoch: {len(loader):,} | total steps: {total_steps:,}")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_one_epoch(
            model, loader, optimizer, loss_fn,
            device=device, scheduler=scheduler, scaler=scaler, ema=ema,
        )
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.4f} | lr={lr_now:.2e}")

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
    torch.save({"model": model.state_dict(), "epoch": args.epochs}, final_path)
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
