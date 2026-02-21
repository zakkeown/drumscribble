# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "drumscribble @ git+https://github.com/zakkeown/drumscribble.git",
#     "datasets",
#     "huggingface_hub[hf_xet]",
#     "pyyaml",
#     "trackio",
# ]
# ///
"""HF Jobs training script for DrumscribbleCNN (pre-computed features).

Loads pre-computed mel/onset/velocity features from HF datasets repos
via `datasets.load_dataset()` + `FeaturesDataset`. Much faster than the
old raw-audio pipeline (5-10 GB download instead of 90 GB+, no mel
computation at train time).

Supports E-GMD, STAR, or multi-dataset (both) training.
Uploads checkpoints to HF Hub with Trackio experiment tracking.

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
from huggingface_hub import HfApi


def load_features(repo_id: str, split: str, token: str | None) -> list[dict]:
    """Load pre-computed features from an HF dataset repo.

    Args:
        repo_id: HF dataset repo (e.g. 'schismaudio/e-gmd').
        split: Dataset split ('train' or 'validation').
        token: HF API token (needed for private repos).

    Returns:
        List of dicts with keys: mel_spectrogram, onset_targets,
        velocity_targets, n_frames.
    """
    import datasets

    print(f"Loading {repo_id} split={split}...")
    ds = datasets.load_dataset(
        repo_id,
        data_files=f"features/{split}-*.parquet",
        split="train",  # data_files loads everything as "train" split
        token=token,
    )
    rows = list(ds)
    print(f"  Loaded {len(rows)} rows from {repo_id}/{split}")
    return rows


def validate(
    model: torch.nn.Module,
    val_rows: list[dict],
    device: str,
    chunk_frames: int,
) -> dict[str, float]:
    """Run validation and compute micro-averaged onset F1.

    Args:
        model: DrumscribbleCNN (should already have EMA weights applied).
        val_rows: List of feature dicts for validation split.
        device: Device string ('cuda' or 'cpu').
        chunk_frames: Number of frames per chunk.

    Returns:
        Dict with val_f1, val_precision, val_recall.
    """
    from torch.utils.data import DataLoader

    from drumscribble.config import FPS, GM_CLASSES
    from drumscribble.data.features import FeaturesDataset
    from drumscribble.evaluate import evaluate_events
    from drumscribble.inference import detections_to_events

    val_ds = FeaturesDataset(val_rows, chunk_frames=chunk_frames)
    if len(val_ds) == 0:
        print("  WARNING: No validation chunks available")
        return {"val_f1": 0.0, "val_precision": 0.0, "val_recall": 0.0}

    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    all_ref_events: list[dict] = []
    all_est_events: list[dict] = []

    model.eval()
    with torch.no_grad():
        for mel, onset_target, vel_target in val_loader:
            mel = mel.to(device)
            onset_target = onset_target.to(device)

            if mel.dim() == 3:
                mel = mel.unsqueeze(1)

            onset_pred, vel_pred, _ = model(mel)

            # Process each example in the batch
            for i in range(mel.shape[0]):
                # Reference events: peaks in onset_target where target >= 1.0
                ref_events = []
                ot = onset_target[i]  # (NUM_CLASSES, T)
                for cls_idx in range(ot.shape[0]):
                    frames = torch.where(ot[cls_idx] >= 1.0)[0]
                    for f in frames:
                        ref_events.append({
                            "time": f.item() / FPS,
                            "note": GM_CLASSES[cls_idx],
                        })

                # Estimated events via inference pipeline
                est_events = detections_to_events(
                    onset_pred[i].sigmoid(),
                    vel_pred[i].sigmoid(),
                    threshold=0.5,
                    fps=FPS,
                )

                all_ref_events.extend(ref_events)
                all_est_events.extend(est_events)

    # Compute micro-averaged metrics
    metrics = evaluate_events(all_ref_events, all_est_events)
    return {
        "val_f1": metrics["f1"],
        "val_precision": metrics["precision"],
        "val_recall": metrics["recall"],
    }


def main():
    parser = argparse.ArgumentParser(description="Train DrumscribbleCNN on HF Jobs")
    parser.add_argument("--dataset", type=str, default="egmd",
                        choices=["egmd", "star", "multi"],
                        help="Dataset to train on")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--chunk-seconds", type=float, default=10.0,
                        help="Duration of each training chunk in seconds")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--egmd-repo", type=str, default="schismaudio/e-gmd")
    parser.add_argument("--star-repo", type=str, default="zkeown/star-drums")
    parser.add_argument("--output-repo", type=str, default="schismaudio/drumscribble-checkpoints")
    parser.add_argument("--dataset-weights", type=float, nargs=2, default=[0.5, 0.5],
                        help="Weights for multi-dataset mode [egmd, star]")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Checkpoint filename in output repo (e.g. checkpoint_epoch10.pt)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Trackio run name (default: auto-generated)")
    parser.add_argument("--trackio-space", type=str, default=None,
                        help="HF Space ID for Trackio dashboard")
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
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = "cpu"
        print("WARNING: No GPU detected, training on CPU")
    print(f"Device: {device}")

    # --- Load pre-computed features ---
    train_rows: list[dict] = []
    val_rows: list[dict] = []

    if args.dataset in ("egmd", "multi"):
        train_rows.extend(load_features(args.egmd_repo, "train", token))
        val_rows.extend(load_features(args.egmd_repo, "validation", token))

    if args.dataset in ("star", "multi"):
        train_rows.extend(load_features(args.star_repo, "train", token))
        val_rows.extend(load_features(args.star_repo, "validation", token))

    print(f"Total training rows: {len(train_rows)}")
    print(f"Total validation rows: {len(val_rows)}")

    # --- Create output repo if needed ---
    try:
        api.repo_info(repo_id=args.output_repo, repo_type="model")
    except Exception:
        print(f"Creating output repo {args.output_repo}...")
        api.create_repo(repo_id=args.output_repo, repo_type="model", private=True)

    # --- Import training components ---
    from torch.utils.data import DataLoader

    from drumscribble.data.augment import SpecAugment
    from drumscribble.data.features import FeaturesDataset
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
    from drumscribble.config import FPS
    chunk_frames = int(args.chunk_seconds * FPS)
    augment = SpecAugment()
    collate_fn = AugmentCollate(augment)

    dataset = FeaturesDataset(train_rows, chunk_frames=chunk_frames)
    print(f"Training chunks: {len(dataset):,}")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn,
    )

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

    # --- Trackio ---
    import trackio

    trackio.init(
        project="drumscribble",
        name=args.run_name,
        space_id=args.trackio_space,
        config={
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "chunk_frames": chunk_frames,
            "egmd_repo": args.egmd_repo,
            "star_repo": args.star_repo,
            "output_repo": args.output_repo,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "train_chunks": len(dataset),
            "parameters": params,
        },
    )

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

        trackio.log({"loss": avg_loss, "lr": lr_now, "epoch": epoch + 1})

        # Save checkpoint and run validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            # --- Validation with EMA weights ---
            ema.apply(model)
            val_metrics = validate(model, val_rows, device, chunk_frames)
            ema.restore(model)

            print(f"  Val F1={val_metrics['val_f1']:.4f} | "
                  f"P={val_metrics['val_precision']:.4f} | "
                  f"R={val_metrics['val_recall']:.4f}")
            trackio.log(val_metrics)

            # --- Save checkpoint ---
            ckpt_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
            ckpt_data = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "val_f1": val_metrics["val_f1"],
            }
            if scaler is not None:
                ckpt_data["scaler"] = scaler.state_dict()
            torch.save(ckpt_data, ckpt_path)
            print(f"  Saved {ckpt_path}")

            api.upload_file(
                path_or_fileobj=str(ckpt_path),
                path_in_repo=f"checkpoint_epoch{epoch+1}.pt",
                repo_id=args.output_repo,
                repo_type="model",
                token=token,
            )
            print(f"  Uploaded checkpoint to {args.output_repo}")

    # --- Save final with EMA weights ---
    ema.apply(model)

    # Final validation
    final_metrics = validate(model, val_rows, device, chunk_frames)
    print(f"\nFinal Val F1={final_metrics['val_f1']:.4f} | "
          f"P={final_metrics['val_precision']:.4f} | "
          f"R={final_metrics['val_recall']:.4f}")
    trackio.log(final_metrics)

    final_path = output_dir / "final.pt"
    torch.save({
        "model": model.state_dict(),
        "epoch": args.epochs,
        "val_f1": final_metrics["val_f1"],
    }, final_path)
    ema.restore(model)

    api.upload_file(
        path_or_fileobj=str(final_path),
        path_in_repo="final.pt",
        repo_id=args.output_repo,
        repo_type="model",
        token=token,
    )
    print(f"\nTraining complete! Final checkpoint (EMA) uploaded to {args.output_repo}")

    trackio.finish()


if __name__ == "__main__":
    main()
