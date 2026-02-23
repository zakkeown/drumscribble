# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "drumscribble @ git+https://github.com/zakkeown/drumscribble.git@482207d",
#     "huggingface_hub[hf_xet]",
#     "pyarrow",
#     "pyyaml",
#     "trackio",
# ]
# ///
"""HF Jobs training script for DrumscribbleCNN (pre-computed features).

Downloads pre-computed mel/onset/velocity feature Parquet shards from
HF dataset repos, then reads them lazily via ParquetFeaturesDataset
(one shard cached in memory at a time). Much faster than the old
raw-audio pipeline (5-10 GB download instead of 90 GB+, no mel
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

# Force unbuffered stdout so HF Jobs logs stream in real-time
sys.stdout.reconfigure(line_buffering=True)

import torch
from huggingface_hub import HfApi


def download_shards(repo_id: str, split: str, token: str | None) -> list[str]:
    """Download feature Parquet shards and return local file paths.

    Downloads shards from features/{split}-*.parquet via hf_hub_download
    (cached locally). Does NOT load data into memory — that happens lazily
    in ParquetFeaturesDataset.

    Args:
        repo_id: HF dataset repo (e.g. 'schismaudio/e-gmd').
        split: Dataset split ('train' or 'validation').
        token: HF API token (needed for private repos).

    Returns:
        List of local file paths to downloaded Parquet shards.
    """
    from huggingface_hub import HfApi, hf_hub_download

    print(f"Downloading shards for {repo_id} split={split}...")
    api = HfApi(token=token)

    all_files = [
        f for f in api.list_repo_files(repo_id, repo_type="dataset")
        if f.startswith(f"features/{split}-") and f.endswith(".parquet")
    ]
    all_files.sort()
    print(f"  Found {len(all_files)} shards")

    paths = []
    for i, filename in enumerate(all_files):
        local_path = hf_hub_download(
            repo_id=repo_id, repo_type="dataset",
            filename=filename, token=token,
        )
        paths.append(local_path)
        if (i + 1) % 10 == 0:
            print(f"  Downloaded {i + 1}/{len(all_files)} shards")

    print(f"  Downloaded {len(paths)} shards from {repo_id}/{split}")
    return paths


_VALIDATE_SCRIPT = r"""
import json, sys, torch
import numpy as np
import pyarrow.parquet as pq
from drumscribble.model.drumscribble import DrumscribbleCNN
from drumscribble.config import FPS, GM_CLASSES, N_MELS, NUM_CLASSES
from drumscribble.evaluate import evaluate_events
from drumscribble.inference import detections_to_events

config_path, result_path = sys.argv[1], sys.argv[2]
with open(config_path) as f:
    cfg = json.load(f)
device = cfg["device"]
chunk_frames = cfg["chunk_frames"]
val_shards = cfg["val_shards"]
model_path = cfg["model_path"]

model = DrumscribbleCNN(
    backbone_dims=(64, 128, 256, 384),
    backbone_depths=(5, 5, 5, 5),
    num_attn_layers=3, num_attn_heads=4,
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

all_ref, all_est = [], []
with torch.no_grad():
    for shard_path in val_shards:
        table = pq.read_table(shard_path,
            columns=["mel_spectrogram", "onset_targets", "velocity_targets", "n_frames"])
        for row_idx in range(table.num_rows):
            n_frames = table.column("n_frames")[row_idx].as_py()
            if n_frames < chunk_frames:
                continue
            mel = np.frombuffer(table.column("mel_spectrogram")[row_idx].as_py(),
                dtype=np.float32).reshape(N_MELS, n_frames)
            onset = np.frombuffer(table.column("onset_targets")[row_idx].as_py(),
                dtype=np.float32).reshape(NUM_CLASSES, n_frames)
            for start in range(0, n_frames - chunk_frames + 1, chunk_frames):
                mel_t = torch.from_numpy(mel[:, start:start+chunk_frames].copy()
                    ).unsqueeze(0).unsqueeze(0).to(device)
                onset_t = torch.from_numpy(onset[:, start:start+chunk_frames].copy())
                onset_pred, vel_pred, _ = model(mel_t)
                for cls_idx in range(NUM_CLASSES):
                    frames = torch.where(onset_t[cls_idx] >= 1.0)[0]
                    for f_val in frames:
                        all_ref.append({"time": f_val.item() / FPS, "note": GM_CLASSES[cls_idx]})
                all_est.extend(detections_to_events(
                    onset_pred[0].sigmoid(), vel_pred[0].sigmoid(), threshold=0.5, fps=FPS))
        del table

metrics = evaluate_events(all_ref, all_est)
with open(result_path, "w") as f:
    json.dump({"val_f1": metrics["f1"], "val_precision": metrics["precision"],
               "val_recall": metrics["recall"]}, f)
"""


def validate(
    model: torch.nn.Module,
    val_shards: list[str],
    device: str,
    chunk_frames: int,
) -> dict[str, float]:
    """Run validation in a subprocess to prevent pyarrow memory leaks.

    Pyarrow's memory allocator does not return freed memory to the OS,
    so repeated in-process validation causes RSS to grow until OOM.
    Running in a subprocess guarantees all memory is reclaimed on exit.

    Args:
        model: DrumscribbleCNN (should already have EMA weights applied).
        val_shards: List of local Parquet shard paths for validation.
        device: Device string ('cuda' or 'cpu').
        chunk_frames: Number of frames per chunk.

    Returns:
        Dict with val_f1, val_precision, val_recall.
    """
    import json
    import subprocess
    import tempfile

    fallback = {"val_f1": 0.0, "val_precision": 0.0, "val_recall": 0.0}

    # Save model weights and config to temp files
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        model_path = f.name
        torch.save(model.state_dict(), f)

    config_path = model_path + ".config.json"
    result_path = model_path + ".result.json"
    script_path = model_path + ".validate.py"

    with open(config_path, "w") as f:
        json.dump({
            "model_path": model_path,
            "val_shards": val_shards,
            "device": device,
            "chunk_frames": chunk_frames,
        }, f)

    with open(script_path, "w") as f:
        f.write(_VALIDATE_SCRIPT)

    try:
        result = subprocess.run(
            [sys.executable, script_path, config_path, result_path],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            stderr_tail = result.stderr[-500:] if result.stderr else "no stderr"
            print(f"  Validation subprocess failed (rc={result.returncode}): {stderr_tail}")
            return fallback

        with open(result_path) as f:
            return json.load(f)
    except subprocess.TimeoutExpired:
        print("  Validation subprocess timed out (600s)")
        return fallback
    except Exception as e:
        print(f"  Validation error: {e}")
        return fallback
    finally:
        for p in [model_path, config_path, result_path, script_path]:
            Path(p).unlink(missing_ok=True)


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
    parser.add_argument("--num-workers", type=int, default=2)
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

    # --- Download feature shards (lazy — no data loaded into memory) ---
    train_shards: list[str] = []
    val_shards: list[str] = []

    if args.dataset in ("egmd", "multi"):
        train_shards.extend(download_shards(args.egmd_repo, "train", token))
        val_shards.extend(download_shards(args.egmd_repo, "validation", token))

    if args.dataset in ("star", "multi"):
        train_shards.extend(download_shards(args.star_repo, "train", token))
        val_shards.extend(download_shards(args.star_repo, "validation", token))

    print(f"Train shards: {len(train_shards)}")
    print(f"Val shards: {len(val_shards)}")

    # --- Create output repo if needed ---
    try:
        api.repo_info(repo_id=args.output_repo, repo_type="model")
    except Exception:
        print(f"Creating output repo {args.output_repo}...")
        api.create_repo(repo_id=args.output_repo, repo_type="model", private=True)

    # --- Import training components ---
    from torch.utils.data import DataLoader

    from drumscribble.data.augment import SpecAugment
    from drumscribble.data.features import ParquetFeaturesDataset, ShardGroupedSampler
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

    dataset = ParquetFeaturesDataset(train_shards, chunk_frames=chunk_frames)
    print(f"Training chunks: {len(dataset):,}")

    sampler = ShardGroupedSampler(dataset)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn,
        worker_init_fn=ParquetFeaturesDataset.worker_init_fn,
        persistent_workers=False,
    )

    # --- Optimizer ---
    optimizer = create_optimizer(model, lr=args.lr, weight_decay=0.05)
    loss_fn = DrumscribbleLoss()

    warmup_epochs = 5
    warmup_steps = warmup_epochs * len(loader)
    total_steps = args.epochs * len(loader)
    scheduler = create_scheduler(optimizer, warmup_steps, total_steps)

    ema = EMAModel(model, decay=0.999)

    # --- AMP (bfloat16 — same throughput as float16 on A10G, no overflow risk) ---
    amp_dtype = torch.bfloat16 if device == "cuda" else None
    if amp_dtype:
        print(f"Using CUDA AMP with {amp_dtype}")

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
            "train_shards": len(train_shards),
            "val_shards": len(val_shards),
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
            device=device, scheduler=scheduler, ema=ema,
            amp_dtype=amp_dtype,
        )
        lr_now = optimizer.param_groups[0]["lr"]
        import resource
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.4f} | lr={lr_now:.2e} | rss={rss_mb:.0f}MB")

        trackio.log({"loss": avg_loss, "lr": lr_now, "epoch": epoch + 1})

        # Save checkpoint and run validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            # --- Validation with EMA weights (subprocess to prevent memory leaks) ---
            import gc; gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            ema.apply(model)
            val_metrics = validate(model, val_shards[:3], device, chunk_frames)
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

            # Free checkpoint memory and delete local file
            del ckpt_data
            ckpt_path.unlink(missing_ok=True)
            import gc; gc.collect()
            torch.cuda.empty_cache() if device == "cuda" else None

    # --- Save final with EMA weights ---
    ema.apply(model)

    # Final validation (subset to limit memory)
    import gc; gc.collect()
    final_metrics = validate(model, val_shards[:3], device, chunk_frames)
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
