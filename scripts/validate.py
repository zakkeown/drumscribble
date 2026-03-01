"""Run validation on a checkpoint using WebDataset feature shards."""
import argparse

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from drumscribble.config import FPS, INDEX_TO_GM_NOTE, NUM_CLASSES
from drumscribble.data.webdataset_loader import create_webdataset_pipeline
from drumscribble.inference import detections_to_events
from drumscribble.evaluate import evaluate_events
from drumscribble.model.drumscribble import DrumscribbleCNN


def events_from_targets(onset_target, vel_target, threshold=0.5):
    """Convert ground-truth frame targets to event dicts for mir_eval."""
    events = []
    for cls_idx in range(onset_target.shape[0]):
        for t in range(onset_target.shape[1]):
            if onset_target[cls_idx, t] >= threshold:
                events.append({
                    "time": t / FPS,
                    "note": INDEX_TO_GM_NOTE.get(cls_idx, 0),
                    "velocity": int(vel_target[cls_idx, t].item() * 127),
                })
    return events


def main():
    parser = argparse.ArgumentParser(description="Validate DrumscribbleCNN")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--config", type=str, default="configs/train/default.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Onset detection threshold")
    parser.add_argument("--shard-root", type=str, required=True,
                        help="Root directory for feature shards")
    parser.add_argument("--datasets", type=str, nargs="+", required=True,
                        help="Dataset names within shard root")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limit number of batches for quick validation")
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

    # Load model
    model = DrumscribbleCNN(
        backbone_dims=tuple(model_cfg["backbone_dims"]),
        backbone_depths=tuple(model_cfg["backbone_depths"]),
        num_attn_layers=model_cfg["num_attn_layers"],
        num_attn_heads=model_cfg["num_attn_heads"],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {epoch})")

    # Load validation dataset via WebDataset
    pipeline = create_webdataset_pipeline(
        shard_root=args.shard_root,
        datasets=args.datasets,
        split="validation",
        shuffle=False,
    )

    loader = DataLoader(
        pipeline,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # Run validation
    all_ref_events = []
    all_est_events = []
    n_batches = 0

    with torch.no_grad():
        for batch_idx, (mel, onset_target, vel_target) in enumerate(tqdm(loader, desc="Validating")):
            if args.max_batches and batch_idx >= args.max_batches:
                break

            mel = mel.to(device)
            onset_pred, vel_pred, offset_pred = model(mel)

            # Move to CPU for event extraction
            onset_pred_cpu = onset_pred.cpu()
            vel_pred_cpu = vel_pred.cpu()
            onset_target_cpu = onset_target.cpu()
            vel_target_cpu = vel_target.cpu()

            # Per-sample event extraction and matching
            for i in range(mel.shape[0]):
                ref = events_from_targets(onset_target_cpu[i], vel_target_cpu[i])
                est = detections_to_events(
                    onset_pred_cpu[i], vel_pred_cpu[i],
                    threshold=args.threshold,
                )
                all_ref_events.extend(ref)
                all_est_events.extend(est)

            n_batches += 1

    # Compute overall metrics
    metrics = evaluate_events(all_ref_events, all_est_events)

    print(f"\n{'='*50}")
    print(f"Validation Results (epoch {epoch}, threshold={args.threshold})")
    print(f"{'='*50}")
    print(f"  Batches evaluated:  {n_batches:,}")
    print(f"  Reference events:   {len(all_ref_events):,}")
    print(f"  Estimated events:   {len(all_est_events):,}")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  Recall:             {metrics['recall']:.4f}")
    print(f"  F1:                 {metrics['f1']:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
