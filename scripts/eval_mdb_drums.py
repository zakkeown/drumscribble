"""Evaluate DrumscribbleCNN on MDB-Drums (unseen dataset).

Downloads the final.pt checkpoint from HF Hub and runs inference on all 23
MDB-Drums tracks, comparing against ground truth annotations.

Usage:
    python scripts/eval_mdb_drums.py \
        --mdb-path ~/Documents/Datasets/mdb-drums \
        --checkpoint schismaudio/drumscribble-checkpoints/final.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

from drumscribble.audio import load_and_preprocess, compute_mel_spectrogram
from drumscribble.config import (
    FPS, GM_NOTE_TO_INDEX, INDEX_TO_GM_NOTE, NUM_CLASSES, EVAL_MAPPINGS,
)
from drumscribble.evaluate import evaluate_events
from drumscribble.inference import detections_to_events
from drumscribble.model.drumscribble import DrumscribbleCNN


# MDB-Drums subclass label -> GM MIDI note
MDB_SUBCLASS_TO_GM = {
    "KD":   36,  # Kick Drum
    "SD":   38,  # Snare
    "SDG":  38,  # Snare ghost
    "SDB":  38,  # Snare buzz
    "SDF":  38,  # Snare flam
    "SDD":  38,  # Snare drag
    "SDNS": 38,  # Snare (no snares)
    "SST":  37,  # Side Stick
    "CHH":  42,  # Closed Hi-Hat
    "OHH":  46,  # Open Hi-Hat
    "PHH":  44,  # Pedal Hi-Hat
    "CRC":  49,  # Crash Cymbal
    "RDC":  51,  # Ride Cymbal
    "RDB":  53,  # Ride Bell
    "SPC":  55,  # Splash Cymbal
    "CHC":  52,  # Chinese Cymbal
    "LFT":  41,  # Low Floor Tom
    "MHT":  47,  # Mid Tom
    "HFT":  50,  # High Tom
    "TMB":  54,  # Tambourine
    "HIT":  None,  # skip unidentifiable hits
}


def parse_mdb_annotations(ann_path: Path) -> list[dict]:
    """Parse MDB-Drums subclass annotation file into event list."""
    events = []
    for line in ann_path.read_text().strip().splitlines():
        parts = line.split("\t")
        time_sec = float(parts[0].strip())
        label = parts[1].strip()
        gm_note = MDB_SUBCLASS_TO_GM.get(label)
        if gm_note is None:
            continue
        if gm_note not in GM_NOTE_TO_INDEX:
            continue
        events.append({"time": time_sec, "note": gm_note})
    return events


def remap_events_for_eval(events: list[dict], mapping: dict[int, int]) -> list[dict]:
    """Remap GM notes to reduced evaluation classes."""
    remapped = []
    for e in events:
        note = e["note"]
        if note in GM_NOTE_TO_INDEX:
            cls_idx = GM_NOTE_TO_INDEX[note]
            if cls_idx in mapping:
                remapped.append({"time": e["time"], "note": mapping[cls_idx]})
    return remapped


def run_inference(
    model: DrumscribbleCNN,
    audio_path: Path,
    device: str,
    chunk_frames: int = 625,
    threshold: float = 0.5,
) -> list[dict]:
    """Run chunked inference on an audio file."""
    waveform = load_and_preprocess(str(audio_path))
    mel = compute_mel_spectrogram(waveform)  # (1, N_MELS, T)
    total_frames = mel.shape[-1]

    all_events = []
    with torch.no_grad():
        for start in range(0, total_frames - chunk_frames + 1, chunk_frames):
            chunk = mel[:, :, start:start + chunk_frames].unsqueeze(0).to(device)
            onset_logits, vel_logits, _ = model(chunk)
            onset_probs = onset_logits[0].sigmoid()
            vel_probs = vel_logits[0].sigmoid()
            chunk_events = detections_to_events(
                onset_probs, vel_probs, threshold=threshold, fps=FPS,
            )
            # Offset times by chunk position
            time_offset = start / FPS
            for e in chunk_events:
                e["time"] += time_offset
            all_events.extend(chunk_events)

    all_events.sort(key=lambda e: e["time"])
    return all_events


def main():
    parser = argparse.ArgumentParser(description="Evaluate on MDB-Drums")
    parser.add_argument("--mdb-path", type=str, required=True,
                        help="Path to MDB-Drums dataset root")
    parser.add_argument("--checkpoint", type=str,
                        default="schismaudio/drumscribble-checkpoints/final.pt",
                        help="HF Hub path or local path to checkpoint")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    mdb = Path(args.mdb_path)
    audio_dir = mdb / "audio" / "drum_only"
    ann_dir = mdb / "annotations" / "subclass"

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    if "/" in args.checkpoint and not Path(args.checkpoint).exists():
        from huggingface_hub import hf_hub_download
        parts = args.checkpoint.split("/")
        repo_id = f"{parts[0]}/{parts[1]}"
        filename = "/".join(parts[2:])
        ckpt_path = hf_hub_download(repo_id, filename)
    else:
        ckpt_path = args.checkpoint

    model = DrumscribbleCNN(
        backbone_dims=(64, 128, 256, 384),
        backbone_depths=(5, 5, 5, 5),
        num_attn_layers=3,
        num_attn_heads=4,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "ema_state_dict" in state:
        model.load_state_dict(state["ema_state_dict"])
        print("Loaded EMA weights")
    elif "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        print("Loaded model weights")
    elif "model" in state:
        model.load_state_dict(state["model"])
        epoch = state.get("epoch", "?")
        val_f1 = state.get("val_f1", "?")
        print(f"Loaded checkpoint (epoch={epoch}, val_f1={val_f1})")
    else:
        model.load_state_dict(state)
        print("Loaded raw state dict")
    model.eval()
    print(f"Model on {device}\n")

    # Collect audio/annotation pairs
    tracks = []
    for wav_path in sorted(audio_dir.glob("*.wav")):
        name = wav_path.stem.replace("_Drum", "")
        ann_path = ann_dir / f"{name}_subclass.txt"
        if ann_path.exists():
            tracks.append((wav_path, ann_path, name))

    print(f"Found {len(tracks)} tracks\n")
    print(f"{'Track':<35} {'F1':>6} {'Prec':>6} {'Rec':>6}  "
          f"{'F1_5':>6} {'P_5':>6} {'R_5':>6}  {'Ref':>5} {'Est':>5}")
    print("-" * 110)

    mdb5_map = EVAL_MAPPINGS["mdb_5"]
    all_ref_26, all_est_26 = [], []
    all_ref_5, all_est_5 = [], []

    for wav_path, ann_path, name in tracks:
        ref_events = parse_mdb_annotations(ann_path)
        est_events = run_inference(model, wav_path, device, threshold=args.threshold)

        # 26-class evaluation
        metrics_26 = evaluate_events(ref_events, est_events)

        # 5-class reduced evaluation
        ref_5 = remap_events_for_eval(ref_events, mdb5_map)
        est_5 = remap_events_for_eval(est_events, mdb5_map)
        metrics_5 = evaluate_events(ref_5, est_5)

        all_ref_26.extend(ref_events)
        all_est_26.extend(est_events)
        all_ref_5.extend(ref_5)
        all_est_5.extend(est_5)

        print(f"{name:<35} {metrics_26['f1']:>6.3f} {metrics_26['precision']:>6.3f} "
              f"{metrics_26['recall']:>6.3f}  {metrics_5['f1']:>6.3f} {metrics_5['precision']:>6.3f} "
              f"{metrics_5['recall']:>6.3f}  {len(ref_events):>5} {len(est_events):>5}")

    print("-" * 110)

    # Overall micro-averaged metrics
    overall_26 = evaluate_events(all_ref_26, all_est_26)
    overall_5 = evaluate_events(all_ref_5, all_est_5)

    print(f"{'OVERALL':<35} {overall_26['f1']:>6.3f} {overall_26['precision']:>6.3f} "
          f"{overall_26['recall']:>6.3f}  {overall_5['f1']:>6.3f} {overall_5['precision']:>6.3f} "
          f"{overall_5['recall']:>6.3f}  {len(all_ref_26):>5} {len(all_est_26):>5}")

    print(f"\n26-class: F1={overall_26['f1']:.4f}  P={overall_26['precision']:.4f}  R={overall_26['recall']:.4f}")
    print(f" 5-class: F1={overall_5['f1']:.4f}  P={overall_5['precision']:.4f}  R={overall_5['recall']:.4f}")


if __name__ == "__main__":
    main()
