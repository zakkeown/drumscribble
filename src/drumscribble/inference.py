"""Inference pipeline: peak picking, NMS, event extraction."""
import torch
import numpy as np
from drumscribble.config import FPS, INDEX_TO_GM_NOTE, NUM_CLASSES


def peak_pick(
    probs: torch.Tensor,
    threshold: float = 0.5,
    pre_max: int = 2,
    post_max: int = 6,
) -> list[int]:
    """Find peaks in a 1D probability sequence.

    Args:
        probs: (T,) tensor of probabilities.
        threshold: Minimum probability for a peak.
        pre_max: Frames before peak to check for local maximum.
        post_max: Frames after peak to check for local maximum.

    Returns:
        List of peak frame indices.

    Note:
        Peaks in the first ``pre_max`` or last ``post_max`` frames are
        excluded. With defaults (pre_max=2, post_max=6) at 62.5 fps,
        this means the first 32 ms and last 96 ms are not searched.
    """
    probs_np = probs.cpu().numpy()
    peaks = []
    for i in range(pre_max, len(probs_np) - post_max):
        if probs_np[i] < threshold:
            continue
        window = probs_np[max(0, i - pre_max) : i + post_max + 1]
        if probs_np[i] == window.max():
            peaks.append(i)
    return peaks


def nms(
    peaks: list[int],
    scores: list[float],
    min_distance: int = 2,
) -> list[tuple[int, float]]:
    """Non-maximum suppression on peaks.

    Returns:
        List of (frame, score) tuples after suppression.
    """
    if not peaks:
        return []

    # Sort by score descending
    order = sorted(range(len(peaks)), key=lambda i: scores[i], reverse=True)
    kept = []
    suppressed = set()

    for i in order:
        if i in suppressed:
            continue
        kept.append((peaks[i], scores[i]))
        for j in order:
            if j != i and abs(peaks[j] - peaks[i]) < min_distance:
                suppressed.add(j)

    return sorted(kept, key=lambda x: x[0])


def detections_to_events(
    onset_probs: torch.Tensor,
    vel_probs: torch.Tensor,
    threshold: float = 0.5,
    nms_frames: int = 2,
    fps: float = FPS,
) -> list[dict]:
    """Convert frame-level predictions to onset events.

    Args:
        onset_probs: (NUM_CLASSES, T) onset probabilities.
        vel_probs: (NUM_CLASSES, T) velocity estimates.
        threshold: Minimum onset probability for detection.
        nms_frames: Minimum distance between detections for NMS.
        fps: Frames per second for time conversion.

    Returns:
        List of dicts with keys: time, note, velocity, class_idx, confidence.
    """
    events = []
    for cls_idx in range(onset_probs.shape[0]):
        peaks = peak_pick(onset_probs[cls_idx], threshold=threshold)
        if not peaks:
            continue
        scores = [onset_probs[cls_idx, p].item() for p in peaks]
        kept = nms(peaks, scores, min_distance=nms_frames)

        for frame, confidence in kept:
            velocity = vel_probs[cls_idx, frame].item()
            events.append({
                "time": frame / fps,
                "note": INDEX_TO_GM_NOTE.get(cls_idx, 0),
                "velocity": int(velocity * 127),
                "class_idx": cls_idx,
                "confidence": confidence,
            })

    events.sort(key=lambda e: e["time"])
    return events
