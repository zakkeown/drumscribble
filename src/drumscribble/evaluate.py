"""Evaluation using mir_eval for onset F1."""

import numpy as np
import mir_eval


def evaluate_events(
    ref_events: list[dict],
    est_events: list[dict],
    onset_tolerance: float = 0.05,
) -> dict[str, float]:
    """Evaluate onset precision/recall/F1 using mir_eval.

    Events are grouped by note class, and per-class onset matching is
    performed using mir_eval's bipartite matching. Micro-averaged
    precision/recall/F1 is computed across all note classes.

    Args:
        ref_events: List of dicts with 'time' and 'note'.
        est_events: List of dicts with 'time' and 'note'.
        onset_tolerance: Tolerance in seconds (default 50ms).

    Returns:
        Dict with precision, recall, f1.
    """
    if not ref_events and not est_events:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    # Group by note class
    ref_by_note: dict[int, list[float]] = {}
    est_by_note: dict[int, list[float]] = {}

    for e in ref_events:
        ref_by_note.setdefault(e["note"], []).append(e["time"])
    for e in est_events:
        est_by_note.setdefault(e["note"], []).append(e["time"])

    all_notes = set(ref_by_note) | set(est_by_note)
    total_tp, total_fp, total_fn = 0, 0, 0

    for note in all_notes:
        ref_times = np.array(sorted(ref_by_note.get(note, [])))
        est_times = np.array(sorted(est_by_note.get(note, [])))

        if len(ref_times) == 0:
            total_fp += len(est_times)
            continue
        if len(est_times) == 0:
            total_fn += len(ref_times)
            continue

        # Build intervals (onset, offset) required by mir_eval.
        # Offset doesn't matter for onset-only matching; use a dummy offset.
        ref_intervals = np.column_stack([ref_times, ref_times + 0.1])
        est_intervals = np.column_stack([est_times, est_times + 0.1])

        # Use mir_eval's bipartite onset matching to get matched pairs
        matching = mir_eval.transcription.match_note_onsets(
            ref_intervals, est_intervals, onset_tolerance=onset_tolerance
        )

        tp = len(matching)
        total_tp += tp
        total_fp += len(est_times) - tp
        total_fn += len(ref_times) - tp

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_onset_f1(
    ref_events: list[dict],
    est_events: list[dict],
    onset_tolerance: float = 0.05,
) -> float:
    """Convenience: return just the F1 score."""
    return evaluate_events(ref_events, est_events, onset_tolerance)["f1"]
