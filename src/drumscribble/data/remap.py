"""Remap STAR 18-class targets to 26-class GM taxonomy."""
import numpy as np

from drumscribble.config import (
    GM_NOTE_TO_INDEX,
    NUM_CLASSES,
    STAR_ABBREV_TO_GM,
    STAR_CLASSES,
    TARGET_WIDENING,
)

# Precompute STAR index -> GM index mapping
_STAR_TO_GM_INDEX = [
    GM_NOTE_TO_INDEX[STAR_ABBREV_TO_GM[abbrev]]
    for abbrev in STAR_CLASSES
]


def remap_star_targets(
    onset_18: np.ndarray,
    vel_18: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remap (18, T) STAR targets to (26, T) GM targets with onset widening.

    Args:
        onset_18: Binary onset targets, shape (18, T).
        vel_18: Velocity targets normalized to [0, 1], shape (18, T).

    Returns:
        Tuple of (onset_26, vel_26), each shape (26, T), float32.
    """
    n_frames = onset_18.shape[1]
    onset_26 = np.zeros((NUM_CLASSES, n_frames), dtype=np.float32)
    vel_26 = np.zeros((NUM_CLASSES, n_frames), dtype=np.float32)

    half_w = len(TARGET_WIDENING) // 2

    for star_idx, gm_idx in enumerate(_STAR_TO_GM_INDEX):
        # Copy velocity directly (no widening)
        vel_26[gm_idx] = vel_18[star_idx]

        # Apply widening to onsets
        active_frames = np.nonzero(onset_18[star_idx])[0]
        for center in active_frames:
            for i, w in enumerate(TARGET_WIDENING):
                frame = center - half_w + i
                if 0 <= frame < n_frames:
                    onset_26[gm_idx, frame] = max(onset_26[gm_idx, frame], w)

    return onset_26, vel_26
