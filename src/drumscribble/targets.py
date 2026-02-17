"""Frame-level target generation with target widening."""
import torch
from drumscribble.config import (
    NUM_CLASSES, FPS, TARGET_WIDENING, GM_NOTE_TO_INDEX, EGMD_NOTE_REMAP,
)


def onsets_to_target_frames(
    events: list[tuple[float, int, int]],
    n_frames: int,
    fps: float = FPS,
    widening: list[float] = TARGET_WIDENING,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert onset events to frame-level targets with widening.

    Args:
        events: List of (time_seconds, class_index, midi_velocity).
        n_frames: Number of output frames.
        fps: Frames per second.
        widening: Target widening values centered on onset frame.

    Returns:
        onset_target: (NUM_CLASSES, n_frames) float tensor [0, 1].
        vel_target: (NUM_CLASSES, n_frames) float tensor [0, 1].
    """
    onset_target = torch.zeros(NUM_CLASSES, n_frames)
    vel_target = torch.zeros(NUM_CLASSES, n_frames)

    half_w = len(widening) // 2

    for time_s, cls_idx, velocity in events:
        center = round(time_s * fps)
        vel_norm = velocity / 127.0

        for i, w in enumerate(widening):
            frame = center - half_w + i
            if 0 <= frame < n_frames:
                onset_target[cls_idx, frame] = max(
                    onset_target[cls_idx, frame].item(), w
                )
                if w == 1.0:
                    vel_target[cls_idx, frame] = vel_norm

    return onset_target, vel_target


def events_to_targets(
    midi_events: list[tuple[float, int, int]],
    n_frames: int,
    fps: float = FPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert (time, gm_note, velocity) events to frame-level targets.

    Handles GM note remapping (e.g. E-GMD Roland TD-17 non-standard notes).
    Unknown notes are silently dropped.
    """
    converted = []
    for time_s, note, velocity in midi_events:
        note = EGMD_NOTE_REMAP.get(note, note)
        if note in GM_NOTE_TO_INDEX:
            converted.append((time_s, GM_NOTE_TO_INDEX[note], velocity))
    return onsets_to_target_frames(converted, n_frames, fps)
