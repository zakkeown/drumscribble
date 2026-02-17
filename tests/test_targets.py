import torch
from drumscribble.targets import onsets_to_target_frames, events_to_targets
from drumscribble.config import NUM_CLASSES, FPS, TARGET_WIDENING


def test_single_onset_widened():
    # One onset at t=1.0s, class index 0, velocity 100
    events = [(1.0, 0, 100)]
    n_frames = 200
    onset_target, vel_target = onsets_to_target_frames(events, n_frames)

    assert onset_target.shape == (NUM_CLASSES, n_frames)
    assert vel_target.shape == (NUM_CLASSES, n_frames)

    center_frame = round(1.0 * FPS)
    # Check widening pattern
    assert onset_target[0, center_frame].item() == 1.0
    assert abs(onset_target[0, center_frame - 1].item() - 0.6) < 1e-5
    assert abs(onset_target[0, center_frame + 1].item() - 0.6) < 1e-5
    assert abs(onset_target[0, center_frame - 2].item() - 0.3) < 1e-5
    assert abs(onset_target[0, center_frame + 2].item() - 0.3) < 1e-5
    # Velocity at center
    assert abs(vel_target[0, center_frame].item() - 100 / 127) < 1e-5


def test_no_events_gives_zeros():
    onset_target, vel_target = onsets_to_target_frames([], 100)
    assert onset_target.sum() == 0
    assert vel_target.sum() == 0


def test_overlapping_onsets_take_max():
    # Two onsets close together on same class
    events = [(1.0, 0, 80), (1.032, 0, 120)]  # 32ms apart = 2 frames
    n_frames = 200
    onset_target, _ = onsets_to_target_frames(events, n_frames)
    # Overlapping widened regions should take max value
    center1 = round(1.0 * FPS)
    center2 = round(1.032 * FPS)
    assert onset_target[0, center1].item() == 1.0
    assert onset_target[0, center2].item() == 1.0


def test_events_to_targets_from_midi_notes():
    # Simulates parsed MIDI: list of (time, gm_note, velocity)
    midi_events = [(0.5, 36, 110), (0.5, 42, 90)]  # kick + closed HH simultaneous
    onset, vel = events_to_targets(midi_events, n_frames=100)
    assert onset.shape == (NUM_CLASSES, 100)
    # Check both classes have activations
    from drumscribble.config import GM_NOTE_TO_INDEX
    kick_idx = GM_NOTE_TO_INDEX[36]
    hh_idx = GM_NOTE_TO_INDEX[42]
    assert onset[kick_idx].max() > 0
    assert onset[hh_idx].max() > 0
