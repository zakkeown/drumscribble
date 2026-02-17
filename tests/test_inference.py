import torch
import numpy as np
from drumscribble.inference import peak_pick, nms
from drumscribble.config import FPS


def test_peak_pick_single():
    probs = torch.zeros(100)
    probs[50] = 0.9
    peaks = peak_pick(probs, threshold=0.5)
    assert len(peaks) == 1
    assert peaks[0] == 50


def test_peak_pick_threshold():
    probs = torch.zeros(100)
    probs[50] = 0.4
    peaks = peak_pick(probs, threshold=0.5)
    assert len(peaks) == 0


def test_nms_removes_nearby():
    peaks = [10, 12, 50, 51, 52]
    scores = [0.8, 0.9, 0.7, 0.95, 0.6]
    result = nms(peaks, scores, min_distance=3)
    # Should keep 12 (higher than 10) and 51 (highest in cluster)
    assert 12 in [r[0] for r in result]
    assert 51 in [r[0] for r in result]
    assert len(result) == 2


def test_detections_to_events():
    from drumscribble.inference import detections_to_events
    from drumscribble.config import NUM_CLASSES

    onset_probs = torch.zeros(NUM_CLASSES, 200)
    vel_probs = torch.zeros(NUM_CLASSES, 200)

    # Place a kick onset
    onset_probs[1, 100] = 0.9  # class index 1 = GM 36
    vel_probs[1, 100] = 0.75

    events = detections_to_events(onset_probs, vel_probs, threshold=0.5)
    assert len(events) >= 1
    assert events[0]["time"] == 100 / FPS
    assert events[0]["velocity"] > 0
