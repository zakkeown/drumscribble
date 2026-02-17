import numpy as np
from drumscribble.evaluate import evaluate_events, evaluate_onset_f1


def test_evaluate_perfect():
    ref_events = [
        {"time": 0.5, "note": 36},
        {"time": 1.0, "note": 38},
    ]
    est_events = [
        {"time": 0.5, "note": 36},
        {"time": 1.0, "note": 38},
    ]
    metrics = evaluate_events(ref_events, est_events)
    assert metrics["f1"] == 1.0


def test_evaluate_missed_onset():
    ref_events = [
        {"time": 0.5, "note": 36},
        {"time": 1.0, "note": 38},
    ]
    est_events = [
        {"time": 0.5, "note": 36},
    ]
    metrics = evaluate_events(ref_events, est_events)
    assert metrics["recall"] < 1.0
    assert metrics["precision"] == 1.0


def test_evaluate_onset_f1_overall():
    ref = [{"time": 0.5, "note": 36}, {"time": 1.0, "note": 36}]
    est = [{"time": 0.5, "note": 36}, {"time": 1.02, "note": 36}]
    f1 = evaluate_onset_f1(ref, est, onset_tolerance=0.05)
    assert f1 == 1.0  # Both within 50ms
