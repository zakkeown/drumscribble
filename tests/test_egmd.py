import csv
import torch
import pytest
import pretty_midi
import torchaudio
from pathlib import Path
from drumscribble.data.egmd import EGMDDataset, parse_midi_to_events
from drumscribble.config import SAMPLE_RATE, NUM_CLASSES, GM_NOTE_TO_INDEX


@pytest.fixture
def fake_egmd(tmp_path):
    """Create a minimal fake E-GMD dataset for testing."""
    pm = pretty_midi.PrettyMIDI()
    drum = pretty_midi.Instrument(program=0, is_drum=True)
    drum.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.5, end=0.6))
    drum.notes.append(pretty_midi.Note(velocity=80, pitch=38, start=1.0, end=1.1))
    pm.instruments.append(drum)

    midi_path = tmp_path / "drummer1" / "session1"
    midi_path.mkdir(parents=True)
    pm.write(str(midi_path / "test.midi"))

    waveform = torch.randn(1, SAMPLE_RATE * 2)
    torchaudio.save(str(midi_path / "test.wav"), waveform, SAMPLE_RATE)

    csv_path = tmp_path / "e-gmd-v1.0.0.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "drummer", "session", "id", "style", "bpm", "beat_type",
            "time_signature", "duration", "split", "midi_filename", "audio_filename",
        ])
        writer.writerow([
            "drummer1", "drummer1/session1", "drummer1/session1/test",
            "rock/basic", "120", "beat", "4-4", "2.0", "train",
            "drummer1/session1/test.midi", "drummer1/session1/test.wav",
        ])

    return tmp_path


def test_parse_midi_to_events(fake_egmd):
    events = parse_midi_to_events(str(fake_egmd / "drummer1/session1/test.midi"))
    assert len(events) == 2
    assert events[0] == pytest.approx((0.5, 36, 100), abs=0.01)
    assert events[1] == pytest.approx((1.0, 38, 80), abs=0.01)


def test_dataset_len(fake_egmd):
    ds = EGMDDataset(fake_egmd, split="train", chunk_seconds=2.0)
    assert len(ds) >= 1


def test_dataset_getitem(fake_egmd):
    ds = EGMDDataset(fake_egmd, split="train", chunk_seconds=2.0)
    mel, onset_target, vel_target = ds[0]
    assert mel.dim() == 3  # (1, 128, T)
    assert onset_target.shape[0] == NUM_CLASSES
    assert vel_target.shape[0] == NUM_CLASSES
    assert mel.shape[-1] == onset_target.shape[-1]
