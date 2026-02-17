import torch
import pytest
import torchaudio
from pathlib import Path
from drumscribble.data.star import STARDataset, parse_star_annotation
from drumscribble.config import SAMPLE_RATE, NUM_CLASSES


@pytest.fixture
def fake_star(tmp_path):
    """Create a minimal fake STAR dataset."""
    ann_dir = tmp_path / "data" / "training" / "test_source" / "annotation"
    ann_dir.mkdir(parents=True)
    ann_file = ann_dir / "001_mix_kit1.txt"
    ann_file.write_text("0.5\tBD\t100\n1.0\tSD\t80\n1.0\tCHH\t70\n")

    audio_dir = tmp_path / "data" / "training" / "test_source" / "audio" / "mix"
    audio_dir.mkdir(parents=True)
    waveform = torch.randn(1, SAMPLE_RATE * 2)
    torchaudio.save(str(audio_dir / "001_mix_kit1.flac"), waveform, SAMPLE_RATE)

    return tmp_path


def test_parse_star_annotation(fake_star):
    ann_path = fake_star / "data/training/test_source/annotation/001_mix_kit1.txt"
    events = parse_star_annotation(str(ann_path))
    assert len(events) == 3
    assert events[0] == (0.5, 36, 100)  # BD -> GM 36
    assert events[1] == (1.0, 38, 80)   # SD -> GM 38
    assert events[2] == (1.0, 42, 70)   # CHH -> GM 42


def test_star_dataset_len(fake_star):
    ds = STARDataset(fake_star, split="training", chunk_seconds=2.0)
    assert len(ds) >= 1


def test_star_dataset_getitem(fake_star):
    ds = STARDataset(fake_star, split="training", chunk_seconds=2.0)
    mel, onset_target, vel_target = ds[0]
    assert mel.dim() == 3
    assert onset_target.shape[0] == NUM_CLASSES
