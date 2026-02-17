"""STAR Drums dataset loader."""
from pathlib import Path
import torch
from torch.utils.data import Dataset

from drumscribble.audio import load_and_preprocess, compute_mel_spectrogram
from drumscribble.targets import events_to_targets
from drumscribble.config import SAMPLE_RATE, FPS, STAR_ABBREV_TO_GM


def parse_star_annotation(path: str) -> list[tuple[float, int, int]]:
    """Parse STAR TSV annotation to (time, gm_note, velocity) events."""
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            time_s = float(parts[0])
            abbrev = parts[1]
            velocity = int(parts[2])
            gm_note = STAR_ABBREV_TO_GM.get(abbrev)
            if gm_note is not None:
                events.append((time_s, gm_note, velocity))
    events.sort(key=lambda x: x[0])
    return events


class STARDataset(Dataset):
    """STAR Drums dataset: FLAC audio + TSV annotations."""

    def __init__(
        self,
        root: str | Path,
        split: str = "training",
        chunk_seconds: float = 10.0,
    ):
        self.root = Path(root) / "data" / split
        self.chunk_seconds = chunk_seconds
        self.chunk_samples = int(chunk_seconds * SAMPLE_RATE)

        self.entries = []
        for source_dir in sorted(self.root.iterdir()):
            if not source_dir.is_dir():
                continue
            ann_dir = source_dir / "annotation"
            audio_dir = source_dir / "audio" / "mix"
            if not ann_dir.exists() or not audio_dir.exists():
                continue
            for ann_file in sorted(ann_dir.glob("*.txt")):
                audio_file = audio_dir / ann_file.name.replace(".txt", ".flac")
                if audio_file.exists():
                    self.entries.append({
                        "audio": str(audio_file),
                        "annotation": str(ann_file),
                    })

        self.chunks = []
        for i, entry in enumerate(self.entries):
            duration_samples = int(60.0 * SAMPLE_RATE)
            for start in range(0, duration_samples - self.chunk_samples + 1, self.chunk_samples):
                self.chunks.append((i, start))

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entry_idx, start_sample = self.chunks[idx]
        entry = self.entries[entry_idx]

        waveform = load_and_preprocess(entry["audio"])
        if start_sample + self.chunk_samples > waveform.shape[1]:
            pad = self.chunk_samples - (waveform.shape[1] - start_sample)
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        chunk = waveform[:, start_sample : start_sample + self.chunk_samples]

        mel = compute_mel_spectrogram(chunk)

        start_time = start_sample / SAMPLE_RATE
        end_time = start_time + self.chunk_seconds
        events = parse_star_annotation(entry["annotation"])
        chunk_events = [
            (t - start_time, note, vel)
            for t, note, vel in events
            if start_time <= t < end_time
        ]

        n_frames = mel.shape[-1]
        onset_target, vel_target = events_to_targets(chunk_events, n_frames)

        return mel, onset_target, vel_target
