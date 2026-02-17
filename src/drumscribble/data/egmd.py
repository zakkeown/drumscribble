"""E-GMD dataset loader."""
import csv
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pretty_midi

from drumscribble.audio import load_and_preprocess, compute_mel_spectrogram
from drumscribble.targets import events_to_targets
from drumscribble.config import SAMPLE_RATE, FPS


def parse_midi_to_events(midi_path: str) -> list[tuple[float, int, int]]:
    """Parse MIDI file to list of (time_seconds, gm_note, velocity)."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    events = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            for note in instrument.notes:
                events.append((note.start, note.pitch, note.velocity))
    events.sort(key=lambda x: x[0])
    return events


class EGMDDataset(Dataset):
    """E-GMD dataset: WAV audio + MIDI annotations."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        chunk_seconds: float = 10.0,
    ):
        self.root = Path(root)
        self.chunk_seconds = chunk_seconds
        self.chunk_samples = int(chunk_seconds * SAMPLE_RATE)
        self.chunk_frames = int(chunk_seconds * FPS)

        csv_path = self.root / "e-gmd-v1.0.0.csv"
        self.entries = []
        skipped = 0
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    audio_path = self.root / row["audio_filename"]
                    midi_path = self.root / row["midi_filename"]
                    if not audio_path.exists() or not midi_path.exists():
                        skipped += 1
                        continue
                    self.entries.append({
                        "audio": str(audio_path),
                        "midi": str(midi_path),
                        "duration": float(row["duration"]),
                    })
        if skipped:
            print(f"E-GMD: skipped {skipped} entries with missing files")

        self.chunks = []
        for i, entry in enumerate(self.entries):
            total_samples = int(entry["duration"] * SAMPLE_RATE)
            if total_samples <= self.chunk_samples:
                self.chunks.append((i, 0))
            else:
                for start in range(0, total_samples - self.chunk_samples + 1, self.chunk_samples):
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

        mel = compute_mel_spectrogram(chunk)  # (1, 128, T)

        start_time = start_sample / SAMPLE_RATE
        end_time = start_time + self.chunk_seconds
        events = parse_midi_to_events(entry["midi"])
        chunk_events = [
            (t - start_time, note, vel)
            for t, note, vel in events
            if start_time <= t < end_time
        ]

        n_frames = mel.shape[-1]
        onset_target, vel_target = events_to_targets(chunk_events, n_frames)

        return mel, onset_target, vel_target
