---
license: cc-by-4.0
task_categories:
  - audio-classification
tags:
  - drums
  - percussion
  - music-information-retrieval
  - drum-transcription
  - midi
  - groove
  - rhythm
  - electronic-drums
pretty_name: "Groove MIDI Dataset"
size_categories:
  - 1K<n<10K
---

# Groove MIDI Dataset

## Quick Start

```python
from datasets import load_dataset

# Stream to avoid downloading the entire dataset
ds = load_dataset("schismaudio/groove-midi-dataset", streaming=True)

# Or download locally
ds = load_dataset("schismaudio/groove-midi-dataset")
```

## Dataset Description

The **Groove MIDI Dataset (GMD)** is a collection of 13.6 hours of aligned MIDI and synthesized audio recordings of drum performances by 10 drummers (80%+ professional). Created by Google Magenta, it contains 1,150 MIDI files covering over 22,000 measures across 18 musical styles.

Each performance was recorded on a Roland TD-11 electronic drum kit, capturing both expressive MIDI (with velocity and precise timing) and synthesized audio output. The dataset is widely used for research in drum transcription, groove modeling, humanization, and rhythmic generation.

GMD is the foundation for several derivative datasets, including [E-GMD](https://huggingface.co/datasets/schismaudio/e-gmd) (re-rendered with 43 kits) and [StemGMD](https://huggingface.co/datasets/schismaudio/stemgmd) (multi-kit isolated stems).

## Dataset Structure

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `midi_filename` | `string` | Relative path to the MIDI file |
| `audio_filename` | `string` | Relative path to the WAV audio file (22050 Hz, mono) |
| `drummer` | `string` | Anonymized drummer ID (drummer1–drummer10) |
| `session` | `string` | Recording session (e.g., session1, eval_session) |
| `style` | `string` | Musical genre (18 styles) |
| `bpm` | `float` | Tempo in beats per minute |
| `beat_type` | `string` | "beat" (steady pattern) or "fill" (fill passage) |
| `time_signature` | `string` | Time signature (e.g., 4-4, 6-8) |
| `duration` | `float` | Duration in seconds |
| `split` | `string` | Dataset split: train, valid, or test |

### Data Splits

| Split | Examples | Description |
|-------|----------|-------------|
| `train` | ~800 | Training set |
| `valid` | ~200 | Validation set |
| `test` | ~150 | Test set |

Splits are defined in the `info.csv` metadata file.

## Class Taxonomy

The dataset uses 9 canonical drum instrument classes, mapped from 22 raw MIDI pitches on the Roland TD-11:

| Class | Instrument | MIDI Pitches |
|-------|-----------|-------------|
| 0 | Kick | 36 |
| 1 | Snare (Head) | 38 |
| 2 | Snare (Rim) | 37, 40 |
| 3 | Closed Hi-Hat | 42, 22 |
| 4 | Open Hi-Hat | 46, 26 |
| 5 | Low Tom | 43, 58 |
| 6 | Mid Tom | 47, 45 |
| 7 | High Tom | 50, 48 |
| 8 | Ride / Crash | 49, 51, 55, 57, 52 |

## Usage Examples

### Parse MIDI onsets

```python
import pretty_midi

midi = pretty_midi.PrettyMIDI("path/to/file.mid")
for instrument in midi.instruments:
    for note in instrument.notes:
        print(f"Time: {note.start:.3f}s, Pitch: {note.pitch}, Velocity: {note.velocity}")
```

### Load metadata

```python
import pandas as pd

info = pd.read_csv("info.csv")
# Filter by style
rock = info[info["style"] == "rock"]
print(f"Rock tracks: {len(rock)}, Total duration: {rock['duration'].sum():.0f}s")
```

## Dataset Creation

### Source Data

All performances were recorded at Magenta's recording studio by 10 professional and semi-professional drummers playing on a **Roland TD-11** electronic drum kit. Drummers performed along with a metronome at specified tempos across 18 musical styles.

### Annotations

Annotations are the raw MIDI output from the electronic drum kit — no manual annotation was needed. MIDI captures exact onset times, pitches, and velocities directly from the pad triggers. Audio was synthesized by the TD-11's internal sound engine.

## Known Limitations

- **Synthesized audio only:** Audio is the TD-11's internal sound engine output, not acoustic drums. This limits generalization to real-world acoustic drum sounds.
- **Single kit:** All recordings use one drum kit (Roland TD-11), so the model may overfit to this kit's timbral characteristics.
- **Limited timbral diversity:** To address this, consider using [E-GMD](https://huggingface.co/datasets/schismaudio/e-gmd) (43 kits) or [StemGMD](https://huggingface.co/datasets/schismaudio/stemgmd) (10 acoustic kits).
- **Electronic pad dynamics:** Velocity response differs from acoustic drums — pad triggers have fixed dynamic curves.

## Related Datasets

This dataset is part of the [Drum Audio Datasets](https://huggingface.co/collections/schismaudio/drum-audio-datasets) collection by [schismaudio](https://huggingface.co/schismaudio). Related datasets:

- [schismaudio/e-gmd](https://huggingface.co/datasets/schismaudio/e-gmd) — 444 hours: E-GMD re-renders GMD performances with 43 different kits
- [schismaudio/stemgmd](https://huggingface.co/datasets/schismaudio/stemgmd) — 1,224 hours: StemGMD provides isolated per-instrument stems from 10 acoustic kits
- [schismaudio/star-drums](https://huggingface.co/datasets/schismaudio/star-drums) — Real-world drum recordings with accompaniment

## Citation

```bibtex
@inproceedings{gillick2019learning,
  title={Learning to Groove with Inverse Sequence Transformations},
  author={Gillick, Jon and Roberts, Adam and Engel, Jesse and Eck, Douglas and Bamman, David},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2019}
}
```

## License

This dataset is released under the [Creative Commons Attribution 4.0 International License (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

You are free to share and adapt this dataset for any purpose, including commercial use, as long as you give appropriate credit to the original authors (Google Magenta).
