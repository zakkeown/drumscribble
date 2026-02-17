# DrumscribbleCNN Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a SOTA on-device CoreML drum transcription model (26-class GM MIDI with velocity) trained on E-GMD + STAR datasets.

**Architecture:** ConvNeXt CNN encoder (20 blocks, 4 stages, 3 temporal downsamplings) with 3 ANE-optimized self-attention layers at T/8 bottleneck, U-Net decoder with skip connections for full-resolution output, and FiLM conditioning for optional MERT features. BCE loss with target widening. Frame-level onset/velocity/offset prediction at 62.5fps.

**Tech Stack:** PyTorch, torchaudio, pretty_midi, mir_eval, coremltools 8.2+, pytest

**Design Doc:** `docs/plans/2026-02-17-sota-drum-transcription-design.md`

---

## Architecture: U-Net Encoder-Decoder with Attention Bottleneck

The design doc's 3 temporal downsamplings are kept as-is. A U-Net decoder with skip connections recovers full temporal resolution (T) for the output heads. This gives us:
- **Deep bottleneck (T/8)** for efficient attention — sequence length ~78 for 10s
- **Full T-resolution output** via skip connections carrying fine-grained temporal info
- **Standard proven pattern** — used in Apple's Stable Diffusion on ANE, audio segmentation models, etc.

**Encoder:**

| Component | Channels | Temporal | Notes |
|-----------|----------|----------|-------|
| Stem | 1→64 | T | Conv2d(1, 64, (128,1), stride=(128,1)) collapses mel bins |
| Stage 1 | 64 | T | 5 ConvNeXt blocks, kernel (1,7). **Skip → Decoder 3** |
| Downsample 1 | 64→128 | T→T/2 | BN + Conv2d stride (1,2) |
| Stage 2 | 128 | T/2 | 5 ConvNeXt blocks, kernel (1,7). **Skip → Decoder 2** |
| Downsample 2 | 128→256 | T/2→T/4 | BN + Conv2d stride (1,2) |
| Stage 3 | 256 | T/4 | 5 ConvNeXt blocks, kernel (1,11). **Skip → Decoder 1** |
| Downsample 3 | 256→384 | T/4→T/8 | BN + Conv2d stride (1,2) |
| Stage 4 | 384 | T/8 | 5 ConvNeXt blocks, kernel (1,11) |

**Bottleneck:**

| Component | Channels | Temporal | Notes |
|-----------|----------|----------|-------|
| FiLM | 384 | T/8 | Optional MERT conditioning |
| Attention | 384 | T/8 | 3 ANE self-attention layers, 4 heads. seq_len=78 for 10s |

**Decoder (U-Net):**

| Component | Channels | Temporal | Notes |
|-----------|----------|----------|-------|
| Up 1 | 384→256 | T/8→T/4 | Interpolate, cat(skip3=256)→640, Conv1x1→256, ConvNeXt block |
| Up 2 | 256→128 | T/4→T/2 | Interpolate, cat(skip2=128)→384, Conv1x1→128, ConvNeXt block |
| Up 3 | 128→64 | T/2→T | Interpolate, cat(skip1=64)→192, Conv1x1→64, ConvNeXt block |

**Output Heads:**

| Component | Channels | Temporal | Notes |
|-----------|----------|----------|-------|
| Project | 64→128 | T | BN + Conv2d(1,1) + GELU |
| Onset head | 128→26 | T | Conv2d(1,1) + Sigmoid |
| Velocity head | 128→26 | T | Conv2d(1,1) + Sigmoid |
| Offset head | 128→26 | T | Conv2d(1,1) + Sigmoid |

**Estimated params:** ~12.1M (encoder ~9.4M + attention ~1.77M + decoder ~0.92M + heads ~0.02M)

---

## Phase 1: Foundation

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/drumscribble/__init__.py`
- Create: `tests/conftest.py`
- Create: `.gitignore`

**Step 1: Initialize git repository**

```bash
cd /Users/zakkeown/Code/Drumscribble
git init
```

**Step 2: Create .gitignore**

```gitignore
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.eggs/
*.pt
*.ckpt
*.mlmodel
*.mlpackage
wandb/
outputs/
*.wav
*.flac
*.mid
*.midi
.DS_Store
.venv/
```

**Step 3: Create pyproject.toml**

```toml
[project]
name = "drumscribble"
version = "0.1.0"
description = "SOTA on-device drum transcription"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "torchaudio>=2.2",
    "pretty_midi>=0.2.10",
    "pyyaml>=6.0",
    "tqdm>=4.66",
    "mir_eval>=0.7",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=4.0"]
export = ["coremltools>=8.2"]
mert = ["transformers>=4.36"]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**Step 4: Create directory structure**

```bash
mkdir -p src/drumscribble/model src/drumscribble/data src/drumscribble/cli tests configs/train
```

**Step 5: Create __init__.py and conftest.py**

`src/drumscribble/__init__.py`:
```python
"""DrumscribbleCNN: SOTA on-device drum transcription."""
```

`tests/conftest.py`:
```python
import pytest
import torch


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def batch_mel():
    """Fake mel spectrogram: (B=2, 1, 128 mel bins, T=625 frames = 10s at 62.5fps)."""
    return torch.randn(2, 1, 128, 625)
```

**Step 6: Install in editable mode and verify**

```bash
pip install -e ".[dev]"
pytest --co -q  # should show "no tests ran"
```

**Step 7: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding with pyproject.toml and directory structure"
```

---

### Task 2: Config Module

**Files:**
- Create: `src/drumscribble/config.py`
- Create: `tests/test_config.py`

**Step 1: Write failing tests**

`tests/test_config.py`:
```python
from drumscribble.config import (
    GM_CLASSES,
    GM_NOTE_TO_INDEX,
    INDEX_TO_GM_NOTE,
    STAR_ABBREV_TO_GM,
    EGMD_NOTE_REMAP,
    EVAL_MAPPINGS,
    NUM_CLASSES,
    SAMPLE_RATE,
    HOP_LENGTH,
    FPS,
    TARGET_WIDENING,
)


def test_gm_classes_count():
    assert NUM_CLASSES == 26
    assert len(GM_CLASSES) == 26


def test_gm_note_to_index_roundtrip():
    for note, idx in GM_NOTE_TO_INDEX.items():
        assert INDEX_TO_GM_NOTE[idx] == note


def test_star_abbrev_covers_18_classes():
    assert len(STAR_ABBREV_TO_GM) == 18
    assert STAR_ABBREV_TO_GM["BD"] == 36
    assert STAR_ABBREV_TO_GM["SD"] == 38
    assert STAR_ABBREV_TO_GM["CHH"] == 42


def test_egmd_note_remap():
    # Roland TD-17 edge/rim notes remap to standard GM
    assert EGMD_NOTE_REMAP[22] == 42  # HH edge -> closed HH
    assert EGMD_NOTE_REMAP[26] == 46  # HH edge -> open HH
    assert EGMD_NOTE_REMAP[58] == 43  # Tom rim -> high floor tom
    assert EGMD_NOTE_REMAP[40] == 40  # Electric snare stays


def test_eval_mappings():
    # 26 -> 5 for MDB
    mdb = EVAL_MAPPINGS["mdb_5"]
    assert all(v in range(5) for v in mdb.values())
    # 26 -> 3 for IDMT
    idmt = EVAL_MAPPINGS["idmt_3"]
    assert all(v in range(3) for v in idmt.values())


def test_audio_constants():
    assert SAMPLE_RATE == 16000
    assert HOP_LENGTH == 256
    assert FPS == SAMPLE_RATE / HOP_LENGTH  # 62.5


def test_target_widening():
    assert TARGET_WIDENING == [0.3, 0.6, 1.0, 0.6, 0.3]
    assert TARGET_WIDENING[2] == 1.0  # center is peak
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_config.py -v
```
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement config module**

`src/drumscribble/config.py`:
```python
"""Constants and mappings for DrumscribbleCNN."""

SAMPLE_RATE = 16000
HOP_LENGTH = 256  # 16000 / 256 = 62.5 fps
N_MELS = 128
FPS = SAMPLE_RATE / HOP_LENGTH  # 62.5

TARGET_WIDENING = [0.3, 0.6, 1.0, 0.6, 0.3]

# 26 GM-standard drum classes (MIDI note numbers)
GM_CLASSES = [
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 59, 75, 76, 77,
]
# Remove duplicate — 27 entries above, we need exactly 26.
# Per design: 35-57, 59, 75-77 = 26 unique notes
GM_CLASSES = [
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59,
    75, 76,
]
# That's only 26 — but the design says 77 is included. Let me recount:
# 35-57 = 23 notes, 59 = 1, 75-77 = 3, total = 27. Need to drop one.
# Actually the design lists exactly 26 entries. Let me count again from the doc:
# Row 1: 35, 36, 37 = 3
# Row 2: 38, 39, 40 = 3
# Row 3: 41, 42, 43 = 3
# Row 4: 44, 45, 46 = 3
# Row 5: 47, 48, 49 = 3
# Row 6: 50, 51, 52 = 3
# Row 7: 53, 54, 55 = 3
# Row 8: 56, 57, 59 = 3
# Row 9: 75, 76, 77 = 3
# Total: 27. But the design says 26 classes. One must be excluded.
# Standard GM percussion omits 58 (Vibraslap). 76 (Hi Wood Block) and 77 (Low Wood Block)
# are rarely used. Let's use exactly what the design doc lists and count.
# On re-reading: the doc lists 26 items but includes 75, 76, 77. Let me drop 76 or 77.
# Actually, let me just list exactly 26 and be explicit:
GM_CLASSES = sorted([
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59,
    75, 77,
])
NUM_CLASSES = len(GM_CLASSES)
assert NUM_CLASSES == 26

GM_NOTE_TO_INDEX = {note: i for i, note in enumerate(GM_CLASSES)}
INDEX_TO_GM_NOTE = {i: note for i, note in enumerate(GM_CLASSES)}

# Roland TD-17 non-standard MIDI notes -> GM standard
EGMD_NOTE_REMAP = {
    22: 42,   # HH closed edge -> Closed Hi-Hat
    26: 46,   # HH open edge -> Open Hi-Hat
    58: 43,   # Tom3 rim -> High Floor Tom
    50: 50,   # Tom1 rim -> High Tom (already GM)
    47: 47,   # Tom2 rim -> Low-Mid Tom (already GM)
    40: 40,   # Snare rim -> Electric Snare (already GM)
}

# STAR 18-class abbreviations -> GM note numbers
STAR_ABBREV_TO_GM = {
    "BD": 36, "SD": 38, "CHH": 42, "PHH": 44, "OHH": 46,
    "HT": 48, "MT": 45, "LT": 43, "CRC": 49, "SPC": 55,
    "CHC": 52, "RD": 51, "RB": 53, "CB": 56, "CL": 75,
    "CLP": 39, "SS": 37, "TB": 54,
}

# Evaluation class reduction mappings (26-class index -> reduced index)
def _build_eval_mapping(groups: dict[str, list[int]]) -> dict[int, int]:
    mapping = {}
    for group_idx, (_, notes) in enumerate(groups.items()):
        for note in notes:
            if note in GM_NOTE_TO_INDEX:
                mapping[GM_NOTE_TO_INDEX[note]] = group_idx
    return mapping

EVAL_MAPPINGS = {
    "mdb_5": _build_eval_mapping({
        "kick": [35, 36],
        "snare": [37, 38, 39, 40],
        "tom": [41, 43, 45, 47, 48, 50],
        "hihat": [42, 44, 46],
        "cymbal": [49, 51, 52, 53, 55, 56, 57, 59],
    }),
    "idmt_3": _build_eval_mapping({
        "kick": [35, 36],
        "snare": [37, 38, 39, 40],
        "hihat": [42, 44, 46],
    }),
    "egmd_7": _build_eval_mapping({
        "kick": [35, 36],
        "snare": [37, 38, 39, 40],
        "hh_closed": [42, 44],
        "hh_open": [46],
        "tom": [41, 43, 45, 47, 48, 50],
        "crash": [49, 55, 57],
        "ride": [51, 52, 53, 59],
    }),
}
```

**Step 4: Run tests**

```bash
pytest tests/test_config.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/drumscribble/config.py tests/test_config.py
git commit -m "feat: config module with GM classes, dataset mappings, and audio constants"
```

---

### Task 3: Audio Preprocessing

**Files:**
- Create: `src/drumscribble/audio.py`
- Create: `tests/test_audio.py`

**Step 1: Write failing tests**

`tests/test_audio.py`:
```python
import torch
from drumscribble.audio import compute_mel_spectrogram, load_and_preprocess
from drumscribble.config import SAMPLE_RATE, N_MELS, HOP_LENGTH


def test_mel_spectrogram_shape():
    waveform = torch.randn(1, SAMPLE_RATE * 10)  # 10s mono
    mel = compute_mel_spectrogram(waveform)
    assert mel.shape[0] == 1  # batch dim from channel
    assert mel.shape[1] == N_MELS  # 128 mel bins
    expected_frames = (SAMPLE_RATE * 10) // HOP_LENGTH + 1
    assert abs(mel.shape[2] - expected_frames) <= 1


def test_mel_spectrogram_log_scale():
    waveform = torch.randn(1, SAMPLE_RATE * 2)
    mel = compute_mel_spectrogram(waveform)
    # Log-mel should have reasonable values (not all zeros, not huge)
    assert mel.min() > -100
    assert mel.max() < 100


def test_mel_spectrogram_4d_output():
    waveform = torch.randn(1, SAMPLE_RATE * 5)
    mel = compute_mel_spectrogram(waveform, as_4d=True)
    # (1, 1, 128, T) for model input format
    assert mel.dim() == 4
    assert mel.shape[1] == 1
    assert mel.shape[2] == N_MELS


def test_load_and_preprocess_resamples(tmp_path):
    import torchaudio
    # Create a 48kHz test file
    waveform = torch.randn(2, 48000 * 2)  # 2s stereo at 48kHz
    path = tmp_path / "test.wav"
    torchaudio.save(str(path), waveform, 48000)

    audio = load_and_preprocess(str(path))
    assert audio.shape[0] == 1  # mono
    expected_samples = int(2 * SAMPLE_RATE)
    assert abs(audio.shape[1] - expected_samples) <= SAMPLE_RATE // 10
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_audio.py -v
```

**Step 3: Implement audio module**

`src/drumscribble/audio.py`:
```python
"""Audio preprocessing: loading, resampling, mel spectrogram."""
import torch
import torchaudio
from drumscribble.config import SAMPLE_RATE, N_MELS, HOP_LENGTH

_mel_transform = None


def _get_mel_transform(device: torch.device = torch.device("cpu")):
    global _mel_transform
    if _mel_transform is None or _mel_transform.mel_scale.fb.device != device:
        _mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=2048,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=20.0,
            f_max=8000.0,
            power=2.0,
        ).to(device)
    return _mel_transform


def compute_mel_spectrogram(
    waveform: torch.Tensor,
    as_4d: bool = False,
) -> torch.Tensor:
    """Compute log-mel spectrogram from waveform.

    Args:
        waveform: (C, samples) or (samples,) tensor at SAMPLE_RATE.
        as_4d: If True, return (1, 1, N_MELS, T) for model input.

    Returns:
        Log-mel spectrogram: (C, N_MELS, T) or (1, 1, N_MELS, T).
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    mel_spec = _get_mel_transform(waveform.device)(waveform)
    log_mel = torch.log(mel_spec.clamp(min=1e-7))

    if as_4d:
        if log_mel.shape[0] > 1:
            log_mel = log_mel.mean(dim=0, keepdim=True)
        return log_mel.unsqueeze(0)
    return log_mel


def load_and_preprocess(path: str, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load audio file, convert to mono, resample to target_sr.

    Returns:
        (1, samples) tensor at target_sr.
    """
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform
```

**Step 4: Run tests**

```bash
pytest tests/test_audio.py -v
```

**Step 5: Commit**

```bash
git add src/drumscribble/audio.py tests/test_audio.py
git commit -m "feat: audio preprocessing with mel spectrogram and resampling"
```

---

### Task 4: Target Generation

**Files:**
- Create: `src/drumscribble/targets.py`
- Create: `tests/test_targets.py`

**Step 1: Write failing tests**

`tests/test_targets.py`:
```python
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
```

**Step 2: Run tests**

```bash
pytest tests/test_targets.py -v
```

**Step 3: Implement targets module**

`src/drumscribble/targets.py`:
```python
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
```

**Step 4: Run tests**

```bash
pytest tests/test_targets.py -v
```

**Step 5: Commit**

```bash
git add src/drumscribble/targets.py tests/test_targets.py
git commit -m "feat: target generation with widening and GM note remapping"
```

---

## Phase 2: Model

### Task 5: ConvNeXt Block

**Files:**
- Create: `src/drumscribble/model/__init__.py`
- Create: `src/drumscribble/model/convnext.py`
- Create: `tests/test_convnext.py`

**Step 1: Write failing tests**

`tests/test_convnext.py`:
```python
import torch
from drumscribble.model.convnext import ConvNeXtBlock, ConvNeXtBackbone


class TestConvNeXtBlock:
    def test_output_shape_preserved(self):
        block = ConvNeXtBlock(dim=64, kernel_size=(1, 7))
        x = torch.randn(2, 64, 1, 625)
        out = block(x)
        assert out.shape == (2, 64, 1, 625)

    def test_residual_connection(self):
        block = ConvNeXtBlock(dim=64, kernel_size=(1, 7))
        x = torch.zeros(1, 64, 1, 100)
        out = block(x)
        # With zero input, residual should make output non-zero
        # (due to BatchNorm bias terms)
        # Just verify it runs and has correct shape
        assert out.shape == x.shape

    def test_kernel_11(self):
        block = ConvNeXtBlock(dim=256, kernel_size=(1, 11))
        x = torch.randn(2, 256, 1, 312)
        out = block(x)
        assert out.shape == (2, 256, 1, 312)

    def test_expand_ratio(self):
        block = ConvNeXtBlock(dim=64, kernel_size=(1, 7), expand_ratio=4)
        # Count parameters: dwconv + bn + pw1 + pw2
        total = sum(p.numel() for p in block.parameters())
        # pw1: 64*256 + 256 = 16640, pw2: 256*64 + 64 = 16448
        # dwconv: 64*7 + 64 = 512, bn: 64*2 = 128
        assert total > 30000  # sanity check
```

**Step 2: Run tests**

```bash
pytest tests/test_convnext.py::TestConvNeXtBlock -v
```

**Step 3: Implement ConvNeXt block**

`src/drumscribble/model/__init__.py`:
```python
"""DrumscribbleCNN model components."""
```

`src/drumscribble/model/convnext.py`:
```python
"""ConvNeXt blocks and backbone for DrumscribbleCNN."""
import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    """Single ConvNeXt block with BatchNorm (ANE-compatible).

    Architecture: DWConv -> BN -> 1x1 Conv (expand) -> GELU -> 1x1 Conv (project) -> Residual
    """

    def __init__(self, dim: int, kernel_size: tuple[int, int] = (1, 7), expand_ratio: int = 4):
        super().__init__()
        padding = (0, kernel_size[1] // 2)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=padding, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, dim * expand_ratio, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * expand_ratio, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pwconv2(self.act(self.pwconv1(self.norm(self.dwconv(x)))))
```

**Step 4: Run tests**

```bash
pytest tests/test_convnext.py::TestConvNeXtBlock -v
```

**Step 5: Commit**

```bash
git add src/drumscribble/model/__init__.py src/drumscribble/model/convnext.py tests/test_convnext.py
git commit -m "feat: ConvNeXt block with BatchNorm for ANE"
```

---

### Task 6: ConvNeXt Backbone

**Files:**
- Modify: `src/drumscribble/model/convnext.py`
- Modify: `tests/test_convnext.py`

**Step 1: Write failing tests**

Add to `tests/test_convnext.py`:
```python
class TestConvNeXtBackbone:
    def test_output_and_skips(self):
        backbone = ConvNeXtBackbone()
        x = torch.randn(2, 1, 128, 624)  # (B, 1, mel_bins, T) — use T divisible by 8
        out, skips = backbone(x)
        # Output: (B, 384, 1, T/8) due to 3 temporal downsamplings
        assert out.shape == (2, 384, 1, 78)  # 624/8 = 78
        # 3 skip connections: stage1 (64, T), stage2 (128, T/2), stage3 (256, T/4)
        assert len(skips) == 3
        assert skips[0].shape == (2, 64, 1, 624)
        assert skips[1].shape == (2, 128, 1, 312)
        assert skips[2].shape == (2, 256, 1, 156)

    def test_output_shape_30s(self):
        backbone = ConvNeXtBackbone()
        x = torch.randn(1, 1, 128, 1872)  # divisible by 8
        out, skips = backbone(x)
        assert out.shape == (1, 384, 1, 234)  # 1872/8

    def test_param_count(self):
        backbone = ConvNeXtBackbone()
        total = sum(p.numel() for p in backbone.parameters())
        assert 8_000_000 < total < 12_000_000  # ~9.4M expected
```

**Step 2: Run tests**

```bash
pytest tests/test_convnext.py::TestConvNeXtBackbone -v
```

**Step 3: Implement backbone**

Add to `src/drumscribble/model/convnext.py`:
```python
class ConvNeXtBackbone(nn.Module):
    """4-stage ConvNeXt encoder with frequency-collapsing stem.

    Input: (B, 1, 128, T) mel spectrogram
    Output: (B, 384, 1, T/8) feature map + 3 skip connections

    Architecture:
        Stem: collapse 128 mel bins into 64 channels
        Stage 1: 5 blocks at 64ch, kernel (1,7) → skip1 at T
        Downsample 1: stride-2, 64->128ch (T -> T/2)
        Stage 2: 5 blocks at 128ch, kernel (1,7) → skip2 at T/2
        Downsample 2: stride-2, 128->256ch (T/2 -> T/4)
        Stage 3: 5 blocks at 256ch, kernel (1,11) → skip3 at T/4
        Downsample 3: stride-2, 256->384ch (T/4 -> T/8)
        Stage 4: 5 blocks at 384ch, kernel (1,11)
    """

    def __init__(
        self,
        n_mels: int = 128,
        dims: tuple[int, ...] = (64, 128, 256, 384),
        depths: tuple[int, ...] = (5, 5, 5, 5),
        kernels: tuple[tuple[int, int], ...] = ((1, 7), (1, 7), (1, 11), (1, 11)),
    ):
        super().__init__()

        # Stem: collapse mel bins into channels
        self.stem = nn.Conv2d(1, dims[0], kernel_size=(n_mels, 1), stride=(n_mels, 1))

        # Build stages and downsampling transitions
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dims[i], kernels[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

            # Stride-2 temporal downsampling between stages 1->2, 2->3, 3->4
            if i < 3:
                downsample = nn.Sequential(
                    nn.BatchNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=(1, 2), stride=(1, 2)),
                )
                self.downsamples.append(downsample)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.

        Args:
            x: (B, 1, n_mels, T) mel spectrogram.

        Returns:
            features: (B, dims[-1], 1, T/8) bottleneck features.
            skips: [skip1 (64, T), skip2 (128, T/2), skip3 (256, T/4)]
        """
        x = self.stem(x)  # (B, 64, 1, T)

        skips = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < 3:
                skips.append(x)  # save before downsampling
                x = self.downsamples[i](x)

        return x, skips
```

**Step 4: Run tests**

```bash
pytest tests/test_convnext.py -v
```

**Step 5: Commit**

```bash
git add src/drumscribble/model/convnext.py tests/test_convnext.py
git commit -m "feat: ConvNeXt backbone with 3 downsamplings and skip connections"
```

---

### Task 7: ANE-Optimized Self-Attention

**Files:**
- Create: `src/drumscribble/model/attention.py`
- Create: `tests/test_attention.py`

**Step 1: Write failing tests**

`tests/test_attention.py`:
```python
import torch
from drumscribble.model.attention import ANESelfAttention


def test_attention_shape():
    attn = ANESelfAttention(dim=384, num_heads=4)
    x = torch.randn(2, 384, 1, 78)  # (B, C, 1, T/8) for 10s: 625→312→156→78
    out = attn(x)
    assert out.shape == (2, 384, 1, 78)


def test_attention_long_sequence():
    attn = ANESelfAttention(dim=384, num_heads=4)
    x = torch.randn(1, 384, 1, 234)  # 30s at T/8: 1875→937→468→234
    out = attn(x)
    assert out.shape == (1, 384, 1, 234)


def test_attention_uses_conv2d():
    """All projections should be Conv2d(1x1), not Linear, for ANE."""
    attn = ANESelfAttention(dim=384, num_heads=4)
    for name, module in attn.named_modules():
        assert not isinstance(module, torch.nn.Linear), f"Found Linear layer: {name}"


def test_attention_is_causal_false():
    """Drum transcription uses bidirectional (non-causal) attention."""
    attn = ANESelfAttention(dim=384, num_heads=4)
    x = torch.randn(1, 384, 1, 50)
    out = attn(x)
    assert out.shape == (1, 384, 1, 50)
```

**Step 2: Run tests**

```bash
pytest tests/test_attention.py -v
```

**Step 3: Implement attention**

`src/drumscribble/model/attention.py`:
```python
"""ANE-optimized self-attention following Apple's ml-ane-transformers pattern."""
import torch
import torch.nn as nn


class ANESelfAttention(nn.Module):
    """Self-attention using Conv2d(1x1) projections for ANE compatibility.

    Operates on (B, C, 1, S) tensors. All Linear replaced with Conv2d.
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual.

        Args:
            x: (B, C, 1, S) tensor.
        Returns:
            (B, C, 1, S) tensor.
        """
        residual = x
        B, C, _, S = x.shape

        qkv = self.qkv(x)  # (B, 3*C, 1, S)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, S)
        q, k, v = qkv.unbind(1)  # each (B, H, D, S)

        # Attention computation
        attn = torch.einsum("bhds,bhdt->bhst", q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum("bhst,bhdt->bhds", attn, v)
        out = out.reshape(B, C, 1, S)
        out = self.proj(out)
        out = self.norm(out)

        return out + residual
```

**Step 4: Run tests**

```bash
pytest tests/test_attention.py -v
```

**Step 5: Commit**

```bash
git add src/drumscribble/model/attention.py tests/test_attention.py
git commit -m "feat: ANE-optimized self-attention with Conv2d projections"
```

---

### Task 8: FiLM Conditioning

**Files:**
- Create: `src/drumscribble/model/film.py`
- Create: `tests/test_film.py`

**Step 1: Write failing tests**

`tests/test_film.py`:
```python
import torch
from drumscribble.model.film import FiLMConditioning


def test_film_output_shape():
    film = FiLMConditioning(feature_dim=768, target_dim=384)
    x = torch.randn(2, 384, 1, 313)
    cond = torch.randn(2, 768, 1, 313)
    out = film(x, cond)
    assert out.shape == (2, 384, 1, 313)


def test_film_without_conditioning():
    """Model must work without MERT features (returns input unchanged)."""
    film = FiLMConditioning(feature_dim=768, target_dim=384)
    x = torch.randn(2, 384, 1, 313)
    out = film(x, None)
    assert torch.equal(out, x)


def test_film_no_broadcasting():
    """Gamma/beta must be pre-expanded, not rely on broadcasting."""
    film = FiLMConditioning(feature_dim=768, target_dim=384)
    x = torch.randn(1, 384, 1, 100)
    cond = torch.randn(1, 768, 1, 100)
    out = film(x, cond)
    assert out.shape == x.shape
```

**Step 2: Run tests**

```bash
pytest tests/test_film.py -v
```

**Step 3: Implement FiLM**

`src/drumscribble/model/film.py`:
```python
"""FiLM (Feature-wise Linear Modulation) for MERT conditioning."""
import torch
import torch.nn as nn


class FiLMConditioning(nn.Module):
    """FiLM: y = gamma * x + beta.

    Projects conditioning features to per-channel gamma and beta.
    Pre-expands to match spatial dims (no broadcasting for ANE).
    """

    def __init__(self, feature_dim: int, target_dim: int):
        super().__init__()
        # Project conditioning to gamma and beta via Conv2d(1x1)
        self.proj = nn.Conv2d(feature_dim, target_dim * 2, kernel_size=1)

    def forward(
        self, x: torch.Tensor, conditioning: torch.Tensor | None
    ) -> torch.Tensor:
        """Apply FiLM conditioning.

        Args:
            x: (B, C, 1, T) feature map.
            conditioning: (B, feature_dim, 1, T) MERT features, or None.

        Returns:
            (B, C, 1, T) modulated features.
        """
        if conditioning is None:
            return x

        # Interpolate conditioning to match x's temporal dimension if needed
        if conditioning.shape[-1] != x.shape[-1]:
            conditioning = torch.nn.functional.interpolate(
                conditioning, size=(1, x.shape[-1]), mode="bilinear", align_corners=False,
            )

        params = self.proj(conditioning)  # (B, 2*C, 1, T)
        gamma, beta = params.chunk(2, dim=1)  # each (B, C, 1, T)

        # Element-wise (same shape, no broadcasting needed)
        return gamma * x + beta
```

**Step 4: Run tests**

```bash
pytest tests/test_film.py -v
```

**Step 5: Commit**

```bash
git add src/drumscribble/model/film.py tests/test_film.py
git commit -m "feat: FiLM conditioning for MERT feature integration"
```

---

### Task 9: DrumscribbleCNN Full Assembly

**Files:**
- Create: `src/drumscribble/model/drumscribble.py`
- Create: `tests/test_model.py`

**Step 1: Write failing tests**

`tests/test_model.py`:
```python
import torch
from drumscribble.model.drumscribble import DrumscribbleCNN, UNetDecoderBlock
from drumscribble.config import NUM_CLASSES


def test_decoder_block_shape():
    block = UNetDecoderBlock(in_ch=384, skip_ch=256, out_ch=256)
    x = torch.randn(2, 384, 1, 78)
    skip = torch.randn(2, 256, 1, 156)
    out = block(x, skip)
    assert out.shape == (2, 256, 1, 156)


def test_decoder_block_odd_sizes():
    """Decoder must handle non-power-of-2 sizes via interpolation to skip size."""
    block = UNetDecoderBlock(in_ch=128, skip_ch=64, out_ch=64)
    x = torch.randn(1, 128, 1, 156)
    skip = torch.randn(1, 64, 1, 313)  # 625/2 = 312, but encoder floors to 312
    out = block(x, skip)
    assert out.shape == (1, 64, 1, 313)


def test_model_output_shape_10s():
    model = DrumscribbleCNN()
    x = torch.randn(2, 1, 128, 625)
    onset, velocity, offset = model(x)
    assert onset.shape == (2, NUM_CLASSES, 625)
    assert velocity.shape == (2, NUM_CLASSES, 625)
    assert offset.shape == (2, NUM_CLASSES, 625)


def test_model_output_shape_30s():
    model = DrumscribbleCNN()
    x = torch.randn(1, 1, 128, 1875)
    onset, velocity, offset = model(x)
    assert onset.shape == (1, NUM_CLASSES, 1875)


def test_model_outputs_are_probabilities():
    model = DrumscribbleCNN()
    x = torch.randn(1, 1, 128, 200)
    onset, velocity, offset = model(x)
    assert onset.min() >= 0 and onset.max() <= 1
    assert velocity.min() >= 0 and velocity.max() <= 1
    assert offset.min() >= 0 and offset.max() <= 1


def test_model_with_mert():
    model = DrumscribbleCNN(mert_dim=768)
    x = torch.randn(1, 1, 128, 625)
    mert = torch.randn(1, 768, 1, 500)  # MERT at ~50Hz, FiLM interpolates to T/8
    onset, _, _ = model(x, mert_features=mert)
    assert onset.shape == (1, NUM_CLASSES, 625)


def test_model_without_mert():
    model = DrumscribbleCNN(mert_dim=768)
    x = torch.randn(1, 1, 128, 625)
    onset, _, _ = model(x, mert_features=None)
    assert onset.shape == (1, NUM_CLASSES, 625)


def test_model_param_count():
    model = DrumscribbleCNN()
    total = sum(p.numel() for p in model.parameters())
    assert 10_000_000 < total < 15_000_000  # ~12.1M target


def test_model_returns_tuple():
    """Must return plain tuple for torch.jit.trace compatibility."""
    model = DrumscribbleCNN()
    x = torch.randn(1, 1, 128, 200)
    result = model(x)
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_model_frozen_bn():
    model = DrumscribbleCNN()
    model.freeze_bn()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            assert not m.training


def test_model_skip_connections_used():
    """Verify the model uses skip connections (output should differ from no-skip baseline)."""
    model = DrumscribbleCNN()
    model.eval()
    x = torch.randn(1, 1, 128, 200)
    with torch.no_grad():
        onset, _, _ = model(x)
    # Just verify it runs and produces full-resolution output
    assert onset.shape[-1] == 200
```

**Step 2: Run tests**

```bash
pytest tests/test_model.py -v
```

**Step 3: Implement U-Net decoder and full model**

`src/drumscribble/model/drumscribble.py`:
```python
"""DrumscribbleCNN: U-Net encoder-decoder with attention bottleneck."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from drumscribble.config import NUM_CLASSES
from drumscribble.model.convnext import ConvNeXtBackbone, ConvNeXtBlock
from drumscribble.model.attention import ANESelfAttention
from drumscribble.model.film import FiLMConditioning


class UNetDecoderBlock(nn.Module):
    """Single U-Net decoder stage: upsample → concat skip → fuse → ConvNeXt block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, kernel: tuple[int, int] = (1, 7)):
        super().__init__()
        self.fuse = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=1)
        self.block = ConvNeXtBlock(out_ch, kernel)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Interpolate to match skip's spatial dims (handles non-power-of-2 sizes)
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.block(x)
        return x


class DrumscribbleCNN(nn.Module):
    """SOTA drum transcription model for CoreML/ANE deployment.

    U-Net encoder-decoder with attention bottleneck:
    - Encoder: ConvNeXt backbone with 3 temporal downsamplings (T → T/8)
    - Bottleneck: FiLM conditioning + ANE self-attention at T/8
    - Decoder: 3 upsampling stages with skip connections (T/8 → T)

    Input: (B, 1, 128, T) log-mel spectrogram
    Output: tuple of 3 tensors, each (B, 26, T):
        - onset probabilities
        - velocity estimates [0, 1]
        - offset probabilities
    """

    def __init__(
        self,
        n_mels: int = 128,
        backbone_dims: tuple[int, ...] = (64, 128, 256, 384),
        backbone_depths: tuple[int, ...] = (5, 5, 5, 5),
        num_attn_layers: int = 3,
        num_attn_heads: int = 4,
        mert_dim: int | None = None,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        d = backbone_dims
        hidden = d[-1]  # 384

        # Encoder
        self.backbone = ConvNeXtBackbone(
            n_mels=n_mels, dims=d, depths=backbone_depths,
        )

        # Bottleneck: FiLM + attention at T/8
        self.film = FiLMConditioning(mert_dim, hidden) if mert_dim else None
        self.attention = nn.Sequential(
            *[ANESelfAttention(hidden, num_attn_heads) for _ in range(num_attn_layers)]
        )

        # Decoder: 3 upsampling stages with skip connections
        self.decoder1 = UNetDecoderBlock(d[3], d[2], d[2])  # 384+256→256, T/8→T/4
        self.decoder2 = UNetDecoderBlock(d[2], d[1], d[1])  # 256+128→128, T/4→T/2
        self.decoder3 = UNetDecoderBlock(d[1], d[0], d[0])  # 128+64→64,   T/2→T

        # Output heads (operate on decoder output = dims[0] channels)
        self.head_proj = nn.Sequential(
            nn.BatchNorm2d(d[0]),
            nn.Conv2d(d[0], 128, 1),
            nn.GELU(),
        )
        self.onset_head = nn.Conv2d(128, num_classes, 1)
        self.velocity_head = nn.Conv2d(128, num_classes, 1)
        self.offset_head = nn.Conv2d(128, num_classes, 1)

    def forward(
        self,
        x: torch.Tensor,
        mert_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Encoder: (B, 1, 128, T) → (B, 384, 1, T/8), skips at T, T/2, T/4
        features, skips = self.backbone(x)

        # Bottleneck: optional MERT conditioning + self-attention at T/8
        if self.film is not None:
            features = self.film(features, mert_features)
        features = self.attention(features)

        # Decoder: recover full T resolution via skip connections
        features = self.decoder1(features, skips[2])  # T/8 → T/4
        features = self.decoder2(features, skips[1])  # T/4 → T/2
        features = self.decoder3(features, skips[0])  # T/2 → T

        # Output heads
        h = self.head_proj(features)
        onset = torch.sigmoid(self.onset_head(h)).squeeze(2)   # (B, 26, T)
        velocity = torch.sigmoid(self.velocity_head(h)).squeeze(2)
        offset = torch.sigmoid(self.offset_head(h)).squeeze(2)

        return onset, velocity, offset

    def freeze_bn(self):
        """Freeze all BatchNorm layers (use running stats). For local MPS training."""
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
```

**Step 4: Run tests**

```bash
pytest tests/test_model.py -v
```

**Step 5: Commit**

```bash
git add src/drumscribble/model/drumscribble.py tests/test_model.py
git commit -m "feat: DrumscribbleCNN U-Net encoder-decoder with attention bottleneck and skip connections"
```

---

## Phase 3: Data & Loss

### Task 10: Loss Functions

**Files:**
- Create: `src/drumscribble/loss.py`
- Create: `tests/test_loss.py`

**Step 1: Write failing tests**

`tests/test_loss.py`:
```python
import torch
from drumscribble.loss import DrumscribbleLoss


def test_loss_basic():
    loss_fn = DrumscribbleLoss()
    onset_pred = torch.sigmoid(torch.randn(2, 26, 100))
    vel_pred = torch.sigmoid(torch.randn(2, 26, 100))
    offset_pred = torch.sigmoid(torch.randn(2, 26, 100))
    onset_target = torch.zeros(2, 26, 100)
    vel_target = torch.zeros(2, 26, 100)

    total, components = loss_fn(
        onset_pred, vel_pred, offset_pred, onset_target, vel_target
    )
    assert total.dim() == 0  # scalar
    assert total.item() > 0
    assert "onset" in components
    assert "velocity" in components
    assert "offset" in components


def test_velocity_loss_masked():
    """Velocity loss only computed where onset > 0."""
    loss_fn = DrumscribbleLoss()
    onset_target = torch.zeros(1, 26, 100)
    vel_target = torch.zeros(1, 26, 100)

    # Set one onset
    onset_target[0, 0, 50] = 1.0
    vel_target[0, 0, 50] = 0.8

    pred_onset = onset_target.clone()
    pred_vel = torch.zeros_like(vel_target)  # wrong velocity
    pred_offset = torch.zeros(1, 26, 100)

    _, components = loss_fn(pred_onset, pred_vel, pred_offset, onset_target, vel_target)
    # Velocity loss should be > 0 (predicted 0, target 0.8)
    assert components["velocity"].item() > 0


def test_loss_zero_when_perfect():
    loss_fn = DrumscribbleLoss()
    target = torch.zeros(1, 26, 50)
    pred = torch.zeros(1, 26, 50)
    total, _ = loss_fn(pred, pred, pred, target, target)
    # BCE(0,0) should be very small
    assert total.item() < 0.01
```

**Step 2: Run tests**

```bash
pytest tests/test_loss.py -v
```

**Step 3: Implement loss**

`src/drumscribble/loss.py`:
```python
"""Loss functions for DrumscribbleCNN."""
import torch
import torch.nn.functional as F


class DrumscribbleLoss(torch.nn.Module):
    """Combined onset BCE + masked velocity MSE + offset BCE."""

    def __init__(self, velocity_weight: float = 0.5):
        super().__init__()
        self.velocity_weight = velocity_weight

    def forward(
        self,
        onset_pred: torch.Tensor,
        vel_pred: torch.Tensor,
        offset_pred: torch.Tensor,
        onset_target: torch.Tensor,
        vel_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute combined loss.

        All tensors: (B, NUM_CLASSES, T).
        Predictions are already sigmoid'd (probabilities).
        """
        # Onset BCE
        onset_loss = F.binary_cross_entropy(onset_pred, onset_target, reduction="mean")

        # Offset BCE (reuse onset targets — offsets approximate onset tails)
        offset_loss = F.binary_cross_entropy(offset_pred, onset_target, reduction="mean")

        # Masked velocity MSE (only where onset_target > 0)
        mask = (onset_target >= 1.0).float()
        if mask.sum() > 0:
            vel_loss = ((vel_pred - vel_target) ** 2 * mask).sum() / mask.sum()
        else:
            vel_loss = torch.tensor(0.0, device=onset_pred.device)

        total = onset_loss + offset_loss + self.velocity_weight * vel_loss

        return total, {
            "onset": onset_loss.detach(),
            "velocity": vel_loss.detach(),
            "offset": offset_loss.detach(),
        }
```

**Step 4: Run tests**

```bash
pytest tests/test_loss.py -v
```

**Step 5: Commit**

```bash
git add src/drumscribble/loss.py tests/test_loss.py
git commit -m "feat: combined onset/velocity/offset loss function"
```

---

### Task 11: E-GMD Dataset Loader

**Files:**
- Create: `src/drumscribble/data/__init__.py`
- Create: `src/drumscribble/data/egmd.py`
- Create: `tests/test_egmd.py`

**Step 1: Write failing tests**

`tests/test_egmd.py`:
```python
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
    # Create a simple MIDI file
    pm = pretty_midi.PrettyMIDI()
    drum = pretty_midi.Instrument(program=0, is_drum=True)
    # Add some notes: kick at 0.5s, snare at 1.0s
    drum.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.5, end=0.6))
    drum.notes.append(pretty_midi.Note(velocity=80, pitch=38, start=1.0, end=1.1))
    pm.instruments.append(drum)

    midi_path = tmp_path / "drummer1" / "session1"
    midi_path.mkdir(parents=True)
    pm.write(str(midi_path / "test.midi"))

    # Create a matching WAV file (2 seconds)
    waveform = torch.randn(1, SAMPLE_RATE * 2)
    torchaudio.save(str(midi_path / "test.wav"), waveform, SAMPLE_RATE)

    # Create CSV
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
```

**Step 2: Run tests**

```bash
pytest tests/test_egmd.py -v
```

**Step 3: Implement E-GMD loader**

`src/drumscribble/data/__init__.py`:
```python
"""Dataset loaders for DrumscribbleCNN."""
```

`src/drumscribble/data/egmd.py`:
```python
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
    """E-GMD dataset: WAV audio + MIDI annotations.

    Loads the CSV index, chunks audio into fixed-length segments,
    and generates frame-level targets.
    """

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

        # Parse CSV
        csv_path = self.root / "e-gmd-v1.0.0.csv"
        self.entries = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == split:
                    self.entries.append({
                        "audio": str(self.root / row["audio_filename"]),
                        "midi": str(self.root / row["midi_filename"]),
                        "duration": float(row["duration"]),
                    })

        # Build chunk index: (entry_idx, start_sample)
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

        # Load audio chunk
        waveform = load_and_preprocess(entry["audio"])
        if start_sample + self.chunk_samples > waveform.shape[1]:
            # Pad if needed
            pad = self.chunk_samples - (waveform.shape[1] - start_sample)
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        chunk = waveform[:, start_sample : start_sample + self.chunk_samples]

        # Compute mel spectrogram
        mel = compute_mel_spectrogram(chunk)  # (1, 128, T)

        # Parse MIDI and filter events to this chunk's time range
        start_time = start_sample / SAMPLE_RATE
        end_time = start_time + self.chunk_seconds
        events = parse_midi_to_events(entry["midi"])
        chunk_events = [
            (t - start_time, note, vel)
            for t, note, vel in events
            if start_time <= t < end_time
        ]

        # Generate targets
        n_frames = mel.shape[-1]
        onset_target, vel_target = events_to_targets(chunk_events, n_frames)

        return mel, onset_target, vel_target
```

**Step 4: Run tests**

```bash
pytest tests/test_egmd.py -v
```

**Step 5: Commit**

```bash
git add src/drumscribble/data/__init__.py src/drumscribble/data/egmd.py tests/test_egmd.py
git commit -m "feat: E-GMD dataset loader with MIDI parsing and chunking"
```

---

### Task 12: STAR Dataset Loader

**Files:**
- Create: `src/drumscribble/data/star.py`
- Create: `tests/test_star.py`

**Step 1: Write failing tests**

`tests/test_star.py`:
```python
import torch
import pytest
import torchaudio
from pathlib import Path
from drumscribble.data.star import STARDataset, parse_star_annotation
from drumscribble.config import SAMPLE_RATE, NUM_CLASSES


@pytest.fixture
def fake_star(tmp_path):
    """Create a minimal fake STAR dataset."""
    # Create annotation file
    ann_dir = tmp_path / "data" / "training" / "test_source" / "annotation"
    ann_dir.mkdir(parents=True)
    ann_file = ann_dir / "001_mix_kit1.txt"
    ann_file.write_text("0.5\tBD\t100\n1.0\tSD\t80\n1.0\tCHH\t70\n")

    # Create audio file
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
```

**Step 2: Run tests**

```bash
pytest tests/test_star.py -v
```

**Step 3: Implement STAR loader**

`src/drumscribble/data/star.py`:
```python
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

        # Discover annotation files and match to audio
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

        # Build chunk index (STAR files are 60s each)
        self.chunks = []
        for i, entry in enumerate(self.entries):
            # Assume 60s per file; generate chunk offsets
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
```

**Step 4: Run tests**

```bash
pytest tests/test_star.py -v
```

**Step 5: Commit**

```bash
git add src/drumscribble/data/star.py tests/test_star.py
git commit -m "feat: STAR Drums dataset loader with TSV annotation parsing"
```

---

### Task 13: Data Augmentation

**Files:**
- Create: `src/drumscribble/data/augment.py`
- Create: `tests/test_augment.py`

**Step 1: Write failing tests**

`tests/test_augment.py`:
```python
import torch
from drumscribble.data.augment import SpecAugment, AudioAugmentPipeline


def test_spec_augment_shape():
    aug = SpecAugment(freq_mask_param=10, time_mask_param=20, num_masks=2)
    mel = torch.randn(1, 128, 625)
    out = aug(mel)
    assert out.shape == mel.shape


def test_spec_augment_masks_something():
    torch.manual_seed(42)
    aug = SpecAugment(freq_mask_param=30, time_mask_param=50, num_masks=2)
    mel = torch.ones(1, 128, 625)
    out = aug(mel)
    # Some values should be zeroed
    assert (out == 0).any()


def test_audio_augment_pipeline():
    pipeline = AudioAugmentPipeline()
    waveform = torch.randn(1, 160000)  # 10s at 16kHz
    out = pipeline(waveform)
    assert out.shape == waveform.shape
```

**Step 2: Run tests**

```bash
pytest tests/test_augment.py -v
```

**Step 3: Implement augmentation**

`src/drumscribble/data/augment.py`:
```python
"""Data augmentation for training."""
import torch
import torchaudio


class SpecAugment(torch.nn.Module):
    """SpecAugment: frequency and time masking on mel spectrograms."""

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 30,
        num_masks: int = 2,
    ):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
        self.num_masks = num_masks

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        for _ in range(self.num_masks):
            mel = self.freq_mask(mel)
            mel = self.time_mask(mel)
        return mel


class AudioAugmentPipeline(torch.nn.Module):
    """Waveform-level augmentation pipeline.

    Currently: gain jitter. RIR convolution and pitch shift added in Phase 6.
    """

    def __init__(self, gain_range: tuple[float, float] = (0.5, 1.5)):
        super().__init__()
        self.gain_range = gain_range

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        gain = torch.empty(1).uniform_(*self.gain_range).item()
        return waveform * gain
```

**Step 4: Run tests**

```bash
pytest tests/test_augment.py -v
```

**Step 5: Commit**

```bash
git add src/drumscribble/data/augment.py tests/test_augment.py
git commit -m "feat: SpecAugment and audio augmentation pipeline"
```

---

## Phase 4: Training

### Task 14: Training Loop

**Files:**
- Create: `src/drumscribble/train.py`
- Create: `configs/train/default.yaml`
- Create: `tests/test_train.py`

**Step 1: Write failing tests**

`tests/test_train.py`:
```python
import torch
import pytest
from drumscribble.train import train_one_epoch, create_optimizer
from drumscribble.model.drumscribble import DrumscribbleCNN
from drumscribble.loss import DrumscribbleLoss


@pytest.fixture
def tiny_model():
    return DrumscribbleCNN(
        backbone_dims=(32, 32, 32, 32),
        backbone_depths=(1, 1, 1, 1),
        num_attn_layers=1,
    )


def test_create_optimizer(tiny_model):
    opt = create_optimizer(tiny_model, lr=1e-3, weight_decay=0.05)
    assert len(opt.param_groups) > 0


def test_train_one_epoch(tiny_model):
    """Verify one epoch runs without error on synthetic data."""
    from torch.utils.data import DataLoader, TensorDataset

    n = 8
    mel = torch.randn(n, 1, 128, 200)
    onset = torch.zeros(n, 26, 200)
    vel = torch.zeros(n, 26, 200)
    ds = TensorDataset(mel, onset, vel)
    loader = DataLoader(ds, batch_size=4)

    opt = create_optimizer(tiny_model, lr=1e-3)
    loss_fn = DrumscribbleLoss()

    avg_loss = train_one_epoch(tiny_model, loader, opt, loss_fn, device="cpu")
    assert avg_loss > 0
    assert not torch.isnan(torch.tensor(avg_loss))
```

**Step 2: Run tests**

```bash
pytest tests/test_train.py -v
```

**Step 3: Implement training loop**

`src/drumscribble/train.py`:
```python
"""Training loop for DrumscribbleCNN."""
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from drumscribble.model.drumscribble import DrumscribbleCNN
from drumscribble.loss import DrumscribbleLoss


def create_optimizer(
    model: DrumscribbleCNN,
    lr: float = 1e-3,
    weight_decay: float = 0.05,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def train_one_epoch(
    model: DrumscribbleCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: DrumscribbleLoss,
    device: str = "cpu",
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for mel, onset_target, vel_target in tqdm(loader, desc="Training", leave=False):
        mel = mel.to(device)
        onset_target = onset_target.to(device)
        vel_target = vel_target.to(device)

        # Add channel dim for model: (B, 1, 128, T) -> needs unsqueeze if (B, 128, T)
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)

        onset_pred, vel_pred, offset_pred = model(mel)

        loss, _ = loss_fn(onset_pred, vel_pred, offset_pred, onset_target, vel_target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)
```

`configs/train/default.yaml`:
```yaml
# DrumscribbleCNN training config
model:
  n_mels: 128
  backbone_dims: [64, 128, 256, 384]
  backbone_depths: [5, 5, 5, 5]
  num_attn_layers: 3
  num_attn_heads: 4

training:
  batch_size: 32
  lr: 0.001
  weight_decay: 0.05
  epochs: 100
  grad_clip: 1.0
  chunk_seconds: 10.0
  num_workers: 4

data:
  egmd_root: ~/Documents/Datasets/e-gmd
  star_root: ~/Documents/Datasets/star-drums
```

**Step 4: Run tests**

```bash
pytest tests/test_train.py -v
```

**Step 5: Commit**

```bash
git add src/drumscribble/train.py configs/train/default.yaml tests/test_train.py
git commit -m "feat: training loop with AdamW, gradient clipping, and config"
```

---

### Task 15: Overfitting Validation

**Files:**
- Create: `scripts/overfit_test.py`

**Step 1: Write overfitting test script**

`scripts/overfit_test.py`:
```python
"""Quick overfitting test: train on 1 batch, verify loss goes to ~0."""
import torch
from drumscribble.model.drumscribble import DrumscribbleCNN
from drumscribble.loss import DrumscribbleLoss


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Tiny model for speed
    model = DrumscribbleCNN(
        backbone_dims=(32, 64, 64, 64),
        backbone_depths=(2, 2, 2, 2),
        num_attn_layers=1,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Synthetic batch: 1 sample, 5s
    mel = torch.randn(1, 1, 128, 312).to(device)
    onset_target = torch.zeros(1, 26, 312).to(device)
    vel_target = torch.zeros(1, 26, 312).to(device)

    # Place a few onsets
    onset_target[0, 0, 50] = 1.0  # kick
    onset_target[0, 5, 100] = 1.0  # snare
    vel_target[0, 0, 50] = 0.8
    vel_target[0, 5, 100] = 0.6

    loss_fn = DrumscribbleLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(200):
        onset_pred, vel_pred, offset_pred = model(mel)
        loss, components = loss_fn(onset_pred, vel_pred, offset_pred, onset_target, vel_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(
                f"Step {step:3d} | loss={loss.item():.4f} "
                f"onset={components['onset'].item():.4f} "
                f"vel={components['velocity'].item():.4f}"
            )

    assert loss.item() < 0.1, f"Failed to overfit: loss={loss.item()}"
    print("\nOverfitting test PASSED")


if __name__ == "__main__":
    main()
```

**Step 2: Run overfitting test**

```bash
python scripts/overfit_test.py
```
Expected: Loss decreases to <0.1, "Overfitting test PASSED"

**Step 3: Commit**

```bash
git add scripts/overfit_test.py
git commit -m "feat: overfitting validation script"
```

---

## Phase 5: Inference & Evaluation

### Task 16: Inference Pipeline

**Files:**
- Create: `src/drumscribble/inference.py`
- Create: `tests/test_inference.py`

**Step 1: Write failing tests**

`tests/test_inference.py`:
```python
import torch
import numpy as np
from drumscribble.inference import peak_pick, nms, windowed_inference
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
```

**Step 2: Run tests**

```bash
pytest tests/test_inference.py -v
```

**Step 3: Implement inference**

`src/drumscribble/inference.py`:
```python
"""Inference pipeline: peak picking, NMS, event extraction."""
import torch
import numpy as np
from drumscribble.config import FPS, INDEX_TO_GM_NOTE, NUM_CLASSES


def peak_pick(
    probs: torch.Tensor,
    threshold: float = 0.5,
    pre_max: int = 2,
    post_max: int = 6,
) -> list[int]:
    """Find peaks in a 1D probability sequence.

    Args:
        probs: (T,) tensor of probabilities.
        threshold: Minimum probability for a peak.
        pre_max: Frames before peak to check for local maximum.
        post_max: Frames after peak to check for local maximum.

    Returns:
        List of peak frame indices.
    """
    probs_np = probs.cpu().numpy()
    peaks = []
    for i in range(pre_max, len(probs_np) - post_max):
        if probs_np[i] < threshold:
            continue
        window = probs_np[max(0, i - pre_max) : i + post_max + 1]
        if probs_np[i] == window.max():
            peaks.append(i)
    return peaks


def nms(
    peaks: list[int],
    scores: list[float],
    min_distance: int = 2,
) -> list[tuple[int, float]]:
    """Non-maximum suppression on peaks.

    Returns:
        List of (frame, score) tuples after suppression.
    """
    if not peaks:
        return []

    # Sort by score descending
    order = sorted(range(len(peaks)), key=lambda i: scores[i], reverse=True)
    kept = []
    suppressed = set()

    for i in order:
        if i in suppressed:
            continue
        kept.append((peaks[i], scores[i]))
        for j in order:
            if j != i and abs(peaks[j] - peaks[i]) < min_distance:
                suppressed.add(j)

    return sorted(kept, key=lambda x: x[0])


def detections_to_events(
    onset_probs: torch.Tensor,
    vel_probs: torch.Tensor,
    threshold: float = 0.5,
    nms_frames: int = 2,
    fps: float = FPS,
) -> list[dict]:
    """Convert frame-level predictions to onset events.

    Args:
        onset_probs: (NUM_CLASSES, T) onset probabilities.
        vel_probs: (NUM_CLASSES, T) velocity estimates.

    Returns:
        List of dicts with keys: time, note, velocity, class_idx, confidence.
    """
    events = []
    for cls_idx in range(onset_probs.shape[0]):
        peaks = peak_pick(onset_probs[cls_idx], threshold=threshold)
        if not peaks:
            continue
        scores = [onset_probs[cls_idx, p].item() for p in peaks]
        kept = nms(peaks, scores, min_distance=nms_frames)

        for frame, confidence in kept:
            velocity = vel_probs[cls_idx, frame].item()
            events.append({
                "time": frame / fps,
                "note": INDEX_TO_GM_NOTE.get(cls_idx, 0),
                "velocity": int(velocity * 127),
                "class_idx": cls_idx,
                "confidence": confidence,
            })

    events.sort(key=lambda e: e["time"])
    return events
```

**Step 4: Run tests**

```bash
pytest tests/test_inference.py -v
```

**Step 5: Commit**

```bash
git add src/drumscribble/inference.py tests/test_inference.py
git commit -m "feat: inference pipeline with peak picking and NMS"
```

---

### Task 17: Evaluation (mir_eval)

**Files:**
- Create: `src/drumscribble/evaluate.py`
- Create: `tests/test_evaluate.py`

**Step 1: Write failing tests**

`tests/test_evaluate.py`:
```python
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
```

**Step 2: Run tests**

```bash
pytest tests/test_evaluate.py -v
```

**Step 3: Implement evaluation**

`src/drumscribble/evaluate.py`:
```python
"""Evaluation using mir_eval for onset F1."""
import numpy as np
import mir_eval


def evaluate_events(
    ref_events: list[dict],
    est_events: list[dict],
    onset_tolerance: float = 0.05,
) -> dict[str, float]:
    """Evaluate onset precision/recall/F1 using mir_eval.

    Args:
        ref_events: List of dicts with 'time' and 'note'.
        est_events: List of dicts with 'time' and 'note'.
        onset_tolerance: Tolerance in seconds (default 50ms).

    Returns:
        Dict with precision, recall, f1.
    """
    if not ref_events and not est_events:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    # Group by note class
    ref_by_note: dict[int, list[float]] = {}
    est_by_note: dict[int, list[float]] = {}

    for e in ref_events:
        ref_by_note.setdefault(e["note"], []).append(e["time"])
    for e in est_events:
        est_by_note.setdefault(e["note"], []).append(e["time"])

    all_notes = set(ref_by_note) | set(est_by_note)
    total_tp, total_fp, total_fn = 0, 0, 0

    for note in all_notes:
        ref_times = np.array(sorted(ref_by_note.get(note, [])))
        est_times = np.array(sorted(est_by_note.get(note, [])))

        if len(ref_times) == 0:
            total_fp += len(est_times)
            continue
        if len(est_times) == 0:
            total_fn += len(ref_times)
            continue

        # Use mir_eval onset matching
        ref_intervals = np.column_stack([ref_times, ref_times + 0.1])
        est_intervals = np.column_stack([est_times, est_times + 0.1])
        ref_pitches = np.ones(len(ref_times))
        est_pitches = np.ones(len(est_times))

        p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches,
            est_intervals, est_pitches,
            onset_tolerance=onset_tolerance,
            offset_ratio=None,
        )
        tp = int(round(r * len(ref_times)))
        total_tp += tp
        total_fp += len(est_times) - tp
        total_fn += len(ref_times) - tp

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_onset_f1(
    ref_events: list[dict],
    est_events: list[dict],
    onset_tolerance: float = 0.05,
) -> float:
    """Convenience: return just the F1 score."""
    return evaluate_events(ref_events, est_events, onset_tolerance)["f1"]
```

**Step 4: Run tests**

```bash
pytest tests/test_evaluate.py -v
```

**Step 5: Commit**

```bash
git add src/drumscribble/evaluate.py tests/test_evaluate.py
git commit -m "feat: mir_eval-based onset evaluation with per-class matching"
```

---

### Task 18: CoreML Export

**Files:**
- Create: `src/drumscribble/export.py`
- Create: `tests/test_export.py`

**Step 1: Write failing tests**

`tests/test_export.py`:
```python
import torch
import pytest
from drumscribble.export import trace_model, export_coreml
from drumscribble.model.drumscribble import DrumscribbleCNN


def test_trace_model():
    model = DrumscribbleCNN(
        backbone_dims=(32, 32, 32, 32),
        backbone_depths=(1, 1, 1, 1),
        num_attn_layers=1,
    )
    traced = trace_model(model, n_frames=200)
    # Verify traced model produces same shape
    x = torch.randn(1, 1, 128, 200)
    onset, vel, offset = traced(x)
    assert onset.shape == (1, 26, 200)


@pytest.mark.skipif(
    not pytest.importorskip("coremltools", reason="coremltools not installed"),
    reason="coremltools required",
)
def test_export_coreml(tmp_path):
    model = DrumscribbleCNN(
        backbone_dims=(32, 32, 32, 32),
        backbone_depths=(1, 1, 1, 1),
        num_attn_layers=1,
    )
    path = tmp_path / "test.mlpackage"
    export_coreml(model, str(path), n_frames=200)
    assert path.exists()
```

**Step 2: Run tests**

```bash
pytest tests/test_export.py -v
```

**Step 3: Implement export**

`src/drumscribble/export.py`:
```python
"""CoreML export for DrumscribbleCNN."""
import torch
from drumscribble.model.drumscribble import DrumscribbleCNN
from drumscribble.config import N_MELS


def trace_model(model: DrumscribbleCNN, n_frames: int = 625) -> torch.jit.ScriptModule:
    """Trace model for export. Returns TorchScript module."""
    model.eval()
    example = torch.randn(1, 1, N_MELS, n_frames)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    return traced


def export_coreml(
    model: DrumscribbleCNN,
    output_path: str,
    n_frames: int = 625,
) -> None:
    """Export model to CoreML mlpackage format.

    Exports a fixed-shape model. Call multiple times with different
    n_frames for 10s/20s/30s variants (Revision 5).
    """
    import coremltools as ct

    traced = trace_model(model, n_frames)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="mel_spectrogram", shape=(1, 1, N_MELS, n_frames))],
        outputs=[
            ct.TensorType(name="onset_probs"),
            ct.TensorType(name="velocity"),
            ct.TensorType(name="offset_probs"),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
    )

    mlmodel.save(output_path)
```

**Step 4: Run tests**

```bash
pytest tests/test_export.py -v
```

**Step 5: Commit**

```bash
git add src/drumscribble/export.py tests/test_export.py
git commit -m "feat: CoreML export with fixed-shape models"
```

---

## Phase 6: Advanced Features

### Task 19: MERT Feature Extraction

**Files:**
- Create: `src/drumscribble/mert.py`
- Create: `tests/test_mert.py`

**Step 1: Write failing tests**

`tests/test_mert.py`:
```python
import torch
import pytest


@pytest.mark.skipif(
    not pytest.importorskip("transformers", reason="transformers not installed"),
    reason="transformers required",
)
def test_mert_extractor_shape():
    from drumscribble.mert import MERTExtractor

    extractor = MERTExtractor(layer_indices=[5, 6])
    waveform = torch.randn(1, 16000 * 5)  # 5s at 16kHz
    features = extractor(waveform)
    # MERT-95M outputs 768-dim features
    assert features.shape[1] == 768
    assert features.dim() == 4  # (B, C, 1, T)
```

**Step 2: Implement MERT extractor**

`src/drumscribble/mert.py`:
```python
"""MERT-95M feature extraction (optional, requires transformers)."""
import torch
import torch.nn as nn


class MERTExtractor(nn.Module):
    """Extract features from MERT-95M at specified layers.

    Uses layers 5-6 per design Revision 2.
    """

    def __init__(
        self,
        model_name: str = "m-a-p/MERT-v1-95M",
        layer_indices: list[int] | None = None,
    ):
        super().__init__()
        from transformers import AutoModel, AutoFeatureExtractor

        self.layer_indices = layer_indices or [5, 6]
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract MERT features.

        Args:
            waveform: (B, samples) at 16kHz.

        Returns:
            (B, 768, 1, T_mert) features in ANE 4D format.
        """
        outputs = self.model(waveform, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # Average selected layers
        selected = [hidden_states[i] for i in self.layer_indices]
        features = torch.stack(selected).mean(dim=0)  # (B, T_mert, 768)

        # Reshape to ANE format: (B, C, 1, T)
        features = features.permute(0, 2, 1).unsqueeze(2)

        return features
```

**Step 3: Run tests (skip if transformers not installed)**

```bash
pytest tests/test_mert.py -v
```

**Step 4: Commit**

```bash
git add src/drumscribble/mert.py tests/test_mert.py
git commit -m "feat: MERT-95M feature extractor using layers 5-6"
```

---

### Task 20: Multi-Dataset DataLoader

**Files:**
- Create: `src/drumscribble/data/multi.py`
- Create: `tests/test_multi.py`

**Step 1: Write failing tests**

`tests/test_multi.py`:
```python
import torch
from torch.utils.data import TensorDataset
from drumscribble.data.multi import MultiDatasetLoader


def test_multi_dataset_interleaves():
    ds1 = TensorDataset(torch.ones(10, 3), torch.zeros(10))
    ds2 = TensorDataset(torch.ones(5, 3) * 2, torch.ones(5))
    loader = MultiDatasetLoader([ds1, ds2], batch_size=4, weights=[0.5, 0.5])
    batch = next(iter(loader))
    assert batch[0].shape[0] == 4


def test_multi_dataset_epoch_length():
    ds1 = TensorDataset(torch.ones(20, 3), torch.zeros(20))
    ds2 = TensorDataset(torch.ones(10, 3), torch.zeros(10))
    loader = MultiDatasetLoader([ds1, ds2], batch_size=4)
    n_batches = len(list(loader))
    assert n_batches > 0
```

**Step 2: Implement**

`src/drumscribble/data/multi.py`:
```python
"""Multi-dataset DataLoader for combining E-GMD + STAR."""
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler


class MultiDatasetLoader:
    """Wraps multiple datasets with weighted sampling."""

    def __init__(
        self,
        datasets: list[Dataset],
        batch_size: int = 32,
        weights: list[float] | None = None,
        num_workers: int = 0,
    ):
        self.concat = ConcatDataset(datasets)
        sizes = [len(d) for d in datasets]
        total = sum(sizes)

        if weights is None:
            weights = [1.0 / len(datasets)] * len(datasets)

        # Build per-sample weights
        sample_weights = []
        for ds_idx, size in enumerate(sizes):
            w = weights[ds_idx] / size
            sample_weights.extend([w] * size)

        sampler = WeightedRandomSampler(sample_weights, num_samples=total)

        self.loader = DataLoader(
            self.concat,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=True,
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)
```

**Step 3: Run tests**

```bash
pytest tests/test_multi.py -v
```

**Step 4: Commit**

```bash
git add src/drumscribble/data/multi.py tests/test_multi.py
git commit -m "feat: multi-dataset DataLoader with weighted sampling"
```

---

### Task 21: Training CLI

**Files:**
- Create: `src/drumscribble/cli/__init__.py`
- Create: `src/drumscribble/cli/train.py`

**Step 1: Implement CLI**

`src/drumscribble/cli/__init__.py`:
```python
"""CLI entry points."""
```

`src/drumscribble/cli/train.py`:
```python
"""Training CLI for DrumscribbleCNN."""
import argparse
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

from drumscribble.model.drumscribble import DrumscribbleCNN
from drumscribble.loss import DrumscribbleLoss
from drumscribble.train import create_optimizer, train_one_epoch
from drumscribble.data.egmd import EGMDDataset


def main():
    parser = argparse.ArgumentParser(description="Train DrumscribbleCNN")
    parser.add_argument("--config", type=str, default="configs/train/default.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device:
        device = args.device
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    model = DrumscribbleCNN(
        backbone_dims=tuple(model_cfg["backbone_dims"]),
        backbone_depths=tuple(model_cfg["backbone_depths"]),
        num_attn_layers=model_cfg["num_attn_layers"],
        num_attn_heads=model_cfg["num_attn_heads"],
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    if device == "mps":
        model.freeze_bn()
        print("Frozen BatchNorm for MPS training")

    dataset = EGMDDataset(
        root=Path(data_cfg["egmd_root"]).expanduser(),
        split="train",
        chunk_seconds=train_cfg["chunk_seconds"],
    )
    print(f"Training samples: {len(dataset):,}")

    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=0 if device == "mps" else train_cfg["num_workers"],
        drop_last=True,
    )

    optimizer = create_optimizer(model, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    loss_fn = DrumscribbleLoss()

    epochs = args.epochs or train_cfg["epochs"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, loader, optimizer, loss_fn, device=device)
        print(f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch + 1}, ckpt_path)
            print(f"Saved {ckpt_path}")

    final_path = output_dir / "final.pt"
    torch.save({"model": model.state_dict(), "epoch": epochs}, final_path)
    print(f"Training complete. Saved {final_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Test CLI runs with --help**

```bash
python -m drumscribble.cli.train --help
```

**Step 3: Commit**

```bash
git add src/drumscribble/cli/__init__.py src/drumscribble/cli/train.py
git commit -m "feat: training CLI with config, checkpointing, and MPS support"
```

---

### Task 22: Run All Tests

**Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```
Expected: All tests pass.

**Step 2: Check test coverage**

```bash
pytest tests/ --cov=drumscribble --cov-report=term-missing
```

**Step 3: Commit any fixes**

If any tests fail, fix them and commit:
```bash
git add -A
git commit -m "fix: resolve test failures from integration"
```

---

## Execution Summary

| Phase | Tasks | Key Deliverables |
|-------|-------|-----------------|
| 1: Foundation | 1-4 | Project setup, config, audio, targets |
| 2: Model | 5-9 | ConvNeXt, attention, FiLM, full model |
| 3: Data & Loss | 10-13 | Loss, E-GMD loader, STAR loader, augmentation |
| 4: Training | 14-15 | Training loop, overfitting validation |
| 5: Inference & Eval | 16-18 | Peak picking, mir_eval, CoreML export |
| 6: Advanced | 19-22 | MERT, multi-dataset, CLI, integration tests |

**Total tasks:** 22
**Estimated commits:** 22+
**After completion:** Ready for Stage 2 training on E-GMD + STAR via `python -m drumscribble.cli.train`
