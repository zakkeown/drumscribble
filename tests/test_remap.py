"""Tests for STAR 18-class to GM 26-class target remapping."""
import numpy as np
import pytest

from drumscribble.config import (
    GM_NOTE_TO_INDEX, NUM_CLASSES, STAR_ABBREV_TO_GM, STAR_CLASSES,
    TARGET_WIDENING,
)


def test_remap_star_targets_shape():
    """18-class input produces 26-class output."""
    from drumscribble.data.remap import remap_star_targets

    onset_18 = np.zeros((18, 100), dtype=np.float32)
    vel_18 = np.zeros((18, 100), dtype=np.float32)

    onset_26, vel_26 = remap_star_targets(onset_18, vel_18)

    assert onset_26.shape == (26, 100)
    assert vel_26.shape == (26, 100)
    assert onset_26.dtype == np.float32
    assert vel_26.dtype == np.float32


def test_remap_star_targets_mapping():
    """Bass drum (STAR index 0) maps to GM note 36 (GM index 1)."""
    from drumscribble.data.remap import remap_star_targets

    onset_18 = np.zeros((18, 50), dtype=np.float32)
    vel_18 = np.zeros((18, 50), dtype=np.float32)

    # BD is STAR index 0, set onset at frame 25
    onset_18[0, 25] = 1.0
    vel_18[0, 25] = 0.8

    onset_26, vel_26 = remap_star_targets(onset_18, vel_18)

    # BD -> GM note 36 -> GM index 1
    gm_idx = GM_NOTE_TO_INDEX[36]
    assert gm_idx == 1

    # Onset should have widening pattern centered at frame 25
    assert onset_26[gm_idx, 25] == pytest.approx(1.0)
    assert onset_26[gm_idx, 24] == pytest.approx(0.6)
    assert onset_26[gm_idx, 23] == pytest.approx(0.3)
    assert onset_26[gm_idx, 26] == pytest.approx(0.6)
    assert onset_26[gm_idx, 27] == pytest.approx(0.3)

    # Velocity only at center frame
    assert vel_26[gm_idx, 25] == pytest.approx(0.8)
    assert vel_26[gm_idx, 24] == 0.0  # no velocity on widened frames


def test_remap_star_targets_all_classes_mapped():
    """Every STAR class maps to a valid GM index."""
    from drumscribble.data.remap import remap_star_targets

    onset_18 = np.zeros((18, 20), dtype=np.float32)
    vel_18 = np.zeros((18, 20), dtype=np.float32)

    # Set all 18 classes active at frame 10
    for i in range(18):
        onset_18[i, 10] = 1.0
        vel_18[i, 10] = 0.5

    onset_26, vel_26 = remap_star_targets(onset_18, vel_18)

    # Check each STAR class landed in the right GM row
    for star_idx, abbrev in enumerate(STAR_CLASSES):
        gm_note = STAR_ABBREV_TO_GM[abbrev]
        gm_idx = GM_NOTE_TO_INDEX[gm_note]
        assert onset_26[gm_idx, 10] == pytest.approx(1.0), f"{abbrev} missing at GM idx {gm_idx}"
        assert vel_26[gm_idx, 10] == pytest.approx(0.5), f"{abbrev} vel missing"


def test_remap_star_targets_unmapped_rows_zero():
    """GM classes with no STAR equivalent stay zero."""
    from drumscribble.data.remap import remap_star_targets

    onset_18 = np.ones((18, 10), dtype=np.float32)
    vel_18 = np.ones((18, 10), dtype=np.float32)

    onset_26, vel_26 = remap_star_targets(onset_18, vel_18)

    # Find GM indices NOT covered by any STAR class
    mapped_gm_indices = set()
    for abbrev in STAR_CLASSES:
        gm_note = STAR_ABBREV_TO_GM[abbrev]
        mapped_gm_indices.add(GM_NOTE_TO_INDEX[gm_note])

    for gm_idx in range(NUM_CLASSES):
        if gm_idx not in mapped_gm_indices:
            assert onset_26[gm_idx].sum() == 0.0, f"GM idx {gm_idx} should be zero"
            assert vel_26[gm_idx].sum() == 0.0


def test_remap_widening_clamps_to_one():
    """Overlapping widened onsets don't exceed 1.0."""
    from drumscribble.data.remap import remap_star_targets

    onset_18 = np.zeros((18, 10), dtype=np.float32)
    vel_18 = np.zeros((18, 10), dtype=np.float32)

    # Two BD onsets 2 frames apart — widening will overlap
    onset_18[0, 3] = 1.0
    onset_18[0, 5] = 1.0

    onset_26, vel_26 = remap_star_targets(onset_18, vel_18)

    gm_idx = GM_NOTE_TO_INDEX[36]  # BD
    assert onset_26[gm_idx].max() <= 1.0
