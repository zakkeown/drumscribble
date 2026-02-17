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
