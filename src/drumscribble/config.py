"""Constants and mappings for DrumscribbleCNN."""

SAMPLE_RATE = 16000
HOP_LENGTH = 256  # 16000 / 256 = 62.5 fps
N_MELS = 128
FPS = SAMPLE_RATE / HOP_LENGTH  # 62.5

TARGET_WIDENING = [0.3, 0.6, 1.0, 0.6, 0.3]

# 26 GM-standard drum classes (MIDI note numbers)
# 35-57 = 23 notes, 59 = 1, 75 + 77 = 2, total = 26
# (76 Hi Wood Block excluded to reach exactly 26)
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
