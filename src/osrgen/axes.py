from __future__ import annotations


AXIS_COLUMNS = {
    "l0": "script",
    "l1": "surge",
    "l2": "sway",
    "r0": "twist",
    "r1": "roll",
    "r2": "pitch",
}

AXIS_ORDER = tuple(AXIS_COLUMNS.keys())
SUPPORTED_AXES = set(AXIS_COLUMNS)
AXIS_SCRIPT_SUFFIXES = {
    "l0": "",
    "l1": "surge",
    "l2": "sway",
    "r0": "twist",
    "r1": "roll",
    "r2": "pitch",
}
