"""Stanford 2D-3D-S 13-class taxonomy and indoor importance table.

Ids: 0 = void/unknown, 1..13 = the architectural classes used in 360° indoor
segmentation papers (S3DIS / 2D-3D-S).
"""

from __future__ import annotations

STANFORD_CLASS_NAMES: tuple[str, ...] = (
    "void",
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "table",
    "chair",
    "sofa",
    "bookcase",
    "board",
    "clutter",
)

STANFORD_NAME_TO_ID: dict[str, int] = {n: i for i, n in enumerate(STANFORD_CLASS_NAMES)}

# Protect vs sacrifice for indoor ERP retargeting.
# High: structural/occluding openings and fixtures people navigate by.
# Low: large redundant surfaces (ceiling / floor / wall).
STANFORD_IMPORTANCE: dict[str, float] = {
    "void": 0.0,
    "ceiling": 0.08,
    "floor": 0.10,
    "wall": 0.12,
    "beam": 0.45,
    "column": 0.50,
    "window": 0.95,
    "door": 1.00,
    "table": 0.40,
    "chair": 0.40,
    "sofa": 0.45,
    "bookcase": 0.90,
    "board": 0.92,
    "clutter": 0.25,
}

STANFORD_PROTECTED_GROUPS: dict[str, tuple[str, ...]] = {
    "door": ("door",),
    "window": ("window",),
    "board": ("board",),
    "bookcase": ("bookcase",),
}

# Raw 2D-3D-S instance_class strings -> 13-class names.
_CLASS_ALIASES: dict[str, str] = {
    "unknown": "void",
    "<unk>": "void",
    "unk": "void",
    "none": "void",
    "whiteboard": "board",
    "white-board": "board",
    "blackboard": "board",
    "board": "board",
    "bookcase": "bookcase",
    "bookshelf": "bookcase",
    "shelf": "bookcase",
}


def stanford_protected_ids() -> dict[str, frozenset[int]]:
    return {
        group: frozenset(STANFORD_NAME_TO_ID[n] for n in names)
        for group, names in STANFORD_PROTECTED_GROUPS.items()
    }


def normalize_stanford_class(name: str) -> str:
    key = name.strip().lower().replace(" ", "")
    if key in STANFORD_NAME_TO_ID:
        return key
    if key in _CLASS_ALIASES:
        return _CLASS_ALIASES[key]
    return "clutter"
