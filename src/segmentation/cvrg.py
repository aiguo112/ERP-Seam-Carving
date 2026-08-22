"""CVRG-Pano 21-class taxonomy (outdoor ERP GT).

Ids match SegFormer/Data_Loader.py and ERPSEGFormer/labelmap.txt.
Protected groups are Person and Vehicle — the outdoor analogues of the
handoff table. Sky / road / building stay cheap (sacrifice).
"""

from __future__ import annotations

CVRG_CLASS_NAMES: tuple[str, ...] = (
    "background",
    "car",
    "bicycle",
    "bridge",
    "building",
    "bus",
    "fence",
    "ground",
    "motorcycle",
    "parking area",
    "person",
    "pole",
    "road",
    "sidewalk",
    "sky",
    "terrain",
    "traffic light",
    "traffic sign",
    "truck",
    "vegetation",
    "wall",
)

CVRG_NAME_TO_ID: dict[str, int] = {n: i for i, n in enumerate(CVRG_CLASS_NAMES)}

# RGB palette used in CVRG-Pano label PNGs.
CVRG_COLOR_TO_ID: dict[tuple[int, int, int], int] = {
    (0, 0, 0): 0,
    (113, 174, 206): 1,
    (0, 64, 64): 2,
    (224, 128, 0): 3,
    (240, 240, 20): 4,
    (32, 64, 128): 5,
    (32, 224, 128): 6,
    (64, 0, 160): 7,
    (128, 160, 160): 8,
    (230, 63, 228): 9,
    (192, 96, 96): 10,
    (255, 0, 124): 11,
    (250, 125, 187): 12,
    (240, 146, 26): 13,
    (64, 117, 128): 14,
    (240, 64, 64): 15,
    (19, 92, 211): 16,
    (236, 28, 159): 17,
    (144, 96, 128): 18,
    (144, 32, 192): 19,
    (208, 32, 192): 20,
}

CVRG_IMPORTANCE: dict[str, float] = {
    "background": 0.05,
    "car": 0.90,
    "bicycle": 0.90,
    "bridge": 0.25,
    "building": 0.12,
    "bus": 0.90,
    "fence": 0.30,
    "ground": 0.10,
    "motorcycle": 0.90,
    "parking area": 0.12,
    "person": 1.00,
    "pole": 0.35,
    "road": 0.10,
    "sidewalk": 0.12,
    "sky": 0.08,
    "terrain": 0.12,
    "traffic light": 0.40,
    "traffic sign": 0.40,
    "truck": 0.90,
    "vegetation": 0.15,
    "wall": 0.12,
}

CVRG_PROTECTED_GROUPS: dict[str, tuple[str, ...]] = {
    "person": ("person",),
    "vehicle": ("car", "truck", "bus", "motorcycle", "bicycle"),
}
