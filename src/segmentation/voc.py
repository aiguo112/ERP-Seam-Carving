"""PASCAL VOC 21-class names (DeepLabV3 COCO-with-VOC labels)."""

from __future__ import annotations

VOC_CLASS_NAMES: tuple[str, ...] = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

VOC_NAME_TO_ID: dict[str, int] = {n: i for i, n in enumerate(VOC_CLASS_NAMES)}

# DeepLabV3 / PASCAL VOC importance. Higher = more expensive to carve.
VOC_IMPORTANCE: dict[str, float] = {
    "background": 0.05,
    "aeroplane": 0.90,
    "bicycle": 0.90,
    "bird": 0.85,
    "boat": 0.90,
    "bottle": 0.30,
    "bus": 0.90,
    "car": 0.90,
    "cat": 0.85,
    "chair": 0.25,
    "cow": 0.85,
    "diningtable": 0.25,
    "dog": 0.85,
    "horse": 0.85,
    "motorbike": 0.90,
    "person": 1.00,
    "pottedplant": 0.20,
    "sheep": 0.85,
    "sofa": 0.25,
    "train": 0.90,
    "tvmonitor": 0.30,
}

# Super-classes reported by Stage C.
PROTECTED_GROUPS: dict[str, tuple[str, ...]] = {
    "person": ("person",),
    "vehicle": (
        "aeroplane",
        "bicycle",
        "boat",
        "bus",
        "car",
        "motorbike",
        "train",
    ),
    "animal": ("bird", "cat", "cow", "dog", "horse", "sheep"),
}


def protected_class_ids() -> dict[str, frozenset[int]]:
    """Map group name -> VOC class ids."""
    return {
        group: frozenset(VOC_NAME_TO_ID[n] for n in names)
        for group, names in PROTECTED_GROUPS.items()
    }
