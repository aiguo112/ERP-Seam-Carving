"""ADE20K-150 importance for SegFormer outdoor ERP.

Keys are **class names**, not integer ids. Ids are resolved at runtime from
``model.config.id2label``. ADE20K reliably labels person and vehicles
(car/bus/truck); it has a single generic ``animal`` class that does not
behave like VOC's bird/cat/dog group, so that name is left unlisted (low
default) rather than copying a VOC animal weight we do not expect to fire.
"""

from __future__ import annotations

# What nvidia SegFormer-ADE actually predicts for outdoor protection.
ADE20K_IMPORTANCE: dict[str, float] = {
    "person": 1.00,
    "car": 0.90,
    "bus": 0.90,
    "truck": 0.90,
    "van": 0.90,
    "bicycle": 0.90,
    "minibike": 0.90,
    "airplane": 0.90,
    "boat": 0.90,
    "ship": 0.90,
}

ADE20K_PROTECTED_GROUPS: dict[str, tuple[str, ...]] = {
    "person": ("person",),
    "vehicle": (
        "car",
        "bus",
        "truck",
        "van",
        "bicycle",
        "minibike",
        "airplane",
        "boat",
        "ship",
    ),
}
