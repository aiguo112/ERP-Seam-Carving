"""Semantic energy from a discrete class map.

Three named importance maps, selected by taxonomy / segmenter:

* VOC — DeepLabV3
* ADE20K — SegFormer (names from ``model.config.id2label``)
* Stanford-13 — indoor GT

Void pixels (polar caps / missing labels) are forced to energy 0 and must
be excluded from hit metrics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from segmentation.voc import VOC_CLASS_NAMES, VOC_IMPORTANCE, protected_class_ids

# Backward-compatible alias (VOC table). Prefer ``importance_map("voc")``.
DEFAULT_IMPORTANCE = VOC_IMPORTANCE


def normalize_class_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def semantic_energy(
    class_map: np.ndarray,
    importance_dict: Mapping[str, float] | Mapping[int, float] | None = None,
    *,
    class_names: Sequence[str] | None = None,
    id2label: Mapping[int, str] | Mapping[str, str] | None = None,
    void_mask: np.ndarray | None = None,
    default_weight: float = 0.05,
) -> NDArray[np.float32]:
    """Map each class id to an importance weight, float32 ``(H, W)``.

    ``importance_dict`` keys are class **names** (preferred) or integer ids.
    When ``id2label`` is given (SegFormer ``model.config.id2label``), names
    are matched against that map; unlisted model classes stay
    ``default_weight``. ``void_mask`` pixels are set to 0.
    """
    labels = np.asarray(class_map)
    if labels.ndim != 2:
        raise ValueError(f"class_map must be HxW, got {labels.shape}")
    if id2label is not None:
        lut = lut_from_id2label(id2label, importance_dict or {}, default_weight)
    else:
        names = tuple(class_names) if class_names is not None else VOC_CLASS_NAMES
        name_to_id = _name_to_id(names)
        table: Mapping[str, float] | Mapping[int, float]
        if importance_dict is not None:
            table = importance_dict
        elif names == VOC_CLASS_NAMES:
            table = VOC_IMPORTANCE
        else:
            raise ValueError("importance_dict is required when class_names is not VOC")
        lut = _importance_lut(table, name_to_id, max(len(names), 1), default_weight)
    max_id = int(labels.max()) if labels.size else 0
    if max_id >= lut.size:
        extra = np.full(max_id + 1 - lut.size, default_weight, dtype=np.float32)
        lut = np.concatenate([lut, extra])
    idx = np.clip(labels.astype(np.int32, copy=False), 0, lut.size - 1)
    energy = lut[idx]
    if void_mask is not None:
        vm = np.asarray(void_mask, dtype=bool)
        if vm.shape != labels.shape:
            raise ValueError("void_mask must match class_map shape")
        energy = energy.copy()
        energy[vm] = 0.0
    return np.ascontiguousarray(energy, dtype=np.float32)


def lut_from_id2label(
    id2label: Mapping[int, str] | Mapping[str, str],
    importance_by_name: Mapping[str, float] | Mapping[int, float],
    default_weight: float = 0.05,
) -> NDArray[np.float32]:
    """Build an id->weight LUT by matching importance **names** to ``id2label``.

    Never assumes hardcoded class indices. Model classes not in the
    importance map stay ``default_weight``. Importance names missing from
    ``id2label`` are ignored.
    """
    items = {int(k): str(v) for k, v in id2label.items()}
    n = (max(items) + 1) if items else 1
    names = [f"class_{i}" for i in range(n)]
    for i, name in items.items():
        if i < 0:
            continue
        if i >= len(names):
            names.extend(f"class_{j}" for j in range(len(names), i + 1))
        names[i] = name
    return _importance_lut(
        importance_by_name, _name_to_id(names), len(names), default_weight
    )


def group_ids_from_id2label(
    id2label: Mapping[int, str] | Mapping[str, str],
    groups: Mapping[str, Sequence[str]],
) -> dict[str, frozenset[int]]:
    """Resolve protected-group names to ids via ``id2label``."""
    inv = {normalize_class_name(v): int(k) for k, v in id2label.items()}
    out: dict[str, frozenset[int]] = {}
    for group, names in groups.items():
        ids = []
        for name in names:
            i = inv.get(normalize_class_name(name))
            if i is not None:
                ids.append(i)
        out[group] = frozenset(ids)
    return out


def _name_to_id(names: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, name in enumerate(names):
        out[str(name)] = i
        out[normalize_class_name(name)] = i
    return out


def _importance_lut(
    table: Mapping[str, float] | Mapping[int, float],
    name_to_id: dict[str, int],
    n_classes: int,
    default_weight: float,
) -> NDArray[np.float32]:
    lut = np.full(n_classes, float(default_weight), dtype=np.float32)
    for key, value in table.items():
        if isinstance(key, str):
            nid = name_to_id.get(key)
            if nid is None:
                nid = name_to_id.get(normalize_class_name(key))
            if nid is None:
                continue
            lut[nid] = float(value)
        else:
            i = int(key)
            if i < 0:
                raise KeyError(f"class id must be >= 0, got {i}")
            if i >= lut.size:
                new = np.full(i + 1, float(default_weight), dtype=np.float32)
                new[: lut.size] = lut
                lut = new
            lut[i] = float(value)
    return lut


def count_protected_hits(
    seam: np.ndarray,
    class_map: np.ndarray,
    groups: dict[str, frozenset[int]] | None = None,
    void_mask: np.ndarray | None = None,
) -> dict[str, int]:
    """Count non-void seam pixels that land on each protected group."""
    s = np.asarray(seam, dtype=np.int32)
    labels = np.asarray(class_map)
    h = labels.shape[0]
    if s.shape != (h,):
        raise ValueError(f"seam length {s.shape} != height {h}")
    ids = labels[np.arange(h), s]
    valid = np.ones(h, dtype=bool)
    if void_mask is not None:
        vm = np.asarray(void_mask, dtype=bool)
        if vm.shape != labels.shape:
            raise ValueError("void_mask must match class_map shape")
        valid &= ~vm[np.arange(h), s]
    group_ids = groups if groups is not None else protected_class_ids()
    out: dict[str, int] = {}
    for group, class_ids in group_ids.items():
        out[group] = int((np.isin(ids, list(class_ids)) & valid).sum())
    return out


def count_valid_seam_pixels(seam: np.ndarray, void_mask: np.ndarray | None) -> int:
    """Non-void seam length (one pixel per row minus void hits)."""
    s = np.asarray(seam, dtype=np.int32)
    if void_mask is None:
        return int(s.size)
    vm = np.asarray(void_mask, dtype=bool)
    return int((~vm[np.arange(vm.shape[0]), s]).sum())
