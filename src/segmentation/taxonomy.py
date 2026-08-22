"""Named importance maps: VOC, ADE20K, Stanford-13. Select by segmenter/dataset."""

from __future__ import annotations

from segmentation.ade20k import ADE20K_IMPORTANCE, ADE20K_PROTECTED_GROUPS
from segmentation.cvrg import CVRG_IMPORTANCE, CVRG_PROTECTED_GROUPS
from segmentation.stanford import STANFORD_IMPORTANCE, STANFORD_PROTECTED_GROUPS
from segmentation.voc import PROTECTED_GROUPS as VOC_PROTECTED_GROUPS
from segmentation.voc import VOC_IMPORTANCE

TAXONOMIES = ("voc", "ade20k", "stanford13", "cvrg")


def importance_map(taxonomy: str) -> dict[str, float]:
    """Return the named class-importance table (keys are class names)."""
    key = taxonomy.lower().strip()
    if key == "voc":
        return VOC_IMPORTANCE
    if key == "ade20k":
        return ADE20K_IMPORTANCE
    if key == "cvrg":
        return CVRG_IMPORTANCE
    if key in {"stanford13", "stanford", "stanford2d3d"}:
        return STANFORD_IMPORTANCE
    raise ValueError(f"unknown taxonomy '{taxonomy}', expected one of {TAXONOMIES}")


def protected_group_names(taxonomy: str) -> dict[str, tuple[str, ...]]:
    key = taxonomy.lower().strip()
    if key == "voc":
        return VOC_PROTECTED_GROUPS
    if key == "ade20k":
        return ADE20K_PROTECTED_GROUPS
    if key == "cvrg":
        return CVRG_PROTECTED_GROUPS
    if key in {"stanford13", "stanford", "stanford2d3d"}:
        return STANFORD_PROTECTED_GROUPS
    raise ValueError(f"unknown taxonomy '{taxonomy}', expected one of {TAXONOMIES}")


def taxonomy_for(*, dataset: str, segmenter: str | None = None) -> str:
    """Pick a taxonomy from the dataset / segmenter flags, not a global table."""
    ds = dataset.lower().strip()
    if ds == "stanford":
        return "stanford13"
    if ds in {"cvrg", "cvrg-pano", "cvrg_pano"}:
        return "cvrg"
    seg = (segmenter or "").lower().strip()
    if ds == "outdoor" or seg in {"segformer", "ade20k", "segformer-ade20k"}:
        if seg in {"deeplabv3", "deeplab", "voc"}:
            return "voc"
        return "ade20k"
    return "voc"


def id2label_from_names(names: tuple[str, ...] | list[str]) -> dict[int, str]:
    return {i: n for i, n in enumerate(names)}
