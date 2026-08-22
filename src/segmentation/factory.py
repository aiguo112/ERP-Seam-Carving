"""Pick a segmenter by flag. Taxonomy follows the segmenter, not a global table."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

SEGMENTERS = ("deeplabv3", "segformer")


def get_segmenter(name: str = "deeplabv3", device: str | None = None) -> Any:
    """Construct the named ERP segmenter (lazy weights)."""
    key = name.lower().strip()
    if key in {"deeplabv3", "deeplab", "voc"}:
        from segmentation.segmenter import ErpSegmenter

        print("loading torchvision deeplabv3_resnet50 (COCO_WITH_VOC_LABELS_V1) ...")
        return ErpSegmenter(device=device)
    if key in {"segformer", "ade20k", "segformer-ade20k"}:
        from segmentation.segformer_ade20k import Ade20kSegformer, SEGFORMER_MODEL_ID

        print(f"loading HuggingFace {SEGFORMER_MODEL_ID} ...")
        print("(If this hangs or 403s, enable a VPN and retry.)")
        return Ade20kSegformer(device=device)
    raise ValueError(f"unknown segmenter '{name}', expected one of {SEGMENTERS}")


def predict_erp(
    img: np.ndarray,
    name: str = "deeplabv3",
    device: str | None = None,
) -> NDArray[np.int32]:
    """Run the named CNN on an ERP RGB image. May download weights (VPN if blocked)."""
    return get_segmenter(name, device=device).predict(img)
