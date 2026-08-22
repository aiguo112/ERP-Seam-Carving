"""ERP semantic segmentation via torchvision DeepLabV3-ResNet50.

Pinned weights: ``DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1``
(torchvision >= 0.15). 21 PASCAL VOC classes. Perspective CNN on an
equirectangular image is imperfect — poles are stretched and the model
was not trained on 360° photos. Longitude is circular-padded so the
ERP cut is not a fake image border.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from io_erp import circular_pad
from segmentation.voc import VOC_CLASS_NAMES

MODEL_ARCH = "deeplabv3_resnet50"
WEIGHTS_NAME = "COCO_WITH_VOC_LABELS_V1"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
WRAP_PAD = 32
MAX_SIDE = 512


class ErpSegmenter:
    """Lazy-loaded DeepLabV3; ``predict`` returns HxW int32 VOC class ids."""

    class_names = VOC_CLASS_NAMES
    taxonomy = "voc"

    def __init__(self, device: str | None = None, max_side: int = MAX_SIDE) -> None:
        self.max_side = int(max_side)
        self._device_arg = device
        self._model: Any = None
        self._device: Any = None

    @property
    def id2label(self) -> dict[int, str]:
        """VOC names indexed by the DeepLabV3 class id."""
        return {i: n for i, n in enumerate(self.class_names)}

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from torchvision.models.segmentation import (
            DeepLabV3_ResNet50_Weights,
            deeplabv3_resnet50,
        )

        weights = getattr(DeepLabV3_ResNet50_Weights, WEIGHTS_NAME)
        self._device = torch.device(
            self._device_arg
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = deeplabv3_resnet50(weights=weights)
        model.to(self._device)
        model.eval()
        self._model = model

    def predict(self, img: np.ndarray) -> NDArray[np.int32]:
        """Per-pixel VOC class map, same HxW as the ERP image.

        Does not undistort the sphere. Only the left/right cut is padded
        with wrap-around before the CNN.
        """
        self._load()
        import torch
        import torch.nn.functional as F

        rgb = np.asarray(img)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"expected HxWx3 RGB, got {rgb.shape}")
        rgb = rgb[..., :3]
        h, w = rgb.shape[:2]
        padded = circular_pad(rgb, WRAP_PAD)

        tensor = _to_imagenet_tensor(padded).unsqueeze(0)
        _, _, ph, pw = tensor.shape
        scale = self.max_side / max(ph, pw)
        if scale < 1.0:
            nh, nw = max(1, int(round(ph * scale))), max(1, int(round(pw * scale)))
            tensor = F.interpolate(tensor, size=(nh, nw), mode="bilinear", align_corners=False)

        with torch.inference_mode():
            logits = self._model(tensor.to(self._device))["out"]
            logits = F.interpolate(
                logits, size=(padded.shape[0], padded.shape[1]), mode="bilinear", align_corners=False
            )
            classes = logits.argmax(1)[0].detach().cpu().numpy().astype(np.int32)

        cropped = classes[WRAP_PAD : WRAP_PAD + h, WRAP_PAD : WRAP_PAD + w]
        if cropped.shape != (h, w):
            raise RuntimeError(f"class map shape {cropped.shape} != image {(h, w)}")
        return np.ascontiguousarray(cropped)


def _to_imagenet_tensor(rgb: np.ndarray):
    import torch

    arr = np.asarray(rgb)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    return (tensor - mean) / std
