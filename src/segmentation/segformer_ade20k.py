"""HuggingFace SegFormer (ADE20K) for outdoor 360 ERP.

Pinned checkpoint: ``nvidia/segformer-b0-finetuned-ade-512-512``.
Requires ``transformers``; first load downloads weights from Hugging Face
(use a VPN if that host is blocked). Perspective weights on ERP are
imperfect; longitude is wrap-padded so the 0/W cut is not a fake border.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from io_erp import circular_pad

SEGFORMER_MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
WRAP_PAD = 32
MAX_SIDE = 512


class Ade20kSegformer:
    """Lazy-loaded SegFormer-B0 ADE20K; ``predict`` returns HxW int32 class ids."""

    model_id = SEGFORMER_MODEL_ID
    taxonomy = "ade20k"

    def __init__(self, device: str | None = None, max_side: int = MAX_SIDE) -> None:
        self.max_side = int(max_side)
        self._device_arg = device
        self._model: Any = None
        self._processor: Any = None
        self._device: Any = None
        self._id2label: dict[int, str] | None = None

    @classmethod
    def load_id2label(cls) -> dict[int, str]:
        """Read ``config.id2label`` for the pinned checkpoint (config only)."""
        try:
            from transformers import SegformerConfig
        except ImportError as exc:
            raise ImportError("transformers is required for SegFormer ADE20K") from exc
        try:
            config = SegformerConfig.from_pretrained(SEGFORMER_MODEL_ID, revision="main")
        except Exception as exc:  # noqa: BLE001
            raise ConnectionError(
                "Could not download "
                f"{SEGFORMER_MODEL_ID} config from Hugging Face. "
                "If you are behind a firewall, turn on a VPN and retry. "
                f"Original error: {exc}"
            ) from exc
        return _id2label_from_config(config.id2label)

    @property
    def id2label(self) -> dict[int, str]:
        """The loaded model's ``config.id2label`` (never a hardcoded id table)."""
        if self._id2label is None:
            if self._model is not None:
                self._id2label = _id2label_from_config(self._model.config.id2label)
            else:
                self._id2label = self.load_id2label()
        return self._id2label

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        try:
            self._processor = SegformerImageProcessor.from_pretrained(
                SEGFORMER_MODEL_ID,
                revision="main",
                do_resize=False,
                do_reduce_labels=False,
            )
            self._model = SegformerForSemanticSegmentation.from_pretrained(
                SEGFORMER_MODEL_ID,
                revision="main",
            )
        except Exception as exc:  # noqa: BLE001
            raise ConnectionError(
                "Could not download "
                f"{SEGFORMER_MODEL_ID} from Hugging Face. "
                "If you are behind a firewall, turn on a VPN and retry. "
                f"Original error: {exc}"
            ) from exc
        self._device = torch.device(
            self._device_arg or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model.to(self._device)
        self._model.eval()
        self._id2label = _id2label_from_config(self._model.config.id2label)

    def predict(self, img: np.ndarray) -> NDArray[np.int32]:
        """Per-pixel ADE20K class map, same HxW as the ERP image."""
        self._load()
        import torch
        import torch.nn.functional as F
        from PIL import Image

        rgb = np.asarray(img)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(f"expected HxWx3 RGB, got {rgb.shape}")
        rgb = rgb[..., :3]
        h, w = rgb.shape[:2]
        padded = circular_pad(rgb, WRAP_PAD)
        ph, pw = padded.shape[:2]
        scale = self.max_side / max(ph, pw)
        if scale < 1.0:
            nh, nw = max(1, int(round(ph * scale))), max(1, int(round(pw * scale)))
            resized = np.array(
                Image.fromarray(padded).resize((nw, nh), Image.BILINEAR)
            )
        else:
            resized = padded
            nh, nw = ph, pw

        inputs = self._processor(images=Image.fromarray(np.ascontiguousarray(resized)), return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = self._model(**inputs).logits
            logits = F.interpolate(
                logits, size=(ph, pw), mode="bilinear", align_corners=False
            )
            classes = logits.argmax(1)[0].detach().cpu().numpy().astype(np.int32)

        cropped = classes[WRAP_PAD : WRAP_PAD + h, WRAP_PAD : WRAP_PAD + w]
        if cropped.shape != (h, w):
            raise RuntimeError(f"class map shape {cropped.shape} != image {(h, w)}")
        return np.ascontiguousarray(cropped)


def _id2label_from_config(raw: dict) -> dict[int, str]:
    """Normalize HF id2label keys (str or int) to ``{int: name}``."""
    return {int(k): str(v) for k, v in raw.items()}
