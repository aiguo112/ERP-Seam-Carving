"""Load / save ERP images and pad with periodic longitude.

ERP assumption: ``x`` is longitude. Column ``0`` is adjacent to column
``W-1``, so horizontal padding must wrap. Latitude (``y``) is not periodic;
the poles are real boundaries.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


def load_erp(path: str | Path) -> NDArray[np.uint8]:
    """Load an ERP image as RGB uint8, shape ``(H, W, 3)``."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected a color ERP image, got shape {rgb.shape}")
    return rgb


def save_erp(path: str | Path, image: np.ndarray) -> None:
    """Save an RGB or grayscale image. Creates parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = _to_uint8(image)
    if arr.ndim == 2:
        out = arr
    elif arr.ndim == 3 and arr.shape[2] == 3:
        out = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"unsupported image shape {arr.shape}")
    if not cv2.imwrite(str(path), out):
        raise OSError(f"failed to write {path}")


def circular_pad(img: np.ndarray, pad: int) -> np.ndarray:
    """Pad ``pad`` pixels: wrap in ``x`` (longitude), replicate in ``y``.

    Never treats the left/right ERP edges as boundaries. Poles are padded
    with edge-replicate so north is not glued to south.
    """
    if pad < 0:
        raise ValueError(f"pad must be >= 0, got {pad}")
    arr = np.asarray(img)
    if pad == 0:
        return arr
    if arr.ndim == 2:
        wrapped = np.pad(arr, ((0, 0), (pad, pad)), mode="wrap")
        return np.pad(wrapped, ((pad, pad), (0, 0)), mode="edge")
    if arr.ndim == 3:
        wrapped = np.pad(arr, ((0, 0), (pad, pad), (0, 0)), mode="wrap")
        return np.pad(wrapped, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    raise ValueError(f"img must be HxW or HxWxC, got shape {arr.shape}")


def _to_uint8(image: np.ndarray) -> NDArray[np.uint8]:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.5:
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)
