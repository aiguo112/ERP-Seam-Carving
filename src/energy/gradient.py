"""Classical gradient energy (Stage A baseline).

ERP assumption: longitude is periodic. Sobel is run on a circularly padded
image so the left/right cut is not treated as a boundary.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from io_erp import circular_pad


def _luminance(img: np.ndarray) -> NDArray[np.float32]:
    arr = np.asarray(img)
    if arr.ndim == 2:
        gray = arr.astype(np.float32, copy=False)
    elif arr.ndim == 3:
        rgb = arr.astype(np.float32, copy=False)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    else:
        raise ValueError(f"img must be HxW or HxWxC, got shape {arr.shape}")
    if np.issubdtype(np.asarray(img).dtype, np.integer) or float(np.max(gray)) > 1.5:
        gray = gray / 255.0
    return gray.astype(np.float32, copy=False)


def gradient_energy(img: np.ndarray) -> NDArray[np.float32]:
    """Sobel gradient magnitude, float32 ``(H, W)``.

    Uses :func:`io_erp.circular_pad` (wrap in ``x``, replicate in ``y``)
    before a 3×3 Sobel so energy is well-defined across the ERP cut.
    """
    gray = _luminance(img)
    h, w = gray.shape
    pad = 1
    padded = circular_pad(gray, pad)
    # Sobel on the padded field; default OpenCV borders only hit the
    # already-padded halo, not the ERP left/right cut.
    gx = cv2.Sobel(padded, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(padded, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    energy = mag[pad : pad + h, pad : pad + w]
    return np.ascontiguousarray(energy, dtype=np.float32)
