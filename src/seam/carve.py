"""Remove vertical seams to reduce ERP width."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from seam.dp import find_vertical_seam


EnergyFn = Callable[[np.ndarray], NDArray[np.float32]]


def remove_vertical_seam(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Drop one pixel per row. Height unchanged; width becomes ``W - 1``."""
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    s = np.asarray(seam, dtype=np.int32)
    if s.shape != (h,):
        raise ValueError(f"seam shape {s.shape} does not match height {h}")
    if np.any(s < 0) or np.any(s >= w):
        raise ValueError("seam column out of range")
    if w < 2:
        raise ValueError("cannot remove a seam from width < 2")

    keep = np.ones((h, w), dtype=bool)
    keep[np.arange(h), s] = False
    if arr.ndim == 2:
        return arr[keep].reshape(h, w - 1)
    if arr.ndim == 3:
        return arr[keep].reshape(h, w - 1, arr.shape[2])
    raise ValueError(f"unsupported image ndim {arr.ndim}")


def reduce_width(
    img: np.ndarray,
    energy_fn: EnergyFn,
    target_w: int,
    cyclic: bool = False,
    on_seam: Callable[[np.ndarray, np.ndarray, np.ndarray, int], None] | None = None,
) -> np.ndarray:
    """Remove vertical seams until the image width equals ``target_w``.

    Energy is recomputed after every removal. ``on_seam(image, seam,
    energy, i)`` is called *before* each removal (for overlays).
    """
    current = np.array(img, copy=True)
    h, w = current.shape[:2]
    if target_w < 1 or target_w > w:
        raise ValueError(f"target_w must be in [1, {w}], got {target_w}")
    n = w - target_w
    for i in range(n):
        energy = energy_fn(current)
        seam = find_vertical_seam(energy, cyclic=cyclic)
        if on_seam is not None:
            on_seam(current, seam, energy, i)
        current = remove_vertical_seam(current, seam)
    return current


def map_seams_to_original(seams: list[np.ndarray]) -> list[NDArray[np.int32]]:
    """Lift each seam from its shrunk frame back to original column indices.

    ``seams[i]`` is the path removed from the image after ``i`` prior
    removals. The returned paths are 8-connected in the *original* ERP.
    """
    original: list[NDArray[np.int32]] = []
    for seam in seams:
        x = np.asarray(seam, dtype=np.int32).copy()
        for prev in original:
            x = x + (x >= prev).astype(np.int32)
        original.append(x)
    return original
