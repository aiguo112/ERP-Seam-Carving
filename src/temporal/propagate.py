"""Non-learned temporal baseline: compensate, then regularize in that frame.

No sequence model. ``S_pred`` is the previous seam warped by the estimated
cyclic yaw. Local refine = banded cyclic DP around ``S_pred``. The temporal
penalty is ``λ_t * cyclic(|x - S_pred(y)|)`` added to spherical energy —
never ``|x - S_{t-1}|``, which would fight correct pan.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from seam.dp import find_vertical_seam
from temporal.jitter import cyclic_abs_diff, warp_seam

BAND_BIG = np.float32(1.0e6)


def cyclic_distance_map(center: np.ndarray, width: int) -> NDArray[np.float32]:
    """HxW cyclic |x - center[y]| on an ERP of width ``width``."""
    c = np.asarray(center, dtype=np.float64).reshape(-1)
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    x = np.arange(width, dtype=np.float64)[None, :]
    d = np.abs(x - c[:, None])
    return np.minimum(d, float(width) - d).astype(np.float32)


def compensated_penalty(
    energy: np.ndarray,
    s_pred: np.ndarray,
    lambda_t: float,
) -> NDArray[np.float32]:
    """``E + λ_t * cyclic(|x - S_pred|)``. ``λ_t = 0`` leaves energy unchanged."""
    e = np.asarray(energy, dtype=np.float32)
    if lambda_t == 0.0:
        return e
    dist = cyclic_distance_map(s_pred, e.shape[1])
    if dist.shape != e.shape:
        raise ValueError(f"S_pred map {dist.shape} vs energy {e.shape}")
    return (e + np.float32(lambda_t) * dist).astype(np.float32, copy=False)


def apply_pred_band(
    energy: np.ndarray,
    s_pred: np.ndarray,
    radius: int,
) -> NDArray[np.float32]:
    """Hard local refine: columns farther than ``radius`` from ``S_pred`` are banned."""
    e = np.asarray(energy, dtype=np.float32).copy()
    dist = cyclic_distance_map(s_pred, e.shape[1])
    e[dist > float(radius)] = BAND_BIG
    return e


def find_temporal_seam(
    energy: np.ndarray,
    *,
    s_pred: np.ndarray | None = None,
    lambda_t: float = 0.0,
    band_radius: int | None = None,
    cyclic: bool = True,
) -> NDArray[np.int32]:
    """Cyclic DP on spherical energy, optionally biased toward ``S_pred``."""
    e = np.asarray(energy, dtype=np.float32)
    if s_pred is not None:
        e = compensated_penalty(e, s_pred, lambda_t)
        if band_radius is not None:
            e = apply_pred_band(e, s_pred, int(band_radius))
    return find_vertical_seam(e, cyclic=cyclic)


def seam_pred(prev_seam: np.ndarray, pan_x: float, width: int) -> NDArray[np.float64]:
    """``S_pred = wrap(S_{t-1} + pan)``. The motion-compensated initializer."""
    return warp_seam(prev_seam, pan_x, width)


def pred_deviation(seam: np.ndarray, s_pred: np.ndarray, width: int) -> float:
    """Mean cyclic |S - S_pred| (what the compensated penalty is shrinking)."""
    return float(cyclic_abs_diff(seam, s_pred, width).mean())
