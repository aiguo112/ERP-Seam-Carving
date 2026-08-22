"""Weighted sum of gradient, spherical-gradient, and semantic energies."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def minmax_normalize(energy: np.ndarray, eps: float = 1e-8) -> NDArray[np.float32]:
    """Scale an HxW map to ``[0, 1]``. Constant maps become zeros."""
    e = np.asarray(energy, dtype=np.float32)
    lo = float(e.min())
    hi = float(e.max())
    if hi - lo < eps:
        return np.zeros_like(e)
    return ((e - lo) / (hi - lo)).astype(np.float32, copy=False)


def combine_energies(
    gradient: np.ndarray,
    spherical: np.ndarray,
    semantic: np.ndarray,
    lambda_g: float = 1.0,
    lambda_p: float = 1.0,
    lambda_s: float = 1.0,
    *,
    normalize: bool = True,
) -> NDArray[np.float32]:
    """``E = λ_g E_g + λ_p E_p + λ_s E_s`` with optional min-max on each term.

    ``spherical`` should be ``apply_spherical_weight(gradient, H)`` — i.e.
    gradient × cos(φ) — not a separate image.
    """
    if gradient.shape != spherical.shape or gradient.shape != semantic.shape:
        raise ValueError(
            "energy maps must share shape, got "
            f"{gradient.shape}, {spherical.shape}, {semantic.shape}"
        )
    if min(lambda_g, lambda_p, lambda_s) < 0:
        raise ValueError("lambdas must be non-negative")
    g = minmax_normalize(gradient) if normalize else np.asarray(gradient, dtype=np.float32)
    p = minmax_normalize(spherical) if normalize else np.asarray(spherical, dtype=np.float32)
    s = minmax_normalize(semantic) if normalize else np.asarray(semantic, dtype=np.float32)
    return np.ascontiguousarray(
        np.float32(lambda_g) * g + np.float32(lambda_p) * p + np.float32(lambda_s) * s,
        dtype=np.float32,
    )
