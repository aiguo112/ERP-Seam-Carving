"""Spherical area weighting for ERP energy maps.

ERP latitude: ``φ = (y / (H - 1) - 0.5) * π``. Each pixel covers spherical
area ∝ ``cos(φ)``, so pole rows are oversampled and nearly redundant.

Multiplying gradient energy by ``cos(φ)`` makes polar pixels cheaper to
carve. A vertical seam still has **one pixel per row**, so the fraction of
seam pixels that *sit* in the top/bottom polar bands is structurally fixed
(~30% for a 15% band). What changes is *routing*: the DP is willing to
meander in those cheap polar rows in order to hit a low-cost equatorial
column. More lateral motion in the polar 15% bands is the **intended
behavior, not a bug**.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from energy.gradient import gradient_energy


def latitude_weight(H: int) -> NDArray[np.float32]:
    """Return ``cos(φ)`` for each row, shape ``(H,)``.

    ``φ = (y / (H - 1) - 0.5) * π``. On this range ``cos(φ) ≥ 0``,
    peaking at the equator and falling to 0 at the poles.
    """
    if H < 1:
        raise ValueError(f"H must be >= 1, got {H}")
    if H == 1:
        return np.ones(1, dtype=np.float32)
    y = np.arange(H, dtype=np.float32)
    phi = (y / np.float32(H - 1) - np.float32(0.5)) * np.float32(np.pi)
    return np.cos(phi).astype(np.float32)


def apply_spherical_weight(energy: np.ndarray, H: int) -> NDArray[np.float32]:
    """Multiply an HxW energy map by ``cos(φ)`` per row.

    Polar rows (``cos(φ) → 0``) become cheap because they represent less
    spherical area. That is intended, not a bug: later DP should prefer to
    wander there rather than spend budget on the oversampled poles.
    """
    e = np.asarray(energy, dtype=np.float32)
    if e.ndim != 2:
        raise ValueError(f"energy must be HxW, got {e.shape}")
    if e.shape[0] != H:
        raise ValueError(f"H={H} does not match energy height {e.shape[0]}")
    w = latitude_weight(H)[:, None]
    return (e * w).astype(np.float32, copy=False)


def spherical_gradient_energy(img: np.ndarray) -> NDArray[np.float32]:
    """Sobel magnitude × ``cos(φ)``. Drop-in energy for Stage B."""
    e = gradient_energy(img)
    return apply_spherical_weight(e, e.shape[0])


def polar_energy_content(
    energy: np.ndarray,
    band: float = 0.15,
    *,
    polar_valid: np.ndarray | None = None,
) -> dict[str, float]:
    """Unweighted polar vs equatorial energy. Predicts when ``cos(φ)`` has work.

    On outdoor ERP, poles are often already cheap (sky / black caps), so the
    naive Sobel baseline already wanders there and spherical weighting is
    nearly redundant. Polar texture should break that redundancy.

    ``energy`` must be the *unweighted* map (the baseline the DP sees before
    ``cos(φ)``). ``polar_valid`` optionally drops polar void (near-black caps);
    the primary predictor uses all polar pixels, matching what carving sees.
    """
    e = np.asarray(energy, dtype=np.float32)
    if e.ndim != 2:
        raise ValueError(f"energy must be HxW, got {e.shape}")
    polar = polar_row_mask(e.shape[0], band)
    eq = ~polar
    polar_hw = np.broadcast_to(polar[:, None], e.shape)
    eq_pix = np.broadcast_to(eq[:, None], e.shape)
    if polar_valid is not None:
        pv = np.asarray(polar_valid, dtype=bool)
        if pv.shape != e.shape:
            raise ValueError("polar_valid must match energy shape")
        polar_pix = polar_hw & pv
    else:
        polar_pix = polar_hw
    polar_mean = float(e[polar_pix].mean()) if polar_pix.any() else 0.0
    eq_mean = float(e[eq_pix].mean()) if eq_pix.any() else 0.0
    ratio = polar_mean / (eq_mean + 1e-8)
    return {
        "polar_grad": polar_mean,
        "eq_grad": eq_mean,
        "polar_ratio": ratio,
    }


def polar_row_mask(height: int, band: float = 0.15) -> NDArray[np.bool_]:
    """True on the top and bottom ``band`` fraction of rows (polar ERP caps)."""
    if not 0.0 <= band < 0.5:
        raise ValueError(f"band must be in [0, 0.5), got {band}")
    n = int(np.floor(band * height))
    mask = np.zeros(height, dtype=bool)
    if n > 0:
        mask[:n] = True
        mask[height - n :] = True
    return mask


def polar_pixel_fraction(
    seams: list[np.ndarray],
    height: int,
    band: float = 0.15,
) -> float:
    """Fraction of seam pixels whose row lies in the polar bands.

    For vertical seams (one pixel per row) this equals the polar-row
    occupancy ``2 * band`` (up to flooring) for every method.
    """
    if not seams:
        return 0.0
    mask = polar_row_mask(height, band)
    polar = 0
    total = 0
    for seam in seams:
        s = np.asarray(seam)
        if s.shape[0] != height:
            raise ValueError(f"seam length {s.shape[0]} != height {height}")
        polar += int(mask.sum())
        total += int(s.shape[0])
    return polar / total


def polar_routing_share(
    seams: list[np.ndarray],
    height: int,
    widths: int | list[int] | tuple[int, ...],
    band: float = 0.15,
    *,
    cyclic: bool = True,
) -> float:
    """Fraction of lateral seam motion ``|Δx|`` that originates in polar rows.

    This is the quantity that **increases** under ``cos(φ)`` weighting:
    cheap polar pixels absorb the 8-connected wandering so the equator can
    stay on a low-cost longitude.

    ``widths`` is the image width *at each seam* (shrinks after every removal).
    Using the original W would mis-measure wrap steps on later seams.
    """
    if not seams or height < 2:
        return 0.0
    if isinstance(widths, int):
        width_list = [widths] * len(seams)
    else:
        width_list = list(widths)
    if len(width_list) != len(seams):
        raise ValueError("widths must match the number of seams")
    mask = polar_row_mask(height, band)
    polar_dx = 0.0
    total_dx = 0.0
    for seam, width in zip(seams, width_list, strict=True):
        s = np.asarray(seam, dtype=np.int32)
        for y in range(height - 1):
            dx = abs(int(s[y + 1]) - int(s[y]))
            if cyclic:
                dx = min(dx, width - dx)
            total_dx += dx
            if mask[y]:
                polar_dx += dx
    if total_dx == 0.0:
        return 0.0
    return polar_dx / total_dx


def mean_polar_step(
    seams: list[np.ndarray],
    height: int,
    widths: int | list[int] | tuple[int, ...],
    band: float = 0.15,
    *,
    cyclic: bool = True,
) -> float:
    """Mean cyclic ``|Δx|`` of steps that start in a polar-band row.

    Spherical weighting should raise this: polar pixels are cheap, so the
    seam uses those rows to travel. Intended, not a bug.
    """
    if not seams or height < 2:
        return 0.0
    if isinstance(widths, int):
        width_list = [widths] * len(seams)
    else:
        width_list = list(widths)
    mask = polar_row_mask(height, band)
    steps: list[float] = []
    for seam, width in zip(seams, width_list, strict=True):
        s = np.asarray(seam, dtype=np.int32)
        for y in range(height - 1):
            if not mask[y]:
                continue
            dx = abs(int(s[y + 1]) - int(s[y]))
            if cyclic:
                dx = min(dx, width - dx)
            steps.append(float(dx))
    if not steps:
        return 0.0
    return float(np.mean(steps))
