"""Content damage: does temporal smoothing steal from spherical energy?

The temporal penalty is excluded from this measurement. We score seams on
the Stage-B spherical map that independent carving actually optimizes,
otherwise the regularizer would grade its own homework.

Headline: equatorial path-energy ratio vs the independent seam on the same
frame. Polar rows are cheap by Stage-B design and are not counted as content.
There is no video semantic GT here; spherical energy is the content proxy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from energy.spherical import polar_row_mask


def equator_row_mask(height: int, band: float = 0.15) -> NDArray[np.bool_]:
    """True on non-polar rows (the content that Stage B is trying to protect)."""
    return ~polar_row_mask(height, band)


def path_energy(
    energy: np.ndarray,
    seam: np.ndarray,
    row_mask: np.ndarray | None = None,
) -> float:
    """Mean spherical energy along the seam, optionally restricted to rows."""
    e = np.asarray(energy, dtype=np.float64)
    s = np.asarray(seam, dtype=np.int32).reshape(-1)
    if e.ndim != 2 or s.shape[0] != e.shape[0]:
        raise ValueError(f"energy {e.shape} vs seam {s.shape}")
    vals = e[np.arange(s.shape[0]), s]
    if row_mask is not None:
        m = np.asarray(row_mask, dtype=bool).reshape(-1)
        if m.shape[0] != s.shape[0]:
            raise ValueError("row_mask length must match seam")
        vals = vals[m]
    if vals.size == 0:
        return 0.0
    return float(vals.mean())


def equatorial_path_energy(energy: np.ndarray, seam: np.ndarray, band: float = 0.15) -> float:
    """Mean spherical energy on equatorial rows only."""
    h = int(np.asarray(energy).shape[0])
    return path_energy(energy, seam, equator_row_mask(h, band))


def high_energy_hit_rate(
    energy: np.ndarray,
    seam: np.ndarray,
    *,
    band: float = 0.15,
    q: float = 0.80,
) -> float:
    """Fraction of equatorial seam pixels at or above the frame's equatorial q-quantile.

    Independent seams should sit in cheap pixels. A rise means the temporal
    path is cutting through high-gradient (salient) equatorial content.
    """
    e = np.asarray(energy, dtype=np.float64)
    s = np.asarray(seam, dtype=np.int32).reshape(-1)
    eq = equator_row_mask(e.shape[0], band)
    eq_vals = e[eq, :]
    if eq_vals.size == 0:
        return 0.0
    thresh = float(np.quantile(eq_vals, q))
    rows = np.flatnonzero(eq)
    seam_vals = e[rows, s[rows]]
    return float(np.mean(seam_vals >= thresh))


@dataclass(frozen=True)
class ContentDamage:
    """Temporal vs independent, averaged over frames that used a predecessor."""

    path_energy_ind: float
    path_energy_tmp: float
    equator_energy_ind: float
    equator_energy_tmp: float
    damage_ratio: float
    hit_ind: float
    hit_tmp: float
    n_frames: int


def content_damage(
    energies: list[np.ndarray],
    seams_ind: list[np.ndarray],
    seams_tmp: list[np.ndarray],
    used_pred: list[bool],
    *,
    band: float = 0.15,
    q: float = 0.80,
) -> ContentDamage:
    """``damage_ratio = P_eq(temporal) / P_eq(independent)``.

    Only frames where the temporal method had an ``S_pred`` (not a gap restart).
    ``damage_ratio > 1``: temporal pushed the seam into higher spherical energy.
    """
    pe_i: list[float] = []
    pe_t: list[float] = []
    eq_i: list[float] = []
    eq_t: list[float] = []
    h_i: list[float] = []
    h_t: list[float] = []
    n = 0
    for e, si, st, used in zip(energies, seams_ind, seams_tmp, used_pred, strict=True):
        if not used:
            continue
        pe_i.append(path_energy(e, si))
        pe_t.append(path_energy(e, st))
        eq_i.append(equatorial_path_energy(e, si, band))
        eq_t.append(equatorial_path_energy(e, st, band))
        h_i.append(high_energy_hit_rate(e, si, band=band, q=q))
        h_t.append(high_energy_hit_rate(e, st, band=band, q=q))
        n += 1
    if n == 0:
        return ContentDamage(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0)
    ei = float(np.mean(eq_i))
    et = float(np.mean(eq_t))
    return ContentDamage(
        path_energy_ind=float(np.mean(pe_i)),
        path_energy_tmp=float(np.mean(pe_t)),
        equator_energy_ind=ei,
        equator_energy_tmp=et,
        damage_ratio=et / max(ei, 1e-12),
        hit_ind=float(np.mean(h_i)),
        hit_tmp=float(np.mean(h_t)),
        n_frames=n,
    )
