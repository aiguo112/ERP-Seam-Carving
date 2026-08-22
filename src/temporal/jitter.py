"""Frame-to-frame seam jitter (no temporal model).

``J = mean over contiguous pairs of mean_y cyclic(|S_t(y) - S_{t-1}(y)|)``.
Longitude is periodic, so a jump from ``W-1`` to ``0`` is 1 px, not ``W-1``.

Residual J warps ``S_{t-1}`` by the dominant equatorial horizontal pan
(phase correlation, cyclic in x) before the same cyclic distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

EQUATOR_BAND = 0.15


def cyclic_abs_diff(a: np.ndarray, b: np.ndarray, width: int) -> NDArray[np.float64]:
    """Per-row circular distance on an ERP of width ``width``."""
    x0 = np.asarray(a, dtype=np.float64)
    x1 = np.asarray(b, dtype=np.float64)
    if x0.shape != x1.shape:
        raise ValueError(f"seam shapes {x0.shape} vs {x1.shape}")
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    d = np.abs(x0 - x1)
    return np.minimum(d, width - d)


def _default_pairs(n: int) -> list[tuple[int, int]]:
    return [(t - 1, t) for t in range(1, n)]


def seam_jitter(
    seams: list[np.ndarray],
    width: int,
    pairs: list[tuple[int, int]] | None = None,
) -> float:
    """Mean cyclic path distance on the given pairs (default: consecutive)."""
    if pairs is None:
        pairs = _default_pairs(len(seams))
    if not pairs:
        return 0.0
    acc = 0.0
    for i, j in pairs:
        d = cyclic_abs_diff(seams[j], seams[i], width)
        acc += float(d.mean())
    return acc / len(pairs)


def warp_seam(seam: np.ndarray, pan_x: float, width: int) -> NDArray[np.float64]:
    """Cyclic horizontal wrap of a vertical seam by ``pan_x`` pixels."""
    return np.mod(np.asarray(seam, dtype=np.float64) + float(pan_x), float(width))


def residual_seam_jitter(
    seams: list[np.ndarray],
    pans: list[float],
    width: int,
    pairs: list[tuple[int, int]],
) -> float:
    """Mean cyclic |S_t - warp(S_{t-1}, pan)|. ``pans[k]`` matches ``pairs[k]``."""
    if not pairs:
        return 0.0
    if len(pans) != len(pairs):
        raise ValueError(f"pans ({len(pans)}) vs pairs ({len(pairs)})")
    acc = 0.0
    for (i, j), pan in zip(pairs, pans, strict=True):
        pred = warp_seam(seams[i], pan, width)
        d = cyclic_abs_diff(seams[j], pred, width)
        acc += float(d.mean())
    return acc / len(pairs)


def equatorial_horizontal_pan(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Dominant cyclic yaw (px) mapping ``frame_a`` -> ``frame_b``.

    Phase correlation on the equatorial band. No Hanning along x so ERP
    wrap-around is a valid shift. Hanning along y only (band edges).
    """
    a = _equator_gray(frame_a)
    b = _equator_gray(frame_b)
    if a.shape != b.shape:
        raise ValueError(f"equator shapes {a.shape} vs {b.shape}")
    if a.std() < 1.0 or b.std() < 1.0:
        return 0.0
    a = a - float(a.mean())
    b = b - float(b.mean())
    wy = np.hanning(a.shape[0]).astype(np.float64)[:, None]
    a = a * wy
    b = b * wy
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    # Fb * conj(Fa): peak at +dx when b = roll(a, dx) (content yaw a -> b).
    r = fb * np.conj(fa)
    mag = np.abs(r)
    r = np.divide(r, mag, out=np.zeros_like(r), where=mag > 1e-12)
    corr = np.fft.ifft2(r).real
    dx, _dy = _peak_shift_2d(corr)
    return float(dx)


def photometric_warp_scores(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    pan_x: float,
) -> tuple[float, float]:
    """Equatorial |I_t - I_{t-1}| vs |I_t - roll(I_{t-1}, pan)| in 0-255 gray.

    If the pan estimate is tracking yaw, the warped residual is smaller.
    If it is fitting noise, warping does not help (ratio warped/raw >= 1).
    """
    a = _gray(frame_a)
    b = _gray(frame_b)
    if a.shape != b.shape:
        raise ValueError(f"frame shapes {a.shape} vs {b.shape}")
    h = a.shape[0]
    lo, hi = int(EQUATOR_BAND * h), int((1.0 - EQUATOR_BAND) * h)
    raw = float(
        np.mean(np.abs(a[lo:hi].astype(np.float32) - b[lo:hi].astype(np.float32)))
    )
    shift = int(round(float(pan_x)))
    warped = np.roll(a, shift, axis=1)
    wrp = float(
        np.mean(
            np.abs(warped[lo:hi].astype(np.float32) - b[lo:hi].astype(np.float32))
        )
    )
    return raw, wrp


def photometric_gain(
    frames: list[np.ndarray],
    pairs: list[tuple[int, int]],
    pans: list[float],
) -> tuple[float, float, float]:
    """Mean raw residual, warped residual, and warped/raw over pairs."""
    if not pairs:
        return 0.0, 0.0, 1.0
    if len(pans) != len(pairs):
        raise ValueError(f"pans ({len(pans)}) vs pairs ({len(pairs)})")
    raws: list[float] = []
    wrps: list[float] = []
    for (i, j), pan in zip(pairs, pans, strict=True):
        raw, wrp = photometric_warp_scores(frames[i], frames[j], pan)
        raws.append(raw)
        wrps.append(wrp)
    m_raw = float(np.mean(raws))
    m_wrp = float(np.mean(wrps))
    return m_raw, m_wrp, m_wrp / max(m_raw, 1e-6)


def motion_score(
    frames: list[np.ndarray],
    pairs: list[tuple[int, int]] | None = None,
) -> float:
    """Mean equatorial |I_t - I_{t-1}| in 0-255 gray. Proxy for camera/scene motion."""
    if pairs is None:
        pairs = _default_pairs(len(frames))
    if not pairs:
        return 0.0
    acc = 0.0
    for i, j in pairs:
        a = _gray(frames[i])
        b = _gray(frames[j])
        h = a.shape[0]
        lo, hi = int(EQUATOR_BAND * h), int((1.0 - EQUATOR_BAND) * h)
        acc += float(
            np.mean(
                np.abs(a[lo:hi].astype(np.float32) - b[lo:hi].astype(np.float32))
            )
        )
    return acc / len(pairs)


def motion_label(score: float) -> str:
    """Coarse regime from the equatorial gray delta."""
    if score < 3.0:
        return "static"
    if score < 8.0:
        return "slow"
    if score < 16.0:
        return "moderate"
    return "fast pan"


@dataclass(frozen=True)
class JitterStats:
    J: float
    J_residual: float
    pan_mean_abs: float
    n_pairs: int
    M: float


def compute_jitter_stats(
    frames: list[np.ndarray],
    seams: list[np.ndarray],
    width: int,
    pairs: list[tuple[int, int]],
) -> JitterStats:
    """Raw J, motion-compensated residual J, and mean |pan| on the same pairs."""
    if not pairs:
        return JitterStats(J=0.0, J_residual=0.0, pan_mean_abs=0.0, n_pairs=0, M=0.0)
    pans = [equatorial_horizontal_pan(frames[i], frames[j]) for i, j in pairs]
    return JitterStats(
        J=seam_jitter(seams, width, pairs),
        J_residual=residual_seam_jitter(seams, pans, width, pairs),
        pan_mean_abs=float(np.mean(np.abs(np.asarray(pans, dtype=np.float64)))),
        n_pairs=len(pairs),
        M=motion_score(frames, pairs),
    )


def _equator_gray(img: np.ndarray) -> NDArray[np.float64]:
    g = _gray(img)
    h = g.shape[0]
    lo, hi = int(EQUATOR_BAND * h), int((1.0 - EQUATOR_BAND) * h)
    return g[lo:hi].astype(np.float64)


def _peak_shift_2d(corr: np.ndarray) -> tuple[float, float]:
    h, w = corr.shape
    y, x = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dx = _subpixel_cyclic(corr[int(y), :], int(x))
    dy = _subpixel_cyclic(corr[:, int(x)], int(y))
    if dx > w / 2.0:
        dx -= w
    if dy > h / 2.0:
        dy -= h
    return float(dx), float(dy)


def _subpixel_cyclic(row: np.ndarray, i: int) -> float:
    n = int(row.shape[0])
    cm = float(row[(i - 1) % n])
    c0 = float(row[i])
    cp = float(row[(i + 1) % n])
    denom = 2.0 * (cm - 2.0 * c0 + cp)
    if abs(denom) < 1e-12:
        peak = float(i)
    else:
        peak = float(i) + (cm - cp) / denom
    return float(peak % n)


def _gray(img: np.ndarray) -> NDArray[np.float32]:
    arr = np.asarray(img)
    if arr.ndim == 2:
        g = arr.astype(np.float32)
    else:
        rgb = arr[..., :3].astype(np.float32)
        g = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return g
