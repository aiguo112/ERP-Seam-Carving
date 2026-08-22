"""Latitude-aware head hits for PanoCrowd point GT.

A head is a point. A vertical seam hits it if, on the head's row, cyclic
|x_seam - x_head| is within r(phi), where r grows as 1/cos(phi) because ERP
stretches longitude toward the poles.

r_eq = W * (theta_deg / 360): a 2 degree longitude disk at the equator.
cos(phi) is clipped at cos(75 deg) so polar radius stays finite.

Hits are scored on original-width seams (map_seams_to_original), not on
the shrinking frame. Energy disks use the same r(phi) so we protect what
we measure.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

THETA_DEG = 2.0
COS_CLIP = float(np.cos(np.deg2rad(75.0)))  # ~0.2588
# Body-column proxy (locked): meridian strip from the head downward.
BODY_LAT_DEG = 12.0  # head-to-torso angular height in latitude
SHOULDER_SCALE = 2.5  # shoulders ~2.5 head half-widths, per-row r(phi)


def latitude_rad(y: float, height: int) -> float:
    if height < 2:
        return 0.0
    return (float(y) / float(height - 1) - 0.5) * float(np.pi)


def hit_radius_px(
    y: float,
    height: int,
    width: int,
    theta_deg: float = THETA_DEG,
    cos_clip: float = COS_CLIP,
) -> float:
    """Pixel half-width of the head disk at row y."""
    phi = latitude_rad(y, height)
    r_eq = float(width) * (float(theta_deg) / 360.0)
    return max(1.0, r_eq / max(abs(np.cos(phi)), float(cos_clip)))


def equator_radius_px(width: int, theta_deg: float = THETA_DEG) -> float:
    return float(width) * (float(theta_deg) / 360.0)


def cyclic_dx(a: float, b: float, width: int) -> float:
    d = abs(float(a) - float(b))
    return float(min(d, float(width) - d))


def heads_hit_mask(
    seams_original: list[np.ndarray],
    heads: np.ndarray,
    height: int,
    width: int,
    *,
    n_seams: int | None = None,
    theta_deg: float = THETA_DEG,
) -> NDArray[np.bool_]:
    """True per head if any of the first ``n_seams`` original-space seams hit it."""
    pts = np.asarray(heads, dtype=np.float64).reshape(-1, 2)
    n = pts.shape[0]
    if n == 0 or not seams_original:
        return np.zeros(n, dtype=bool)
    use = seams_original if n_seams is None else seams_original[:n_seams]
    hit = np.zeros(n, dtype=bool)
    for i in range(n):
        x, y = float(pts[i, 0]), float(pts[i, 1])
        yi = int(np.clip(round(y), 0, height - 1))
        r = hit_radius_px(yi, height, width, theta_deg=theta_deg)
        for seam in use:
            s = np.asarray(seam, dtype=np.int32)
            if cyclic_dx(float(s[yi]), x, width) <= r:
                hit[i] = True
                break
    return hit


def head_hit_rate(
    seams_original: list[np.ndarray],
    heads: np.ndarray,
    height: int,
    width: int,
    *,
    n_seams: int | None = None,
    theta_deg: float = THETA_DEG,
) -> float:
    mask = heads_hit_mask(
        seams_original, heads, height, width, n_seams=n_seams, theta_deg=theta_deg
    )
    if mask.size == 0:
        return 0.0
    return float(mask.mean())


def paint_person_map(
    height: int,
    width: int,
    heads: np.ndarray,
    theta_deg: float = THETA_DEG,
) -> NDArray[np.int32]:
    """Class map: 1 inside a latitude-aware head disk, 0 elsewhere. Wraps in x."""
    cmap = np.zeros((height, width), dtype=np.int32)
    pts = np.asarray(heads, dtype=np.float64).reshape(-1, 2)
    r_eq = max(equator_radius_px(width, theta_deg), 1.0)
    for x, y in pts:
        r_x = hit_radius_px(y, height, width, theta_deg=theta_deg)
        x0 = int(round(x))
        y0 = int(round(y))
        rx = int(np.ceil(r_x)) + 1
        ry = int(np.ceil(r_eq)) + 1
        y_lo = max(0, y0 - ry)
        y_hi = min(height, y0 + ry + 1)
        for yy in range(y_lo, y_hi):
            dy = (yy - y) / r_eq
            dy2 = dy * dy
            if dy2 > 1.0:
                continue
            for dx_i in range(-rx, rx + 1):
                dx = (x0 + dx_i - x) / r_x
                if dx * dx + dy2 <= 1.0:
                    cmap[yy, (x0 + dx_i) % width] = 1
    return cmap


def body_extent_px(height: int, body_lat_deg: float = BODY_LAT_DEG) -> int:
    """Rows from the head downward covering ``body_lat_deg`` of latitude."""
    if height < 2:
        return 1
    return max(1, int(round((float(body_lat_deg) / 180.0) * float(height - 1))))


def body_radius_px(
    y: float,
    height: int,
    width: int,
    *,
    theta_deg: float = THETA_DEG,
    shoulder_scale: float = SHOULDER_SCALE,
) -> float:
    """Half-width of the body column at row y (latitude-aware)."""
    return float(shoulder_scale) * hit_radius_px(y, height, width, theta_deg=theta_deg)


def bodies_hit_mask(
    seams_original: list[np.ndarray],
    heads: np.ndarray,
    height: int,
    width: int,
    *,
    n_seams: int | None = None,
    theta_deg: float = THETA_DEG,
    body_lat_deg: float = BODY_LAT_DEG,
    shoulder_scale: float = SHOULDER_SCALE,
) -> NDArray[np.bool_]:
    """True per person if any of the first k seams cuts the body column.

    Column: same longitude as the head, from the head row downward by
    ``body_lat_deg`` of latitude. Width at each row is ``shoulder_scale``
    times the 2 deg head radius at that row (ERP stretch).
    """
    pts = np.asarray(heads, dtype=np.float64).reshape(-1, 2)
    n = pts.shape[0]
    if n == 0 or not seams_original:
        return np.zeros(n, dtype=bool)
    use = seams_original if n_seams is None else seams_original[:n_seams]
    dy = body_extent_px(height, body_lat_deg)
    hit = np.zeros(n, dtype=bool)
    for i in range(n):
        x, y = float(pts[i, 0]), float(pts[i, 1])
        y0 = int(np.clip(round(y), 0, height - 1))
        y1 = min(height - 1, y0 + dy)
        for seam in use:
            s = np.asarray(seam, dtype=np.int32)
            for yy in range(y0, y1 + 1):
                r = body_radius_px(
                    yy, height, width, theta_deg=theta_deg, shoulder_scale=shoulder_scale
                )
                if cyclic_dx(float(s[yy]), x, width) <= r:
                    hit[i] = True
                    break
            if hit[i]:
                break
    return hit


def body_hit_rate(
    seams_original: list[np.ndarray],
    heads: np.ndarray,
    height: int,
    width: int,
    *,
    n_seams: int | None = None,
    theta_deg: float = THETA_DEG,
    body_lat_deg: float = BODY_LAT_DEG,
    shoulder_scale: float = SHOULDER_SCALE,
) -> float:
    mask = bodies_hit_mask(
        seams_original,
        heads,
        height,
        width,
        n_seams=n_seams,
        theta_deg=theta_deg,
        body_lat_deg=body_lat_deg,
        shoulder_scale=shoulder_scale,
    )
    if mask.size == 0:
        return 0.0
    return float(mask.mean())


def paint_body_map(
    height: int,
    width: int,
    heads: np.ndarray,
    *,
    theta_deg: float = THETA_DEG,
    body_lat_deg: float = BODY_LAT_DEG,
    shoulder_scale: float = SHOULDER_SCALE,
) -> NDArray[np.int32]:
    """Class map: 1 inside body columns. Used only if the proxy check trips."""
    cmap = np.zeros((height, width), dtype=np.int32)
    pts = np.asarray(heads, dtype=np.float64).reshape(-1, 2)
    dy = body_extent_px(height, body_lat_deg)
    for x, y in pts:
        y0 = int(np.clip(round(y), 0, height - 1))
        y1 = min(height, y0 + dy + 1)
        x0 = int(round(x))
        for yy in range(y0, y1):
            r = body_radius_px(
                yy, height, width, theta_deg=theta_deg, shoulder_scale=shoulder_scale
            )
            rx = int(np.ceil(r)) + 1
            for dx_i in range(-rx, rx + 1):
                xx = x0 + dx_i
                if cyclic_dx(xx, x, width) <= r:
                    cmap[yy, xx % width] = 1
    return cmap


def equatorial_person_cover(
    class_map: np.ndarray,
    band: float = 0.15,
) -> float:
    """Fraction of equatorial pixels (not polar caps) that sit in a person disk."""
    h, w = class_map.shape[:2]
    n = int(np.floor(band * h))
    eq = class_map[n : h - n] if n > 0 else class_map
    if eq.size == 0:
        return 0.0
    return float((eq == 1).mean())
