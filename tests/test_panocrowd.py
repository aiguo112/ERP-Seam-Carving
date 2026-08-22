"""PanoCrowd head-hit geometry and density bins."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.panocrowd import density_bin  # noqa: E402
from energy.head_hits import (  # noqa: E402
    BODY_LAT_DEG,
    COS_CLIP,
    THETA_DEG,
    bodies_hit_mask,
    body_extent_px,
    body_hit_rate,
    cyclic_dx,
    equator_radius_px,
    head_hit_rate,
    heads_hit_mask,
    hit_radius_px,
    paint_person_map,
)
from seam.carve import map_seams_to_original  # noqa: E402


def test_density_bins_match_panocrowd_groups() -> None:
    assert density_bin(0) == "sparse"
    assert density_bin(49) == "sparse"
    assert density_bin(50) == "medium"
    assert density_bin(199) == "medium"
    assert density_bin(200) == "dense"


def test_equator_radius_is_two_degrees() -> None:
    w = 1800
    assert abs(equator_radius_px(w) - 10.0) < 1e-9
    h = 101
    y_eq = 50
    assert abs(hit_radius_px(y_eq, h, w) - 10.0) < 1e-6


def test_polar_radius_grows_as_one_over_cos() -> None:
    w, h = 360, 181
    y_eq = 90
    # phi = 60 deg: y/(H-1) - 0.5 = 60/180 = 1/3 -> y = 90 + 60 = 150
    y_60 = 150
    r_eq = hit_radius_px(y_eq, h, w)
    r_60 = hit_radius_px(y_60, h, w)
    assert abs(r_eq - 2.0) < 1e-6
    assert abs(r_60 - 4.0) < 1e-5
    y_pole = 0
    r_pole = hit_radius_px(y_pole, h, w)
    assert abs(r_pole - (2.0 / COS_CLIP)) < 1e-5


def test_wraparound_head_hit() -> None:
    h, w = 40, 80
    y = 20
    heads = np.array([[0.0, float(y)]], dtype=np.float64)
    seam = np.full(h, w - 1, dtype=np.int32)
    r = hit_radius_px(y, h, w)
    assert r >= 1.0
    assert cyclic_dx(float(w - 1), 0.0, w) <= r
    assert bool(heads_hit_mask([seam], heads, h, w)[0])


def test_miss_outside_radius() -> None:
    h, w = 40, 180
    y = 20
    heads = np.array([[40.0, float(y)]], dtype=np.float64)
    seam = np.full(h, 80, dtype=np.int32)
    assert not bool(heads_hit_mask([seam], heads, h, w)[0])


def test_nested_seams_cannot_lower_hit_rate() -> None:
    h, w = 32, 64
    heads = np.array([[10.0, 16.0], [40.0, 16.0]], dtype=np.float64)
    s1 = np.full(h, 10, dtype=np.int32)
    s2 = np.full(h, 40, dtype=np.int32)
    mapped = map_seams_to_original([s1, s2])
    r8 = head_hit_rate(mapped, heads, h, w, n_seams=1)
    r24 = head_hit_rate(mapped, heads, h, w, n_seams=2)
    assert r24 >= r8


def test_paint_wraps_and_marks_person() -> None:
    h, w = 40, 80
    heads = np.array([[0.0, 20.0]], dtype=np.float64)
    cmap = paint_person_map(h, w, heads, theta_deg=THETA_DEG)
    assert cmap[20, 0] == 1
    assert cmap[20, 1] == 1
    assert cmap[20, w - 1] == 1
    assert cmap[20, 40] == 0


def test_body_extent_is_12deg_latitude() -> None:
    h = 181
    assert body_extent_px(h) == 12  # 12/180 * 180


def test_body_column_hits_torso_misses_head() -> None:
    h, w = 101, 180
    y_head = 20.0
    heads = np.array([[40.0, y_head]], dtype=np.float64)
    seam = np.full(h, 80, dtype=np.int32)
    y0 = 20
    seam[y0 + 3] = 40
    assert not bool(heads_hit_mask([seam], heads, h, w)[0])
    assert bool(bodies_hit_mask([seam], heads, h, w)[0])


def test_body_does_not_extend_upward() -> None:
    h, w = 101, 180
    heads = np.array([[40.0, 30.0]], dtype=np.float64)
    seam = np.full(h, 80, dtype=np.int32)
    seam[20] = 40
    assert not bool(bodies_hit_mask([seam], heads, h, w)[0])


def test_body_hit_rate_at_least_head_hit_rate() -> None:
    h, w = 64, 128
    heads = np.array([[10.0, 20.0], [60.0, 24.0]], dtype=np.float64)
    s1 = np.full(h, 10, dtype=np.int32)
    mapped = map_seams_to_original([s1])
    assert body_hit_rate(mapped, heads, h, w) >= head_hit_rate(mapped, heads, h, w)
