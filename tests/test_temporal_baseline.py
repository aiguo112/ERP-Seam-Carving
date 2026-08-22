"""Non-learned temporal baseline: no sequence model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from temporal.damage import content_damage, equatorial_path_energy, path_energy  # noqa: E402
from temporal.jitter import equatorial_horizontal_pan, photometric_warp_scores  # noqa: E402
from temporal.propagate import (  # noqa: E402
    compensated_penalty,
    cyclic_distance_map,
    find_temporal_seam,
    seam_pred,
)


def test_cyclic_distance_wraps() -> None:
    center = np.array([0.0, 9.0])
    d = cyclic_distance_map(center, 10)
    assert d[0, 0] == 0
    assert d[0, 9] == 1
    assert d[1, 0] == 1


def test_zero_lambda_leaves_energy() -> None:
    e = np.arange(12, dtype=np.float32).reshape(3, 4)
    pred = np.array([1.0, 1.0, 1.0])
    out = compensated_penalty(e, pred, 0.0)
    np.testing.assert_array_equal(out, e)


def test_penalty_pulls_seam_to_pred_on_flat_energy() -> None:
    h, w = 16, 32
    energy = np.ones((h, w), dtype=np.float32)
    pred = np.full(h, 20.0)
    found = find_temporal_seam(energy, s_pred=pred, lambda_t=1.0)
    np.testing.assert_array_equal(found, np.full(h, 20, dtype=np.int32))


def test_band_forbids_far_columns() -> None:
    h, w = 12, 40
    energy = np.ones((h, w), dtype=np.float32)
    energy[:, 2] = 0.0  # cheap valley far from pred
    pred = np.full(h, 30.0)
    found = find_temporal_seam(energy, s_pred=pred, lambda_t=0.0, band_radius=3)
    assert int(np.max(np.abs(found - 30))) <= 3
    assert 2 not in set(int(x) for x in found)


def test_seam_pred_is_cyclic_shift() -> None:
    s = np.array([0, 1, 2], dtype=np.int32)
    p = seam_pred(s, 3.0, 10)
    np.testing.assert_allclose(p, np.array([3.0, 4.0, 5.0]))
    p2 = seam_pred(s, -1.0, 10)
    np.testing.assert_allclose(p2, np.array([9.0, 0.0, 1.0]))


def test_damage_ratio_one_when_seams_match() -> None:
    e = np.random.default_rng(0).random((20, 30)).astype(np.float32)
    s = np.full(20, 4, dtype=np.int32)
    d = content_damage([e], [s], [s], [True])
    assert abs(d.damage_ratio - 1.0) < 1e-9
    assert d.n_frames == 1


def test_damage_ratio_rises_when_seam_hits_hot_column() -> None:
    e = np.ones((20, 16), dtype=np.float32) * 0.1
    e[:, 8] = 5.0
    cheap = np.full(20, 1, dtype=np.int32)
    hot = np.full(20, 8, dtype=np.int32)
    d = content_damage([e], [cheap], [hot], [True])
    assert d.damage_ratio > 10.0
    assert equatorial_path_energy(e, hot) > equatorial_path_energy(e, cheap)


def test_gap_frames_excluded_from_damage() -> None:
    e = np.ones((8, 8), dtype=np.float32)
    s = np.zeros(8, dtype=np.int32)
    d = content_damage([e], [s], [s], [False])
    assert d.n_frames == 0
    assert d.damage_ratio == 1.0


def test_photometric_warp_helps_on_true_roll() -> None:
    rng = np.random.default_rng(2)
    img = (40 + 80 * rng.random((48, 96, 3))).astype(np.uint8)
    img[:, 20:28] = 220
    rolled = np.roll(img, 11, axis=1)
    raw, wrp = photometric_warp_scores(img, rolled, 11.0)
    assert wrp < raw * 0.5
    pan = equatorial_horizontal_pan(img, rolled)
    assert abs(pan - 11.0) < 0.75
    raw2, wrp2 = photometric_warp_scores(img, rolled, pan)
    assert wrp2 < raw2


def test_wrong_pan_does_not_help() -> None:
    rng = np.random.default_rng(3)
    img = (40 + 80 * rng.random((48, 96, 3))).astype(np.uint8)
    img[:, 20:28] = 220
    rolled = np.roll(img, 11, axis=1)
    raw, wrp = photometric_warp_scores(img, rolled, -11.0)
    assert wrp >= raw * 0.9


def test_path_energy_uses_named_column() -> None:
    e = np.zeros((4, 5), dtype=np.float32)
    e[:, 3] = 2.0
    s = np.full(4, 3, dtype=np.int32)
    assert abs(path_energy(e, s) - 2.0) < 1e-6
