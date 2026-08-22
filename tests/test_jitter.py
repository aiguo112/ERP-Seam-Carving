"""Seam jitter unit tests (no video files required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.video_erp import contiguous_pairs, frame_is_near_black  # noqa: E402
from temporal.jitter import (  # noqa: E402
    compute_jitter_stats,
    cyclic_abs_diff,
    equatorial_horizontal_pan,
    motion_score,
    residual_seam_jitter,
    seam_jitter,
    warp_seam,
)


def test_cyclic_wrap_is_one_pixel() -> None:
    a = np.array([0, 0], dtype=np.int32)
    b = np.array([9, 9], dtype=np.int32)
    d = cyclic_abs_diff(a, b, 10)
    assert int(d[0]) == 1


def test_static_seams_zero_jitter() -> None:
    s = np.arange(8, dtype=np.int32)
    assert seam_jitter([s, s.copy(), s.copy()], 16) == 0.0


def test_shifted_seams_match_offset() -> None:
    s0 = np.full(10, 20, dtype=np.int32)
    s1 = np.full(10, 25, dtype=np.int32)
    assert abs(seam_jitter([s0, s1], 80) - 5.0) < 1e-6


def test_static_frames_low_motion_score() -> None:
    img = np.full((32, 64, 3), 80, dtype=np.uint8)
    img[:, 20:24] = 200
    frames = [img.copy() for _ in range(6)]
    assert motion_score(frames) < 0.5


def test_rolling_frames_high_motion_score() -> None:
    img = np.zeros((32, 64, 3), dtype=np.uint8)
    img[:, 10:18] = 220
    frames = [np.roll(img, 8 * t, axis=1) for t in range(6)]
    assert motion_score(frames) > motion_score([img] * 6)


def test_independent_carve_jitters_when_content_rolls() -> None:
    from energy.spherical import spherical_gradient_energy
    from seam.carve import map_seams_to_original, reduce_width
    from temporal.jitter import seam_jitter as jitter

    h, w = 40, 80
    base = np.full((h, w, 3), 40, dtype=np.uint8)
    base += (np.arange(w) % 11).astype(np.uint8)[None, :, None]
    base[:, 30:38] = 200

    def first_seam(img):
        held: list[np.ndarray] = []

        def on_seam(frame, seam, energy, i):
            if i == 0:
                held.append(np.asarray(seam, dtype=np.int32).copy())

        reduce_width(
            img, spherical_gradient_energy, w - 1, cyclic=True, on_seam=on_seam
        )
        return map_seams_to_original(held)[0]

    static = [first_seam(base) for _ in range(4)]
    rolled = [first_seam(np.roll(base, 6 * t, axis=1)) for t in range(4)]
    assert jitter(rolled, w) > jitter(static, w)


def test_near_black_empty_vs_letterbox() -> None:
    empty = np.zeros((32, 64, 3), dtype=np.uint8)
    assert frame_is_near_black(empty)
    faded = np.full((32, 64, 3), 4, dtype=np.uint8)
    assert frame_is_near_black(faded)
    # dark poles but visible equator (Reloaded-like letterbox)
    letterbox = np.zeros((40, 64, 3), dtype=np.uint8)
    letterbox[10:30] = 80
    assert not frame_is_near_black(letterbox)


def test_contiguous_pairs_skip_dropped_slot() -> None:
    # linspace slots 0,1,2,3,4; slot 2 dropped -> kept slots 0,1,3,4
    slots = np.array([0, 1, 3, 4], dtype=np.int32)
    pairs = contiguous_pairs(slots)
    assert pairs == [(0, 1), (2, 3)]
    seams = [
        np.full(4, 0, dtype=np.int32),
        np.full(4, 1, dtype=np.int32),
        np.full(4, 50, dtype=np.int32),  # would be a gap jump if counted
        np.full(4, 51, dtype=np.int32),
    ]
    j = seam_jitter(seams, 80, pairs)
    assert abs(j - 1.0) < 1e-6
    j_naive = seam_jitter(seams, 80)
    assert j_naive > 10.0


def test_phasecorr_recovers_horizontal_roll() -> None:
    rng = np.random.default_rng(0)
    img = (40 + 80 * rng.random((48, 96, 3))).astype(np.uint8)
    img[:, 20:28] = 220
    rolled = np.roll(img, 17, axis=1)
    pan = equatorial_horizontal_pan(img, rolled)
    assert abs(pan - 17.0) < 0.75
    wrapped = np.roll(img, -5, axis=1)
    pan_w = equatorial_horizontal_pan(img, wrapped)
    assert abs(pan_w - (-5.0)) < 0.75


def test_residual_zero_when_seam_tracks_pan() -> None:
    w = 80
    s0 = np.full(12, 20, dtype=np.int32)
    pan = 17.0
    s1 = np.asarray(warp_seam(s0, pan, w), dtype=np.int32)
    pairs = [(0, 1)]
    j = seam_jitter([s0, s1], w, pairs)
    jres = residual_seam_jitter([s0, s1], [pan], w, pairs)
    assert abs(j - 17.0) < 1e-6
    assert jres < 0.5


def test_residual_survives_when_seam_ignores_pan() -> None:
    w = 80
    s0 = np.full(12, 20, dtype=np.int32)
    s1 = np.full(12, 20, dtype=np.int32)  # seam stuck; content panned
    pairs = [(0, 1)]
    j = seam_jitter([s0, s1], w, pairs)
    jres = residual_seam_jitter([s0, s1], [17.0], w, pairs)
    assert j < 1e-6
    assert abs(jres - 17.0) < 1e-6


def test_stats_residual_on_rolled_pattern() -> None:
    rng = np.random.default_rng(1)
    base = (50 + 60 * rng.random((40, 80, 3))).astype(np.uint8)
    base[:, 30:38] = 210
    frames = [np.roll(base, 9 * t, axis=1) for t in range(4)]
    seams = [np.full(40, (10 + 9 * t) % 80, dtype=np.int32) for t in range(4)]
    pairs = [(0, 1), (1, 2), (2, 3)]
    stats = compute_jitter_stats(frames, seams, 80, pairs)
    assert stats.n_pairs == 3
    assert abs(stats.J - 9.0) < 0.5
    assert abs(stats.pan_mean_abs - 9.0) < 1.0
    assert stats.J_residual < 1.5
