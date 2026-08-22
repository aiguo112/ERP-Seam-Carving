"""Cyclic DP: 8-connected predecessors wrap with (x ± 1) mod W."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from io_erp import save_erp  # noqa: E402
from seam.dp import (  # noqa: E402
    cyclic_predecessors,
    find_vertical_seam,
    seam_wraps,
)
from viz import overlay_seam, save_cyclic_seam_proof  # noqa: E402

OUT = ROOT / "output"
H, W = 40, 32


def wrapping_low_energy_path(height: int = H, width: int = W) -> np.ndarray:
    """8-connected diagonal that crosses the right edge onto column 0."""
    # Starts near the right: x = W-5, then +1 each row, so it hits W-1 then 0.
    return ((width - 5) + np.arange(height, dtype=np.int32)) % width


def synthetic_wrapping_case(
    height: int = H, width: int = W
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Image + energy whose *only* cheap path wraps around x=W-1 → x=0.

    Energy is 1 everywhere except 0 on that path, so cyclic DP has a unique
    optimum. The RGB image is a visualization of the same valley (dark path
    on a light field) — not Sobel energy, which would light up the path edges.
    """
    true_seam = wrapping_low_energy_path(height, width)
    energy = np.ones((height, width), dtype=np.float32)
    energy[np.arange(height), true_seam] = 0.0

    img = np.full((height, width, 3), 210, dtype=np.uint8)
    img[np.arange(height), true_seam] = (20, 40, 80)
    img[np.arange(height), (true_seam - 1) % width] = (40, 70, 110)
    img[np.arange(height), (true_seam + 1) % width] = (40, 70, 110)
    return img, energy, true_seam


def test_cyclic_predecessors_are_modulo_w() -> None:
    assert cyclic_predecessors(0, W) == (W - 1, 0, 1)
    assert cyclic_predecessors(W - 1, W) == (W - 2, W - 1, 0)
    assert cyclic_predecessors(5, W) == (4, 5, 6)


def test_cyclic_dp_follows_wrapping_valley() -> None:
    img, energy, true_seam = synthetic_wrapping_case()
    found = find_vertical_seam(energy, cyclic=True)

    assert found.shape == (H,)
    assert found.dtype == np.int32
    np.testing.assert_array_equal(found, true_seam)
    assert seam_wraps(found, W)

    step = np.abs(np.diff(found))
    assert np.all((step <= 1) | (step == W - 1))
    assert np.all((found >= 0) & (found < W))

    overlay = save_cyclic_seam_proof(
        img,
        found,
        OUT / "cyclic_wrap_proof.png",
        title="cyclic DP seam wraps across x=0 / x=W-1",
    )
    save_erp(OUT / "cyclic_wrap_overlay.png", overlay_seam(img, found, thickness=1))
    assert (OUT / "cyclic_wrap_proof.png").is_file()
    assert (OUT / "cyclic_wrap_overlay.png").is_file()
    # Red overlay must appear on both the right edge and the left edge.
    assert np.any(overlay[:, W - 1, 0] > 200)
    assert np.any(overlay[:, 0, 0] > 200)


def test_noncyclic_dp_cannot_cross_the_cut() -> None:
    _, energy, true_seam = synthetic_wrapping_case()
    found = find_vertical_seam(energy, cyclic=False)
    assert not seam_wraps(found, W)
    cost = float(energy[np.arange(H), found].sum())
    assert cost > 0.0
    assert not np.array_equal(found, true_seam)
