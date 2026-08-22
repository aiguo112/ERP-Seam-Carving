"""Spherical cos(φ) weighting: polar pixels are cheap by design."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from energy.gradient import gradient_energy  # noqa: E402
from energy.spherical import (  # noqa: E402
    apply_spherical_weight,
    latitude_weight,
    mean_polar_step,
    polar_energy_content,
    polar_pixel_fraction,
    polar_row_mask,
    polar_routing_share,
    spherical_gradient_energy,
)
from seam.carve import reduce_width  # noqa: E402


def test_latitude_weight_matches_formula() -> None:
    h = 9
    w = latitude_weight(h)
    y = np.arange(h, dtype=np.float64)
    phi = (y / (h - 1) - 0.5) * np.pi
    np.testing.assert_allclose(w, np.cos(phi), atol=1e-6)
    assert abs(float(w[4]) - 1.0) < 1e-5
    assert float(w[0]) < 0.05
    assert float(w[-1]) < 0.05


def test_apply_spherical_weight_cheapens_poles() -> None:
    e = np.ones((11, 6), dtype=np.float32)
    s = apply_spherical_weight(e, 11)
    mid = s.shape[0] // 2
    assert float(s[mid].mean()) > float(s[0].mean())
    assert float(s[mid].mean()) > float(s[-1].mean())


def test_polar_energy_content_ratio() -> None:
    e = np.ones((20, 8), dtype=np.float32)
    e[:3] = 4.0
    e[-3:] = 4.0
    stats = polar_energy_content(e, band=0.15)
    assert abs(stats["polar_grad"] - 4.0) < 1e-5
    assert abs(stats["eq_grad"] - 1.0) < 1e-5
    assert abs(stats["polar_ratio"] - 4.0) < 1e-5
    valid = np.ones_like(e, dtype=bool)
    valid[:3] = False
    dropped = polar_energy_content(e, band=0.15, polar_valid=valid)
    assert dropped["polar_grad"] == 4.0  # only bottom cap remains, still 4


def test_polar_pixel_fraction_matches_band() -> None:
    h = 20
    seams = [np.zeros(h, dtype=np.int32)]
    n = int(np.floor(0.15 * h))
    assert polar_pixel_fraction(seams, h, 0.15) == (2 * n) / h
    assert polar_row_mask(h, 0.15).sum() == 2 * n


def test_spherical_increases_polar_routing() -> None:
    """On polar-noisy ERP, cos(φ) should shift |Δx| into the polar bands."""
    rng = np.random.default_rng(1)
    h, w = 80, 120
    img = np.full((h, w, 3), 80, dtype=np.uint8)
    img[h // 3 : 2 * h // 3, w // 2 - 4 : w // 2 + 4] = 250
    n_cap = int(0.15 * h)
    img[:n_cap] = rng.integers(0, 255, (n_cap, w, 3), dtype=np.uint8)
    img[h - n_cap :] = rng.integers(0, 255, (n_cap, w, 3), dtype=np.uint8)

    n = 8
    seams_plain: list[np.ndarray] = []
    seams_sph: list[np.ndarray] = []
    widths_plain: list[int] = []
    widths_sph: list[int] = []

    def collect(store: list[np.ndarray], widths: list[int]):
        def on_seam(frame, seam, energy, i) -> None:
            store.append(np.asarray(seam, dtype=np.int32).copy())
            widths.append(int(frame.shape[1]))

        return on_seam

    reduce_width(
        img, gradient_energy, w - n, cyclic=True, on_seam=collect(seams_plain, widths_plain)
    )
    reduce_width(
        img,
        spherical_gradient_energy,
        w - n,
        cyclic=True,
        on_seam=collect(seams_sph, widths_sph),
    )

    pix_p = polar_pixel_fraction(seams_plain, h)
    pix_s = polar_pixel_fraction(seams_sph, h)
    assert abs(pix_p - pix_s) < 1e-9

    route_p = polar_routing_share(seams_plain, h, widths_plain, cyclic=True)
    route_s = polar_routing_share(seams_sph, h, widths_sph, cyclic=True)
    step_p = mean_polar_step(seams_plain, h, widths_plain, cyclic=True)
    step_s = mean_polar_step(seams_sph, h, widths_sph, cyclic=True)
    assert step_s > step_p or route_s > route_p
