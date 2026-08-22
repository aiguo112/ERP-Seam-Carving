"""Semantic energy lookup and protected-class hit counts (no CNN required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from energy.semantic import count_protected_hits, semantic_energy  # noqa: E402
from segmentation.voc import VOC_NAME_TO_ID  # noqa: E402
from seam.carve import map_seams_to_original, reduce_width, remove_vertical_seam  # noqa: E402


def test_semantic_energy_uses_importance_table() -> None:
    h, w = 8, 10
    cmap = np.zeros((h, w), dtype=np.int32)
    cmap[:, 3] = VOC_NAME_TO_ID["person"]
    cmap[:, 5] = VOC_NAME_TO_ID["car"]
    e = semantic_energy(cmap)
    assert e.shape == (h, w) and e.dtype == np.float32
    assert float(e[0, 3]) == 1.0
    assert abs(float(e[0, 5]) - 0.9) < 1e-6
    assert abs(float(e[0, 0]) - 0.05) < 1e-6


def test_custom_importance_by_id_and_name() -> None:
    cmap = np.zeros((4, 4), dtype=np.int32)
    cmap[1, 1] = VOC_NAME_TO_ID["dog"]
    e = semantic_energy(cmap, {"dog": 0.5, 0: 0.0})
    assert float(e[1, 1]) == 0.5
    assert float(e[0, 0]) == 0.0


def test_protected_hit_counts() -> None:
    cmap = np.zeros((6, 8), dtype=np.int32)
    cmap[2, 4] = VOC_NAME_TO_ID["person"]
    cmap[3, 4] = VOC_NAME_TO_ID["bus"]
    cmap[4, 4] = VOC_NAME_TO_ID["cat"]
    seam = np.full(6, 4, dtype=np.int32)
    hits = count_protected_hits(seam, cmap)
    assert hits == {"person": 1, "vehicle": 1, "animal": 1}


def test_map_seams_to_original() -> None:
    s0 = np.array([2, 2, 2], dtype=np.int32)
    s1 = np.array([2, 2, 2], dtype=np.int32)
    orig = map_seams_to_original([s0, s1])
    np.testing.assert_array_equal(orig[0], [2, 2, 2])
    np.testing.assert_array_equal(orig[1], [3, 3, 3])


def test_semantic_carving_hits_fewer_people() -> None:
    """Flat person blob is cheap for Sobel but expensive for semantic energy."""
    from energy.gradient import gradient_energy
    from energy.semantic import semantic_energy as sem_e
    from energy.spherical import apply_spherical_weight
    from energy.combine import combine_energies as comb

    h, w = 48, 80
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    img += (np.arange(w) % 7).astype(np.uint8)[None, :, None]
    cmap = np.zeros((h, w), dtype=np.int32)
    cmap[:, 30:50] = VOC_NAME_TO_ID["person"]
    img[:, 30:50] = 40

    n = 6

    def run(lambda_s: float) -> int:
        holder = [cmap.copy()]
        hits = 0

        def energy_fn(frame):
            g = gradient_energy(frame)
            p = apply_spherical_weight(g, g.shape[0])
            s = sem_e(holder[0])
            return comb(g, p, s, 1.0, 0.5, lambda_s)

        def on_seam(frame, seam, energy, i):
            nonlocal hits
            hits += count_protected_hits(seam, holder[0])["person"]
            holder[0] = remove_vertical_seam(holder[0], seam)

        reduce_width(img, energy_fn, w - n, cyclic=True, on_seam=on_seam)
        return hits

    assert run(4.0) < run(0.0)
