"""Stanford 2D-3D-S decode, void caps, indoor importance."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.stanford2d3d import (  # noqa: E402
    VOID_RGB,
    decode_stanford_semantic,
    encode_index_color,
    polar_black_void,
)
from energy.semantic import count_protected_hits, semantic_energy  # noqa: E402
from segmentation.stanford import (  # noqa: E402
    STANFORD_CLASS_NAMES,
    STANFORD_IMPORTANCE,
    STANFORD_NAME_TO_ID,
    stanford_protected_ids,
)


def test_encode_index_roundtrip() -> None:
    r, g, b = encode_index_color(42)
    idx = r * 256 * 256 + g * 256 + b
    assert idx == 42


def test_decode_maps_instance_class_and_void_color() -> None:
    labels = ["door_1_office_1_1", "wall_2_office_1_1"]
    h, w = 8, 10
    sem = np.zeros((h, w, 3), dtype=np.uint8)
    sem[:] = VOID_RGB
    dcol = encode_index_color(0)
    wcol = encode_index_color(1)
    sem[4, 3] = dcol
    sem[4, 7] = wcol
    class_map, void = decode_stanford_semantic(sem, labels)
    assert class_map[4, 3] == STANFORD_NAME_TO_ID["door"]
    assert class_map[4, 7] == STANFORD_NAME_TO_ID["wall"]
    assert void[0, 0]
    assert not void[4, 3]


def test_polar_black_void_only_in_caps() -> None:
    h, w = 20, 16
    img = np.full((h, w, 3), 80, dtype=np.uint8)
    img[:2] = 0
    img[h // 2, :] = 0  # dark equator — must NOT be void
    void = polar_black_void(img, band=0.15)
    assert void[:2].all()
    assert not void[h // 2].any()


def test_door_importance_higher_than_wall() -> None:
    assert STANFORD_IMPORTANCE["door"] > STANFORD_IMPORTANCE["wall"]
    assert STANFORD_IMPORTANCE["window"] > STANFORD_IMPORTANCE["ceiling"]
    assert STANFORD_IMPORTANCE["void"] == 0.0


def test_semantic_energy_zeros_void() -> None:
    h, w = 6, 8
    cmap = np.full((h, w), STANFORD_NAME_TO_ID["door"], dtype=np.int32)
    void = np.zeros((h, w), dtype=bool)
    void[0] = True
    e = semantic_energy(
        cmap,
        STANFORD_IMPORTANCE,
        class_names=STANFORD_CLASS_NAMES,
        void_mask=void,
    )
    assert float(e[1, 0]) == 1.0
    assert float(e[0, 0]) == 0.0


def test_hits_exclude_void() -> None:
    cmap = np.zeros((5, 6), dtype=np.int32)
    cmap[:, 2] = STANFORD_NAME_TO_ID["door"]
    void = np.zeros_like(cmap, dtype=bool)
    void[0:2, 2] = True
    seam = np.full(5, 2, dtype=np.int32)
    hits = count_protected_hits(seam, cmap, stanford_protected_ids(), void)
    assert hits["door"] == 3
    assert hits["window"] == 0


def test_list_panoramas_pairs_rgb_and_semantic(tmp_path: Path) -> None:
    import json

    import cv2

    from data.stanford2d3d import list_panoramas, load_stanford_sample

    area = tmp_path / "area_1" / "pano"
    rgb_dir = area / "rgb"
    sem_dir = area / "semantic"
    rgb_dir.mkdir(parents=True)
    sem_dir.mkdir(parents=True)
    labels = ["door_1_office_1_1"]
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "semantic_labels.json").write_text(json.dumps(labels), encoding="utf-8")

    h, w = 32, 64
    rgb = np.full((h, w, 3), 90, dtype=np.uint8)
    rgb[:4] = 0
    sem = np.zeros((h, w, 3), dtype=np.uint8)
    sem[:] = VOID_RGB
    dcol = encode_index_color(0)
    sem[10:20, 20:30] = dcol
    stem = "camera_x_office_1_frame_equirectangular_domain"
    cv2.imwrite(str(rgb_dir / f"{stem}__rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(sem_dir / f"{stem}__semantic.png"), cv2.cvtColor(sem, cv2.COLOR_RGB2BGR))

    panos = list_panoramas(tmp_path)
    assert len(panos) == 1
    sample = load_stanford_sample(panos[0], tmp_path / "assets" / "semantic_labels.json", max_width=32)
    assert sample.image.shape[1] == 32
    assert sample.void_mask[:2].any()
    assert STANFORD_NAME_TO_ID["door"] in sample.class_map


def test_semantic_reduces_door_hits_beyond_gradient() -> None:
    """Flat door column: (c) must hit fewer door pixels than (a)."""
    from energy.combine import combine_energies
    from energy.gradient import gradient_energy
    from energy.spherical import apply_spherical_weight
    from seam.carve import reduce_width, remove_vertical_seam

    h, w = 48, 80
    img = np.full((h, w, 3), 90, dtype=np.uint8)
    img += (np.arange(w) % 7).astype(np.uint8)[None, :, None]
    cmap = np.full((h, w), STANFORD_NAME_TO_ID["wall"], dtype=np.int32)
    cmap[:, 30:50] = STANFORD_NAME_TO_ID["door"]
    img[:, 30:50] = 40
    void = np.zeros((h, w), dtype=bool)
    void[:4] = True
    groups = stanford_protected_ids()
    n = 6

    def run(lg: float, lp: float, ls: float) -> int:
        holder = [cmap.copy()]
        vholder = [void.copy()]
        hits = 0

        def energy_fn(frame):
            g = gradient_energy(frame)
            p = apply_spherical_weight(g, g.shape[0])
            s = semantic_energy(
                holder[0],
                STANFORD_IMPORTANCE,
                class_names=STANFORD_CLASS_NAMES,
                void_mask=vholder[0],
            )
            return combine_energies(g, p, s, lg, lp, ls)

        def on_seam(frame, seam, energy, i):
            nonlocal hits
            hits += count_protected_hits(seam, holder[0], groups, vholder[0])["door"]
            holder[0] = remove_vertical_seam(holder[0], seam)
            vholder[0] = remove_vertical_seam(vholder[0].astype(np.uint8), seam).astype(bool)

        reduce_width(img, energy_fn, w - n, cyclic=True, on_seam=on_seam)
        return hits

    hits_a = run(1.0, 0.0, 0.0)
    hits_c = run(1.0, 1.0, 4.0)
    assert hits_c < hits_a
