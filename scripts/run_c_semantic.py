"""Stage C ablation on real ERP: isolate the semantic term.

Four configs, same image, same seam count, cyclic DP::

    (a) gradient only
    (b) gradient + spherical
    (c) gradient + spherical + semantic
    (d) semantic only   (sanity: where do seams go with no appearance term?)

Stanford 2D-3D-S uses GT 13-class masks (void excluded). Outdoor 360 uses
the segmenter's own taxonomy (SegFormer -> ADE20K names from id2label;
DeepLabV3 -> VOC). Hugging Face or Stanford downloads may need a VPN.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.cvrg_pano import list_panoramas as list_cvrg_panoramas  # noqa: E402
from data.cvrg_pano import load_cvrg_sample  # noqa: E402
from data.paths import cvrg_root, require_path  # noqa: E402
from data.stanford2d3d import (  # noqa: E402
    find_labels_json,
    list_panoramas as list_stanford_panoramas,
    load_stanford_sample,
    polar_black_void,
)
from energy.combine import combine_energies  # noqa: E402
from energy.gradient import gradient_energy  # noqa: E402
from energy.semantic import group_ids_from_id2label, semantic_energy  # noqa: E402
from energy.spherical import apply_spherical_weight, polar_routing_share  # noqa: E402
from io_erp import load_erp, save_erp  # noqa: E402
from metrics import accumulate_hits, format_table, hit_percentages, semantic_delta_line  # noqa: E402
from seam.carve import map_seams_to_original, reduce_width, remove_vertical_seam  # noqa: E402
from segmentation.cvrg import CVRG_CLASS_NAMES  # noqa: E402
from segmentation.stanford import STANFORD_CLASS_NAMES  # noqa: E402
from segmentation.taxonomy import (  # noqa: E402
    id2label_from_names,
    importance_map,
    protected_group_names,
    taxonomy_for,
)
from segmentation.voc import VOC_CLASS_NAMES, VOC_NAME_TO_ID  # noqa: E402
from viz import overlay_protected_classes, overlay_seams, save_quad  # noqa: E402


def make_semantic_demo(height: int = 160, width: int = 320) -> tuple[np.ndarray, np.ndarray]:
    """Painted VOC blobs (not an honest real-ERP table)."""
    rng = np.random.default_rng(0)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    rgb[..., 2] = 0.35 + 0.40 * (1.0 - y)
    rgb[..., 1] = 0.22 + 0.18 * y
    rgb[..., 0] = 0.16 + 0.10 * y
    rgb += 0.06 * rng.standard_normal(rgb.shape).astype(np.float32)
    class_map = np.zeros((height, width), dtype=np.int32)
    yy = np.arange(height)[:, None]
    xx = np.arange(width)[None, :]

    def _blob(cy, cx, ry, rx, rgb_val, class_name: str) -> None:
        mask = ((yy - cy * (height - 1)) / (ry * height)) ** 2 + (
            (xx - cx * (width - 1)) / (rx * width)
        ) ** 2 <= 1.0
        rgb[mask] = np.array(rgb_val, dtype=np.float32)
        class_map[mask] = VOC_NAME_TO_ID[class_name]

    _blob(0.52, 0.28, 0.18, 0.08, (0.72, 0.48, 0.38), "person")
    _blob(0.58, 0.62, 0.10, 0.14, (0.22, 0.24, 0.28), "car")
    _blob(0.55, 0.85, 0.09, 0.07, (0.45, 0.28, 0.12), "dog")
    rgb[:, 4:7, :] = 0.12
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8), class_map


def energy_fn_for(
    class_map_ref: list[np.ndarray],
    void_ref: list[np.ndarray | None],
    lambda_g: float,
    lambda_p: float,
    lambda_s: float,
    importance,
    id2label: dict[int, str],
) -> Callable[[np.ndarray], NDArray[np.float32]]:
    def energy_fn(img: np.ndarray) -> NDArray[np.float32]:
        g = gradient_energy(img)
        p = apply_spherical_weight(g, g.shape[0])
        s = semantic_energy(
            class_map_ref[0],
            importance,
            id2label=id2label,
            void_mask=void_ref[0],
        )
        return combine_energies(g, p, s, lambda_g, lambda_p, lambda_s)

    return energy_fn


def run_config(
    image: np.ndarray,
    class_map: np.ndarray,
    void_mask: np.ndarray | None,
    n_seams: int,
    lambda_g: float,
    lambda_p: float,
    lambda_s: float,
    importance,
    id2label: dict[int, str],
    groups: dict[str, frozenset[int]],
) -> dict:
    cmap_ref = [np.array(class_map, copy=True)]
    void_ref: list[np.ndarray | None] = [
        None if void_mask is None else np.array(void_mask, copy=True)
    ]
    seams: list[np.ndarray] = []
    widths: list[int] = []
    hits = {g: 0 for g in groups}
    n_valid = 0

    def on_seam(frame: np.ndarray, seam: np.ndarray, energy: np.ndarray, i: int) -> None:
        nonlocal n_valid
        n_valid += accumulate_hits(seam, cmap_ref[0], groups, void_ref[0], hits)
        seams.append(np.asarray(seam, dtype=np.int32).copy())
        widths.append(int(frame.shape[1]))
        cmap_ref[0] = remove_vertical_seam(cmap_ref[0], seam)
        if void_ref[0] is not None:
            void_ref[0] = remove_vertical_seam(void_ref[0].astype(np.uint8), seam).astype(bool)

    carved = reduce_width(
        image,
        energy_fn_for(cmap_ref, void_ref, lambda_g, lambda_p, lambda_s, importance, id2label),
        image.shape[1] - n_seams,
        cyclic=True,
        on_seam=on_seam,
    )
    polar = polar_routing_share(seams, image.shape[0], widths, cyclic=True)
    pct = hit_percentages(hits, n_valid)
    return {
        "carved": carved,
        "seams": seams,
        "hits": hits,
        "valid_px": n_valid,
        "polar_route": polar,
        "pct": pct,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Honest Stage C ablation: does semantic beat gradient-only on real ERP?"
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=None,
        help="Outdoor ERP path (implies --dataset outdoor if dataset is demo).",
    )
    parser.add_argument(
        "--dataset",
        choices=("stanford", "outdoor", "demo", "cvrg"),
        default="demo",
        help="cvrg = CVRG-Pano GT; stanford = 2D-3D-S GT; outdoor = CNN; demo = blobs.",
    )
    parser.add_argument("--stanford-root", type=Path, default=None)
    parser.add_argument(
        "--cvrg-root",
        type=Path,
        default=cvrg_root(),
    )
    parser.add_argument("--split", default="val", help="CVRG split: train/val/test/all.")
    parser.add_argument("--index", type=int, default=0, help="Which pano (sorted).")
    parser.add_argument(
        "--segmenter",
        choices=("deeplabv3", "segformer"),
        default="segformer",
        help="Outdoor CNN. SegFormer/ADE20K is the outdoor default.",
    )
    parser.add_argument("-n", "--seams", type=int, default=24)
    parser.add_argument("--max-width", type=int, default=1024)
    parser.add_argument("--lambda-s", type=float, default=4.0, help="λ_s used in config (c).")
    parser.add_argument("--out", type=Path, default=Path("output") / "semantic")
    args = parser.parse_args(argv)
    if args.image is not None and args.dataset == "demo":
        args.dataset = "outdoor"

    configs = (
        ("(a) gradient", 1.0, 0.0, 0.0),
        ("(b) +spherical", 1.0, 1.0, 0.0),
        ("(c) +semantic", 1.0, 1.0, float(args.lambda_s)),
        ("(d) semantic-only", 0.0, 0.0, 1.0),
    )

    if args.dataset == "stanford":
        image, class_map, void_mask, importance, id2label, groups, label_src = (
            _load_stanford(args)
        )
    elif args.dataset == "cvrg":
        image, class_map, void_mask, importance, id2label, groups, label_src = (
            _load_cvrg(args)
        )
    elif args.dataset == "outdoor":
        image, class_map, void_mask, importance, id2label, groups, label_src = (
            _load_outdoor(args)
        )
    else:
        image, class_map = make_semantic_demo()
        void_mask = np.zeros(class_map.shape, dtype=bool)
        tax = taxonomy_for(dataset="demo")
        importance = importance_map(tax)
        id2label = id2label_from_names(VOC_CLASS_NAMES)
        groups = group_ids_from_id2label(id2label, protected_group_names(tax))
        label_src = "DEMO painted VOC blobs - not a real-ERP result"

    w = image.shape[1]
    if args.seams < 1 or args.seams >= w:
        raise SystemExit(f"--seams must be in [1, {w - 1}], got {args.seams}")

    print(f"image {image.shape}  seams={args.seams}  cyclic=True")
    print(f"labels: {label_src}")
    print("void pixels:", int(void_mask.sum()), f"/ {void_mask.size}")

    results = []
    overlays = []
    tinted = overlay_protected_classes(image, class_map, groups=groups)
    for name, lg, lp, ls in configs:
        print(f"running {name}  lambdas g={lg} p={lp} s={ls} ...")
        out = run_config(
            image, class_map, void_mask, args.seams, lg, lp, ls,
            importance, id2label, groups,
        )
        out["config"] = name
        results.append(out)
        mapped = map_seams_to_original(out["seams"])
        overlays.append(overlay_seams(tinted, mapped, color=(255, 32, 32), thickness=0))

    group_names = list(groups.keys())
    print()
    print(format_table(results, group_names))
    print()
    print(semantic_delta_line(results[0]["pct"], results[2]["pct"], group_names, vs="(a) gradient-only"))
    print(
        semantic_delta_line(
            results[1]["pct"],
            results[2]["pct"],
            group_names,
            vs="(b) gradient+spherical (isolates semantic)",
        )
    )
    print(
        "polar_route is Stage B routing share (|dx| in top/bottom 15% rows), "
        "not the 30% occupancy."
    )

    args.out.mkdir(parents=True, exist_ok=True)
    save_erp(args.out / "original.png", image)
    save_erp(args.out / "class_tint.png", tinted)
    save_erp(args.out / "overlay_a_gradient.png", overlays[0])
    save_erp(args.out / "overlay_b_spherical.png", overlays[1])
    save_erp(args.out / "overlay_c_semantic.png", overlays[2])
    save_erp(args.out / "overlay_d_semantic_only.png", overlays[3])
    save_quad(
        overlays,
        [r["config"] for r in results],
        args.out / "ablation_quad.png",
    )
    save_quad(
        [tinted, overlays[0], overlays[1], overlays[2]],
        [
            "original + protected tint",
            "(a) gradient only",
            "(b) gradient + spherical",
            "(c) + semantic",
        ],
        args.out / "comparison_abc.png",
    )
    print(f"wrote {args.out.resolve()}")
    return 0


def _load_stanford(args) -> tuple:
    if args.stanford_root is None:
        raise SystemExit(
            "Stanford 2D-3D-S is not bundled. Pass --stanford-root PATH "
            "(local area_*/pano/{rgb,semantic} plus assets/semantic_labels.json). "
            "The dataset download is large; a VPN may be required off-campus."
        )
    panos = list_stanford_panoramas(args.stanford_root)
    if not panos:
        raise SystemExit(f"no equirectangular pairs under {args.stanford_root}")
    if args.index < 0 or args.index >= len(panos):
        raise SystemExit(f"--index must be in [0, {len(panos) - 1}], found {len(panos)} panos")
    labels = find_labels_json(args.stanford_root)
    sample = load_stanford_sample(panos[args.index], labels, max_width=args.max_width)
    print(f"loaded {sample.rgb_path}")
    tax = taxonomy_for(dataset="stanford")
    id2label = id2label_from_names(STANFORD_CLASS_NAMES)
    return (
        sample.image,
        sample.class_map,
        sample.void_mask,
        importance_map(tax),
        id2label,
        group_ids_from_id2label(id2label, protected_group_names(tax)),
        f"Stanford2D3D GT ({panos[args.index].area}/{panos[args.index].stem})",
    )


def _load_cvrg(args) -> tuple:
    args.cvrg_root = require_path(args.cvrg_root, flag="--cvrg-root", env="CVRG_PANO_ROOT")
    panos = list_cvrg_panoramas(args.cvrg_root, split=args.split)
    if not panos:
        raise SystemExit(f"no CVRG pairs under {args.cvrg_root} split={args.split}")
    if args.index < 0 or args.index >= len(panos):
        raise SystemExit(f"--index must be in [0, {len(panos) - 1}], found {len(panos)}")
    sample = load_cvrg_sample(panos[args.index], max_width=args.max_width)
    print(f"loaded {sample.rgb_path}")
    tax = taxonomy_for(dataset="cvrg")
    id2label = id2label_from_names(CVRG_CLASS_NAMES)
    return (
        sample.image,
        sample.class_map,
        sample.void_mask,
        importance_map(tax),
        id2label,
        group_ids_from_id2label(id2label, protected_group_names(tax)),
        f"CVRG-Pano GT ({panos[args.index].split}/{panos[args.index].stem})",
    )


def _load_outdoor(args) -> tuple:
    if args.image is None:
        raise SystemExit("--dataset outdoor requires an image path")
    image = load_erp(args.image)
    if args.max_width and image.shape[1] > args.max_width:
        import cv2

        scale = args.max_width / image.shape[1]
        nw, nh = args.max_width, max(1, int(round(image.shape[0] * scale)))
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    from segmentation.factory import get_segmenter

    tax = taxonomy_for(dataset="outdoor", segmenter=args.segmenter)
    seg = get_segmenter(args.segmenter)
    class_map = seg.predict(image)
    id2label = dict(seg.id2label)
    void_mask = polar_black_void(image)
    src = (
        f"{type(seg).__name__} {tax} predictions "
        "(importance matched on id2label names)"
    )
    return (
        image,
        class_map,
        void_mask,
        importance_map(tax),
        id2label,
        group_ids_from_id2label(id2label, protected_group_names(tax)),
        src,
    )


if __name__ == "__main__":
    raise SystemExit(main())
