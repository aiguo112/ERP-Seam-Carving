"""C1 set-scale gate: cos(phi) vs plain Sobel on a real ERP set.

Controlled like C3: same images, same cyclic DP, only the energy changes.
Headline is polar ROUTING share, not occupancy (occupancy is structural ~30%).

Dataset: CVRG-Pano val union test (n=120), not cherry-picked.
If mean routing does not rise and a majority of images do not rise, C1
is still a demo and there is no paper.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.cvrg_pano import list_panoramas  # noqa: E402
from data.paths import cvrg_root, require_path  # noqa: E402
from energy.gradient import gradient_energy  # noqa: E402
from energy.spherical import (  # noqa: E402
    mean_polar_step,
    polar_pixel_fraction,
    polar_routing_share,
    spherical_gradient_energy,
)
from io_erp import load_erp, save_erp  # noqa: E402
from seam.carve import reduce_width  # noqa: E402
from viz import overlay_seam  # noqa: E402

POLAR_BAND = 0.15
CVRG_ROOT = cvrg_root()


def resize_max_width(img: np.ndarray, max_width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max_width is None or w <= max_width:
        return img
    scale = max_width / w
    return cv2.resize(
        img,
        (int(max_width), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def carve_seams(image: np.ndarray, energy_fn, n_seams: int) -> tuple[list[np.ndarray], list[int], np.ndarray]:
    seams: list[np.ndarray] = []
    widths: list[int] = []
    first: np.ndarray | None = None

    def on_seam(frame, seam, energy, i) -> None:
        nonlocal first
        seams.append(np.asarray(seam, dtype=np.int32).copy())
        widths.append(int(frame.shape[1]))
        if first is None:
            first = overlay_seam(frame, seam, thickness=1)

    reduce_width(
        image, energy_fn, image.shape[1] - n_seams, cyclic=True, on_seam=on_seam
    )
    assert first is not None
    return seams, widths, first


def stats_for(seams: list[np.ndarray], height: int, widths: list[int]) -> dict[str, float]:
    return {
        "occupancy": polar_pixel_fraction(seams, height, POLAR_BAND),
        "routing": polar_routing_share(
            seams, height, widths, POLAR_BAND, cyclic=True
        ),
        "polar_step": mean_polar_step(
            seams, height, widths, POLAR_BAND, cyclic=True
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Set-scale spherical vs naive ERP.")
    parser.add_argument("--cvrg-root", type=Path, default=CVRG_ROOT)
    parser.add_argument("--split", default="val+test")
    parser.add_argument("-n", "--seams", type=int, default=24)
    parser.add_argument("--max-width", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("output") / "spherical_set")
    args = parser.parse_args(argv)
    args.cvrg_root = require_path(args.cvrg_root, flag="--cvrg-root", env="CVRG_PANO_ROOT")

    if args.split == "val+test":
        panos = list_panoramas(args.cvrg_root, "val") + list_panoramas(
            args.cvrg_root, "test"
        )
    else:
        panos = list_panoramas(args.cvrg_root, args.split)
    if args.limit:
        panos = panos[: args.limit]
    if not panos:
        raise SystemExit(f"no panos under {args.cvrg_root} split={args.split}")

    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"C1 set-scale: n={len(panos)}  split={args.split}  seams={args.seams}  "
        f"W<={args.max_width}  cyclic DP both  band={POLAR_BAND}"
    )
    print("occupancy must match; routing share is the claim")
    print()

    rows: list[dict] = []
    for k, pano in enumerate(panos):
        image = resize_max_width(load_erp(pano.rgb_path), args.max_width)
        h, w = image.shape[:2]
        n_seams = min(args.seams, w - 1)
        sp, wp, ov_p = carve_seams(image, gradient_energy, n_seams)
        ss, ws, ov_s = carve_seams(image, spherical_gradient_energy, n_seams)
        a = stats_for(sp, h, wp)
        b = stats_for(ss, h, ws)
        rec = {
            "stem": pano.stem,
            "split": pano.split,
            "H": h,
            "W": w,
            "n_seams": n_seams,
            "occ_plain": a["occupancy"],
            "occ_sph": b["occupancy"],
            "route_plain": a["routing"],
            "route_sph": b["routing"],
            "step_plain": a["polar_step"],
            "step_sph": b["polar_step"],
            "d_route": b["routing"] - a["routing"],
            "d_step": b["polar_step"] - a["polar_step"],
        }
        rows.append(rec)
        print(
            f"{k + 1:3d}/{len(panos)} {pano.split:4s} {pano.stem:<16}  "
            f"occ {a['occupancy']:.3f}/{b['occupancy']:.3f}  "
            f"route {a['routing']:.3f}->{b['routing']:.3f}  "
            f"d={rec['d_route']:+.3f}",
            flush=True,
        )

    d = np.array([r["d_route"] for r in rows], dtype=np.float64)
    occ_p = np.array([r["occ_plain"] for r in rows])
    occ_s = np.array([r["occ_sph"] for r in rows])
    rp = np.array([r["route_plain"] for r in rows])
    rs = np.array([r["route_sph"] for r in rows])
    n_up = int((d > 1e-6).sum())
    n_down = int((d < -1e-6).sum())
    n_tie = len(d) - n_up - n_down
    occ_gap = float(np.max(np.abs(occ_p - occ_s)))

    print()
    print(
        f"occupancy  plain={occ_p.mean():.4f}+/-{occ_p.std():.4f}  "
        f"sph={occ_s.mean():.4f}+/-{occ_s.std():.4f}  max|diff|={occ_gap:.4f}"
    )
    print(
        f"routing    plain={rp.mean():.4f}+/-{rp.std():.4f}  "
        f"sph={rs.mean():.4f}+/-{rs.std():.4f}"
    )
    print(
        f"paired d_route  mean={d.mean():+.4f}  std={d.std():.4f}  "
        f"median={np.median(d):+.4f}  up={n_up} down={n_down} tie={n_tie} / {len(d)}"
    )

    gate_occ = occ_gap < 0.02
    gate_mean = float(d.mean()) > 0.0
    gate_maj = n_up > len(d) / 2
    if gate_occ and gate_mean and gate_maj:
        print(
            "GATE PASS: occupancy is structural; spherical raises routing "
            "on a majority of images. C1 is a set-scale result."
        )
    else:
        print("GATE FAIL: C1 does not scale. Do not write a method paper on this.")
        if not gate_occ:
            print("  occupancy differed across methods (should be identical).")
        if not gate_mean:
            print("  mean routing share did not increase.")
        if not gate_maj:
            print("  spherical routing rose on a minority of images.")

    # keep two overlays: largest gain and a near-median
    order = np.argsort(d)
    picks = {int(order[-1]): "max_gain", int(order[len(order) // 2]): "median"}
    for idx, tag in picks.items():
        pano = panos[idx]
        image = resize_max_width(load_erp(pano.rgb_path), args.max_width)
        _, _, ov_p = carve_seams(image, gradient_energy, rows[idx]["n_seams"])
        _, _, ov_s = carve_seams(image, spherical_gradient_energy, rows[idx]["n_seams"])
        save_erp(args.out / f"{tag}_plain.png", ov_p)
        save_erp(args.out / f"{tag}_sph.png", ov_s)

    summary = {
        "n": len(rows),
        "split": args.split,
        "seams": args.seams,
        "max_width": args.max_width,
        "band": POLAR_BAND,
        "occ_plain_mean": float(occ_p.mean()),
        "occ_sph_mean": float(occ_s.mean()),
        "occ_max_abs_diff": occ_gap,
        "route_plain_mean": float(rp.mean()),
        "route_plain_std": float(rp.std()),
        "route_sph_mean": float(rs.mean()),
        "route_sph_std": float(rs.std()),
        "d_route_mean": float(d.mean()),
        "d_route_std": float(d.std()),
        "d_route_median": float(np.median(d)),
        "n_up": n_up,
        "n_down": n_down,
        "n_tie": n_tie,
        "gate_pass": bool(gate_occ and gate_mean and gate_maj),
    }
    (args.out / "metrics.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
