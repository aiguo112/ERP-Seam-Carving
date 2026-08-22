"""CVRG-Pano GT batch: screen stress vs easy, then four-config (c) vs (b).

Hits are GT Person/Vehicle, void excluded. Polar metric is routing share.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data.cvrg_pano import list_panoramas, load_cvrg_sample  # noqa: E402
from data.paths import cvrg_root, require_path  # noqa: E402
from energy.gradient import gradient_energy  # noqa: E402
from energy.semantic import group_ids_from_id2label  # noqa: E402
from segmentation.cvrg import CVRG_CLASS_NAMES, CVRG_NAME_TO_ID  # noqa: E402
from segmentation.taxonomy import (  # noqa: E402
    id2label_from_names,
    importance_map,
    protected_group_names,
    taxonomy_for,
)
from io_erp import save_erp  # noqa: E402
from viz import overlay_protected_classes, overlay_seams  # noqa: E402

import run_c_semantic as run_c  # noqa: E402


PERSON_ID = CVRG_NAME_TO_ID["person"]
VEHICLE_IDS = tuple(
    CVRG_NAME_TO_ID[n] for n in ("car", "truck", "bus", "motorcycle", "bicycle")
)
EQ_LO, EQ_HI = 0.15, 0.85


def _eq_slice(h: int) -> slice:
    return slice(int(EQ_LO * h), max(int(EQ_LO * h) + 1, int(EQ_HI * h)))


def screen_one(sample) -> dict:
    img = sample.image
    cmap = sample.class_map
    h, w = cmap.shape
    eq = np.zeros(h, dtype=bool)
    eq[_eq_slice(h)] = True
    eq2 = eq[:, None]
    person = cmap == PERSON_ID
    vehicle = np.isin(cmap, VEHICLE_IDS)
    n_eq = int(eq.sum() * w)
    person_eq = float((person & eq2).sum()) / max(n_eq, 1)
    vehicle_eq = float((vehicle & eq2).sum()) / max(n_eq, 1)
    g = gradient_energy(img)
    p_pix = person & eq2
    bg = (~person) & (~vehicle) & eq2
    g_person = float(g[p_pix].mean()) if p_pix.any() else float("nan")
    g_bg = float(g[bg].mean()) if bg.any() else float("nan")
    ratio = g_person / (g_bg + 1e-8) if p_pix.any() and bg.any() else float("inf")
    return {
        "person_eq": person_eq,
        "vehicle_eq": vehicle_eq,
        "grad_ratio": ratio,
        "person_px": int(person.sum()),
        "vehicle_px": int(vehicle.sum()),
    }


def pick_bins(rows: list[dict], n_stress: int = 7, n_easy: int = 3) -> list[dict]:
    has_person = [r for r in rows if r["person_px"] > 0]
    stress = [
        r
        for r in has_person
        if r["person_eq"] >= 0.008 and r["grad_ratio"] < 1.15
    ]
    stress.sort(key=lambda r: (r["person_eq"], -r["grad_ratio"]), reverse=True)
    easy = [
        r
        for r in has_person
        if r["person_eq"] < 0.004 or r["grad_ratio"] >= 1.2
    ]
    easy.sort(key=lambda r: r["person_eq"])
    chosen = []
    seen = set()
    for r in stress[:n_stress] + easy[:n_easy]:
        if r["stem"] in seen:
            continue
        r["bin"] = "stress" if r in stress[:n_stress] else "easy"
        chosen.append(r)
        seen.add(r["stem"])
    if len(chosen) < n_stress + n_easy:
        leftovers = [r for r in has_person if r["stem"] not in seen]
        leftovers.sort(key=lambda r: r["person_eq"], reverse=True)
        for r in leftovers:
            if len(chosen) >= n_stress + n_easy:
                break
            r["bin"] = "fill"
            chosen.append(r)
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CVRG-Pano GT four-config batch.")
    parser.add_argument("--cvrg-root", type=Path, default=cvrg_root())
    parser.add_argument("--split", default="val")
    parser.add_argument("-n", "--seams", type=int, default=24)
    parser.add_argument("--max-width", type=int, default=1024)
    parser.add_argument("--lambda-s", type=float, default=4.0)
    parser.add_argument("--n-stress", type=int, default=7)
    parser.add_argument("--n-easy", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("output") / "cvrg_batch")
    args = parser.parse_args(argv)
    args.cvrg_root = require_path(args.cvrg_root, flag="--cvrg-root", env="CVRG_PANO_ROOT")

    panos = list_panoramas(args.cvrg_root, split=args.split)
    if args.split == "val":
        panos = panos + list_panoramas(args.cvrg_root, split="test")
    print(f"screening {len(panos)} panos from {args.cvrg_root} ...")
    rows = []
    for i, pano in enumerate(panos):
        sample = load_cvrg_sample(pano, max_width=512)
        stats = screen_one(sample)
        stats.update(stem=pano.stem, split=pano.split, pano=pano, index=i)
        rows.append(stats)
        if (i + 1) % 20 == 0:
            print(f"  screened {i + 1}/{len(panos)}")

    chosen = pick_bins(rows, args.n_stress, args.n_easy)
    print(f"selected {len(chosen)} images")
    for r in chosen:
        print(
            f"  {r['bin']:<6} {r['split']}/{r['stem']:<12} "
            f"person_eq={r['person_eq']*100:.2f}%  veh_eq={r['vehicle_eq']*100:.2f}%  "
            f"grad_ratio={r['grad_ratio']:.2f}"
        )

    tax = taxonomy_for(dataset="cvrg")
    importance = importance_map(tax)
    id2label = id2label_from_names(CVRG_CLASS_NAMES)
    groups = group_ids_from_id2label(id2label, protected_group_names(tax))
    configs = (
        ("(a) gradient", 1.0, 0.0, 0.0),
        ("(b) +spherical", 1.0, 1.0, 0.0),
        ("(c) +semantic", 1.0, 1.0, float(args.lambda_s)),
        ("(d) semantic-only", 0.0, 0.0, 1.0),
    )

    args.out.mkdir(parents=True, exist_ok=True)
    print()
    header = (
        f"{'bin':<6} {'image':<16} {'(b) per%':>9} {'(c) per%':>9} {'c-b per':>8} "
        f"{'(b) veh%':>9} {'(c) veh%':>9} {'c-b veh':>8} {'(b) polar':>9} {'(d) polar':>9}"
    )
    print(header)
    print("-" * len(header))
    deltas_p = []
    deltas_v = []
    for r in chosen:
        sample = load_cvrg_sample(r["pano"], max_width=args.max_width)
        by_name = {}
        for name, lg, lp, ls in configs:
            out = run_c.run_config(
                sample.image,
                sample.class_map,
                sample.void_mask,
                args.seams,
                lg, lp, ls,
                importance,
                id2label,
                groups,
            )
            by_name[name] = out
        pb, pc = by_name["(b) +spherical"]["pct"], by_name["(c) +semantic"]["pct"]
        dp = pc.get("person", 0.0) - pb.get("person", 0.0)
        dv = pc.get("vehicle", 0.0) - pb.get("vehicle", 0.0)
        if r["bin"] == "stress":
            deltas_p.append(dp)
            deltas_v.append(dv)
        print(
            f"{r['bin']:<6} {r['stem']:<16} "
            f"{pb.get('person', 0):9.2f} {pc.get('person', 0):9.2f} {dp:8.2f} "
            f"{pb.get('vehicle', 0):9.2f} {pc.get('vehicle', 0):9.2f} {dv:8.2f} "
            f"{by_name['(b) +spherical']['polar_route']:9.3f} "
            f"{by_name['(d) semantic-only']['polar_route']:9.3f}"
        )
        tinted = overlay_protected_classes(sample.image, sample.class_map, groups=groups)
        from seam.carve import map_seams_to_original

        dest = args.out / r["bin"] / r["stem"]
        dest.mkdir(parents=True, exist_ok=True)
        save_erp(dest / "original.png", sample.image)
        save_erp(dest / "class_tint.png", tinted)
        for key, fname in (
            ("(a) gradient", "overlay_a.png"),
            ("(b) +spherical", "overlay_b.png"),
            ("(c) +semantic", "overlay_c.png"),
            ("(d) semantic-only", "overlay_d.png"),
        ):
            mapped = map_seams_to_original(by_name[key]["seams"])
            save_erp(dest / fname, overlay_seams(tinted, mapped, color=(255, 32, 32), thickness=0))

    print()
    if deltas_p:
        med_p = float(np.median(deltas_p))
        med_v = float(np.median(deltas_v))
        print(
            f"stress median (c)-(b): person {med_p:+.2f} pp, vehicle {med_v:+.2f} pp "
            f"(n={len(deltas_p)})"
        )
        if abs(med_p) < 0.05 and abs(med_v) < 0.05:
            print("Finding: semantic delta stays near zero on this GT stress set.")
        else:
            print("Finding: semantic term moved protected hits on the stress set.")
    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
