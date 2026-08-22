"""C2 PanoCrowd: does semantic break gradient+spherical redundancy?

LOCKED BEFORE NUMBERS. Dual-outcome. Medium density is the claim bin.

Dataset: PanoCrowd official val union test (not train). Density bins from
the dataset's own counts, grouped as sparse 0-49 / medium 50-199 / dense 200+.
Medium is the pre-registered stress: people enough that gradient cannot
dodge them all, corridors still possible.

Energy: GT head disks, not SegFormer. Same isolation as CVRG (GT semantics).
A segmenter is a different experiment. Disks use the same latitude-aware
r(phi) as the hit test so we protect what we score.

Comparison: (b) gradient+spherical vs (c) +semantic, cyclic DP, lambda_s=4.
Carve 48 seams once; report head-hit rate at k in {8, 24, 48}. Primary k=24.

Hit: a head is hit if any of the first k original-space seams, on the head
row, is within cyclic r(phi). r_eq = W * 2deg/360, r(phi) = r_eq / max(cos(phi),
cos 75deg). Metric = fraction of heads hit (not pixel %).

Coverage (diagnostic, not the outcome): equatorial fraction of person-disk
pixels. High coverage is the uniform-energy flattening caveat.

Support bar (primary k=24, medium n>=30 else empty-bin):

  characterized-boundary -- mean (c)-(b) hit rate <= -0.05 AND majority of
      medium images have (c)<(b), AND sparse mean d > medium mean d
      (sparse closer to zero). Thesis win: semantic matters when crowds
      stop doing the baseline's job.
  unconditional-help -- medium meets the -0.05/majority bar AND sparse
      also mean d <= -0.05. Semantic helps even when people are rare;
      not a density-conditioned boundary.
  weak -- medium majority (c)<(b) but mean d in (-0.05, 0]. Direction
      only; do not headline.
  dense-null -- medium does not meet majority+direction at 5pp. Semantic
      stays redundant even in the stress bin. Paper B is redundancy and
      empty regimes; flattening note if dense coverage is high.
  empty-bin -- medium n < 30. Do not claim.

Reduction sweep does not move the bar. If only k=48 meets -0.05, that is
robustness failure: report it, do not promote 48 as the result.

Post-hoc analyses (coverage vs d, polar routing) are exploratory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data.panocrowd import DEFAULT_ROOT, list_panocrowd, load_panocrowd_sample  # noqa: E402
from energy.head_hits import (  # noqa: E402
    THETA_DEG,
    equatorial_person_cover,
    head_hit_rate,
    paint_person_map,
)
from energy.semantic import group_ids_from_id2label  # noqa: E402
from seam.carve import map_seams_to_original  # noqa: E402
from io_erp import save_erp  # noqa: E402
from viz import overlay_seams  # noqa: E402

import run_c_semantic as run_c  # noqa: E402

PRIMARY_K = 24
SWEEP_K = (8, 24, 48)
MAX_SEAMS = 48
LAMBDA_S = 4.0
MAX_WIDTH = 512
WIN_PP = 0.05  # 5 percentage points of head-hit rate
MIN_BIN_N = 30

ID2LABEL = {0: "background", 1: "person"}
IMPORTANCE = {"background": 0.05, "person": 1.0}
GROUPS = group_ids_from_id2label(ID2LABEL, {"person": ("person",)})


def _carve(sample, n_seams: int, lambda_s: float) -> dict:
    cmap = paint_person_map(sample.image.shape[0], sample.image.shape[1], sample.heads)
    void = np.zeros(cmap.shape, dtype=bool)
    out = run_c.run_config(
        sample.image,
        cmap,
        void,
        n_seams,
        1.0,
        1.0,
        float(lambda_s),
        IMPORTANCE,
        ID2LABEL,
        GROUPS,
    )
    mapped = map_seams_to_original(out["seams"])
    h, w = sample.image.shape[:2]
    rates = {
        k: head_hit_rate(mapped, sample.heads, h, w, n_seams=k)
        for k in SWEEP_K
        if k <= n_seams
    }
    return {
        "rates": rates,
        "polar_route": out["polar_route"],
        "mapped": mapped,
        "cover": equatorial_person_cover(cmap),
    }


def _bin_stats(rows: list[dict], key: str) -> dict:
    ds = np.array([r[key] for r in rows], dtype=np.float64)
    if ds.size == 0:
        return {"n": 0, "mean_d": 0.0, "n_protect": 0, "frac_protect": 0.0}
    n_protect = int((ds < -1e-12).sum())
    return {
        "n": int(ds.size),
        "mean_d": float(ds.mean()),
        "median_d": float(np.median(ds)),
        "n_protect": n_protect,
        "frac_protect": n_protect / float(ds.size),
        "mean_b": float(np.mean([r["hit_b"] for r in rows])),
        "mean_c": float(np.mean([r["hit_c"] for r in rows])),
        "mean_cover": float(np.mean([r["cover"] for r in rows])),
    }


def verdict(sparse: dict, medium: dict) -> tuple[str, str]:
    if medium["n"] < MIN_BIN_N:
        return (
            "empty-bin",
            "medium n<30. Do not claim a density result.",
        )
    medium_win = medium["mean_d"] <= -WIN_PP and medium["frac_protect"] > 0.5
    if not medium_win:
        if medium["mean_d"] < 0.0 and medium["frac_protect"] > 0.5:
            return (
                "weak",
                "medium majority (c)<(b) but mean reduction is below 5pp. Direction only.",
            )
        return (
            "dense-null",
            "medium does not meet 5pp + majority. Semantic stays redundant in the stress bin.",
        )
    sparse_also = sparse["n"] >= MIN_BIN_N and sparse["mean_d"] <= -WIN_PP
    if sparse_also:
        return (
            "unconditional-help",
            "semantic reduces head hits in sparse AND medium. Not a density-conditioned boundary.",
        )
    if sparse["n"] >= MIN_BIN_N and sparse["mean_d"] > medium["mean_d"]:
        return (
            "characterized-boundary",
            "medium breaks redundancy; sparse is weaker. Thesis win.",
        )
    return (
        "medium-win",
        "medium meets the 5pp bar; sparse comparison is not a cleaner interaction.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PanoCrowd (b) vs (c) head-hit gate.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", default="val+test")
    parser.add_argument("--max-width", type=int, default=MAX_WIDTH)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("output") / "panocrowd")
    args = parser.parse_args(argv)

    items = list_panocrowd(args.root, split=args.split)
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"no PanoCrowd items under {args.root} split={args.split}")

    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"C2 PanoCrowd LOCKED: n={len(items)} split={args.split} W<={args.max_width} "
        f"k={list(SWEEP_K)} primary={PRIMARY_K} theta={THETA_DEG}deg lambda_s={LAMBDA_S}"
    )
    print("GT head disks, not SegFormer. (b) spherical vs (c) +semantic.")
    print("bar: medium mean d_hit <= -0.05 and majority (c)<(b)")
    print()

    rows: list[dict] = []
    for i, item in enumerate(items):
        sample = load_panocrowd_sample(item, max_width=args.max_width)
        n_seams = min(MAX_SEAMS, sample.image.shape[1] - 1)
        b = _carve(sample, n_seams, 0.0)
        c = _carve(sample, n_seams, LAMBDA_S)
        rec = {
            "stem": item.stem,
            "split": item.split,
            "human_num": sample.human_num,
            "bin": sample.density_bin,
            "H": int(sample.image.shape[0]),
            "W": int(sample.image.shape[1]),
            "n_heads": int(sample.heads.shape[0]),
            "cover": b["cover"],
            "polar_b": b["polar_route"],
            "polar_c": c["polar_route"],
            "hit_b": b["rates"][PRIMARY_K],
            "hit_c": c["rates"][PRIMARY_K],
            "d_hit": c["rates"][PRIMARY_K] - b["rates"][PRIMARY_K],
        }
        for k in SWEEP_K:
            rec[f"hit_b_k{k}"] = b["rates"][k]
            rec[f"hit_c_k{k}"] = c["rates"][k]
            rec[f"d_k{k}"] = c["rates"][k] - b["rates"][k]
        rows.append(rec)
        print(
            f"{i + 1:3d}/{len(items)} {sample.density_bin:<7} {item.stem:<8} "
            f"n={sample.human_num:4d} cover={b['cover']:.3f} "
            f"hit {b['rates'][PRIMARY_K]:.3f}->{c['rates'][PRIMARY_K]:.3f} "
            f"d={rec['d_hit']:+.3f}",
            flush=True,
        )

    by_bin = {name: [r for r in rows if r["bin"] == name] for name in ("sparse", "medium", "dense")}
    stats = {name: _bin_stats(part, "d_hit") for name, part in by_bin.items()}
    print()
    for name in ("sparse", "medium", "dense"):
        s = stats[name]
        print(
            f"{name:8s} n={s['n']:3d}  hit_b={s.get('mean_b', 0):.3f}  hit_c={s.get('mean_c', 0):.3f}  "
            f"d={s['mean_d']:+.4f}  protect={s['n_protect']}/{s['n']}  "
            f"cover={s.get('mean_cover', 0):.3f}"
        )
        for k in SWEEP_K:
            sk = _bin_stats(by_bin[name], f"d_k{k}")
            print(f"         k={k:2d}  d={sk['mean_d']:+.4f}  protect={sk['n_protect']}/{sk['n']}")

    label, note = verdict(stats["sparse"], stats["medium"])
    print()
    print(f"VERDICT {label}: {note}")

    # two overlays from medium: largest protection and a near-zero
    medium = by_bin["medium"]
    if medium:
        order = np.argsort([r["d_hit"] for r in medium])
        picks = {int(order[0]): "medium_max_protect"}
        picks[int(order[len(order) // 2])] = "medium_median"
        stem_to_item = {it.stem: it for it in items}
        for idx, tag in picks.items():
            rec = medium[idx]
            sample = load_panocrowd_sample(stem_to_item[rec["stem"]], max_width=args.max_width)
            n_seams = min(MAX_SEAMS, sample.image.shape[1] - 1)
            b = _carve(sample, n_seams, 0.0)
            c = _carve(sample, n_seams, LAMBDA_S)
            save_erp(
                args.out / f"{tag}_b.png",
                overlay_seams(sample.image, b["mapped"][:PRIMARY_K], color=(255, 32, 32), thickness=0),
            )
            save_erp(
                args.out / f"{tag}_c.png",
                overlay_seams(sample.image, c["mapped"][:PRIMARY_K], color=(32, 200, 64), thickness=0),
            )

    summary = {
        "n": len(rows),
        "split": args.split,
        "primary_k": PRIMARY_K,
        "theta_deg": THETA_DEG,
        "lambda_s": LAMBDA_S,
        "win_pp": WIN_PP,
        "bins": stats,
        "verdict": label,
        "note": note,
        "energy": "GT head disks, not SegFormer",
    }
    (args.out / "metrics.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
