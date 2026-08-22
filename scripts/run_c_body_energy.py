"""C2b body-energy: (b) vs (c) on body columns. NEW experiment, not a C2 swap.

The 2 deg head-disk run stays as C2 (measured heads). This run has its own
energy, metric, bar, and output dir.

Column (same as the proxy check): meridian strip, 12 deg latitude down from
the GT head, half-width 2.5 x r_head(phi). Energy AND hits use this column.

Sanity (locked, before the table): overlay columns on RGB for a fixed set of
stems. If columns systematically paint floor/sky/neighbor instead of bodies
on 2+ of those images, STOP. Do not run (b) vs (c) on a broken proxy.

Subset (locked from (b) only, not from (c)): high-body = medium images with
(b) body-hit >= 20%. That is the ~30% tail where the baseline already takes
body. Complement = the rest of medium.

Support bar, k=24, n>=30 else empty:

  characterized-boundary -- high-body meets mean (c)-(b) <= -5pp AND
      majority (c)<(b); complement does not. Semantic is redundant when
      the baseline misses bodies, and protects when seams cut them.

  body-protects-unconditional -- all-medium meets 5pp+majority (and the
      complement is not a null). Semantic reduces body-hits generally.

  body-redundant -- all-medium fails 5pp+majority AND high-body fails.
      Genuine C2 null, now robust to the 2 deg target objection.

  weak -- majority (c)<(b) on the claim bin but mean d in (-5pp, 0].
      Direction only.

  empty-subset -- high-body n<30. Do not claim a tail boundary.

k in {8,24,48}; primary 24. Do not promote 48. Head-disk C2 is not rewritten.
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
    BODY_LAT_DEG,
    SHOULDER_SCALE,
    body_hit_rate,
    equatorial_person_cover,
    paint_body_map,
)
from energy.semantic import group_ids_from_id2label  # noqa: E402
from io_erp import save_erp  # noqa: E402
from seam.carve import map_seams_to_original  # noqa: E402
from viz import overlay_protected_classes, overlay_seams  # noqa: E402

import run_c_semantic as run_c  # noqa: E402

PRIMARY_K = 24
SWEEP_K = (8, 24, 48)
MAX_SEAMS = 48
LAMBDA_S = 4.0
MAX_WIDTH = 512
WIN_PP = 0.05
MIN_BIN_N = 30
HIGH_BODY = 0.20
ID2LABEL = {0: "background", 1: "person"}
IMPORTANCE = {"background": 0.05, "person": 1.0}
GROUPS = group_ids_from_id2label(ID2LABEL, {"person": ("person",)})

# Fixed sanity stems: high-body medium, zero-body medium, sparse, dense.
SANITY_STEMS = ("149", "219", "166", "151", "157", "729")


def _carve(sample, n_seams: int, lambda_s: float) -> dict:
    cmap = paint_body_map(sample.image.shape[0], sample.image.shape[1], sample.heads)
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
        k: body_hit_rate(mapped, sample.heads, h, w, n_seams=k)
        for k in SWEEP_K
        if k <= n_seams
    }
    return {
        "rates": rates,
        "mapped": mapped,
        "cover": equatorial_person_cover(cmap),
        "cmap": cmap,
    }


def _meets(stats: dict) -> bool:
    return stats["n"] >= MIN_BIN_N and stats["mean_d"] <= -WIN_PP and stats["frac_protect"] > 0.5


def _stats(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n": 0,
            "mean_d": 0.0,
            "n_protect": 0,
            "frac_protect": 0.0,
            "mean_b": 0.0,
            "mean_c": 0.0,
            "mean_cover": 0.0,
        }
    ds = np.array([r["d_hit"] for r in rows], dtype=np.float64)
    n_protect = int((ds < -1e-12).sum())
    return {
        "n": len(rows),
        "mean_d": float(ds.mean()),
        "median_d": float(np.median(ds)),
        "n_protect": n_protect,
        "frac_protect": n_protect / float(len(rows)),
        "mean_b": float(np.mean([r["hit_b"] for r in rows])),
        "mean_c": float(np.mean([r["hit_c"] for r in rows])),
        "mean_cover": float(np.mean([r["cover"] for r in rows])),
    }


def verdict(medium: dict, high: dict, rest: dict) -> tuple[str, str]:
    if medium["n"] < MIN_BIN_N:
        return "empty-bin", "medium n<30. Do not claim."
    if high["n"] < MIN_BIN_N:
        if _meets(medium):
            return (
                "body-protects-unconditional",
                "high-body subset n<30; all-medium meets 5pp. No tail boundary claim.",
            )
        if medium["mean_d"] < 0 and medium["frac_protect"] > 0.5:
            return "weak", "high-body n<30; all-medium direction only."
        return "body-redundant", "high-body n<30; all-medium fails 5pp+majority."
    if _meets(high) and not _meets(rest):
        return (
            "characterized-boundary",
            "semantic reduces body-hits where (b) already cuts bodies; not on the rest.",
        )
    if _meets(medium):
        return (
            "body-protects-unconditional",
            "semantic reduces body-hits on medium overall, not only the high-body tail.",
        )
    if (not _meets(medium)) and (not _meets(high)):
        if (medium["mean_d"] < 0 and medium["frac_protect"] > 0.5) or (
            high["mean_d"] < 0 and high["frac_protect"] > 0.5
        ):
            return "weak", "direction only on medium or high-body; below 5pp or not majority."
        return (
            "body-redundant",
            "body-hits also redundant. C2 null is robust to the 2 deg target objection.",
        )
    return "weak", "mixed; do not headline a boundary or a clean null."


def write_sanity(items, out: Path) -> list[str]:
    by_stem = {it.stem: it for it in items}
    written: list[str] = []
    out.mkdir(parents=True, exist_ok=True)
    for stem in SANITY_STEMS:
        if stem not in by_stem:
            print(f"sanity missing stem {stem}")
            continue
        sample = load_panocrowd_sample(by_stem[stem], max_width=MAX_WIDTH)
        cmap = paint_body_map(sample.image.shape[0], sample.image.shape[1], sample.heads)
        tinted = overlay_protected_classes(
            sample.image, cmap, groups=GROUPS, alpha=0.5
        )
        path = out / f"sanity_{stem}.png"
        save_erp(path, tinted)
        written.append(str(path))
        print(
            f"sanity {stem} bin={sample.density_bin} n={sample.human_num} "
            f"cover={equatorial_person_cover(cmap):.3f} -> {path}"
        )
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C2b body-energy (b) vs (c).")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", default="val+test")
    parser.add_argument("--max-width", type=int, default=MAX_WIDTH)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sanity-only", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("output") / "panocrowd_body_energy")
    args = parser.parse_args(argv)

    items = list_panocrowd(args.root, split=args.split)
    if args.limit:
        items = items[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    print(
        f"C2b body-energy LOCKED: split={args.split} k={PRIMARY_K} "
        f"body={BODY_LAT_DEG}deg x{SHOULDER_SCALE} lambda_s={LAMBDA_S} "
        f"high-body>={HIGH_BODY:.0%} bar=-{WIN_PP:.0%}+majority"
    )
    print("this does not replace the 2 deg head-disk table")
    print()
    write_sanity(items, args.out / "sanity")
    if args.sanity_only:
        print("sanity-only: inspect overlays before running the table")
        return 0

    print()
    print("running (b) vs (c) on body-column energy ...")
    rows: list[dict] = []
    for i, item in enumerate(items):
        sample = load_panocrowd_sample(item, max_width=args.max_width)
        n_seams = min(MAX_SEAMS, sample.image.shape[1] - 1)
        b = _carve(sample, n_seams, 0.0)
        c = _carve(sample, n_seams, LAMBDA_S)
        rec = {
            "stem": item.stem,
            "bin": sample.density_bin,
            "human_num": sample.human_num,
            "cover": b["cover"],
            "hit_b": b["rates"][PRIMARY_K],
            "hit_c": c["rates"][PRIMARY_K],
            "d_hit": c["rates"][PRIMARY_K] - b["rates"][PRIMARY_K],
            "high_body": bool(b["rates"][PRIMARY_K] >= HIGH_BODY),
        }
        for k in SWEEP_K:
            rec[f"d_k{k}"] = c["rates"][k] - b["rates"][k]
            rec[f"hit_b_k{k}"] = b["rates"][k]
            rec[f"hit_c_k{k}"] = c["rates"][k]
        rows.append(rec)
        print(
            f"{i + 1:3d}/{len(items)} {sample.density_bin:<7} {item.stem:<8} "
            f"{'HIGH' if rec['high_body'] else '    '} "
            f"hit {rec['hit_b']:.3f}->{rec['hit_c']:.3f} d={rec['d_hit']:+.3f}",
            flush=True,
        )

    medium = [r for r in rows if r["bin"] == "medium"]
    high = [r for r in medium if r["high_body"]]
    rest = [r for r in medium if not r["high_body"]]
    sparse = [r for r in rows if r["bin"] == "sparse"]
    dense = [r for r in rows if r["bin"] == "dense"]
    stats = {
        "sparse": _stats(sparse),
        "medium": _stats(medium),
        "dense": _stats(dense),
        "high_body": _stats(high),
        "medium_rest": _stats(rest),
    }
    print()
    for name in ("sparse", "medium", "high_body", "medium_rest", "dense"):
        s = stats[name]
        print(
            f"{name:12s} n={s['n']:3d}  b={s['mean_b']:.3f}  c={s['mean_c']:.3f}  "
            f"d={s['mean_d']:+.4f}  protect={s['n_protect']}/{s['n']}"
        )
        if name in ("medium", "high_body") and s["n"]:
            for k in SWEEP_K:
                key = f"d_k{k}"
                ds = np.array([r[key] for r in (medium if name == "medium" else high)])
                print(
                    f"{'':12s} k={k:2d}  d={float(ds.mean()):+.4f}  "
                    f"protect={int((ds < -1e-12).sum())}/{ds.size}"
                )

    label, note = verdict(stats["medium"], stats["high_body"], stats["medium_rest"])
    print()
    print(f"VERDICT {label}: {note}")

    # overlays: one high-body medium pair if present
    if high:
        rec = min(high, key=lambda r: r["d_hit"])
        item = next(it for it in items if it.stem == rec["stem"])
        sample = load_panocrowd_sample(item, max_width=args.max_width)
        n_seams = min(MAX_SEAMS, sample.image.shape[1] - 1)
        b = _carve(sample, n_seams, 0.0)
        c = _carve(sample, n_seams, LAMBDA_S)
        tinted = overlay_protected_classes(sample.image, b["cmap"], groups=GROUPS, alpha=0.4)
        save_erp(
            args.out / "highbody_b.png",
            overlay_seams(tinted, b["mapped"][:PRIMARY_K], color=(255, 32, 32), thickness=0),
        )
        save_erp(
            args.out / "highbody_c.png",
            overlay_seams(tinted, c["mapped"][:PRIMARY_K], color=(32, 200, 64), thickness=0),
        )

    summary = {
        "n": len(rows),
        "primary_k": PRIMARY_K,
        "body_lat_deg": BODY_LAT_DEG,
        "shoulder_scale": SHOULDER_SCALE,
        "high_body_threshold": HIGH_BODY,
        "win_pp": WIN_PP,
        "verdict": label,
        "note": note,
        "bins": stats,
        "replaces_c2_head": False,
    }
    (args.out / "metrics.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
