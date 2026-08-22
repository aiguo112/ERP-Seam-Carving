"""C2 body-proxy check: was the 4.3% (b) hit rate a 2 deg disk artifact?

LOCKED BEFORE NUMBERS. Scores the spherical baseline (b) against a
person-scale body column, without changing energy.

Body column (locked):
  - same longitude as the GT head (meridian strip)
  - from the head row DOWNWARD by 12 deg of latitude
    (y increases down; feet are below heads in ERP)
  - half-width at row y = 2.5 * r_head(y)  (shoulders, latitude-aware)

Metric: fraction of people whose body column is cut by any of the first
k=24 original-space seams. Nested: a head hit is a body hit.

Support bar -- medium bin, k=24, n>=30:

  genuine-redundancy -- mean (b) body-hit < 10%.
      Even person-scale targets are rarely hit. The 2 deg ceiling was not
      hiding damage. C2 dense-null stands.

  proxy-artifact -- mean (b) body-hit >= 20%.
      Seams were cutting people; 2 deg heads hid it. Re-run (b) vs (c)
      with body columns as BOTH energy and metric, same 5pp+majority bar
      as C2. Do not keep the head-disk null.

  ambiguous -- body-hit in [10%, 20).
      Do not treat C2 as a clean null or as an artifact. Report both
      rates; do not draft the populated-bin null as settled.

This check does not reopen C2's 5pp bar. It only asks whether that bar
was computed on the wrong target.
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
    THETA_DEG,
    body_hit_rate,
    equatorial_person_cover,
    head_hit_rate,
    paint_body_map,
    paint_person_map,
)
from energy.semantic import group_ids_from_id2label  # noqa: E402
from seam.carve import map_seams_to_original  # noqa: E402

import run_c_semantic as run_c  # noqa: E402

PRIMARY_K = 24
MAX_SEAMS = 48
MAX_WIDTH = 512
LAMBDA_S = 4.0
MIN_BIN_N = 30
GENUINE_MAX = 0.10
ARTIFACT_MIN = 0.20
WIN_PP = 0.05

ID2LABEL = {0: "background", 1: "person"}
IMPORTANCE = {"background": 0.05, "person": 1.0}
GROUPS = group_ids_from_id2label(ID2LABEL, {"person": ("person",)})


def _carve_b(sample, n_seams: int) -> list[np.ndarray]:
    cmap = paint_person_map(sample.image.shape[0], sample.image.shape[1], sample.heads)
    void = np.zeros(cmap.shape, dtype=bool)
    out = run_c.run_config(
        sample.image,
        cmap,
        void,
        n_seams,
        1.0,
        1.0,
        0.0,
        IMPORTANCE,
        ID2LABEL,
        GROUPS,
    )
    return map_seams_to_original(out["seams"])


def _carve_body_energy(sample, n_seams: int, lambda_s: float) -> list[np.ndarray]:
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
    return map_seams_to_original(out["seams"])


def _verdict(mean_body: float, n: int) -> tuple[str, str]:
    if n < MIN_BIN_N:
        return "empty-bin", "medium n<30. Do not claim."
    if mean_body < GENUINE_MAX:
        return (
            "genuine-redundancy",
            "body-hit under 10%. 2 deg disks were not hiding person damage. C2 stands.",
        )
    if mean_body >= ARTIFACT_MIN:
        return (
            "proxy-artifact",
            "body-hit >= 20%. Head disks hid person cuts. Re-run (b) vs (c) on body columns.",
        )
    return (
        "ambiguous",
        "body-hit in [10%, 20). Do not settle C2 as a clean null or as an artifact.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PanoCrowd body-column proxy check.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", default="val+test")
    parser.add_argument("--max-width", type=int, default=MAX_WIDTH)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--rerun-if-artifact",
        action="store_true",
        default=True,
        help="If body-hit >= 20 percent, re-run (b) vs (c) with body energy+metric.",
    )
    parser.add_argument("--out", type=Path, default=Path("output") / "panocrowd_body")
    args = parser.parse_args(argv)

    items = list_panocrowd(args.root, split=args.split)
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"no items under {args.root}")

    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"C2 body-proxy LOCKED: n={len(items)}  k={PRIMARY_K}  "
        f"body={BODY_LAT_DEG}deg lat down  shoulder={SHOULDER_SCALE}x  "
        f"genuine<={GENUINE_MAX:.0%}  artifact>={ARTIFACT_MIN:.0%}"
    )
    print("scoring (b) spherical seams on head disks vs body columns")
    print()

    rows: list[dict] = []
    for i, item in enumerate(items):
        sample = load_panocrowd_sample(item, max_width=args.max_width)
        n_seams = min(MAX_SEAMS, sample.image.shape[1] - 1)
        mapped = _carve_b(sample, n_seams)
        h, w = sample.image.shape[:2]
        hit_h = head_hit_rate(mapped, sample.heads, h, w, n_seams=PRIMARY_K)
        hit_b = body_hit_rate(mapped, sample.heads, h, w, n_seams=PRIMARY_K)
        rec = {
            "stem": item.stem,
            "bin": sample.density_bin,
            "human_num": sample.human_num,
            "hit_head": hit_h,
            "hit_body": hit_b,
            "cover_head": equatorial_person_cover(
                paint_person_map(h, w, sample.heads)
            ),
            "cover_body": equatorial_person_cover(paint_body_map(h, w, sample.heads)),
        }
        rows.append(rec)
        print(
            f"{i + 1:3d}/{len(items)} {sample.density_bin:<7} {item.stem:<8} "
            f"head={hit_h:.3f}  body={hit_b:.3f}  "
            f"cover {rec['cover_head']:.3f}->{rec['cover_body']:.3f}",
            flush=True,
        )

    def _mean(part: list[dict], key: str) -> float:
        if not part:
            return 0.0
        return float(np.mean([r[key] for r in part]))

    print()
    by_bin: dict[str, list[dict]] = {}
    for name in ("sparse", "medium", "dense"):
        part = [r for r in rows if r["bin"] == name]
        by_bin[name] = part
        print(
            f"{name:8s} n={len(part):3d}  hit_head={_mean(part, 'hit_head'):.4f}  "
            f"hit_body={_mean(part, 'hit_body'):.4f}  "
            f"cover_body={_mean(part, 'cover_body'):.3f}"
        )

    medium = by_bin["medium"]
    mean_body = _mean(medium, "hit_body")
    mean_head = _mean(medium, "hit_head")
    label, note = _verdict(mean_body, len(medium))
    print()
    print(
        f"medium (b) head={mean_head:.4f}  body={mean_body:.4f}  "
        f"ratio={mean_body / max(mean_head, 1e-9):.2f}"
    )
    print(f"VERDICT {label}: {note}")

    follow: dict | None = None
    if label == "proxy-artifact" and args.rerun_if_artifact:
        print()
        print("re-running (b) vs (c) with body-column energy AND body-hit metric ...")
        drows = []
        for i, item in enumerate(items):
            sample = load_panocrowd_sample(item, max_width=args.max_width)
            n_seams = min(MAX_SEAMS, sample.image.shape[1] - 1)
            h, w = sample.image.shape[:2]
            mb = _carve_body_energy(sample, n_seams, 0.0)
            mc = _carve_body_energy(sample, n_seams, LAMBDA_S)
            hb = body_hit_rate(mb, sample.heads, h, w, n_seams=PRIMARY_K)
            hc = body_hit_rate(mc, sample.heads, h, w, n_seams=PRIMARY_K)
            drows.append(
                {
                    "stem": item.stem,
                    "bin": sample.density_bin,
                    "hit_b": hb,
                    "hit_c": hc,
                    "d_hit": hc - hb,
                }
            )
            print(
                f"  body-energy {i + 1}/{len(items)} {sample.density_bin:<7} "
                f"{hb:.3f}->{hc:.3f} d={hc - hb:+.3f}",
                flush=True,
            )
        med = [r for r in drows if r["bin"] == "medium"]
        ds = np.array([r["d_hit"] for r in med])
        n_protect = int((ds < -1e-12).sum())
        mean_d = float(ds.mean()) if ds.size else 0.0
        frac = n_protect / max(len(med), 1)
        body_win = mean_d <= -WIN_PP and frac > 0.5
        print(
            f"body-energy medium n={len(med)}  d={mean_d:+.4f}  "
            f"protect={n_protect}/{len(med)}  win_5pp={body_win}"
        )
        follow = {
            "medium_n": len(med),
            "mean_d": mean_d,
            "n_protect": n_protect,
            "frac_protect": frac,
            "meets_c2_bar": body_win,
            "rows": drows,
        }

    summary = {
        "n": len(rows),
        "primary_k": PRIMARY_K,
        "body_lat_deg": BODY_LAT_DEG,
        "shoulder_scale": SHOULDER_SCALE,
        "genuine_max": GENUINE_MAX,
        "artifact_min": ARTIFACT_MIN,
        "medium_n": len(medium),
        "medium_hit_head": mean_head,
        "medium_hit_body": mean_body,
        "verdict": label,
        "note": note,
        "bins": {
            name: {
                "n": len(part),
                "hit_head": _mean(part, "hit_head"),
                "hit_body": _mean(part, "hit_body"),
                "cover_body": _mean(part, "cover_body"),
            }
            for name, part in by_bin.items()
        },
        "body_energy_followup": None
        if follow is None
        else {k: v for k, v in follow.items() if k != "rows"},
    }
    payload = {"summary": summary, "rows": rows}
    if follow is not None:
        payload["body_energy_rows"] = follow["rows"]
    (args.out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
