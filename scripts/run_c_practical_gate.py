"""Practical (predicted) content-conditioned gate on PanoCrowd medium.

NOT equivalent to the oracle C2b gate.

Oracle C2b: GT head points -> constructed 12 deg x 2.5 body columns.
This run: SegFormer/DeepLab *person pixels* (no head detector). Gate statistic
is the fraction of predicted person connected-components cut by (b) seams.
Energy (c) protects those predicted person pixels, not GT columns.

Evaluation still reports GT body-column hit rates so the oracle table can be
compared without pretending the predicted mask is GT.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data.panocrowd import DEFAULT_ROOT, list_panocrowd, load_panocrowd_sample  # noqa: E402
from energy.head_hits import body_hit_rate  # noqa: E402
from energy.semantic import group_ids_from_id2label  # noqa: E402
from io_erp import save_erp  # noqa: E402
from seam.carve import map_seams_to_original  # noqa: E402
from segmentation.factory import get_segmenter  # noqa: E402
from viz import overlay_protected_classes, overlay_seams  # noqa: E402

import run_c_semantic as run_c  # noqa: E402

PRIMARY_K = 24
LAMBDA_S = 4.0
MAX_WIDTH = 512
TAU = 0.20
N_VIZ = 8

# Same LUT as C2b, applied to a predicted 0/1 person map (not GT columns).
ID2LABEL = {0: "background", 1: "person"}
IMPORTANCE = {"background": 0.05, "person": 1.0}
GROUPS = group_ids_from_id2label(ID2LABEL, {"person": ("person",)})


def person_ids(id2label: dict[int, str]) -> list[int]:
    ids = []
    for k, name in id2label.items():
        if " ".join(str(name).strip().lower().split()) == "person":
            ids.append(int(k))
    return ids


def predicted_person_map(class_map: np.ndarray, id2label: dict[int, str]) -> np.ndarray:
    """HxW int32: 1 on predicted person pixels, 0 elsewhere. Not GT."""
    ids = person_ids(id2label)
    mask = np.isin(np.asarray(class_map), ids) if ids else np.zeros(class_map.shape, dtype=bool)
    return mask.astype(np.int32)


def instance_hit_rate(
    seams_original: list[np.ndarray],
    person_map: np.ndarray,
    *,
    n_seams: int,
) -> tuple[float, int, int]:
    """Fraction of predicted person CCs hit by any of the first k original seams.

    Connected components of the predicted mask. Not GT people, not head points.
    """
    binary = (np.asarray(person_map) == 1).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(binary, connectivity=8)
    n_inst = int(n_labels - 1)
    if n_inst <= 0 or not seams_original:
        return 0.0, 0, 0
    use = seams_original[:n_seams]
    hit = np.zeros(n_labels, dtype=bool)
    h = labels.shape[0]
    for seam in use:
        s = np.asarray(seam, dtype=np.int32)
        lab = labels[np.arange(h), np.clip(s, 0, labels.shape[1] - 1)]
        hit[lab] = True
    hit[0] = False
    n_hit = int(hit[1:].sum())
    return float(n_hit) / float(n_inst), n_hit, n_inst


def _carve(image: np.ndarray, person_map: np.ndarray, n_seams: int, lambda_s: float) -> dict:
    void = np.zeros(person_map.shape, dtype=bool)
    out = run_c.run_config(
        image,
        person_map,
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
    return {"mapped": mapped, "carved": out["carved"], "seams": out["seams"]}


def mean_median_ci(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    n = int(a.size)
    if n == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "ci95_lo": 0.0, "ci95_hi": 0.0}
    mean = float(a.mean())
    median = float(np.median(a))
    if n == 1:
        return {"n": 1, "mean": mean, "median": median, "ci95_lo": mean, "ci95_hi": mean}
    se = float(a.std(ddof=1) / np.sqrt(n))
    return {
        "n": n,
        "mean": mean,
        "median": median,
        "ci95_lo": mean - 1.96 * se,
        "ci95_hi": mean + 1.96 * se,
    }


def agg_hits(rows: list[dict], b_key: str, a_key: str) -> dict:
    if not rows:
        return {
            "n": 0,
            "hit_before": mean_median_ci([]),
            "hit_after": mean_median_ci([]),
            "delta": mean_median_ci([]),
            "n_after_lt_before": 0,
        }
    before = [float(r[b_key]) for r in rows]
    after = [float(r[a_key]) for r in rows]
    delta = [a - b for a, b in zip(after, before, strict=True)]
    return {
        "n": len(rows),
        "hit_before": mean_median_ci(before),
        "hit_after": mean_median_ci(after),
        "delta": mean_median_ci(delta),
        "n_after_lt_before": int(sum(1 for d in delta if d < -1e-12)),
    }


def save_panel(
    path: Path,
    original: np.ndarray,
    pred_tint: np.ndarray,
    seams_b: np.ndarray,
    seams_g: np.ndarray,
    carved: np.ndarray,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(22, 3.6))
    panels = (
        (original, "ERP"),
        (pred_tint, "predicted person (not GT)"),
        (seams_b, "(b) baseline seams"),
        (seams_g, "gated seams"),
        (carved, "final ERP (gated)"),
    )
    for ax, (im, lab) in zip(axes, panels, strict=True):
        ax.imshow(im)
        ax.set_title(lab, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def load_oracle(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    blob = json.loads(path.read_text(encoding="utf-8"))
    return {str(r["stem"]): r for r in blob.get("rows", [])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Practical predicted-person gate. Not the oracle C2b experiment."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", default="val+test")
    parser.add_argument("--segmenter", choices=("segformer", "deeplabv3"), default="segformer")
    parser.add_argument("--max-width", type=int, default=MAX_WIDTH)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument(
        "--oracle-json",
        type=Path,
        default=Path("output") / "panocrowd_body_energy" / "metrics.json",
    )
    parser.add_argument("--out", type=Path, default=Path("output") / "panocrowd_practical_gate")
    args = parser.parse_args(argv)

    items = list_panocrowd(args.root, split=args.split)
    if args.limit:
        items = items[: args.limit]

    oracle_by_stem = load_oracle(args.oracle_json)
    args.out.mkdir(parents=True, exist_ok=True)
    viz_dir = args.out / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"PRACTICAL GATE (not oracle): segmenter={args.segmenter} "
        f"split={args.split} k={PRIMARY_K} tau={args.tau:.2f} "
        f"lambda_s={LAMBDA_S} W<={args.max_width}"
    )
    print("gate uses predicted person connected-components, not GT heads")
    print("GT body-column hits are evaluation-only")
    print()

    print(f"loading {args.segmenter} ...")
    seg = get_segmenter(args.segmenter)
    id2label = dict(seg.id2label)

    rows: list[dict] = []
    viz_on: list = []
    viz_off: list = []

    for i, item in enumerate(items):
        sample = load_panocrowd_sample(item, max_width=args.max_width)
        if sample.density_bin != "medium":
            continue
        n_seams = min(PRIMARY_K, sample.image.shape[1] - 1)
        class_map = seg.predict(sample.image)
        pred_map = predicted_person_map(class_map, id2label)
        n_pred_px = int((pred_map == 1).sum())

        b = _carve(sample.image, pred_map, n_seams, 0.0)
        c = _carve(sample.image, pred_map, n_seams, LAMBDA_S)
        h, w = sample.image.shape[:2]
        mapped_b = b["mapped"]
        mapped_c = c["mapped"]

        pred_hit_b, pred_n_hit, pred_n_inst = instance_hit_rate(
            mapped_b, pred_map, n_seams=PRIMARY_K
        )
        pred_hit_c, _, _ = instance_hit_rate(mapped_c, pred_map, n_seams=PRIMARY_K)
        gate_on = bool(pred_hit_b >= args.tau)

        gt_b = body_hit_rate(mapped_b, sample.heads, h, w, n_seams=PRIMARY_K)
        gt_c = body_hit_rate(mapped_c, sample.heads, h, w, n_seams=PRIMARY_K)
        mapped_g = mapped_c if gate_on else mapped_b
        carved_g = c["carved"] if gate_on else b["carved"]
        gt_g = gt_c if gate_on else gt_b
        pred_hit_g = pred_hit_c if gate_on else pred_hit_b

        o = oracle_by_stem.get(item.stem, {})
        rec = {
            "stem": item.stem,
            "bin": "medium",
            "human_num_gt": sample.human_num,
            "n_pred_person_px": n_pred_px,
            "n_pred_instances": pred_n_inst,
            "pred_hit_b": pred_hit_b,
            "pred_hit_c": pred_hit_c,
            "pred_hit_gated": pred_hit_g,
            "gate_on": gate_on,
            "gt_body_hit_b": gt_b,
            "gt_body_hit_c_always_on": gt_c,
            "gt_body_hit_gated": gt_g,
            "d_gt_always_on": gt_c - gt_b,
            "d_gt_gated": gt_g - gt_b,
            "d_pred_always_on": pred_hit_c - pred_hit_b,
            "d_pred_gated": pred_hit_g - pred_hit_b,
            "oracle_high_body": o.get("high_body"),
            "oracle_gt_hit_b": o.get("hit_b"),
            "oracle_gt_hit_c": o.get("hit_c"),
            "oracle_d_hit": o.get("d_hit"),
            "segmenter": args.segmenter,
            "person_source": "predicted_person_pixels_not_gt_columns",
        }
        rows.append(rec)
        pack = (rec, sample.image.copy(), pred_map.copy(), b, c)
        if gate_on:
            viz_on.append(pack)
            viz_on.sort(key=lambda t: t[0]["pred_hit_b"], reverse=True)
            viz_on = viz_on[:4]
        else:
            viz_off.append(pack)
            viz_off.sort(key=lambda t: t[0]["pred_hit_b"])
            viz_off = viz_off[:4]
        print(
            f"{len(rows):3d} {item.stem:<8} gate={'ON ' if gate_on else 'off'} "
            f"pred_inst={pred_n_inst:3d} pred_hit_b={pred_hit_b:.3f} "
            f"gt_b={gt_b:.3f} gt_c={gt_c:.3f} gt_g={gt_g:.3f}",
            flush=True,
        )

    on_rows = [r for r in rows if r["gate_on"]]
    off_rows = [r for r in rows if not r["gate_on"]]
    oracle_on = [r for r in rows if r.get("oracle_high_body") is True]
    agree = sum(
        1
        for r in rows
        if r.get("oracle_high_body") is not None
        and bool(r["gate_on"]) == bool(r["oracle_high_body"])
    )
    n_joined = sum(1 for r in rows if r.get("oracle_high_body") is not None)

    summary = {
        "experiment": "practical_predicted_person_gate",
        "not_oracle": True,
        "person_source": (
            "SegFormer/DeepLab person class pixels; connected components for the "
            "gate rate. Not GT heads. Not constructed 12deg body columns."
        ),
        "segmenter": args.segmenter,
        "k": PRIMARY_K,
        "tau": args.tau,
        "lambda_g": 1.0,
        "lambda_p": 1.0,
        "lambda_s_on": LAMBDA_S,
        "n_medium": len(rows),
        "n_gated_on": len(on_rows),
        "n_gated_off": len(off_rows),
        "gt_body_hit_baseline_all": agg_hits(rows, "gt_body_hit_b", "gt_body_hit_b"),
        "gt_body_hit_always_on_semantic": agg_hits(
            rows, "gt_body_hit_b", "gt_body_hit_c_always_on"
        ),
        "gt_body_hit_predicted_gate": agg_hits(
            rows, "gt_body_hit_b", "gt_body_hit_gated"
        ),
        "gt_body_hit_predicted_gate_ON_subset": agg_hits(
            on_rows, "gt_body_hit_b", "gt_body_hit_gated"
        ),
        "gt_body_hit_predicted_gate_OFF_subset": agg_hits(
            off_rows, "gt_body_hit_b", "gt_body_hit_gated"
        ),
        "pred_instance_hit_always_on": agg_hits(rows, "pred_hit_b", "pred_hit_c"),
        "pred_instance_hit_gate": agg_hits(rows, "pred_hit_b", "pred_hit_gated"),
        "oracle_join": {
            "n_with_oracle_row": n_joined,
            "n_agree_gate_vs_oracle_high_body": agree,
            "oracle_json": str(args.oracle_json),
            "oracle_high_body_subset_gt": agg_hits(
                oracle_on, "gt_body_hit_b", "gt_body_hit_c_always_on"
            ),
        },
        "note": (
            "Do not report these numbers as the C2b oracle -35.2pp. "
            "Predicted person pixels are not GT body columns."
        ),
    }

    print()
    print(f"N medium={summary['n_medium']}  gate ON={summary['n_gated_on']}  OFF={summary['n_gated_off']}")
    for name in (
        "gt_body_hit_always_on_semantic",
        "gt_body_hit_predicted_gate",
        "gt_body_hit_predicted_gate_ON_subset",
        "gt_body_hit_predicted_gate_OFF_subset",
    ):
        s = summary[name]
        d = s["delta"]
        print(
            f"{name}: n={s['n']}  before_mean={s['hit_before']['mean']:.4f}  "
            f"after_mean={s['hit_after']['mean']:.4f}  "
            f"d_mean={d['mean']:+.4f}  median={d['median']:+.4f}  "
            f"ci95=[{d['ci95_lo']:+.4f},{d['ci95_hi']:+.4f}]  "
            f"(c or g)<b {s['n_after_lt_before']}/{s['n']}"
        )
    if n_joined:
        print(
            f"gate vs oracle high-body agree {agree}/{n_joined} "
            "(different detectors; not a GT-equivalence test)"
        )

    chosen = (viz_on + viz_off)[:N_VIZ]

    for rec, image, pred_map, b, c in chosen:
        tint = overlay_protected_classes(image, pred_map, groups=GROUPS, alpha=0.5)
        mapped_b = b["mapped"][:PRIMARY_K]
        mapped_g = (c["mapped"] if rec["gate_on"] else b["mapped"])[:PRIMARY_K]
        seams_b = overlay_seams(image, mapped_b, color=(255, 32, 32), thickness=0)
        seams_g = overlay_seams(image, mapped_g, color=(32, 200, 64), thickness=0)
        carved = c["carved"] if rec["gate_on"] else b["carved"]
        title = (
            f"stem={rec['stem']} predicted_gate={'ON' if rec['gate_on'] else 'OFF'} "
            f"pred_hit_b={rec['pred_hit_b']:.3f} gt_b={rec['gt_body_hit_b']:.3f} "
            f"gt_gated={rec['gt_body_hit_gated']:.3f} (not GT mask)"
        )
        save_panel(
            viz_dir / f"panel_{rec['stem']}.png",
            image,
            tint,
            seams_b,
            seams_g,
            carved,
            title,
        )
        save_erp(viz_dir / f"{rec['stem']}_erp.png", image)
        save_erp(viz_dir / f"{rec['stem']}_pred.png", tint)
        save_erp(viz_dir / f"{rec['stem']}_seams_b.png", seams_b)
        save_erp(viz_dir / f"{rec['stem']}_seams_gated.png", seams_g)
        save_erp(viz_dir / f"{rec['stem']}_final.png", carved)

    out_json = args.out / "metrics.json"
    out_json.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out_json}")
    print(f"wrote {len(chosen)} panels under {viz_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
