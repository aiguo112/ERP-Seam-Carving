"""C1 polar-texture correlation: does routing delta appear when poles have content?

Primary predictor (locked): polar_ratio = mean unweighted Sobel in the polar
15% bands / mean unweighted Sobel on the equator, measured on the same
resized image the C1 carve used. That is the energy the naive baseline sees.

Prediction: polar_ratio correlates positively with d_route. High polar
texture is the content condition that breaks spherical redundancy.

Support bar (locked, not after seeing rho):
  supported -- Spearman rho(polar_ratio, d_route) >= 0.30 AND top tercile
               mean d_route > bottom tercile mean d_route
  weak      -- rho > 0 but below 0.30, or terciles ordered without the rho bar
  fail      -- rho <= 0 (texture does not predict the spherical effect)

Secondary (reported, not the claim): polar_ratio excluding near-black caps,
and route_plain as a mediator (already-cheap poles already wander).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data.cvrg_pano import list_panoramas, polar_black_void  # noqa: E402
from energy.gradient import gradient_energy  # noqa: E402
from energy.spherical import polar_energy_content  # noqa: E402
from io_erp import load_erp  # noqa: E402

from run_b_cvrg_set import CVRG_ROOT, POLAR_BAND, resize_max_width  # noqa: E402

BIG_PP = 0.01  # the 19% tail from C1 (>1 pp routing delta)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(np.asarray(x, dtype=np.float64)))
    ry = np.argsort(np.argsort(np.asarray(y, dtype=np.float64)))
    return float(np.corrcoef(rx.astype(np.float64), ry.astype(np.float64))[0, 1])


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))[0, 1])


def tercile_ids(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = np.percentile(x, [100.0 / 3.0, 200.0 / 3.0])
    return x <= lo, (x > lo) & (x < hi), x >= hi


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Correlate C1 d_route with polar texture.")
    parser.add_argument("--metrics", type=Path, default=Path("output") / "spherical_set" / "metrics.json")
    parser.add_argument("--cvrg-root", type=Path, default=CVRG_ROOT)
    parser.add_argument("--max-width", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("output") / "spherical_set")
    args = parser.parse_args(argv)

    blob = json.loads(args.metrics.read_text(encoding="utf-8"))
    by_stem = {r["stem"]: r for r in blob["rows"]}
    panos = list_panoramas(args.cvrg_root, "val") + list_panoramas(args.cvrg_root, "test")
    panos = [p for p in panos if p.stem in by_stem]
    if len(panos) != len(by_stem):
        missing = sorted(set(by_stem) - {p.stem for p in panos})
        raise SystemExit(f"metrics stems not on disk: {missing[:8]}")

    rows: list[dict] = []
    for pano in panos:
        rec = by_stem[pano.stem]
        image = resize_max_width(load_erp(pano.rgb_path), args.max_width)
        energy = gradient_energy(image)
        all_pix = polar_energy_content(energy, POLAR_BAND)
        void = polar_black_void(image, band=POLAR_BAND)
        novoid = polar_energy_content(energy, POLAR_BAND, polar_valid=~void)
        rows.append(
            {
                **rec,
                "polar_grad": all_pix["polar_grad"],
                "eq_grad": all_pix["eq_grad"],
                "polar_ratio": all_pix["polar_ratio"],
                "polar_ratio_novoid": novoid["polar_ratio"],
                "polar_grad_novoid": novoid["polar_grad"],
            }
        )

    ratio = np.array([r["polar_ratio"] for r in rows])
    ratio_nv = np.array([r["polar_ratio_novoid"] for r in rows])
    polar_g = np.array([r["polar_grad"] for r in rows])
    d = np.array([r["d_route"] for r in rows])
    rp = np.array([r["route_plain"] for r in rows])
    big = d > BIG_PP

    rho = spearman(ratio, d)
    r_pear = pearson(ratio, d)
    rho_nv = spearman(ratio_nv, d)
    rho_abs = spearman(polar_g, d)
    rho_plain = spearman(ratio, rp)
    rho_med = spearman(rp, d)

    lo, mid, hi = tercile_ids(ratio)
    d_lo, d_mid, d_hi = float(d[lo].mean()), float(d[mid].mean()), float(d[hi].mean())
    frac_lo = float(big[lo].mean())
    frac_mid = float(big[mid].mean())
    frac_hi = float(big[hi].mean())

    print(
        f"n={len(rows)}  primary polar_ratio = polar Sobel / eq Sobel  "
        f"(unweighted, same W<={args.max_width} as C1)"
    )
    print(
        f"Spearman polar_ratio vs d_route     {rho:+.3f}  "
        f"Pearson {r_pear:+.3f}"
    )
    print(f"Spearman polar_grad vs d_route      {rho_abs:+.3f}  (absolute, not ratio)")
    print(f"Spearman polar_ratio_novoid vs d    {rho_nv:+.3f}  (drop near-black caps)")
    print(
        f"mediator: polar_ratio vs route_plain {rho_plain:+.3f}  "
        f"(expect negative: textured poles, naive won't wander)"
    )
    print(
        f"mediator: route_plain vs d_route     {rho_med:+.3f}  "
        f"(expect negative: already-wandering baseline, little left to gain)"
    )
    print()
    print(
        f"polar_ratio terciles  n lo/mid/hi={int(lo.sum())}/{int(mid.sum())}/{int(hi.sum())}"
    )
    print(
        f"  mean d_route   low={d_lo:+.4f}  mid={d_mid:+.4f}  high={d_hi:+.4f}"
    )
    print(
        f"  frac d>1pp     low={frac_lo:.3f}  mid={frac_mid:.3f}  high={frac_hi:.3f}"
    )
    print(
        f"d>1pp images ({int(big.sum())}/{len(rows)}): polar_ratio mean "
        f"{ratio[big].mean():.3f} vs rest {ratio[~big].mean():.3f}"
    )

    if rho >= 0.30 and d_hi > d_lo:
        verdict = "supported"
        note = (
            "C1 is a predicted effect: spherical routing gain appears when "
            "poles have texture relative to the equator."
        )
    elif rho > 0.0 and d_hi > d_lo:
        verdict = "weak"
        note = (
            "Sign is as predicted but rho is below the 0.30 bar. Do not "
            "headline 'effect appears exactly when poles have texture'."
        )
    else:
        verdict = "fail"
        note = (
            "Polar texture does not predict the routing delta. Keep C1 as "
            "a small mean effect, not a content-conditioned result."
        )
    print()
    print(f"VERDICT {verdict}: {note}")

    args.out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.scatter(ratio[~big], d[~big] * 100.0, s=18, c="#4c6ef5", alpha=0.75, label="d <= 1pp")
    ax.scatter(ratio[big], d[big] * 100.0, s=22, c="#e03131", alpha=0.85, label="d > 1pp")
    ax.axhline(0.0, color="#868e96", lw=0.8)
    ax.set_xlabel("polar / equatorial Sobel (unweighted)")
    ax.set_ylabel("routing share delta (pp)")
    ax.set_title(f"C1 polar texture vs routing delta  Spearman={rho:+.2f}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / "polar_texture_scatter.png", dpi=140)
    plt.close(fig)

    summary = {
        "n": len(rows),
        "spearman_polar_ratio_d": rho,
        "pearson_polar_ratio_d": r_pear,
        "spearman_polar_grad_d": rho_abs,
        "spearman_polar_ratio_novoid_d": rho_nv,
        "spearman_polar_ratio_route_plain": rho_plain,
        "spearman_route_plain_d": rho_med,
        "tercile_d_mean": {"low": d_lo, "mid": d_mid, "high": d_hi},
        "tercile_frac_d_gt_1pp": {"low": frac_lo, "mid": frac_mid, "high": frac_hi},
        "big_n": int(big.sum()),
        "polar_ratio_big": float(ratio[big].mean()) if big.any() else None,
        "polar_ratio_rest": float(ratio[~big].mean()),
        "verdict": verdict,
        "note": note,
        "support_bar": "Spearman>=0.30 and high tercile mean d > low tercile",
    }
    (args.out / "polar_texture.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.out / 'polar_texture.json'}")
    print(f"wrote {args.out / 'polar_texture_scatter.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
