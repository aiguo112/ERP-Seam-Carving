#!/usr/bin/env python3
"""Recompute Tables I-III of the paper from results/metrics/*.json.

Nothing is trained and no dataset is touched: each JSON holds the per-image
metrics of one experiment, so every aggregate can be rebuilt and compared with
the values printed in the manuscript.

    python scripts/reproduce_tables.py

Output is written as markdown so it can be pasted straight into a docs page.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"


def load(name: str) -> dict:
    return json.loads((METRICS / name).read_text(encoding="utf-8"))


def paired_ci(deltas: list[float]) -> tuple[float, float, float]:
    """Mean and normal-approximation 95% CI of a paired difference (paper Eq. 7)."""
    m, s, n = mean(deltas), stdev(deltas), len(deltas)
    half = 1.96 * s / math.sqrt(n)
    return m, m - half, m + half


def pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}"


def table_rq1() -> None:
    rows = load("spherical_set_metrics.json")["rows"]
    rp = [r["route_plain"] for r in rows]
    rs = [r["route_sph"] for r in rows]
    d = [r["d_route"] for r in rows]
    occ = max(abs(r["occ_plain"] - r["occ_sph"]) for r in rows)
    up = sum(1 for x in d if x > 0)
    down = sum(1 for x in d if x < 0)

    print("## Table I - polar routing, CVRG-Pano val+test (n=%d, k=24)\n" % len(rows))
    print("| Method | Polar occupancy | Routing share | Mean polar step |")
    print("|---|---|---|---|")
    print(f"| Plain Sobel, cyclic | {mean([r['occ_plain'] for r in rows]):.3f} | "
          f"{mean(rp):.4f} +/- {stdev(rp):.4f} | {mean([r['step_plain'] for r in rows]):.3f} |")
    print(f"| Sobel x cos(phi), cyclic | {mean([r['occ_sph'] for r in rows]):.3f} | "
          f"{mean(rs):.4f} +/- {stdev(rs):.4f} | {mean([r['step_sph'] for r in rows]):.3f} |")
    print(f"| Paired delta | 0.000 | {mean(d):+.4f} +/- {stdev(d):.4f} | "
          f"{mean([r['d_step'] for r in rows]):+.3f} |")
    print(f"\nRouting delta = {mean(d) * 100:+.2f} pp; {up}/{len(d)} up, {down}/{len(d)} down; "
          f"{sum(1 for x in d if x > 0.01)}/{len(d)} images move by more than +1 pp.\n")


def table_rq2a() -> None:
    rows = load("panocrowd_metrics.json")["rows"]
    print("## Table II - always-on 2 deg GT head disks, PanoCount val+test (k=24)\n")
    print("| Bin | n | (b) head-hit | (c) head-hit | delta (pp) | (c)<(b) | eq. cover |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for b in ("sparse", "medium", "dense"):
        sub = [r for r in rows if r["bin"] == b]
        hb = [r["hit_b_k24"] for r in sub]
        hc = [r["hit_c_k24"] for r in sub]
        dd = [r["d_k24"] for r in sub]
        print(f"| {b} | {len(sub)} | {pct(mean(hb), 1)}% | {pct(mean(hc), 1)}% | "
              f"{mean(dd) * 100:+.2f} | {sum(1 for x in dd if x < 0)}/{len(sub)} | "
              f"{pct(mean([r['cover'] for r in sub]), 1)}% |")
    med = [r for r in rows if r["bin"] == "medium"]
    zeros = sum(1 for r in med if r["hit_b_k24"] == 0 and r["hit_c_k24"] == 0)
    print(f"\nMedium: {zeros}/{len(med)} frames are 0 -> 0; k=48 delta "
          f"{mean([r['d_k48'] for r in med]) * 100:+.2f} pp.\n")


def table_rq2b() -> None:
    body = [r for r in load("panocrowd_body_energy_metrics.json")["rows"] if r["bin"] == "medium"]
    hi = [r for r in body if r["high_body"]]
    comp = [r for r in body if not r["high_body"]]
    gate = load("panocrowd_practical_gate_metrics.json")["rows"]
    on = [r for r in gate if r["gate_on"]]
    off = [r for r in gate if not r["gate_on"]]

    def row(label: str, sub: list[dict], before: str, after: str, n_on: int, ci: bool) -> None:
        b = [r[before] for r in sub]
        a = [r[after] for r in sub]
        m, lo, hi_ = paired_ci([y - x for x, y in zip(b, a)])
        ci_txt = f"[{lo * 100:+.2f}, {hi_ * 100:+.2f}]" if ci else "-"
        print(f"| {label} | {len(sub)} | {n_on} | {pct(mean(b))}% | {pct(mean(a))}% | "
              f"{m * 100:+.2f} | {ci_txt} |")

    print("## Table III(A) - evaluation on constructed GT body columns (PanoCount medium, k=24)\n")
    print("| Method | n | Gate ON | Before | After | delta (pp) | 95% CI |")
    print("|---|---:|---:|---:|---:|---:|---|")
    row("Oracle high-body", hi, "hit_b_k24", "hit_c_k24", len(hi), False)
    row("Oracle complement", comp, "hit_b_k24", "hit_c_k24", 0, False)
    row("Practical gate, all medium", gate, "gt_body_hit_b", "gt_body_hit_gated", len(on), True)
    row("Practical gate, ON subset", on, "gt_body_hit_b", "gt_body_hit_gated", len(on), True)
    row("Practical gate, OFF subset", off, "gt_body_hit_b", "gt_body_hit_gated", 0, False)

    print("\n## Table III(B) - evaluation on predicted-person instances\n")
    print("| Method | n | Before | After | delta (pp) |")
    print("|---|---:|---:|---:|---:|")
    pb = [r["pred_hit_b"] for r in gate]
    pc = [r["pred_hit_c"] for r in gate]
    print(f"| Always-on predicted-person | {len(gate)} | {pct(mean(pb))}% | {pct(mean(pc))}% | "
          f"{(mean(pc) - mean(pb)) * 100:+.2f} |")

    d = [(r["gt_body_hit_gated"] - r["gt_body_hit_b"]) * 100 for r in on]
    down = sum(1 for x in d if x < -1e-9)
    tie = sum(1 for x in d if abs(x) <= 1e-9)
    up = sum(1 for x in d if x > 1e-9)
    oh = [r for r in gate if r["oracle_high_body"]]
    caught = sum(1 for r in on if r["oracle_high_body"])
    print(f"\nGate ON subset: {down} down / {tie} tie / {up} up (n={len(on)}).")
    print(f"Gate vs. oracle high-body: catches {caught}/{len(oh)}, misses {len(oh) - caught}, "
          f"fires on {len(on) - caught} images outside the oracle set.\n")


def main() -> None:
    print("# Recomputed tables (from results/metrics/*.json)\n")
    table_rq1()
    table_rq2a()
    table_rq2b()
    print("All values above should match the manuscript exactly; they are computed from the "
          "same per-image dumps the paper's tables were built from.")


if __name__ == "__main__":
    main()
