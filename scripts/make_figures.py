#!/usr/bin/env python3
"""Regenerate the quantitative figures in README.md from results/metrics/*.json.

No dataset, no GPU and no model inference is needed: every JSON under results/
is the per-image metrics dump of one experiment, and all charts below are
recomputed from those rows exactly as reported in the paper.

    python scripts/make_figures.py            # writes figures/fig_*.png

Photo composites (fig_method_overview.png, fig_head_body.jpg,
fig_qualitative.jpg, fig_temporal_control.jpg) are static assets under
figures/ and are not touched here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)

BLUE = "#2E5EAA"
RUST = "#C8553D"
SAGE = "#5E8C61"
GREY = "#8A8A8A"
LIGHT = "#B9C6DA"


def load(name: str) -> dict:
    return json.loads((METRICS / name).read_text(encoding="utf-8"))


def paired_ci(deltas: list[float]) -> tuple[float, float, float]:
    """Mean and normal-approximation 95% CI of a paired difference (paper Eq. 7)."""
    m, s, n = mean(deltas), stdev(deltas), len(deltas)
    half = 1.96 * s / math.sqrt(n)
    return m, m - half, m + half


# --------------------------------------------------------------------------- #
# RQ1: does spherical energy change where seams go?
# --------------------------------------------------------------------------- #
def fig_routing() -> None:
    rows = load("spherical_set_metrics.json")["rows"]
    plain = [r["route_plain"] for r in rows]
    sph = [r["route_sph"] for r in rows]
    d = np.array([r["d_route"] for r in rows]) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))

    ax = axes[0]
    xs = [0, 1]
    vals = [mean(plain), mean(sph)]
    errs = [stdev(plain), stdev(sph)]
    ax.bar(
        xs, vals, yerr=errs, width=0.55, color=[GREY, BLUE],
        edgecolor="white", error_kw={"elinewidth": 1.2, "capsize": 4, "ecolor": "#333"},
    )
    for x, v, e in zip(xs, vals, errs):
        ax.text(x, v + e + 0.002, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(xs, ["Sobel\n(≡ plain)", "Sobel × cos φ\n(spherical)"])
    ax.set_ylim(0.28, 0.315)
    ax.set_ylabel("Polar routing share")
    ax.set_title(f"Routing: Δ = {d.mean():+.2f} pp (n={len(rows)})", loc="left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    ax = axes[1]
    up, down = int((d > 0).sum()), int((d < 0).sum())
    ax.hist(d, bins=24, color=LIGHT, edgecolor="white")
    ax.axvline(0, color=RUST, linewidth=1.1, linestyle="--")
    ax.axvline(d.mean(), color=BLUE, linewidth=1.1)
    ax.set_xlabel("Per-image Δ polar routing (pp)")
    ax.set_ylabel("Images")
    ax.set_title(f"{up}/{len(d)} up, {down}/{len(d)} down", loc="left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIG / "fig_routing.png")
    plt.close(fig)


def fig_polar_texture() -> None:
    data = load("spherical_set_polar_texture.json")
    rows = data["rows"]
    rho = data["summary"]["spearman_polar_ratio_d"]
    x = np.array([r["polar_ratio"] for r in rows])
    y = np.array([r["d_route"] for r in rows]) * 100.0
    big = y > 1.0

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.scatter(x[~big], y[~big], s=16, color=BLUE, alpha=0.6, linewidths=0)
    ax.scatter(x[big], y[big], s=22, color=RUST, alpha=0.95, linewidths=0, label="Δ > +1 pp")
    ax.axhline(0, color="#888", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Unweighted polar / equator Sobel ratio")
    ax.set_ylabel("Δ polar routing (pp)")
    ax.set_title(f"Texture does not predict the shift (Spearman ρ = {rho:.2f})", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "fig_polar_texture.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# RQ2: when does semantic protection matter?
# --------------------------------------------------------------------------- #
def fig_head_hits() -> None:
    rows = [r for r in load("panocrowd_metrics.json")["rows"]]
    bins = ["sparse", "medium", "dense"]
    names = {"sparse": "sparse\n0–49", "medium": "medium\n50–199", "dense": "dense\n200+"}

    b_vals, c_vals, covers, ns, deltas = [], [], [], [], []
    for b in bins:
        sub = [r for r in rows if r["bin"] == b]
        b_vals.append(mean([r["hit_b_k24"] for r in sub]) * 100)
        c_vals.append(mean([r["hit_c_k24"] for r in sub]) * 100)
        covers.append(mean([r["cover"] for r in sub]) * 100)
        deltas.append(mean([r["d_k24"] for r in sub]) * 100)
        ns.append(len(sub))

    x = np.arange(len(bins))
    w = 0.36
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.bar(x - w / 2, b_vals, w, color=GREY, label="(b) spherical baseline")
    ax.bar(x + w / 2, c_vals, w, color=SAGE, label="(c) always-on 2° head disks")
    for xi, bv, cv, d, n in zip(x, b_vals, c_vals, deltas, ns):
        ax.text(xi - w / 2, bv + 0.08, f"{bv:.1f}%", ha="center", fontsize=7.5)
        ax.text(xi + w / 2, cv + 0.08, f"{cv:.1f}%", ha="center", fontsize=7.5)
        ax.text(xi, max(bv, cv) + 0.55, f"Δ {d:+.2f} pp\nn={n}", ha="center", fontsize=7.2)
    ax.set_xticks(x, [names[b] for b in bins])
    ax.set_ylim(0, 6.2)
    ax.set_ylabel("GT 2° head-hit (%)")
    ax.set_title("Always-on head protection: little work left to do", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "fig_head_hits.png")
    plt.close(fig)


def fig_body_hits() -> None:
    body = [r for r in load("panocrowd_body_energy_metrics.json")["rows"] if r["bin"] == "medium"]
    hi = [r for r in body if r["high_body"]]
    comp = [r for r in body if not r["high_body"]]

    gate = load("panocrowd_practical_gate_metrics.json")["rows"]
    on = [r for r in gate if r["gate_on"]]
    off = [r for r in gate if not r["gate_on"]]

    def pair(sub, before, after):
        return (
            mean([r[before] for r in sub]) * 100,
            mean([r[after] for r in sub]) * 100,
        )

    groups_oracle = [
        ("high-body\n(n=49)", pair(hi, "hit_b_k24", "hit_c_k24")),
        ("complement\n(n=122)", pair(comp, "hit_b_k24", "hit_c_k24")),
    ]
    groups_gate = [
        ("all medium\n(n=171)", pair(gate, "gt_body_hit_b", "gt_body_hit_gated")),
        ("gate ON\n(n=56)", pair(on, "gt_body_hit_b", "gt_body_hit_gated")),
        ("gate OFF\n(n=115)", pair(off, "gt_body_hit_b", "gt_body_hit_gated")),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0))
    for ax, groups, title in zip(
        axes,
        [groups_oracle, groups_gate],
        ["Oracle (GT body columns)", "Practical (SegFormer-B0 gate)"],
    ):
        x = np.arange(len(groups))
        w = 0.36
        before = [g[1][0] for g in groups]
        after = [g[1][1] for g in groups]
        ax.bar(x - w / 2, before, w, color=GREY, label="before (b)")
        ax.bar(x + w / 2, after, w, color=SAGE, label="after (gated / c)")
        for xi, bv, av in zip(x, before, after):
            ax.text(xi - w / 2, bv + 0.35, f"{bv:.1f}", ha="center", fontsize=7.5)
            ax.text(xi + w / 2, av + 0.35, f"{av:.1f}", ha="center", fontsize=7.5)
            ax.text(xi, max(bv, av) + 2.4, f"{av - bv:+.1f} pp", ha="center", fontsize=7.5)
        ax.set_xticks(x, [g[0] for g in groups])
        ax.set_ylabel("GT body-hit (%)")
        ax.set_title(title, loc="left")
        ax.set_ylim(0, max(before) * 1.32)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_body_hits.png")
    plt.close(fig)


def fig_gate() -> None:
    gate = load("panocrowd_practical_gate_metrics.json")["rows"]
    on = [r for r in gate if r["gate_on"]]

    # Per-image change on the activated subset, sorted: the paper's "34 down / 5 tie / 17 up".
    d = np.array([(r["gt_body_hit_gated"] - r["gt_body_hit_b"]) * 100 for r in on])
    order = np.argsort(d)
    d_sorted = d[order]
    down, tie, up = int((d < -1e-9).sum()), int((np.abs(d) <= 1e-9).sum()), int((d > 1e-9).sum())

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), gridspec_kw={"width_ratios": [2.0, 1.0]})

    ax = axes[0]
    colors = [SAGE if v < -1e-9 else (RUST if v > 1e-9 else GREY) for v in d_sorted]
    ax.bar(np.arange(len(d_sorted)), d_sorted, color=colors, width=0.9, linewidth=0)
    ax.axhline(0, color="#333", linewidth=0.9)
    ax.set_xlabel("Gate-ON images (sorted by Δ)")
    ax.set_ylabel("Δ GT body-hit (pp)")
    ax.set_title(f"Gate ON subset: {down} down · {tie} tie · {up} up", loc="left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)

    # Agreement between the predicted gate and the oracle high-body set.
    oh_on = sum(1 for r in on if r["oracle_high_body"])
    oh_off = sum(1 for r in gate if r["oracle_high_body"] and not r["gate_on"])
    n_on = len(on)
    n_off = len(gate) - n_on
    mat = np.array([[oh_on, oh_off], [n_on - oh_on, n_off - oh_off]])

    ax = axes[1]
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=11,
                    color="white" if mat[i, j] > mat.max() * 0.6 else "#222")
    ax.set_xticks([0, 1], ["gate ON", "gate OFF"])
    ax.set_yticks([0, 1], ["oracle high", "not high"])
    ax.set_title("Gate vs. oracle set", loc="left")
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_xticks(np.arange(-0.5, 2), minor=True)
    ax.set_yticks(np.arange(-0.5, 2), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    fig.tight_layout()
    fig.savefig(FIG / "fig_gate.png")
    plt.close(fig)


def main() -> None:
    fig_routing()
    fig_polar_texture()
    fig_head_hits()
    fig_body_hits()
    fig_gate()
    for p in sorted(FIG.glob("fig_*.png")):
        print(f"{p.relative_to(ROOT)}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
