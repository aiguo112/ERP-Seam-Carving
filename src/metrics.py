"""Ablation metrics: protected-class hit rates (void excluded) + polar routing."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from energy.semantic import count_protected_hits, count_valid_seam_pixels


def hit_percentages(
    hits: Mapping[str, int],
    n_valid: int,
) -> dict[str, float]:
    """Seam-object intersection % of non-void seam pixels."""
    if n_valid <= 0:
        return {k: 0.0 for k in hits}
    return {k: 100.0 * v / n_valid for k, v in hits.items()}


def accumulate_hits(
    seam: np.ndarray,
    class_map: np.ndarray,
    groups: dict[str, frozenset[int]],
    void_mask: np.ndarray | None,
    bucket: dict[str, int],
) -> int:
    """Add this seam's protected hits into ``bucket``. Returns valid pixel count."""
    for key, value in count_protected_hits(seam, class_map, groups, void_mask).items():
        bucket[key] = bucket.get(key, 0) + value
    return count_valid_seam_pixels(seam, void_mask)


def format_table(
    rows: list[dict],
    group_names: list[str],
) -> str:
    """Plain-text ablation table."""
    cols = ["config", "polar_route"] + [f"{g}%" for g in group_names] + ["valid_px"]
    widths = {c: max(len(c), 12) for c in cols}
    for row in rows:
        widths["config"] = max(widths["config"], len(str(row["config"])))
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    lines = [header, "-" * len(header)]
    for row in rows:
        cells = [
            str(row["config"]).ljust(widths["config"]),
            f"{row['polar_route']:.3f}".ljust(widths["polar_route"]),
        ]
        pct = row["pct"]
        for g in group_names:
            cells.append(f"{pct.get(g, 0.0):6.2f}".ljust(widths[f"{g}%"]))
        cells.append(str(row["valid_px"]).ljust(widths["valid_px"]))
        lines.append("  ".join(cells))
    return "\n".join(lines)


def semantic_delta_line(
    pct_grad: Mapping[str, float],
    pct_full: Mapping[str, float],
    group_names: list[str],
    *,
    vs: str = "gradient-only",
) -> str:
    """Honest one-liner: does (c) beat the named baseline on protected-class hits?"""
    bits = []
    reduced = 0
    for g in group_names:
        d = pct_full.get(g, 0.0) - pct_grad.get(g, 0.0)
        bits.append(f"{g} {d:+.2f} pp")
        if d < -0.05:
            reduced += 1
    summary = ", ".join(bits)
    if reduced == 0:
        return (
            f"semantic vs {vs}: {summary}. "
            "No measurable reduction in protected-object hits on this image."
        )
    return (
        f"semantic vs {vs}: {summary}. "
        "Semantic term reduced hits on at least one protected class "
        "(beyond that baseline)."
    )
