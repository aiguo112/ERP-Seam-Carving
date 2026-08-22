"""Stage E0: non-learned compensated-frame temporal baseline. No Mamba.

1. Independent Stage-B seam (D0).
2. Flow-warped propagation: S_pred = wrap(S_{t-1} + pan), then banded DP.
3. Compensated-frame penalty: E + lambda_t * cyclic(|x - S_pred|). Sweep lambda_t.

Reports J, residual J, and content damage (equatorial spherical path energy
vs independent). Penalty is excluded from the damage score.

Restarts independent after a dropped-frame gap. Contiguous pairs only.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.paths import default_videos  # noqa: E402
from data.video_erp import SampledClip, list_clips, sample_frames  # noqa: E402
from energy.spherical import spherical_gradient_energy  # noqa: E402
from io_erp import save_erp  # noqa: E402
from seam.dp import find_vertical_seam  # noqa: E402
from temporal.damage import content_damage  # noqa: E402
from temporal.jitter import (  # noqa: E402
    compute_jitter_stats,
    equatorial_horizontal_pan,
    photometric_gain,
)
from temporal.propagate import find_temporal_seam, seam_pred  # noqa: E402
from viz import overlay_seam  # noqa: E402

# lambda_t units: spherical energy per pixel of cyclic deviation from S_pred.
DEFAULT_LAMBDAS = (0.005, 0.02, 0.08, 0.3)


@dataclass(frozen=True)
class Config:
    name: str
    lambda_t: float
    band_radius: int | None


def independent_seam(energy: np.ndarray) -> np.ndarray:
    return find_vertical_seam(energy, cyclic=True)


def pair_set(pairs: list[tuple[int, int]]) -> set[tuple[int, int]]:
    return set(pairs)


def carve_temporal(
    energies: list[np.ndarray],
    frames: list[np.ndarray],
    pairs: list[tuple[int, int]],
    cfg: Config,
) -> tuple[list[np.ndarray], list[bool], list[float]]:
    """Causal chain. Restart independent when (t-1, t) is not a contiguous pair."""
    allowed = pair_set(pairs)
    width = int(energies[0].shape[1])
    seams: list[np.ndarray] = []
    used_pred: list[bool] = []
    pans: list[float] = []
    for t, energy in enumerate(energies):
        if t > 0 and (t - 1, t) in allowed:
            pan = equatorial_horizontal_pan(frames[t - 1], frames[t])
            pred = seam_pred(seams[t - 1], pan, width)
            seam = find_temporal_seam(
                energy,
                s_pred=pred,
                lambda_t=cfg.lambda_t,
                band_radius=cfg.band_radius,
            )
            used_pred.append(True)
            pans.append(float(pan))
        else:
            seam = independent_seam(energy)
            used_pred.append(False)
        seams.append(np.asarray(seam, dtype=np.int32))
    return seams, used_pred, pans


def configs_for(lambdas: tuple[float, ...], band: int) -> list[Config]:
    out = [
        Config("ind", 0.0, None),
        Config(f"prop_R{band}", 0.0, band),
    ]
    for lam in lambdas:
        tag = f"{lam:g}".replace(".", "p")
        out.append(Config(f"lt_{tag}", float(lam), None))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compensated-frame temporal baseline (no sequence model)."
    )
    parser.add_argument("--videos", nargs="*", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--max-width", type=int, default=512)
    parser.add_argument("--band", type=int, default=16)
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="*",
        default=list(DEFAULT_LAMBDAS),
        help="lambda_t sweep (energy units per px from S_pred).",
    )
    parser.add_argument("--out", type=Path, default=Path("output") / "temporal_baseline")
    args = parser.parse_args(argv)

    paths = list(args.videos) if args.videos else default_videos()
    if not paths:
        raise SystemExit("no videos found; pass --videos PATHS (videos are not in the repo)")
    clips = [list_clips(p)[0] for p in paths]
    cfgs = configs_for(tuple(float(x) for x in args.lambdas), int(args.band))
    args.out.mkdir(parents=True, exist_ok=True)

    print("content damage: D = P_eq(temporal) / P_eq(independent) on spherical energy")
    print("  (temporal penalty excluded; polar rows excluded; only frames with S_pred)")
    print("penalty is lambda_t * cyclic(|x - S_pred|), never |x - S_{t-1}|")
    print()

    all_rows: list[dict] = []
    for clip in clips:
        sampled: SampledClip = sample_frames(
            clip.path, max_frames=args.frames, max_width=args.max_width
        )
        frames = sampled.frames
        pairs = sampled.contiguous_pairs()
        w0 = int(frames[0].shape[1])
        print(
            f"=== {clip.name}  keep={len(frames)}/{sampled.n_attempted}  "
            f"drop={sampled.n_dropped}  pairs={len(pairs)} ==="
        )
        energies = [spherical_gradient_energy(img) for img in frames]
        mean_e = float(np.mean([float(e.mean()) for e in energies]))
        print(f"  mean spherical energy={mean_e:.4f}  (lambda_t relative to this scale)")

        seams_ind = [independent_seam(e) for e in energies]
        stats_ind = compute_jitter_stats(frames, seams_ind, w0, pairs)
        pans = [
            equatorial_horizontal_pan(frames[i], frames[j]) for i, j in pairs
        ]
        m_raw, m_wrp, gain = photometric_gain(frames, pairs, pans)
        print(
            f"  pan |mean|={stats_ind.pan_mean_abs:.2f}  "
            f"photo raw={m_raw:.2f} warped={m_wrp:.2f}  warped/raw={gain:.3f}"
        )
        if gain < 0.95:
            print("  pan check: warp reduces photometric residual (yaw estimate is tracking).")
        elif gain > 1.05:
            print(
                "  pan check: warp INCREASES photometric residual "
                "(estimate is not a clean yaw; Jres on this clip is contaminated)."
            )
        else:
            print("  pan check: warp is near-neutral (weak or non-yaw motion).")
        print(
            f"  {'cfg':<12} {'J':>8} {'Jres':>8} {'|pan|':>8} {'D':>7} {'Peq_d':>8} {'hit80':>14}"
        )

        clip_rows = []
        for cfg in cfgs:
            if cfg.name == "ind":
                seams = seams_ind
                used = [True] * len(frames)
            else:
                seams, used, _pans_chain = carve_temporal(energies, frames, pairs, cfg)
            stats = compute_jitter_stats(frames, seams, w0, pairs)
            dmg = content_damage(energies, seams_ind, seams, used)
            hit = f"{dmg.hit_ind:.3f}->{dmg.hit_tmp:.3f}"
            print(
                f"  {cfg.name:<12} {stats.J:8.2f} {stats.J_residual:8.2f} "
                f"{stats.pan_mean_abs:8.2f} {dmg.damage_ratio:7.3f} "
                f"{dmg.equator_energy_tmp - dmg.equator_energy_ind:+8.4f} {hit:>14}"
            )
            row = {
                "clip": clip.name,
                "cfg": cfg.name,
                "lambda_t": cfg.lambda_t,
                "band": cfg.band_radius,
                "J": stats.J,
                "Jres": stats.J_residual,
                "pan": stats.pan_mean_abs,
                "M": stats.M,
                "pairs": stats.n_pairs,
                "photo_raw": m_raw,
                "photo_warped": m_wrp,
                "photo_gain": gain,
                "damage_ratio": dmg.damage_ratio,
                "Peq_ind": dmg.equator_energy_ind,
                "Peq_tmp": dmg.equator_energy_tmp,
                "hit_ind": dmg.hit_ind,
                "hit_tmp": dmg.hit_tmp,
                "n_pred_frames": dmg.n_frames,
                "mean_energy": mean_e,
                "keep": len(frames),
                "drop": sampled.n_dropped,
            }
            clip_rows.append(row)
            all_rows.append(row)

            if cfg.name in {"ind", f"prop_R{args.band}"} or (
                cfg.lambda_t == max(float(x) for x in args.lambdas)
            ):
                ov = overlay_seam(frames[len(frames) // 2], seams[len(frames) // 2], thickness=1)
                save_erp(args.out / f"{clip.name}_{cfg.name}_mid.png", ov)

        ind = next(r for r in clip_rows if r["cfg"] == "ind")
        # Do not crown max-lambda: it always zeros Jres by locking to S_pred.
        print(
            f"  vs ind J={ind['J']:.2f} Jres={ind['Jres']:.2f}. "
            "Read D with Jres: large lambda can zero flicker by cutting content."
        )
        prop = next((r for r in clip_rows if r["cfg"].startswith("prop_")), None)
        if prop is not None:
            print(
                f"  prop: Jres {ind['Jres']:.2f}->{prop['Jres']:.2f}  "
                f"D={prop['damage_ratio']:.3f}"
            )
        print()

    out_json = args.out / "metrics.json"
    out_json.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"wrote {out_json}")
    print("No Mamba/Transformer. If residual J is already gone, stop here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
