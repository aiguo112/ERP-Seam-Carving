"""Finer-stride motion check: does a validated fast-pan clip exist?

Linspace-36 on ERP2 was ~232 source frames (~7.7s). That aliases phase
correlation. This sweep uses a mid-clip window at source strides 1, 2, 4, 8
(and the original-scale 32 as a negative control).

Certified validated_fast only if:
  |pan - dense_u| <= 8 px, pan-warp photometric residual <= 0.85 of raw,
  leftover v is not tilt-scale, and |pan| or |dense_u| >= 8 px at W=512.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.paths import default_videos  # noqa: E402
from data.video_erp import list_clips, sample_strided_frames  # noqa: E402
from temporal.flow import certify_motion, pair_flow_report, summarize_clip  # noqa: E402

from data.paths import default_videos  # noqa: E402

STRIDES = (1, 2, 4, 8, 32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stride sweep for pan vs dense flow.")
    parser.add_argument("--videos", nargs="*", type=Path, default=None)
    parser.add_argument("--n-frames", type=int, default=16)
    parser.add_argument("--max-width", type=int, default=512)
    parser.add_argument("--start-frac", type=float, default=0.45)
    parser.add_argument("--out", type=Path, default=Path("output") / "stride_sweep")
    args = parser.parse_args(argv)

    paths = list(args.videos) if args.videos else default_videos()
    if not paths:
        raise SystemExit("no videos found; pass --videos PATHS (videos are not in the repo)")
    args.out.mkdir(parents=True, exist_ok=True)

    print("certify validated_fast: |pan-dense_u|<=8px, photo_pan<=0.85 raw,")
    print("  leftover v small, |shift|>=8px at W=512")
    print("linspace-36 ERP2 was stride ~232 frames; 32 is a sanity alias check")
    print()
    print(
        f"{'clip':<10} {'stride':>6} {'cert':<16} {'|pan|':>7} {'u':>7} {'v':>7} "
        f"{'|p-u|':>7} {'photP':>6} {'photD':>6}"
    )
    print("-" * 88)

    rows: list[dict] = []
    for path in paths:
        clip = list_clips(path)[0]
        for stride in STRIDES:
            sampled = sample_strided_frames(
                clip.path,
                stride=int(stride),
                n_frames=args.n_frames,
                start_frac=args.start_frac,
                max_width=args.max_width,
            )
            pairs = sampled.contiguous_pairs()
            reports = [
                pair_flow_report(sampled.frames[a], sampled.frames[b]) for a, b in pairs
            ]
            verdict = summarize_clip(reports)
            cert, why = certify_motion(verdict)
            dt = (stride / max(sampled.fps, 1e-6))
            print(
                f"{clip.name:<10} {stride:6d} {cert:<16} {verdict.pan_mean_abs:7.2f} "
                f"{verdict.mean_u:7.2f} {verdict.mean_v:7.2f} {verdict.pan_dense_gap:7.2f} "
                f"{verdict.photo_pan_ratio:6.3f} {verdict.photo_flow_ratio:6.3f}"
            )
            print(f"{'':10} {dt:6.2f}s  {why}")
            rows.append(
                {
                    "clip": clip.name,
                    "stride": int(stride),
                    "dt_sec": float(dt),
                    "cert": cert,
                    "why": why,
                    "n_pairs": verdict.n_pairs,
                    "keep": len(sampled.frames),
                    "drop": sampled.n_dropped,
                    "fps": sampled.fps,
                    "pan_mean_abs": verdict.pan_mean_abs,
                    "mean_u": verdict.mean_u,
                    "mean_v": verdict.mean_v,
                    "pan_dense_gap": verdict.pan_dense_gap,
                    "photo_pan_ratio": verdict.photo_pan_ratio,
                    "photo_flow_ratio": verdict.photo_flow_ratio,
                    "resid_mean_v": verdict.resid_mean_v,
                    "flow_verdict": verdict.label,
                }
            )

    out_json = args.out / "metrics.json"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print()
    fast = [r for r in rows if r["cert"] == "validated_fast"]
    if fast:
        print("validated-fast windows:")
        for r in fast:
            print(
                f"  {r['clip']} stride={r['stride']}  |pan|={r['pan_mean_abs']:.2f}  "
                f"u={r['mean_u']:.2f}  dt={r['dt_sec']:.3f}s"
            )
    else:
        print(
            "no validated-fast window. Fast regime has no trustworthy anchor "
            "beyond ERP4's small-motion independent jitter."
        )
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
