"""Find a high-motion window, then certify pan vs dense at stride 1 and 8."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.paths import default_videos  # noqa: E402
from data.video_erp import list_clips, sample_strided_frames  # noqa: E402
from temporal.flow import certify_motion, pair_flow_report, summarize_clip  # noqa: E402
from temporal.jitter import motion_score  # noqa: E402

FRACS = (0.12, 0.30, 0.45, 0.62, 0.80)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Probe high-motion windows on ERP clips.")
    parser.add_argument("--videos", nargs="*", type=Path, default=None)
    args = parser.parse_args(argv)
    clips = list(args.videos) if args.videos else default_videos()
    if not clips:
        raise SystemExit("no videos found; pass --videos PATHS (videos are not in the repo)")
    out = Path("output") / "stride_sweep"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    print(f"{'clip':<8} {'frac':>5} {'M':>7}  (stride=1 probe)")
    hottest: dict[str, float] = {}
    for path in CLIPS:
        clip = list_clips(path)[0]
        best_frac, best_m = FRACS[0], -1.0
        for frac in FRACS:
            sampled = sample_strided_frames(
                clip.path, stride=1, n_frames=12, start_frac=frac, max_width=256
            )
            m = motion_score(sampled.frames, sampled.contiguous_pairs())
            print(f"{clip.name:<8} {frac:5.2f} {m:7.2f}")
            if m > best_m:
                best_m, best_frac = m, frac
        hottest[clip.name] = best_frac
        print(f"  hottest frac={best_frac:.2f} M={best_m:.2f}")

    print()
    print(
        f"{'clip':<8} {'frac':>5} {'str':>3} {'cert':<16} {'|pan|':>7} {'u':>7} "
        f"{'|p-u|':>7} {'photP':>6} {'photD':>6}"
    )
    for path in CLIPS:
        clip = list_clips(path)[0]
        frac = hottest[clip.name]
        for stride in (1, 8):
            sampled = sample_strided_frames(
                clip.path, stride=stride, n_frames=16, start_frac=frac, max_width=512
            )
            reports = [
                pair_flow_report(sampled.frames[a], sampled.frames[b])
                for a, b in sampled.contiguous_pairs()
            ]
            v = summarize_clip(reports)
            cert, why = certify_motion(v)
            print(
                f"{clip.name:<8} {frac:5.2f} {stride:3d} {cert:<16} {v.pan_mean_abs:7.2f} "
                f"{v.mean_u:7.2f} {v.pan_dense_gap:7.2f} {v.photo_pan_ratio:6.3f} "
                f"{v.photo_flow_ratio:6.3f}"
            )
            print(f"{'':8}       {why}")
            rows.append(
                {
                    "clip": clip.name,
                    "frac": frac,
                    "stride": stride,
                    "cert": cert,
                    "why": why,
                    "pan_mean_abs": v.pan_mean_abs,
                    "mean_u": v.mean_u,
                    "mean_v": v.mean_v,
                    "pan_dense_gap": v.pan_dense_gap,
                    "photo_pan_ratio": v.photo_pan_ratio,
                    "photo_flow_ratio": v.photo_flow_ratio,
                    "n_pairs": v.n_pairs,
                }
            )

    (out / "hot_windows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fast = [r for r in rows if r["cert"] == "validated_fast"]
    print()
    if fast:
        print("validated-fast:", fast)
    else:
        print("still no validated-fast window after scanning 5 positions x hottest certify.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
