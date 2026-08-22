"""Validate whether a cyclic horizontal pan is an adequate motion model.

No sequence model. DIS dense flow (wrap in x) vs the phase-correlation
pan used for S_pred. Residual flow is computed AFTER rolling I_{t-1} by pan.

ERP4 is the control (small pan; S_pred ~= S_{t-1}). ERP2 is the headline mixed
clip. ERP5 is the known-noisy pan row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.paths import default_videos  # noqa: E402
from data.video_erp import list_clips, sample_frames  # noqa: E402
from io_erp import save_erp  # noqa: E402
from temporal.flow import (  # noqa: E402
    ClipFlowVerdict,
    dense_flow_cyclic,
    flow_to_bgr,
    pair_flow_report,
    summarize_clip,
)

from data.paths import default_videos  # noqa: E402


def print_verdict(name: str, v: ClipFlowVerdict) -> None:
    print(f"=== {name}  pairs={v.n_pairs}  verdict={v.label} ===")
    print(f"  {v.reason}")
    print(
        f"  |pan|={v.pan_mean_abs:.2f}  dense_u={v.mean_u:.2f}  dense_v={v.mean_v:.2f}  "
        f"std_u={v.std_u:.2f}  |pan-dense|={v.pan_dense_gap:.2f}"
    )
    print(
        f"  residual after pan: u={v.resid_mean_u:.2f}  v={v.resid_mean_v:.2f}  "
        f"std_u={v.resid_std_u:.2f}"
    )
    print(
        f"  photo ratio  pan={v.photo_pan_ratio:.3f}  dense={v.photo_flow_ratio:.3f}  "
        f"pan+resid={v.photo_resid_ratio:.3f}  (raw={v.photo_raw:.2f})"
    )
    if v.label == "yaw_ok":
        print("  horizontal S_pred is adequate on this clip.")
    elif v.label == "small_motion":
        print("  S_pred is nearly S_{t-1}; residual-J wins do not depend on the pan number.")
    elif v.label == "stride_too_sparse":
        print("  asterisk this clip: appearance change is not a warp at this sample stride.")
    else:
        print("  asterisk this clip: replace the horizontal-shift S_pred (or drop the row).")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dense-flow check of the pan model.")
    parser.add_argument("--videos", nargs="*", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--max-width", type=int, default=512)
    parser.add_argument("--max-pairs", type=int, default=16)
    parser.add_argument("--out", type=Path, default=Path("output") / "flow_validate")
    args = parser.parse_args(argv)

    paths = list(args.videos) if args.videos else default_videos()
    if not paths:
        raise SystemExit("no videos found; pass --videos PATHS (videos are not in the repo)")
    args.out.mkdir(parents=True, exist_ok=True)

    print("spec: residual dense flow AFTER pan-warp.")
    print("  yaw_ok / small_motion => keep horizontal S_pred")
    print("  replace_model / stride_too_sparse => asterisk the clip")
    print()

    rows: list[dict] = []
    for path in paths:
        clip = list_clips(path)[0]
        sampled = sample_frames(
            clip.path, max_frames=args.frames, max_width=args.max_width
        )
        pairs = sampled.contiguous_pairs()
        if args.max_pairs and len(pairs) > args.max_pairs:
            idxs = [
                int(round(x))
                for x in np.linspace(0, len(pairs) - 1, args.max_pairs)
            ]
            seen: set[int] = set()
            chosen = []
            for i in idxs:
                if i not in seen:
                    seen.add(i)
                    chosen.append(pairs[i])
            pairs = chosen
        reports = []
        mid = pairs[len(pairs) // 2] if pairs else None
        for a, b in pairs:
            reports.append(pair_flow_report(sampled.frames[a], sampled.frames[b]))
        verdict = summarize_clip(reports)
        print_verdict(clip.name, verdict)
        if mid is not None:
            flow = dense_flow_cyclic(sampled.frames[mid[0]], sampled.frames[mid[1]])
            bgr = flow_to_bgr(flow)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            save_erp(args.out / f"{clip.name}_flow.png", rgb)
        row = {
            "clip": clip.name,
            "verdict": verdict.label,
            "reason": verdict.reason,
            "n_pairs": verdict.n_pairs,
            "pan_mean_abs": verdict.pan_mean_abs,
            "mean_u": verdict.mean_u,
            "mean_v": verdict.mean_v,
            "std_u": verdict.std_u,
            "resid_mean_u": verdict.resid_mean_u,
            "resid_mean_v": verdict.resid_mean_v,
            "resid_std_u": verdict.resid_std_u,
            "photo_raw": verdict.photo_raw,
            "photo_pan_ratio": verdict.photo_pan_ratio,
            "photo_flow_ratio": verdict.photo_flow_ratio,
            "photo_resid_ratio": verdict.photo_resid_ratio,
            "pan_dense_gap": verdict.pan_dense_gap,
        }
        rows.append(row)

    out_json = args.out / "metrics.json"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
