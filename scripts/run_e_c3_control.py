"""C3 control: sampling-induced phantom yaw, not a broken ERP estimator.

Same equatorial cyclic phase-correlation estimator, same ERP2 frame I_t:

  A. I_t vs roll(I_t, k)  -- known cyclic yaw on a real spherical image.
     If pan ~= k and photometry drops, the estimator *can* measure ERP yaw.
  B. I_t vs I_{t+1}       -- stride 1. Should agree with dense u near 0.
  C. I_t vs I_{t+S}       -- original linspace gap (~232 frames / 7.7s).
     Headline phantom lives here: large pan, photometry does not drop.

If A works and B is ~0 and C fabricates, the failure is the gap, not the projector.
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

from data.paths import erp2_video, require_path
from data.video_erp import sample_frames  # noqa: E402
from temporal.flow import equator_flow_stats, dense_flow_cyclic  # noqa: E402
from temporal.jitter import (  # noqa: E402
    equatorial_horizontal_pan,
    photometric_warp_scores,
)

SYNTH_K = (0, 8, 16, 32, 64, 128, 141)


def read_frame(path: Path, index: int, max_width: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        raise ValueError(f"could not read frame {index} from {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if w > max_width:
        scale = max_width / w
        rgb = cv2.resize(
            rgb,
            (int(max_width), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return rgb


def row(kind: str, pan: float, u: float | None, raw: float, wrp: float) -> dict:
    return {
        "kind": kind,
        "pan": float(pan),
        "dense_u": None if u is None else float(u),
        "photo_raw": float(raw),
        "photo_pan": float(wrp),
        "photo_ratio": float(wrp / max(raw, 1e-6)),
    }


def report_pair(kind: str, a: np.ndarray, b: np.ndarray, *, dense: bool) -> dict:
    pan = equatorial_horizontal_pan(a, b)
    raw, wrp = photometric_warp_scores(a, b, pan)
    u = None
    if dense:
        mu, _mv, _su, _sv = equator_flow_stats(dense_flow_cyclic(a, b))
        u = mu
    rec = row(kind, pan, u, raw, wrp)
    extra = ""
    if u is not None:
        extra = f"  dense_u={u:7.2f}"
    print(
        f"{kind:<28} pan={pan:7.2f}{extra}  photo_raw={raw:6.2f}  "
        f"photo_pan={wrp:6.2f}  ratio={rec['photo_ratio']:.3f}"
    )
    return rec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C3 sampling-vs-estimator control.")
    parser.add_argument("--video", type=Path, default=erp2_video())
    parser.add_argument("--max-width", type=int, default=512)
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--out", type=Path, default=Path("output") / "c3_control")
    args = parser.parse_args(argv)
    args.video = require_path(args.video, flag="--video", env="ERP2_VIDEO")
    if not args.video.is_file():
        raise SystemExit(f"missing {args.video}")
    args.out.mkdir(parents=True, exist_ok=True)

    sampled = sample_frames(
        args.video, max_frames=args.frames, max_width=args.max_width
    )
    pairs = sampled.contiguous_pairs()
    pans = [
        equatorial_horizontal_pan(sampled.frames[i], sampled.frames[j])
        for i, j in pairs
    ]
    k = int(np.argmax(np.abs(pans)))
    i, j = pairs[k]
    t0 = int(sampled.source_indices[i])
    t1 = int(sampled.source_indices[j])
    gap = t1 - t0
    fps = float(sampled.fps)
    print(
        f"anchor linspace pair {i}->{j}  src {t0}->{t1}  gap={gap} frames "
        f"({gap / max(fps, 1):.2f}s)  |pan|={abs(pans[k]):.2f}"
    )
    print()

    a = sampled.frames[i]
    b_sparse = sampled.frames[j]
    b_next = read_frame(args.video, t0 + 1, args.max_width)

    print("--- A. synthetic cyclic yaw on the same ERP frame (estimator sanity) ---")
    synth = []
    for dk in SYNTH_K:
        rolled = np.roll(a, int(dk), axis=1)
        use_dense = abs(dk) <= 16
        rec = report_pair(f"synth roll {dk:3d}px", a, rolled, dense=use_dense)
        rec["true_k"] = int(dk)
        synth.append(rec)

    print()
    print("--- B. real pair, stride 1, same I_t ---")
    real_1 = report_pair("real t -> t+1", a, b_next, dense=True)

    print()
    print("--- C. real pair, original linspace gap, same I_t ---")
    real_s = report_pair(f"real t -> t+{gap}", a, b_sparse, dense=True)

    rec_0 = next(r for r in synth if r["true_k"] == 0)
    worst_hi_err = max(abs(r["pan"] - r["true_k"]) for r in synth if r["true_k"] >= 32)
    a_ok = worst_hi_err < 4.0 and all(
        r["photo_ratio"] < 0.15 for r in synth if r["true_k"] >= 8
    )
    print()
    print("control outcome:")
    print(
        f"  A synthetic: max |pan-k| for k>=32 is {worst_hi_err:.2f} px; "
        f"{'PASS (estimator recovers ERP yaw)' if a_ok else 'FAIL (estimator broken on ERP yaw)'}"
    )
    print(
        f"  B stride-1: pan={real_1['pan']:.2f} dense_u={real_1['dense_u']:.2f} "
        f"photo_ratio={real_1['photo_ratio']:.3f}"
    )
    print(
        f"  C gap {gap}: pan={real_s['pan']:.2f} dense_u={real_s['dense_u']:.2f} "
        f"photo_ratio={real_s['photo_ratio']:.3f}"
    )
    if a_ok and abs(real_1["pan"]) < 8 and abs(real_s["pan"]) > 32 and real_s["photo_ratio"] > 0.9:
        print(
            "  C3 framing: same estimator, same I_t: correct on known yaw and "
            "stride-1, fabricates at the sparse gap. Sampling-induced."
        )
    elif not a_ok:
        print(
            "  C3 risk: estimator does not recover a known cyclic yaw on this "
            "ERP frame. Do not claim a field-level sampling trap."
        )
    else:
        print("  mixed: inspect the table before claiming C3.")

    out = {
        "video": str(args.video),
        "t0": t0,
        "t1_sparse": t1,
        "gap_frames": gap,
        "gap_sec": gap / max(fps, 1.0),
        "width": int(a.shape[1]),
        "synth": synth,
        "real_stride1": real_1,
        "real_sparse": real_s,
        "identity_photo_ratio": rec_0["photo_ratio"],
        "synth_max_err_k32plus": worst_hi_err,
    }
    path = args.out / "metrics.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
