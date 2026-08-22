"""Export C3 A/B/C equator strips for the workshop PDF."""

from __future__ import annotations

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

from data.paths import erp2_video, require_path  # noqa: E402
from temporal.jitter import EQUATOR_BAND  # noqa: E402

METRICS = ROOT / "output" / "c3_control" / "metrics.json"
OUT = ROOT / "output" / "fig_c3_control.png"


def read_frame(video: Path, index: int, max_width: int = 512) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        raise SystemExit(f"could not read frame {index}")
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


def equator(img: np.ndarray) -> np.ndarray:
    h = img.shape[0]
    lo, hi = int(EQUATOR_BAND * h), int((1.0 - EQUATOR_BAND) * h)
    return img[lo:hi]


def main() -> int:
    video = require_path(erp2_video(), flag="--video / ERP2_VIDEO", env="ERP2_VIDEO")
    meta = json.loads(METRICS.read_text(encoding="utf-8"))
    t0 = int(meta["t0"])
    t1 = int(meta["t1_sparse"])
    pan_c = float(meta["real_sparse"]["pan"])
    a = read_frame(video, t0)
    b1 = read_frame(video, t0 + 1)
    bs = read_frame(video, t1)
    roll141 = np.roll(a, 141, axis=1)
    warped = np.roll(a, int(round(pan_c)), axis=1)
    resid = np.abs(equator(warped).astype(np.int16) - equator(bs).astype(np.int16))
    resid = np.clip(resid * 3, 0, 255).astype(np.uint8)

    panels = [
        ("(A) $I_t$", equator(a)),
        ("(A) roll $I_t$ by 141 px", equator(roll141)),
        ("(B) $I_{t+1}$ (0.03 s)", equator(b1)),
        ("(C) $I_{t+232}$ (7.74 s)", equator(bs)),
        (f"(C) roll $I_t$ by {pan_c:.0f} px", equator(warped)),
        ("(C) |warp $-$ sparse| $\\times$ 3", resid),
    ]
    # Wide 2x3; placed as a single-column bottom float in the paper.
    fig, axes = plt.subplots(2, 3, figsize=(11.6, 4.15))
    for ax, (title, im) in zip(axes.ravel(), panels, strict=True):
        ax.imshow(im)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout(pad=0.35, w_pad=0.25, h_pad=0.35)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
