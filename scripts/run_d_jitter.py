"""Stage D0: does independent per-frame carving jitter on ERP video?

No temporal model. Stage B energy (gradient x cos(phi), cyclic DP) on each
frame independently. J = mean_t mean_y cyclic(|S_t - S_{t-1}|).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.paths import video_root  # noqa: E402
from data.video_erp import list_clips, sample_frames  # noqa: E402
from energy.gradient import gradient_energy  # noqa: E402
from energy.spherical import spherical_gradient_energy  # noqa: E402
from io_erp import save_erp  # noqa: E402
from seam.carve import map_seams_to_original, reduce_width  # noqa: E402
from temporal.jitter import compute_jitter_stats, motion_label, motion_score  # noqa: E402
from viz import overlay_seam  # noqa: E402


def energy_fn_stage_b(img: np.ndarray) -> np.ndarray:
    return spherical_gradient_energy(img)


def energy_fn_stage_a(img: np.ndarray) -> np.ndarray:
    return gradient_energy(img)


def carve_frame(img: np.ndarray, n_seams: int, spherical: bool) -> tuple[np.ndarray, np.ndarray]:
    """Independent carve. Returns (carved, first seam in original x)."""
    fn = energy_fn_stage_b if spherical else energy_fn_stage_a
    seams: list[np.ndarray] = []

    def on_seam(frame, seam, energy, i):
        if i == 0:
            seams.append(np.asarray(seam, dtype=np.int32).copy())

    carved = reduce_width(
        img, fn, img.shape[1] - n_seams, cyclic=True, on_seam=on_seam
    )
    mapped = map_seams_to_original(seams)
    return carved, mapped[0]


def write_side_by_side(
    originals: list[np.ndarray],
    overlayed: list[np.ndarray],
    carved: list[np.ndarray],
    path: Path,
    fps: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = originals[0].shape[0]
    panels = []
    for orig, ov, car in zip(originals, overlayed, carved, strict=True):
        c = car
        if c.shape[0] != h:
            c = cv2.resize(c, (c.shape[1], h), interpolation=cv2.INTER_AREA)
        if c.shape[1] != ov.shape[1]:
            pad = ov.shape[1] - c.shape[1]
            c = cv2.copyMakeBorder(c, 0, 0, 0, max(pad, 0), cv2.BORDER_CONSTANT, value=(0, 0, 0))
            c = c[:, : ov.shape[1]]
        row = np.concatenate([ov, c], axis=1)
        panels.append(cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
    hh, ww = panels[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 8.0), (ww, hh))
    if not writer.isOpened():
        raise OSError(f"could not open VideoWriter for {path}")
    for bgr in panels:
        writer.write(bgr)
    writer.release()


def resolve_root(cli_root: Path | None) -> tuple[Path, str]:
    if cli_root is not None:
        return cli_root, "cli"
    env_root = video_root()
    if env_root is not None and (env_root.is_dir() or env_root.is_file()):
        return env_root, "env"
    raise SystemExit(
        "Video root not found. Pass --root PATH (videos are not in the repo), "
        "or set ERP_VIDEO_ROOT / PAVS10K_ROOT."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-frame independent carving jitter on ERP video. No temporal model."
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "--videos",
        nargs="*",
        type=Path,
        default=None,
        help="Explicit video paths (overrides --root discovery).",
    )
    parser.add_argument("--clips", type=int, default=5)
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--max-width", type=int, default=512)
    parser.add_argument("-n", "--seams", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("output") / "jitter")
    parser.add_argument(
        "--gradient-only",
        action="store_true",
        help="Stage A energy (no spherical). Default is Stage B cyclic+spherical.",
    )
    args = parser.parse_args(argv)

    if args.videos:
        clips = [list_clips(p)[0] for p in args.videos]
        source = "explicit"
        root = Path(".")
        print(f"explicit {len(clips)} videos")
    else:
        root, source = resolve_root(args.root)
        clips = list_clips(root)
        if not clips:
            raise SystemExit(f"no video files under {root}")
        print(f"root={root}  source={source}  n_videos={len(clips)}")
        if source != "pavs":
            print("NOTE: this is NOT PAVS10K. PAVS10K was not on disk; using local ERP videos.")

    probed = []
    for clip in clips:
        try:
            sampled = sample_frames(clip.path, max_frames=12, max_width=256)
        except (ValueError, FileNotFoundError) as exc:
            print("skip", clip.path, exc)
            continue
        m = motion_score(sampled.frames, sampled.contiguous_pairs())
        probed.append((clip, m, sampled.fps))
    if not probed:
        raise SystemExit("no readable clips")
    probed.sort(key=lambda t: t[1])
    # span static -> fast: take lowest, some mids, highest
    k = min(args.clips, len(probed))
    if k == 1:
        chosen = probed
    elif k == 2:
        chosen = [probed[0], probed[-1]]
    else:
        idxs = np.linspace(0, len(probed) - 1, k).round().astype(int)
        seen = set()
        chosen = []
        for i in idxs:
            if int(i) in seen:
                continue
            seen.add(int(i))
            chosen.append(probed[int(i)])
        while len(chosen) < k and len(chosen) < len(probed):
            for item in probed:
                if item[0].path not in {c[0].path for c in chosen}:
                    chosen.append(item)
                    break

    spherical = not args.gradient_only
    energy_name = "stageB spherical+cyclic" if spherical else "stageA gradient+cyclic"
    args.out.mkdir(parents=True, exist_ok=True)
    print()
    print(
        f"{'clip':<16} {'regime':<12} {'M':>7} {'J':>8} {'Jres':>8} "
        f"{'|pan|':>8} {'pairs':>6} {'keep':>5} {'drop':>5}"
    )
    print("-" * 88)
    rows = []
    for clip, _m_probe, _fps_probe in chosen:
        sampled = sample_frames(
            clip.path, max_frames=args.frames, max_width=args.max_width
        )
        frames = sampled.frames
        pairs = sampled.contiguous_pairs()
        first_seams: list[np.ndarray] = []
        overlayed: list[np.ndarray] = []
        carved_list: list[np.ndarray] = []
        w0 = frames[0].shape[1]
        print(
            f"carving {clip.name}  keep={len(frames)}/{sampled.n_attempted}  "
            f"drop_black={sampled.n_dropped_black}  pairs={len(pairs)} ..."
        )
        for img in frames:
            carved, seam0 = carve_frame(img, args.seams, spherical=spherical)
            first_seams.append(seam0)
            overlayed.append(overlay_seam(img, seam0, thickness=1))
            carved_list.append(carved)
        stats = compute_jitter_stats(frames, first_seams, w0, pairs)
        label = motion_label(stats.M)
        print(
            f"{clip.name:<16} {label:<12} {stats.M:7.2f} {stats.J:8.2f} "
            f"{stats.J_residual:8.2f} {stats.pan_mean_abs:8.2f} "
            f"{stats.n_pairs:6d} {len(frames):5d} {sampled.n_dropped:5d}"
        )
        rows.append(
            {
                "clip": clip.name,
                "regime": label,
                "M": stats.M,
                "J": stats.J,
                "Jres": stats.J_residual,
                "pan": stats.pan_mean_abs,
                "pairs": stats.n_pairs,
                "keep": len(frames),
                "drop": sampled.n_dropped,
            }
        )
        vid = args.out / f"{clip.name}_{label}.mp4"
        write_side_by_side(
            frames, overlayed, carved_list, vid, fps=min(sampled.fps, 12.0)
        )
        save_erp(args.out / f"{clip.name}_overlay0.png", overlayed[0])
        save_erp(args.out / f"{clip.name}_overlayN.png", overlayed[-1])
        print(f"  wrote {vid}")

    print()
    print(f"energy={energy_name}  seams={args.seams}  max_width={args.max_width}")
    print(
        "J / Jres use only contiguous sample pairs "
        "(no jitter across a dropped-frame gap)."
    )
    if len(rows) >= 2:
        low = [r for r in rows if r["regime"] in {"static", "slow"}]
        high = [r for r in rows if r["regime"] in {"moderate", "fast pan"}]
        if low and high:
            js, jf = float(np.mean([r["J"] for r in low])), float(np.mean([r["J"] for r in high]))
            print(
                f"mean J low-motion={js:.2f} vs high-motion={jf:.2f}  "
                f"(ratio {jf / max(js, 1e-6):.2f}x)"
            )
        else:
            print("Need both a low-M and high-M clip to scope the temporal contribution.")
        fast = [r for r in rows if r["clip"] in {"ERP2", "ERP4"} or r["regime"] == "fast pan"]
        if fast:
            print("motion-compensated residual (fast-pan clips):")
            for r in fast:
                frac = r["Jres"] / max(r["J"], 1e-6)
                print(
                    f"  {r['clip']}: J={r['J']:.2f}  Jres={r['Jres']:.2f}  "
                    f"|pan|={r['pan']:.2f}  residual/raw={frac:.2f}  "
                    f"pairs={r['pairs']}"
                )
                if frac <= 0.35:
                    print(
                        "    most raw J is yaw: the seam is tracking the pan, "
                        "not independently wandering."
                    )
                elif frac >= 0.75:
                    print(
                        "    residual stays high: this is not a fixable yaw artifact; "
                        "the seam is not following the dominant pan."
                    )
                else:
                    print(
                        "    mixed: some of the jitter is yaw, some is leftover "
                        "path change after compensating the pan."
                    )
    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
