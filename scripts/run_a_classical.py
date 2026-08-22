"""Stage A: classical gradient seam carving (non-cyclic DP)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from energy.gradient import gradient_energy  # noqa: E402
from io_erp import load_erp, save_erp  # noqa: E402
from seam.carve import reduce_width  # noqa: E402
from seam.dp import find_vertical_seam  # noqa: E402
from viz import overlay_seam, save_comparison  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage A baseline: Sobel energy + standard (non-cyclic) vertical seams."
    )
    parser.add_argument("image", type=Path, help="Path to an ERP PNG/JPG.")
    parser.add_argument(
        "-n",
        "--seams",
        type=int,
        default=40,
        help="Number of vertical seams to remove (width reduction).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output"),
        help="Directory for carved image, overlay, and comparison.",
    )
    parser.add_argument(
        "--cyclic",
        action="store_true",
        help="Use cyclic 8-connected DP (off by default for Stage A).",
    )
    args = parser.parse_args(argv)

    image = load_erp(args.image)
    h, w, _ = image.shape
    if args.seams < 0 or args.seams >= w:
        raise SystemExit(f"--seams must be in [0, {w - 1}], got {args.seams}")

    cyclic = bool(args.cyclic)
    first_overlay_saved = False

    def on_seam(frame, seam, energy, i: int) -> None:
        nonlocal first_overlay_saved
        if first_overlay_saved:
            return
        overlay = overlay_seam(frame, seam, thickness=1)
        save_erp(args.out / "seam_overlay.png", overlay)
        first_overlay_saved = True

    target_w = w - args.seams
    carved = reduce_width(
        image,
        gradient_energy,
        target_w,
        cyclic=cyclic,
        on_seam=on_seam,
    )

    if args.seams == 0:
        energy = gradient_energy(image)
        seam = find_vertical_seam(energy, cyclic=cyclic)
        save_erp(args.out / "seam_overlay.png", overlay_seam(image, seam))

    save_erp(args.out / "original.png", image)
    save_erp(args.out / "carved.png", carved)
    save_comparison(image, carved, args.out / "comparison.png")
    print(f"cyclic={cyclic}  {image.shape} -> {carved.shape}")
    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
