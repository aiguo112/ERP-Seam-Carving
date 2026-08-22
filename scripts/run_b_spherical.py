"""Stage B: spherical area weighting vs plain gradient energy.

ERP pixels are not equal: area ∝ cos(φ), so the poles are oversampled.
This script carves the same image twice:

1. Plain Sobel energy.
2. Sobel energy multiplied by cos(φ).

Polar pixels become cheap in (2). A vertical seam still has one pixel per
row, so the *count* of seam pixels in the top/bottom 15% of rows is the
same for both methods. What increases under spherical weighting is polar
*routing* (lateral |Δx| in those bands): the DP meanders in the redundant
caps and spends its budget on the equator. That is intended, not a bug.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from energy.gradient import gradient_energy  # noqa: E402
from energy.spherical import (  # noqa: E402
    mean_polar_step,
    polar_pixel_fraction,
    polar_routing_share,
    spherical_gradient_energy,
)
from io_erp import load_erp, save_erp  # noqa: E402
from seam.carve import reduce_width  # noqa: E402
from viz import overlay_seam, save_labeled_pair  # noqa: E402

POLAR_BAND = 0.15


def make_erp_demo(height: int = 160, width: int = 320, seed: int = 0) -> np.ndarray:
    """ERP-like canvas: equatorial valley + uniformly expensive polar texture.

    Unweighted Sobel sees the polar checker as costly, so the seam stays
    nearly vertical in the caps and spends its lateral motion on the
    equator. After ``cos(φ)`` the caps are almost free, so the seam walks
    in the polar 15% bands toward a cheap equatorial column.
    """
    rng = np.random.default_rng(seed)
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    rgb[..., 2] = 0.28 + 0.40 * (1.0 - y)
    rgb[..., 1] = 0.18 + 0.20 * y
    rgb[..., 0] = 0.12 + 0.10 * y
    rgb += 0.02 * rng.standard_normal(rgb.shape).astype(np.float32)

    # Cheap equatorial column, close enough that a polar-cap walk can reach it.
    n = int(np.floor(0.15 * height))
    valley = min(n + 2, width // 4)
    eq = slice(n, height - n)
    rgb[eq, valley - 1 : valley + 2, :] = 0.04

    # Object to avoid on the equator.
    rgb[eq, width // 2 : width // 2 + 8, :] = 0.95

    # Polar checkerboard: roughly uniform high gradient in the caps.
    yy = np.arange(height)[:, None]
    xx = np.arange(width)[None, :]
    checker = ((xx // 2 + yy // 2) % 2).astype(np.float32)
    rgb[:n] = 0.15 + 0.75 * checker[:n, :, None]
    rgb[height - n :] = 0.15 + 0.75 * checker[height - n :, :, None]

    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def _carve(
    image: np.ndarray,
    energy_fn,
    n_seams: int,
    cyclic: bool,
) -> tuple[np.ndarray, list[np.ndarray], list[int], np.ndarray]:
    seams: list[np.ndarray] = []
    widths: list[int] = []
    first_overlay: np.ndarray | None = None

    def on_seam(frame: np.ndarray, seam: np.ndarray, energy: np.ndarray, i: int) -> None:
        nonlocal first_overlay
        seams.append(np.asarray(seam, dtype=np.int32).copy())
        widths.append(int(frame.shape[1]))
        if first_overlay is None:
            first_overlay = overlay_seam(frame, seam, thickness=1)

    carved = reduce_width(
        image,
        energy_fn,
        image.shape[1] - n_seams,
        cyclic=cyclic,
        on_seam=on_seam,
    )
    assert first_overlay is not None
    return carved, seams, widths, first_overlay


def _report(
    name: str,
    seams: list[np.ndarray],
    height: int,
    widths: list[int],
    cyclic: bool,
) -> dict[str, float]:
    pix = polar_pixel_fraction(seams, height, POLAR_BAND)
    route = polar_routing_share(seams, height, widths, POLAR_BAND, cyclic=cyclic)
    step = mean_polar_step(seams, height, widths, POLAR_BAND, cyclic=cyclic)
    print(f"{name:12s}  polar-band pixel fraction (top/bottom {POLAR_BAND:.0%}): {pix:.3f}")
    print(f"{name:12s}  polar-band routing share (abs dx in those rows):   {route:.3f}")
    print(f"{name:12s}  mean abs dx per polar-band row:                    {step:.3f}")
    return {
        "polar_pixel_fraction": pix,
        "polar_routing_share": route,
        "mean_polar_step": step,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage B: compare plain vs cos(φ)-weighted gradient carving."
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=None,
        help="ERP PNG/JPG. Omit to use a built-in polar-noise demo.",
    )
    parser.add_argument("-n", "--seams", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path("output") / "spherical")
    parser.add_argument(
        "--no-cyclic",
        action="store_true",
        help="Disable longitude wrap in DP (default is cyclic=True).",
    )
    args = parser.parse_args(argv)

    cyclic = not args.no_cyclic
    if args.image is None:
        image = make_erp_demo()
    else:
        image = load_erp(args.image)

    h, w, _ = image.shape
    if args.seams < 1 or args.seams >= w:
        raise SystemExit(f"--seams must be in [1, {w - 1}], got {args.seams}")

    args.out.mkdir(parents=True, exist_ok=True)
    save_erp(args.out / "original.png", image)

    carved_plain, seams_plain, widths_plain, overlay_plain = _carve(
        image, gradient_energy, args.seams, cyclic
    )
    carved_sph, seams_sph, widths_sph, overlay_sph = _carve(
        image, spherical_gradient_energy, args.seams, cyclic
    )

    save_erp(args.out / "carved_plain.png", carved_plain)
    save_erp(args.out / "carved_spherical.png", carved_sph)
    save_erp(args.out / "first_seam_plain.png", overlay_plain)
    save_erp(args.out / "first_seam_spherical.png", overlay_sph)
    save_labeled_pair(
        overlay_plain,
        overlay_sph,
        args.out / "first_seam_comparison.png",
        left_title="plain gradient (first seam)",
        right_title="spherical × cos(φ) (first seam)",
        polar_band=POLAR_BAND,
    )
    save_labeled_pair(
        carved_plain,
        carved_sph,
        args.out / "comparison.png",
        left_title=f"plain gradient  {carved_plain.shape[1]}px",
        right_title=f"spherical × cos(φ)  {carved_sph.shape[1]}px",
        polar_band=POLAR_BAND,
    )

    print(f"cyclic={cyclic}  {image.shape} -> {carved_plain.shape}")
    print(
        "Polar-band pixel fraction is ~2*15% for ANY vertical seam (one pixel "
        "per row). Spherical weighting instead increases polar routing share."
    )
    stats_plain = _report("plain", seams_plain, h, widths_plain, cyclic)
    stats_sph = _report("spherical", seams_sph, h, widths_sph, cyclic)

    if stats_sph["mean_polar_step"] > stats_plain["mean_polar_step"]:
        print(
            "verified: spherical uses polar rows for MORE lateral travel "
            "(cheap pixels, less spherical area) -- intended, not a bug."
        )
    elif stats_sph["polar_routing_share"] > stats_plain["polar_routing_share"]:
        print(
            "verified: spherical routes MORE lateral seam motion through polar "
            "rows (cheap, less spherical area) -- intended, not a bug."
        )
    else:
        print(
            "note: polar routing share did not increase on this image "
            "(content may already be polar-cheap)."
        )
    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
