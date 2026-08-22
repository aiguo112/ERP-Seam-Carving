"""Visualize seams overlaid on an ERP image."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def overlay_seam(
    img: np.ndarray,
    seam: np.ndarray,
    *,
    color: tuple[int, int, int] = (255, 32, 32),
    thickness: int = 1,
) -> NDArray[np.uint8]:
    """Return an RGB copy of ``img`` with the vertical seam painted.

    ``thickness`` paints neighboring columns. For a seam on the ERP cut,
    neighbors wrap so the overlay is visible on both left and right edges.
    """
    rgb = _as_rgb_uint8(img)
    h, w, _ = rgb.shape
    s = np.asarray(seam, dtype=np.int32)
    if s.shape != (h,):
        raise ValueError(f"seam length {s.shape} != height {h}")
    out = rgb.copy()
    t = max(int(thickness), 0)
    color_arr = np.asarray(color, dtype=np.uint8)
    rows = np.arange(h)
    for dx in range(-t, t + 1):
        cols = np.mod(s + dx, w)
        out[rows, cols] = color_arr
    return out


def save_cyclic_seam_proof(
    img: np.ndarray,
    seam: np.ndarray,
    path: str | Path,
    *,
    title: str = "cyclic vertical seam (wraps across x=0 / x=W-1)",
) -> NDArray[np.uint8]:
    """Save an overlay plus an x(y) plot that marks wrap steps. Returns overlay."""
    overlay = overlay_seam(img, seam, thickness=1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    s = np.asarray(seam, dtype=np.int32)
    h, w = overlay.shape[:2]
    wrap_rows = np.where(np.abs(np.diff(s)) == w - 1)[0] if w > 1 else np.array([], dtype=int)
    wrapped = wrap_rows.size > 0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].imshow(overlay)
    axes[0].set_title(title)
    axes[0].set_xlabel("x (longitude, periodic)")
    axes[0].set_ylabel("y (latitude)")
    for y in wrap_rows:
        axes[0].annotate(
            "wrap",
            xy=(float(s[y]), float(y)),
            xytext=(w * 0.45, float(y)),
            color="yellow",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "yellow"},
        )

    axes[1].plot(s, np.arange(h), color="crimson", linewidth=1.5)
    axes[1].invert_yaxis()
    axes[1].set_xlim(-1, w)
    axes[1].set_title(
        "seam x(y)" + (" — WRAP" if wrapped else " — no wrap")
    )
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].axvline(-0.5, color="gray", linestyle=":")
    axes[1].axvline(w - 0.5, color="gray", linestyle=":")
    for y in wrap_rows:
        axes[1].scatter([s[y], s[y + 1]], [y, y + 1], c="gold", zorder=3)
        axes[1].annotate("wrap", (s[y], y), color="goldenrod", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return overlay


def overlay_protected_classes(
    img: np.ndarray,
    class_map: np.ndarray,
    groups: dict[str, frozenset[int]] | None = None,
    colors: dict[str, tuple[int, int, int]] | None = None,
    *,
    alpha: float = 0.45,
) -> NDArray[np.uint8]:
    """Tint protected-class pixels on an RGB ERP copy."""
    from segmentation.voc import protected_class_ids

    rgb = _as_rgb_uint8(img)
    labels = np.asarray(class_map)
    if labels.shape != rgb.shape[:2]:
        raise ValueError("class_map must match image HxW")
    out = rgb.astype(np.float32)
    group_ids = groups if groups is not None else protected_class_ids()
    default_colors = {
        "person": (255, 40, 180),
        "vehicle": (255, 220, 40),
        "animal": (40, 220, 90),
        "door": (255, 40, 180),
        "window": (40, 180, 255),
        "board": (255, 220, 40),
        "bookcase": (40, 220, 90),
    }
    palette = colors or default_colors
    for group, ids in group_ids.items():
        mask = np.isin(labels, list(ids))
        if not np.any(mask):
            continue
        rgb_c = palette.get(group, (255, 80, 80))
        color = np.array(rgb_c, dtype=np.float32)
        out[mask] = (1.0 - alpha) * out[mask] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_seams(
    img: np.ndarray,
    seams: list[np.ndarray],
    *,
    color: tuple[int, int, int] = (255, 32, 32),
    thickness: int = 0,
) -> NDArray[np.uint8]:
    """Paint many vertical seams onto one RGB copy."""
    out = _as_rgb_uint8(img)
    for seam in seams:
        out = overlay_seam(out, seam, color=color, thickness=thickness)
    return out


def save_labeled_pair(
    left: np.ndarray,
    right: np.ndarray,
    path: str | Path,
    *,
    left_title: str,
    right_title: str,
    polar_band: float | None = 0.15,
) -> None:
    """Side-by-side images with optional polar-band guidelines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, im, title in (
        (axes[0], left, left_title),
        (axes[1], right, right_title),
    ):
        rgb = _as_rgb_uint8(im)
        ax.imshow(rgb)
        ax.set_title(title)
        ax.axis("off")
        if polar_band is not None:
            h = rgb.shape[0]
            n = int(np.floor(polar_band * h))
            if n > 0:
                ax.axhline(n - 0.5, color="cyan", linewidth=0.8, alpha=0.85)
                ax.axhline(h - n - 0.5, color="cyan", linewidth=0.8, alpha=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_quad(
    images: list[np.ndarray],
    titles: list[str],
    path: str | Path,
) -> None:
    """2x2 panel figure (ablation overlays)."""
    if len(images) != 4 or len(titles) != 4:
        raise ValueError("save_quad expects 4 images and 4 titles")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, im, title in zip(axes.ravel(), images, titles, strict=True):
        ax.imshow(_as_rgb_uint8(im))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_comparison(before: np.ndarray, after: np.ndarray, path: str | Path) -> None:
    """Save a side-by-side before / after figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(_as_rgb_uint8(before))
    axes[0].set_title(f"before {before.shape[1]}×{before.shape[0]}")
    axes[0].axis("off")
    axes[1].imshow(_as_rgb_uint8(after))
    axes[1].set_title(f"after {after.shape[1]}×{after.shape[0]}")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _as_rgb_uint8(image: np.ndarray) -> NDArray[np.uint8]:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"expected HxW or HxWx3, got {arr.shape}")
    arr = arr[..., :3]
    if arr.dtype == np.uint8:
        return arr
    if arr.max() <= 1.5:
        return np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)
