"""CVRG-Pano outdoor ERP RGB + pixel GT masks.

Layout::

    {root}/all-rgb/img-N.png
    {root}/all-rgb-masks/img-N.png
    {root}/{train,val,test}/{images,labels}/...

Masks are RGB palettes (not integer ids). Unmapped colors become background.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from io_erp import load_erp
from segmentation.cvrg import CVRG_COLOR_TO_ID, CVRG_NAME_TO_ID

POLAR_BAND = 0.15
POLAR_BLACK_MAX = 12


@dataclass(frozen=True)
class CvrgPano:
    rgb_path: Path
    mask_path: Path
    split: str
    stem: str


@dataclass
class CvrgSample:
    image: NDArray[np.uint8]
    class_map: NDArray[np.int32]
    void_mask: NDArray[np.bool_]
    rgb_path: Path
    mask_path: Path


def list_panoramas(root: str | Path, split: str = "val") -> list[CvrgPano]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"CVRG-Pano root does not exist: {root}")
    key = split.lower().strip()
    if key in {"train", "val", "test"}:
        rgb_dir = root / key / "images"
        mask_dir = root / key / "labels"
        split_name = key
    elif key in {"all", "all-rgb"}:
        rgb_dir = root / "all-rgb"
        mask_dir = root / "all-rgb-masks"
        split_name = "all"
    else:
        raise ValueError(f"split must be train/val/test/all, got {split}")
    if not rgb_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"missing {rgb_dir} or {mask_dir}")
    found: list[CvrgPano] = []
    for rgb in sorted(rgb_dir.glob("*.png")):
        mask = mask_dir / rgb.name
        if not mask.is_file():
            continue
        found.append(CvrgPano(rgb, mask, split_name, rgb.stem))
    return found


def load_cvrg_sample(
    pano: CvrgPano,
    *,
    max_width: int | None = 1024,
) -> CvrgSample:
    image = load_erp(pano.rgb_path)
    raw = _read_png_rgb(pano.mask_path)
    if raw.shape[:2] != image.shape[:2]:
        raw = cv2.resize(raw, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    class_map = decode_cvrg_mask(raw)
    void_mask = polar_black_void(image)
    class_map = class_map.copy()
    class_map[void_mask] = CVRG_NAME_TO_ID["background"]
    if max_width is not None and image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        new_w = int(max_width)
        new_h = max(1, int(round(image.shape[0] * scale)))
        image, class_map, void_mask = _resize_triple(image, class_map, void_mask, new_h, new_w)
    return CvrgSample(
        image=image,
        class_map=class_map,
        void_mask=void_mask,
        rgb_path=pano.rgb_path,
        mask_path=pano.mask_path,
    )


_PACKED_LUT: NDArray[np.uint8] | None = None


def decode_cvrg_mask(mask_rgb: np.ndarray) -> NDArray[np.int32]:
    """RGB palette PNG -> 21-class ids. Unknown colors -> background (0)."""
    arr = np.asarray(mask_rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"CVRG mask must be HxWx3, got {arr.shape}")
    packed = (
        arr[..., 0].astype(np.int32) * 256 * 256
        + arr[..., 1].astype(np.int32) * 256
        + arr[..., 2].astype(np.int32)
    )
    return _packed_lut()[packed].astype(np.int32)


def _packed_lut() -> NDArray[np.uint8]:
    global _PACKED_LUT
    if _PACKED_LUT is None:
        lut = np.zeros(256 * 256 * 256, dtype=np.uint8)
        for (r, g, b), cid in CVRG_COLOR_TO_ID.items():
            lut[r * 256 * 256 + g * 256 + b] = cid
        _PACKED_LUT = lut
    return _PACKED_LUT


def polar_black_void(image: np.ndarray, band: float = POLAR_BAND) -> NDArray[np.bool_]:
    rgb = np.asarray(image)
    h = rgb.shape[0]
    n = int(np.floor(band * h))
    polar = np.zeros(h, dtype=bool)
    if n > 0:
        polar[:n] = True
        polar[h - n :] = True
    black = np.max(rgb[..., :3], axis=2) <= POLAR_BLACK_MAX
    return black & polar[:, None]


def _read_png_rgb(path: Path) -> NDArray[np.uint8]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read CVRG mask: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _resize_triple(
    image: np.ndarray,
    class_map: np.ndarray,
    void_mask: np.ndarray,
    height: int,
    width: int,
) -> tuple[NDArray[np.uint8], NDArray[np.int32], NDArray[np.bool_]]:
    rgb = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    labels = cv2.resize(class_map.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST)
    void = cv2.resize(void_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return rgb, labels.astype(np.int32), void.astype(bool)
