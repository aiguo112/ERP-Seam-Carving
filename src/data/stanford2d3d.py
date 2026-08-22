"""Stanford 2D-3D-S equirectangular panoramas + semantic GT.

Official layout (per area)::

    area_N/pano/rgb/*equirectangular_domain__rgb.png
    area_N/pano/semantic/*equirectangular_domain__semantic.png
    assets/semantic_labels.json

Semantic PNGs are 24-bit base-256 indices into ``semantic_labels.json``.
Missing labels are color ``#0D0D0D``. RGB panoramas also have black polar
caps (camera FOV); those are marked void and get **zero semantic energy**.
Metrics must exclude void pixels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from io_erp import load_erp
from segmentation.stanford import (
    STANFORD_CLASS_NAMES,
    STANFORD_NAME_TO_ID,
    normalize_stanford_class,
)

VOID_RGB = (0x0D, 0x0D, 0x0D)
POLAR_BAND = 0.15
POLAR_BLACK_MAX = 12  # max RGB channel in polar rows to count as FOV void


@dataclass(frozen=True)
class StanfordPano:
    rgb_path: Path
    semantic_path: Path
    area: str
    stem: str


@dataclass
class StanfordSample:
    image: NDArray[np.uint8]
    class_map: NDArray[np.int32]
    void_mask: NDArray[np.bool_]
    rgb_path: Path
    semantic_path: Path


def find_labels_json(root: str | Path) -> Path:
    root = Path(root)
    candidates = [
        root / "assets" / "semantic_labels.json",
        root / "semantic_labels.json",
        root.parent / "assets" / "semantic_labels.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "semantic_labels.json not found under "
        f"{root} (expected assets/semantic_labels.json)"
    )


def list_panoramas(root: str | Path) -> list[StanfordPano]:
    """Discover RGB/semantic ERP pairs under a 2D-3D-S root."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Stanford2D3D root does not exist: {root}")
    found: list[StanfordPano] = []
    for rgb in sorted(root.rglob("*equirectangular_domain__rgb.png")):
        if "semantic_pretty" in rgb.parts:
            continue
        sem = _semantic_path_for_rgb(rgb)
        if sem is None or not sem.is_file():
            continue
        area = _area_name(rgb, root)
        stem = rgb.name.replace("_domain__rgb.png", "")
        found.append(StanfordPano(rgb, sem, area, stem))
    if not found:
        # Fallback: any *rgb*.png next to a matching semantic file.
        for rgb in sorted(root.rglob("*_rgb.png")):
            if "semantic" in rgb.parts:
                continue
            sem = rgb.parent.parent / "semantic" / rgb.name.replace("_rgb.png", "_semantic.png")
            if not sem.is_file():
                continue
            found.append(StanfordPano(rgb, sem, _area_name(rgb, root), rgb.stem))
    return found


def load_stanford_sample(
    pano: StanfordPano,
    labels_json: str | Path,
    *,
    max_width: int | None = 1024,
) -> StanfordSample:
    """Load RGB + 13-class GT + void mask. Optionally downsample (nearest GT)."""
    image = load_erp(pano.rgb_path)
    raw_sem = _read_png_rgb(pano.semantic_path)
    if raw_sem.shape[:2] != image.shape[:2]:
        raw_sem = cv2.resize(
            raw_sem, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    labels = _load_label_strings(labels_json)
    class_map, void_from_sem = decode_stanford_semantic(raw_sem, labels)
    void_mask = void_from_sem | polar_black_void(image)
    class_map = class_map.copy()
    class_map[void_mask] = STANFORD_NAME_TO_ID["void"]

    if max_width is not None and image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        new_w = int(max_width)
        new_h = max(1, int(round(image.shape[0] * scale)))
        image, class_map, void_mask = _resize_triple(image, class_map, void_mask, new_h, new_w)

    return StanfordSample(
        image=image,
        class_map=class_map,
        void_mask=void_mask,
        rgb_path=pano.rgb_path,
        semantic_path=pano.semantic_path,
    )


def decode_stanford_semantic(
    semantic_rgb: np.ndarray,
    labels: list[str],
) -> tuple[NDArray[np.int32], NDArray[np.bool_]]:
    """Decode 24-bit instance PNG to 13-class ids and a void mask."""
    arr = np.asarray(semantic_rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"semantic image must be HxWx3, got {arr.shape}")
    index = (
        arr[..., 0].astype(np.int32) * 256 * 256
        + arr[..., 1].astype(np.int32) * 256
        + arr[..., 2].astype(np.int32)
    )
    missing = (
        (arr[..., 0] == VOID_RGB[0])
        & (arr[..., 1] == VOID_RGB[1])
        & (arr[..., 2] == VOID_RGB[2])
    )
    n = len(labels)
    oob = index >= n
    void = missing | oob

    lut = np.zeros(max(n, 1), dtype=np.int32)
    for i, raw in enumerate(labels):
        lut[i] = STANFORD_NAME_TO_ID[normalize_stanford_class(_instance_class(raw))]

    class_map = np.zeros(index.shape, dtype=np.int32)
    valid = ~void
    class_map[valid] = lut[index[valid]]
    class_map[void] = STANFORD_NAME_TO_ID["void"]
    return class_map, void


def polar_black_void(image: np.ndarray, band: float = POLAR_BAND) -> NDArray[np.bool_]:
    """Black polar caps (camera FOV), not dark furniture on the equator."""
    rgb = np.asarray(image)
    h = rgb.shape[0]
    n = int(np.floor(band * h))
    polar = np.zeros(h, dtype=bool)
    if n > 0:
        polar[:n] = True
        polar[h - n :] = True
    black = np.max(rgb[..., :3], axis=2) <= POLAR_BLACK_MAX
    return black & polar[:, None]


def encode_index_color(index: int) -> tuple[int, int, int]:
    """24-bit index -> RGB, inverse of the 2D-3D-S encoder."""
    r = (index >> 16) & 255
    g = (index >> 8) & 255
    b = index & 255
    return r, g, b


def _instance_class(label: str) -> str:
    # "{class}_{instance}_{roomType}_{roomNum}_{areaNum}"
    return str(label).split("_", 1)[0]


def _load_label_strings(path: str | Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("semantic_labels.json must be a JSON list of strings")
    return [str(x) for x in data]


def _semantic_path_for_rgb(rgb: Path) -> Path | None:
    name = rgb.name.replace("domain__rgb.png", "domain__semantic.png")
    candidates = [
        rgb.with_name(name),
        rgb.parent.parent / "semantic" / name,
        rgb.parent.parent / "semantic" / rgb.name.replace("__rgb.png", "__semantic.png"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _area_name(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
        return rel.parts[0]
    except ValueError:
        return path.parts[0]


def _read_png_rgb(path: Path) -> NDArray[np.uint8]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"could not read semantic PNG: {path}")
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
