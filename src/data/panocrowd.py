"""PanoCount ERP stills + head-point JSON (module name is historical).

Layout::

    {root}/images/{id}.jpg
    {root}/json/{id}.json
    {root}/splits/{train,val,test}.txt   # one image id per line

JSON has ``human_num`` and ``points: [{x, y}, ...]`` in native pixel coords.
Images are 2:1 ERP (typically 1664x832). The stills are not shipped here;
see the paper / PanoCount authors for access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from data.paths import panocount_root
from io_erp import load_erp

# Optional local default via PANOCOUNT_ROOT / PANOCROWD_ROOT. Datasets are not shipped.
DEFAULT_ROOT = panocount_root()

# Locked density bins: PanoCrowd's own 0-49 / 50-99+100-199 / 200+ grouping.
# Medium (50-199) is the pre-registered stress regime.
BIN_SPARSE = "sparse"
BIN_MEDIUM = "medium"
BIN_DENSE = "dense"


@dataclass(frozen=True)
class PanoCrowdItem:
    image_path: Path
    json_path: Path
    split: str
    stem: str
    image_id: str


@dataclass
class PanoCrowdSample:
    image: NDArray[np.uint8]
    heads: NDArray[np.float64]  # (N, 2) x, y in the loaded image
    human_num: int
    density_bin: str
    image_path: Path
    json_path: Path


def density_bin(human_num: int) -> str:
    n = int(human_num)
    if n < 50:
        return BIN_SPARSE
    if n < 200:
        return BIN_MEDIUM
    return BIN_DENSE


def list_panocrowd(
    root: str | Path | None = None,
    split: str = "val+test",
) -> list[PanoCrowdItem]:
    if root is None:
        root = DEFAULT_ROOT
    if root is None:
        raise FileNotFoundError(
            "PanoCount root is unset. Pass root=... or set PANOCOUNT_ROOT."
        )
    root = Path(root)
    img_dir = root / "images"
    json_dir = root / "json"
    split_dir = root / "splits"
    if not img_dir.is_dir() or not json_dir.is_dir():
        raise FileNotFoundError(f"missing {img_dir} or {json_dir}")

    key = split.lower().strip()
    if key == "val+test":
        ids = _read_ids(split_dir / "val.txt") + _read_ids(split_dir / "test.txt")
        split_name = "val+test"
    elif key in {"train", "val", "test"}:
        ids = _read_ids(split_dir / f"{key}.txt")
        split_name = key
    elif key == "all":
        ids = sorted({p.stem for p in json_dir.glob("*.json")})
        split_name = "all"
    else:
        raise ValueError(f"split must be train/val/test/val+test/all, got {split}")

    found: list[PanoCrowdItem] = []
    for image_id in ids:
        jpg = img_dir / f"{image_id}.jpg"
        if not jpg.is_file():
            jpg = img_dir / f"{image_id}.jpeg"
        js = json_dir / f"{image_id}.json"
        if not jpg.is_file() or not js.is_file():
            continue
        found.append(PanoCrowdItem(jpg, js, split_name, jpg.stem, str(image_id)))
    return found


def load_panocrowd_sample(
    item: PanoCrowdItem,
    *,
    max_width: int | None = 512,
) -> PanoCrowdSample:
    image = load_erp(item.image_path)
    meta = json.loads(item.json_path.read_text(encoding="utf-8"))
    pts = meta.get("points") or []
    heads = np.array([[float(p["x"]), float(p["y"])] for p in pts], dtype=np.float64)
    if heads.size == 0:
        heads = np.zeros((0, 2), dtype=np.float64)
    human_num = int(meta.get("human_num", heads.shape[0]))
    h0, w0 = image.shape[:2]
    if max_width is not None and w0 > max_width:
        scale = max_width / w0
        new_w = int(max_width)
        new_h = max(1, int(round(h0 * scale)))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if heads.shape[0]:
            heads = heads * np.array([new_w / w0, new_h / h0], dtype=np.float64)
    return PanoCrowdSample(
        image=image,
        heads=heads,
        human_num=human_num,
        density_bin=density_bin(human_num),
        image_path=item.image_path,
        json_path=item.json_path,
    )


def _read_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            ids.append(s)
    return ids
