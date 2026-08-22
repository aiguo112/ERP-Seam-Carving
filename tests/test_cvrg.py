"""CVRG-Pano palette decode."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.cvrg_pano import decode_cvrg_mask  # noqa: E402
from segmentation.cvrg import CVRG_COLOR_TO_ID, CVRG_IMPORTANCE, CVRG_NAME_TO_ID  # noqa: E402


def test_cvrg_palette_decodes_person_and_car() -> None:
    h, w = 4, 6
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    person_rgb = next(k for k, v in CVRG_COLOR_TO_ID.items() if v == CVRG_NAME_TO_ID["person"])
    car_rgb = next(k for k, v in CVRG_COLOR_TO_ID.items() if v == CVRG_NAME_TO_ID["car"])
    rgb[1, 2] = person_rgb
    rgb[2, 4] = car_rgb
    cmap = decode_cvrg_mask(rgb)
    assert cmap[1, 2] == CVRG_NAME_TO_ID["person"]
    assert cmap[2, 4] == CVRG_NAME_TO_ID["car"]
    assert cmap[0, 0] == CVRG_NAME_TO_ID["background"]


def test_cvrg_person_outranks_sky() -> None:
    assert CVRG_IMPORTANCE["person"] > CVRG_IMPORTANCE["sky"]
    assert CVRG_IMPORTANCE["car"] > CVRG_IMPORTANCE["road"]
