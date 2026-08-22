"""SegFormer ADE20K importance is resolved via model.config.id2label names."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_segformer_importance_resolves_person_car_via_id2label() -> None:
    from energy.semantic import lut_from_id2label, normalize_class_name
    from segmentation.segformer_ade20k import Ade20kSegformer
    from segmentation.taxonomy import importance_map

    id2label = Ade20kSegformer.load_id2label()
    assert len(id2label) >= 150

    seg = Ade20kSegformer()
    seg._id2label = id2label
    table = importance_map(seg.taxonomy)
    assert seg.taxonomy == "ade20k"
    assert "animal" not in table
    assert table["person"] == 1.0
    assert table["car"] >= 0.89

    lut = lut_from_id2label(seg.id2label, table)
    by_name = {normalize_class_name(v): int(k) for k, v in seg.id2label.items()}
    assert "person" in by_name and "car" in by_name
    assert float(lut[by_name["person"]]) == 1.0
    assert float(lut[by_name["car"]]) >= 0.89
    if "sky" in by_name:
        assert float(lut[by_name["sky"]]) <= 0.10
    if "animal" in by_name:
        assert float(lut[by_name["animal"]]) <= 0.10
