"""Dense-flow yaw check (synthetic; no video files)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from temporal.flow import (  # noqa: E402
    dense_flow_cyclic,
    equator_flow_stats,
    pair_flow_report,
    verdict_from_means,
    certify_motion,
    ClipFlowVerdict,
)


def _textured(h: int = 64, w: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = (40 + 90 * rng.random((h, w, 3))).astype(np.uint8)
    img[:, 20:28] = 220
    img[10:18, :] = 30
    return img


def test_verdict_small_motion() -> None:
    label, _ = verdict_from_means(
        pan_abs=3.0,
        mean_u=2.0,
        resid_u=1.0,
        resid_v=0.5,
        resid_std_u=2.0,
        photo_pan_ratio=0.99,
        photo_flow_ratio=0.98,
    )
    assert label == "small_motion"


def test_verdict_stride_too_sparse() -> None:
    label, _ = verdict_from_means(
        pan_abs=80.0,
        mean_u=10.0,
        resid_u=20.0,
        resid_v=15.0,
        resid_std_u=30.0,
        photo_pan_ratio=0.99,
        photo_flow_ratio=0.97,
    )
    assert label == "stride_too_sparse"


def test_verdict_yaw_ok() -> None:
    label, _ = verdict_from_means(
        pan_abs=40.0,
        mean_u=38.0,
        resid_u=1.0,
        resid_v=0.4,
        resid_std_u=3.0,
        photo_pan_ratio=0.50,
        photo_flow_ratio=0.48,
    )
    assert label == "yaw_ok"


def test_verdict_replace_when_dense_beats_pan() -> None:
    label, _ = verdict_from_means(
        pan_abs=40.0,
        mean_u=8.0,
        resid_u=2.0,
        resid_v=1.0,
        resid_std_u=4.0,
        photo_pan_ratio=0.92,
        photo_flow_ratio=0.50,
    )
    assert label == "replace_model"


def test_dense_flow_recovers_yaw_roll() -> None:
    img = _textured()
    rolled = np.roll(img, 9, axis=1)
    flow = dense_flow_cyclic(img, rolled)
    mu, mv, _su, _sv = equator_flow_stats(flow)
    assert abs(mu - 9.0) < 2.0
    assert abs(mv) < 2.0


def test_residual_flow_after_correct_pan_is_small() -> None:
    img = _textured(seed=1)
    rolled = np.roll(img, 11, axis=1)
    rep = pair_flow_report(img, rolled)
    assert abs(rep.pan - 11.0) < 1.0
    assert abs(rep.resid_mean_u) < 3.0
    assert abs(rep.resid_mean_v) < 3.0
    assert rep.photo_pan < 0.6 * rep.photo_raw


def _verdict(**kwargs) -> ClipFlowVerdict:
    base = dict(
        label="x",
        reason="",
        n_pairs=8,
        pan_mean_abs=12.0,
        mean_u=11.0,
        mean_v=0.4,
        std_u=3.0,
        resid_mean_u=1.0,
        resid_mean_v=0.5,
        resid_std_u=3.0,
        photo_raw=20.0,
        photo_pan_ratio=0.50,
        photo_flow_ratio=0.48,
        photo_resid_ratio=0.47,
        pan_dense_gap=1.0,
    )
    base.update(kwargs)
    return ClipFlowVerdict(**base)


def test_certify_validated_fast() -> None:
    label, _ = certify_motion(_verdict())
    assert label == "validated_fast"


def test_certify_validated_slow() -> None:
    label, _ = certify_motion(
        _verdict(pan_mean_abs=3.0, mean_u=2.5, pan_dense_gap=0.5)
    )
    assert label == "validated_slow"


def test_certify_static_window_is_slow() -> None:
    label, _ = certify_motion(
        _verdict(
            pan_mean_abs=0.02,
            mean_u=0.01,
            pan_dense_gap=0.01,
            photo_pan_ratio=1.0,
            mean_v=0.0,
            resid_mean_v=0.0,
        )
    )
    assert label == "validated_slow"


def test_certify_rejects_disagreement() -> None:
    label, _ = certify_motion(_verdict(pan_mean_abs=40.0, mean_u=2.0, pan_dense_gap=38.0))
    assert label == "not_validated"
