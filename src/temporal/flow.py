"""Dense-flow check: is a cyclic horizontal pan an adequate ERP motion model?

Scalar phase-correlation pan can look large while warping the image does
nothing. Residual J is then distance to a possibly-wrong ``S_pred``.

For each contiguous pair:

1. Scalar pan (phase correlation), as used for ``S_pred``.
2. Dense DIS optical flow with wrap in ``x`` (ERP longitude).
3. Residual dense flow AFTER rolling ``I_{t-1}`` by that pan. If the pan is
   real yaw, leftover ``(u, v)`` is small. If it is tilt / roll / parallax /
   an alias, leftover flow stays structured or undoes the pan.
4. Equatorial photometric residual: identity vs pan-warp vs dense-warp.

Clip verdict (fail-closed for headline rows)::

    small_motion     |pan| small, leftover flow small. ``S_pred ~= S_{t-1}``.
    yaw_ok           pan explains appearance; residual flow is small.
    replace_model    leftover tilt/parallax, pan vs dense disagree, or dense
                     warp beats pan by a lot. Horizontal ``S_pred`` is wrong.
    stride_too_sparse  neither pan nor dense warp explains appearance
                     (linspace gap is not a warp).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from io_erp import circular_pad
from temporal.jitter import EQUATOR_BAND, equatorial_horizontal_pan

# Residual flow after a correct yaw should sit inside these (px).
RESID_U_MAX = 8.0
RESID_V_MAX = 8.0
RESID_STD_U_MAX = 12.0
# |dense mean_u - pan| above this => phasecorr is not the dense yaw.
PAN_DENSE_DISAGREE = 32.0
SMALL_PAN = 16.0
# Photometric ratios (warped / raw). Lower is a better explanation of appearance.
PAN_HELPS = 0.85
PAN_NEUTRAL = 0.95
FLOW_HELPS = 0.90
DENSE_BEATS_PAN = 0.10

# Finer-stride certification (consecutive / small-gap pairs at W=512).
AGREE_PX = 8.0
FAST_PX = 8.0
CERT_PHOTO_PAN = 0.85


@dataclass(frozen=True)
class PairFlow:
    pan: float
    mean_u: float
    mean_v: float
    std_u: float
    std_v: float
    resid_mean_u: float
    resid_mean_v: float
    resid_std_u: float
    photo_raw: float
    photo_pan: float
    photo_flow: float
    photo_resid: float


@dataclass(frozen=True)
class ClipFlowVerdict:
    label: str
    reason: str
    n_pairs: int
    pan_mean_abs: float
    mean_u: float
    mean_v: float
    std_u: float
    resid_mean_u: float
    resid_mean_v: float
    resid_std_u: float
    photo_raw: float
    photo_pan_ratio: float
    photo_flow_ratio: float
    photo_resid_ratio: float
    pan_dense_gap: float


def _gray(img: np.ndarray) -> NDArray[np.float32]:
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    rgb = arr[..., :3].astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def equator_slice(height: int, band: float = EQUATOR_BAND) -> slice:
    lo = int(band * height)
    hi = int((1.0 - band) * height)
    return slice(lo, max(hi, lo + 1))


def _gray_u8(img: np.ndarray) -> NDArray[np.uint8]:
    g = _gray(img)
    return np.clip(g, 0, 255).astype(np.uint8)


def dense_flow_cyclic(prev: np.ndarray, nxt: np.ndarray) -> NDArray[np.float32]:
    """DIS dense flow, wrap-padded in longitude. ``u`` is +x (same sign as np.roll).

    DIS is reliable on modest displacements. Uncompensated mean_u is *not* used
    to confirm a 100px+ pan; residual flow after the pan-warp is.
    """
    g0 = _gray_u8(prev)
    g1 = _gray_u8(nxt)
    if g0.shape != g1.shape:
        raise ValueError(f"flow shapes {g0.shape} vs {g1.shape}")
    h, w = g0.shape
    pad = min(64, max(8, w // 8))
    p0 = circular_pad(g0, pad)
    p1 = circular_pad(g1, pad)
    dis = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow_pad = dis.calc(p0, p1, None)
    return np.ascontiguousarray(flow_pad[pad : pad + h, pad : pad + w], dtype=np.float32)


def warp_cyclic_flow(img: np.ndarray, flow: np.ndarray) -> NDArray[np.float32]:
    """Backward-warp ``img`` by flow so dest ~= next if flow maps prev -> next.

    OpenCV Farneback: ``next(x+u, y+v) ~ prev(x, y)``. Aligning prev onto next
    is ``dest(x, y) = prev(x - u, y - v)`` with wrap in ``x``.
    """
    g = _gray(img)
    h, w = g.shape
    fl = np.asarray(flow, dtype=np.float32)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.mod(xs - fl[..., 0], float(w)).astype(np.float32)
    map_y = np.clip(ys - fl[..., 1], 0.0, float(h - 1)).astype(np.float32)
    return cv2.remap(g, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def equator_flow_stats(flow: np.ndarray) -> tuple[float, float, float, float]:
    eq = flow[equator_slice(flow.shape[0])]
    u = eq[..., 0].astype(np.float64)
    v = eq[..., 1].astype(np.float64)
    return float(u.mean()), float(v.mean()), float(u.std()), float(v.std())


def equator_photo(a: np.ndarray, b: np.ndarray) -> float:
    ga = _gray(a) if a.ndim == 3 else np.asarray(a, dtype=np.float32)
    gb = _gray(b) if b.ndim == 3 else np.asarray(b, dtype=np.float32)
    sl = equator_slice(ga.shape[0])
    return float(np.mean(np.abs(ga[sl].astype(np.float32) - gb[sl].astype(np.float32))))


def pair_flow_report(prev: np.ndarray, nxt: np.ndarray) -> PairFlow:
    pan = equatorial_horizontal_pan(prev, nxt)
    flow = dense_flow_cyclic(prev, nxt)
    mean_u, mean_v, std_u, std_v = equator_flow_stats(flow)
    shifted = np.roll(prev, int(round(pan)), axis=1)
    resid = dense_flow_cyclic(shifted, nxt)
    ru, rv, rsu, _rsv = equator_flow_stats(resid)
    g_next = _gray(nxt)
    raw = equator_photo(prev, nxt)
    photo_pan = equator_photo(np.roll(_gray(prev), int(round(pan)), axis=1), g_next)
    photo_flow = equator_photo(warp_cyclic_flow(prev, flow), g_next)
    photo_resid = equator_photo(warp_cyclic_flow(shifted, resid), g_next)
    return PairFlow(
        pan=float(pan),
        mean_u=mean_u,
        mean_v=mean_v,
        std_u=std_u,
        std_v=std_v,
        resid_mean_u=ru,
        resid_mean_v=rv,
        resid_std_u=rsu,
        photo_raw=raw,
        photo_pan=photo_pan,
        photo_flow=photo_flow,
        photo_resid=photo_resid,
    )


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def verdict_from_means(
    *,
    pan_abs: float,
    mean_u: float,
    resid_u: float,
    resid_v: float,
    resid_std_u: float,
    photo_pan_ratio: float,
    photo_flow_ratio: float,
) -> tuple[str, str]:
    """Fail-closed labels for a clip. See module docstring."""
    pan_dense_gap = abs(abs(mean_u) - pan_abs)
    resid_small = (
        abs(resid_u) < RESID_U_MAX
        and abs(resid_v) < RESID_V_MAX
        and resid_std_u < RESID_STD_U_MAX
    )

    if pan_abs < SMALL_PAN and photo_pan_ratio > PAN_NEUTRAL:
        if resid_small:
            return (
                "small_motion",
                "pan is small and leftover flow is small; S_pred is nearly S_{t-1}",
            )
        return (
            "replace_model",
            "claimed pan is small but leftover dense flow is structured",
        )

    if photo_pan_ratio >= PAN_NEUTRAL and photo_flow_ratio >= FLOW_HELPS:
        return (
            "stride_too_sparse",
            "neither pan nor dense warp cuts photometric residual; not a warp at this stride",
        )

    if photo_pan_ratio <= PAN_HELPS and resid_small:
        return "yaw_ok", "pan warp explains appearance; residual flow is small"

    if pan_dense_gap > PAN_DENSE_DISAGREE:
        return (
            "replace_model",
            "phase-correlation pan disagrees with dense mean u",
        )

    if abs(resid_v) >= RESID_V_MAX or resid_std_u >= RESID_STD_U_MAX:
        return (
            "replace_model",
            "leftover flow after pan has tilt or spatially varying u (not yaw)",
        )

    if photo_flow_ratio + DENSE_BEATS_PAN < photo_pan_ratio:
        return (
            "replace_model",
            "dense warp explains appearance substantially better than pan",
        )

    return (
        "replace_model",
        "horizontal pan does not pass the yaw checks (fail-closed)",
    )


def summarize_clip(pairs: list[PairFlow]) -> ClipFlowVerdict:
    if not pairs:
        return ClipFlowVerdict(
            "replace_model",
            "no pairs",
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            0.0,
        )
    pan_abs = _mean([abs(p.pan) for p in pairs])
    mean_u = _mean([p.mean_u for p in pairs])
    mean_v = _mean([p.mean_v for p in pairs])
    std_u = _mean([p.std_u for p in pairs])
    resid_u = _mean([p.resid_mean_u for p in pairs])
    resid_v = _mean([p.resid_mean_v for p in pairs])
    resid_std_u = _mean([p.resid_std_u for p in pairs])
    raw = _mean([p.photo_raw for p in pairs])
    pan_ph = _mean([p.photo_pan for p in pairs])
    flow_ph = _mean([p.photo_flow for p in pairs])
    resid_ph = _mean([p.photo_resid for p in pairs])
    pr = pan_ph / max(raw, 1e-6)
    fr = flow_ph / max(raw, 1e-6)
    rr = resid_ph / max(raw, 1e-6)
    label, reason = verdict_from_means(
        pan_abs=pan_abs,
        mean_u=mean_u,
        resid_u=resid_u,
        resid_v=resid_v,
        resid_std_u=resid_std_u,
        photo_pan_ratio=pr,
        photo_flow_ratio=fr,
    )
    return ClipFlowVerdict(
        label=label,
        reason=reason,
        n_pairs=len(pairs),
        pan_mean_abs=pan_abs,
        mean_u=mean_u,
        mean_v=mean_v,
        std_u=std_u,
        resid_mean_u=resid_u,
        resid_mean_v=resid_v,
        resid_std_u=resid_std_u,
        photo_raw=raw,
        photo_pan_ratio=pr,
        photo_flow_ratio=fr,
        photo_resid_ratio=rr,
        pan_dense_gap=abs(abs(mean_u) - pan_abs),
    )


def certify_motion(v: ClipFlowVerdict) -> tuple[str, str]:
    """Label a strided window: validated_fast / validated_slow / not_validated.

    Validated yaw: pan and dense mean u agree within ``AGREE_PX``, pan-warp
    photometric residual drops to ``CERT_PHOTO_PAN`` of raw, leftover v is
    small. Fast: that yaw is at least ``FAST_PX`` pixels per pair at W=512.
    """
    agree = v.pan_dense_gap <= AGREE_PX
    nearly_static = max(v.pan_mean_abs, abs(v.mean_u)) < 1.0
    photo_ok = v.photo_pan_ratio <= CERT_PHOTO_PAN or nearly_static
    tilt_ok = abs(v.resid_mean_v) < RESID_V_MAX and abs(v.mean_v) < max(
        0.5 * max(abs(v.mean_u), v.pan_mean_abs), RESID_V_MAX
    )
    fast = max(v.pan_mean_abs, abs(v.mean_u)) >= FAST_PX
    if agree and photo_ok and tilt_ok:
        if fast:
            return (
                "validated_fast",
                "pan matches dense u, pan-warp cuts photometry, leftover is not tilt",
            )
        if nearly_static:
            return (
                "validated_slow",
                "pan and dense u agree near 0; window is static (photo cannot improve)",
            )
        return (
            "validated_slow",
            "yaw model holds but per-pair shift is below the fast threshold",
        )
    bits = []
    if not agree:
        bits.append(f"|pan-dense|={v.pan_dense_gap:.1f}>{AGREE_PX:g}")
    if not photo_ok:
        bits.append(f"photo_pan={v.photo_pan_ratio:.2f}>{CERT_PHOTO_PAN:g}")
    if not tilt_ok:
        bits.append(f"v={v.mean_v:.1f} resid_v={v.resid_mean_v:.1f}")
    return "not_validated", "; ".join(bits) if bits else "failed yaw checks"


def flow_to_bgr(flow: np.ndarray) -> NDArray[np.uint8]:
    """HSV flow viz (OpenCV style). Magnitude scaled to the clip max."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2.0).astype(np.uint8)
    hsv[..., 1] = 255
    scale = float(np.max(mag)) if np.max(mag) > 1e-6 else 1.0
    hsv[..., 2] = np.clip(mag / scale * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
