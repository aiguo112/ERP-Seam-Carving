"""Discover ERP video clips (PAVS10K layout or a flat folder of mp4/mkv)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

# Empty / faded frames: skip at sample time. Dark-but-valid VR (letterbox) is kept.
NEAR_BLACK_LUMA = 8.0
NEAR_BLACK_FRAC = 0.90
NEAR_BLACK_PIXEL = 8.0


@dataclass(frozen=True)
class VideoClip:
    path: Path
    name: str


@dataclass(frozen=True)
class SampledClip:
    """Evenly sampled frames after dropping near-black / decode failures."""

    frames: list[NDArray[np.uint8]]
    fps: float
    source_indices: NDArray[np.int32]
    sample_slots: NDArray[np.int32]
    n_attempted: int
    n_dropped_black: int
    n_dropped_decode: int

    @property
    def n_dropped(self) -> int:
        return int(self.n_dropped_black + self.n_dropped_decode)

    def contiguous_pairs(self) -> list[tuple[int, int]]:
        """Kept-list index pairs whose sample slots were adjacent (no drop between)."""
        return contiguous_pairs(self.sample_slots)


def list_clips(root: str | Path) -> list[VideoClip]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"video root does not exist: {root}")
    if root.is_file() and root.suffix.lower() in VIDEO_EXTS:
        return [VideoClip(root, root.stem)]
    found: list[VideoClip] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        if any(p in path.parts for p in ("2x2", "3x3", "crop")):
            continue
        found.append(VideoClip(path, path.stem))
    return found


def frame_is_near_black(
    img: np.ndarray,
    *,
    luma_min: float = NEAR_BLACK_LUMA,
    black_frac: float = NEAR_BLACK_FRAC,
    pixel: float = NEAR_BLACK_PIXEL,
) -> bool:
    """True for empty/faded frames, not for letterboxed-but-visible ERP."""
    arr = np.asarray(img)
    g = arr.mean(axis=2) if arr.ndim == 3 else arr.astype(np.float64)
    return float(g.mean()) < luma_min or float((g <= pixel).mean()) > black_frac


def contiguous_pairs(sample_slots: np.ndarray) -> list[tuple[int, int]]:
    """Pairs (k-1, k) in the kept list whose linspace slots differ by 1.

    A dropped (black/failed) sample between two kept frames is a gap: do not
    measure jitter across it. Intended linspace spacing is not a gap.
    """
    slots = np.asarray(sample_slots, dtype=np.int32).reshape(-1)
    pairs: list[tuple[int, int]] = []
    for k in range(1, slots.size):
        if int(slots[k]) == int(slots[k - 1]) + 1:
            pairs.append((k - 1, k))
    return pairs


def sample_frames(
    path: str | Path,
    *,
    max_frames: int = 48,
    max_width: int = 640,
    skip_near_black: bool = True,
) -> SampledClip:
    """Evenly sample frames, resize keeping aspect, drop near-black at read time."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if n < 2:
        cap.release()
        raise ValueError(f"video too short: {path}")
    take = min(int(max_frames), n)
    idxs = np.linspace(0, n - 1, take, dtype=int)
    frames: list[NDArray[np.uint8]] = []
    source_indices: list[int] = []
    sample_slots: list[int] = []
    n_dropped_black = 0
    n_dropped_decode = 0
    for slot, i in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            n_dropped_decode += 1
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if max_width is not None and w > max_width:
            scale = max_width / w
            rgb = cv2.resize(
                rgb,
                (int(max_width), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        if skip_near_black and frame_is_near_black(rgb):
            n_dropped_black += 1
            continue
        frames.append(rgb)
        source_indices.append(int(i))
        sample_slots.append(int(slot))
    cap.release()
    if len(frames) < 2:
        raise ValueError(f"could not decode frames from {path}")
    return SampledClip(
        frames=frames,
        fps=fps,
        source_indices=np.asarray(source_indices, dtype=np.int32),
        sample_slots=np.asarray(sample_slots, dtype=np.int32),
        n_attempted=int(take),
        n_dropped_black=n_dropped_black,
        n_dropped_decode=n_dropped_decode,
    )


def sample_strided_frames(
    path: str | Path,
    *,
    stride: int,
    n_frames: int = 20,
    start_frac: float = 0.45,
    max_width: int = 512,
    skip_near_black: bool = True,
) -> SampledClip:
    """Read ``n_frames`` at a fixed source stride (not linspace across the clip).

    ``start_frac`` places the window in the file (0 = start, 0.5 = middle).
    Consecutive decode with ``grab`` so stride-1 does not seek every frame.
    """
    if int(stride) < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    take = min(int(n_frames), n)
    span = (take - 1) * int(stride)
    if n < 2 or span < 1:
        cap.release()
        raise ValueError(f"video too short for stride={stride}: {path}")
    start = int(np.clip(float(start_frac) * (n - 1 - span), 0, max(n - 1 - span, 0)))
    wanted = [start + k * int(stride) for k in range(take)]
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames: list[NDArray[np.uint8]] = []
    source_indices: list[int] = []
    sample_slots: list[int] = []
    n_dropped_black = 0
    n_dropped_decode = 0
    current = start
    for slot, want in enumerate(wanted):
        while current < want:
            if not cap.grab():
                n_dropped_decode += take - slot
                current = want + 1
                break
            current += 1
        if current != want:
            continue
        ok, bgr = cap.read()
        current += 1
        if not ok or bgr is None:
            n_dropped_decode += 1
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if max_width is not None and w > max_width:
            scale = max_width / w
            rgb = cv2.resize(
                rgb,
                (int(max_width), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        if skip_near_black and frame_is_near_black(rgb):
            n_dropped_black += 1
            continue
        frames.append(rgb)
        source_indices.append(int(want))
        sample_slots.append(int(slot))
    cap.release()
    if len(frames) < 2:
        raise ValueError(f"could not decode strided frames from {path}")
    return SampledClip(
        frames=frames,
        fps=fps,
        source_indices=np.asarray(source_indices, dtype=np.int32),
        sample_slots=np.asarray(sample_slots, dtype=np.int32),
        n_attempted=int(take),
        n_dropped_black=n_dropped_black,
        n_dropped_decode=n_dropped_decode,
    )
