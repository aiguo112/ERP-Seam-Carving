"""Vertical-seam dynamic programming.

A vertical seam is one pixel per row, 8-connected (column changes by at
most 1). ``cyclic=True`` wraps that connectivity across ``x=0`` / ``x=W-1``:
the predecessor set of ``(y, x)`` is ``{(x-1) mod W, x, (x+1) mod W}``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cyclic_predecessors(x: int, width: int) -> tuple[int, int, int]:
    """8-connected predecessor columns of ``x``, computed modulo ``W``.

    ERP longitude is periodic: column ``0`` is adjacent to column ``W-1``.
    """
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    return ((x - 1) % width, x % width, (x + 1) % width)


def seam_wraps(seam: np.ndarray, width: int) -> bool:
    """True if any 8-connected step crosses the ``x=0`` / ``x=W-1`` cut."""
    s = np.asarray(seam, dtype=np.int32)
    if s.size < 2 or width < 2:
        return False
    return bool(np.any(np.abs(np.diff(s)) == width - 1))


def find_vertical_seam(energy: np.ndarray, cyclic: bool = False) -> NDArray[np.int32]:
    """Min-energy top-to-bottom seam: column index per row, shape ``(H,)``.

    Recurrence::

        M[0, x] = E[0, x]
        M[y, x] = E[y, x] + min_{x' in P(x)} M[y-1, x']

    ``P(x) = {x-1, x, x+1}``. If ``cyclic``, each index is taken **modulo W**
    so the seam may wrap across the ERP cut; it is still one pixel per row.
    If not cyclic, predecessors outside ``[0, W)`` are dropped.
    """
    e = np.asarray(energy, dtype=np.float32)
    if e.ndim != 2:
        raise ValueError(f"energy must be HxW, got {e.shape}")
    h, w = e.shape
    if h < 1 or w < 1:
        raise ValueError(f"energy too small: {e.shape}")

    m = np.empty((h, w), dtype=np.float32)
    m[0] = e[0]
    if w == 1:
        m[:, 0] = np.cumsum(e[:, 0])
        return np.zeros(h, dtype=np.int32)

    for y in range(1, h):
        prev = m[y - 1]
        if cyclic:
            # {x-1, x, x+1} modulo W: roll(+1) is M[y-1, x-1], roll(-1) is M[y-1, x+1].
            left = np.roll(prev, 1)
            right = np.roll(prev, -1)
            m[y] = e[y] + np.minimum(np.minimum(left, prev), right)
        else:
            left = np.empty(w, dtype=np.float32)
            right = np.empty(w, dtype=np.float32)
            left[0] = np.inf
            left[1:] = prev[:-1]
            right[-1] = np.inf
            right[:-1] = prev[1:]
            m[y] = e[y] + np.minimum(np.minimum(left, prev), right)

    seam = np.empty(h, dtype=np.int32)
    x = int(np.argmin(m[-1]))
    seam[-1] = x
    for y in range(h - 2, -1, -1):
        x = _min_predecessor(m[y], x, cyclic=cyclic)
        seam[y] = x
    return seam


def _min_predecessor(row: NDArray[np.float32], x: int, *, cyclic: bool) -> int:
    w = int(row.shape[0])
    if cyclic:
        candidates = cyclic_predecessors(x, w)
    else:
        candidates = tuple(c for c in (x - 1, x, x + 1) if 0 <= c < w)
    best_x = candidates[0]
    best_v = float(row[best_x])
    for c in candidates:
        v = float(row[c])
        if v < best_v or (v == best_v and c == x):
            best_v = v
            best_x = c
    return best_x
