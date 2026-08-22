"""Dataset locations. Nothing under these paths is shipped in the repo."""

from __future__ import annotations

import os
from pathlib import Path


def env_path(*names: str) -> Path | None:
    for name in names:
        raw = os.environ.get(name, "").strip().strip('"')
        if raw:
            return Path(raw)
    return None


def panocount_root() -> Path | None:
    """PanoCount stills + head-point JSON (request-based; not in this repo)."""
    return env_path("PANOCOUNT_ROOT", "PANOCROWD_ROOT")


def cvrg_root() -> Path | None:
    return env_path("CVRG_PANO_ROOT", "CVRG_ROOT")


def erp2_video() -> Path | None:
    return env_path("ERP2_VIDEO")


def video_root() -> Path | None:
    return env_path("ERP_VIDEO_ROOT", "PAVS10K_ROOT")


def default_videos() -> list[Path]:
    """Optional local clips via ERP2_VIDEO and ERP_VIDEOS (os.pathsep-separated)."""
    found: list[Path] = []
    p = erp2_video()
    if p is not None:
        found.append(p)
    extra = os.environ.get("ERP_VIDEOS", "").strip()
    if extra:
        found.extend(Path(part.strip()) for part in extra.split(os.pathsep) if part.strip())
    root = video_root()
    if root is not None and root.is_dir():
        found.extend(sorted(root.glob("ERP*.mp4")))
    seen: set[str] = set()
    unique: list[Path] = []
    for path in found:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def require_path(path: Path | None, *, flag: str, env: str) -> Path:
    if path is None:
        raise SystemExit(
            f"Datasets are not included in this repository. "
            f"Pass {flag} or set {env}."
        )
    return path
