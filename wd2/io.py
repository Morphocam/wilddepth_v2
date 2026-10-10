"""Data access for the WildDepth-v2 RGB-LiDAR-radar subset (CPS2 rig, Longleat July 8).

Layout of one snippet directory (``$WD2_RADAR_ROOT/snippet_N``)::

    rgb/<id>.png                 640x480 RealSense colour
    lidar/<id>.pcd               Livox, fields x y z rgb (binary float32)
    radar/<id>.pcd               Altos V2, fields x y z h s v (binary float32, zero-padded)
    depthm/<id>.png              projected LiDAR, uint16 millimetres, 0 = no return
    refined_extrinsics/<id>.txt  4x4 LiDAR->camera (fallback: refined_extrinsic.txt)
    radar_match_log.csv          per-frame radar-minus-LiDAR timestamp offset
    frame_errors.csv             per-frame extrinsic rot/trans error estimate

The integer ``<id>`` is a global 10 Hz frame index shared by all sensors.
"""
from __future__ import annotations

import csv
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
RADAR_ROOT = Path(os.environ.get("WD2_RADAR_ROOT", "/mnt/hdd/WildDepth/radar/snippets"))
CACHE_ROOT = Path(os.environ.get("WD2_CACHE", "/mnt/hdd/WildDepth/wd2_cache"))
W, H = 640, 480


def intrinsic() -> np.ndarray:
    return np.loadtxt(REPO / "configs/calib/intrinsic_realsense.txt").reshape(3, 3)


def radar2lidar() -> np.ndarray:
    return np.loadtxt(REPO / "configs/calib/radar2lidar.txt").reshape(4, 4)


# ---------------------------------------------------------------- PCD parsing
def _parse_pcd(path: os.PathLike) -> tuple[list[str], np.ndarray]:
    raw = Path(path).read_bytes()
    marker = b"DATA binary\n"
    idx = raw.find(marker)
    if idx < 0:
        raise ValueError(f"{path}: not a binary PCD")
    fields, n = [], 0
    for line in raw[:idx].decode("latin1").splitlines():
        p = line.split()
        if p and p[0] == "FIELDS":
            fields = p[1:]
        elif p and p[0] == "POINTS":
            n = int(p[1])
    arr = np.frombuffer(raw[idx + len(marker): idx + len(marker) + n * 4 * len(fields)],
                        dtype=np.float32).reshape(n, len(fields))
    return fields, arr[~np.all(arr[:, :3] == 0, axis=1)]  # drop zero padding


def load_radar(path: os.PathLike) -> dict[str, np.ndarray]:
    """Radar frame as dict x,y,z,h (Doppler m/s),s (strength),v (coarse motion class)."""
    fields, arr = _parse_pcd(path)
    out = {f: arr[:, i].astype(np.float64) for i, f in enumerate(fields)}
    for k in ("x", "y", "z", "h", "s", "v"):
        out.setdefault(k, np.zeros(len(arr)))
    return out


def load_lidar_xyz(path: os.PathLike) -> np.ndarray:
    """(N,3) LiDAR points in the LiDAR frame, full resolution."""
    return _parse_pcd(path)[1][:, :3].astype(np.float64)


# ---------------------------------------------------------------- snippets
def list_snippets(root: Path = RADAR_ROOT) -> list[str]:
    """Snippet names containing radar+lidar. Raises instead of silently returning []."""
    root = Path(root)
    snips = [d.name for d in root.iterdir()
             if d.is_dir() and (d / "radar").is_dir() and (d / "lidar").is_dir()]
    if not snips:
        raise FileNotFoundError(f"no snippet_*/radar dirs under {root}; set WD2_RADAR_ROOT")
    return sorted(snips, key=lambda s: int(s.split("_")[1]))


def list_frames(snip: str, root: Path = RADAR_ROOT) -> list[int]:
    """Frame ids with RGB, LiDAR and radar all present (the synchronized triples)."""
    d = Path(root) / snip
    # size check: one LiDAR file in the release is 0 bytes (snippet_2/lidar/3000.pcd)
    ids = [{int(p.stem) for p in (d / sub).glob(f"*.{ext}") if p.stat().st_size > 0}
           for sub, ext in (("rgb", "png"), ("lidar", "pcd"), ("radar", "pcd"))]
    return sorted(ids[0] & ids[1] & ids[2])


def snippet_dir(snip: str) -> Path:
    return RADAR_ROOT / snip


def load_rgb(snip: str, f: int) -> np.ndarray:
    return np.asarray(Image.open(snippet_dir(snip) / "rgb" / f"{f}.png").convert("RGB"))


def load_depthm(snip: str, f: int) -> np.ndarray:
    """Projected LiDAR depth in metres (0 = none)."""
    return np.asarray(Image.open(snippet_dir(snip) / "depthm" / f"{f}.png")).astype(np.float64) / 1000.0


def lidar2cam(snip: str, f: int) -> np.ndarray:
    d = snippet_dir(snip)
    p = d / "refined_extrinsics" / f"{f}.txt"
    return np.loadtxt(p if p.exists() else d / "refined_extrinsic.txt").reshape(4, 4)


def radar_path(snip: str, f: int) -> Path:
    return snippet_dir(snip) / "radar" / f"{f}.pcd"


def lidar_path(snip: str, f: int) -> Path:
    return snippet_dir(snip) / "lidar" / f"{f}.pcd"


@lru_cache(maxsize=64)
def match_log(snip: str) -> dict[int, float]:
    p = snippet_dir(snip) / "radar_match_log.csv"
    out: dict[int, float] = {}
    if p.exists():
        for row in csv.DictReader(p.open()):
            try:
                out[int(row["frame_id"])] = float(row["delta_s_radar_minus_lidar"])
            except (KeyError, ValueError):
                pass
    return out


def radar_frozen(snip: str, sample: int = 12) -> bool:
    """True if sampled radar files are byte-identical (the July-7 stale-sensor failure)."""
    files = sorted((snippet_dir(snip) / "radar").glob("*.pcd"), key=lambda p: int(p.stem))
    if len(files) < 2:
        return False
    step = max(1, len(files) // sample)
    return len({hashlib.md5(f.read_bytes()).hexdigest() for f in files[::step]}) == 1


# ---------------------------------------------------------------- cached predictions
def cache_path(snip: str, f: int) -> Path:
    return CACHE_ROOT / snip / f"{f}.npz"


def load_cache(snip: str, f: int) -> dict | None:
    """Precomputed per-frame outputs of scripts/00_precompute.py, or None.

    Keys: boxes (K,4) xyxy, conf (K,), masks (K,H,W) bool, moge (H,W) float16 metres.
    """
    p = cache_path(snip, f)
    if not p.exists():
        return None
    z = np.load(p)
    masks = np.unpackbits(z["masks_packed"], axis=-1)[..., : W].astype(bool) \
        if z["masks_packed"].size else np.zeros((0, H, W), bool)
    return {"boxes": z["boxes"], "conf": z["conf"], "masks": masks,
            "moge": z["moge"].astype(np.float32)}
