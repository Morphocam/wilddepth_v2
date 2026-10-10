"""One synchronized frame with everything projected into the camera, for the experiments."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geometry as G
from . import io

MD_CONF = 0.3          # detection threshold used in all experiments
MIN_MASK_PX = 30


@dataclass
class Frame:
    snip: str
    f: int
    boxes: np.ndarray            # (K,4)
    conf: np.ndarray
    masks: np.ndarray            # (K,H,W) bool
    moge: np.ndarray             # (H,W) metres, 0 invalid
    lidar_l: np.ndarray          # (N,3) LiDAR frame
    lu: np.ndarray               # LiDAR projected u,v,z,in-image
    lv: np.ndarray
    lz: np.ndarray
    lok: np.ndarray
    radar: dict                  # raw radar dict
    radar_l: np.ndarray          # (M,3) radar in LiDAR frame
    ru: np.ndarray
    rv: np.ndarray
    rz: np.ndarray
    rok: np.ndarray
    L2C: np.ndarray
    extras: dict = field(default_factory=dict)


def load_frame(snip: str, f: int, extrinsic: str = "anchor", need_cache: bool = True) -> Frame | None:
    """extrinsic: 'per_frame' (refined_extrinsics/<id>.txt) or 'anchor' (refined_extrinsic.txt)."""
    if io.lidar_path(snip, f).stat().st_size == 0:
        return None
    c = io.load_cache(snip, f)
    if c is None and need_cache:
        return None
    if c is None:
        c = {"boxes": np.zeros((0, 4)), "conf": np.zeros(0),
             "masks": np.zeros((0, io.H, io.W), bool), "moge": np.zeros((io.H, io.W), np.float32)}
    keep = c["conf"] >= MD_CONF
    if len(keep) and len(c["masks"]) == len(keep):
        keep &= c["masks"].reshape(len(keep), -1).sum(1) >= MIN_MASK_PX
    K = io.intrinsic()
    L2C = (np.loadtxt(io.snippet_dir(snip) / "refined_extrinsic.txt").reshape(4, 4)
           if extrinsic == "anchor" else io.lidar2cam(snip, f))
    Pl = io.load_lidar_xyz(io.lidar_path(snip, f))
    lu, lv, lz, lok = G.project(G.transform(L2C, Pl), K)
    r = io.load_radar(io.radar_path(snip, f))
    Rl = G.transform(io.radar2lidar(), np.stack([r["x"], r["y"], r["z"]], 1))
    ru, rv, rz, rok = G.project(G.transform(L2C, Rl), K)
    return Frame(snip, f, c["boxes"][keep], c["conf"][keep], c["masks"][keep], c["moge"],
                 Pl, lu, lv, lz, lok, r, Rl, ru, rv, rz, rok, L2C)


def radar_in_box(fr: Frame, box, vpad_frac: float = 1.0) -> np.ndarray:
    """Radar returns associated to a box at the OBJECT level.

    Horizontal extent = the box (azimuth is accurate); vertical extent padded by
    vpad_frac * box height on both sides, because the 1-2 m elevation error of the
    extrinsic moves radar returns up/down in the image.
    """
    x0, y0, x1, y1 = box
    pad = vpad_frac * (y1 - y0)
    return fr.rok & (fr.ru >= x0) & (fr.ru <= x1) & (fr.rv >= y0 - pad) & (fr.rv <= y1 + pad)


def radar_range_estimates(z: np.ndarray, prior: float | None = None, gap: float = 2.0,
                          min_n: int = 3, prior_tol: float = 0.35) -> dict:
    """LiDAR-free estimates of an animal's range from the radar returns in its box.

    A box's returns mix the animal with background seen past it, so a plain median is
    biased far. Candidates (selected on held-out snippets, see e3_radar_scale.py):
      median   robust median of all returns
      nearest  first cluster in range with >= min_n returns within `gap` metres
      mode     densest `gap`-wide range window
      prior    returns within +-prior_tol of an RGB prior (MoGe-2 metric), then median
    """
    z = np.sort(np.asarray(z, float)[np.isfinite(z)])
    out = dict(median=None, nearest=None, mode=None, prior=None)
    if len(z) < min_n:
        return out
    out["median"] = G.robust_median(z)
    cnt = np.searchsorted(z, z + gap, side="right") - np.arange(len(z))
    ok = np.flatnonzero(cnt >= min_n)
    if len(ok):
        i = ok[0]
        out["nearest"] = float(np.median(z[i: i + cnt[i]]))
        j = int(np.argmax(cnt))
        out["mode"] = float(np.median(z[j: j + cnt[j]]))
    if prior is not None and prior > 0:
        sel = z[np.abs(z - prior) <= prior_tol * prior]
        if len(sel) >= min_n:
            out["prior"] = float(np.median(sel))
    return out


def lidar_in_mask(fr: Frame, k: int) -> np.ndarray:
    return G.mask_values(fr.lu, fr.lv, fr.lok, fr.masks[k])


def moge_in_mask(fr: Frame, k: int) -> float | None:
    v = fr.moge[fr.masks[k]]
    return G.robust_median(v[v > 0.1])
