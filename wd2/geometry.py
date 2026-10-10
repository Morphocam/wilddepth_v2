"""Projection, association and robust statistics shared by all experiments."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .io import H, W

# Anisotropic radar<->LiDAR association radii (x forward, y lateral, z up), metres.
# The long z axis absorbs the 1-2 m elevation error of the radar extrinsic.
RXYZ = np.array([0.6, 1.0, 1.5])


def transform(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    return (T[:3, :3] @ P.T + T[:3, 3:4]).T


def project(P_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Camera-frame points -> (u, v, z, in_image mask)."""
    z = P_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * P_cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * P_cam[:, 1] / z + K[1, 2]
    ok = (z > 0.5) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, z, ok


def radar_to_cam(radar: dict, R2L: np.ndarray, L2C: np.ndarray) -> np.ndarray:
    P = np.stack([radar["x"], radar["y"], radar["z"]], 1)
    return transform(L2C, transform(R2L, P))


def robust_median(z: np.ndarray, min_n: int = 3) -> float | None:
    """Median after a MAD band that drops background leaking into a mask/box."""
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    if len(z) < min_n:
        return None
    m = np.median(z)
    mad = np.median(np.abs(z - m)) + 1e-6
    keep = (np.abs(z - m) <= max(1.5, 4 * mad)) & (z < m + 3.0)
    return float(np.median(z[keep])) if keep.sum() >= min_n else float(m)


def in_box(u, v, box, pad: float = 0.0):
    x0, y0, x1, y1 = box
    return (u >= x0 - pad) & (u <= x1 + pad) & (v >= y0 - pad) & (v <= y1 + pad)


def mask_values(u, v, ok, mask: np.ndarray) -> np.ndarray:
    """Boolean selector of projected points falling inside a binary mask."""
    sel = np.zeros(len(u), bool)
    ui = np.clip(u[ok].astype(int), 0, W - 1)
    vi = np.clip(v[ok].astype(int), 0, H - 1)
    sel[np.flatnonzero(ok)] = mask[vi, ui]
    return sel


def fit_affine(d_mde, d_metric, iters: int = 5, keep_pct: float = 80):
    """Trimmed least squares D_metric = s*D_mde + t. Returns (s,t) or None."""
    x, y = np.asarray(d_mde, float), np.asarray(d_metric, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 or np.ptp(x) < 1e-3:
        return None
    keep = np.ones(len(x), bool)
    s = t = 0.0
    for _ in range(iters):
        A = np.stack([x[keep], np.ones(keep.sum())], 1)
        (s, t), *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        res = np.abs(s * x + t - y)
        new = res <= np.percentile(res, keep_pct)
        if new.sum() < 2:
            break
        keep = new
    return float(s), float(t)


def fit_scale(d_mde, d_metric):
    """Scale-only fit (robust median ratio) -- used when only one anchor exists."""
    x, y = np.asarray(d_mde, float), np.asarray(d_metric, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if ok.sum() < 1:
        return None
    return float(np.median(y[ok] / x[ok])), 0.0


def doppler_transfer(lidar_xyz: np.ndarray, radar_xyz_l: np.ndarray, radar_h: np.ndarray,
                     radii: np.ndarray = RXYZ) -> np.ndarray:
    """Each LiDAR point inherits |Doppler| of the nearest radar return inside an ellipsoid.

    Both clouds in the LiDAR frame. Returns per-LiDAR |h| (NaN where no radar within radii).
    """
    out = np.full(len(lidar_xyz), np.nan)
    if len(radar_xyz_l) == 0 or len(lidar_xyz) == 0:
        return out
    tree = cKDTree(radar_xyz_l / radii)
    d, j = tree.query(lidar_xyz / radii, distance_upper_bound=1.0)
    hit = np.isfinite(d)
    out[hit] = np.abs(radar_h[j[hit]])
    return out


def ego_motion_fit(radar: dict, iters: int = 5, tol: float = 0.3):
    """Fit Doppler h(az) = vx cos(az) + vy sin(az) + c with iterative inlier refit.

    A static scene seen from a moving platform follows this cosine, so a frame is
    ego-motion if the fit explains the Doppler (R^2 > 0.5) or implies a real platform
    speed (|V| > 1 m/s with > 70% inliers) -- same rule as the original radar study.
    Returns (speed m/s, R^2, inlier fraction, is_ego).
    """
    x, y, h = radar["x"], radar["y"], radar["h"]
    if len(h) < 10:
        return 0.0, 0.0, 0.0, False
    az = np.arctan2(y, x)
    A = np.stack([np.cos(az), np.sin(az), np.ones_like(az)], 1)
    idx = np.ones(len(h), bool)
    c = np.zeros(3)
    for _ in range(iters):
        c, *_ = np.linalg.lstsq(A[idx], h[idx], rcond=None)
        idx = np.abs(A @ c - h) < tol
        if idx.sum() < 10:
            break
    V = float(np.hypot(c[0], c[1]))
    inl = float(np.mean(np.abs(A @ c - h) < tol))
    r2 = float(1 - np.sum((A @ c - h) ** 2) / max(1e-9, np.sum((h - h.mean()) ** 2)))
    return V, r2, inl, (r2 > 0.5) or (V > 1.0 and inl > 0.7)
