"""Minimal tracking-by-detection and metric Kalman smoothing.

The 2D tracker is SORT-style IoU association with Hungarian matching (no appearance),
which is adequate for the slow, sparse animals in this subset and has no extra weights.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


class IoUTracker:
    def __init__(self, iou_thr: float = 0.3, max_age: int = 5):
        self.iou_thr, self.max_age = iou_thr, max_age
        self.tracks: dict[int, dict] = {}
        self.next_id = 0

    def update(self, frame: int, boxes: np.ndarray) -> list[int]:
        """Returns a track id for each box."""
        live = [t for t, v in self.tracks.items() if frame - v["last"] <= self.max_age]
        ids = [-1] * len(boxes)
        if live and len(boxes):
            tb = np.array([self.tracks[t]["box"] for t in live])
            M = iou(tb, boxes)
            r, c = linear_sum_assignment(-M)
            for i, j in zip(r, c):
                if M[i, j] >= self.iou_thr:
                    ids[j] = live[i]
        for j, b in enumerate(boxes):
            if ids[j] < 0:
                ids[j] = self.next_id
                self.next_id += 1
            self.tracks[ids[j]] = {"box": b, "last": frame}
        return ids


def kalman_cv_smooth(t: np.ndarray, z: np.ndarray, r_meas: np.ndarray, q: float = 0.5):
    """Constant-velocity Kalman filter + RTS smoother in 3D.

    t: (T,) seconds; z: (T,3) positions with NaN rows for missing measurements;
    r_meas: (T,) measurement std in metres; q: acceleration noise (m/s^2).
    Returns smoothed positions (T,3) and velocities (T,3).
    """
    T = len(t)
    n = 6
    x = np.zeros((T, n))
    P = np.zeros((T, n, n))
    xp = np.zeros((T, n))
    Pp = np.zeros((T, n, n))
    H = np.hstack([np.eye(3), np.zeros((3, 3))])
    first = np.flatnonzero(np.isfinite(z[:, 0]))
    if len(first) == 0:
        return np.full((T, 3), np.nan), np.full((T, 3), np.nan)
    xk = np.concatenate([z[first[0]], np.zeros(3)])
    Pk = np.diag([r_meas[first[0]] ** 2] * 3 + [4.0] * 3)
    Fs = []
    for k in range(T):
        dt = t[k] - t[k - 1] if k else 0.0
        F = np.eye(n)
        F[:3, 3:] = dt * np.eye(3)
        G = np.vstack([0.5 * dt ** 2 * np.eye(3), dt * np.eye(3)])
        Q = q ** 2 * G @ G.T
        xk, Pk = F @ xk, F @ Pk @ F.T + Q
        Fs.append(F)
        xp[k], Pp[k] = xk, Pk
        if np.isfinite(z[k, 0]):
            R = r_meas[k] ** 2 * np.eye(3)
            S = H @ Pk @ H.T + R
            K = Pk @ H.T @ np.linalg.inv(S)
            xk = xk + K @ (z[k] - H @ xk)
            Pk = (np.eye(n) - K @ H) @ Pk
        x[k], P[k] = xk, Pk
    xs = x.copy()
    for k in range(T - 2, -1, -1):
        C = P[k] @ Fs[k + 1].T @ np.linalg.inv(Pp[k + 1])
        xs[k] = x[k] + C @ (xs[k + 1] - xp[k + 1])
    return xs[:, :3], xs[:, 3:]
