"""Distance-error metrics with bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np

BINS = [(5, 20), (20, 30), (30, 55)]


def errors(pred, gt) -> dict:
    p, g = np.asarray(pred, float), np.asarray(gt, float)
    ok = np.isfinite(p) & np.isfinite(g) & (g > 0)
    p, g = p[ok], g[ok]
    if len(g) == 0:
        return dict(n=0)
    e = p - g
    ratio = np.maximum(p / g, g / np.maximum(p, 1e-6))
    return dict(n=int(len(g)), MAE=float(np.mean(np.abs(e))), medAE=float(np.median(np.abs(e))),
                RMSE=float(np.sqrt(np.mean(e ** 2))), AbsRel=float(np.mean(np.abs(e) / g)),
                d125=float(np.mean(ratio < 1.25)))


def bootstrap_ci(pred, gt, groups, stat: str = "AbsRel", n: int = 500, seed: int = 0):
    """95% CI resampling whole groups (e.g. frames) to respect within-frame correlation."""
    p, g, grp = np.asarray(pred, float), np.asarray(gt, float), np.asarray(groups)
    ok = np.isfinite(p) & np.isfinite(g)
    p, g, grp = p[ok], g[ok], grp[ok]
    ug = np.unique(grp)
    if len(ug) < 2:
        return (np.nan, np.nan)
    idx_by = {u: np.flatnonzero(grp == u) for u in ug}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx_by[u] for u in rng.choice(ug, len(ug))])
        vals.append(errors(p[pick], g[pick])[stat])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
