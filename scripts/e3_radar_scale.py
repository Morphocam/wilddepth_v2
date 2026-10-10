"""E3: metric-scale calibration of MoGe-2 by anchor source, under ONE fair protocol.

Stage 1 (extract) -- per animal (MegaDetector >= 0.3, SAM2 mask, >= 10 single-sweep LiDAR
points in the mask, GT in 5-55 m) record:
  gt        robust median of a random half B of the mask's LiDAR points (GT for every method)
  lidarA    robust median of the other half A (held-out LC-box anchor -> not circular)
  moge      robust median of MoGe-2 depth in the mask
  radar_*   LiDAR-free object-level radar range estimates from returns projected into the box
            (vertical padding 1 box height for the elevation error), see
            wd2.frame.radar_range_estimates: median / nearest / mode / prior(MoGe-gated)
  radar_snap  the ORIGINAL radar-study anchor: radar returns snapped to their nearest LiDAR
            point (anisotropic ellipsoid) and the LiDAR depth used. Needs LiDAR at inference;
            reported only to explain the earlier number.
and per frame: UC scale (one LiDAR point nearest the image centre), LC_scene affine (all
LiDAR < 55 m), RC_pixel affine (all radar returns < 55 m as per-pixel anchors).

Stage 2 (summarize) -- per frame fit s*MoGe+t from the anchors of all animals in the frame
(affine with >= 2 anchors, scale-only with 1) and score against gt. The radar estimator is
chosen on DEV snippets (odd ids) and reported on TEST snippets (even ids).

    python scripts/e3_radar_scale.py extract [--extrinsic per_frame|anchor]
    python scripts/e3_radar_scale.py summarize [--tag per_frame]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zlib
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wd2 import geometry as G, io, metrics as M  # noqa: E402
from wd2.frame import (lidar_in_mask, load_frame, moge_in_mask, radar_in_box,  # noqa: E402
                       radar_range_estimates)

DEPTH_DIR = Path(os.environ["WD2_DEPTH_DIR"]) if os.environ.get("WD2_DEPTH_DIR") else None
RMAX = 55.0
MIN_LIDAR = 10
RADAR_EST = ["median", "nearest", "mode", "prior"]
OUT = io.REPO / "results"


def _num(x):
    return np.nan if x is None else float(x)


def extract_snippet(args):
    snip, frames, extrinsic = args
    rng = np.random.default_rng(zlib.crc32(snip.encode()))
    ego = {}
    p = OUT / "e10_ego_per_frame.csv"
    if p.exists():
        ego = {int(r["frame"]): int(r["is_ego"]) for r in csv.DictReader(p.open()) if r["snippet"] == snip}
    fe = io.snippet_dir(snip) / "frame_errors.csv"
    ferr = {int(r["stem"]): float(r["rot_err_deg"]) for r in csv.DictReader(fe.open())} if fe.exists() else {}
    recs = []
    for f in frames:
        fr = load_frame(snip, f, extrinsic=extrinsic)
        if fr is None or len(fr.boxes) == 0:
            continue
        moge_metric = fr.moge
        if DEPTH_DIR is not None:          # alternative MDE (e.g. Marigold); radar prior stays MoGe-2
            dp = DEPTH_DIR / snip / f"{f}.npy"
            if not dp.exists():
                continue
            fr.moge = np.load(dp).astype(np.float32)
        mo = fr.moge
        ui, vi = np.clip(fr.lu.astype(int), 0, io.W - 1), np.clip(fr.lv.astype(int), 0, io.H - 1)
        sc = fr.lok & (fr.lz < RMAX)
        dm = mo[vi[sc], ui[sc]]
        good = dm > 0.1
        lcs = G.fit_affine(dm[good], fr.lz[sc][good]) if good.sum() >= 3 else None
        cand = np.flatnonzero(fr.lok & (fr.lz < 150))
        cand = cand[mo[vi[cand], ui[cand]] > 0.1] if len(cand) else cand
        uc = None
        if len(cand):
            j = cand[np.argmin(np.hypot(fr.lu[cand] - io.W / 2, fr.lv[cand] - io.H / 2))]
            uc = fr.lz[j] / mo[vi[j], ui[j]]
        ru, rv = np.clip(fr.ru.astype(int), 0, io.W - 1), np.clip(fr.rv.astype(int), 0, io.H - 1)
        rs = fr.rok & (fr.rz < RMAX)
        rdm = mo[rv[rs], ru[rs]]
        rg = rdm > 0.1
        rcp = G.fit_affine(rdm[rg], fr.rz[rs][rg]) if rg.sum() >= 3 else None
        ltree = cKDTree(fr.lidar_l / G.RXYZ) if len(fr.lidar_l) else None
        lz_all = G.transform(fr.L2C, fr.lidar_l)[:, 2]
        for k in range(len(fr.boxes)):
            sel = lidar_in_mask(fr, k)
            z = fr.lz[sel]
            z = z[z < 150]
            mm = moge_in_mask(fr, k)
            if len(z) < MIN_LIDAR or mm is None:
                continue
            perm = rng.permutation(len(z))
            gt, la = G.robust_median(z[perm[len(z) // 2:]]), G.robust_median(z[perm[: len(z) // 2]])
            if gt is None or la is None or not (5 <= gt <= RMAX):
                continue
            rsel = radar_in_box(fr, fr.boxes[k], 1.0) & (fr.rz < 150)
            mv = moge_metric[fr.masks[k]]
            est = radar_range_estimates(fr.rz[rsel], prior=G.robust_median(mv[mv > 0.1]))
            snap = None
            if rsel.sum() >= 3 and ltree is not None:
                d, jj = ltree.query(fr.radar_l[rsel] / G.RXYZ)
                ok = d <= 1.0
                if ok.sum() >= 3:
                    snap = G.robust_median(lz_all[jj[ok]])
            r = dict(snippet=snip, frame=f, k=k, gt=gt, lidarA=la, moge=mm, n_lidar=len(z),
                     n_radar=int(rsel.sum()), conf=float(fr.conf[k]), is_ego=ego.get(f, -1),
                     extr_rot_err=ferr.get(f, np.nan), uc_s=_num(uc),
                     lcs_s=_num(lcs and lcs[0]), lcs_t=_num(lcs and lcs[1]),
                     rcp_s=_num(rcp and rcp[0]), rcp_t=_num(rcp and rcp[1]), radar_snap=_num(snap))
            r.update({f"radar_{m}": _num(est[m]) for m in RADAR_EST})
            recs.append(r)
    return recs


def extract(a):
    if a.frames_file:
        sel = defaultdict(list)
        for line in open(a.frames_file):
            sn, fi = line.split()
            sel[sn].append(int(fi))
        jobs = [(s, sorted(v), a.extrinsic) for s, v in sel.items()]
    else:
        jobs = [(s, io.list_frames(s)[:: a.stride], a.extrinsic) for s in io.list_snippets()]
    with Pool(a.workers) as pool:
        recs = [r for rs in pool.map(extract_snippet, jobs) for r in rs]
    with open(OUT / f"e3_records_{a.tag}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    print(f"{len(recs)} animals from {len({(r['snippet'], r['frame']) for r in recs})} frames")


# ------------------------------------------------------------------ stage 2
def frame_fit(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
    if not pts:
        return None
    x, y = np.array(pts).T
    return G.fit_affine(x, y) if len(pts) >= 2 else G.fit_scale(x, y)


def predict(recs, radar_key):
    """Per-animal predictions for all methods, fitting per frame."""
    by = defaultdict(list)
    for i, r in enumerate(recs):
        by[(r["snippet"], r["frame"])].append(i)
    P = defaultdict(lambda: np.full(len(recs), np.nan))
    for idx in by.values():
        rs = [recs[i] for i in idx]
        mo = np.array([r["moge"] for r in rs])
        lcb = frame_fit(mo, [r["lidarA"] for r in rs])
        rad = np.array([r[radar_key] for r in rs])
        rcb = frame_fit(mo, rad)
        ok = np.isfinite(rad)
        rcs = G.fit_scale(mo[ok], rad[ok]) if ok.any() else None
        snp = frame_fit(mo, [r["radar_snap"] for r in rs])
        for j, i in enumerate(idx):
            r = rs[j]
            m = r["moge"]
            P["MoGe-2 metric (no anchor)"][i] = m
            P["UC (1 LiDAR pt)"][i] = r["uc_s"] * m
            P["LC-scene (LiDAR)"][i] = r["lcs_s"] * m + r["lcs_t"]
            if lcb:
                P["LC-box held-out (LiDAR)"][i] = lcb[0] * m + lcb[1]
            if rcb:
                P["RC-box (radar)"][i] = rcb[0] * m + rcb[1]
            if rcs:
                P["RC-box scale-only (radar)"][i] = rcs[0] * m
            loo = frame_fit(np.delete(mo, j), np.delete(rad, j))
            if loo:
                P["RC-box LOO (radar)"][i] = loo[0] * m + loo[1]
            P["RC-pixel (radar)"][i] = r["rcp_s"] * m + r["rcp_t"]
            P["Radar range only"][i] = r[radar_key]
            if snp:
                P["Radar snapped to LiDAR*"][i] = snp[0] * m + snp[1]
            P["LiDAR half-A only (bound)"][i] = r["lidarA"]
    return P


def score(recs, P, mask):
    gt = np.array([r["gt"] for r in recs])
    grp = np.array([f"{r['snippet']}/{r['frame']}" for r in recs])
    out = {}
    for name, p in P.items():
        sel = mask & np.isfinite(p)
        e = M.errors(p[sel], gt[sel])
        e["coverage"] = float(sel.sum() / max(mask.sum(), 1))
        if e["n"] > 5:
            e["AbsRel_CI95"] = M.bootstrap_ci(p[sel], gt[sel], grp[sel], "AbsRel", n=300)
            e["bins"] = {f"{lo}-{hi}": M.errors(p[sel & (gt >= lo) & (gt < hi)], gt[sel & (gt >= lo) & (gt < hi)])
                         for lo, hi in M.BINS}
        out[name] = e
    return out


def summarize(a):
    recs = list(csv.DictReader(open(OUT / f"e3_records_{a.tag}.csv")))
    for r in recs:
        for k, v in r.items():
            if k not in ("snippet",):
                r[k] = float(v)
    sid = np.array([int(r["snippet"].split("_")[1]) for r in recs])
    dev, test = sid % 2 == 1, sid % 2 == 0
    # --- choose the LiDAR-free radar estimator on DEV (lowest AbsRel of radar range itself)
    gt = np.array([r["gt"] for r in recs])
    sel_tab = {}
    for m in RADAR_EST:
        v = np.array([r[f"radar_{m}"] for r in recs])
        ok = dev & np.isfinite(v)
        sel_tab[m] = dict(dev_coverage=float(ok.sum() / dev.sum()), dev_AbsRel=M.errors(v[ok], gt[ok]).get("AbsRel"),
                          dev_medRel=float(np.median(np.abs(v[ok] - gt[ok]) / gt[ok])) if ok.any() else None)
    best = min(RADAR_EST, key=lambda m: sel_tab[m]["dev_medRel"] if sel_tab[m]["dev_medRel"] is not None else 1e9)
    print("radar estimator selection on DEV snippets:", json.dumps(sel_tab, indent=1), "-> chosen:", best)
    P = predict(recs, f"radar_{best}")
    has_radar = np.isfinite(P["RC-box (radar)"])
    static = np.array([r["is_ego"] == 0 for r in recs])
    goodcal = np.array([r["extr_rot_err"] < 2.0 for r in recs])
    subsets = {
        "test_all": test,
        "test_frames_with_radar_anchor": test & has_radar,
        "test_frames_with_radar_anchor_static_goodcal": test & has_radar & static & goodcal,
        "test_animals_with_own_radar_estimate": test & np.isfinite(P["Radar range only"]),
        "all_snippets_frames_with_radar_anchor": has_radar,
    }
    summary = dict(radar_estimator_selection=sel_tab, radar_estimator=best, n_records=len(recs),
                   n_test=int(test.sum()), n_dev=int(dev.sum()),
                   frac_animals_with_radar_estimate=float(np.mean(np.isfinite([r[f"radar_{best}"] for r in recs]))))
    for name, m in subsets.items():
        summary[name] = score(recs, P, m)
        summary[name]["_n_animals"] = int(m.sum())
        print(f"\n== {name}: {int(m.sum())} animals")
        for meth, e in summary[name].items():
            if isinstance(e, dict) and e.get("n", 0) > 5:
                print(f"  {meth:<28} n={e['n']:>5} cov={e['coverage']:.2f}  MAE={e['MAE']:6.2f} medAE={e['medAE']:5.2f} "
                      f"AbsRel={e['AbsRel']:.3f} [{e['AbsRel_CI95'][0]:.3f},{e['AbsRel_CI95'][1]:.3f}] d1.25={e['d125']:.3f}")
    json.dump(summary, open(OUT / f"e3_summary_{a.tag}.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "summarize"])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--extrinsic", default="anchor", choices=["per_frame", "anchor"])
    ap.add_argument("--tag", default="anchor")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--frames-file", help="optional 'snippet frame' list (subset runs)")
    a = ap.parse_args()
    extract(a) if a.stage == "extract" else summarize(a)


if __name__ == "__main__":
    main()
