"""E6 metric 3D tracking (+ E2 temporal stability, E9 distance sampling, E8 Doppler velocity).

Stage 1 (extract): per snippet, run an IoU tracker over consecutive frames (MegaDetector >= 0.3)
and record per observation: mask centroid, LiDAR GT distance (all mask points), MoGe-2 mask depth,
per-frame UC / LC-scene fits, the LiDAR-free radar range (MoGe-gated estimator chosen in E3),
and -- for the E8 analysis only -- the signed Doppler of radar returns within 2 m of the LiDAR
range of the animal (this association uses LiDAR and is not a method).

Stage 2 (summarize): tracks with >= MIN_LEN observations are lifted to camera-frame 3D with each
depth source and smoothed with the same constant-velocity Kalman/RTS smoother; the LiDAR-lifted
track is the reference.  Sources:
  MoGe-2 metric      raw metric output
  UC per frame       one LiDAR point near the image centre, every frame
  LC per frame       scene LiDAR affine, every frame
  LC once per track  LiDAR scale from the FIRST frame of the track only, then MoGe-2
  Radar range        LiDAR-free radar range (gaps where no estimate)
  RC per track       MoGe-2 x median(radar/MoGe) over the track -- RGB + radar, no LiDAR
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wd2 import geometry as G, io  # noqa: E402
from wd2.frame import lidar_in_mask, load_frame, moge_in_mask, radar_in_box, radar_range_estimates  # noqa: E402
from wd2.track import IoUTracker, kalman_cv_smooth  # noqa: E402

OUT = io.REPO / "results"
MIN_LEN = 20
RMAX = 55.0
SOURCES = ["MoGe-2 metric", "UC per frame", "LC per frame", "LC once per track", "Radar range", "RC per track"]


def extract_snippet(snip):
    ego = {int(r["frame"]): int(r["is_ego"]) for r in csv.DictReader(open(OUT / "e10_ego_per_frame.csv"))
           if r["snippet"] == snip}
    tr = IoUTracker()
    rows = []
    K = io.intrinsic()
    for f in io.list_frames(snip):
        fr = load_frame(snip, f)
        if fr is None:
            continue
        ids = tr.update(f, fr.boxes)
        if len(fr.boxes) == 0:
            continue
        mo = fr.moge
        ui, vi = np.clip(fr.lu.astype(int), 0, io.W - 1), np.clip(fr.lv.astype(int), 0, io.H - 1)
        sc = fr.lok & (fr.lz < RMAX)
        dm = mo[vi[sc], ui[sc]]
        g = dm > 0.1
        lcs = G.fit_affine(dm[g], fr.lz[sc][g]) if g.sum() >= 3 else None
        cand = np.flatnonzero(fr.lok & (fr.lz < 150))
        cand = cand[mo[vi[cand], ui[cand]] > 0.1] if len(cand) else cand
        uc = np.nan
        if len(cand):
            j = cand[np.argmin(np.hypot(fr.lu[cand] - io.W / 2, fr.lv[cand] - io.H / 2))]
            uc = fr.lz[j] / mo[vi[j], ui[j]]
        for k in range(len(fr.boxes)):
            ys, xs = np.nonzero(fr.masks[k])
            sel = lidar_in_mask(fr, k)
            gt = G.robust_median(fr.lz[sel & (fr.lz < 150)], min_n=5)
            mm = moge_in_mask(fr, k)
            rsel = radar_in_box(fr, fr.boxes[k], 1.0) & (fr.rz < 150)
            rest = radar_range_estimates(fr.rz[rsel], prior=mm)["prior"]
            dop_med = dop_std = np.nan
            n_dop = 0
            if gt is not None:
                near = rsel & (np.abs(fr.rz - gt) < 2.0)
                n_dop = int(near.sum())
                if n_dop >= 3:
                    hh = fr.radar["h"][near]
                    dop_med, dop_std = float(np.median(hh)), float(np.std(hh))
            rows.append(dict(snippet=snip, frame=f, track=ids[k], u=float(xs.mean()), v=float(ys.mean()),
                             conf=float(fr.conf[k]), gt=np.nan if gt is None else gt, n_lidar=int(sel.sum()),
                             moge=np.nan if mm is None else mm, uc_s=uc,
                             lcs_s=np.nan if lcs is None else lcs[0], lcs_t=np.nan if lcs is None else lcs[1],
                             radar=np.nan if rest is None else rest, dop_med=dop_med, dop_std=dop_std,
                             n_dop=n_dop, is_ego=ego.get(f, -1)))
    return rows


def backproject(u, v, d, K):
    return np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], 1)


def summarize():
    rows = list(csv.DictReader(open(OUT / "e6_observations.csv")))
    for r in rows:
        for k in r:
            if k != "snippet":
                r[k] = float(r[k])
    K = io.intrinsic()
    tracks = defaultdict(list)
    for r in rows:
        tracks[(r["snippet"], int(r["track"]))].append(r)
    res = {s: dict(pos=[], dist_rel=[], speed_err=[], within10=[], jitter=[]) for s in SOURCES + ["LiDAR"]}
    per_track_dist = defaultdict(list)
    e8 = []
    n_tracks = 0
    for key, obs in tracks.items():
        obs.sort(key=lambda r: r["frame"])
        gt = np.array([o["gt"] for o in obs])
        if len(obs) < MIN_LEN or np.isfinite(gt).mean() < 0.5 or np.nanmedian(gt) > RMAX:
            continue
        n_tracks += 1
        t = np.array([o["frame"] for o in obs]) * 0.1
        u, v = np.array([o["u"] for o in obs]), np.array([o["v"] for o in obs])
        mo = np.array([o["moge"] for o in obs])
        first = np.flatnonzero(np.isfinite(gt) & np.isfinite(mo))[0]
        radar = np.array([o["radar"] for o in obs])
        ratio = radar / mo
        rc_s = np.nanmedian(ratio) if np.isfinite(ratio).any() else np.nan
        dsrc = {
            "LiDAR": gt,
            "MoGe-2 metric": mo,
            "UC per frame": np.array([o["uc_s"] for o in obs]) * mo,
            "LC per frame": np.array([o["lcs_s"] for o in obs]) * mo + np.array([o["lcs_t"] for o in obs]),
            "LC once per track": gt[first] / mo[first] * mo,
            "Radar range": radar,
            "RC per track": rc_s * mo,
        }
        r_meas = lambda d: 0.1 + 0.05 * np.nan_to_num(d, nan=20.0)  # noqa: E731
        ref_p, ref_v = kalman_cv_smooth(t, backproject(u, v, gt, K), r_meas(gt))
        ref_speed = np.linalg.norm(ref_v, axis=1)
        ref_range = np.linalg.norm(ref_p, axis=1)
        for s, d in dsrc.items():
            if np.isfinite(d).sum() < 5:
                continue
            p, vel = kalman_cv_smooth(t, backproject(u, v, d, K), r_meas(d))
            ok = np.isfinite(gt)
            res[s]["pos"].append(np.sqrt(np.nanmean(np.sum((p[ok] - ref_p[ok]) ** 2, 1))))
            res[s]["dist_rel"].append(np.nanmean(np.abs(p[ok, 2] - gt[ok]) / gt[ok]))
            res[s]["within10"].append(np.nanmean(np.abs(p[ok, 2] - gt[ok]) / gt[ok] < 0.1))
            mv = ref_speed > 0.3
            if mv.sum() >= 5:
                res[s]["speed_err"].append(np.mean(np.abs(np.linalg.norm(vel[mv], axis=1) - ref_speed[mv])))
            dd = np.diff(d)
            consec = np.diff(t) < 0.15
            if (np.isfinite(dd) & consec).sum() >= 5:
                res[s]["jitter"].append(np.nanstd(dd[consec]))
            per_track_dist[s].append(np.nanmedian(d))
        per_track_dist["LiDAR ref"].append(np.nanmedian(gt))
        rr = np.gradient(ref_range, t)
        for i, o in enumerate(obs):
            if np.isfinite(o["dop_med"]):
                e8.append((o["dop_med"], rr[i], o["dop_std"], ref_speed[i], o["is_ego"]))

    out = {"n_tracks": n_tracks, "n_observations": len(rows)}
    for s, m in res.items():
        out[s] = {k: (float(np.nanmedian(v)) if len(v) else None) for k, v in m.items()}
        out[s]["n_tracks"] = len(m["pos"])
        out[s]["n_tracks_speed"] = len(m["speed_err"])
    # E9: distribution of per-track distances vs LiDAR; half-normal sigma = sqrt(mean d^2)
    ref = np.array(per_track_dist["LiDAR ref"])
    sig_ref = float(np.sqrt(np.mean(ref ** 2)))
    out["E9"] = {"sigma_halfnormal_lidar": sig_ref}
    for s in SOURCES:
        d = np.array(per_track_dist[s], float)
        d = d[np.isfinite(d)]
        if len(d) >= 5:
            sig = float(np.sqrt(np.mean(d ** 2)))
            out["E9"][s] = dict(n=len(d), ks=float(stats.ks_2samp(d, ref).statistic),
                                ks_p=float(stats.ks_2samp(d, ref).pvalue), sigma=sig,
                                sigma_bias_pct=100 * (sig - sig_ref) / sig_ref)
    # E8: signed Doppler vs LiDAR range rate
    if e8:
        a = np.array(e8)
        ok = np.isfinite(a[:, 1])
        out["E8"] = dict(n=int(ok.sum()),
                         pearson_r=float(stats.pearsonr(a[ok, 0], a[ok, 1])[0]),
                         pearson_r_nonego=float(stats.pearsonr(a[ok & (a[:, 4] == 0), 0], a[ok & (a[:, 4] == 0), 1])[0])
                         if (ok & (a[:, 4] == 0)).sum() > 10 else None,
                         pearson_r_ego=float(stats.pearsonr(a[ok & (a[:, 4] == 1), 0], a[ok & (a[:, 4] == 1), 1])[0])
                         if (ok & (a[:, 4] == 1)).sum() > 10 else None,
                         spearman_doppler_std_vs_speed=float(stats.spearmanr(a[ok, 2], a[ok, 3])[0]))
        np.save(OUT / "e8_pairs.npy", a)
    json.dump(out, open(OUT / "e6_summary.json", "w"), indent=1)
    json.dump({k: list(map(float, v)) for k, v in per_track_dist.items()}, open(OUT / "e9_track_distances.json", "w"))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "summarize"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.stage == "summarize":
        return summarize()
    with Pool(a.workers) as pool:
        rows = [r for rs in pool.map(extract_snippet, io.list_snippets()) for r in rs]
    with open(OUT / "e6_observations.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summarize()


if __name__ == "__main__":
    main()
