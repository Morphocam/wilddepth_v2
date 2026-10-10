"""E10 + E11 (CPU): radar-subset statistics and the calibration / timing error budget.

Per snippet: synchronized frames, frozen-radar check, radar returns per frame and range,
LiDAR points per frame, radar-LiDAR timestamp offset, Doppler availability, ego-motion
frames, and the per-frame LiDAR->camera extrinsic error estimate shipped in frame_errors.csv.

Radar->LiDAR residual (E11): static radar returns (|h| < 0.5 m/s) are matched to the nearest
accumulated-LiDAR point in the ground plane (BEV); the in-plane distance and the vertical gap
to that neighbour are reported separately (gate: BEV neighbour within 1 m).

Outputs: results/e10_snippet_stats.csv, results/e11_calib_residuals.npz, results/e10_e11_summary.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wd2 import geometry as G, io  # noqa: E402

OUT = io.REPO / "results"
RESID_SAMPLE = 40          # frames per snippet for the residual
ACCUM = 5                  # LiDAR sweeps accumulated for the residual reference
DOPPLER_MOVING = 0.5       # m/s


def frame_errors(snip):
    p = io.snippet_dir(snip) / "frame_errors.csv"
    if not p.exists():
        return np.array([]), np.array([])
    rows = list(csv.DictReader(p.open()))
    return (np.array([float(r["rot_err_deg"]) for r in rows]),
            np.array([float(r["trans_err_m"]) for r in rows]))


def main():
    OUT.mkdir(exist_ok=True)
    R2L = io.radar2lidar()
    rows, bev_all, dz_all = [], [], []
    ego_rows = []
    for snip in io.list_snippets():
        frames = io.list_frames(snip)
        n_radar_files = len(list((io.snippet_dir(snip) / "radar").glob("*.pcd")))
        frozen = io.radar_frozen(snip)
        n_ret, rng, n_lid, dop_frac, ego = [], [], [], [], []
        for f in frames:
            r = io.load_radar(io.radar_path(snip, f))
            n_ret.append(len(r["x"]))
            if len(r["x"]):
                rng.append(np.hypot(r["x"], r["y"]))
            dop_frac.append(np.mean(np.abs(r["h"]) > DOPPLER_MOVING) if len(r["h"]) else 0.0)
            V, r2, inl, is_ego = G.ego_motion_fit(r)
            ego.append(is_ego)
            ego_rows.append((snip, f, V, r2, inl, int(is_ego)))
        for f in frames[:: max(1, len(frames) // 50)]:
            n_lid.append(len(io.load_lidar_xyz(io.lidar_path(snip, f))))
        dts = np.abs([v for k, v in io.match_log(snip).items() if k in set(frames)])
        rot, trans = frame_errors(snip)
        rng = np.concatenate(rng) if rng else np.zeros(1)

        # ---- E11 residual
        samp = frames[:: max(1, len(frames) // RESID_SAMPLE)][:RESID_SAMPLE]
        for f in samp:
            i = frames.index(f)
            win = frames[max(0, i - ACCUM + 1): i + 1]
            P = np.vstack([io.load_lidar_xyz(io.lidar_path(snip, g)) for g in win])
            r = io.load_radar(io.radar_path(snip, f))
            st = np.abs(r["h"]) < DOPPLER_MOVING
            if st.sum() == 0 or len(P) < 100:
                continue
            rp = G.transform(R2L, np.stack([r["x"][st], r["y"][st], r["z"][st]], 1))
            d2, j = cKDTree(P[:, :2]).query(rp[:, :2])
            g = d2 <= 1.0
            bev_all.append(d2[g])
            dz_all.append(np.abs(rp[g, 2] - P[j[g], 2]))

        rows.append(dict(
            snippet=snip, radar_files=n_radar_files, sync_frames=len(frames), radar_frozen=int(frozen),
            radar_returns_med=float(np.median(n_ret)), radar_range_p1=float(np.percentile(rng, 1)),
            radar_range_p99=float(np.percentile(rng, 99)), lidar_pts_med=float(np.median(n_lid)),
            dt_mean_ms=float(dts.mean() * 1e3) if len(dts) else np.nan,
            dt_p95_ms=float(np.percentile(dts, 95) * 1e3) if len(dts) else np.nan,
            dt_max_ms=float(dts.max() * 1e3) if len(dts) else np.nan,
            frac_frames_doppler=float(np.mean(np.array(dop_frac) > 0.02)),
            frac_ego_frames=float(np.mean(ego)),
            extr_rot_err_deg_med=float(np.median(rot)) if len(rot) else np.nan,
            extr_trans_err_m_med=float(np.median(trans)) if len(trans) else np.nan))
        print(rows[-1], flush=True)

    with open(OUT / "e10_snippet_stats.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "e10_ego_per_frame.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["snippet", "frame", "ego_speed", "r2", "inlier_frac", "is_ego"])
        w.writerows(ego_rows)
    bev, dz = np.concatenate(bev_all), np.concatenate(dz_all)
    np.savez_compressed(OUT / "e11_calib_residuals.npz", bev=bev.astype(np.float32), dz=dz.astype(np.float32))

    def pct(a):
        return {k: float(np.percentile(a, q)) for k, q in (("p25", 25), ("med", 50), ("p75", 75), ("p90", 90))}
    all_dt = [abs(v) for s in io.list_snippets() for v in io.match_log(s).values()]
    summary = dict(
        snippets=len(rows), sync_frames=int(sum(r["sync_frames"] for r in rows)),
        radar_files=int(sum(r["radar_files"] for r in rows)),
        frozen_snippets=[r["snippet"] for r in rows if r["radar_frozen"]],
        radar_returns_per_frame_med=float(np.median([r["radar_returns_med"] for r in rows])),
        lidar_pts_per_frame_med=float(np.median([r["lidar_pts_med"] for r in rows])),
        radar_range_p1_p99=[float(min(r["radar_range_p1"] for r in rows)), float(max(r["radar_range_p99"] for r in rows))],
        dt_ms=dict(mean=float(np.mean(all_dt) * 1e3), p95=float(np.percentile(all_dt, 95) * 1e3), max=float(np.max(all_dt) * 1e3)),
        frac_frames_with_doppler=float(np.average([r["frac_frames_doppler"] for r in rows], weights=[r["sync_frames"] for r in rows])),
        frac_ego_frames=float(np.average([r["frac_ego_frames"] for r in rows], weights=[r["sync_frames"] for r in rows])),
        extr_rot_err_deg_med=float(np.nanmedian([r["extr_rot_err_deg_med"] for r in rows])),
        extr_trans_err_m_med=float(np.nanmedian([r["extr_trans_err_m_med"] for r in rows])),
        residual_bev_m=pct(bev), residual_dz_m=pct(dz), residual_n=int(len(bev)))
    json.dump(summary, open(OUT / "e10_e11_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
