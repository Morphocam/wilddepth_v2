"""E5: Doppler-gated temporal LiDAR accumulation for dense animal depth.

For each animal in a NON-ego frame f (static platform, so past sweeps are already registered):
  sweeps  = LiDAR sweeps f-K+1 .. f-1 (history) and f (current)
  held-out GT = half B of the current sweep's points in the mask (random split)
  anchor set:
    single     current-sweep half A only
    accum      history sweeps + half A              (standard accumulation)
    gated      as accum, but if Doppler says the animal moves (>= 3 transferred moving
               LiDAR points in the mask), history points inside the dilated box are dropped
    oracle     as gated, but using the flow-based motion label instead of Doppler
  dense depth = MoGe-2 aligned to the in-mask anchors by trimmed affine (SSA);
  error = per-pixel MAE of dense depth vs GT points (half B) inside the mask.
Also records the accumulated/single disagreement used by the GT-free selection rule.

Output: results/e5_records.csv, results/e5_summary.json
"""
from __future__ import annotations

import csv
import json
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wd2 import geometry as G, io  # noqa: E402
from wd2.frame import lidar_in_mask, load_frame  # noqa: E402

OUT = io.REPO / "results"
K_ACC = int(__import__("os").environ.get("WD2_K_ACC", 5))
TAU = 0.5
DILATE = 15


def dense_mae(moge, mask, au, av, az, gu, gv, gz):
    """SSA-align MoGe to anchors (au,av,az) inside mask; MAE at GT pixels."""
    if len(az) < 3:
        return np.nan
    x = moge[av, au]
    ok = x > 0.1
    st = G.fit_affine(x[ok], az[ok]) if ok.sum() >= 3 else None
    if st is None:
        return np.nan
    pred = st[0] * moge[gv, gu] + st[1]
    return float(np.mean(np.abs(pred - gz)))


def process(args):
    snip, ego, flowmov = args
    frames = io.list_frames(snip)
    rng = np.random.default_rng(zlib.crc32(snip.encode()))
    K = io.intrinsic()
    recs = []
    for i, f in enumerate(frames):
        if ego.get(f, 1) != 0 or i < K_ACC - 1 or frames[i - K_ACC + 1] != f - K_ACC + 1:
            continue       # need K consecutive static-platform sweeps
        if any(ego.get(g, 1) != 0 for g in frames[i - K_ACC + 1: i]):
            continue
        fr = load_frame(snip, f)
        if fr is None or len(fr.boxes) == 0:
            continue
        hist = np.vstack([io.load_lidar_xyz(io.lidar_path(snip, g)) for g in frames[i - K_ACC + 1: i]])
        hu, hv, hz, hok = G.project(G.transform(fr.L2C, hist), K)
        inh = G.doppler_transfer(fr.lidar_l, fr.radar_l, fr.radar["h"])
        moving_l = np.nan_to_num(inh) > TAU
        for k in range(len(fr.boxes)):
            m = fr.masks[k]
            sel = lidar_in_mask(fr, k)
            idx = np.flatnonzero(sel & (fr.lz < 150))
            if len(idx) < 10:
                continue
            perm = rng.permutation(idx)
            A, B = perm[: len(idx) // 2], perm[len(idx) // 2:]
            gu, gv, gz = fr.lu[B].astype(int), fr.lv[B].astype(int), fr.lz[B]
            au, av, az = fr.lu[A].astype(int), fr.lv[A].astype(int), fr.lz[A]
            hsel = G.mask_values(hu, hv, hok, m) & (hz < 150)
            md = cv2.dilate(m.astype(np.uint8), np.ones((DILATE, DILATE), np.uint8)).astype(bool)
            hs_in_dil = G.mask_values(hu, hv, hok, md)
            doppler_mov = int((sel & moving_l).sum()) >= 3
            fmov = bool(flowmov.get((f, k), False))
            cu = np.concatenate([au, hu[hsel].astype(int)])
            cv_ = np.concatenate([av, hv[hsel].astype(int)])
            cz = np.concatenate([az, hz[hsel]])
            e_single = dense_mae(fr.moge, m, au, av, az, gu, gv, gz)
            e_acc = dense_mae(fr.moge, m, cu, cv_, cz, gu, gv, gz)
            e_gate = e_single if doppler_mov else e_acc
            e_oracle = e_single if fmov else e_acc
            disagree = abs((G.robust_median(cz) or np.nan) - (G.robust_median(az) or np.nan))
            recs.append(dict(snippet=snip, frame=f, k=k, dist=float(np.median(gz)), n_hist=int(hsel.sum()),
                             n_hist_dil=int(hs_in_dil.sum()), doppler_moving=int(doppler_mov),
                             flow_moving=int(fmov), disagree=disagree, mae_single=e_single,
                             mae_accum=e_acc, mae_gated=e_gate, mae_flowgated=e_oracle))
    return recs


def summarize(recs):
    a = {k: np.array([float(r[k]) for r in recs]) for k in recs[0] if k != "snippet"}
    ok = np.isfinite(a["mae_single"]) & np.isfinite(a["mae_accum"])
    out = {"n_animals": int(ok.sum())}
    groups = {"all": ok, "doppler_moving": ok & (a["doppler_moving"] == 1),
              "flow_moving": ok & (a["flow_moving"] == 1),
              "doppler_and_disagree>0.2": ok & (a["doppler_moving"] == 1) & (a["disagree"] > 0.2),
              "static(neither)": ok & (a["doppler_moving"] == 0) & (a["flow_moving"] == 0)}
    for g, s in groups.items():
        if s.sum():
            out[g] = {"n": int(s.sum())}
            for m in ("mae_single", "mae_accum", "mae_gated", "mae_flowgated"):
                out[g][m] = float(np.mean(a[m][s]))
    # gate quality: an animal 'benefits' if single-sweep beats accumulation
    benefit = ok & (a["mae_single"] < a["mae_accum"])
    for name, gate in (("doppler", a["doppler_moving"] == 1), ("flow", a["flow_moving"] == 1),
                       ("doppler_and_disagree", (a["doppler_moving"] == 1) & (a["disagree"] > 0.2)),
                       ("doppler_or_flow", (a["doppler_moving"] == 1) | (a["flow_moving"] == 1))):
        g = ok & gate
        out[f"gate_{name}"] = dict(fires=int(g.sum()), fire_rate=float(g.sum() / ok.sum()),
                                   precision=float((g & benefit).sum() / max(g.sum(), 1)),
                                   recall=float((g & benefit).sum() / max(benefit.sum(), 1)))
    out["frac_benefit"] = float(benefit.sum() / ok.sum())
    return out


def main():
    ego_all, flow = {}, {}
    for r in csv.DictReader(open(OUT / "e10_ego_per_frame.csv")):
        ego_all.setdefault(r["snippet"], {})[int(r["frame"])] = int(r["is_ego"])
    p = OUT / "e4_animals.csv"
    if p.exists():
        for r in csv.DictReader(open(p)):
            if r["flow_px"] not in ("", "nan"):
                flow.setdefault(r["snippet"], {})[(int(r["frame"]), int(r["k"]))] = float(r["flow_px"]) > 1.0
    jobs = [(s, ego_all.get(s, {}), flow.get(s, {})) for s in io.list_snippets()]
    with Pool(8) as pool:
        recs = [r for rs in pool.map(process, jobs) for r in rs]
    with open(OUT / f"e5_records_k{K_ACC}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    out = summarize(recs)
    json.dump(out, open(OUT / f"e5_summary_k{K_ACC}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
