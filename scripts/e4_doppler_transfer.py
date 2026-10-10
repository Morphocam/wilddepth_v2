"""E4: Doppler as a motion cue, transferred from sparse radar to dense single-sweep LiDAR.

For every synchronized frame that is NOT an ego-motion frame (per-frame cosine fit, E10):
  * radar returns with |h| > tau are 'moving';
  * each LiDAR point inherits |h| of the nearest radar return inside an ellipsoid (radii);
    inherited |h| > tau -> moving LiDAR point;
  * precision = fraction of in-image moving points that fall inside an animal box
    (radar: box padded vertically by one box height for the elevation error; LiDAR: exact box);
  * densification = moving LiDAR points / moving radar points (in image).
Per animal:
  * radar_moving: >= 3 moving LiDAR points (after transfer) inside the SAM2 mask;
  * flow_moving (independent reference): Farneback flow between frame f-1 and f, animal
    median flow minus background median flow > FLOW_PX px (camera motion removed). Flow sees
    lateral motion; radar sees radial motion -- the two are complementary, not identical.
Ablation: radii x tau.

Outputs: results/e4_frames.csv, results/e4_animals.csv, results/e4_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wd2 import geometry as G, io  # noqa: E402
from wd2.frame import lidar_in_mask, load_frame  # noqa: E402

OUT = io.REPO / "results"
RADII = {"aniso_0.6_1.0_1.5": (0.6, 1.0, 1.5), "iso_0.6": (0.6, 0.6, 0.6),
         "iso_1.0": (1.0, 1.0, 1.0), "aniso_0.6_1.0_2.5": (0.6, 1.0, 2.5)}
TAUS = (0.3, 0.5, 0.7, 1.0)
DEFAULT = ("aniso_0.6_1.0_1.5", 0.5)
MIN_MOVING_PTS = 3
FLOW_PX = 1.0


def box_member(u, v, boxes, vpad=0.0):
    inside = np.zeros(len(u), bool)
    for x0, y0, x1, y1 in boxes:
        p = vpad * (y1 - y0)
        inside |= (u >= x0) & (u <= x1) & (v >= y0 - p) & (v <= y1 + p)
    return inside


def animal_flow(snip, f, masks):
    prev = io.snippet_dir(snip) / "rgb" / f"{f - 1}.png"
    if not prev.exists() or len(masks) == 0:
        return [np.nan] * len(masks)
    a = cv2.cvtColor(cv2.imread(str(prev)), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.imread(str(io.snippet_dir(snip) / "rgb" / f"{f}.png")), cv2.COLOR_BGR2GRAY)
    fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    anyanimal = masks.any(0)
    bg = np.median(fl[~anyanimal], axis=0)
    return [float(np.linalg.norm(np.median(fl[m], axis=0) - bg)) if m.sum() else np.nan for m in masks]


def process(args):
    snip, frames, ego = args
    frows, arows = [], []
    for f in frames:
        if ego.get(f, 1) != 0:           # only verified non-ego frames
            continue
        fr = load_frame(snip, f)
        if fr is None or len(fr.boxes) == 0:
            continue
        h = np.abs(fr.radar["h"])
        flows = animal_flow(snip, f, fr.masks)
        rin = box_member(fr.ru, fr.rv, fr.boxes, vpad=1.0)
        lz_ok = fr.lok
        per = {}
        for rname, radii in RADII.items():
            inh = G.doppler_transfer(fr.lidar_l, fr.radar_l, fr.radar["h"], np.array(radii))
            for tau in TAUS:
                rm = fr.rok & (h > tau)
                lm = lz_ok & (np.nan_to_num(inh) > tau)
                lin = box_member(fr.lu[lm], fr.lv[lm], fr.boxes)
                per[(rname, tau)] = (lm, lin)
                frows.append(dict(snippet=snip, frame=f, radii=rname, tau=tau, n_boxes=len(fr.boxes),
                                  radar_moving=int(rm.sum()), radar_moving_in_box=int((rm & rin).sum()),
                                  lidar_moving=int(lm.sum()), lidar_moving_in_box=int(lin.sum())))
        lm, _ = per[DEFAULT]
        for k in range(len(fr.boxes)):
            sel = lidar_in_mask(fr, k)
            gt = G.robust_median(fr.lz[sel])
            arows.append(dict(snippet=snip, frame=f, k=k, dist=gt if gt is not None else np.nan,
                              n_lidar=int(sel.sum()), n_moving_lidar=int((sel & lm).sum()),
                              flow_px=flows[k]))
    return frows, arows


def summarize():
    fr = list(csv.DictReader(open(OUT / "e4_frames.csv")))
    an = list(csv.DictReader(open(OUT / "e4_animals.csv")))
    out = {"n_nonego_frames_with_animals": len({(r["snippet"], r["frame"]) for r in fr})}
    abl = {}
    for rname in RADII:
        for tau in TAUS:
            rows = [r for r in fr if r["radii"] == rname and float(r["tau"]) == tau]
            ev = [r for r in rows if int(r["radar_moving_in_box"]) >= MIN_MOVING_PTS]
            rm, rmb = sum(int(r["radar_moving"]) for r in ev), sum(int(r["radar_moving_in_box"]) for r in ev)
            lm, lmb = sum(int(r["lidar_moving"]) for r in ev), sum(int(r["lidar_moving_in_box"]) for r in ev)
            abl[f"{rname}|{tau}"] = dict(event_frames=len(ev), radar_moving=rm, lidar_moving=lm,
                                         densification=lm / max(rm, 1),
                                         radar_precision=rmb / max(rm, 1), lidar_precision=lmb / max(lm, 1),
                                         lidar_moving_in_box_per_frame=lmb / max(len(ev), 1))
    out["ablation"] = abl
    # per-animal agreement with the flow reference (default setting)
    flow = np.array([float(r["flow_px"]) for r in an])
    nm = np.array([int(r["n_moving_lidar"]) for r in an])
    ok = np.isfinite(flow)
    radar_mov, flow_mov = nm >= MIN_MOVING_PTS, flow > FLOW_PX
    tp = int((radar_mov & flow_mov & ok).sum())
    out["per_animal"] = dict(
        n_animals=int(ok.sum()), radar_moving=int((radar_mov & ok).sum()), flow_moving=int((flow_mov & ok).sum()),
        both=tp, precision_vs_flow=tp / max(int((radar_mov & ok).sum()), 1),
        recall_vs_flow=tp / max(int((flow_mov & ok).sum()), 1),
        frac_animals_with_doppler=float(np.mean(radar_mov)))
    per_event = {}
    for r in fr:
        if (r["radii"], float(r["tau"])) == (DEFAULT[0], DEFAULT[1]) and int(r["radar_moving_in_box"]) >= MIN_MOVING_PTS:
            per_event.setdefault(r["snippet"], []).append(r)
    out["events_by_snippet"] = {
        s: dict(frames=len(v), radar_moving=sum(int(r["radar_moving"]) for r in v),
                lidar_moving=sum(int(r["lidar_moving"]) for r in v),
                lidar_precision=sum(int(r["lidar_moving_in_box"]) for r in v) / max(sum(int(r["lidar_moving"]) for r in v), 1))
        for s, v in sorted(per_event.items(), key=lambda kv: -len(kv[1]))}
    top = sorted((r for rs in per_event.values() for r in rs), key=lambda r: -int(r["lidar_moving_in_box"]))[:60]
    out["top_frames"] = [dict(snippet=r["snippet"], frame=int(r["frame"]), radar_moving=int(r["radar_moving"]),
                              lidar_moving=int(r["lidar_moving"]), lidar_in_box=int(r["lidar_moving_in_box"]),
                              precision=int(r["lidar_moving_in_box"]) / max(int(r["lidar_moving"]), 1)) for r in top]
    json.dump(out, open(OUT / "e4_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "ablation"}, indent=1))
    print("default:", json.dumps(abl[f"{DEFAULT[0]}|{DEFAULT[1]}"], indent=1))
    for k, v in abl.items():
        print(f"{k:<24} ev={v['event_frames']:>4} dens={v['densification']:.2f} "
              f"P_radar={v['radar_precision']:.2f} P_lidar={v['lidar_precision']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "summarize"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.stage == "summarize":
        return summarize()
    ego_all = {}
    for r in csv.DictReader(open(OUT / "e10_ego_per_frame.csv")):
        ego_all.setdefault(r["snippet"], {})[int(r["frame"])] = int(r["is_ego"])
    jobs = [(s, io.list_frames(s), ego_all.get(s, {})) for s in io.list_snippets()]
    with Pool(a.workers) as pool:
        res = pool.map(process, jobs)
    for name, idx in (("e4_frames.csv", 0), ("e4_animals.csv", 1)):
        rows = [r for x in res for r in x[idx]]
        with open(OUT / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    summarize()


if __name__ == "__main__":
    main()
