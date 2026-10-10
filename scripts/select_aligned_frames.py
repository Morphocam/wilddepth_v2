"""Select well-aligned example frames for Figures 1 and 3 by explicit scores (auditable).

These are BEST-CASE illustrations; typical alignment is quantified in the error-budget figure and the
paper's captions say so.

LiDAR alignment (per animal, SAM2 mask):
    n_same  = LiDAR points in the mask within 1 m of the mask's median depth
    purity  = n_same / all points in the mask
    spill   = same-depth points in a 15-px ring around the mask (excluding other animals' masks)
              / (n_same + that count)    -- misprojection pushes animal points outside the mask
    lidar   = min(n_same, 150)/150 * purity * (1 - spill)
Radar alignment (radar subset only):
    on      = radar returns within 1.5 m of the LiDAR range that project inside the box
              (rows padded by only 25% of the box height)
    frac_on = on / radar returns in the 1-box-height padded box
    err     = |radar range estimate - LiDAR| / LiDAR  (MoGe-gated estimator of E3)
    radar   = min(on, 12)/12 * frac_on * (err < 0.05)
    moving  = on-animal returns with |Doppler| > 0.5 m/s

Output: results/aligned_frames.json (top candidates per category, with scores).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wd2 import geometry as G, io  # noqa: E402
from wd2.frame import load_frame, moge_in_mask, radar_range_estimates  # noqa: E402

RES = io.REPO / "results"
REL = Path(os.environ.get("WD2_RELEASE", "/mnt/hdd/WildDepth/wilddepth_release"))
RING = np.ones((31, 31), np.uint8)


def lidar_alignment(u, v, z, ok, masks, k):
    """Alignment score of projected LiDAR (u,v,z,ok) against mask k."""
    ui = np.clip(u.astype(int), 0, masks.shape[2] - 1)
    vi = np.clip(v.astype(int), 0, masks.shape[1] - 1)
    inm = ok & masks[k][vi, ui]
    if inm.sum() < 10:
        return None
    med = float(np.median(z[inm]))
    same = np.abs(z - med) < 1.0
    n_same = int((inm & same).sum())
    purity = n_same / inm.sum()
    ring = cv2.dilate(masks[k].astype(np.uint8), RING).astype(bool) & ~masks.any(0)
    n_ring = int((ok & ring[vi, ui] & same).sum())
    spill = n_ring / max(n_same + n_ring, 1)
    return dict(med=med, n_same=n_same, purity=float(purity), spill=float(spill),
                lidar=min(n_same, 150) / 150 * purity * (1 - spill))


def score_cps2(args):
    snip, f = args
    fr = load_frame(snip, f)
    if fr is None or len(fr.boxes) == 0:
        return None
    animals = []
    for k in range(len(fr.boxes)):
        la = lidar_alignment(fr.lu, fr.lv, fr.lz, fr.lok, fr.masks, k)
        if la is None or not (5 <= la["med"] <= 55):
            continue
        x0, y0, x1, y1 = fr.boxes[k]
        hgt = y1 - y0
        near = np.abs(fr.rz - la["med"]) < 1.5
        tight = fr.rok & (fr.ru >= x0) & (fr.ru <= x1) & (fr.rv >= y0 - 0.25 * hgt) & (fr.rv <= y1 + 0.25 * hgt)
        padded = fr.rok & (fr.ru >= x0) & (fr.ru <= x1) & (fr.rv >= y0 - hgt) & (fr.rv <= y1 + hgt)
        on = tight & near
        est = radar_range_estimates(fr.rz[padded & (fr.rz < 150)], prior=moge_in_mask(fr, k))["prior"]
        err = abs(est - la["med"]) / la["med"] if est is not None else 1.0
        frac_on = on.sum() / max(padded.sum(), 1)
        moving = int((on & (np.abs(fr.radar["h"]) > 0.5)).sum())
        animals.append(dict(k=k, **la, n_on=int(on.sum()), frac_on=float(frac_on), radar_est=est,
                            radar_err=float(err), moving=moving,
                            radar=min(int(on.sum()), 12) / 12 * frac_on * (err < 0.05)))
    if not animals:
        return None
    best = max(animals, key=lambda a: a["lidar"] * a["radar"])
    return dict(snippet=snip, frame=f, n_animals=len(animals), animals=animals,
                score=best["lidar"] * best["radar"], dist=best["med"],
                moving_score=max(a["lidar"] * a["radar"] * min(a["moving"], 8) / 8 for a in animals),
                herd_score=float(np.sum(sorted([a["lidar"] * a["radar"] for a in animals])[-3:])) if len(animals) >= 3 else 0.0)


def cps2_candidates(max_frames=2500):
    recs = list(csv.DictReader(open(RES / "e3_records_anchor.csv")))
    ego = {(r["snippet"], int(r["frame"])): int(r["is_ego"]) for r in csv.DictReader(open(RES / "e10_ego_per_frame.csv"))}
    keys = set()
    for r in recs:
        key = (r["snippet"], int(float(r["frame"])))
        if ego.get(key, 1) != 0 or r["radar_prior"] in ("", "nan") or float(r["n_lidar"]) < 20:
            continue
        if abs(float(r["radar_prior"]) - float(r["gt"])) / float(r["gt"]) < 0.05:
            keys.add(key)
    keys = sorted(keys)
    if len(keys) > max_frames:
        keys = [keys[i] for i in np.linspace(0, len(keys) - 1, max_frames).astype(int)]
    with Pool(16) as p:
        out = [x for x in p.map(score_cps2, keys) if x]
    return out, len(keys)


def release_candidates(dirs, scale, per_dir=10):
    from ultralytics import SAM, YOLO
    md = YOLO(os.environ.get("WD2_MD_WEIGHTS", "/mnt/hdd/WildDepth/depth_eval/weights/MDV6-yolov9-c.pt"))
    sam = SAM(os.environ.get("WD2_SAM_WEIGHTS", "/mnt/hdd/WildDepth/depth_eval/sam2.1_b.pt"))
    out = []
    for d in dirs:
        ps = sorted((d / "rgb").glob("*.jpg"))
        for p in [ps[i] for i in np.linspace(0, len(ps) - 1, min(per_dir, len(ps))).astype(int)]:
            dp = d / "depthm" / (p.stem + ".png")
            if not dp.exists():
                continue
            rgb = np.asarray(Image.open(p).convert("RGB"))
            if rgb.mean() / 255 > 0.62:
                continue
            H, W = rgb.shape[:2]
            b = md(rgb[..., ::-1].copy(), conf=0.5, verbose=False)[0].boxes
            keep = (b.cls.cpu().numpy() == 0)
            boxes = b.xyxy.cpu().numpy()[keep]
            conf = b.conf.cpu().numpy()[keep]
            if not len(boxes):
                continue
            m = sam(rgb[..., ::-1].copy(), bboxes=boxes.tolist(), verbose=False)[0].masks
            if m is None:
                continue
            masks = m.data.cpu().numpy().astype(bool)
            depth = np.asarray(Image.open(dp)).astype(float) / scale
            vv, uu = np.nonzero(depth > 0)
            z = depth[vv, uu]
            best = None
            for k in range(len(boxes)):
                area = masks[k].mean()
                if not (0.004 <= area <= 0.2):
                    continue
                la = lidar_alignment(uu.astype(float), vv.astype(float), z, np.ones(len(z), bool), masks, k)
                if la is None or not (5 <= la["med"] <= 60):
                    continue
                sc = la["lidar"] * float(conf[k]) * np.sqrt(min(area, 0.05) / 0.05)
                if best is None or sc > best["score"]:
                    best = dict(rgb=str(p), depth=str(dp), scale=scale, score=sc, box=[float(x) for x in boxes[k]], **la)
            if best:
                out.append(best)
    return sorted(out, key=lambda x: -x["score"])


def main():
    cps2, n_cand = cps2_candidates()
    res = {"n_cps2_candidates": n_cand, "n_cps2_scored": len(cps2)}
    res["cps2_best"] = sorted(cps2, key=lambda x: -x["score"])[:15]
    res["cps2_moving"] = sorted(cps2, key=lambda x: -x["moving_score"])[:15]
    res["cps2_herd"] = sorted(cps2, key=lambda x: -x["herd_score"])[:15]
    res["cps2_near"] = sorted((c for c in cps2 if 8 <= c["dist"] <= 20), key=lambda x: -x["score"])[:15]
    res["cps2_far"] = sorted((c for c in cps2 if c["dist"] > 25), key=lambda x: -x["score"])[:15]
    bub = [d for rec in ("19371340", "19372084") for d in sorted((REL / rec).iterdir()) if (d / "rgb").is_dir()]
    mor = [d for rec in ("19152060", "19154614") for day in ("July07", "July08") if (REL / rec / day).is_dir()
           for d in sorted((REL / rec / day).iterdir()) if (d / "rgb").is_dir()]
    res["bubye"] = release_candidates(bub, 100.0)[:10]
    res["morphocam"] = release_candidates(mor, 1000.0)[:10]
    json.dump(res, open(RES / "aligned_frames.json", "w"), indent=1, default=float)
    for k in ("cps2_best", "cps2_moving", "cps2_herd", "cps2_near", "cps2_far"):
        print(k, [(c["snippet"], c["frame"], round(c["score"], 3), round(c["moving_score"], 3), round(c["dist"], 1))
                  for c in res[k][:5]])
    for k in ("bubye", "morphocam"):
        print(k, [(Path(c["rgb"]).parent.parent.name, Path(c["rgb"]).stem, round(c["score"], 3),
                   round(c["purity"], 2), round(c["spill"], 2), c["n_same"]) for c in res[k][:5]])


if __name__ == "__main__":
    main()
