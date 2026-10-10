"""E11b: which LiDAR->camera extrinsic should the ground truth use?

Each CPS2 snippet ships a per-frame refined extrinsic (refined_extrinsics/<id>.txt) and one
anchor extrinsic (refined_extrinsic.txt). frame_errors.csv shows the per-frame solution drifting
away from the anchor, by up to ~25 deg / 10 m in some snippets -- implausible for a rigid mount.

Independent check (does not share the parameter under test): project LiDAR with each extrinsic
and measure the inter-quartile range of LiDAR depth inside each SAM2 animal mask. A correctly
aligned projection puts the animal's own points in the mask (small IQR); a misaligned one
mixes in background (large IQR). Reported per snippet and by drift magnitude.

Output: results/e11b_extrinsic_check.csv / .json
"""
from __future__ import annotations

import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wd2 import io  # noqa: E402
from wd2.frame import lidar_in_mask, load_frame  # noqa: E402

OUT = io.REPO / "results"
STRIDE = 5


def iqr_list(fr):
    out = []
    for k in range(len(fr.boxes)):
        z = fr.lz[lidar_in_mask(fr, k)]
        z = z[z < 150]
        out.append((len(z), float(np.subtract(*np.percentile(z, [75, 25]))) if len(z) >= 5 else np.nan))
    return out


def process(snip):
    fe = io.snippet_dir(snip) / "frame_errors.csv"
    ferr = {int(r["stem"]): (float(r["rot_err_deg"]), float(r["trans_err_m"]))
            for r in csv.DictReader(fe.open())} if fe.exists() else {}
    rows = []
    for f in io.list_frames(snip)[::STRIDE]:
        a, b = load_frame(snip, f, "per_frame"), load_frame(snip, f, "anchor")
        if a is None or len(a.boxes) == 0:
            continue
        for k, ((na, ia), (nb, ib)) in enumerate(zip(iqr_list(a), iqr_list(b))):
            rows.append(dict(snippet=snip, frame=f, k=k, rot_drift=ferr.get(f, (np.nan,))[0],
                             trans_drift=ferr.get(f, (np.nan, np.nan))[1],
                             n_per_frame=na, iqr_per_frame=ia, n_anchor=nb, iqr_anchor=ib))
    return rows


def main():
    with Pool(8) as p:
        rows = [r for rs in p.map(process, io.list_snippets()) for r in rs]
    with open(OUT / "e11b_extrinsic_check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    a = np.array([[r["rot_drift"], r["iqr_per_frame"], r["iqr_anchor"], r["n_per_frame"], r["n_anchor"]] for r in rows], float)
    ok = np.isfinite(a[:, 1]) & np.isfinite(a[:, 2])
    out = {}
    for name, sel in [("all", ok), ("drift<1deg", ok & (a[:, 0] < 1)), ("1-5deg", ok & (a[:, 0] >= 1) & (a[:, 0] < 5)),
                      (">=5deg", ok & (a[:, 0] >= 5))]:
        if sel.sum():
            out[name] = dict(n=int(sel.sum()), iqr_per_frame_med=float(np.median(a[sel, 1])),
                             iqr_anchor_med=float(np.median(a[sel, 2])),
                             frac_per_frame_better=float(np.mean(a[sel, 1] < a[sel, 2])),
                             n_lidar_per_frame_med=float(np.median(a[sel, 3])), n_lidar_anchor_med=float(np.median(a[sel, 4])))
    per_snip = {}
    for s in sorted({r["snippet"] for r in rows}, key=lambda s: int(s.split("_")[1])):
        m = np.array([r["snippet"] == s for r in rows]) & ok
        if m.sum():
            per_snip[s] = dict(n=int(m.sum()), drift_med=float(np.nanmedian(a[m, 0])),
                               iqr_per_frame=float(np.median(a[m, 1])), iqr_anchor=float(np.median(a[m, 2])))
    out["per_snippet"] = per_snip
    json.dump(out, open(OUT / "e11b_extrinsic_check.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "per_snippet"}, indent=1))
    for s, v in per_snip.items():
        print(f"{s:<11} n={v['n']:>4} drift={v['drift_med']:6.2f}deg  IQR per-frame={v['iqr_per_frame']:6.2f}  anchor={v['iqr_anchor']:6.2f}")


if __name__ == "__main__":
    main()
