"""Paper figures from results/*. Each figure is skipped if its inputs are missing.

Colour follows the ENTITY, fixed across all figures (validated categorical palette, light mode):
  LiDAR  #2a78d6   radar  #eb6834   RGB-only / MDE  #1baf7a   fused RGB+radar  #4a3aa7
Aqua is below 3:1 contrast on white, so every series drawn in it is also direct-labelled.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wd2 import io  # noqa: E402

RES = io.REPO / "results"
FIG = io.REPO / "figures"
C = {"lidar": "#2a78d6", "radar": "#eb6834", "rgb": "#1baf7a", "fused": "#4a3aa7",
     "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "axes.edgecolor": C["ink2"], "axes.labelcolor": C["ink"], "xtick.color": C["ink2"],
    "ytick.color": C["ink2"], "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "pdf.fonttype": 42, "legend.frameon": False,
})


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_calib_budget():
    p = RES / "e11_calib_residuals.npz"
    if not p.exists():
        return
    z = np.load(p)
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.3), gridspec_kw={"wspace": 0.35})
    for key, lab, col, ypos in (("bev", "in-plane (range, azimuth)", C["lidar"], 0.55),
                                ("dz", "vertical (elevation)", C["radar"], 0.25)):
        xs = np.sort(z[key])
        axs[0].plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=col)
        med = np.median(xs)
        axs[0].plot([med], [0.5], "o", color=col, markersize=4)
        axs[0].text(med * 1.25, ypos, f"{lab}\nmedian {med:.2f} m", color=C["ink"], fontsize=7)
    axs[0].set_xscale("log")
    axs[0].set_xlim(1e-2, 30)
    axs[0].set_xlabel("static radar return → nearest LiDAR point (m)")
    axs[0].set_ylabel("fraction of returns")
    axs[0].set_title("Radar→LiDAR extrinsic residual (CDF)", loc="left")
    dts = np.abs([v for s in io.list_snippets() for v in io.match_log(s).values()]) * 1e3
    xs = np.sort(dts)
    axs[1].plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=C["radar"])
    axs[1].set_xscale("log")
    axs[1].set_xlabel("|radar − LiDAR| timestamp offset (ms)")
    axs[1].set_ylabel("fraction of frames")
    p95 = np.percentile(dts, 95)
    axs[1].axvline(p95, color=C["ink2"], linewidth=1, linestyle="--")
    axs[1].text(p95 * 1.1, 0.2, f"95th pct\n{p95:.0f} ms", color=C["ink"], fontsize=7)
    axs[1].set_title("Radar–LiDAR synchronization", loc="left")
    save(fig, "calib_budget")


def fig_hidden_trackA():
    p = RES / "e1_scores.json"
    if not p.exists():
        return
    s = json.load(open(p))
    names = {"A/moge2": "MoGe-2", "A/da2m": "DA-V2 metric", "A/marigold": "Marigold", "A/depthpro": "Depth Pro",
             "A/dpt": "DPT†", "A/da2": "DA-V2 rel.†"}
    rows = [(names[k], s[k]["lc_absrel"]) for k in names if k in s]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(3.3, 0.35 * len(rows) + 0.6))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], height=0.55, color=C["rgb"])
    for i, (n, v) in enumerate(rows):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7, color=C["ink"])
    ax.set_yticks(y, [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("LC AbsRel (lower is better)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Hidden test, Track A", loc="left")
    save(fig, "hidden_trackA")


def fig_radar_scale_vs_range():
    p = RES / "e3_summary_anchor.json"
    if not p.exists():
        return
    s = json.load(open(p))["test_frames_with_radar_anchor"]
    series = [("LC-box held-out (LiDAR)", "LiDAR, box", C["lidar"], "-"),
              ("LC-scene (LiDAR)", "LiDAR, scene", C["lidar"], "--"),
              ("RC-box scale-only (radar)", "radar + MoGe-2", C["fused"], "-"),
              ("Radar range only", "radar only", C["radar"], "-"),
              ("MoGe-2 metric (no anchor)", "MoGe-2 only", C["rgb"], "-")]
    bins = list(next(iter(v for v in s.values() if isinstance(v, dict) and "bins" in v))["bins"])
    x = np.arange(len(bins))
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for key, lab, col, ls in series:
        if key not in s or "bins" not in s[key]:
            continue
        y = [s[key]["bins"][b].get("AbsRel", np.nan) for b in bins]
        ax.plot(x, y, color=col, linestyle=ls, marker="o", markersize=4)
        ax.text(x[-1] + 0.08, y[-1], lab, va="center", fontsize=7, color=C["ink"])
    ax.set_xticks(x, [b.replace("-", "–") + " m" for b in bins])
    ax.set_xlim(-0.2, len(bins) - 0.2 + 1.1)
    ax.set_ylabel("camera-to-animal AbsRel")
    ax.set_title("Metric scale by anchor source (test snippets)", loc="left")
    save(fig, "radar_scale_vs_range")


def fig_density():
    p = RES / "e3_records_anchor.csv"
    if not p.exists():
        return
    r = list(csv.DictReader(open(p)))
    gt = np.array([float(x["gt"]) for x in r])
    nl = np.array([float(x["n_lidar"]) for x in r])
    nr = np.array([float(x["n_radar"]) for x in r])
    bands = [(5, 20), (20, 30), (30, 55)]
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    x = np.arange(len(bands))
    w = 0.36
    ml = [np.median(nl[(gt >= a) & (gt < b)]) for a, b in bands]
    mr = [np.median(nr[(gt >= a) & (gt < b)]) for a, b in bands]
    ax.bar(x - w / 2 - 0.01, ml, w, color=C["lidar"], label="LiDAR points in SAM2 mask (1 sweep)")
    ax.bar(x + w / 2 + 0.01, mr, w, color=C["radar"], label="radar returns associated to box")
    for i in range(len(bands)):
        ax.text(x[i] - w / 2, ml[i], f"{ml[i]:.0f}", ha="center", va="bottom", fontsize=7)
        ax.text(x[i] + w / 2, mr[i], f"{mr[i]:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [f"{a}–{b} m" for a, b in bands])
    ax.set_ylabel("points per animal (median)")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "radar_lidar_density")


def fig_doppler_transfer():
    p = RES / "e4_summary.json"
    if not p.exists():
        return
    from wd2.frame import load_frame
    from wd2 import geometry as G
    s = json.load(open(p))
    top, seen = [], set()
    s["top_frames"] = sorted(s.get("top_frames_all", s["top_frames"]), key=lambda t: -t["lidar_in_box"])
    for t in s["top_frames"]:
        if t["snippet"] not in seen and t["precision"] >= 0.9:
            top.append(t)
            seen.add(t["snippet"])
        if len(top) == 3:
            break
    if not top:
        return
    fig, axs = plt.subplots(len(top), 2, figsize=(6.8, 2.5 * len(top)), gridspec_kw={"wspace": 0.05, "hspace": 0.2})
    axs = np.atleast_2d(axs)
    for row, t in zip(axs, top):
        fr = load_frame(t["snippet"], t["frame"])
        rgb = io.load_rgb(t["snippet"], t["frame"])
        h = np.abs(fr.radar["h"])
        inh = np.nan_to_num(G.doppler_transfer(fr.lidar_l, fr.radar_l, fr.radar["h"]))
        for ax, (u, v, mv, lab, col) in zip(row, [
                (fr.ru, fr.rv, fr.rok & (h > 0.5), "moving radar returns", C["radar"]),
                (fr.lu, fr.lv, fr.lok & (inh > 0.5), "LiDAR after Doppler transfer", C["fused"])]):
            ax.imshow(rgb)
            ax.scatter(u[mv], v[mv], s=5, c=col, edgecolors="white", linewidths=0.3)
            for x0, y0, x1, y1 in fr.boxes:
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", lw=0.8))
            ax.set_axis_off()
            ax.set_title(f"{t['snippet'].replace('snippet_', 'seq ')}, frame {t['frame']}: {lab} (n={int(mv.sum())})",
                         fontsize=7, loc="left")
    save(fig, "doppler_transfer")


def fig_tracking():
    p = RES / "e6_summary.json"
    q = RES / "e9_track_distances.json"
    if not (p.exists() and q.exists()):
        return
    d = json.load(open(q))
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    bins = np.linspace(0, 55, 23)
    ref = np.array(d["LiDAR ref"])
    ax.hist(ref, bins=bins, color="#cfcfcb", label="LiDAR (reference)")
    for key, lab, col in (("MoGe-2 metric", "MoGe-2 only", C["rgb"]), ("RC per track", "MoGe-2 + radar", C["fused"]),
                          ("UC per frame", "UC (1 LiDAR pt)", C["radar"])):
        v = np.array(d.get(key, []), float)
        v = v[np.isfinite(v)]
        if len(v):
            ax.hist(v, bins=bins, histtype="step", color=col, linewidth=1.6, label=lab)
    ax.set_xlabel("per-track median distance (m)")
    ax.set_ylabel("tracks")
    ax.legend(loc="upper right")
    ax.set_title("Detection distances by depth source", loc="left")
    save(fig, "distance_sampling")
    a = RES / "e8_pairs.npy"
    if a.exists():
        e = np.load(a)
        ok = np.isfinite(e[:, 1])
        ok = np.isfinite(e[:, 1]) & (e[:, 4] == 0)
        edges = np.linspace(-1.5, 1.5, 13)
        mid = (edges[:-1] + edges[1:]) / 2
        q = np.array([np.percentile(e[ok & (e[:, 1] >= a) & (e[:, 1] < b), 0], [25, 50, 75])
                      if (ok & (e[:, 1] >= a) & (e[:, 1] < b)).sum() >= 20 else [np.nan] * 3
                      for a, b in zip(edges[:-1], edges[1:])])
        fig, ax = plt.subplots(figsize=(3.2, 2.6))
        ax.fill_between(mid, q[:, 0], q[:, 2], color=C["radar"], alpha=0.2, linewidth=0)
        ax.plot(mid, q[:, 1], color=C["radar"], marker="o", markersize=4)
        ax.plot([-1.5, 1.5], [-1.5, 1.5], color=C["ink2"], linewidth=1, linestyle="--")
        ax.text(0.9, 1.15, "ideal", color=C["ink2"], fontsize=7)
        ax.text(0.55, -0.35, "median Doppler\n(band: IQR)", color=C["ink"], fontsize=7)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel("LiDAR range rate of the animal (m/s)")
        ax.set_ylabel("in-box Doppler (m/s)")
        ax.set_title("Doppler vs. LiDAR radial velocity (static rig)", loc="left")
        save(fig, "doppler_vs_lidar_velocity")


if __name__ == "__main__":
    for fn in (fig_calib_budget, fig_hidden_trackA, fig_radar_scale_vs_range, fig_density,
               fig_doppler_transfer, fig_tracking):
        fn()
