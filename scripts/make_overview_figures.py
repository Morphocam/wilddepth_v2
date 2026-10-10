"""Figure 1 (dataset overview) and Figure 3 (RGB-LiDAR-radar examples) from the data.

Radar-subset, Bubye and MorphoCam frames are the best-aligned examples ranked by
scripts/select_aligned_frames.py (best case, stated in the captions).

Frames are chosen automatically, not by hand: per site, MegaDetector scores sampled frames and the
frame maximizing detection confidence x sqrt(animal area) is used (confidence >= 0.5, 0.2-25% of the image); for Figure 3,
one walking-herd frame with Doppler, one static frame, and one frame with a distant (> 30 m) animal
are taken from the radar-subset records of E3/E4.

Colour: range = one-hue sequential blue (light near -> dark far); Doppler = diverging
blue (receding) <-> grey (static) <-> red (approaching); static LiDAR context = neutral grey.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wd2 import geometry as G, io  # noqa: E402
from wd2.frame import lidar_in_mask, load_frame  # noqa: E402

FIG = io.REPO / "figures"
RES = io.REPO / "results"
REL = Path(os.environ.get("WD2_RELEASE", "/mnt/hdd/WildDepth/wilddepth_release"))
KGA = Path(os.environ.get("WD2_KGALAGADI", "/mnt/Longleat/frankenstein/wildpose"))
MD_W = os.environ.get("WD2_MD_WEIGHTS", "/mnt/hdd/WildDepth/depth_eval/weights/MDV6-yolov9-c.pt")

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RANGE_CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"])
DOP_CMAP = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#9ec5f4", "#f0efec", "#f0a8a7", "#e34948"])
RANGE_NORM = Normalize(5, 60)
DOP_NORM = Normalize(-1.2, 1.2)

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "pdf.fonttype": 42, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK})


# ------------------------------------------------------------------ frame selection
_MD = None


def md_best(paths, n=24):
    """Return (path, score, box) of the sampled frame maximizing confidence x sqrt(area fraction)."""
    global _MD
    if _MD is None:
        from ultralytics import YOLO
        _MD = YOLO(MD_W)
    paths = [paths[i] for i in np.linspace(0, len(paths) - 1, min(n, len(paths))).astype(int)]
    best = (None, 0.0, None)
    for p in paths:
        im = np.asarray(Image.open(p).convert("RGB"))
        if im.mean() / 255 > 0.62:          # skip washed-out / overexposed frames
            continue
        r = _MD(im[..., ::-1].copy(), conf=0.3, verbose=False)[0].boxes
        H, W = im.shape[:2]
        for c, cl, b in zip(r.conf.cpu().numpy(), r.cls.cpu().numpy(), r.xyxy.cpu().numpy()):
            frac = (b[2] - b[0]) * (b[3] - b[1]) / (H * W)
            score = float(c) * np.sqrt(min(frac, 0.1))
            if cl == 0 and c >= 0.5 and 0.002 <= frac <= 0.25 and score > best[1]:
                best = (p, score, b)
    return best


ALIGNED = RES / "aligned_frames.json"     # written by scripts/select_aligned_frames.py


def site_frames():
    """Kgalagadi: MegaDetector choice (RGB only). Bubye / MorphoCam: best LiDAR alignment (select_aligned_frames)."""
    cache = RES / "overview_frames.json"
    out = json.load(open(cache)) if cache.exists() else {}
    if "kgalagadi" not in out:
        kga = []
        for d in sorted(KGA.glob("2022-12-12_00*")):
            kga += sorted((d / "sync_rgb").glob("*"))
        p, c, b = md_best(kga, n=60)
        out["kgalagadi"] = dict(rgb=str(p), depth=None, conf=c, box=[float(x) for x in b])
        json.dump(out, open(cache, "w"), indent=1)
    al = json.load(open(ALIGNED))
    for key in ("bubye", "morphocam"):
        def sz(c):   # clean silhouette, fully in frame: alignment x sqrt(size)
            x0, y0, x1, y1 = c["box"]
            inside = x0 > 0.02 * 1920 and y0 > 0.02 * 1080 and x1 < 0.98 * 1920 and y1 < 0.98 * 1080
            ok = c["purity"] >= 0.85 and c["spill"] <= 0.2 and inside
            return ok * c["lidar"] * np.sqrt(min((x1 - x0) * (y1 - y0) / (1920 * 1080), 0.05) / 0.05)
        out[key] = max(al[key], key=sz)
    return out


def _first(lst, avoid=()):
    for c in lst:
        if c["snippet"] not in avoid:
            return c
    return lst[0]


_SAM = None


def release_mask(rgb, box):
    """SAM2 mask of the animal in a released frame (same model as the benchmark pipeline)."""
    global _SAM
    if _SAM is None:
        from ultralytics import SAM
        _SAM = SAM(os.environ.get("WD2_SAM_WEIGHTS", "/mnt/hdd/WildDepth/depth_eval/sam2.1_b.pt"))
    return _SAM(rgb[..., ::-1].copy(), bboxes=[box], verbose=False)[0].masks.data.cpu().numpy()[0].astype(bool)


def best_animal(cand):
    ok = [a for a in cand["animals"] if a["radar_est"] is not None and a["radar_err"] < 0.05 and a["n_on"] >= 3]
    return max(ok, key=lambda a: a["lidar"] * a["radar"]) if ok else None


def rerank(cands, key="score", avoid=(), top=15, min_frac_on=0.0):
    """Re-rank the top candidates by alignment score x sqrt(size of the best-aligned animal)."""
    best, bs = None, -1.0
    for c in [c for c in cands if c["snippet"] not in avoid][:top]:
        a = best_animal(c)
        if a is None or a["frac_on"] < min_frac_on:
            continue
        fr = load_frame(c["snippet"], c["frame"])
        x0, y0, x1, y1 = fr.boxes[a["k"]]
        area = (x1 - x0) * (y1 - y0) / (io.W * io.H)
        sc = c[key] * np.sqrt(min(area, 0.04) / 0.04)
        if sc > bs:
            best, bs = c, sc
    return best or _first(cands, avoid)


def radar_examples():
    """Best-aligned radar-subset frames for Fig. 3: moving animal, standing animals, distant animals."""
    cache = RES / "fig3_frames.json"
    if cache.exists():
        return [tuple(x) for x in json.load(open(cache))]
    al = json.load(open(ALIGNED))
    mov = rerank(al["cps2_moving"], key="moving_score")
    near = rerank(al["cps2_near"], avoid={mov["snippet"]})
    far = rerank(al["cps2_far"], avoid={mov["snippet"], near["snippet"]}, min_frac_on=0.3)
    out = [(mov["snippet"], mov["frame"], f"moving animal, {mov['dist']:.0f} m", mov),
           (near["snippet"], near["frame"], f"standing animals, {near['dist']:.0f} m", near),
           (far["snippet"], far["frame"], f"distant animals, {far['dist']:.0f} m", far)]
    json.dump(out, open(cache, "w"), indent=1, default=float)
    return out


def overview_examples():
    """Fig. 1: CPS2 tile (best overall, not a Fig. 3 sequence) and a herd with Doppler on the animals."""
    al = json.load(open(ALIGNED))
    used = {e[0] for e in radar_examples()}
    tile = rerank(al["cps2_best"], avoid=used)
    herd = rerank([c for c in al["cps2_herd"] if c["moving_score"] > 0.2] or al["cps2_herd"], key="herd_score",
                  avoid=used | {tile["snippet"]})
    return tile, herd


def on_animal(fr):
    """Boolean masks over LiDAR / radar points that lie ON a detected animal (mask + depth, box + range)."""
    lon = np.zeros(len(fr.lu), bool)
    ron = np.zeros(len(fr.ru), bool)
    for k in range(len(fr.boxes)):
        sel = lidar_in_mask(fr, k) & (fr.lz < 150)
        if sel.sum() < 5:
            continue
        med = np.median(fr.lz[sel])
        lon |= sel & (np.abs(fr.lz - med) < 1.0)
        x0, y0, x1, y1 = fr.boxes[k]
        hgt = y1 - y0
        ron |= (fr.rok & (fr.ru >= x0) & (fr.ru <= x1) & (fr.rv >= y0 - 0.25 * hgt) & (fr.rv <= y1 + 0.25 * hgt)
                & (np.abs(fr.rz - med) < 1.5))
    return lon, ron


def draw_lidar(ax, fr, lon, s_on=2.4, step=3):
    bg = np.flatnonzero(fr.lok & (fr.lz < 120) & ~lon)[::step]
    ax.scatter(fr.lu[bg], fr.lv[bg], c=fr.lz[bg], cmap=RANGE_CMAP, norm=RANGE_NORM, s=0.25, alpha=0.35, linewidths=0)
    on = np.flatnonzero(lon)
    ax.scatter(fr.lu[on], fr.lv[on], c=fr.lz[on], cmap=RANGE_CMAP, norm=RANGE_NORM, s=s_on,
               edgecolors="#0d366b", linewidths=0.2)


def draw_radar(ax, fr, ron, s_on=16):
    bg = np.flatnonzero(fr.rok & (fr.rz < 120) & ~ron)
    ax.scatter(fr.ru[bg], fr.rv[bg], c=fr.radar["h"][bg], cmap=DOP_CMAP, norm=DOP_NORM, s=4, alpha=0.45,
               linewidths=0)
    on = np.flatnonzero(ron)
    on = on[np.argsort(np.abs(fr.radar["h"][on]))]
    ax.scatter(fr.ru[on], fr.rv[on], c=fr.radar["h"][on], cmap=DOP_CMAP, norm=DOP_NORM, s=s_on,
               edgecolors=INK, linewidths=0.45)


def label_best(ax, fr, cand, fs=6.0):
    """One label: LiDAR vs radar range of the best-aligned animal -- the paper's selling point."""
    a = best_animal(cand)
    if a is None:
        return
    x0, y0, x1, y1 = fr.boxes[a["k"]]
    above = y0 > 45
    ax.text((x0 + x1) / 2, y0 - 4 if above else y1 + 4,
            f"LiDAR {a['med']:.1f} m · radar {a['radar_est']:.1f} m", fontsize=fs, color=INK,
            va="bottom" if above else "top", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.9))


# ------------------------------------------------------------------ drawing helpers
def crop_box(box, W, H, aspect=4 / 3, ctx=2.6, min_frac=0.25):
    """4:3 crop centred on the animal, ctx x its size, at least min_frac of the image width."""
    x0, y0, x1, y1 = box
    cw = max(ctx * (x1 - x0), ctx * (y1 - y0) * aspect, min_frac * W)
    cw = min(cw, W, H * aspect)
    ch = cw / aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(np.clip(cx - cw / 2, 0, W - cw))
    top = int(np.clip(cy - ch / 2, 0, H - ch))
    return left, top, int(cw), int(ch)


def draw_bev(ax, fr, lon=None, ron=None, xmax=None, ymax=None):
    if lon is None:
        lon, ron = on_animal(fr)
    L = fr.lidar_l
    if xmax is None:
        d = fr.lz[lon]
        xmax = float(np.clip(1.4 * np.max(d) + 5 if len(d) else 40, 20, 60))
    ymax = ymax or 0.55 * xmax
    keep = (L[:, 0] > 0) & (L[:, 0] < xmax) & (np.abs(L[:, 1]) < ymax) & ~lon
    ax.scatter(-L[keep, 1], L[keep, 0], s=0.15, color="#c9c8c2", linewidths=0, rasterized=True)
    ax.scatter(-L[lon, 1], L[lon, 0], s=1.6, color=INK, linewidths=0, rasterized=True)
    R = fr.radar_l
    rk = (R[:, 0] > 0) & (R[:, 0] < xmax) & (np.abs(R[:, 1]) < ymax)
    bg = rk & ~ron
    ax.scatter(-R[bg, 1], R[bg, 0], c=fr.radar["h"][bg], cmap=DOP_CMAP, norm=DOP_NORM, s=4, alpha=0.45, linewidths=0)
    on = np.flatnonzero(rk & ron)
    ax.scatter(-R[on, 1], R[on, 0], c=fr.radar["h"][on], cmap=DOP_CMAP, norm=DOP_NORM, s=14,
               edgecolors=INK, linewidths=0.45)
    half = np.arctan(io.W / 2 / io.intrinsic()[0, 0])
    for sgn in (-1, 1):
        ax.plot([0, sgn * xmax * np.tan(half)], [0, xmax], color=INK2, linewidth=0.6, linestyle="--")
    ax.set_xlim(-ymax, ymax)
    ax.set_ylim(0, xmax)
    ax.set_aspect("equal")
    ax.set_xlabel("lateral (m)", labelpad=1)
    ax.set_ylabel("forward (m)", labelpad=1)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(color=GRID, linewidth=0.5)
    for sp in ("top", "left"):
        ax.spines[sp].set_visible(False)


def cbar(fig, rect, cmap, norm, label, ticks=None, ticklabels=None):
    cax = fig.add_axes(rect) if isinstance(rect, (list, tuple)) else rect
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", ticks=ticks)
    if ticklabels:
        cb.ax.set_xticklabels(ticklabels)
    cb.set_label(label, fontsize=7)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=6.5, length=2)


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote", name)


DOP_TICKS = dict(ticks=[-1.2, 0, 1.2], ticklabels=["−1.2", "0", "+1.2"])


# ------------------------------------------------------------------ Figure 1
def figure1():
    sites = site_frames()
    tile, herdc = overview_examples()
    fr = load_frame(tile["snippet"], tile["frame"])
    frb = load_frame(herdc["snippet"], herdc["frame"])
    stats = json.load(open(RES / "e10_e11_summary.json"))
    fig = plt.figure(figsize=(7.1, 4.15))
    outer0 = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.32], hspace=0.30)
    lower = outer0[1].subgridspec(2, 1, height_ratios=[1.25, 0.05], hspace=0.30)
    outer = [outer0[0], lower[0], lower[1]]
    gs = outer[0].subgridspec(1, 4, wspace=0.08)
    gb = outer[1].subgridspec(1, 4, width_ratios=[1.35, 0.95, 0.12, 1.6], wspace=0.05)
    tiles = [
        ("kgalagadi", "Kgalagadi, South Africa", "WildPose zoom RGB + LiDAR", "26 species · 12–140 m"),
        ("bubye", "Bubye Valley, Zimbabwe", "4K RGB + Livox Mid-70", "6 species · 5–50 m"),
        ("morphocam", "Longleat, UK", "RGB + Livox Avia", "10 species · 5–50 m"),
        ("cps2", "Longleat, UK", "RGB + LiDAR + 4D radar", f"{stats['sync_frames']:,} synced frames"),
    ]
    for i, (key, title, rig, sub) in enumerate(tiles):
        ax = fig.add_subplot(gs[0, i])
        if key == "cps2":
            lon, ron = on_animal(fr)
            a = best_animal(tile)
            l, t, cw, ch = crop_box(fr.boxes[a["k"]], io.W, io.H, ctx=2.4, min_frac=0.4)
            ax.imshow(io.load_rgb(tile["snippet"], tile["frame"]))
            draw_lidar(ax, fr, lon, s_on=3.0, step=2)
            draw_radar(ax, fr, ron, s_on=18)
            label_best(ax, fr, tile, fs=5.3)
            ax.set_xlim(l, l + cw)
            ax.set_ylim(t + ch, t)
        else:
            st = sites[key]
            rgb = np.asarray(Image.open(st["rgb"]).convert("RGB"))
            Hh, Ww = rgb.shape[:2]
            l, t, cw, ch = crop_box(st["box"], Ww, Hh)
            ax.imshow(rgb[t:t + ch, l:l + cw])
            if st.get("depth") and Path(st["depth"]).exists():
                d = np.asarray(Image.open(st["depth"])).astype(float)[t:t + ch, l:l + cw] / st["scale"]
                step = max(1, cw // 320)
                v, u = np.nonzero(d[::step, ::step] > 0)
                z = d[::step, ::step][v, u]
                mk = release_mask(rgb, st["box"])[t:t + ch, l:l + cw][::step, ::step]
                on = mk[v, u] & (np.abs(z - st.get("med", -99)) < 1.0)
                ax.scatter(u[~on] * step, v[~on] * step, c=z[~on], cmap=RANGE_CMAP, norm=RANGE_NORM, s=0.4,
                           alpha=0.35, linewidths=0)
                ax.scatter(u[on] * step, v[on] * step, c=z[on], cmap=RANGE_CMAP, norm=RANGE_NORM, s=2.2,
                           edgecolors="#0d366b", linewidths=0.2)
            else:
                ax.text(0.03, 0.04, "RGB (LiDAR in release)", transform=ax.transAxes, fontsize=5.5, color="white")
            ax.set_xlim(0, cw)
            ax.set_ylim(ch, 0)
        ax.set_axis_off()
        ax.set_title(title, fontsize=7, loc="left", pad=2, fontweight="bold")
        ax.text(0, -0.04, rig, transform=ax.transAxes, va="top", fontsize=6.3, color=INK)
        ax.text(0, -0.16, sub, transform=ax.transAxes, va="top", fontsize=6.3, color=INK2)

    axi = fig.add_subplot(gb[0, 0])
    lonb, ronb = on_animal(frb)
    axi.imshow(io.load_rgb(frb.snip, frb.f))
    draw_lidar(axi, frb, lonb, s_on=2.0)
    draw_radar(axi, frb, ronb)
    for x0, y0, x1, y1 in frb.boxes:
        axi.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", lw=0.6))
    label_best(axi, frb, herdc, fs=5.5)
    axi.set_xlim(0, io.W)
    axi.set_ylim(io.H, 0)
    axi.set_axis_off()
    axi.set_title(f"Longleat herd (seq {frb.snip.split('_')[1]}): LiDAR and radar on the animals", fontsize=7, loc="left", pad=2)
    axb = fig.add_subplot(gb[0, 1])
    draw_bev(axb, frb, lonb, ronb)
    axb.set_ylabel("")
    axb.set_xlabel("")
    axb.set_title("bird's-eye view (m)", fontsize=7, loc="left", pad=2)

    axt = fig.add_subplot(gb[0, 3])
    axt.set_axis_off()
    tasks = [("T1", "Animal distance", "RGB", "hidden Track A"),
             ("T2", "Depth completion", "RGB + 1 LiDAR sweep", "hidden Track B"),
             ("T3", "Radar metric scale", "RGB + radar", "radar subset"),
             ("T4", "Motion segmentation", "LiDAR + radar Doppler", "radar subset"),
             ("T5", "Metric 3D tracking", "RGB (+ radar / LiDAR)", "radar subset")]
    axt.text(0.0, 1.0, "Benchmark tasks", fontsize=8, fontweight="bold", va="top", color=INK)
    for j, (tid, name, inp, where) in enumerate(tasks):
        y = 0.88 - j * 0.155
        axt.add_patch(plt.Rectangle((0, y - 0.12), 1.0, 0.135, transform=axt.transAxes, facecolor="#f4f3f0",
                                    edgecolor="none"))
        axt.text(0.03, y - 0.05, tid, fontsize=8, fontweight="bold", color=INK, va="center", transform=axt.transAxes)
        axt.text(0.14, y - 0.02, name, fontsize=7.3, color=INK, va="center", transform=axt.transAxes)
        axt.text(0.14, y - 0.085, f"{inp} · {where}", fontsize=6, color=INK2, va="center", transform=axt.transAxes)
    # legend strip: full width, aligned with the panels above
    gl = outer[2].subgridspec(1, 2, wspace=0.12)
    cbar(fig, fig.add_subplot(gl[0, 0]), RANGE_CMAP, RANGE_NORM, "LiDAR range (m)", ticks=[5, 10, 20, 30, 40, 50, 60])
    cbar(fig, fig.add_subplot(gl[0, 1]), DOP_CMAP, DOP_NORM,
         "radar Doppler (m/s):  receding  −  /  +  approaching", ticks=[-1.2, -0.6, 0, 0.6, 1.2],
         ticklabels=["−1.2", "−0.6", "0", "+0.6", "+1.2"])
    save(fig, "overview_v2")


# ------------------------------------------------------------------ Figure 3
def figure3():
    exs = radar_examples()
    fig, axs = plt.subplots(len(exs), 3, figsize=(7.1, 2.0 * len(exs)),
                            gridspec_kw={"width_ratios": [1.33, 1.33, 0.9], "wspace": 0.06, "hspace": 0.12})
    heads = ["RGB + LiDAR (range)", "RGB + radar (Doppler) and detections", "bird's-eye view (black: LiDAR on animals)"]
    for r, (row, (snip, f, lab, cand)) in enumerate(zip(axs, exs)):
        fr = load_frame(snip, f)
        rgb = io.load_rgb(snip, f)
        lon, ron = on_animal(fr)
        ax = row[0]
        ax.imshow(rgb)
        draw_lidar(ax, fr, lon)
        ax.set_xlim(0, io.W)
        ax.set_ylim(io.H, 0)
        ax.set_axis_off()
        ax.text(-0.03, 0.5, f"{lab}\nseq {snip.split('_')[1]}, frame {f}", transform=ax.transAxes, rotation=90,
                ha="right", va="center", fontsize=6.8, color=INK)
        ax2 = row[1]
        ax2.imshow(rgb)
        draw_radar(ax2, fr, ron)
        for x0, y0, x1, y1 in fr.boxes:
            ax2.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", lw=0.7))
        label_best(ax2, fr, cand)
        ax2.set_xlim(0, io.W)
        ax2.set_ylim(io.H, 0)
        ax2.set_axis_off()
        n_mov = int((ron & (np.abs(fr.radar["h"]) > 0.5)).sum())
        ax2.text(0.02, 0.03, f"{int(ron.sum())} radar returns on animals ({n_mov} moving)", transform=ax2.transAxes,
                 fontsize=6, color="white")
        draw_bev(row[2], fr, lon, ron)
        if r < len(exs) - 1:
            row[2].set_xlabel("")
        if r == 0:
            for a_, hd in zip(row, heads):
                a_.set_title(hd, fontsize=7.3, loc="left", pad=3)
    cbar(fig, [0.12, 0.05, 0.25, 0.008], RANGE_CMAP, RANGE_NORM, "LiDAR range (m)", ticks=[5, 20, 40, 60])
    cbar(fig, [0.45, 0.05, 0.25, 0.008], DOP_CMAP, DOP_NORM, "Doppler (m/s): receding − / + approaching", **DOP_TICKS)
    save(fig, "radar_qualitative")


if __name__ == "__main__":
    figure1()
    figure3()
