"""Side study: does Marigold (prs-eth, diffusion MDE) change the E3 conclusions vs MoGe-2?

Samples N frames with animals (evenly across snippets that are already cached), runs
Marigold v1.1 (4 DDIM steps, no ensembling, 768 px), and writes affine-invariant depth to
$WD2_CACHE/../wd2_marigold/<snippet>/<id>.npy plus the frame list. Then:

    WD2_DEPTH_DIR=/mnt/hdd/WildDepth/wd2_marigold python scripts/e3_radar_scale.py extract \
        --frames-file results/marigold_subset_frames.txt --tag marigold_subset
    python scripts/e3_radar_scale.py extract --frames-file results/marigold_subset_frames.txt --tag moge_subset
    python scripts/e3_radar_scale.py summarize --tag marigold_subset   (and moge_subset)

Marigold is affine-invariant: only affine protocols (LC-scene, LC-box, RC-box affine) are
meaningful for it; UC / scale-only rows are not.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wd2 import io  # noqa: E402

OUT_DIR = io.CACHE_ROOT.parent / "wd2_marigold"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    sys.path.insert(0, str(io.REPO / "scripts"))
    from e1_hidden_baselines import load_marigold, marigold_depth

    cands = []
    for s in io.list_snippets():
        fs = [f for f in io.list_frames(s) if io.cache_path(s, f).exists()]
        fs = [f for f in fs if (io.load_cache(s, f)["conf"] >= 0.3).any()] if fs else []
        if fs:
            cands.append((s, fs))
    per = max(1, a.n // len(cands))
    pick = [(s, f) for s, fs in cands for f in np.array(fs)[np.linspace(0, len(fs) - 1, min(per, len(fs))).astype(int)]]
    (io.REPO / "results/marigold_subset_frames.txt").write_text("".join(f"{s} {f}\n" for s, f in pick))
    pipe = load_marigold()
    t0 = time.time()
    for s, f in pick:
        p = OUT_DIR / s / f"{f}.npy"
        if p.exists():
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, marigold_depth(pipe, io.load_rgb(s, f)).astype(np.float32))
    print(f"{len(pick)} frames from {len(cands)} snippets, {time.time() - t0:.0f}s")
    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
