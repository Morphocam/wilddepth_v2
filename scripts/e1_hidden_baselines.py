"""E1: organiser baselines on the frozen hidden test (Track A: RGB only; Track B: RGB + 1 sweep).

Predictions are written as 16-bit millimetre PNGs (the submission format) and scored with the
competition's own scoring programs against the frozen reference. The frozen SHA-256 manifest is
verified first; nothing is written to the public leaderboard (use benchmark/leaderboard/
score_submission.py for that once the numbers are approved).

    python scripts/e1_hidden_baselines.py --models moge2 da2 dpt depthpro --trackb
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wd2 import geometry as G, io  # noqa: E402

BENCH = Path(os.environ.get("WD2_BENCH", "/mnt/hdd/WildDepth/benchmark"))
BUNDLE = BENCH / "codabench"
FROZEN = BENCH / "frozen/v1.3"
INP = BUNDLE / "input_data/test_hidden"
REF = BUNDLE / "reference_data/test_hidden"
PRED_ROOT = io.CACHE_ROOT.parent / "wd2_hidden_preds"
OUT = io.REPO / "results"
HF = {"da2": ("depth-anything/Depth-Anything-V2-Large-hf", "disparity"),
      "da2m": ("depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf", "depth"),
      "dpt": ("Intel/dpt-large", "disparity"),
      "depthpro": ("apple/DepthPro-hf", "depth")}


def verify_frozen():
    r = subprocess.run(["sha256sum", "-c", str(FROZEN / "SHA256SUMS"), "--quiet"], cwd=BUNDLE,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("frozen reference changed -- refusing to score\n" + r.stdout[:500])
    print("frozen reference v1.3 verified")


def frames():
    import csv
    return [(r["frame_id"], INP / r["rgb_path"]) for r in csv.DictReader(open(INP / "frames.csv"))]


def save_mm(path, depth_m):
    d = np.nan_to_num(depth_m, nan=0.0, posinf=0.0)
    cv2.imwrite(str(path), np.clip(d * 1000, 0, 65535).astype(np.uint16))


class Model:
    def __init__(self, name):
        self.name = name
        if name == "moge2":
            from moge.model.v2 import MoGeModel
            self.m = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").cuda().eval()
        elif name == "marigold":
            self.m = load_marigold()
        else:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            hid, self.kind = HF[name]
            self.proc = AutoImageProcessor.from_pretrained(hid)
            self.m = AutoModelForDepthEstimation.from_pretrained(hid).cuda().eval()

    @torch.no_grad()
    def __call__(self, rgb):
        h, w = rgb.shape[:2]
        if self.name == "marigold":
            return marigold_depth(self.m, rgb)
        if self.name == "moge2":
            x = torch.tensor(rgb / 255.0, dtype=torch.float32, device="cuda").permute(2, 0, 1)
            d = self.m.infer(x)["depth"].float().cpu().numpy()
            d[~np.isfinite(d)] = 0
            return d
        inp = self.proc(images=rgb, return_tensors="pt").to("cuda")
        pred = self.m(**inp).predicted_depth
        pred = torch.nn.functional.interpolate(pred[:, None].float(), size=(h, w), mode="bilinear")[0, 0].cpu().numpy()
        if self.kind == "disparity":            # relative inverse depth -> depth (scale/shift fixed by scorer)
            pred = 1.0 / np.maximum(pred, 1e-3 * pred.max())
        return pred


def load_marigold():
    """Marigold v1.1 (prs-eth). Affine-invariant depth in [0,1]; 4 DDIM steps, no ensembling."""
    from diffusers import MarigoldDepthPipeline
    return MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-depth-v1-1", torch_dtype=torch.float16).to("cuda")


@torch.no_grad()
def marigold_depth(pipe, rgb, steps: int = 4):
    from PIL import Image
    out = pipe(Image.fromarray(rgb), num_inference_steps=steps, ensemble_size=1, processing_resolution=768,
               match_input_resolution=True, generator=torch.Generator("cuda").manual_seed(0))
    d = np.asarray(out.prediction).squeeze().astype(np.float32)
    return 1.0 + 100.0 * d      # relative depth, positive; absolute scale is fixed by UC/LC alignment


def sparse_fill(lid):
    """Nearest-neighbour fill of the sparse sweep."""
    from scipy.ndimage import distance_transform_edt
    valid = lid > 0
    if not valid.any():
        return np.zeros_like(lid, float)
    _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    return lid[iy, ix].astype(float)


def ssa(pred, lid, rmax=55.0):
    """MoGe-2 aligned to the sparse sweep by trimmed affine (the paper's SSA, scene anchors)."""
    v = (lid > 0) & (lid < rmax) & (pred > 0.1)
    st = G.fit_affine(pred[v], lid[v]) if v.sum() >= 3 else None
    return pred if st is None else st[0] * pred + st[1]


def score(track, pred_dir):
    sys.path.insert(0, str(BUNDLE / ("scoring_program" if track == "A" else "scoring_program_trackb")))
    mod = __import__("score" if track == "A" else "score_trackb")
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
        os.symlink(REF, os.path.join(tmp, "ref"))
        os.symlink(pred_dir, os.path.join(tmp, "res"))
        mod.main(tmp, out)
        s = json.load(open(os.path.join(out, "scores.json")))
    sys.path.pop(0)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["moge2", "da2", "dpt", "depthpro"])
    ap.add_argument("--trackb", action="store_true")
    a = ap.parse_args()
    verify_frozen()
    fl = frames()
    results = json.load(open(OUT / "e1_scores.json")) if (OUT / "e1_scores.json").exists() else {}
    moge_dir = None
    for name in a.models:
        d = PRED_ROOT / f"A_{name}"
        d.mkdir(parents=True, exist_ok=True)
        if len(list(d.glob("*.png"))) < len(fl):
            m = Model(name)
            for fid, p in fl:
                rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                save_mm(d / f"{fid}.png", m(rgb))
            del m
            torch.cuda.empty_cache()
        results[f"A/{name}"] = score("A", d)
        print(name, results[f"A/{name}"], flush=True)
        if name == "moge2":
            moge_dir = d
    if a.trackb:
        nn, sa = PRED_ROOT / "B_sparse_nn_fill", PRED_ROOT / "B_moge2_ssa"
        nn.mkdir(parents=True, exist_ok=True)
        sa.mkdir(parents=True, exist_ok=True)
        for fid, p in fl:
            lid = cv2.imread(str(INP / "lidar" / f"{fid}.png"), -1).astype(float) / 1000.0
            save_mm(nn / f"{fid}.png", sparse_fill(lid))
            if moge_dir is not None:
                pr = cv2.imread(str(moge_dir / f"{fid}.png"), -1).astype(float) / 1000.0
                save_mm(sa / f"{fid}.png", ssa(pr, lid))
        results["B/sparse_nn_fill"] = score("B", nn)
        print("B nn", results["B/sparse_nn_fill"])
        if moge_dir is not None:
            results["B/moge2_ssa"] = score("B", sa)
            print("B moge2+ssa", results["B/moge2_ssa"])
    json.dump(results, open(OUT / "e1_scores.json", "w"), indent=1)


if __name__ == "__main__":
    main()
