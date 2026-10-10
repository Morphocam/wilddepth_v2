"""Stage 0 (GPU): per-frame animal boxes, SAM2 masks and MoGe-2 depth for every
synchronized RGB-LiDAR-radar frame. Resumable; writes $WD2_CACHE/<snippet>/<id>.npz.

    python scripts/00_precompute.py                       # all snippets, all frames
    python scripts/00_precompute.py --snippets snippet_28 --stride 5

Weights (paths via env or defaults):
    WD2_MD_WEIGHTS   MegaDetector V6 yolov9-c  (ultralytics format)
    WD2_SAM_WEIGHTS  SAM2.1-b                   (ultralytics format)
    MoGe-2 ViT-L normal from HuggingFace (Ruicheng/moge-2-vitl-normal)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wd2 import io  # noqa: E402

MD_W = os.environ.get("WD2_MD_WEIGHTS", "/mnt/hdd/WildDepth/depth_eval/weights/MDV6-yolov9-c.pt")
SAM_W = os.environ.get("WD2_SAM_WEIGHTS", "/mnt/hdd/WildDepth/depth_eval/sam2.1_b.pt")
MD_CONF = 0.2      # stored threshold; experiments filter higher (0.3)
ANIMAL = 0         # MegaDetector classes: 0 animal, 1 person, 2 vehicle
MIN_AREA = 400     # px^2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snippets", nargs="*")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--no-moge", action="store_true")
    a = ap.parse_args()

    from ultralytics import SAM, YOLO
    from moge.model.v2 import MoGeModel
    dev = "cuda"
    md, sam = YOLO(MD_W), SAM(SAM_W)
    moge = None if a.no_moge else MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(dev).eval()

    snips = a.snippets or io.list_snippets()
    for snip in snips:
        frames = io.list_frames(snip)[:: a.stride]
        out = io.CACHE_ROOT / snip
        out.mkdir(parents=True, exist_ok=True)
        todo = [f for f in frames if not io.cache_path(snip, f).exists()]
        t0 = time.time()
        for f in todo:
            rgb = io.load_rgb(snip, f)
            r = md(rgb[..., ::-1].copy(), conf=MD_CONF, verbose=False)[0]
            b = r.boxes
            keep = (b.cls.cpu().numpy() == ANIMAL)
            boxes = b.xyxy.cpu().numpy()[keep]
            conf = b.conf.cpu().numpy()[keep]
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) else np.zeros(0)
            boxes, conf = boxes[area >= MIN_AREA], conf[area >= MIN_AREA]
            if len(boxes):
                m = sam(rgb[..., ::-1].copy(), bboxes=boxes.tolist(), verbose=False)[0].masks
                masks = m.data.cpu().numpy().astype(bool) if m is not None else np.zeros((0, io.H, io.W), bool)
            else:
                masks = np.zeros((0, io.H, io.W), bool)
            if moge is not None:
                with torch.no_grad():
                    x = torch.tensor(rgb / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
                    o = moge.infer(x)
                    depth = o["depth"].float().cpu().numpy()
                    depth[~np.isfinite(depth)] = 0
            else:
                depth = np.zeros((io.H, io.W), np.float32)
            np.savez_compressed(io.cache_path(snip, f), boxes=boxes.astype(np.float32),
                                conf=conf.astype(np.float32),
                                masks_packed=np.packbits(masks, axis=-1),
                                moge=depth.astype(np.float16))
        dt = time.time() - t0
        print(f"{snip}: {len(todo)}/{len(frames)} frames in {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
