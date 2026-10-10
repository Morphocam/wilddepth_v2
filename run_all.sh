#!/usr/bin/env bash
# Reproduce every WildDepth-v2 number in results/*.json.  ~2 h GPU + ~1 h CPU on one RTX-class GPU.
set -euo pipefail
export WD2_RADAR_ROOT=${WD2_RADAR_ROOT:-/mnt/hdd/WildDepth/radar/snippets}
export WD2_CACHE=${WD2_CACHE:-/mnt/hdd/WildDepth/wd2_cache}
cd "$(dirname "$0")"
python scripts/00_precompute.py                                   # GPU: boxes, SAM2 masks, MoGe-2 (resumable)
python scripts/e10_e11_data_and_calib.py                          # dataset stats, sync, residual, ego frames
python scripts/e11b_extrinsic_check.py                            # per-frame vs anchor LiDAR->camera extrinsic
python scripts/e3_radar_scale.py extract --extrinsic anchor --tag anchor
python scripts/e3_radar_scale.py summarize --tag anchor
python scripts/e3_radar_scale.py extract --extrinsic per_frame --tag per_frame   # sensitivity
python scripts/e3_radar_scale.py summarize --tag per_frame
python scripts/e4_doppler_transfer.py extract                     # Doppler -> LiDAR motion masks (+ flow labels)
for k in 5 15 30; do WD2_K_ACC=$k python scripts/e5_gated_accumulation.py; done
python scripts/e6_tracking.py extract                             # E6 tracking + E2 jitter, E8 Doppler, E9 distance sampling
python scripts/e1_hidden_baselines.py --models moge2 da2m marigold dpt depthpro da2 --trackb   # GPU, organisers only
# Marigold robustness study (supplement)
python scripts/x_marigold_subset.py --n 240
F=results/marigold_subset_frames.txt
WD2_DEPTH_DIR=${WD2_CACHE%/*}/wd2_marigold python scripts/e3_radar_scale.py extract --frames-file $F --tag marigold_subset
python scripts/e3_radar_scale.py extract --frames-file $F --tag moge_subset
python scripts/e3_radar_scale.py summarize --tag marigold_subset
python scripts/e3_radar_scale.py summarize --tag moge_subset
python scripts/make_figures.py
python scripts/select_aligned_frames.py    # ranks best-aligned example frames (Fig. 1, 3)
python scripts/make_overview_figures.py   # Fig. 1 overview, Fig. 3 radar examples (needs release + WildPose drive)
python scripts/make_tables.py
