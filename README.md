# WildDepth-v2 — code for the RGB–LiDAR–radar wildlife paper

Code and experiment scripts for **WildDepth-v2: A Synchronized RGB–LiDAR–Radar Dataset and
Benchmark for Metric Depth, Motion, and Tracking of Wildlife** (IJCV special issue on Computer
Vision Approaches for Animal Tracking and Modeling, extension of WildDepth, ECCV 2026).

Every number the paper reports from the radar subset and the hidden benchmark is produced by a
script in `scripts/` and lands in `results/*.json`. `results/RESULTS.md` maps each paper claim
to its file.

## Layout
```
wd2/                    library
  io.py                 data access (PCD parsing, snippets, frames, calibration, cache)
  geometry.py           projection, robust fits, Doppler transfer, ego-motion fit
  frame.py              one synchronized frame projected into the camera; radar association
  metrics.py            distance metrics + frame-level bootstrap CIs
  track.py              IoU tracker, constant-velocity Kalman + RTS smoother
configs/calib/          radar->LiDAR extrinsic, RealSense intrinsics
scripts/
  00_precompute.py      GPU: MegaDetector V6, SAM2.1 masks, MoGe-2 depth for every frame
  e10_e11_data_and_calib.py   dataset statistics, sync, ego frames, radar->LiDAR residual
  e11b_extrinsic_check.py     per-frame vs anchor LiDAR->camera extrinsic (in-mask IQR test)
  e3_radar_scale.py     E3  metric-scale calibration by anchor source (fair protocol)
  e4_doppler_transfer.py      E4  Doppler -> LiDAR motion masks, precision/recall, ablation
  e5_gated_accumulation.py    E5  Doppler-gated LiDAR accumulation + gate ablation
  e6_tracking.py        E6  metric 3D tracking by depth source (+E2 jitter, E8 Doppler, E9 distance sampling)
  e1_hidden_baselines.py      E1  baselines on the frozen hidden test (Tracks A, B)
  make_figures.py / make_tables.py   paper figures and LaTeX tables from results/
tests/                  unit tests (pytest)
results/                small JSON summaries (committed); large CSVs are regenerated
```

## Setup
```bash
conda env create -f environment.yml
conda activate wilddepth2
pip install git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183
pip install --no-deps git+https://github.com/microsoft/MoGe.git
pip install -e .
pytest -q tests
```
Weights: MegaDetector V6 (yolov9-c, ultralytics format) and SAM2.1-b (`sam2.1_b.pt`) paths via
`WD2_MD_WEIGHTS` / `WD2_SAM_WEIGHTS`; MoGe-2 and the Track-A baselines download from HuggingFace.

## Data
| variable | default | contents |
|---|---|---|
| `WD2_RADAR_ROOT` | `/mnt/hdd/WildDepth/radar/snippets` | 37 July-8 snippets: `rgb/ lidar/ radar/ depthm/ refined_extrinsics/ radar_match_log.csv` |
| `WD2_CACHE` | `/mnt/hdd/WildDepth/wd2_cache` | per-frame outputs of `00_precompute.py` (~10 GB) |
| `WD2_BENCH` | `/mnt/hdd/WildDepth/benchmark` | frozen hidden-test bundle (organisers only) |

`io.list_snippets()` raises if the root holds no snippets (the old loader silently returned 0).

## Reproduce
```bash
./run_all.sh          # ~1.5 h GPU + ~1 h CPU on one RTX-class GPU
```

## Conventions and known data issues
- Frame ids are a global 10 Hz index shared by RGB, LiDAR and radar; a frame is used only when
  all three exist and are non-empty (one LiDAR file, `snippet_2/lidar/3000.pcd`, is 0 bytes).
- Radar is associated to animals at the **object level** only (box columns, vertical padding of
  one box height) because the radar extrinsic's elevation error is 1–2 m.
- Methods that need LiDAR at inference are marked; "radar" methods never read LiDAR.
- Ground truth for E3/E5 uses a random half of the mask's LiDAR points; the other half is the
  LiDAR anchor, so LiDAR-anchored baselines are not scored on their own anchor points.
- See `results/RESULTS.md` for the LiDAR->camera extrinsic drift issue found in E11b.

## License
TODO: choose a license before making the repository public.
