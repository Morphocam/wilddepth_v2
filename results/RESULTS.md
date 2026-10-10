# Results ledger — WildDepth-v2 (run 2026-09-30)

Every number below comes from a script in `scripts/` run on this machine; the file it is read
from is given. Environment: conda `wilddepth2` (py3.11, torch 2.11 cu128), RTX PRO 4000.
Radar subset: 37 July-8 snippets, **19,716 synchronized frames**; MegaDetector V6 (yolov9-c,
conf >= 0.3), SAM2.1-b masks, MoGe-2 ViT-L for every frame (`00_precompute.py`).

## Data and calibration (E10, E11, E11b) — `e10_e11_summary.json`, `e11b_extrinsic_check.json`
| quantity | value |
|---|---|
| synchronized RGB+LiDAR+radar frames / radar files | 19,716 / 22,348 (1 empty LiDAR file dropped) |
| frozen-radar snippets (July 8) | 0 |
| radar returns per frame (median of snippet medians) | 679 (LiDAR 19,390 points) |
| radar–LiDAR timestamp offset | mean 16.8 ms, p95 31.5 ms, max 350 ms |
| frames with Doppler (>2% returns \|h\|>0.5) / ego-motion frames | 24% / 35% |
| radar->LiDAR residual, static returns (827k) | in-plane median **0.10 m** (p75 0.26, p90 0.52; gated at 1 m); vertical median **1.84 m** (p75 4.4) |
| per-frame LiDAR->camera extrinsic drift from anchor (frame_errors.csv) | median 2.1 deg / 0.26 m; up to 25 deg / 10 m (seq 31, 37) |
| E11b: in-mask LiDAR IQR, drift >= 5 deg (4,156 animals) | per-frame 1.27 m vs **anchor 0.83 m**; anchor has 116 vs 50 LiDAR points on the animal; per-frame better in only 37% |

**Consequence:** all experiments use the anchor extrinsic (`refined_extrinsic.txt`). The shipped
`depthm/` maps were projected with the drifting per-frame extrinsic (match to 1 mm), and the
public Zenodo record 19103277 contains byte-identical maps -> **the public CPS2 depth maps are
misaligned in high-drift sequences** (e.g. 26, 29, 31, 34, 35, 37). Needs a release note / re-issue.

## E3 radar as a metric anchor — `e3_summary_anchor.json` (sensitivity: `e3_summary_per_frame.json`)
48,723 animals / 18,879 frames. GT = held-out half of the mask's LiDAR. Radar estimator chosen on
odd (dev) snippets, reported on even (test) snippets. Test, frames with a radar anchor (20,869):

| anchor | LiDAR at inference | AbsRel [95% CI] | medAE |
|---|---|---|---|
| MoGe-2 metric, none | no | 0.146 [0.145, 0.148] | 1.93 m |
| UC 1 LiDAR point | yes | 0.442 | 6.53 m |
| LC-scene | yes | 0.198 | 2.49 m |
| LC-box held-out | yes | **0.052** | 0.17 m |
| RC-box affine (radar) | **no** | **0.120** [0.113, 0.131] | 0.60 m |
| RC-pixel (radar) | no | 0.483 | 4.54 m |
| radar snapped to LiDAR (old study's anchor) | yes | 0.572 | 2.04 m |
| same 15,196 animals: MoGe-2 / radar range (MoGe-gated) / LC-box | no / no / yes | 0.144 / **0.064** / 0.046 | |

Radar-only estimator (no RGB): nearest range cluster, dev median rel. err. 4.2% at 90% coverage, heavy tail (AbsRel 0.35).
Per distance: radar-only is best below 30 m and worst at 30–55 m (`figures/radar_scale_vs_range`).
Conclusions identical with the per-frame extrinsic.
**Corrections:** the old "radar 8% beats LiDAR 10%" compared box radar to scene LiDAR; the old
"radar-refine 0.85 m" anchor used LiDAR depth; neither is a LiDAR-free result, and the latter
does not replicate at scale (0.572).

## E4 Doppler -> LiDAR motion masks — `e4_summary.json`
12,596 non-ego frames with animals; 2,783 event frames. Default (radii 0.6/1.0/1.5 m, tau 0.5 m/s):
densification **3.7x**, precision radar 0.65 / **LiDAR 0.72**; per sequence 0.39–0.98 (walking herds
27/28/31: 0.96–0.98). Isotropic 0.6 m: 1.75x at the same precision. Doppler flags 8.5% of animals;
vs an optical-flow motion label: precision 0.57, recall 0.28 (radial vs lateral motion).
**Correction:** the handoff's 84–100% precision was 5 hand-picked frames.

## E5 Doppler-gated accumulation — `e5_summary_k{5,15,30}.json`
Doppler-moving animals, dense MAE single / accumulated / gated: 0.5 s 2.02/1.96/2.02; 1.5 s
1.81/2.04/1.81; **3 s 1.64/2.60/1.64 (-37%)**. Gate precision 0.54 / 0.72 / 0.85. All animals at 3 s:
accumulated 2.25, Doppler-gated 2.17, flow-gated 2.08.
**Correction:** the old -44% is not reproduced as stated; smear matters only for multi-second windows.

## E6 metric tracking (+E2, E9) — `e6_summary.json`
431 tracks (>= 20 observations), 50,338 observations, IoU tracker + CV Kalman/RTS, LiDAR-lifted reference.
| source | LiDAR | pos RMSE | dist AbsRel | speed err | jitter | sigma bias (E9) |
|---|---|---|---|---|---|---|
| MoGe-2 | no | 2.20 m | 0.149 | 0.15 m/s | 0.31 m | -13.2% (KS p<1e-6) |
| UC per frame | yes | 6.27 | 0.399 | 0.28 | 1.72 | -8.1% |
| LC per frame | yes | 1.52 | 0.089 | 0.11 | 0.28 | -3.6% |
| LiDAR scale first frame | 1 frame | 0.52 | 0.031 | 0.10 | 0.35 | +0.2% |
| radar range | no | 0.53 | 0.036 | 0.13 | 0.72 | -2.1% (p=0.62) |
| **MoGe-2 x radar scale per track** | **no** | **0.58** | **0.033** | **0.10** | 0.35 | **-2.1%** (p=0.58) |
LiDAR self-jitter 0.12 m.

## E8 Doppler vs LiDAR radial velocity — `e6_summary.json` ("E8"), `e8_pairs.npy`
29,347 observations with >= 3 radar returns within 2 m of the animal's LiDAR range (analysis uses LiDAR).
All: r = 0.57 (static rig 0.61). Moving animals (\|range rate\| > 0.3 m/s, static rig, n = 3,080):
r = 0.60, Theil–Sen slope **0.47**, sign agreement 72%, median error 0.29 m/s. Doppler spread vs
speed: Spearman 0.54.
**Correction:** the old "r = 0.01, velocity unrecoverable" does not hold; Doppler tracks radial
velocity but under-reads it by ~half.

## E1 hidden test — `e1_scores.json` (frozen v1.3 verified; leaderboard NOT updated)
Track A LC AbsRel: MoGe-2 **0.094**, DA-V2 metric-outdoor 0.169, Marigold 0.252, DPT† 0.274,
Depth Pro 0.337, DA-V2 relative† 0.398 (†disparity inverted without shift — not a fair use of these models).
Track B (902 frames): NN-fill animal RMSE 6.10 / bg 4.51 m; MoGe-2 + SSA to the sweep: animal 5.33 / bg 4.52 m.
Composite scores still use the placeholder REF_DENSE.

## Marigold side study — `e3_summary_{moge,marigold}_subset.json`
233 frames / ~300 animals, same anchors: LC-scene 0.167 vs **0.132**; LC-box 0.060 vs 0.056;
RC-box 0.137 vs 0.131; RC leave-own-out 0.257 vs 0.194. Hidden Track A: 0.094 vs 0.252.
Marigold is affine-invariant, so it cannot be used unanchored or as the radar-association prior.
Use as a robustness check in the supplement.
