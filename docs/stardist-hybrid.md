# Sensitive StarDist proposals with surface-loss contours

[Documentation](README.md) · [Models](models.md) · [Analysis](fish-analysis.md)

The focused app now has a **Use sensitive StarDist hybrid** preset under
**Models and device**. It selects the original sensitive StarDist, the current
surface-loss crop U-Net and the signal-center U-Net. The current default remains
the center/box detector with the same two downstream models.

StarDist supplies bounding boxes and physical-distance-transform interior seeds.
Its saved sensitivity is preserved, instead of applying the app's conservative
StarDist preset. The original sensitive detector's saved preliminary geometry
filters still apply; the final crop contours have no roundness filter.
The crop U-Net redraws contours from membrane pixels on its calibrated grid.
It starts with an empty label volume, rather than using the detector masks.
Contours require a connected component at the seed, volume at least 5 µm³,
at least 65% supported surface samples and mean membrane contrast at least 0.04.
Crop/scan-edge contours and duplicate/conflicting proposals are rejected.
Accepted cells retain voxel ownership; removed overlaps are rechecked for
connectivity, size and membrane support. Failed proposals are discarded, never
replaced with StarDist contours. This uses the default center/box pipeline's
existing contour acceptance checks.

The signal-center U-Net then runs inside accepted masks. Cells without signals
are removed, and physical mesh volumes, nearest-surface B points, A–B distances
and normal Imaris/two-sheet Excel exports follow. No model was retrained.

## Full held-out comparison

All 144 held-out scans (71 test and 73 validation) have now been evaluated with
all three modes, for 432 completed runs and zero failures. The test set contains
508 complete manual cells. Test-set volume MAPE is 6.59% for the default, 10.43%
for the hybrid and 8.95% for consensus; distance MAE is 0.112, 0.154 and 0.129 µm,
respectively. On the same 160 complete-cell volume pairs, the default is 5.93%
and hybrid/consensus are both 8.58%. The hybrid improves complete-cell recall
(373/508 versus 315/508), while consensus finds 300/508.

[Full comparison, validation results and confidence intervals](../reports/all-heldout-comparison/README.md)
include per-scan and paired measurement data. The earlier four-scan pilot below
is preserved as a preliminary experiment. Full results favour the default for
measurement accuracy. Unmatched cells remain unscored because annotations are
incomplete, so cell precision still requires a reviewed reference set.

## Fixed validation sample

Four scans from four acquisition groups were sampled with seed 20261009 before
this experiment, using the same sample as the cell-recovery benchmark. There
are 40 manual reference cells. Both pipelines use the same surface-loss and
signal-center weights. Manual cell matches require whole-object IoU ≥ 0.5.
Distances require matched cells and same-channel manual/NN signal centers within
0.6 µm. Original manual mesh volumes and A–B distances are the references.
Previously completed baseline runs were reused; their metrics were recalculated.

| Metric | Current default | Sensitive StarDist hybrid |
|---|---:|---:|
| Cells before signal filtering | 216 | 346 |
| Manual matches before filtering | 24/40 (60%) | 32/40 (80%) |
| Cells retained with signals | 47 | 50 |
| Manual matches after filtering | 13 | 17 |
| Signals retained | 49 | 54 |
| Volume mean absolute percentage error, own matched subset | 9.42% (13 cells) | 13.34% (17 cells) |
| Distance MAE, own matched subset | 0.179 µm (8 signals) | 0.163 µm (9 signals) |
| Volume error, same 12 retained manual cells | 9.87% | 6.40% |
| Distance MAE, same 8 manual signals | 0.179 µm | 0.181 µm |

The hybrid recovers five additional retained manual cells and loses one, for a
net gain of four. Contours improve on shared cells, but some newly recovered
cells have large volume errors (up to 50.44%). Distance accuracy on shared signals
is effectively unchanged. More annotated recall therefore does not establish
an improvement in every measurement.

| Scan | Default manual matches, before/after filtering | Hybrid manual matches, before/after filtering |
|---|---:|---:|
| 121113_HAG4_Lam633_22A488a.ims | 8/8 | 9/8 |
| dsLacZ3.ims | 4/2 | 9/4 |
| c.ims | 6/2 | 8/5 |
| 080212_HAG50-2_Lam633m_K9546a.ims | 6/1 | 6/0 |

These are exploratory validation results, with small matched subsets. Unmarked
cells remain unknown, so unmatched predictions cannot estimate cell precision.
No threshold was tuned on test scans. Nine synthetic proposals across blank,
noise-only and bright-puncta-without-membrane images were all rejected using the
actual surface-loss checkpoint; this does not replace a manual precision review.

The native Linux GUI smoke test checked 2D/3D display, A/B geometry, automatic and
button Excel exports with exactly two sheets, positive mesh volumes and unchanged
source image bytes. Imaris itself and Windows were not run in this environment.
The test suite passed 182 tests with six optional-dependency tests skipped.

Results: [summary](../reports/stardist-hybrid/summary.json). Full per-scan runs and
measurements are in the historical run folder (removed during cleanup).
Reproduce with `.venv/bin/python scripts/benchmark_stardist_hybrid.py` and
`FISH_SPOTS_PYTHON` set to the PyTorch environment's Python executable.
`--resume RUN_DIRECTORY` reuses completed runs. The full audit and original
sensitive checkpoint must exist in this workspace. The preset also recognizes
`models/cells-sensitive`; it never substitutes the balanced `models/cells` model.
