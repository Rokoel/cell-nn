# Cells found by both approaches

In the focused app, open **Models and device** and click **Use both-detector
consensus**. This selects the original sensitive StarDist, surface-loss contour
U-Net and signal-center U-Net, and enables **Require both cell detectors**.

Both the sensitive StarDist hybrid and center/box pipeline produce accepted
surface-loss contours. A one-to-one match with whole-instance IoU ≥ 0.5 is
required before a cell can reach signal detection, measurement or export. The
complete hybrid contour is retained; the two masks are never intersected or
averaged. This retains the shape from the hybrid while using agreement as an
additional detection gate. It does not retrain either detector or guarantee
that they cannot make the same mistake; both use the same downstream contour NN.

Source fingerprints, membrane channel and time must agree. Physical grids are
aligned using saved spacing/origin. Provisional recovery masks cannot establish
agreement. Sparse cell IDs stay stable, and evaluation counts actual cells
rather than the largest ID. Reports record pair IDs, IoUs and both model runs.
Rejected cells do not reappear through a fallback. Signal filtering, distances,
Imaris scenes and the two-sheet Excel export are unchanged.

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

## Validation

The same four held-out validation scans from four acquisition groups, with 40
manual reference cells, were used. Complete baseline and hybrid cell runs were
reused for the benchmark. Consensus, signal detection, filtering, measurement
and exports then ran normally. A separate fresh native GUI run exercised all
eight processing stages, including both detectors, 2D/3D views, A/B geometry and
automatic/button Excel export, and verified unchanged source image bytes.

| Metric | Default center/box | Sensitive hybrid | Consensus |
|---|---:|---:|---:|
| Cells before signal filtering | 216 | 346 | 189 |
| Manual cells matched before filtering | 24/40 | 32/40 | 25/40 |
| Cells retained with signals | 47 | 50 | 44 |
| Manual cells matched after filtering | 13 | 17 | 15 |
| Volume mean absolute percentage error | 9.42% (13) | 13.34% (17) | 11.38% (15) |
| Distance MAE | 0.179 µm (8) | 0.163 µm (9) | 0.163 µm (9) |

The sample is unchanged; each method's matched subset differs. On the same 12
retained manual cells, volume error is 9.87% for the default and 6.40% for both
hybrid and consensus. On the same eight manual signals, distance MAE is 0.179 µm
for the default and 0.181 µm for both others. Consensus changes which cells are
kept, not their hybrid contour geometry.

| Scan | Default manual matches before/after | Hybrid before/after | Consensus before/after |
|---|---:|---:|---:|
| 121113_HAG4_Lam633_22A488a.ims | 8/8 | 9/8 | 8/8 |
| dsLacZ3.ims | 4/2 | 9/4 | 5/3 |
| c.ims | 6/2 | 8/5 | 6/4 |
| 080212_HAG50-2_Lam633m_K9546a.ims | 6/1 | 6/0 | 6/0 |

Agreement reduces detections and removes some inaccurate newly recovered cells,
but a consensus manual match can still have a large volume error (50.44% in
this sample). Incomplete annotations prevent a precision estimate: the 29
unmatched retained consensus cells are unscored, not established false positives.
The subset is small, especially for distances. No test-set threshold tuning was
performed. The current default remains unchanged.

[Summary](../reports/cell-consensus/summary.json). Full runs:
`artifacts/cell-consensus-validation-20261009-212626`.
Reproduce using `scripts/benchmark_cell_consensus.py`, with `FISH_SPOTS_PYTHON`
set to the PyTorch environment's Python executable. `--resume` reuses completed
runs; `--reuse-hybrid` accepts a completed hybrid benchmark for the same scans.
The original sensitive weights and training audit must exist in the workspace.
Imaris itself and Windows were not run in this environment.
