# Membrane peak refinement — 2026-10-07

Implemented an optional post-processing step for the existing best **center/box detector + cropped 3D U-Net**. It moves existing surface points toward brighter supported local cell-channel peaks along surface normals, smooths their displacement and reconstructs masks on the unchanged physical grid. It creates no new cells and imposes no roundness or convexity constraint. See [usage and safeguards](../../docs/membrane-snap.md).

**Result: a small reduction in undersizing and volume error; essentially unchanged signal-to-border distances.** This is available as **Models and device → Snap to membrane (experimental)** in FISH Analysis, off by default. The original neural checkpoints and conservative signal-model default are unchanged.

## Validation selection

Used all 73 validation scans and the fixed 441 complete manually matched cells at baseline whole-cell IoU ≥ 0.5. Three predeclared settings were compared with the baseline. No test data selected parameters. Selected **0.30 µm peak search, halfway movement**, with brightness smoothing 0.10 µm and local displacement smoothing 0.25 µm. The requested continuous mesh shift is bounded by 0.15 µm; voxel rasterization limits final spatial resolution.

Macro averages over scans, using fine-grid voxelized manual targets:

| Metric | Baseline | Selected refinement |
|---|---:|---:|
| Mean symmetric boundary error | 0.14537 µm | 0.14454 µm |
| Mean absolute volume percentage error | 8.51% | 8.27% |
| Mean signed volume error | −3.76% | −3.20% |
| Mean IoU | 0.83053 | 0.83094 |

Full-strength movement reduced volume error further, but performed worse than halfway movement on the primary boundary metric. [Calibration details](validation-calibration.json), [frozen configuration](selected-config.json).

## Held-out comparison

Applied the frozen setting to all **71 test scans / 4,513 detected cells**. Refinement took 283 seconds on this Linux machine. Accepted 4,296 changes; rejected 130 for overlapping another cell, 76 for reduced membrane evidence, 9 for changes below voxel resolution and 2 for scan/search-boundary contact. The average accepted volume ratio was 1.0088. These acceptance counts measure algorithm checks, not manual correctness.

The paired measurement comparison keeps original cell identities and includes all outliers. Volumes use actual closed physical triangle meshes, unlike the voxel targets used for validation calibration.

| Cohort / metric | Before | After |
|---|---:|---:|
| 314 complete matched cells: volume MAPE | 9.63% | 9.27% |
| Same 314 cells: median absolute volume error | 6.36% | 6.08% |
| Same 314 cells: volume RMSE | 16.225 µm³ | 15.901 µm³ |
| Same 314 cells: mean symmetric surface error | 0.15944 µm | 0.15801 µm |
| Same 314 cells: mean surface HD95 | 0.46421 µm | 0.46318 µm |
| 230 eligible manual A centers: distance MAE | 0.16004 µm | 0.15933 µm |
| Same 230 centers: distance RMSE | 0.23762 µm | 0.23817 µm |
| 116 original signal-retained matched cells: volume MAPE | 6.98% | 6.62% |
| Same 116 cells: volume RMSE | 8.080 µm³ | 7.804 µm³ |
| 105 original CNN signal centers: distance MAE | 0.13768 µm | 0.13732 µm |
| Same 105 centers: distance RMSE | 0.17614 µm | 0.17592 µm |

All 314 original matches remained above IoU 0.5. No originally eligible manual A center or frozen CNN A center moved outside its cell. The signal network was **not rerun across the test cohort**: the final two distance rows freeze the existing predicted A centers and recompute only the nearest point B. This isolates boundary effects and is not a new signal-detection benchmark.

Of the 314 cells, volume error improved for 237, worsened for 67 and stayed unchanged for 10. The average reduction was **0.354 percentage points**, with an acquisition-group bootstrap 95% interval of **0.224–0.509**. Among the 116 originally retained cells, 95 improved, 17 worsened and 4 stayed unchanged. Only 5 of the 105 frozen CNN distances changed (4 improved, 1 worsened): local movement often leaves the closest boundary point unchanged. [Paired uncertainty](paired-uncertainty.json).

[Detailed comparison](test-comparison/comparison.json), [cell and manual-distance Excel](test-comparison/boundary-comparison.xlsx), [fixed pipeline cohort Excel](test-comparison/fixed-measurements.xlsx), [comparison graph](test-comparison/boundary-comparison.png), [inference summary](inference-summary.json).

## Interpretation and limits

The correction modestly reduces underestimation. It cannot recover a badly misplaced contour whose membrane lies beyond the local search, and local brightness can still attract a contour toward the wrong structure. The safeguards prevent the checked overlaps/large changes; they do not establish biological correctness. Sparse manual cell labels cannot establish whole-field false-positive rates.

The historically reused test split is an exploratory comparison. Manual surfaces and A–B lines have their own drawing uncertainty. The very small boundary/distance improvements should not be interpreted as meaningful improvements in individual measurement precision. No outliers were removed. The off-by-default checkbox lets the user compare original and snapped outputs on their scans.

## Verification and reproduction

**144 tests passed**, including synthetic undersizing, oversizing, blank channels, anisotropic spacing, concavity, neighbor protection, cached-run provenance, Excel diagnostics, controller stage ordering and the small-window layout. A real Qt app run used the bundled cell/boundary/signal models, refined 31 of 33 initial cells and exported 7 signal-retained cells with 7 A–B measurements. The checks verified cell volumes, Excel values, closest-point B geometry, Imaris scene A/B labels, 2D/3D rendering and unchanged source pixels. Linux was tested; native Windows and the Imaris binary were not available. [Verification record](verification.json).

Reproduce calibration with `scripts/calibrate_membrane_snap.py --audit <fine-audit> --report <validation-cell-report> --output <calibration.json>`. Apply the frozen settings with `python -m fish_viewer.refinement.membrane --cells-report <test-cell-report> --audit <fine-audit> --config-file reports/membrane-snap/selected-config.json --output <new-run-root>`. Compare with `scripts/compare_refined_boundaries.py --audit <fine-audit> --baseline-report <test-cell-report> --refined-report <snap-report> --baseline-measurements reports/center-box-contours/test-measurements/report.json --output <comparison-folder>`.

The fine audit is `artifacts/refinement-fish/audit/audit.json`; baseline validation and test prediction paths are recorded in the calibration and inference artifacts. The four previously excluded scans remain excluded. Original FISH scans and earlier prediction runs are preserved.
