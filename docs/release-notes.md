# FISH Analysis v0.5.0

This release adds the training-image annotator, a boundary-recognition U-Net trained from its annotations, and optional learned guidance for existing 3D contours. The established default remains the center/box detector → surface-loss contour U-Net → signal-center U-Net. Learned guidance, sensitive StarDist hybrid, consensus, membrane snapping and cell recovery remain experimental options.

Linux and Windows source/model ZIPs include all three apps (Analysis, Viewer and Annotator), eleven model folders, launch/setup scripts, evaluation reports and model checksums. Python 3.12 and internet access for initial dependency setup are required. Windows inference uses CPU; Linux also supports GPU setup. Original FISH scans, training caches and user annotation images are not distributed.

## New annotation and boundary tools

- Annotate a random single-image crop from the saved training split using only the audited cell channel. Mark “Likely cell border” or “Likely not cell border” with translucent brushes, eraser, Fill, Clear all and Ctrl+Z undo. Zoom/pan and brightness controls are included.
- Save & next persists original cell-channel pixels, labels and calibrated source coordinates. Skip discards all current marks. Untouched pixels remain unknown and contribute no training loss.
- The new 121,969-parameter boundary U-Net was refitted on all 45 annotated crops from 40 scans / 32 acquisition groups for 35 epochs. A separate development checkpoint used 37 training crops and eight validation crops from disjoint acquisition groups. On labeled validation pixels it achieved **71.7% precision, 81.9% recall, 76.5% F1 and 61.9% IoU** at threshold 0.6. These scores describe broad brush bands, not exact measurement surfaces; they are not scores from the final all-annotation refit.
- Analysis → processing options → **Learned boundary guidance (experimental)** uses those predictions for bounded, smooth 3D contour corrections. Raw membrane evidence, topology, cell cores and non-overlap remain acceptance checks. It creates no new cell instances. Signal-model weights are unchanged.
- The annotator requires the original dataset and a saved training-split audit; use `--audit` and `--fish-root` when relocating them. Launch with `run-annotator.sh` / `run-annotator.bat`.

## Full held-out pipeline comparison

All **144 held-out scans** were tested: 71 test and 73 validation scans. Counts below refer to complete manual cells matched at IoU ≥0.5 before signal filtering. Volume/distance errors are conditional on matched complete cells with accepted signals; all outliers are retained. Different methods have different matched subsets.

| Pipeline | Cells detected before signal filtering | Complete manual cells matched | Volume MAPE | Distance MAE |
|---|---:|---:|---:|---:|
| Current default | 9,826 | 758 | **5.70%** (499 cells) | **0.116 µm** (396 signals) |
| Sensitive StarDist → surface-loss contours → signal U-Net | 18,285 | 869 | 8.97% (539 cells) | 0.136 µm (442 signals) |
| Both-detector consensus | 9,004 | 729 | 7.88% (463 cells) | 0.126 µm (375 signals) |
| Current default + learned boundary guidance | 9,826 | 760 | 5.70% (502 cells) | 0.117 µm (396 signals) |

The sensitive model is now bundled as `models/cells-sensitive`; its proposal masks supply boxes/seeds, and failed contour predictions are rejected. Consensus keeps whole hybrid contours only where both approaches match at IoU ≥0.5; it does not intersect their voxels.

### Learned guidance on the same references

Across the same **496 matched complete cells**, volume MAPE decreased **5.71% → 5.59%**. The paired change was −0.122 percentage points (acquisition-group bootstrap 95% interval −0.225 to −0.013). On the same **393 manual signals**, distance error was effectively unchanged (paired MAE change −0.000057 µm; 95% interval −0.001385 to +0.001144). This is a small volume improvement, not a demonstrated distance improvement or substantial overall gain, so guidance stays off by default.

Test-only results (71 scans): current default **6.59% volume MAPE / 0.112 µm distance MAE**; learned guidance **6.63% / 0.115 µm**. On the same 188 test-cell references, volume MAPE improved **6.58% → 6.39%**. In the guidance comparison, baseline cell detections were reused; guidance, signal inference, filtering and physical mesh measurements were freshly executed. The 144-scan default/hybrid/consensus comparison freshly ran both detector pipelines, with consensus reusing their cell sets and rerunning signals/measurements.

Manual cell annotations are incomplete, so unmatched detections cannot establish cell precision. Validation influenced earlier model selection, and test acquisitions were previously inspected in exploratory comparisons; this is not a new blind cohort. Brush-band recognition metrics must not be interpreted as volume/distance accuracy.

## Exports and verification

Excel exports contain exactly two sheets: **Cell volumes** and **Signal distances**, with manual/NN measurements and signal channels where available. Both signal channels are processed; cells are retained when at least one accepted signal lies inside them in any channel. The pipeline creates nearest-surface B points and A–B distance lines. Imaris image data and the compatible FISH scene structure are preserved.

GitHub Actions tests and loads the bundled checkpoints on Linux and Windows, verifies the packaged copies, and publishes both ZIPs plus SHA256SUMS.txt only after both builds pass. Local verification includes the masked-loss unknown-pixel gradient, annotation fill/undo/save/skip behavior, geometry/Excel checks, and a native GUI run through learned guidance, signals, filtering, measurements, Imaris/Excel export and 2D/3D display. Native Imaris opening is not covered by those checks.

Detailed results and training curves are bundled under `reports/all-heldout-comparison` and `reports/annotation-boundary`; usage is in `docs/training-annotator.md` and `docs/annotation-boundary-model.md`.
