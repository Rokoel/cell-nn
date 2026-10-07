# FISH Analysis v0.3.0

This release replaces radial cell detection with a tiny 3D center/box detector
and a seeded 3D U-Net that draws a contour within each padded cell crop. It can
represent concave and elongated cells without roundness or star-convex limits.
Contours still require visible membrane evidence; failed crops are rejected
without a guessed outline. The existing signal CNN is unchanged.

## Applications and bundled models

- **FISH Analysis:** batch processing, 2D/3D inspection, cell contours, signals
  separated by channel, removal of cells without signals, closest-border B
  points and 3D A–B distances, Imaris/CSV/Excel export, and reopening saved results.
  Excel includes calibrated cell volumes, signal coordinates, scores, distances
  and per-channel counts. A cell is retained if any searched channel contains at
  least one accepted signal center inside it; no boundary clearance is required.
- **FISH Viewer:** image exploration, original annotation display, channel and
  display controls, screenshots, saved views, classical contour experiments,
  StarDist training/application, and signal training. Its processing controls
  retain the older cell pipeline; use Analysis for the new center/crop models.
- **Training controls:** Analysis has separate center/box and boundary training
  windows with preprocessing, epoch/batch progress and live loss/metric graphs.
  Training requires a collection audit and source scans, which are not bundled.
- **Models:** `models/proposals` (latest center/box detector), `models/refinement`
  (latest crop U-Net), and `models/signals` (existing signal CNN). The previous
  StarDist model in `models/cells` and boundary model in `models/refinement-base`
  are included for comparison. Exact model checksums are in `models/origins.json`.

New installations select the center/crop pair automatically. Existing saved
model choices are retained: select **Models and device → Use bundled center and
crop models** to use the latest pipeline.

## Latest model evaluation

The acquisition-group split contains 215 training, 73 validation and 71 test
scans. All 1,493 complete, closed training cell annotations were used without
roundness selection. Unmarked regions have zero detector loss and are not
negative cell examples. The four previously excluded legacy scans and their
duplicates remain excluded. Checkpoints and cutoffs were frozen using validation
data before this test run. Center confidence is 0.65; crop mask cutoff is 0.5.

**Held-out test: 71 scans, with all measurement outliers included.** Cell matching
requires one-to-one 3D intersection-over-union (IoU) of at least 0.5. Incomplete
manual cells are excluded from the cell-recall denominator.

| Metric | Result | Evaluated objects |
| --- | ---: | --- |
| Annotated cell recall before signal filtering | **61.8%** | **314 / 508** complete manual cells |
| Matched cells retained after signal filtering | **116** | Separate any-signal filter, not detector recall |
| Volume mean absolute percentage error | **6.98%** | 116 retained matched cells |
| Volume median absolute percentage error | **5.13%** | Same 116 cells |
| Volume RMSE | **8.08 µm³** | Same 116 cells |
| Volume error before signal filtering | **9.63% MAPE** | All 314 matched cells |
| Signal-to-border distance MAE | **0.138 µm** | 105 matched signals |
| Signal-to-border distance RMSE | **0.176 µm** | Same 105 signals |
| Distance median absolute percentage error | **21.70%** | Same 105 signals |
| Distance signed bias, prediction minus manual | **−0.086 µm** | Distances tend to be underestimated |

Signals are matched within the same cell/channel and at most 0.6 µm from the
manual A point. The current pipeline matched 105 of 233 eligible manual signals;
eligibility depends on the current detected cells and channel assignment, so
this is not a whole-dataset signal recall or a like-for-like comparison with
the earlier signal model benchmark. The final pipeline produced 722 retained
cells and 756 signals across the 71 scans; unmatched predictions are not
established false positives because human cell annotations are incomplete.

**Validation:** 441/574 complete cells matched (**76.8% recall**), with 7.70%
conditional volume MAPE and mean symmetric surface error **0.134 µm**. Validation
results informed selection and are not an independent test estimate.

For a like-for-like comparison, the following rows use only manual identities
matched by both the previous StarDist pipeline and the new pipeline:

| Shared objects; all outliers included | Previous | Latest |
| --- | ---: | ---: |
| 65 cells: volume MAPE | 9.02% | **5.17%** |
| 65 cells: volume RMSE | 10.29 µm³ | **8.01 µm³** |
| 61 signals: distance MAE | 0.189 µm | **0.144 µm** |
| 61 signals: distance RMSE | 0.243 µm | **0.184 µm** |

The 95% acquisition-group bootstrap interval for the paired volume MAPE
improvement is 1.13–5.39 percentage points. For distance MAE improvement it is
−0.0021 to 0.0715 µm, which includes no improvement. The historical test split
has been reused during development, so these results remain exploratory.
Errors are conditional on matched objects and do not measure missed cells or
prove real-image precision. Seven synthetic empty/noise controls produced zero
accepted contours, but real unmatched predictions still need human review.

Detailed reports, CSV/Excel tables, graphs and model provenance are bundled and
available in [the evaluation report](https://github.com/Rokoel/cell-nn/blob/v0.3.0/reports/center-box-contours/README.md).

## Download and run

Download the ZIP for Linux or Windows and extract it. Install 64-bit Python 3.12,
then run `setup-analysis.bat` (Windows) or `bash setup-analysis.sh` (Linux).
Start `run-analysis.bat` / `bash run-analysis.sh` for the automatic pipeline, or
`run-viewer.bat` / `bash run-viewer.sh` for the Viewer. Both apps and all five
model checkpoints are included; the original FISH dataset is not needed for
inference. Setup requires internet to install Python dependencies. These are
source/model distributions, not standalone executables. Native Windows uses
CPU inference; Linux supports the optional CUDA dependency extra.

GitHub Actions tests Linux and Windows builds, verifies model checksums and CPU
inference from both the checkout and packaged copy, and publishes both bundles
with `SHA256SUMS.txt` only after both builds pass. Local validation passed 119
tests and a real Linux desktop pipeline/export smoke test. Exported `.ims`
images preserve the source image; annotations are in the matching `.imx` scene,
which should be opened with the `.ims` beside it. Native Imaris opening has not
been tested locally.
