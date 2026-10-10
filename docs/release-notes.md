# FISH Analysis v0.6.0

Fine-tuned StarDist 2D is now the default cell-contour model, followed by the existing signal-center 3D U-Net. The single cell intensity channel is enhanced, XY contours are predicted, and adjacent contours are linked into calibrated 3D cells. The default minimum roundness remains 0.72. Cells without signals are removed; signal centers, point Bs, distance lines, volumes and per-channel distances are exported to Imaris and the two-sheet Excel workbook.

This release also includes explicit cell/signal channel roles, stain-aware automatic channel selection, a preprocessed 3D StarDist option, and the signals-first experimental preset. Existing presets and their models remain available. Analysis has no training buttons. The app and checkpoints are bundled for Linux and Windows.

## Held-out measurements

All 71 held-out test scans were tested with fixed models and thresholds. No test-set tuning was performed.

| Default pipeline, roundness 0.72 and any-signal retention | Result |
|---|---:|
| Matched signals with actual manual A–B distances | 112 |
| Signal-to-border distance MAE | 0.175 µm |
| Signal-to-border distance RMSE | 0.247 µm |
| Median absolute distance error | 0.124 µm |
| Mean signed distance error | −0.040 µm |
| Distance MAE 95% acquisition-bootstrap CI | 0.136–0.211 µm |
| Mean absolute distance percentage error | 40.16% |
| Median absolute distance percentage error | 26.07% |
| Matched retained complete cells for volume comparison | 121 |
| Mean absolute volume percentage error | 5.64% |
| Final retained cells, including unscored cells | 478 |
| Predicted signals, including unscored detections | 524 |

Without roundness or signal retention, fine-tuned 2D finds 1,399 contours and matches **197/505 complete manual cells**, with **7.72% mean absolute volume error** and **0.245 µm mean symmetric surface error**. On the 163 cells shared with preprocessed 3D, volume error is **7.39% versus 11.16%**. Preprocessed 3D matches more cells (265/505), and its mean per-cell HD95 is better (0.692 versus 0.871 µm). These tradeoffs remain relevant despite the requested default change.

Errors are conditional on whole-cell IoU ≥0.5 and, for distances, same-cell/channel signal matches within 0.6 µm. The distance reference is the actual manual A–B line. All outliers are included. Incomplete annotations cannot establish field precision; conditional errors do not measure missed objects. The test fold was inspected in earlier development, so these are exploratory estimates. Older headline means use different matched objects and cannot alone establish a distance improvement.

Detailed tables, graphs, matching policies and acquisition-bootstrap intervals are bundled under `reports/stardist2d/`, including `distances/`.

## Attribution

The default checkpoint was fine-tuned for 40 epochs from the official single-channel fluorescence model **2D_versatile_fluo**, using 1,482 complete manual training cells from 215 scans. Acquisition groups were kept separate across training (215), validation (73) and testing (71). Unmarked cells were ignored, not treated as negative examples. The pretrained archive SHA256 was verified against the official registry.

Please cite **Uwe Schmidt, Martin Weigert, Coleman Broaddus and Gene Myers. Cell Detection with Star-convex Polygons. MICCAI 2018.** [Paper](https://arxiv.org/abs/1806.03535), [DOI: 10.1007/978-3-030-00934-2_30](https://doi.org/10.1007/978-3-030-00934-2_30). [Official models](https://github.com/stardist/stardist-models). The original weights' BSD-3-Clause license is bundled at `models/cells-stardist2d/PRETRAINED-LICENSE.txt`.

## Verification and installation

Local verification includes 220 non-integration regression tests, default-preset migration checks, all bundled checkpoint loads on CPU, synthetic cell/signal controls, and a native GUI run through the full pipeline and Imaris/Excel exports. GitHub Actions repeats CPU checks and verifies the packaged models on Linux and Windows before publishing both bundles and SHA256SUMS.txt. Native Imaris application opening is not tested.

Extract your platform's ZIP and run `setup-analysis.sh` or `setup-analysis.bat`, then `run-analysis.sh` or `run-analysis.bat`. All neural checkpoints are included; no model download or training is required. Linux CUDA remains optional; CPU inference works on both platforms.
