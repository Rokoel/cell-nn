# Direct signal-center detectors

[Documentation](README.md) · [Models](models.md) · [Analysis](fish-analysis.md)

Generated arrays and historical predictions have been removed. Rebuild training
caches before running reproduction commands; [training data](training-data.md)
explains the retained metadata and dependency order.

Two signal detectors predict centers directly from a dense 3D heatmap:

- `models/signals-center-unet`: a small 3D U-Net (199,993 parameters).
- `models/signals-spotiflow`: the upstream Spotiflow 0.6.5 network, configured for three input channels and 3D detection (757,300 parameters), trained with an adapter that masks unknown annotations.

They use the signal image, membrane image, and predicted cell mask. Only local maxima of the **learned heatmap** are proposed; the old image-based candidate finder is not used. Spotiflow also predicts its stereographic flow to refine centers. These are from-scratch FISH models, not the upstream pretrained models or unmodified Spotiflow CLI training. From v0.4.0 the U-Net in `models/signals-center-unet` is the default. The previous conservative CNN in `models/signals` and Spotiflow remain available for comparison.

## Use in either app

The standard `setup-analysis` script installs the separate runtime automatically. To install it alone, run with `./setup-spots.sh` on Linux or `setup-spots.bat` on Windows. Linux defaults to CPU PyTorch; for a compatible NVIDIA GPU use `FISH_TORCH_INDEX=https://download.pytorch.org/whl/cu128 ./setup-spots.sh`. Windows defaults to CPU; a matching CUDA-enabled PyTorch installation can be used in `.venv-spots` for a compatible NVIDIA GPU. The GUI still runs in its existing environment. `FISH_SPOTS_PYTHON` can point to an alternative Python executable containing these dependencies.

In FISH Analysis, expand the model options and select either direct-center folder as the **signal model**. In the full viewer, select that folder in the Signals model field, then apply it to neural cell contours. Both outputs use the existing signal centers, scores, per-cell/per-channel counts, cell filtering, shortest surface distances, point Bs, distance lines, Excel and Imaris exports. Signals on the cell boundary are allowed if their centers belong to the predicted cell. The direct models do not change the cell detector.

The viewer can open each saved training `report.json` in the Signals results panel to display training loss and validation precision/recall. Full training is launched by the scripts below; the legacy GUI training button still launches the candidate classifier.

## Supervision and comparison

The frozen split contains 215 training, 73 validation and 71 test scans. The four previously excluded heterogeneous scans remain excluded. The corpus contains 676 eligible training A points, 362 validation points and 228 test points, inside predicted cells overlapping fully marked, unambiguous manual cells. All 676 training points receive centered samples every epoch. A is a signal center; B is never a signal training target. Channels are inferred from the manual A markers using the existing rules.

At 0.15 µm isotropic sampling, each crop contains the cell bounding box plus 1.2 µm context. Resampling does not increase the acquired axial resolution. A Gaussian target has 0.20 µm standard deviation. Training uses 48³ patches, brightness/noise and flip/rotation augmentation, masked weighted heatmap loss, and an additional masked Spotiflow flow loss. Unmarked cells/regions and neighborhoods of excluded or unassigned A markers have **zero training loss**, not negative labels. Scoring uses the original benchmark ROI, separately from these training exclusions, to avoid hiding false detections.

Both runs use the same data, 60 maximum epochs, batch size 4, seed 42, and early stopping after 15 epochs without improved validation selection. The U-Net uses AdamW at 0.0005 with float16 mixed precision. Spotiflow initially used that recipe, but its flow training produced a non-finite loss during epoch 17. Recovery starts from the finite best epoch 15 checkpoint, uses bfloat16 mixed precision, restarts AdamW at 0.0001, and continues the original epoch budget. The original failed-stage report is retained alongside the recovery report. This is an architecture experiment with a documented numerical recovery, not a perfectly equal-optimizer comparison or an exhaustive assessment of Spotiflow. Checkpoints and cutoffs are chosen on validation to maximize recall subject to 90% precision; if that constraint cannot be reached, the fallback is F0.5. The test fold is evaluated only after freezing these choices. This test fold has been used in earlier experiments, so the results are exploratory rather than an untouched prospective accuracy estimate.

Predictions and manual points are matched one-to-one within the same predicted cell and channel, within 0.6 µm. The cell predictions, manual intersections, inferred channels and matching rules are identical to the existing detector comparison. Unknown regions are not scored. Reported signal localization error applies only to matched signals; it is neither signal-to-border distance error nor accuracy of unmatched detections. Precision/recall and paired recall differences include 2,000 acquisition-group bootstrap samples. File-local channel indices do not identify the same fluorophore across scans.

Synthetic blank, constant (normalized to zero), low-noise, and membrane-only controls test whether the selected threshold produces empty-cell hallucinations. They supplement the real-data comparison and do not establish field precision.

## Reproduce

Run these commands from the repository root after installing the optional runtime. Keep output directories distinct; existing corpora and training runs are not overwritten.

```bash
.venv-spots/bin/python scripts/build_signal_centers.py --help
.venv-spots/bin/python scripts/train_signal_centers.py --index artifacts/signal-centers/data/index.json --engine signal-center-unet3d --output artifacts/signal-centers/unet-training
.venv-spots/bin/python scripts/train_signal_centers.py --index artifacts/signal-centers/data/index.json --engine signal-spotiflow3d --output artifacts/signal-centers/spotiflow-training
.venv-spots/bin/python scripts/evaluate_signal_centers.py --index artifacts/signal-centers/data/index.json --model models/signals-center-unet --split test --baseline-report reports/signal-refresh/test/report.json --output reports/signal-centers/unet-test
```

On Windows substitute `.venv-spots\Scripts\python.exe`. The original FISH data and local corpus are required for retraining and scoring; the distributed model folders suffice for inference.

See the [Spotiflow training documentation](https://weigertlab.org/spotiflow/train.html) and [source](https://github.com/weigertlab/spotiflow) for the upstream architecture. Our masked adapter is in [the training module](../fish_viewer/spots/training.py).

For a fresh Spotiflow run the code now uses bfloat16 on CUDA automatically.
The original recovery checkpoint was a historical run output and was removed.
The selected final checkpoint remains in `models/signals-spotiflow`. The command
below describes recovery after regenerating an original training run:

```bash
.venv-spots/bin/python scripts/train_signal_centers.py --index artifacts/signal-centers/data/index.json --engine signal-spotiflow3d --initial-model artifacts/signal-centers/spotiflow-training/model --learning-rate .0001 --output artifacts/signal-centers/spotiflow-training-recovered
```
