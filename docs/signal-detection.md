# Signal detection inside neural cell contours

The first signal network is trained and integrated into the viewer. It finds **signal centers**, reports counts per cell and channel, and saves a classifier score for each detection. Original images and Imaris annotations remain unchanged.

## Use the trained model

1. Restart with `./run-viewer.sh FISH` and open a scan.
2. Use **Auto-select contours… → Balanced collection**, or open existing neural contours from a saved cell run.
3. Click **Find signals / signal training…** below the scan name.
4. Leave **Detect signals in current scan** selected. The completed signal model is selected automatically; **Browse…** can choose another signal model folder.
5. Click **Run**. The panel shows candidate extraction and network scoring progress. When it finishes, signal markers are loaded into the viewer automatically.

Open the panel again for **Results / counts**. It shows XYZ coordinates in µm, the cell ID, channel, count, and detection score. **Open output folder** exposes `report.json`, `signals.csv`, and `cell-counts.csv`. Zero counts are included for every neural cell and every analyzed channel. The GUI and CSV channel numbers are one-based; JSON channel indices are zero-based.

Signals are listed in **Annotations** as “NN signals · cell … · channel …”. Select a group to see scores and use **Focus selected object** to inspect it with surrounding cell context. **Predicted signal centers** controls their visibility independently; the existing measurement-endpoint visibility also applies to point markers. Markers retain their channel colors in 2D and 3D. Loading different cell contours clears the old signal overlay, since cell IDs may have changed.

**Open saved signal run…** opens application results or training graphs. A running CLI training report can be watched. **Train signal detector from audit** starts a new network from the prepared audit, with live loss, validation precision/recall/F0.5 graphs and epoch/batch progress. Training-curve precision/recall use score 0.5; the final selected threshold is separate.

## Model and measured results

- [Model folder](../artifacts/signal-fish/training/signal-train-20260919-020312-268747/model/), including `signal-model.json` and `best.weights.h5`.
- [Training report](../artifacts/signal-fish/training/signal-train-20260919-020312-268747/report.json), [compact summary](../artifacts/signal-fish/training/signal-train-20260919-020312-268747/summary.json), and [per-scan test scores](../artifacts/signal-fish/training/signal-train-20260919-020312-268747/test-by-scan.csv).
- [Training graphs](../artifacts/signal-fish/training/signal-train-20260919-020312-268747/training.png) and [representative review crops](../artifacts/signal-fish/training/signal-train-20260919-020312-268747/review-crops.png).
- [Saved-checkpoint negative controls](../artifacts/signal-fish/controls/controls.json).

Training stopped after **37 epochs**, selecting **epoch 22** by class-balanced validation cross-entropy. The network has 139,761 parameters. Threshold selection used validation acquisitions only, choosing the most recall among tested thresholds reaching at least 90% precision and five detections. The selected score threshold is **0.995**.

| Scored region split | Reference signals | Matched | False detections | Missed | Precision | Recall | Matched-center RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 255 | 147 | 11 | 108 | 93.0% | 57.6% | 0.177 µm |
| Test | 144 | 74 | 9 | 70 | 89.2% | 51.4% | 0.202 µm |

Matching is one-to-one, within **0.6 µm**, and requires the same neural cell and inferred signal channel. Extra detections in the scored regions are false positives. Missed references count as false negatives. Localization error is calculated only for matched centers. The saved reports include per-channel scores and cell/channel count error.

These metrics apply to the intersection of neural cell interiors with fully marked manual cells whose signal channels could be assigned unambiguously. They are **not precision/recall for every cell or the entire scan**. Channel labels are inferred from image evidence, rather than independently reviewed color labels. Unmarked cells remain unscored. In particular, five eligible test references on channel index 2 were not reached by the candidate finder; no recall claim is supported for that small subgroup.

The candidate stage could reach 115/144 eligible test references (79.9%) before neural rejection. Its misses and the NN's additional rejections are both counted in the final 51.4% recall. A simple candidate-contrast threshold, selected on validation with the same precision preference, detected 5/144 test references with three false positives. This is a limited classical baseline, not a comparison against every available spot detector.

All four synthetic negative controls passed with the saved checkpoint: blank, constant intensity, noise, and membrane-only input with a cell mask present. The latter two generated 69 and 96 candidates respectively, all rejected. These controls do not guarantee zero errors on real images. A score near 1 is not a calibrated probability that a detection is correct.

## Data and supervision

The user confirmed that A marks a signal, and that all signals within marked cells are annotated. B and C endpoints are not signal positives. Duplicate A coordinates from identical images/alternative scenes are merged. The four previously excluded outlier scans and their aliases remain excluded. The audit verifies that aliases have identical raw-channel fingerprints and retains the original disjoint acquisition splits.

| Split | Scans audited | Unique A markers | Channel assigned | Usable references inside NN cells | Positive candidate patches | Negative candidate patches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Training | 215 | 1,982 | 1,488 | 487 | 367 | 8,994 |
| Validation | 73 | 699 | 591 | 255 | 218 | 4,596 |
| Test | 71 | 645 | 473 | 144 | 116 | 2,742 |

The [signal audit](../artifacts/signal-fish/audit/audit.json) covers all **359 retained scans**. It uses the saved balanced model's neural contours; manual contours delimit where exhaustive signal annotations justify negative supervision. Manual cell surfaces are never substituted for neural contours as the detection domain.

For each A marker, the audit compares local peak contrast against a surrounding shell in non-membrane channels. Low-evidence or ambiguous channel assignments exclude their affected manual cells from supervision. This prevents uncertain assignments from becoming false negative labels. The records retain all markers, assignments, exclusions, and points that fall outside neural contours. A small physical tolerance associates near-boundary A points with manual annotations, but acceptance inside a neural cell uses the actual discrete cell mask with **no erosion or required boundary gap**.

## Architecture and crop pipeline

1. Read native-resolution channels and normalize each channel independently. The original cell contour archive supplies calibrated masks and the membrane channel.
2. Find permissive difference-of-Gaussians peaks in non-membrane channels, retaining only centers inside a neural cell. Physical filter scales are 0.18/0.50 µm; candidate separation is 0.35 µm. This explicit candidate stage is why blank images cannot produce arbitrary network centers.
3. Extract a **24³ crop at 0.15 µm spacing** around each candidate. The three inputs are its fluorescence channel, membrane fluorescence, and the candidate's neural cell mask. Surrounding image context remains visible to the model.
4. A compact Conv3D classifier rejects unsuitable candidates. Balanced positive/negative training batches use flips, XY rotations, intensity variation, and mild noise. Candidates near uncertain matching distances are excluded from the loss but retained for detection evaluation.
5. Apply the validation-selected score threshold and within-cell/channel suppression. Emit centers and counts, including zero-signal cells.

The CNN uses 16/32/48 convolution filters, pooling, a 64-unit dense layer and dropout; Adam learning rate is 0.0003. Training allowed 100 epochs with patience 15, 80 batches per epoch, and batch size 32. Model weights, thresholds, acquisition splits and reports remain separate from the Imaris files.

Peak finding followed by thresholding is a standard microscopy baseline described by [Big-FISH](https://big-fish.readthedocs.io/en/stable/detection/spots.html). This implementation uses its own difference-of-Gaussians stage and a trained classifier; it does not depend on Big-FISH. A future alternative is a dense point detector such as [Spotiflow](https://github.com/weigertlab/spotiflow), which supports 3D point detection and could address candidate-stage misses. No Spotiflow model was trained in this run.

## CLI reproduction

```bash
# Audit annotations and cache candidate crops. Use a fresh output folder.
.venv/bin/python -m fish_viewer.signals --mode audit \
  --cell-report artifacts/balanced-fish/training/neural-train-20260918-151950-694838/report.json \
  --collection-audit artifacts/balanced-fish/audit/audit.json \
  --output /path/to/signal-audit

# Train and evaluate using the fixed acquisition split.
.venv/bin/python -m fish_viewer.signals --mode train \
  --audit artifacts/signal-fish/audit/audit.json --output signal_runs --device gpu

# Apply inside existing neural contours for a scan.
.venv/bin/python -m fish_viewer.signals --mode apply \
  --image /path/to/scan.ims --cell-predictions /path/to/contours.npz \
  --model artifacts/signal-fish/training/signal-train-20260919-020312-268747/model \
  --output signal_runs

# Check the saved checkpoint on synthetic no-signal inputs.
.venv/bin/python -m fish_viewer.signals --mode controls \
  --model artifacts/signal-fish/training/signal-train-20260919-020312-268747/model \
  --output /path/to/control-results
```

TensorFlow uses the existing optional neural dependencies. The viewer remains independent of TensorFlow; workers run in separate processes. On native Windows, use the environment's `Scripts\python.exe` and CPU/auto device. The first time point is processed. Linux GPU training and the real desktop workflow were tested; native Windows was not tested here.
