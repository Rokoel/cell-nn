# StarDist 3D training

The viewer now has a **Neural training…** panel for training and applying a real StarDist 3D network. TensorFlow runs in a separate process, so training does not block the viewer or share its OpenGL context. Models and predictions are saved separately from the source Imaris files.

## Start

On this Linux machine the dependencies are installed. Launch `./run-viewer.sh FISH`, open **Neural training…**, and select **Train StarDist 3D + validate + test**. The reference folder is selected initially. Defaults are a maximum of 150 epochs with early stopping, 40 batches per epoch, batch size 1, 32 × 64 × 64 voxel patches, 64 rays, and a compact 3D U-Net with 16 initial feature channels.

- **Setup:** choose files/folders, device, physical cell limits, patch size, learning rate, and stopping settings.
- **Live scores:** training/validation total loss, validation Dice/IoU, detection rate, and matched-cell boundary RMSE. Contour curves use complete reference cells. Full-volume validation is measured initially and every five epochs by default.
- **Preprocessing:** selected image channel, normalized network input, manual training labels, and predicted labels. The log records complete/cut/overlapping annotation counts and the acquisition split.
- **Results:** per-image scores against all references. Open predictions in the same 2D/3D viewer; green surfaces are generated predictions. JSON reports additionally separate complete and cut reference cells and retain scores before shape filtering.
- **Cancel:** requests cancellation at the next checkpoint. A second click on **Stop now** terminates the worker. Completed checkpoints remain. Closing the viewer also stops its neural worker.

Use **Open saved run…** to load `report.json`. Its model folder is selected automatically. Choose **Apply a saved StarDist model** to process additional files using that model's saved preprocessing, threshold, and shape settings. **Load settings…** and **Save settings…** operate on neural settings JSON files. New training starts from random weights; weight checkpoints currently support inference, not continuation of optimizer state through the GUI.

## Completed reference experiment

The trained run is saved in [`artifacts/neural-extended/neural-train-20260915-025102-143388/`](../artifacts/neural-extended/neural-train-20260915-025102-143388/). Open its [`report.json`](../artifacts/neural-extended/neural-train-20260915-025102-143388/report.json) in the panel to inspect the graphs, use its model, or open the existing predictions without retraining.

The run trained on ten complete disjoint cells from two `101212` images. It used seven usable training-style validation objects from `131212`; full-volume validation scores additionally include other saved reference meshes. Training stopped after 83 epochs, selecting epoch 73 by validation loss. The validation threshold sweep selected probability 0.50. The full run took approximately two minutes on the RTX 5060 Ti after CUDA kernels were cached. A preceding 30-epoch pilot had poor detection; the longer fit supports the current default training budget.

On the reserved `141212` scan:

| Metric | Classical contours | StarDist 3D |
| --- | ---: | ---: |
| Complete-cell mean Dice (7 references) | 0.204 | **0.736** |
| Complete cells detected, IoU ≥ 0.5 | 2 / 7 | **7 / 7** |
| All-reference mean Dice (10 references) | 0.227 | **0.720** |
| All references detected, IoU ≥ 0.5 | 3 / 10 | **8 / 10** |
| Complete matched-cell boundary RMSE | 0.522 µm | **0.393 µm** |
| Total predicted contours | 222 | 647 |

The comparison uses the same test image, calibrated grid, reference meshes, diameter/sphericity/elongation filters, and edge policy (edge exclusion off). Classical threshold and smoothing settings come from the preceding tuning run, with no additional tuning. Each method retains its own preprocessing. Missed cells receive zero overlap; boundary RMSE only covers matched cells and therefore has a different denominator between methods. Raw measurements and classical prediction overlays are saved in [`comparison.json`](../artifacts/neural-extended/neural-train-20260915-025102-143388/comparison.json).

**This does not establish precision:** 637 neural predictions have no matched manual reference and remain unscored. They may include both valid unannotated cells and false detections. Seven complete test cells are too few for a strong generalization claim, and this acquisition was already inspected during earlier development. The next useful annotation work is to review representative predicted cells and add explicitly reviewed background regions, then evaluate on a fresh acquisition.

![Training and validation curves](../artifacts/neural-extended/neural-train-20260915-025102-143388/training.png)

![Comparison using common physical filters](../artifacts/neural-extended/neural-train-20260915-025102-143388/comparison.png)

## Supervision and preprocessing

Time point 1 is read at full source resolution. The existing ring-evidence channel selector chooses the membrane channel automatically, independently of manual annotations. The selected pixels receive percentile normalization and calibrated resampling to approximately 0.30 µm. Neural preprocessing does **not** use the classical pipeline's background subtraction. The physical scale stays fixed for larger fields; tiled inference controls GPU memory. Raw channels for one source file and the selected training patches must fit in RAM.

Saved manual meshes become integer 3D instance labels:

| Value | Meaning |
| --- | --- |
| Positive integer | One complete, non-overlapping manually annotated cell. |
| `-1` | Unknown, ignored by the training losses. This includes unannotated areas, padding, cut cells, and discarded overlapping annotations. |
| `0` | Explicitly reviewed background, if separately supplied. Absence of a contour never implies background. |

Cells touching an outer voxel plane are excluded from training but retained for separate evaluation. Among the remaining complete candidates, intersecting objects are excluded in their entirety. Complete candidates take precedence over excluded cut annotations. The audit reports these exclusions. Overlap checks are conservative: even small voxelized intersections between complete candidates discard the affected objects.

Training crops are centered on complete objects. Each crop retains its full target cell; any neighboring object cut by that crop is ignored. This prevents an empty supervised region after output-grid subsampling. Padding is reflected image context with ignored labels. Augmentation uses XY quarter-turns, axis reflections, intensity variation, and noise. If an annotated object cannot fit inside the chosen patch with context, increase the patch size.

StarDist's probability loss ignores negative labels. The distance background regularizer is explicitly disabled too, preventing it from supervising unknown voxels indirectly. Tests check the corresponding gradients. With no reviewed background, the pilot has **object-only supervision**. That can teach shape and within-cell probability structure, but does not establish reliable rejection of background or precision.

Optional reviewed background can be supplied beside an image as `image.background.npz`, with boolean `background` on the processing ZYX grid, string `fingerprint`, `spacing_zyx`, and `origin_xyz`. These must match that image's exported `labels.json`; overlap with any reference cell is rejected. The file is intended for explicitly reviewed annotations, not an automatic inverse of the manual masks. A background-painting editor is not implemented in this version.

## Validation and predictions

At least three acquisition groups with usable manual cells are required. Groups use the directory and acquisition prefix, as in the classical pipeline. The final sorted group(s) are test data, the preceding group(s) are validation, and the remainder train the network. Exact pixel duplicates are skipped. All patches from a given acquisition stay together.

Validation loss selects `best.weights.h5`; `last.weights.h5` is also saved after complete epochs. Patience controls early stopping. Test data do not enter training, checkpoint selection, or threshold calibration.

After checkpoint selection, a bounded probability-threshold sweep uses **validation acquisitions only** and the existing partial-reference metric. It maximizes complete-reference Dice, preferring the higher threshold on a tie. Stock dense-label threshold optimization is unsuitable here because it would penalize unannotated cells as false positives. `report.json` records every calibration candidate, the initial configuration, and the selected configuration. Epoch contour curves use the initial threshold; final predictions use the calibrated threshold. A diamond marks validation scores for the selected checkpoint and threshold; a star marks the final test scores. Calibration candidates are recorded in the log/report.

Inference proposes local maxima in the network probability map, rejects degenerate radial predictions, then runs StarDist's 3D instance suppression and reconstruction. At most 4,000 local proposals per image enter this stage; `proposal_limit_reached` flags the limit. This bounds the geometric work for weak models that assign medium confidence to large regions. Very crowded or much larger fields may require smaller fields or a higher proposal limit in the implementation. The final volume/axis-ratio/sphericity filters use physical units. Edge exclusion is **off by default** for the neural pipeline so that complete/cut-cell evaluation remains visible.

Dice/IoU and detection penalize missed references with zero. Unmatched predictions remain unscored. Boundary RMSE is conditional on a match. Complete/cut subsets and pre-filter scores are saved in the report. Partial annotations cannot measure precision; do not interpret a high recall or low loss as proof that all candidate contours are correct.

The reference folder offers ten complete disjoint training cells from `101212`, seven validation cells from `131212`, and seven complete test cells from `141212`. Original evaluation meshes also include cut and overlapping cells. This is a small pilot. The test acquisition was inspected during the earlier classical experiments, so its results are exploratory; use a fresh independently annotated acquisition for a final generalization claim.

## Saved files

Each run has a unique directory under `neural_runs/`:

- `report.json`, `settings.json`, `epochs.csv`, and `cells-evaluation.csv`.
- `training.png` and `training.svg`, standalone graphs.
- `model/config.json`, `model/fish-model.json`, `model/thresholds.json`, and the best/last `.weights.h5` files.
- Per-image `image.npy`, signed `labels.npy`, and `labels.json` with audit, calibration, and source fingerprint.
- Per-image `contours.npz` and `contours.cells.csv`, compatible with the viewer's prediction overlays.

Keep the whole model folder together. Reports and model manifests include parameters and package versions. GPU floating-point behavior may prevent bit-identical retraining even with the same random seed. Reports contain absolute paths, which need updating if files are moved. GUI worker diagnostics live in the output folder's `.jobs/<job-id>/worker.log`.

## Installation and CLI

Use Python 3.12. From the project directory:

```bash
# Linux, NVIDIA GPU
.venv/bin/python -m pip install -e '.[neural-gpu]'
.venv/bin/python -m fish_viewer.neural --check-device --device gpu

# Train with the tested budget (up to 150 epochs, 40 batches/epoch)
.venv/bin/python -m fish_viewer.neural FISH/3abx3ad_101212 --device gpu

# Apply an existing model
.venv/bin/python -m fish_viewer.neural /path/to/images --mode apply --model /path/to/run/model

# Reproduce the fixed-settings comparison on a run's reserved images
.venv/bin/python scripts/compare_contours.py --neural-report /path/to/run/report.json --classical-settings /path/to/classical/settings.json
```

For CPU execution install `.[neural]` and select `--device cpu`. On native Windows use `.venv\Scripts\python.exe` and `.[neural]`; current TensorFlow GPU training requires Linux or WSL2, while the native desktop viewer can use CPU inference. Windows execution has not been tested here.

The Linux worker automatically adds pip's CUDA library directories to its own loader path. It does not modify system drivers. The RTX 5060 Ti was verified with GPU convolution and gradients under TensorFlow 2.21.0. Its first run may spend time compiling CUDA kernels; subsequent runs use the cache. If GPU is explicitly requested but unavailable, the run fails clearly. Auto mode reports whether it selected CPU or GPU.

References: [StarDist sparse annotation and data requirements](https://github.com/stardist/stardist-docs/blob/main/docs/faq.md), [StarDist 3D implementation](https://github.com/stardist/stardist/blob/main/stardist/models/model3d.py), and [TensorFlow installation and platform support](https://www.tensorflow.org/install/pip).
