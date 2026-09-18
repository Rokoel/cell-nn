# Saved neural models

All paths below are relative to the project folder, `/home/rokoel/Documents/cell-nn` on this machine. Keep the entire `model` directory: it contains the checkpoint, architecture, preprocessing, and acceptance settings.

| Model | Model folder | Run report |
| --- | --- | --- |
| Full collection, many detections | [model](../artifacts/full-fish/training/neural-train-20260915-033619-354714/model/) | [report](../artifacts/full-fish/training/neural-train-20260915-033619-354714/report.json) |
| Conservative, fewer detections | [model](../artifacts/precision-fish/training/neural-train-20260915-222552-439128/model/) | [report](../artifacts/precision-fish/training/neural-train-20260915-222552-439128/report.json) |
| Balanced, September 18 run | [model](../artifacts/balanced-fish/training/neural-train-20260918-151950-694838/model/) | [report](../artifacts/balanced-fish/training/neural-train-20260918-151950-694838/report.json) |
| Extended small pilot | [model](../artifacts/neural-extended/neural-train-20260915-025102-143388/model/) | [report](../artifacts/neural-extended/neural-train-20260915-025102-143388/report.json) |
| Earlier 30-epoch pilot | [model](../artifacts/neural-final/neural-train-20260915-024738-944658/model/) | [report](../artifacts/neural-final/neural-train-20260915-024738-944658/report.json) |
| Initial completed pilot | [model](../artifacts/neural-pilot/neural-train-20260915-023859-283019/model/) | [report](../artifacts/neural-pilot/neural-train-20260915-023859-283019/report.json) |

The full-collection baseline completed 150 epochs and selected epoch 146. The conservative model completed 120 and selected epoch 119. The balanced model completed 150 and selected epoch 150. The extended pilot completed 83 and selected epoch 73; it used only a small reference-folder split and is not equivalent to full-collection training.

`artifacts/neural-calibrated/` contains a cancelled experiment. The first `neural-pilot` run ending `023600-864863` failed. `artifacts/full-fish/gpu-check/` and `artifacts/neural-gui-training-smoke/` contain short diagnostic models. These are preserved but are not the main models to use. A machine-readable inventory is in [model-index.json](../artifacts/model-index.json).

## Open a model

For the current scan, click **Auto-select contours…** below its name and choose a model. This automatically applies visible-boundary checks without roundness limits, then opens the contours. See [automatic contours and reports](auto-contours.md). For advanced controls and historical settings:

1. Launch `./run-viewer.sh FISH` and open **Neural training…**.
2. Select **Apply a saved StarDist model**.
3. **Choose model…** and select a `model` folder from the table. Its saved preprocessing and acceptance thresholds are restored automatically.
4. Select images and run. An acceptance preset can deliberately override the saved settings afterward.

To inspect training curves and existing predictions, use **Results → Open saved run…** and select the corresponding `report.json`. A running report can also be watched for live progress. The actual selected checkpoint is `best.weights.h5`; `last.weights.h5` is retained separately.

See [conservative training](precision-training.md) and [balanced training](balanced-training.md) for the differences between the latest runs.

## Signal detector

The [signal CNN model](../artifacts/signal-fish/training/signal-train-20260919-020312-268747/model/) and its [training report](../artifacts/signal-fish/training/signal-train-20260919-020312-268747/report.json) are separate from the cell models above. It completed 37 epochs, selected epoch 22, and uses score threshold 0.995. Open it through **Find signals / signal training…**, after loading neural cell contours. See the [signal guide](signal-detection.md) for validation scope, centers/counts, and exports.
