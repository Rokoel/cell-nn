# Bundled models

[Documentation](README.md) · [Analysis guide](fish-analysis.md) · [Evaluation](evaluation.md)

The current Analysis defaults are **both-detector consensus** for cells and
**signal-center 3D U-Net** for signals. Presets select their required checkpoints
automatically. The full Viewer also supports choosing a saved StarDist model.

All inference checkpoints are under `models/`. They remain usable after removing
historical runs and generated training arrays. Keep each model folder together.

## Cell and boundary models

| Folder | Purpose | Evidence |
|---|---|---|
| [`cells-sensitive`](../models/cells-sensitive/) | Sensitive StarDist 3D; consensus/hybrid input | [Full collection](full-fish-training.md), [consensus](cell-consensus.md) |
| [`proposals`](../models/proposals/) | Cell-center and bounding-box detector | [Center/box report](../reports/center-box-contours/README.md) |
| [`refinement-bias-candidate`](../models/refinement-bias-candidate/) | Surface-loss crop mask model used by applicable presets | [Contour-bias report](../reports/contour-bias/README.md) |
| [`cells`](../models/cells/) | Historical balanced StarDist 3D | [Balanced training](balanced-training.md) |
| [`cells-conservative`](../models/cells-conservative/) | Conservative StarDist 3D | [Conservative training](precision-training.md) |
| [`cells-preprocessed`](../models/cells-preprocessed/) | Fine-tuned StarDist 3D with saved membrane preprocessing | [Preprocessing report](../reports/preprocessed-stardist/README.md) |
| [`cells-stardist2d`](../models/cells-stardist2d/) | Fine-tuned XY StarDist with linked slices | [2D report](../reports/stardist2d/README.md) |
| [`refinement`](../models/refinement/) | Earlier selected crop contour model | [Center/box report](../reports/center-box-contours/README.md) |
| [`refinement-base`](../models/refinement-base/) | Earlier boundary-refinement baseline | [Boundary report](../reports/boundary-refinement/README.md) |
| [`boundary-annotations`](../models/boundary-annotations/) | Sparse painted-boundary guidance model | [Annotation report](../reports/annotation-boundary/README.md) |

These models use the TensorFlow runtime in `.venv/`. The painted-boundary folder
also retains its development checkpoint and training report to distinguish
held-out development scores from the final all-annotation refit.

## Signal models

| Folder | Purpose | Runtime |
|---|---|---|
| [`signals-center-unet`](../models/signals-center-unet/) | Default direct signal-center 3D U-Net | PyTorch in `.venv-spots/` |
| [`signals-spotiflow`](../models/signals-spotiflow/) | Alternative masked Spotiflow model | PyTorch/Spotiflow in `.venv-spots/` |
| [`signals`](../models/signals/) | Original candidate CNN baseline | TensorFlow in `.venv/` |
| [`signals-refreshed`](../models/signals-refreshed/) | Refreshed candidate CNN experiment | TensorFlow in `.venv/` |

See [signal-center evaluation](../reports/signal-centers/README.md) and
[the refreshed-CNN comparison](../reports/signal-refresh/README.md).
The separate signal environment supports inference too; deleting it disables
the current source-app default signal detector.

## Provenance and historical training

[`models/origins.json`](../models/origins.json) retains original source locations
and hashes. Historical source paths are provenance, not live model locations.
Completed training reports, split definitions and settings have been archived
locally in `training_metadata/`; generated predictions and caches were removed.
See [training-data reconstruction](training-data.md).

Saved model thresholds and preprocessing belong to that checkpoint. Scores from
old runs use their recorded acceptance settings and cannot automatically be
attributed to current presets. [Evaluation](evaluation.md) explains the cohorts.
