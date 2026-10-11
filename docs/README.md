# Documentation

Start with the [Analysis guide](fish-analysis.md) to process scans, or the
[model inventory](models.md) to understand what the bundled checkpoints do.

## Use the applications

| Guide | Covers |
|---|---|
| [FISH Analysis](fish-analysis.md) | Setup, channel roles, pipeline presets, editing and exports |
| [Viewer](viewer.md) | File support, 2D/3D viewing, channels and keyboard controls |
| [Training annotator](training-annotator.md) | Painting training-only boundary crops |
| [Imaris export](imaris-export.md) | File pairing, compatibility and export checks |
| [Standalone releases](releasing.md) | EXE/AppImage setup, runtime downloads and packaging |

## Understand models and results

| Guide | Covers |
|---|---|
| [Model inventory](models.md) | Bundled checkpoint paths, runtimes and report links |
| [Evaluation guide](evaluation.md) | Cohorts, denominators and limits of reported scores |
| [Cell consensus](cell-consensus.md) | Agreement between the two cell detectors |
| [Center/box contours](center-box-contours.md) | Cell proposals followed by crop-based masks |
| [StarDist hybrid](stardist-hybrid.md) | Combining sensitive StarDist with learned masks |
| [Preprocessed StarDist](preprocessed-stardist.md) | Channel-aware membrane preprocessing |
| [StarDist 2D](stardist2d.md) | Slice-based polygons linked into 3D cells |
| [Boundary refinement](boundary-refinement.md) | Seeded 3D masks and boundary supervision |
| [Contour bias](contour-bias.md) | Surface losses, calibration and correction controls |
| [Annotation boundary model](annotation-boundary-model.md) | Learned guidance from painted crops |
| [Membrane snap](membrane-snap.md) | Experimental local intensity refinement |
| [Signal centers](signal-centers.md) | Direct-center U-Net/Spotiflow supervision and comparison |
| [Signal detection](signal-detection.md) | Original candidate CNN and refreshed training |
| [Measurement accuracy](measurement-accuracy.md) | Historical held-out volume/distance results |

## Train and develop

Generated training caches and old predictions are no longer stored. Follow the
[training-data guide](training-data.md) before running historical training commands.
Their paths describe caches to regenerate, rather than datasets already on disk.

| Guide | Covers |
|---|---|
| [Training data](training-data.md) | Retained data, reconstruction metadata and rebuild order |
| [Neural training](neural-training.md) | Sparse labels, acquisition splits and StarDist training |
| [Full collection](full-fish-training.md) | Original baseline dataset and split |
| [Conservative training](precision-training.md) | Exclusions and stricter shape selection |
| [Balanced training](balanced-training.md) | Broader shape selection and historical results |
| [Automatic contours](auto-contours.md) | Viewer inference and visible-boundary acceptance |
| [Classical pipeline](contour-pipeline.md) | Threshold tuning, score definitions and limitations |
| [Architecture](architecture.md) | Modules, workers and runtime separation |
| [Release notes](release-notes.md) | Current bundled release behavior |
