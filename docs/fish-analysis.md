# FISH Analysis

Run `./run-analysis.sh` on Linux or `run-analysis.bat` on Windows. This focused
app processes cells and signals, retains cells with signals, measures nearest
3D membrane distances, and exports Imaris and Excel files. Training belongs in
the Viewer/Annotator tools; Analysis has no training buttons or model pickers.

## Installation

Use Python 3.12. The launch scripts set up the application environment on first
use; an internet connection is needed to download dependencies. Run
`setup-spots.sh` / `setup-spots.bat` to install the isolated PyTorch runtime for
the signal-center U-Net. Linux supports CPU or a configured GPU environment;
native Windows uses CPU for TensorFlow models. All model checkpoints are bundled.

## Processing

1. Add scans or recursively add a folder. Select one or several scans.
2. Preview each scan and open **Channels…** to choose its cell channel and signal
   channels. Choices are saved per file; unchecked signal channels are ignored.
3. Choose an output folder.
4. Under **Pipeline and settings**, choose a pipeline, minimum roundness and device.
5. Click **Analyze selected**. Scans run sequentially; progress includes scan,
   stage and worker counts. Cancel stops processing while preserving finished runs.

The settings panel scrolls. The image preview supports slices and 3D display,
channel visibility, brightness/contrast and zoom. Saved analysis runs can be
opened again; **Export Excel…** writes the measurement workbook.

## Fixed pipeline presets

All presets use the current signal-center U-Net at `models/signals-center-unet`.
Only the current cell/contour models for each preset are exposed:

| Preset | Cell detection and contours |
| --- | --- |
| Fine-tuned StarDist 2D (default) | `models/cells-stardist2d`, enhanced XY contours linked into 3D cells |
| Conservative StarDist | `models/cells-conservative`, its own surfaces |
| Preprocessed StarDist (experimental) | `models/cells-preprocessed`, saved membrane enhancement + its own surfaces |
| Balanced StarDist (first release cells) | `models/cells`, its own surfaces |
| Center/box + surface-loss contours | `models/proposals` → `models/refinement-bias-candidate` |
| Sensitive StarDist + surface-loss contours | `models/cells-sensitive` → `models/refinement-bias-candidate` |
| Both-detector consensus | Whole hybrid contours with a center/box match at IoU ≥0.5 |
| Signals first + center/box + contours (experimental) | Full-scan signal proposals → signal-supported center/box candidates → `models/refinement-bias-candidate` |

No preset enables membrane snapping, annotation-boundary guidance or recovery.
Older checkpoints remain on disk for reproducibility, but are not model choices
in Analysis. Switching presets clears incompatible processing flags.

**Minimum roundness** is the physical 3D sphericity
`π^(1/3) (6V)^(2/3) / A`, using volume and surface area on the calibrated voxel
mask. The default is **0.72**. Increasing it rejects less spherical cells;
**0 disables it**. This gate applies to finished contours before signal detection,
for every pipeline. It never reshapes a surviving contour into a sphere.
A value around 0.65 permits more elongation. Inspect accepted cells: roundness
alone cannot prove that a border agrees with the image.

## Signals-first experiment

The first stage tiles the scan in bounded 96³ volumes, resampled to the signal
model's training spacing, separately for every channel except the selected cell
channel. Learned peaks at the saved signal threshold gate approximate cell boxes,
including their contour padding. The center/box model supplies cell centers;
signal centers are not treated as cell centers. Unsupported boxes are discarded
**before** running the contour U-Net.

The signal model was trained inside predicted-cell masks. Its scouting pass
uses an all-valid mask because cells do not yet exist; this is an unvalidated
input context and may miss signals or propose background puncta. Scouting points
are candidates only. After contours and roundness checks, signals are predicted
again inside actual cell masks. Only these final detections enter measurements
and exports. Scouting cannot force an unsupported contour or fabricate a cell.

See [the exploratory comparison](../reports/signal-first/README.md). This is not
a newly trained contour model; candidate gating and shape rejection do not move
an accepted contour to a better border. Historical 144-scan errors in
[the previous pipeline comparison](../reports/all-heldout-comparison/README.md)
refer to the previous center/box default, not the new roundness-filtered presets.

## Measurements and exports

Signals are detected separately in all non-cell channels. A cell survives if at
least one accepted signal center lies inside it in any channel; no boundary gap
is required. Volumes come from closed 3D meshes. B is the closest mesh point to
A in physical coordinates, equivalent to first contact of an expanding sphere;
A–B lines and distances are saved.

Excel has exactly two sheets: **Cell volumes** and **Signal distances**, with
manual and predicted measurements and separate signal channels where present.
Manual annotations are used for exported comparisons, never to guide inference.
The input Imaris scan is read-only; compatible image/scene data are copied into
a new output file. Native Imaris opening must still be checked in Imaris.

The first published release, v0.2.2 (and the earlier v0.2.0/v0.2.1 tags), bundled
the balanced StarDist checkpoint now stored in `models/cells` and the original
candidate-based signal CNN in `models/signals`. There was no contour U-Net.
The balanced preset restores those **cell weights**, while keeping today's
signal-center U-Net.

Channel selection and the additional trained checkpoint are described in
[Membrane-preprocessed StarDist](preprocessed-stardist.md).

## StarDist citation

The default is fine-tuned from the official `2D_versatile_fluo` single-channel fluorescence model. Cite Schmidt, Weigert, Broaddus and Myers, **Cell Detection with Star-convex Polygons**, MICCAI 2018, [DOI](https://doi.org/10.1007/978-3-030-00934-2_30), [paper](https://arxiv.org/abs/1806.03535). The pretrained weights’ BSD-3-Clause license is included in the model directory. See [the contour benchmark](../reports/stardist2d/README.md) and the release notes for distance measurements.
