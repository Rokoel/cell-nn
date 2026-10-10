# FISH Analysis

A separate desktop app for the final processing pipeline. It reuses the viewer's
readers, 2D/3D rendering, trained cell and signal models, and Imaris exporter.
The main view stays focused on analysis; optional center/box and boundary-model
training opens in separate windows under **Models and device**.

## Start

On the existing Linux workspace, run `./run-analysis.sh`.
On Windows, use `run-analysis.bat` after setup. You can also run
`python -m fish_viewer.analysis` or the installed `fish-analysis` command.
Optional command-line arguments add scans or folders when the app opens.

For another computer, extract the **FISH-Analysis.zip** bundle. It includes the
application source, native Imaris scene template, the latest center/box detector
and crop U-Net, the default signal-center U-Net and optional older/Spotiflow
signal checkpoints. Setup installs separate TensorFlow and PyTorch environments. The previous StarDist and boundary checkpoints are preserved
for comparison. Both app launchers are
included: use `run-analysis` for the complete automatic workflow, or `run-viewer`
for image exploration and the older classical/StarDist training panels.
The original FISH dataset and training artifacts are not needed.

1. Install 64-bit Python 3.12. On Windows, also install the Microsoft Visual C++
   Redistributable required by [TensorFlow](https://www.tensorflow.org/install/pip#windows-native).
2. On Windows, run `setup-analysis.bat`; on Linux, run `bash setup-analysis.sh`.
   Setup installs the Python dependencies and requires internet access.
3. Start `run-analysis.bat` or `bash run-analysis.sh`.

Native Windows uses CPU inference. Current TensorFlow CUDA acceleration is
supported on Linux; see the [official TensorFlow installation guide](https://www.tensorflow.org/install/pip).
For an NVIDIA Linux machine, install the GPU extra with
`.venv/bin/python -m pip install -e '.[neural-gpu]'` after the normal setup.
The Linux desktop needs Qt's X11/XWayland dependencies and working OpenGL, as does
the existing viewer. Windows and Linux launchers use the same application code.

## Use

1. **Add scans…** or **Add folder…**. Folder loading includes subfolders.
2. Select one scan, or Ctrl/Shift-select several. **Select all** queues the list.
3. Choose the results folder and click **Analyze selected**.
4. Watch the current scan, step, tile/candidate progress, and elapsed time.
5. Inspect **Image**, **Distances**, and **Cell counts**. Selecting a distance row
   locates its signal and cell; double-clicking it opens the image tab.
6. **Open result folder** opens the completed export. Open the `.imx` in Imaris
   with its matching `.ims` beside it.
7. **Export Excel…** saves a workbook from the displayed analysis. A workbook is
   also saved automatically beside every exported scan. Its sheets contain cell
   volumes (µm³) and signal-to-border distances (µm) in exactly two sheets.
   Each list distinguishes **Manual** and **NN** measurements. Original manual
   annotations are read from the source scan and its Imaris scene, when present;
   manual A–B distances retain their original values. Signal channels are listed
   separately, and an ambiguous manual channel remains blank.
   Older saved analyses also support Excel export; their mesh volumes are computed
   when opened.

Under **Models and device**, **Recover missed cells (experimental)** enables
three additional checks for the bundled center/box and crop models:

- Local membrane contrast and continuity across slices can support dim cells
  that fail the original fixed brightness checks. Noise is estimated from the
  unsmoothed image so smoothing cannot turn a noisy region into convincing evidence.
- Rejected candidates are retried with larger crops and small seed shifts.
  Retry recovery requires agreement between image-supported NN contours.
- The selected center U-Net or Spotiflow searches provisional candidate masks
  and a 0.6 µm band around them. Signals scoring at least 0.9 can trigger another
  contouring attempt. A signal must be inside the final accepted contour.

The recovery pass preserves baseline contours, rejects empty regions and
overlapping candidates, and never exports provisional search masks as cells.
Final signal detection runs again on accepted contours before filtering and
distance measurement. All detected signal channels are considered. It adds two
visible progress stages and records candidate evidence and retry diagnostics
in the saved JSON reports; Excel still contains only the two measurement lists.
Recovery is off initially while its precision is evaluated. Unmarked cells
remain unknown, so unmatched predictions cannot by themselves measure precision.

The five steps run automatically: cell contours → signals inside those cells →
remove cells with zero signals → create B points and A–B lines → export. Saved
model thresholds are reused. Roundness and elongation limits remain disabled;
visible membrane evidence is still required. Signals whose centers lie inside
the predicted cell are eligible, including near-border signals. Processing uses
time point 1; the exported image preserves all original time points and channels.

With a StarDist cell model, an optional **Boundaries** model adds 3D U-Net
refinement between cell detection and signal detection. It predicts detailed
voxel masks without a radial shape constraint, and shows per-cell progress.
Leave the boundary field empty to use the original StarDist contours; the new
center/box detector requires a boundary model. See [boundary refinement](boundary-refinement.md) for
training, safeguards and saved acceptance/fallback information.

Under **Models and device**, **Use sensitive StarDist hybrid** selects the original
sensitive StarDist checkpoint, the surface-loss crop U-Net, and the signal-center
U-Net. **Redraw StarDist proposals** preserves the saved detector sensitivity
and uses StarDist masks only to obtain padded boxes and interior seeds. The crop
U-Net draws new masks from membrane pixels, without a roundness filter. Empty,
weakly supported, incomplete and duplicate contours are rejected; rejected
proposals never retain their StarDist masks. Signals, filtering, A–B measurements
and the two-sheet Excel export then run as usual.

This is a selectable experiment; the center/box pipeline remains the default.
The preset finds `models/cells-sensitive` if installed, or the original sensitive
training checkpoint in this workspace. It is disabled if that checkpoint is
absent. It does not substitute the balanced `models/cells` model. You can also
select your own StarDist checkpoint and enable **Redraw StarDist proposals**.
The preset turns off center/box recovery, which requires the center/box detector.
**Use bundled center and crop models** returns to the default detector and turns
off StarDist proposal redrawing. See [hybrid evaluation](stardist-hybrid.md).

**Use both-detector consensus** selects the hybrid models and enables
**Require both cell detectors**. Both cell pipelines run on the scan, using the
same surface-loss contour model. Whole hybrid contours are retained only when
matched one-to-one to center/box contours at full-instance IoU ≥ 0.5. This does
not intersect or shrink the masks. Signal detection and measurements run after
agreement filtering. Two additional progress stages show the second cell
pipeline and the agreement filter. Saved reports record both model runs and
matched IDs/IoUs; Excel retains exactly the two measurement sheets.
The consensus preset is optional and leaves the default unchanged. Selecting
**Use sensitive StarDist hybrid** turns off consensus, and **Use bundled center
and crop models** returns to the original default. See [consensus results](cell-consensus.md).

An optional **Snap to membrane (experimental)** checkbox under **Models and device**
adds local cell-channel brightness refinement before signal detection. It searches
up to 0.30 µm along boundary normals and moves points halfway toward supported
peaks. Unsafe changes keep their original contours. Original and snapped runs are
both saved; volume and A–B measurements use the snapped masks when enabled.
See [membrane refinement](membrane-snap.md) for controls, safeguards and evaluation.

Every non-membrane channel is searched separately, including scans with two signal
channels. A cell survives when **any** of those channels has at least one accepted
signal. All accepted signals across those channels are retained and counted
separately; a cell does not need a signal in every channel.

Model folders and CPU/auto/GPU selection are under **Models and device**.
The StarDist checkpoint is `models/cells`. The default signal model is
`models/signals-center-unet`; **Use bundled signal model** restores it even when
another choice was saved. The previous conservative CNN is preserved at
`models/signals`, and Spotiflow is available at `models/signals-spotiflow`.
The new default found 163/228 eligible manual signals: 71.5% recall and 79.9%
precision. See [direct signal models](signal-centers.md) for the comparison.
The default center-and-box detector goes in **Cells** (`models/proposals`)
and requires the crop U-Net in **Boundaries** (`models/refinement-bias-candidate`). Its failed
crops are rejected without a guessed contour. New installations discover this
pair when bundled; former bundled defaults migrate once, while custom model choices
are retained. The previous contour model remains at `models/refinement`. See
[contour improvements](contour-bias.md) for the latest results and their scope. See
[the two-network cell pipeline](center-box-contours.md) for training labels and
validation. Analyzing scans does not retrain these models.

Switch between **2D layers** and **3D volume**. Scroll to change slices; Ctrl-scroll
zooms the 2D image. **Channels…** contains channel visibility, color, brightness,
contrast and full-detail 3D controls. Cells, points and distance lines can be
toggled independently. These display changes do not affect measurements.

## Distance definition

For every signal center **A**, **B** is the closest point on the triangle mesh of
that signal's **assigned cell**. The distance is `sqrt(sum((B - A)**2))` in µm.
This is the first contact of a sphere expanding around A until it touches the
cell surface. It searches triangle interiors and edges as well as vertices,
uses the calibrated physical XYZ coordinates, and uses the same mesh displayed
and exported by the application. It is a 3D distance, not a slice projection,
center-of-cell radius, or a distance to another nearby cell.

The native FISH scenes contain `bpMeasurementPoints` objects with points named
`A` and `B`, `LineMode=Polygon`, and distance labels enabled. The app exports that
same structure, with one independent A/B pair per signal. Scores and identities
are in the measurement object name; A/B endpoint names remain literal `A`/`B`.
Signal counts count A only, never B. CSV coordinates and distances retain full
numeric precision; displayed values are rounded for readability.

If a contour touches the image edge and the closest B lies on the artificial
mesh closure there, its CSV/table row is flagged **Image-edge contact**. This is
not evidence of a membrane outside the acquired volume. Distances inherit the
accuracy of the predicted cell contour and signal location.

## Cell volumes

The primary volume is the enclosed volume of the closed triangle mesh displayed
and exported for that cell, in µm³. It uses physical XYZ calibration, consistent
triangle orientation, and rejects open or non-manifold geometry. The workbook
also records voxel-count volume as a separate sampling-dependent measurement
for new runs; these two definitions can differ slightly. **Cell counts** shows
the primary mesh volume. Image-edge truncation is flagged where applicable.
Cell volume is not a signal volume or the volume of a fitted sphere.

## Saved results

Each scan gets a unique `analysis-...` directory. Existing outputs are never
overwritten, and original scans remain untouched.

- `analysis-result.json`: completed analysis index, reopened using **Open saved analysis…**.
- `export/*_measured.ims`: byte-identical copy of the original image.
- `export/*_measured.imx`: retained surfaces, A/B markers and distance lines.
- `export/*_measured.signals.csv`: signal/cell IDs, channel, score, A XYZ, B XYZ,
  distance in µm, and image-edge flag. CSV channels are one-based.
- `export/*_measured.cell-counts.csv`: per-cell/channel counts, including zero
  counts for a particular channel in a retained cell, and cell volume in µm³.
- `export/*_measured.xlsx`: numeric cell volumes, A/B positions and distances,
  signal scores, per-channel counts and units/model metadata.
- `export/*_measured.annotations.json`: full numerical annotations and provenance.
- `work/`: intermediate results, measured report, and contour archive.
- Stage logs remain in the run folder for diagnosis.

Move the entire analysis directory to preserve app reopening; the index uses
relative paths. The `.ims`/`.imx` pair alone is sufficient for Imaris, with CSVs
beside it when numerical results are needed. Cancellation preserves completed
stages; it does not label an incomplete analysis as complete. A second click on
**Stop now** terminates the active worker.

## Validation

Analytic tests cover nearest triangle interiors, edges, vertices, zero distances,
physical coordinate transforms, and restriction to the assigned cell. Tests also
check stale/wrong B rejection, CSV counts, exact A/B line export and reload,
empty results, cancellation, and reopening a moved result after the original
image is unavailable. Export revalidates B against the meshes before publishing.

The manual audit compared 21 A/B pairs in two native FISH scenes. Median manual
distance minus computed shortest distance was 0.00160 µm and 0.000554 µm;
median manual B-to-mesh residuals were 0.000249 µm and 0.000275 µm. This checks
the interpretation of the manual method, not neural segmentation accuracy.

The real Linux desktop smoke test runs the chosen cell and signal models and verifies every
exported B lies on its own cell and has the shortest A–B distance. It also checks
tables, image byte preservation, 2D/3D overlays and line visibility. The test is
`scripts/smoke_analysis.py`; results and screenshots are under `artifacts/analysis-smoke`.
The contour experiment suite passed 163 tests. A real scan completed all five steps
using the center/box detector, surface-trained crop U-Net and signal-center U-Net,
with 18 retained cells and 18 signals. All bundled models passed CPU loading and inference.
Native Windows desktop execution and opening in Imaris itself have not been
tested locally. The release workflow separately verifies Windows CPU inference.
The exporter retains the FISH Imaris 6.3.1 XML profile.

To rebuild the source/model bundle, run
`.venv/bin/python scripts/package_analysis.py --output artifacts/FISH-Analysis-new`.

The [direct signal-center models](signal-centers.md) predict learned center heatmaps
without the image candidate finder. The standard setup installs their runtime;
existing installations should rerun setup when upgrading to v0.4.0.

## Annotation-trained boundary guidance

Enable **Learned boundary guidance (experimental)** under processing options to
use `models/boundary-annotations` for bounded 3D contour corrections. The default
cell detector, surface-loss contour model and signal-center model stay selected.
The option is off by default: the complete held-out benchmark found a small
volume improvement on shared cell references and no distance improvement.
See [model usage](annotation-boundary-model.md) and
[complete results](../reports/annotation-boundary/README.md).

Launch the separate single-image annotator with `run-annotator.sh` (Linux) or
`run-annotator.bat` (Windows). It requires your training-split audit and scans;
these are not included in releases. Use `--audit` and `--fish-root` to relocate
them. It supports two brush categories, Fill, eraser and Ctrl+Z; unknown pixels
remain ignored. See [annotation guide](training-annotator.md).
