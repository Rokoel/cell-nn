# FISH Analysis

A separate desktop app for the final processing pipeline. It reuses the viewer's
readers, 2D/3D rendering, balanced cell model, signal model, and Imaris exporter.
There are no training panels or preprocessing tuning tabs in this app.

## Start

On the existing Linux workspace, run `./run-analysis.sh`.
On Windows, use `run-analysis.bat` after setup. You can also run
`python -m fish_viewer.analysis` or the installed `fish-analysis` command.
Optional command-line arguments add scans or folders when the app opens.

For another computer, extract the **FISH-Analysis.zip** bundle. It includes the
application source, native Imaris scene template, and both trained checkpoints.
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

The five steps run automatically: cell contours → signals inside those cells →
remove cells with zero signals → create B points and A–B lines → export. Saved
model thresholds are reused. Roundness and elongation limits remain disabled;
visible membrane evidence is still required. Signals whose centers lie inside
the predicted cell are eligible, including near-border signals. Processing uses
time point 1; the exported image preserves all original time points and channels.

Model folders and CPU/auto/GPU selection are under **Models and device**.
The portable bundle defaults to `models/cells` and `models/signals`. In the main
workspace it uses the completed balanced September 18 cell checkpoint and
September 19 signal checkpoint. No retraining takes place.

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

## Saved results

Each scan gets a unique `analysis-...` directory. Existing outputs are never
overwritten, and original scans remain untouched.

- `analysis-result.json`: completed analysis index, reopened using **Open saved analysis…**.
- `export/*_measured.ims`: byte-identical copy of the original image.
- `export/*_measured.imx`: retained surfaces, A/B markers and distance lines.
- `export/*_measured.signals.csv`: signal/cell IDs, channel, score, A XYZ, B XYZ,
  distance in µm, and image-edge flag. CSV channels are one-based.
- `export/*_measured.cell-counts.csv`: per-cell/channel counts, including zero
  counts for a particular channel in a retained cell.
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

The real Linux desktop smoke test uses both trained models and verifies every
exported B lies on its own cell and has the shortest A–B distance. It also checks
tables, image byte preservation, 2D/3D overlays and line visibility. The test is
`scripts/smoke_analysis.py`; results and screenshots are under `artifacts/analysis-smoke`.
The full suite passed 99 tests. A real scan completed all five steps on the Linux
GPU and again on CPU using the packaged source and model copies; both produced
6 retained cells and 6 signals with identical distances (0.1118–1.4360 µm).
Native Windows execution and opening in Imaris itself have not been tested in
this Linux environment. The exporter retains the FISH Imaris 6.3.1 XML profile.

To rebuild the source/model bundle, run
`.venv/bin/python scripts/package_analysis.py --output artifacts/FISH-Analysis-new`.
