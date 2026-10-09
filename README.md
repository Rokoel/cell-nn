# FISH Viewer

For the final automatic workflow, launch **`./run-analysis.sh`** (Linux) or
**`run-analysis.bat`** (Windows). **FISH Analysis** is a separate, focused app that
detects cells and signals, removes cells without signals, measures A–B distances
to each cell border, and saves Imaris files and CSVs automatically. See the
[FISH Analysis guide](docs/fish-analysis.md) for setup, controls, models and output files.

The default pipeline uses the **3D center/box detector → surface-trained crop
U-Net → 3D signal-center U-Net** (`models/proposals`,
`models/refinement-bias-candidate`, `models/signals-center-unet`). On fixed test
comparisons, contour volume error improved from 9.63% to 7.98% across 314 matched
cells, and from 7.88% to 5.71% for 150 matched cells containing signals. Distance
MAE improved from 0.129 to 0.112 µm for 140 frozen signal centers. The paired
experiment retains original masks on rejected edits; fresh app inference uses
its normal rejection rules. See [reports and graphs](reports/contour-bias/README.md)
for cohorts and limitations. Linux/Windows releases include both apps, the
latest models, and previous checkpoints for comparison.

A Python desktop viewer with classical and neural 3D cell contour pipelines for Imaris microscopy stacks. Uses **PySide6 + PyVista/VTK** on Linux and Windows. Images and original annotations are opened read-only.

For automatic contours, open **Contour pipeline…** in the viewer toolbar. The reference folder is selected by default when present. The panel provides automatic channel selection, 3D shape filtering, parameter tuning, live graphs, and prediction overlays. See the [pipeline guide](docs/contour-pipeline.md) for instructions, score definitions, and measured limitations.

For the neural experiment, open **Neural training…**. The new **StarDist 3D** workflow trains on sparse manual contours, separates acquisition groups for training/validation/testing, saves checkpoints, and displays live losses and contour scores. See the [neural training guide](docs/neural-training.md). Optional dependencies are already installed on this machine.

See [saved models and their locations](docs/models.md) to choose between the full-collection, conservative, balanced, and earlier pilot models. **Choose model…** restores the selected model's own thresholds. The [balanced workflow](docs/balanced-training.md) allows more elongation while retaining membrane evidence and synthetic negative controls.

Open a scan and click **Auto-select contours…** below its name to choose a model and display its generated contours automatically. This workflow ignores roundness and elongation limits while requiring visible boundary evidence. **Open saved run…** is beside it and also above the processing tabs and in Results. The viewer and processing panels scroll vertically on smaller screens. See [automatic contours and reports](docs/auto-contours.md), including the collection prepared for future training without shape curation.

FISH Analysis also offers **Models and device → Snap to membrane (experimental)**: bounded local brightness refinement of the best center/crop model's contours. It preserves the original run and uses the refined masks for volumes and A–B distances. On 314 fixed manual test matches, volume error fell from 9.63% to 9.27%; distance accuracy barely changed. See [usage](docs/membrane-snap.md) and [evaluation](reports/membrane-snap/README.md).

After loading neural cell contours, click **Find signals / signal training… → Run** to detect signal centers inside those cells. A refreshed 3D signal CNN was trained on the latest predicted masks with a finder that searches three spot sizes. On the same test regions it found 132/228 manual signals versus 123/228, but precision fell from 87.2% to 76.3%. The previous checkpoint remains available in `models/signals`; the default is now `models/signals-center-unet`, which found 163/228 references (71.5% recall, 79.9% precision). Both provide per-cell/channel counts, scores, CSV exports and 2D/3D markers. See [signal detection](docs/signal-detection.md) and [the fixed-mask comparison](reports/signal-refresh/README.md).

**Scan workflow / Export Imaris…** runs cell and signal detection, keeps cells containing signals, and exports an unchanged `.ims` image plus an `.imx` scene. Open the `.imx` in Imaris with its matching `.ims` beside it. Exports now use the complete Imaris 6.3.1 structure from FISH; re-export earlier results to apply the compatibility fixes. See [Imaris export and annotation-free test files](docs/imaris-export.md) for validation and current limits.

The [conservative training guide](docs/precision-training.md) describes the stricter workflow: exclude the four legacy outliers and their copies, select spherical training annotations, require membrane evidence, and optionally export only human-confirmed contours. The GUI shows phase/epoch/batch/image progress and can watch command-line training runs.

The [balanced model is trained](artifacts/balanced-fish/training/neural-train-20260918-151950-694838/report.json): 150 epochs, all 359 revised inputs processed. On the same 71 test scans it produces **1,925 contours**, between the conservative model's 581 and the original baseline's 22,009, and detects 172/505 complete reference cells versus the conservative model's 104/505. These are saved-run results with its historical acceptance settings; real-image precision still requires review. Open the report with **Open saved run…**.

The [full FISH baseline guide](docs/full-fish-training.md) covers all 392 existing images and the 1,523/564/505 annotation split. Its completed model reaches mean Dice **0.790** on complete test annotations, but generates many extra contours whose correctness is unknown. This baseline is preserved for comparison.

## Start on this Linux machine

The project virtual environment is already prepared:

```bash
./run-viewer.sh FISH
```

Or open the reference scene directly:

```bash
.venv/bin/python -m fish_viewer "FISH/3abx3ad_101212/141212_3ab3ad_Lam633m_dsRed543_CG488b.imx"
```

Without a path, the app starts with an empty viewer. Use **Open files…** or **Open folder…**. A folder is indexed recursively; double-click a dataset to load it. Image pixels are loaded only for the selected dataset, time point, and resolution. Different saved scenes referencing one image remain separately accessible. Hover over an entry for the image and scene paths.

## Install on another computer

Use Python **3.12** (64-bit recommended). Python 3.11+ is supported, subject to dependency wheel availability. Imaris itself is not required.

**Linux:**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
./run-viewer.sh
```

**Windows, PowerShell:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m fish_viewer "C:\path\to\FISH"
```

After installation, `run-viewer.bat` also launches the app on Windows. Copy the project source files to Windows and create a new virtual environment there; a Linux `.venv` cannot be reused on Windows. This is a source application, not a bundled Windows installer.

3D rendering requires working OpenGL 3.2+ graphics drivers. Linux uses Qt's `xcb` backend, including XWayland when the desktop runs Wayland; VTK's native window widget is not compatible with a native Qt Wayland window. If a minimal Linux installation reports a missing xcb plugin dependency, install the relevant distribution packages (on Ubuntu, commonly `libxcb-cursor0`, `libxkbcommon-x11-0`, `libegl1`, and `libgl1`).

## Viewing

| Control | Behavior |
| --- | --- |
| **2D slices / 3D volume**, or **Tab** | Switch views; the 3D camera is retained when returning to the same dataset. |
| Slice slider, spin box, scroll wheel, arrow keys | Move through the stack. |
| XY / XZ / YZ | Choose the slicing plane. Physical voxel proportions are preserved. |
| Ctrl + scroll / drag in 2D | Zoom / pan. |
| Drag / scroll / Shift + drag in 3D | Rotate / zoom / pan. |
| **Fit view**, or **F** | Frame the complete dataset. |
| **Channels** | Independently change visibility, color, black/white levels, brightness, contrast, gamma, and opacity. |
| **Auto contrast** | Use approximate 1st/99.8th intensity percentiles; reset brightness, contrast, and gamma. |
| **Reset** on a channel | Restore that channel's original file display values. |
| **Annotations** | Toggle all borders, endpoints, or lines; toggle individual saved objects. |
| Select a measurement | Display the total and segment distances in calibrated physical units. |
| **Focus selected object** | Zoom to a cell or measurement; in 2D, move to its center slice. |
| **Cell opacity / Wireframe** | Change cell surface appearance in 3D. |
| **Full volume detail** | Use every voxel at the selected resolution in 3D. Default preview samples only axes larger than 256 voxels. |
| **Resolution / Time point** | Read the corresponding stored image data. Available choices come from the file. |
| **Screenshot…** | Export the current viewport to PNG. |
| **Save view… / Load view…** | Store/restore display settings in a separate `.fish-view.json` file. |

In 2D, cell borders are actual **mesh/plane intersections**, not projections of the whole cell. Measurement lines are clipped to the current slice thickness, and endpoint markers appear only in that slice. The scale bar follows zoom.

In 3D, the saved triangle meshes are real surface objects, initially translucent. Image channels are combined additively into one RGBA volume so they share the same volume rendering. Preview sampling can miss a very small punctum; enable **Full volume detail** to inspect every voxel. 2D always uses all pixels of the selected stored resolution.

Display adjustments never change intensity arrays or source files. Display settings persist in memory while switching datasets during a session. Save them explicitly to reuse them after restarting. Saved view files include channels, visibility, slice, time, resolution, 3D display options, and the 3D camera; PNG export captures the current view.

## File and annotation support

- HDF5 `.ims`: calibrated images, multiple channels, time points, stored resolution levels, and valid image sizes excluding HDF5 block padding.
- Older TIFF-based Imaris `.ims` files, including the supplied Imaris 4.x files in `FISH antiLAD`.
- Imaris 5.0/6.3 `.imx` scenes: triangulated cell surfaces, measurement points, group-local pairwise measurements, A/B polylines, and unmeshed manual contour curves.
- Matching companion scenes are discovered automatically. Old Windows acquisition paths are resolved against sibling files, and an explicit scene reference takes precedence over its filename.
- Empty manual contour editing objects are ignored. Nonempty contours without a saved mesh appear as 3D closed curves and dashed outlines in the intersecting 2D slice; no cell surface is invented between them.
- XML scenes embedded at recognized HDF5 scene paths are also read. Newer proprietary binary `Scene`/`Scene8` object tables are **not supported**; the app reports this instead of silently hiding missing annotations. The supplied HDF5 files use external `.imx` scenes.

The supplied reference files represent marked locations as measurement endpoints. Original A/B endpoints and lines remain visible. The signal pipeline uses the user's confirmed convention **A = signal**, infers its fluorescence channel, and trains on the fully marked cell regions. Classical and neural pipelines generate cell contours; the signal CNN adds centers and counts inside neural contours. Signal-volume segmentation, annotation editing, standalone `.lsm` browsing, and dedicated Imaris Spots/Filaments schemas are not implemented.

Keep each `.imx` with the image it references. Four supplied scenes reference missing `.ims` images; these remain listed with a missing-image indication.

## Verification and developer tools

```bash
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
.venv/bin/python scripts/audit_files.py FISH --pixels --output artifacts/file-audit.json
.venv/bin/python scripts/smoke_viewer.py
.venv/bin/python scripts/smoke_viewer.py --pipeline
```

On Windows, substitute `.venv\Scripts\python.exe`. The unit/GUI tests use offscreen Qt for 2D; integration tests skip when FISH data is absent. The desktop smoke test opens and closes a real window to exercise OpenGL volume rendering, channel visibility, full-resolution sampling, surfaces, selection, and saved settings. It saves screenshots in `artifacts/`. It expects the local reference files.

Add `--pipeline` to the desktop smoke test to run two real GUI tuning trials and check generated 2D/3D contours. The complete collection audit checks every image container and scene, optionally decoding all channels, time points, and resolution levels. `artifacts/file-audit.json` records results and missing-image paths. Native Windows execution has not been tested in this Linux environment.

Local validation passed **99 tests**, including real StarDist CPU training, checkpoint reload, sparse-label loss gradients, and GUI worker integration. The real Linux desktop checks also cover live GPU training, saved-model application, graphs, and neural 2D/3D overlays. The small-window check covers report access, model selection, application without roundness filters, and automatic overlays (`scripts/smoke_viewer.py --neural-report /path/to/report.json --auto-contours`). Signal checks cover A/B semantics, channel ambiguity, physical crop mapping, cell-restricted matching, CSV counts, and saved-model negative controls. The real signal GUI smoke test verifies inference, counts, and 2D/3D signal overlays. Imaris export checks compare directly against native FISH scene structure, validate mesh blocks and timestamps, and verify annotation-free copies preserve image bytes. Other tests cover automatic channel selection, physical 3D shape measurements, annotation matching, missing and unknown cells, acquisition splitting, edge exclusion, cancellation, and saved predictions. The earlier full collection audit successfully decoded **1,656 channel volumes from 392 image files** and parsed **396 scenes**, containing **3,519 surfaces**, **3,550 measurement groups**, and **4 unmeshed contour curves**. No decoding or scene parsing errors were found; four referenced image files are absent from the supplied collection. Scene totals include alternative scenes that reference the same image.

Inspect a dataset without launching Qt:

```bash
.venv/bin/python -m fish_viewer --inspect "path/to/file.ims"
```

The code separates data models, read-only image access, scene parsing, views, and GUI controls in `fish_viewer/`. This allows later training-data extraction to use the readers without starting the GUI.

Format reference: [Oxford Instruments' Imaris file format description](https://imaris.oxinst.com/support/imaris-file-format). Rendering references: [PyVista volume rendering](https://docs.pyvista.org/examples/02-plot/volume_rendering.html) and [Qt for Python](https://doc.qt.io/qtforpython-6/).

Experimental direct signal-center models (small 3D U-Net and masked Spotiflow)
can be selected in either app after installing the separate `setup-spots` runtime.
See [setup, supervision and comparison](docs/signal-centers.md).
