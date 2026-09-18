# FISH Viewer

A Python desktop viewer with classical and neural 3D cell contour pipelines for Imaris microscopy stacks. Uses **PySide6 + PyVista/VTK** on Linux and Windows. Images and original annotations are opened read-only.

For automatic contours, open **Contour pipeline…** in the viewer toolbar. The reference folder is selected by default when present. The panel provides automatic channel selection, 3D shape filtering, parameter tuning, live graphs, and prediction overlays. See the [pipeline guide](docs/contour-pipeline.md) for instructions, score definitions, and measured limitations.

For the neural experiment, open **Neural training…**. The new **StarDist 3D** workflow trains on sparse manual contours, separates acquisition groups for training/validation/testing, saves checkpoints, and displays live losses and contour scores. See the [neural training guide](docs/neural-training.md). Optional dependencies are already installed on this machine.

See [saved models and their locations](docs/models.md) to choose between the full-collection, conservative, balanced, and earlier pilot models. **Choose model…** restores the selected model's own thresholds. The [balanced workflow](docs/balanced-training.md) allows more elongation while retaining membrane evidence and synthetic negative controls.

Open a scan and click **Auto-select contours…** below its name to choose a model and display its generated contours automatically. This workflow ignores roundness and elongation limits while requiring visible boundary evidence. **Open saved run…** is beside it and also above the processing tabs and in Results. The viewer and processing panels scroll vertically on smaller screens. See [automatic contours and reports](docs/auto-contours.md), including the collection prepared for future training without shape curation.

After loading neural cell contours, click **Find signals / signal training… → Run** to detect signal centers inside those cells. The trained 3D signal CNN provides per-cell/channel counts, detection scores, CSV exports, and 2D/3D markers. It uses A annotations and excludes uncertain channel labels from supervision. See [signal detection](docs/signal-detection.md) for the model, training graphs, and scoped test results: **89.2% precision / 51.4% recall** in eligible annotated regions inside neural cells.

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

Local validation passed **68 tests**, including real StarDist CPU training, checkpoint reload, sparse-label loss gradients, and GUI worker integration. The real Linux desktop checks also cover live GPU training, saved-model application, graphs, and neural 2D/3D overlays. The small-window check covers report access, model selection, application without roundness filters, and automatic overlays (`scripts/smoke_viewer.py --neural-report /path/to/report.json --auto-contours`). Signal checks cover A/B semantics, channel ambiguity, physical crop mapping, cell-restricted matching, CSV counts, and saved-model negative controls. The real signal GUI smoke test verifies inference, counts, and 2D/3D signal overlays. Other tests cover automatic channel selection, physical 3D shape measurements, annotation matching, missing and unknown cells, acquisition splitting, edge exclusion, cancellation, and saved predictions. The earlier full collection audit successfully decoded **1,656 channel volumes from 392 image files** and parsed **396 scenes**, containing **3,519 surfaces**, **3,550 measurement groups**, and **4 unmeshed contour curves**. No decoding or scene parsing errors were found; four referenced image files are absent from the supplied collection. Scene totals include alternative scenes that reference the same image.

Inspect a dataset without launching Qt:

```bash
.venv/bin/python -m fish_viewer --inspect "path/to/file.ims"
```

The code separates data models, read-only image access, scene parsing, views, and GUI controls in `fish_viewer/`. This allows later training-data extraction to use the readers without starting the GUI.

Format reference: [Oxford Instruments' Imaris file format description](https://imaris.oxinst.com/support/imaris-file-format). Rendering references: [PyVista volume rendering](https://docs.pyvista.org/examples/02-plot/volume_rendering.html) and [Qt for Python](https://doc.qt.io/qtforpython-6/).
