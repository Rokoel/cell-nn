# FISH Viewer

[Documentation](README.md) · [Analysis](fish-analysis.md) · [Models](models.md)

Explore original pixels and annotations in 2D/3D, apply saved models, or open
the separate training panels. Analysis provides the focused automatic workflow.

## Open a scan

After setting up the source environment:

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

After installation, `run-viewer.bat` also launches the app on Windows. Create a fresh environment on each platform; a Linux `.venv` cannot be reused on Windows. For a standalone application, use the [release builds](releasing.md).

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
