# Training image annotator

Launch `./run-annotator.sh` on Linux or `run-annotator.bat` on Windows, using the existing app environment (`setup-analysis.sh` / `setup-analysis.bat` if needed). No neural models or ML runtimes are needed. An installed package also provides `fish-annotator`.

The app randomly selects a training scan, one native Z slice, and a 128 × 128-pixel crop. Only the cell channel identified by the saved audit is read and displayed, in grayscale. There are no signal channels, projections, or layer controls. It uses `artifacts/refinement-fish/audit/audit.json`: only training cases, with the four previously excluded scans omitted. Validation/test cases and aliases are never sampled; conflicting acquisition groups, fingerprints or paths are rejected. It does not favor manually marked cells.

Choose **Likely cell border** (yellow) or **Likely not cell border** (blue), then paint using the left mouse button. Marks are translucent. Each pixel holds one category; painting replaces earlier marks. Brush diameter is in native pixels, independent of zoom. **Eraser** restores unknown pixels. **Clear all** removes every mark on the image.

Toggle **Fill**, select a category, then click a region. Fill replaces the four-connected region of the clicked annotation label. A closed painted contour stops the fill; a gap allows it to spread outside. It uses annotation boundaries, not raw image brightness. Toggle Fill off to return to the brush. Fill also works with the eraser.

**Ctrl+Z** or **Undo** reverses a complete stroke, fill, erasure or Clear all action. Up to 100 edits are retained for the current image; history resets when moving to another image. Wheel to zoom; right/middle mouse drag to pan; **Fit view** resets zoom. Brightness affects only display; contrast is automatically estimated over the crop (1st–99.5th percentiles).

**Save & next** (Ctrl+Enter) stores the image and marks and immediately loads another. It is enabled only when at least one pixel is marked. **Skip** discards all current marks and loads another, with no annotation written. Painting/filling without a selected category has no effect. Untouched/erased pixels remain **unknown**, not negative examples. Nothing is saved automatically when closing; unsaved marks trigger a discard prompt.

Each saved image has its own directory under `training_annotations/`:

- `annotation.npz`: original cell-channel intensity array `image` (YX), categorical `labels` (YX, uint8).
- `metadata.json`: schema version **2**, categories, original source scan/time/channel, Z/Y/X offset, calibrated spacing and crop origin, training split/group, sampling seed, provenance and pixel counts.

Labels: **0 unknown**, **1 likely not cell border**, **3 likely cell border**. Only nonzero labels should contribute to future training losses. These are user judgments, not confirmed truth. Old schema-1 multi-channel/3D annotations remain untouched and must be interpreted according to their own metadata; they are not silently converted to the new categories.

This app collects annotations; it does not retrain models or create Imaris surface objects. Original scans stay read-only. Saving commits both files together; failed saves keep current marks available for retry.

Optional arguments:

```sh
./run-annotator.sh --crop-xy 192 --seed 42 --output /path/to/annotations
./run-annotator.sh --audit /path/to/audit.json
```

On another machine, use `--fish-root "D:\data\FISH"` (Windows) or `--fish-root /path/to/FISH` (Linux) to relocate the dataset while retaining the audited split. Supply `--audit` too if the audit was moved. Missing scans are omitted; startup fails if none are available. The audit must identify the cell channel; missing channel information produces an error rather than guessing a signal channel. The audited source fingerprint is recorded as provenance, not freshly recomputed on every crop.

Pixels are read directly from the original scan, without padded HDF5 pixels or the resized training cache. Metadata `shape_zyx` has depth 1 and records the original Z index in `offset_zyx`. Coordinate origin is the lower voxel edge; pixel centers are `origin_xyz + (x + 0.5, y + 0.5, 0.5) * spacing_xyz`.
