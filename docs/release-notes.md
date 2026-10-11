# FISH Analysis v0.8.2

This release improves scan review, documents the current cell pipelines and fixes
Linux offline startup. All existing checkpoints remain bundled; model weights and
the default both-detector consensus / signal-center 3D U-Net are unchanged.

## Review and processing

- A processing summary shows the selected preset, device and channel roles.
- Scan rows show pending, running, completed, failed or cancelled status, with
  channel warnings and failure details. **Select unfinished scans** prepares retries.
- **Annotations and editing** opens beside the image for contour and point editing.
- The **Results** tab summarizes manual/program cells, signals and distances, with
  direct access to exported files, Excel and edited Imaris scenes.
- Shared worker startup, cancellation and bundled-model discovery were refactored.

## Linux startup

The AppImage automatically detects matching `.runtime.part*` assets beside it,
including when launched from the desktop. Keep the original filenames. Downloading
these files beforehand supports offline first-launch installation; otherwise the
app downloads and verifies them from this release. Completed download chunks are
retained if setup is interrupted. Allow about 11 GB free for first-launch setup
and about 8 GB for the installed runtime.

Linux bundles CUDA support for TensorFlow and PyTorch; a compatible NVIDIA driver
is required for GPU inference. Windows uses CPU inference and includes its required
Microsoft C++ runtime DLLs. No Python installation is required on either platform.

## Documentation and evidence

The README includes real screenshots of completed contour review and measurements,
a workflow diagram for both-detector consensus and fine-tuned StarDist 2D, and a
measurement-error chart generated from the retained test reports. The chart shows
sample sizes and confidence intervals; the two reports use different matched
populations and channel protocols, so it is not a paired accuracy ranking.

Historical logs, runs and generated caches were removed from the source workspace;
original training inputs, bundled checkpoints and evaluation reports were retained.
See the training-data guide for rebuilding caches before retraining.

## Verification

The corrected local Linux AppImage passed normal cold-cache startup with network
access blocked, every default pipeline stage on an original scan, Imaris/Excel
exports and reopening the results. That example retained 20 cells and 21 signals;
these counts are a functionality check, not an accuracy estimate. The source suite
passed 330 tests (7 optional checks skipped).

Release publication requires both Windows and Linux packaged applications to pass
checkpoint loading, CPU runtime checks and the full synthetic consensus pipeline.
Linux runtime installation is tested with adjacent assets, no directory override
and network access blocked. GPU execution was additionally checked locally.
