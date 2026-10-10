# FISH Analysis v0.7.1

Both-detector consensus is the default: sensitive StarDist proposals are redrawn
with the surface-loss contour model and retained when the center/box detector
agrees. The signal-center 3D U-Net remains the signal model. Existing presets and
checkpoints are included; no model was retrained for this release.

- Open annotated or unannotated IMS images and IMX scenes. Read embedded XML and recognized HDF5 spot/mesh arrays, preserve imported objects alongside new detections, and replace stale embedded scenes during export.
- Separate Manual and Program annotation groups with visibility controls. Edited imported objects become Program annotations.
- Select/delete cells or signals; add, move and remove contour points or signal centers in 2D. Ctrl+Z restores edits and deletions. Edits and undo histories survive switching scans within the session.
- Recalculate edited cell volumes and nearest-border B points/distances. Preserve original manual A–B measurements until the associated geometry is edited.
- Export the latest annotations as a new IMS/IMX pair. Excel now has one Measurements sheet containing volumes, channels and signal distances.
- Smooth 3D display surfaces without changing measured/exported geometry. Put channel-role settings beside the analysis controls.
- Windows setup automatically downloads Microsoft's signed x64 Visual C++ runtime and installs it before TensorFlow. Run setup-analysis.bat after updating; setup needs internet access and may request administrator approval or a restart. The installer binary itself is obtained directly from Microsoft.

## Existing model metrics

The prior all-held-out comparison processed all 71 test scans and 73 validation
scans. Its consensus test results were **1,029 retained cells**, **174/508 complete
manual cell matches**, **8.95% volume MAPE** (174 matched cells), and **0.129 µm
distance MAE** (137 matched signals). Across all 144 held-out scans, consensus had
**7.88% volume MAPE** (463 matches) and **0.126 µm distance MAE** (375 matches).
These are historical measurements with the benchmark's saved settings, before
the current GUI roundness/channel-role defaults; they are not a new accuracy
estimate for v0.7.1. Different matched subsets and incomplete manual annotations
limit comparisons and do not establish detection precision. All outliers were
included. See the bundled `reports/all-heldout-comparison/` for details.

## Validation and compatibility

235 regression tests passed (7 skipped); model loading and direct signal controls also passed on CPU.
Regression coverage includes imported/embedded annotations, source pixel
preservation, point edits, mesh topology, deletion, undo, provenance and latest
geometry exports. Real FISH annotations were exported/reloaded, and the consensus
pipeline and native Linux 2D/3D interface were exercised. Windows runtime setup is
checked by the Windows release build. Opening the edited files in a licensed
Imaris installation has not been tested here. Unknown proprietary scene layouts
are reported and blocked from export rather than silently discarded.

## Attribution

StarDist 2D: Schmidt, Weigert, Broaddus and Myers, **Cell Detection with Star-convex
Polygons**, MICCAI 2018, [paper](https://arxiv.org/abs/1806.03535),
[DOI](https://doi.org/10.1007/978-3-030-00934-2_30). The optional fine-tuned 2D
checkpoint retains its upstream BSD-3-Clause license. Model provenance and exact
checkpoint hashes are bundled in `models/origins.json`.
