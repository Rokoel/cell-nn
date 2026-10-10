# FISH Analysis v0.7.3

Contour editing now shows only the current slice's border points. Select a
contour and drag its dots directly; the grab area is 24 screen pixels across
at every zoom level, with overlapping targets choosing the nearest dot.
Connected border lines follow the dot during dragging. Editing a 3D slice
makes its intersections explicit mesh vertices so moving one point keeps the
other points on that slice fixed. Ctrl+Z restores the entire edit, and edited
volumes, signal-to-border distances and exported geometry remain synchronized.

All existing models and presets are bundled. Both-detector consensus remains
the default, with the signal-center 3D U-Net. No model weights were changed.

## Updated model metrics

All **144 held-out scans** were evaluated using explicit membrane channels,
including **58 corrected channel assignments**, and the current app's
**0.72 roundness gate**. Unchanged-channel cell runs were reused only after
checking model hashes and input fingerprints; signal inference, filtering and
measurements were rerun on every scan.

| Split | Scans | Matched cells / signals | Volume MAPE | Distance MAE | Distance RMSE |
|---|---:|---:|---:|---:|---:|
| Test | 71 | 299 / 270 | 8.59% | 0.130 µm | 0.178 µm |
| Validation | 73 | 359 / 332 | 5.95% | 0.118 µm | 0.160 µm |
| All held-out | 144 | 658 / 602 | 7.15% | 0.123 µm | 0.168 µm |

Comparisons use matched complete manual cells and matched same-channel signals
with actual manual A–B measurements. All outliers are included. Test volumes
remain 7.86% too small on average, and distances average 0.077 µm too short.
Acquisition-bootstrap 95% intervals: test volume MAPE 5.67–11.80%; distance
MAE 0.109–0.146 µm. Sparse annotations do not establish detection precision;
validation guided model selection and the historical test fold was inspected
in prior experiments. Reports, individual measurement pairs, channel assignments
and the reproducible benchmark are included under
`reports/consensus-correct-channels/` and `scripts/benchmark_consensus_channels.py`.

## Distribution and verification

This release uses the existing Linux/Windows source-and-model ZIP bundles.
Run the included setup script with Python 3.12. The next workflow change adds
standalone Windows EXE and Linux AppImage builds; these are not assets of v0.7.3.
Windows setup installs Microsoft's signed Visual C++ runtime before TensorFlow.

Contour dragging, fixed neighbor positions, live line updates, repeated edits,
mesh closure, undo, edited export/reload, and the real Linux GUI were verified.

## Attribution

StarDist 2D: Schmidt, Weigert, Broaddus and Myers, **Cell Detection with Star-convex
Polygons**, MICCAI 2018, [paper](https://arxiv.org/abs/1806.03535),
[DOI](https://doi.org/10.1007/978-3-030-00934-2_30). The optional fine-tuned 2D
checkpoint retains its upstream BSD-3-Clause license. Model provenance and exact
checkpoint hashes are bundled in `models/origins.json`.
