# FISH Analysis v0.8.1

The Linux AppImage now bundles CUDA libraries for **both TensorFlow cell
contouring and PyTorch signal detection**. Choose auto/CPU/GPU under Pipeline
and settings. Auto uses an available NVIDIA GPU and otherwise runs on CPU.
A compatible host NVIDIA driver is required; Python and a separate CUDA toolkit
are not required. The Windows EXE retains CPU inference and bundled Microsoft
C++ runtime DLLs. CPU packages no longer expose an unsupported GPU option or
restore an incompatible saved GPU selection.

The AppImage uses stronger compression to fit the larger GPU runtime into a
single GitHub release asset. `--check-runtime --device gpu` verifies a TensorFlow
3D convolution and the default signal-center U-Net on CUDA in separate workers.
GitHub Actions builds both platforms automatically and checks installed inference,
model checkpoints, Imaris export and the single-sheet Excel export on CPU.
GPU execution is checked separately on a machine with an NVIDIA GPU.

The models, default both-detector consensus pipeline and metrics are unchanged.

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

Download the EXE on Windows, or make the AppImage executable on Linux.
See `docs/releasing.md` for runtime requirements and the FUSE-free launch option.

Contour dragging, fixed neighbor positions, live line updates, repeated edits,
mesh closure, undo, edited export/reload, and the real Linux GUI were verified.

## Attribution

StarDist 2D: Schmidt, Weigert, Broaddus and Myers, **Cell Detection with Star-convex
Polygons**, MICCAI 2018, [paper](https://arxiv.org/abs/1806.03535),
[DOI](https://doi.org/10.1007/978-3-030-00934-2_30). The optional fine-tuned 2D
checkpoint retains its upstream BSD-3-Clause license. Model provenance and exact
checkpoint hashes are bundled in `models/origins.json`.
