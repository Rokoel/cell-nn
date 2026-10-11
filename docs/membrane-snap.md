# Local membrane refinement

[Documentation](README.md) · [Models](models.md) · [Analysis](fish-analysis.md)

FISH Analysis can optionally refine the best bundled center/box + cropped 3D U-Net contours using the existing cell-channel image. Open **Models and device**, enable **Snap to membrane (experimental)**, and analyze a scan. This option is off by default. It does not retrain or replace either neural model.

For each existing cell, the step constructs a closed surface, estimates local outward normals, and searches in both directions for brighter membrane peaks. By default the search extends 0.30 µm and each point moves halfway toward its supported peak: at most 0.15 µm before voxel rasterization. The peak search uses 0.10 µm Gaussian smoothing, local brightness/prominence requirements and a preference for nearby peaks. Displacements are smoothed locally without forcing a rounded or convex shape. Some points can move inward where the original contour was too large.

Flat or weak profiles and peaks at the search limit do not move the contour. The whole-cell change is rejected if it overlaps another detected cell, loses its core or connectivity, changes topology, reaches the scan/search boundary, changes volume by more than the allowed 0.85–1.25 ratio, or reduces membrane evidence. Rejected cells keep their original contours. No new cells are created and no roundness filter is used.

The shifted surface is rasterized onto the original calibrated voxel grid. Display surfaces, signal membership, cell volumes and point-B distances are subsequently computed from this same mask. The final surface has the resolution of the stored voxel grid, so individual shifts below a voxel can be lost or appear as a voxel step. This is an image-based correction, not a guarantee of agreement with manual drawings; a nearby bright structure can still be the wrong membrane.

Each analysis retains the original `cells` (or `refine`) run and a separate `snap` run under `work/`. Its `report.json` and `contours.npz` include per-cell acceptance, rejection reasons, proposed volume change, peak support, shift sizes and before/after membrane evidence. Excel exports identify whether each retained cell was snapped. Reprocessing an already snapped contour run is rejected to prevent accumulated movements.

To refine a saved contour run without running either neural model again:

```bash
.venv/bin/python -m fish_viewer.refinement.membrane \
  --cells-report /path/to/original-cell-run/report.json \
  --output /path/to/new-results --max-shift 0.30 --strength 0.5
```

On Windows use `.venv\Scripts\python.exe`. An optional `--audit` loads the existing fine cell-channel cache after checking source fingerprint, channel and physical grid. Otherwise the original scan is loaded using the saved cell channel. `--config-file` accepts a JSON `MembraneConfig` object. The four previously excluded scans remain excluded from dataset calibration/evaluation through the existing audit.

Validation uses the fixed original complete manual-cell matches, all outliers included. The baseline also participates in parameter selection. Held-out comparison uses the same cell identities and manual A points; a second comparison freezes the original signal-network centers and retained cells. See [the evaluation report](../reports/membrane-snap/README.md) for results and limitations.
