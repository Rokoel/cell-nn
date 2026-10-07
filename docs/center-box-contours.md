# Cell centers → padded boxes → crop contours

The alternative pipeline replaces the StarDist detector. A tiny 3D network
predicts center scores and six distances to axis-aligned box faces. Each box
receives extra physical context. A separate seeded 3D U-Net sees that small
image and predicts the chosen cell's mask. The existing signal network, any-
channel signal retention, A/B distances and calibrated Imaris/Excel export are
reused. There is no roundness or star-convex constraint.

The detector uses 3D context so each center has one cell identity across slices.
The implementation is inspired by [Objects as Points](https://arxiv.org/abs/1904.07850);
it is a small custom 3D architecture, not the published CenterNet checkpoint.

## Sparse annotations

Every complete, closed marked training cell supplies one crop per epoch. The
detector's center loss uses voxels inside complete marked cells; the remaining
image is unknown and has zero loss. Box loss is restricted to marked center
neighborhoods. Other marked centers in a crop receive their own targets.
Unmarked cells never become negative cell examples. Test cells are not training
crops and cannot select checkpoints or thresholds. The four excluded scans,
duplicate handling and acquisition split are retained.

For the contour network, a crop is conditioned on a particular target seed.
Its mask means “belongs to this chosen cell,” not “contains any cell.” A neighbor
is not the selected target; this does not label it as a globally nonexistent
cell. No unmarked cell supplies a negative training crop. Generated noise images
provide explicitly synthetic empty controls, including correlated texture.
These are separate from source scans.

Sparse labels do not establish whole-field false-positive rates. Supervising
no-cell decisions would require reviewed empty regions. Center scores and
successful contour checks do not certify that a cell exists.

## Conservative application

The center network supplies an interior point without requiring a bright
signal or spherical cell. The crop U-Net must find a connected mask containing
that point. Its surface must have visible membrane support after mild physical
smoothing and contrast against local context. Incomplete crop/scan-edge masks,
very small fragments and overlapping duplicate proposals are rejected.
Neighbor ownership is resolved without overwriting accepted cells, and
membrane evidence is checked again afterward.

Failed crops produce no contour. There is no box-shaped, spherical or StarDist
fallback. Reasons, center scores and calibrated box extents remain available
in cell worker reports and signal-report provenance.

## App and commands

Under **Models and device**, choose the center detector folder in **Cells** and
the crop U-Net folder in **Boundaries**. **Use bundled center and crop models**
selects both folders together. **Signals** keeps the existing model.
Both cell networks report progress while processing a scan. Choose the original
StarDist folder in **Cells** to use the earlier pipeline. Training buttons open
separate windows with preprocessing, batch/epoch progress and live graphs.

```bash
.venv/bin/python -m fish_viewer.proposals --mode train \
  --audit artifacts/refinement-fish/audit/audit.json \
  --patches artifacts/proposal-fish/patches \
  --output artifacts/proposal-fish/training --device gpu

# Point/box inference alone, without drawing contours.
.venv/bin/python -m fish_viewer.proposals --mode detect \
  --image path/to/scan.ims --model path/to/center-model \
  --output path/to/results --device auto

# Both cell networks; manual contours are not used as inference inputs.
.venv/bin/python -m fish_viewer.proposals --mode apply \
  --image path/to/scan.ims --model path/to/center-model \
  --boundary-model path/to/crop-model --output path/to/results --device auto
```

The worker can reuse cached membrane images with `--audit ... --split validation`
or `--split test`. Manual labels are used only for evaluation after inference.
`scripts/evaluate_cell_centers.py` calibrates center confidence on validation
cells and cannot update a model from test data. `scripts/calibrate_cell_proposals.py`
compares validation contour matches across center cutoffs. Calibration is a
separate step from training a new model.

CPU inference uses the same weights and geometry on Linux and native Windows.
Linux can use the existing TensorFlow GPU setup. Local verification does not
substitute for a native Windows or Imaris runtime test.
