# Channel roles and membrane-preprocessed StarDist

In FISH Analysis, select a scan, open **Channels…**, and choose its **Cell channel**
and one or more **Signal channel** checkboxes. Unchecked channels are excluded
from signal detection. Display visibility and brightness remain independent.
Roles are saved per input file; a saved result uses its original input file's
roles. Selected signal channels drive signal counts, cell retention, and export.
The actual cell channel is excluded from signal detection even in automatic mode.

Automatic cell-channel selection now uses Lam/Lamin/LBR wavelengths declared in
filenames when they match channel excitation metadata. Image ring evidence ranks
those candidates. With no usable stain declaration, all channels are considered.
An explicit role always overrides automatic selection. Display colors do not
identify biological stains: green may be Lam488 in one acquisition and FISH488
in another. Check roles before processing ambiguous scans.

The **Preprocessed StarDist (experimental)** preset uses
`models/cells-preprocessed` and the existing signal-center U-Net. It applies its
saved preprocessing automatically. The conservative preset remains the default.
For elongated cells, reduce the minimum-roundness slider or set it to zero;
visible-membrane checks still apply. Do not preprocess inputs manually or feed enhanced images to older checkpoints.

The new fixed recipe operates in physical 3D coordinates: 0.18 µm Gaussian
smoothing, partial subtraction of a broad 6 µm background, percentile scaling,
35% blended CLAHE with 4 µm tiles, and 15% blended grayscale closing over a
0.35 µm neighborhood. It makes weak connected membrane ridges easier to see
without filling whole cells or aggressively joining adjacent cells. These are
fixed parameters, not parameters selected using held-out results.

The network sees enhanced single-channel intensity volumes. Acceptance checks
still use the unenhanced membrane image, so contrast enhancement alone cannot
justify a contour over blank space. Signal detection and physical measurements
use the original channels and calibrated coordinates.

Training uses the original acquisition-group split and exclusions, with all 215
training scans and 1,482 complete manual cells. Unannotated cells are ignored,
not labeled as background; separate synthetic negative volumes are included.
All 359 audited scans receive the same recipe; 73 validation scans select model
weights and acceptance settings, and 71 test scans are held out. Channel choices
are corrected using filenames and image metadata, independently of manual masks.
108 training, 19 validation, and 36 test channel choices changed.

The new checkpoint fine-tunes the conservative network with its 64 StarDist rays
and a reduced learning rate. Its initialization uses exactly the same training,
validation, and test scan sets. A separate fresh 128-ray experiment failed
validation contour checks and was preserved without promotion. StarDist still describes star-convex
objects: enhancement and extra rays cannot recover every deeply concave or
branching surface. For those shapes a crop-based mask network remains a separate
option. Compare complete manual-cell overlaps, mesh volume errors, and physical
surface distances, not just visual smoothness or an image MSE.

See `reports/preprocessed-stardist` for the completed held-out comparison, training
provenance, and example images. The baseline uses the same corrected raw channels
and existing conservative weights, so channel correction is not credited to the
new model alone. Measurements are before signal retention and the optional app
roundness filter. Sparse annotations cannot establish precision of all detections.
