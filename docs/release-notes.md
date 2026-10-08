# FISH Analysis v0.4.0

The automatic pipeline now defaults to the best center/box cell detector
(`models/proposals`), cropped contour U-Net (`models/refinement`), and the new
3D signal-center U-Net (`models/signals-center-unet`). Contour checkpoints and
thresholds are unchanged. Experimental membrane snapping remains off by default.
Former built-in defaults migrate once; custom model selections are preserved.

Both Linux and Windows bundles include both desktop apps, all eight neural model
folders, model checksums, reports, and setup/launch scripts. Standard setup now
installs the separate CPU PyTorch environment alongside TensorFlow. Compatible
Linux NVIDIA installations can opt into CUDA PyTorch through `FISH_TORCH_INDEX`.
Previous signal CNNs, StarDist, older contour U-Net and Spotiflow remain selectable.
These are Python source/model bundles; Python 3.12 and internet access for initial
setup are required. GitHub Actions verifies tests and all checkpoints on CPU on
both platforms before publishing archives and SHA256SUMS.txt.

## Latest model metrics

On the fixed 71-scan test split, using the same predicted cells and scored manual
regions (228 eligible manual A points):

| Signal model | Matched signals | Precision | Recall | Matched center RMSE |
| --- | ---: | ---: | ---: | ---: |
| **Default signal-center U-Net** | **163/228** | **79.9%** | **71.5%** | **0.147 µm** |
| Previous conservative CNN | 123/228 | 87.2% | 53.9% | 0.184 µm |
| Optional Spotiflow | 148/228 | 81.8% | 64.9% | 0.170 µm |

The U-Net increases recall, with 41 unmatched scored detections versus 18 for the
previous CNN. Unknown/unmarked regions are excluded from scoring and supervision.
Signal channels are inferred separately; all accepted signals inside predicted
cells count, including centers on boundaries. Cells with no signals are removed.

For the default U-Net's 140 matched signals with valid manual A/B comparisons,
signal-to-border distance MAE is **0.129 µm**, RMSE **0.177 µm**, and median absolute
percentage error **22.3%**. Acquisition-group bootstrap 95% MAE interval is
0.103–0.157 µm. These are border-distance errors, distinct from center localization.

The unchanged contour pipeline matched **314/508** complete manual cells (61.8%).
Its historical signal-retained 116-cell subset had **6.98% mean absolute volume
percentage error**; 65 cells shared with the older contour pipeline had 5.17%.
Across all 314 matched complete cells before signal filtering, volume MAPE is
**9.63%** (median 6.36%). With the new U-Net, the 150 matched cells retained by
signals in the scored regions have **7.88% volume MAPE**. These are different
cohorts: the roughly 6% result does not apply to every cell or to the new retained
subset. All figures include outliers. The test split was reused during experiments,
so results are exploratory rather than independent confirmation.

Reports, plots, per-cell/per-signal comparisons and matching rules are bundled in
`reports/center-box-contours` and `reports/signal-centers`, including
`reports/signal-centers/measurements`.

## Analysis and exports

The existing workflow remains: cell contours → signals → remove empty cells →
nearest surface B points and A–B lines → Imaris, CSV and Excel. Excel includes cell
volumes, per-channel counts, signal scores, coordinates and distances. Imaris
exports preserve source image data and the FISH Imaris 6.3.1 scene structure.
Both 2D and 3D viewing and the experimental membrane-snap controls are included.
