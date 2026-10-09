# FISH Analysis v0.4.1

The default pipeline now combines:

- **Cells:** `models/proposals` — 3D center/box detector.
- **Boundaries:** `models/refinement-bias-candidate` — surface-trained crop 3D U-Net,
  selected epoch 5, cutoff 0.55, 2 µm padding and 64³ input.
- **Signals:** `models/signals-center-unet` — direct signal-center heatmap U-Net.

The previous contour model remains at `models/refinement`. Former bundled model
selections migrate once; custom choices are preserved. Experimental membrane
snapping remains off by default. Linux and Windows bundles include both desktop
apps, all nine model folders, reports/graphs, setup/launch scripts and model
checksums. Python 3.12 and internet access for initial setup are required.

## Latest contour and distance metrics

All 71 test scans were processed after freezing the validation model and cutoff.
The following paired comparisons include all outliers:

| Measure | Previous contour model | New contour model |
| --- | ---: | ---: |
| Volume MAPE, 314 fixed matched cells | 9.63% | **7.98%** |
| Median absolute volume percentage error, 314 cells | 6.36% | **5.50%** |
| Mean signed volume error, 314 cells | −5.78% | **−4.26%** |
| Mean symmetric surface disagreement | 0.159 µm | **0.146 µm** |
| Volume MAPE, 150 fixed signal-containing matched cells | 7.88% | **5.71%** |
| Distance MAE, 140 fixed U-Net signal centers | 0.129 µm | **0.112 µm** |
| Distance RMSE, same 140 centers | 0.177 µm | **0.153 µm** |
| Median absolute relative distance error | 22.29% | **19.08%** |

The acquisition-group bootstrap 95% interval for volume MAPE improvement is
**1.08–2.25 percentage points**; for distance MAE improvement it is
**0.006–0.033 µm**. Undersizing is reduced, not eliminated.

All 314 original manual matches remain at IoU ≥0.5. The paired experiment keeps
all 4,513 baseline cell identities, applies 4,384 guarded border edits, and retains
129 original masks when an edit fails. Fresh app inference uses its normal
rejection/neighbor rules and does not automatically fall back to the old model.
These are conditional fixed-cell border comparisons, **not a complete fresh-run
detection benchmark**. Signal centers are frozen, and one center outside its new
mask remains scored and flagged. The live app keeps signals only when their
centers lie inside a predicted cell. No roundness constraint is imposed.

Training visited all 1,493 complete marked training cells per epoch, with actual
predicted crops for matched cells. Surface loss and production-like training
crops were changed together; the gain cannot be attributed to surface loss alone.
Asymmetric-loss and residual-correction trials failed to improve their parent
and are not defaults. Model selection used validation data only; the historical
test split has been reused in earlier experiments, so results remain exploratory.

## Signal detection metrics

The signal model and detector checkpoint are unchanged. The earlier scored-region
comparison found **163/228 manual signals**, **79.9% precision**, **71.5% recall**,
and **0.147 µm matched-center RMSE**. Those detection figures were obtained with
the previous contours and have not been remeasured for the new combination.
Unknown/unmarked regions are excluded from signal scoring and supervision.

## Analysis, exports and verification

The workflow remains contours → signals → remove empty cells → nearest surface
B points/A–B lines → Imaris, CSV and Excel. Excel includes volumes, per-channel
counts, signal scores, coordinates and distances. Both signal channels are
processed separately; a cell is retained if it contains at least one accepted
signal in any channel. Imaris exports preserve source image data and the FISH
Imaris 6.3.1 scene profile.

Local release verification passed **164 tests** (6 optional-runtime skips),
all nine bundled checkpoints on CPU, and a fresh GUI run with the new defaults.
The new candidate passed seven synthetic empty/background controls, including
forced proposals. Local CPU GUI checks passed 2D/3D display, real signal detection,
shortest-border geometry, Excel and Imaris exports, and unchanged source pixels.
Native Imaris opening was not available locally. GitHub Actions tests readers,
geometry, losses, default migration and all checkpoints on Linux and Windows
before publishing the source/model bundles and SHA256SUMS.txt.

Detailed comparisons, training curves, distributions, overlays, protocols and
verification records are bundled under `reports/contour-bias`; earlier signal
metrics remain under `reports/signal-centers`.
