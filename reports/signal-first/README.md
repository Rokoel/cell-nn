# Signals-first exploratory test

Three test scans were sampled uniformly without replacement from all 71 held-out
test acquisitions, sorted by path, with seed **1010**. None are from the training
split. The signal-first mode freshly ran full-scan scouting, cell boxes, padded
crop contours, roundness 0.72, final masked signal inference and measurements.
No manual annotations guide inference. Original image/scene files were unchanged.
Image export was omitted here; a separate native Analysis smoke run exercised
Imaris/Excel export and both 2D/3D views (15 cells and 15 signals).

| Method | Cells before signal filtering | Manual cells matched | Cells with signals | Volume MAPE | Distance MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous center/box crop default | 119 | 16 | 26 | 9.46% (8 complete cells) | 0.0926 µm (3 signals) |
| Signals first + crop contours + roundness 0.72 | 87 | 16 | 23 | 9.46% (8 complete cells) | 0.0926 µm (3 signals) |

The **32-cell reduction (26.9%) preserved all 16 manual cell matches** in this
sample. Those 32 cells are unscored, not proven false positives. Signal gating
kept 132 of 187 candidate boxes before contouring. The resulting 87 contours all
passed roundness 0.72, so that additional shape gate rejected no cells in this
particular sample. Scout peaks are never exported as final signals.

**Border accuracy did not improve.** The same contour weights are used, and the
matched masks changed only slightly where competing proposals were removed.
This supports testing candidate reduction, not claiming better contour drawing
or validated precision. Errors use the same eight complete manual cells and
three explicit A–B distance references in both modes; all outliers are retained.
The sample is too small to estimate general measurement accuracy.

Scans:

- `12042013_dsLam-K9_Lam488_LBR644_Fits543_4.ims`
- `261012_HAG4withoutCu_Lam633_K9488c.ims`
- `040612_HAG50-2_Lam633m_K9546e.ims`

The previous baseline is reused from the completed 144-scan benchmark. Its cell,
contour and signal checkpoint **and manifest hashes match** the experiment.
No model was retrained. Manual annotations are incomplete and earlier test
scans have been inspected, so this is an exploratory comparison, not a blind
validation. The whole-scan signal model input uses an all-valid mask, differing
from its cell-mask training context. Final detections are rechecked inside cells.

Raw summaries and per-scan/pair CSVs are beside this file. Local pipeline data:
`artifacts/signal-first-test-20261010/report.json`. Reproduce with
`FISH_SPOTS_PYTHON=.venv-spots/bin/python .venv/bin/python scripts/benchmark_signal_first.py --output NEW_DIRECTORY`.
