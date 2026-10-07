# Earlier StarDist → boundary-refinement benchmark

This report evaluates the first boundary-refinement experiment, before replacing
StarDist with the center-and-box detector. Original measurement reports remain
unchanged under `reports/measurement-accuracy`.

The U-Net trained on all 1,493 complete closed training cells, without roundness
selection. It used 574 validation cells; test masks were never training crops.
An exploratory run completed 43 epochs and was cancelled after full validation
of a frozen epoch-19 checkpoint. That checkpoint, at mask cutoff 0.2 selected
on all validation detector proposals, is preserved in `models/refinement-base`.
The run's original cancelled status is retained in `training/report.json`.

## Matched test objects, all outliers retained

The frozen original cohort of 79 retained manual-cell matches had mean absolute
volume percentage error **10.15% → 8.81%**. Volume RMSE worsened
**10.68 → 11.89 µm³**: improvement was not uniform, and large errors remain.

Of the original 73 matched CNN signal centers, two lay outside the new contours.
For the same 71 remaining centers, distance MAE was **0.189 → 0.135 µm**.
All 73 rows, including lost eligibility, remain in
`test-boundaries/fixed-signals.csv`. None were removed as outliers.

Before signal retention, all 172 original complete matched cells remained at
IoU≥0.5. Mean sampled symmetric surface error fell **0.239 → 0.154 µm**.
Of 132 originally eligible manual A positions, 14 became outside the refined
cell. For the same 118 remaining positions, distance MAE was
**0.172 → 0.109 µm**. These manual A comparisons isolate boundary placement
from signal-network localization.

With fresh signal inference, the first refined pipeline matched 85 retained
cells and 80 signals: volume percentage error **8.52%**, distance MAE
**0.132 µm**. Those cohorts differ from the original benchmark and are not
strict paired improvements. See `test-full-pipeline` for all rows and intervals.

This is an exploratory reused test split. Sparse manual annotations cannot
establish whole-field precision or certify unmarked detections. Invalid
refinements retain StarDist contours in this older pipeline. The newer
center-and-box pipeline rejects failed crops and is documented separately in
`reports/center-box-contours`.
