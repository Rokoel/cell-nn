# Center-and-box detector → crop U-Net

Use **Models and device → Use bundled center and crop models** in FISH Analysis.
`models/proposals` contains the new point/box detector; `models/refinement`
contains the crop U-Net. The original StarDist model remains in `models/cells`,
and the earlier boundary model is preserved in `models/refinement-base`.
The signal model in `models/signals` is unchanged.

## Training and selection

The tiny center-and-box CNN has 20,691 parameters. Every epoch visits all 1,493
complete closed training references from 215 scans; no roundness selection is
applied. All 574 complete validation references from 73 scans select checkpoints.
The center run completed 33 epochs, stopping after validation loss stopped
improving. Epoch 26 was selected. Unknown regions and unmarked cells have zero
detector loss; there are no detector background crops.

The crop U-Net starts from the independently trained epoch-19 boundary model.
Five full fine-tuning epochs added box-scale/position variation, seed jitter
and broader synthetic noise controls. Fine-tuning was intentionally cancelled
during the next epoch; its cancelled status is retained in the training report.
The frozen best completed checkpoint is fine-tuning epoch 3. Unmarked cells do
not supply negative cell crops. The contour target is membership in the chosen
seeded cell, so neighbors are not that particular target.

All validation center predictions selected confidence 0.65, preserving ≥95%
of the maximum matched marked-center count. This gave 439/574 matched centers,
mean seed-placement error 0.556 µm, with ≥95% padded-box coverage in 434 cases.
The crop mask threshold 0.5 minimized symmetric boundary error on all 574
validation crops: volume MAPE 7.68%, mean volume bias −0.10%.
Both checkpoints and cutoffs were frozen before new test inference.

Actual detector-box validation matched 441/574 complete manual cells at whole-
cell IoU≥0.5. Conditional volume MAPE was 7.70%; mean symmetric surface error
was 0.134 µm. See `validation-contours` for all matched rows and intervals.

## Held-out measurements: all 71 scans, all outliers included

The final pipeline matched **116 retained complete cells** and **105 signals**.
Volume MAPE was **6.98%**; distance MAE against manual A/B lines was **0.138 µm**,
RMSE **0.176 µm**, median relative error **21.70%**. Its distance bias was
**−0.086 µm**, so individual distances still tend to be underestimated.

These are different matched objects from the original benchmark. The strict
comparison below uses the intersection of manual identities:

| Same matched objects in both pipelines | Original | Center/box/crop |
| --- | ---: | ---: |
| 65 cells: mean absolute volume percentage error | 9.02% | 5.17% |
| 65 cells: volume RMSE | 10.29 µm³ | 8.01 µm³ |
| 61 signals: distance MAE | 0.189 µm | 0.144 µm |
| 61 signals: distance RMSE | 0.243 µm | 0.184 µm |

Fourteen originally matched retained cells and twelve originally matched
signals were not matched by the new final pipeline; 51 cells and 44 signals
were new matches. They are counted rather than removed as outliers. Before
signal retention, the new pipeline matched 314/508 complete test references.
Sparse annotations cannot classify the remaining predicted cells as false.

The shared objects span nine acquisition groups. A paired acquisition bootstrap
(3,000 draws, seed 42) estimated volume MAPE improvement **3.85 percentage points**
(95% interval **1.13 to 5.39**). Distance MAE improvement was **0.0447 µm**
(95% interval **−0.0021 to 0.0715**). The distance interval includes no improvement;
the reduction in its sample mean is encouraging but uncertain across batches.
This is a reused historical test split and remains exploratory.

## Checks and limitations

Seven known synthetic empty/gradient/noise/plane/incomplete-membrane controls
produced **zero accepted contours**. Correlated noise produced 17 rough proposals,
all rejected by the contour stage. This does not measure false positives in
real unmarked FISH regions. Failed crops have no StarDist or geometric fallback.

All 119 tests passed. All bundled checkpoints passed CPU loading/inference.
The actual Qt pipeline completed a real unannotated scan with seven retained
cells/signals, nearest-triangle A/B distances, numeric Excel volumes/distances,
2D/3D display and native-scene round-trip verification. Its exported IMS pixels
and structure were byte-identical to the source. Native Windows and the Imaris
executable were not run locally.

Training loss/epoch graphs, center/crop calibration, a validation overlay,
paired error graph, CSV/Excel measurements, acquisition intervals and source
fingerprints are included here. Original scans and previous reports were not
modified.
