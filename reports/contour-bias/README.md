# Contour undersizing experiment — October 9, 2026

The surface-loss fine-tune improved border placement. The selected model is
`models/refinement-bias-candidate`, now bundled as the default in v0.4.1.
The previous `models/refinement` checkpoint remains selectable.
In Analysis, open **Models and device**, select this folder for **Boundaries**,
and retain `models/proposals` for Cells and `models/signals-center-unet` for Signals.
The candidate also loads in released v0.4.0 without an app upgrade.

## Results on the fixed test cohorts

All 71 test scans were processed after freezing the validation choice. No outliers
were removed. Volumes use closed physical meshes; distances use shortest Euclidean
distances to the same meshes, matching the app's measurement/export geometry.

| Measure | Current default | Candidate |
| --- | ---: | ---: |
| Volume MAPE, all 314 matched cells | 9.63% | **7.98%** |
| Median absolute volume error, 314 cells | 6.36% | **5.50%** |
| Mean signed volume error, 314 cells | −5.78% | **−4.26%** |
| Symmetric mean surface error | 0.159 µm | **0.146 µm** |
| Surface HD95 | 0.464 µm | **0.411 µm** |
| Volume MAPE, same 150 signal-containing matched cells | 7.88% | **5.71%** |
| Distance MAE, same 140 predicted signal centers | 0.129 µm | **0.112 µm** |
| Distance RMSE, same 140 predicted signal centers | 0.177 µm | **0.153 µm** |
| Median relative distance error, same 140 centers | 22.29% | **19.08%** |
| Mean signed distance error, same 140 centers | −0.072 µm | **−0.051 µm** |

The volume improvement is 1.65 percentage points, with an acquisition-group
bootstrap 95% interval of **1.08–2.25 points**. The distance MAE improvement is
0.017 µm, with a corresponding interval of **0.006–0.033 µm** (2,000 bootstrap
resamples, seed 42). These describe uncertainty within this dataset; they do not
establish accuracy on a new collection.

All 314 original manual matches remain at IoU ≥0.5. All 4,513 original predicted
cell identities remain; 4,384 boundary edits passed the image/geometry guards and
129 retained their original masks. These latter counts describe boundary edits,
not additional or removed detections. The paired experiment applies guarded edits to frozen baseline detections and
retains the original mask on a rejected edit. Selecting the candidate in the app
runs fresh inference with its usual rejection/neighbor rules, without an automatic
old-model fallback; the matched-cell figures below therefore describe guarded
border changes, not a complete fresh-run detection benchmark.

The signal network was not rerun for the
paired measurements. One of the 140 frozen predicted centers lies outside its
new mask; it remains in the error statistics and is explicitly flagged. The live
pipeline filters signals by their center's membership in the new cell.

A separate border-only comparison fixes **manual** A centers. On the same 228
centers inside both masks, distance MAE changes from 0.161 to 0.144 µm. Two of the
230 centers originally inside move outside and are reported as failed membership,
not silently treated as outliers. This cohort differs from the 140 predicted
signal-center comparison above.

These are conditional agreement metrics for manually matched cells, not overall
cell-detection precision or recall. Manual borders are a reference, not perfect
ground truth. The historical test split has been reused in previous experiments;
this is exploratory validation, not a new untouched test set. Undersizing is
reduced, but remains on average; individual difficult cells can still be wrong.

## Experiments in the requested order

1. **Calibrate cutoff:** old-model validation cutoff 0.45 modestly improves volume
   MAPE 7.66% → 7.31% and surface error 0.13658 → 0.13593 µm. Lower cutoffs expand
   too far. These are native-grid validation metrics, not the mesh test metrics.
2. **Check crops/resolution:** keep 2 µm padding and 64³ inputs. The original crop
   contains the complete manual bounding box in 438/441 matched validation cells.
   Smaller padding, larger padding or 96³ inference without retraining performs
   worse. This does not rule out separately training at higher resolution.
3. **Surface-loss fine-tune:** best result, epoch 5 of 8. All 1,493 complete marked
   training cells are visited per epoch. Training also uses actual detector boxes
   for matched cells, so improvement cannot be attributed to the loss alone.
4. **Stronger false-negative penalty:** β=0.6 Tversky trial over-expands contours;
   no checkpoint improves its parent. The initial checkpoint is retained.
5. **Learned residual correction:** neither asymmetric nor an additional balanced
   control improves the parent. Both retain their initial checkpoint.

The final surface-trained checkpoint is recalibrated using validation only.
Cutoff **0.55**, 2 µm padding and 64³ inputs are frozen before test evaluation;
0.55 minimizes the registered boundary-first criterion. The new model's
probabilities are differently calibrated, so its cutoff is not interchangeable
with the old model's cutoff. No blanket dilation or roundness constraint is used.

## Graphs and underlying records

- [Test volume, distance and surface distributions](test-distributions.png)
- [Old-model cutoff and crop calibration](calibration.png)
- [Training progress for the selected fine-tune](surface-loss/training-progress.png)
- [Asymmetric-loss progress](asymmetric-loss/training-progress.png)
- [Residual correction progress](residual-correction/training-progress.png)
- [Balanced correction control progress](residual-balanced-loss/training-progress.png)
- [Eight undersized-cell overlays](test-overlays.png): chosen by original volume
  error; gold = manual, red = original, cyan = candidate. Illustrative difficult
  cases, not a representative sample.
- [Detailed mesh comparison and tables](test-comparison/report.json)
- [Fixed signal distances and retained-cell volumes](test-signals/report.json)
- [Combined machine-readable summary](summary.json)
- [Registered protocol and pre-test addenda](protocol.json)
- [Training stages and validation selection](training-stages.json)
- [Final model's cutoff calibration](post-training-thresholds.json)
- [Methods and reproduction](../../docs/contour-bias.md)

## Verification

[Verification records](verification/) include the test log, empty controls,
GPU correction preflight, v0.4.0 compatibility and app smoke output.
The experiment passed 163 tests; v0.4.1 release verification passes **164 tests**,
with 6 optional-runtime skips, including the new default-migration check. The candidate passes all seven
synthetic empty/background controls, including forced proposals that exercise the
contour network directly. Its checkpoint loads and runs with the stock v0.4.0
reader and architecture. The portable bundle passes model checksums and CPU inference for all included
checkpoints, including direct signal controls. The CPU GUI smoke test completes cell contours, U-Net
signals, nearest border points/lines, Excel volumes/distances, Imaris export and
2D/3D display; source pixels remain unchanged. Export structure is checked by the
app, but a native Imaris launch was not available during this experiment.
