# Signal refresh experiment

The two requested changes are implemented: rebuild signal crops from the latest center/box + cropped U-Net contours, and search several physical spot sizes while preserving nearby dim peaks. The 139,761-parameter compact 3D CNN architecture is unchanged. No Spotiflow model was introduced.

**The previous signal checkpoint remains the default.** The refreshed model gained test matches but increased false detections. Choose `models/signals-refreshed` explicitly to inspect this experiment; `models/signals` retains the previous checkpoint. Cell-model weights and acceptance settings are unchanged.

## Fixed-mask comparison

All methods use the same scans, latest predicted cell masks, fully marked manual-cell intersections, A markers and inferred unambiguous signal channels. Matching is one-to-one within the same cell/channel and 0.6 µm. Unmarked cells are unscored.

### Validation: 73 scans, 362 eligible references

| Method | Matched | False detections | Missed | Precision | Recall | Center RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original finder + original CNN | 198 | 19 | 164 | 91.2% | 54.7% | 0.173 µm |
| New finder + original CNN | 206 | 28 | 156 | 88.0% | 56.9% | 0.157 µm |
| New finder + refreshed CNN | 204 | 20 | 158 | 91.1% | 56.4% | 0.155 µm |

[Comparison graph](validation/comparison.png) · [per-scan scores](validation/scans.csv)

### Test: 71 scans, 228 eligible references

| Method | Matched | False detections | Missed | Precision | Recall | Center RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original finder + original CNN | 123 | 18 | 105 | 87.2% | 53.9% | 0.184 µm |
| New finder + original CNN | 133 | 36 | 95 | 78.7% | 58.3% | 0.158 µm |
| New finder + refreshed CNN | 132 | 41 | 96 | 76.3% | 57.9% | 0.184 µm |

[Comparison graph](test/comparison.png) · [per-scan scores](test/scans.csv)

The refreshed model's acquisition-group bootstrap 95% intervals are **66.7%–83.2% precision** and **37.2%–70.8% recall**. The paired recall change interval is **-1.46 to 7.67 percentage points** (2,000 acquisition-group resamples).

Localization errors describe each model’s matched subset. These center RMSE figures are not a paired comparison restricted to identical matched signals. The historical test fold has been evaluated before, so this is an exploratory result. Checkpoints, candidate settings and score cutoff were selected on validation; test results prompted the decision to retain the previous default. The September 144-reference benchmark used different cell masks and cannot be compared directly with this 228-reference cohort.

## Data and training

| Split | Scans | A markers | Assigned channel | Eligible in NN cells | Positive crops | Reviewed negative crops |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| training | 215 | 1982 | 1488 | 676 | 542 | 68384 |
| validation | 73 | 699 | 591 | 362 | 313 | 42236 |
| test | 71 | 645 | 473 | 228 | 193 | 25268 |

All 359 retained scans were audited. Acquisition groups remain disjoint, aliases were checked by source stats and decoded pixel fingerprints, and the four legacy outliers and their aliases remain excluded. Source annotations delimit supervision; inference uses predicted cell interiors.

Each manual marker supplies at most one positive candidate. Additional near-marker candidates are unknown. Every positive crop is visited each epoch. Negative candidates require fully marked unambiguous manual regions and at least 0.9 µm clearance from every A marker on the same or an uncertain channel, including markers outside the predicted cell. This corrected 123 training, 55 validation and 46 test crop labels to unknown; it preserves the candidates, reference cohort and scoring regions. Unmarked cells never supply negative examples.

An initial 42-epoch run selected epoch 27 by balanced validation loss but failed the 90% precision target. It was not deployed. Refinement ran 24 epochs, selected epoch 9 by validation recall at the precision target, and reused the initial model’s weights. Half of each negative batch comes from the highest-scoring 5% of reviewed training negatives, mined every five epochs; the remainder is uniform. Learning rate is 0.0001.

Final cutoff calibration used every observed validation score and selected **0.967038929** without changing weights. The training charts show epoch-specific preview cutoffs from the earlier score grid; the final calibration and comparison reports show the deployed experimental cutoff. The updated training code also uses exact observed-score selection for future runs.

[Training curves](training/training.png) · [training report](training/report.json) · [cutoff graph](cutoff-calibration/calibration.png) · [initial attempt](initial-attempt.json) · [audit provenance](audit-summary.json)

## Candidate finder and controls

The validation-selected finder uses 0.10, 0.18 and 0.28 µm Gaussian scales, the original 0.025 contrast cutoff, 0.25 µm candidate separation and a 3×3×3 native maximum neighborhood. Final outputs use 0.35 µm separation after CNN scoring. Channel inference retains its separate original evidence cutoff. Candidate-scale CSV values describe filter settings, not measured signal radii.

On validation scans, raw candidate coverage rose from 292/362 (80.7%) to 313/362 (86.5%), with the scored candidate pool growing from 6,418 to 42,974. Lowering the contrast cutoff further produced more candidates without reaching additional references. Extra candidates are noise for the classifier to reject, not established signals.

[Candidate graph](candidate-calibration/candidate-comparison.png) · [candidate calibration](candidate-calibration/report.json)

The calibrated checkpoint passed blank, constant, noise and membrane-only controls with zero accepted points. Noise and membrane-only inputs generated 407 and 432 candidates respectively. This does not establish real-image precision. A classifier score is not a calibrated correctness probability.

[Control results](controls.json) · [compact summary](summary.json)

## Reproduction

Use the scripts `audit_signal_candidates.py`, `refresh_signal_data.py`, `train_refreshed_signals.py`, `calibrate_signal_model.py` and `compare_signal_models.py`. Full images and calibrated cell/collection audits are required for preparing data; they are not in the source/model bundle. The included `training-config.json` records the refinement recipe. Saved model manifests retain the parent checkpoint, audit hash, frozen cell-model hashes, mining policy and cutoff calibration.

Both apps can load the experimental model through their signal model folder field. Analysis’s **Use bundled signal model** button restores the previous default. The experiment was trained and exercised on Linux; native Windows execution of this refresh has not been tested locally.

## Application verification

The full suite passed 130 tests. After the Viewer archive compatibility fix, all 24 signal tests passed, including native and Windows-style archive paths. The real Analysis app completed the experimental model workflow with six retained cells and seven signals, valid A–B geometry, Excel and Imaris export, and unchanged source-image bytes. The Viewer worker rendered 42 signals with verified inside-cell centers, counts and 2D/3D overlays. All six model folders passed checksums and CPU loading/inference from the packaged copy. [Verification details](verification.json).
