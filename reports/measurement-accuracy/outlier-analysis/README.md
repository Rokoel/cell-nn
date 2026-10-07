# Matched-measurement outlier sensitivity

Upper Tukey fence: Q3 + 1.5 * (Q3 - Q1), NumPy linear percentiles.

Volume exclusions use absolute percentage error; distance exclusions use absolute error in µm.
Volume upper fence: 30.280%; excluded 0/79.
Distance upper fence: 0.583759 µm; excluded 1/73.

| Measurement | Scenario | N | MAE | RMSE | Mean absolute % error | Median absolute % error |
|---|---|---:|---:|---:|---:|---:|
| Volume (µm³) | all | 79 | 7.7115 | 10.6820 | 10.15% | 9.80% |
| Volume (µm³) | iqr outliers excluded | 79 | 7.7115 | 10.6820 | 10.15% | 9.80% |
| Volume (µm³) | worst 5 percent trimmed | 75 | 7.4038 | 10.4740 | 9.26% | 8.26% |
| Distance (µm) | all | 73 | 0.1907 | 0.2426 | 84.89% | 31.64% |
| Distance (µm) | iqr outliers excluded | 72 | 0.1824 | 0.2259 | 82.72% | 31.25% |
| Distance (µm) | worst 5 percent trimmed | 69 | 0.1687 | 0.2061 | 83.92% | 30.35% |

## Tolerance coverage

- Volume, all: within 10 percent: 41/79 (51.9%); within 20 percent: 71/79 (89.9%).
- Volume, iqr outliers excluded: within 10 percent: 41/79 (51.9%); within 20 percent: 71/79 (89.9%).
- Volume, worst 5 percent trimmed: within 10 percent: 41/75 (54.7%); within 20 percent: 71/75 (94.7%).
- Distance, all: within 0.1 um: 24/73 (32.9%); within 0.2 um: 46/73 (63.0%); within 0.3 um: 56/73 (76.7%).
- Distance, iqr outliers excluded: within 0.1 um: 24/72 (33.3%); within 0.2 um: 46/72 (63.9%); within 0.3 um: 56/72 (77.8%).
- Distance, worst 5 percent trimmed: within 0.1 um: 24/69 (34.8%); within 0.2 um: 46/69 (66.7%); within 0.3 um: 56/69 (81.2%).

## Excluded by the standard IQR rule

- Volume: none.
- Distance: `FISH/dsNup133/dsNup133_Lam488_K9546_a.ims`, predicted cell 11, signal 4; absolute error 0.787877.

The 5% sensitivity removes four cells and four signals separately, ranked using the same
error definitions as the IQR rule. It does not mean all four are statistical outliers.
The cell and signal comparisons remain separate cohorts.

Percentage error = 100 × |predicted − manual| / manual. Distance percentages are unstable
for very short manual distances; interpret distance MAE in µm first.

2000 acquisition-group resamples, seed 42; exclusions held fixed.
These intervals condition on the selected observations and do not
include uncertainty in outlier selection or manual annotation.

Conditional on matched detections and retained observations; reused test split. Exclusions do not correct failed measurements or establish independent accuracy.

Original CSVs, unfiltered results and graphs are preserved. Audit CSVs and workbook
contain every original primary pair with explicit exclusion flags.

Reproduce: `.venv/bin/python scripts/analyze_measurement_outliers.py`
