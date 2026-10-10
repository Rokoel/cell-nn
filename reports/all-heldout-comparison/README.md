# Full held-out comparison

All 144 unique held-out scans were processed with all three modes (432 completed runs). Test and validation are separate. No weights or thresholds were adjusted.

## Test scans

| Mode | Scans | Cells before / after signal filtering | Complete manual matches before / after | Volume MAPE (pairs) | Distance MAE, µm (pairs) |
|---|---:|---:|---:|---:|---:|
| Current default | 71 | 4483 / 1182 | 315 / 189 of 508 | 6.59% (189) | 0.112 (149) |
| Sensitive StarDist hybrid | 71 | 8917 / 1650 | 373 / 213 of 508 | 10.43% (213) | 0.154 (173) |
| Both-detector consensus | 71 | 4081 / 1029 | 300 / 174 of 508 | 8.95% (174) | 0.129 (137) |

Volumes and distances above use complete manual cells. All-object and cut-cell detection totals are retained in the JSON and CSV reports. Different modes have different matched subsets.

| Metric on the same complete-cell references | Default | Hybrid | Consensus |
|---|---:|---:|---:|
| Volume MAPE (%) (160 pairs) | 5.929 | 8.577 | 8.577 |
| Distance MAE (µm) (130 pairs) | 0.115 | 0.130 | 0.130 |

## Validation scans

| Mode | Scans | Cells before / after signal filtering | Complete manual matches before / after | Volume MAPE (pairs) | Distance MAE, µm (pairs) |
|---|---:|---:|---:|---:|---:|
| Current default | 73 | 5343 / 1636 | 443 / 310 of 574 | 5.16% (310) | 0.118 (247) |
| Sensitive StarDist hybrid | 73 | 9368 / 2024 | 496 / 326 of 574 | 8.01% (326) | 0.125 (269) |
| Both-detector consensus | 73 | 4923 / 1461 | 429 / 289 of 574 | 7.23% (289) | 0.125 (238) |

Volumes and distances above use complete manual cells. All-object and cut-cell detection totals are retained in the JSON and CSV reports. Different modes have different matched subsets.

| Metric on the same complete-cell references | Default | Hybrid | Consensus |
|---|---:|---:|---:|
| Volume MAPE (%) (274 pairs) | 4.907 | 6.902 | 6.902 |
| Distance MAE (µm) (226 pairs) | 0.118 | 0.124 | 0.124 |

## All held-out scans

| Mode | Scans | Cells before / after signal filtering | Complete manual matches before / after | Volume MAPE (pairs) | Distance MAE, µm (pairs) |
|---|---:|---:|---:|---:|---:|
| Current default | 144 | 9826 / 2818 | 758 / 499 of 1082 | 5.70% (499) | 0.116 (396) |
| Sensitive StarDist hybrid | 144 | 18285 / 3674 | 869 / 539 of 1082 | 8.97% (539) | 0.136 (442) |
| Both-detector consensus | 144 | 9004 / 2490 | 729 / 463 of 1082 | 7.88% (463) | 0.126 (375) |

Volumes and distances above use complete manual cells. All-object and cut-cell detection totals are retained in the JSON and CSV reports. Different modes have different matched subsets.

| Metric on the same complete-cell references | Default | Hybrid | Consensus |
|---|---:|---:|---:|
| Volume MAPE (%) (434 pairs) | 5.284 | 7.519 | 7.519 |
| Distance MAE (µm) (356 pairs) | 0.117 | 0.126 | 0.126 |

## Interpretation and reproducibility

Cell matching requires one-to-one whole-object IoU ≥ 0.5. Distance matching requires the same cell and signal channel, with signal centers within 0.6 µm. Physical volumes are measured on closed triangle meshes; distances use nearest points on those meshes and explicit original manual A–B segments. All annotation aliases are considered, exact reference masks are deduplicated, and ambiguous manual channels/assignments or conflicting A–B segments are reported as exclusions. No manual annotations guide inference.

The JSON includes biases, MAE/RMSE, median and 95th-percentile errors, and 95% acquisition-group bootstrap intervals. All outliers are retained. Unmatched cells remain unscored because manual annotations are incomplete; their number is not a false-positive count. Validation was used for model selection and test scans have been inspected in earlier experiments.

Run `scripts/benchmark_all_heldout.py` with `FISH_SPOTS_PYTHON` pointing to the PyTorch environment. `--resume` preserves completed runs and checks model/audit hashes. Disjoint `--shards`/`--shard-index` jobs can share a run directory. `scripts/summarize_all_heldout.py --run RUN_DIRECTORY` refuses to publish if any held-out scan/mode is missing. Only Imaris image/scene export is omitted in this benchmark.

[Test-set distribution graphs](test-comparison.png). Raw per-scan counts and matched volume/distance pairs are available in the CSV files beside this report.
