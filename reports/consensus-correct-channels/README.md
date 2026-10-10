# Both-detector consensus with corrected cell channels

All 144 held-out scans evaluated with the current app models and a minimum roundness of 0.72. Channels use declared Lam/LBR wavelengths, with ring evidence ranking multiple membrane stains. The nine scans without stain declarations were reviewed visually and use their sole 633 nm membrane metadata hint. Cell-channel indices in the CSV and JSON are zero-based.

| Split | Scans | Matched complete cells (retained) | Volume mean absolute error | Distance MAE | Distance RMSE | Matched signals |
|---|---:|---:|---:|---:|---:|---:|
| Test | 71 | 299 | 8.59% | 0.130 µm | 0.178 µm | 270 |
| Validation | 73 | 359 | 5.95% | 0.118 µm | 0.160 µm | 332 |
| All held-out | 144 | 658 | 7.15% | 0.123 µm | 0.168 µm | 602 |

All held-out scans, fixed app models and roundness gate. Explicit stain-aware cell channels. Unchanged-channel cell inference reused only with matching checkpoint/manifest hashes and input fingerprints; signals/filtering/distances rerun. Complete-cell one-to-one IoU >=0.5; same-cell/channel signal centers within 0.6 um. Original manual A-B distances only. All outliers retained. Validation influenced model selection and the historical test fold was used in earlier exploration; these are not blind prospective estimates. Sparse labels do not establish precision of unmatched cells.

Volume errors exclude manual cells cut by the image boundary. Distances use signals in matched complete cells. The JSON includes signed bias, medians, tails, and acquisition-bootstrap 95% intervals. Per-scan counts and individual measurement pairs are preserved in the CSV files. Neither model weights nor app defaults were changed.

Reproduce with `scripts/benchmark_consensus_channels.py`; `--publish-only reports/consensus-correct-channels` verifies that every expected scan completed before publishing.

58 cell-channel assignments differ from the original audit. On test scans, mean signed volume error is -7.86% (negative means volumes are too small). Mean signed distance error is -0.077 µm. Median absolute volume error is 5.76%; median absolute distance error is 0.090 µm.

Acquisition-bootstrap 95% intervals on the test set: volume MAPE 5.67–11.80%; distance MAE 0.109–0.146 µm.
