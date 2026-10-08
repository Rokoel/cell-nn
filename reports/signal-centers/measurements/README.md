# Cell-volume and signal-to-border distance accuracy

Calculated from the frozen U-Net and Spotiflow test predictions on all 71 test scans. These are **end-to-end distances from the predicted signal center to the predicted cell surface**, compared with the manually drawn A–B lines. They are different from the signal-center localization errors in the detector comparison.

Both networks are signal detectors. They share the same center/box → crop-U-Net cell contours. Their volumes are therefore identical for the same cell; selecting cells with detected signals changes the reported subset, not the contour geometry.

## Volumes

On the same **314 complete manually matched cells** (whole-cell IoU ≥ 0.5), before signal filtering, both pipelines have:

| Metric | Both signal models |
| --- | ---: |
| Mean absolute volume percentage error | **9.63%** |
| Median absolute volume percentage error | **6.36%** |
| Mean absolute volume error | 8.80 µm³ |
| Volume RMSE | 16.22 µm³ |
| Mean signed volume bias | −6.77 µm³ |

For comparison, selecting only those matched cells retained by detections in the scored regions gives 150 cells / 7.88% mean absolute volume percentage error for U-Net, and 136 cells / 7.71% for Spotiflow. **Those are different cell subsets, not evidence that one signal network draws better contours.** These subset counts use scored-region detections; full production inference can retain additional cells based on detections in unknown regions.

## Signal-to-border distances

Each model's eligible matched signals, all outliers included:

| Metric | 3D center U-Net | Spotiflow |
| --- | ---: | ---: |
| Signals with eligible manual A–B measurements | **140** | **123** |
| Mean absolute distance error | **0.129 µm** | **0.129 µm** |
| Distance RMSE | 0.177 µm | 0.181 µm |
| Median absolute distance error | 0.096 µm | 0.084 µm |
| Median absolute percentage error | 22.3% | 20.0% |
| Mean absolute percentage error | 46.1% | 33.3% |
| Mean signed distance bias | −0.072 µm | −0.073 µm |
| 95th percentile absolute error | 0.378 µm | 0.333 µm |
| Acquisition-bootstrap 95% interval for MAE | 0.103–0.157 µm | 0.090–0.168 µm |

The negative bias means distances tend to be underestimated. Mean percentage errors are sensitive to very short manual distances; a modest absolute error can produce a very large percentage error. Median percentages and absolute errors are more useful together than either alone.

Of 163 matched U-Net centers, 23 lacked an eligible manual distance comparison; of 148 matched Spotiflow centers, 25 did. Eligible comparisons require a complete manually drawn cell matched at IoU ≥ 0.5, an inferred channel, and an unambiguous positive manual A–B distance. Missing comparisons are counted rather than removed as outliers.

## Same-signal comparison

On the **same 107 manual signals** recovered by both networks:

| Metric | U-Net | Spotiflow |
| --- | ---: | ---: |
| Distance MAE | **0.128 µm** | **0.132 µm** |
| Distance RMSE | 0.176 µm | 0.184 µm |
| Median percentage error | 21.1% | 18.6% |

The U-Net MAE advantage is only **0.0038 µm**. A paired acquisition bootstrap (2,000 resamples over 14 groups) gives a 95% interval of **−0.0146 to +0.0140 µm** for that advantage. This does not establish a meaningful accuracy difference.

On the stricter intersection of **85 signals recovered by all three detectors**, MAE is 0.136 µm for the current conservative detector, 0.108 µm for U-Net and 0.114 µm for Spotiflow. Those numbers apply to that shared subset and should not be substituted for each model's full matched-signal error above.

## Method and limits

Existing complete manual geometry and A–B measurements were reused from the [cell-pipeline measurement benchmark](../../center-box-contours/test-measurements/report.json). Its cell prediction archive paths match the new detector benchmark exactly. The code verified pixel fingerprints and recomputed mesh volumes against the cached values before measuring new distances. Predicted borders use the existing nearest-triangle `SurfaceLocator`, the same method used by the application. Signal matching reproduces the frozen 0.6 µm one-to-one, same-cell/channel detector benchmark; predicted centers are not moved or retuned for this comparison.

These are agreement estimates against manual annotations, conditional on matched cells and signals. They do not establish absolute biological accuracy or error for unmatched detections. All outliers remain included. The historical test fold has been reused and the result remains exploratory. The signal models improve detection coverage, but substantial border-distance disagreement remains.

[Full results and measurement rows](report.json) · [U-Net distance CSV](unet-distances.csv) · [Spotiflow distance CSV](spotiflow-distances.csv)

To reproduce with the original local corpus and cached references, run:

```bash
.venv/bin/python scripts/evaluate_center_measurements.py --output artifacts/signal-center-measurements-repeat
```
