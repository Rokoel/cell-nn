# Annotation-trained boundary guidance benchmark

All 144 held-out scans. Frozen baseline cell detections reused; learned contour guidance, signal-center NN, filtering and physical mesh measurements freshly executed. Complete manual cells only for primary errors, original manual A-B segments only, all outliers retained. No exported scan duplication.

## test

| Method | Scans | Manual complete cells matched before signals | Volume MAPE | Distance MAE |
|---|---:|---:|---:|---:|
| default | 71 | 315 | 6.59% (n=189) | 0.112 µm (n=149) |
| boundary_guided | 71 | 317 | 6.63% (n=190) | 0.115 µm (n=148) |

Same manual references matched by both methods:

- default: volume MAPE 6.58% (n=188), distance MAE 0.113 µm (n=147).
- boundary_guided: volume MAPE 6.39% (n=188), distance MAE 0.114 µm (n=147).

## validation

| Method | Scans | Manual complete cells matched before signals | Volume MAPE | Distance MAE |
|---|---:|---:|---:|---:|
| default | 73 | 443 | 5.16% (n=310) | 0.118 µm (n=247) |
| boundary_guided | 73 | 443 | 5.14% (n=312) | 0.118 µm (n=248) |

Same manual references matched by both methods:

- default: volume MAPE 5.18% (n=308), distance MAE 0.118 µm (n=246).
- boundary_guided: volume MAPE 5.10% (n=308), distance MAE 0.118 µm (n=246).

## all

| Method | Scans | Manual complete cells matched before signals | Volume MAPE | Distance MAE |
|---|---:|---:|---:|---:|
| default | 144 | 758 | 5.70% (n=499) | 0.116 µm (n=396) |
| boundary_guided | 144 | 760 | 5.70% (n=502) | 0.117 µm (n=396) |

Same manual references matched by both methods:

- default: volume MAPE 5.71% (n=496), distance MAE 0.116 µm (n=393).
- boundary_guided: volume MAPE 5.59% (n=496), distance MAE 0.116 µm (n=393).

[Boundary training curves](training.png)

[Group-held-out annotation examples](development-examples.png)

Brush-band metrics do not imply exact surface accuracy. Measurement errors are conditional on matched cells/signals. Manual cell annotation is incomplete. Validation and test acquisitions have been previously inspected; this is not a new blind cohort.