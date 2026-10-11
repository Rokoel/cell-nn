# Balanced cell detection

[Documentation](README.md) · [Models](models.md) · [Analysis](fish-analysis.md)

Generated arrays and historical predictions have been removed. Rebuild training
caches before running reproduction commands; [training data](training-data.md)
explains the retained metadata and dependency order.

This run allows moderately elongated cells while keeping explicit checks for visible membrane evidence. It retains the four requested outlier exclusions and all duplicate exclusions from the conservative run, along with the same acquisition split and cached image preprocessing.

The completed run (`training_metadata/balanced-fish/training/neural-train-20260918-151950-694838/report.json`) trained for 150 epochs, selected epoch 150, and processed all 359 inputs in 29.3 minutes including validation-policy selection and final inference. On the same 71 test scans:

| Model | Contours | Complete references detected (out of 505) |
| --- | ---: | ---: |
| Original full collection | 22,009 | 445 |
| Conservative | 581 | 104 |
| Balanced | 1,925 | 172 |

The balanced model produces 3.31 times as many contours as the conservative model while retaining membrane checks. It detects 84/198 of the original spherical reference subset (conservative: 68/198), and 136/340 of the expanded eligible subset (conservative: 98/340). There are 1,737 predictions unmatched to any saved reference; these remain unscored, not automatically false. All six saved-checkpoint negative controls pass.

Historical per-run comparison exports were removed; the archived training report retains the run metrics. The later no-roundness inference option changes acceptance only; it does not retrain this saved checkpoint or rewrite these historical results.

## Training data and settings

| Split | Images | Eligible complete manual targets |
| --- | ---: | ---: |
| Training | 215 | 1,081 |
| Validation | 73 | 424 |
| Test | 71 | 340 |

The strict run had 690 training targets. The balanced run raises the maximum principal-axis ratio from 1.5 to 1.8 and lowers minimum sphericity from 0.72 to 0.65. Diameter remains 2.5–10 µm. Targets are complete manual annotations within those shape limits; unknown source regions are not treated as background. The run adds 271 synthetic blank/noise controls to the training pool.

Training starts from scratch with the same compact StarDist 3D architecture, 48³ patches, 64 rays, 16 initial filters, 200 batches per epoch, batch size one, learning rate 0.0003, and a maximum of 150 epochs. Validation loss covers every eligible validation crop; patience is 20 epochs. A fixed eight-image validation preview supplies contour graphs every ten epochs. Checkpoint selection uses validation loss.

The audit (`training_metadata/balanced-fish/audit/audit.json`) preserves all 3,290 reference annotations for evaluation, including cut cells and shapes outside the training limits. Settings (`training_metadata/balanced-fish/settings.json`) record the initial configuration. The selected acceptance thresholds are stored in the completed model manifest and run settings.

## Acceptance selection

Every candidate must have local membrane contrast, brightness, and inward/outward gradient alignment. Evidence must be present in every physical hemisphere. Complete absence of image evidence cannot pass merely because the predicted shape is round.

After choosing the checkpoint, four bounded policies are evaluated on validation images only:

| Policy | Center-score threshold | Required supported directions |
| --- | ---: | ---: |
| Stronger evidence | 0.65 | 60% |
| Intermediate | 0.55 | 55% |
| Balanced | 0.55 | 50% |
| More sensitive | 0.45 | 50% |

All use minimum membrane contrast 0.04, normalized rim intensity 0.10, overlap suppression threshold 0.25, and image-edge exclusion. The hemisphere rule requires at least `max(0.35, support − 0.15)` supported directions in each of six hemispheres. These values are empirical thresholds, not calibrated confidence.

A policy must produce zero accepted contours on all six synthetic negative controls: blank, dim constant, bright constant, two noise levels, and a one-sided membrane arc. Among policies detecting at least 60% of eligible validation references at IoU ≥ 0.5, selection chooses the fewest contours. If none reaches that target, selection uses the highest recall among the policies passing controls. No test images are used to choose the policy.

The September 18 run completed 150 epochs and selected epoch 150 by validation loss. All four policies passed the controls. Full-validation results were:

| Center score / support | Eligible references detected | Contours |
| --- | ---: | ---: |
| **0.65 / 60% — selected** | **60.38%** | **2,701** |
| 0.55 / 55% | 64.62% | 3,152 |
| 0.55 / 50% | 67.45% | 3,594 |
| 0.45 / 50% | 67.45% | 3,605 |

The selected policy satisfies the declared recall target with the fewest contours. Its hemisphere threshold is 45%. The selection report (`training_metadata/balanced-fish/training/neural-train-20260918-151950-694838/acceptance-selection.json`) records the complete settings and control results. The conservative model produced 1,372 contours on these same validation images; the full baseline produced 21,036.

Contour count is a conservative selection preference, not a false-positive measurement. Existing manual references are incomplete, so real-image precision remains unknown. Blank-control success does not guarantee every real-image prediction is correct. Human contour review remains available.

As an input-evidence check, the membrane rule was also applied directly to the 424 eligible manual validation masks, without neural predictions. Supported-direction requirements of 60%, 55%, and 50% retained 280, 297, and 311 masks respectively. This confirms that the relaxed gate retains more annotated boundaries while continuing to require evidence on every side; it does not measure false-positive rejection on real images.

## Viewer and reproduction

Use **Neural training… → Choose audit…** with `artifacts/balanced-fish/audit/audit.json` to restore the full collection configuration. **Use balanced cell selection** permits moderate elongation and enables the bounded validation selection. **Use conservative cell selection** restores the stricter preset and disables that selection. Choosing a saved model restores that model's actual selected thresholds.

```bash
.venv/bin/python -m fish_viewer.neural FISH \
  --audit artifacts/balanced-fish/audit/audit.json \
  --settings training_metadata/balanced-fish/settings.json --output neural_runs

# Derive the same curated collection in a new location.
.venv/bin/python -m fish_viewer.neural \
  --curate-audit artifacts/full-fish/audit/audit.json --balanced \
  --settings training_metadata/balanced-fish/settings.json --output /path/to/new-audit
```

The [model index](models.md) links every main saved model and report. Bundled checkpoints and original Imaris annotations are preserved; old run folders were removed.
