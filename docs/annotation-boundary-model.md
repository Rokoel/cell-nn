# Annotation-trained boundary guidance

This experiment adds a small 2D U-Net trained from `training_annotations/`. The existing center/box detector, surface-loss 3D contour model and signal-center U-Net retain their weights. The new model predicts cell-membrane probability, then guides bounded movement of the existing 3D surfaces before signal detection and A–B measurement.

## Training

```sh
.venv/bin/python -m fish_viewer.boundary --annotations training_annotations --output models/boundary-annotations --device gpu
```

Windows uses `.venv\Scripts\python.exe`; CPU training/inference is available with `--device cpu`. TensorFlow is required, as for the existing contour model. This initial trainer expects schema-2 128 × 128 crops. Labels are 3 (border), 1 (non-border), and 0 (unknown). Unknown pixels have zero loss and zero direct logit gradient. Non-border includes cell interiors as well as external background and noise; it is not an empty-cell label.

The model has 121,969 parameters. Loss combines per-crop class-balanced cross entropy and masked Dice on crops with positive labels. Sampling uses approximately 75% border-containing crops and 25% negative-only crops. Rotation, flipping and intensity augmentation preserve unknown labels. The acquisition audit determines split/channel membership; annotations from validation/test scans are rejected.

A group-disjoint development split selects the epoch and threshold. The final model is independently refitted on all available annotations for the selected epoch count. `development.weights.h5` preserves the evaluated development checkpoint, `boundary.weights.h5` is the final all-annotation model, and `boundary-model.json` records provenance and calibration. Metrics in the manifest describe the development checkpoint, not the all-annotation refit. `training-report.json` contains per-epoch losses, validation metrics and group assignments.

## 3D integration

In FISH Analysis (`run-analysis.sh` / `run-analysis.bat`), expand processing options and select **Learned boundary guidance (experimental)**. The model is read from `models/boundary-annotations`. The existing membrane search-radius control sets the maximum search in either direction (default 0.3 µm). Displacement strength remains 0.5, giving a maximum vertex movement of 0.15 µm at the default radius. When both membrane-snap and learned-guidance options are checked, the learned map is used for the one refinement stage.

Slices are evaluated using overlapping 128-pixel tiles, percentile normalization and physical XY rescaling to the training pixel scale. Probability tiles are blended, then mapped to the exact calibrated 3D contour grid. The existing refinement volume is the input; no signal-channel pixels are used. This is a slice-wise predictor with a smooth 3D mesh correction, not a new volumetric detector.

Surface normals search for nearby supported peaks in learned boundary probability. Local 3D smoothing regularizes motion. Corrections preserve cell cores, connectivity, topology and neighboring instance voxels, and must remain inside conservative volume limits. The original raw membrane evidence must support the corrected surface and cannot materially deteriorate. Failed corrections retain the original contour. The model does not create new cell instances or force spherical shapes.

Command-line application to a completed cell run:

```sh
.venv/bin/python -m fish_viewer.refinement.membrane \
  --cells-report /path/to/cells/report.json \
  --boundary-model models/boundary-annotations \
  --output artifacts/boundary-apply --device gpu
```

An optional `--audit artifacts/refinement-fish/audit/audit.json` reuses the calibrated membrane cache. Run signal inference, cell filtering and measurement on the resulting contours, as the app does automatically. Reports preserve the guidance model path, checkpoint checksum, settings and per-cell acceptance/fallback diagnostics.

The broad brush labels specify boundary bands, not exact measurement surfaces. Probability thresholds selected against those bands do not prove micrometer-scale localization accuracy. The existing default stays available, and adopting the experiment as a default requires comparing matched volumes and A–B distances on held-out scans. Cell precision remains unidentifiable where manual cell annotations are incomplete.

## Completed experiment (10 October 2026)

Training used all 45 saved crops from 40 scans and 32 acquisition groups. Development used 37 crops from 26 groups for fitting and eight crops from six other groups for validation; seven validation crops contain border marks and one is negative-only. The selected development checkpoint achieved precision 71.7%, recall 81.9%, F1 76.5% and IoU 61.9% on labeled pixels at threshold 0.6. The final model was refitted on all 45 crops for 35 epochs. These are broad brush-band metrics, not exact contour localization scores.

The full comparison completed all 144 held-out scans (71 test, 73 validation) without failures. Existing default cell detections were reused; learned guidance, the unchanged signal-center U-Net, cell filtering and mesh measurements were freshly run. Original-image exports were omitted from the batch benchmark; a separate native GUI smoke test exercised Imaris/Excel exports and 2D/3D display with guidance enabled.

Across all held-out scans, overall matched volume MAPE stayed approximately 5.70%; distance MAE changed from 0.1158 to 0.1171 µm. On the same 496 complete-cell references, volume MAPE decreased from 5.71% to 5.59%, a paired change of −0.122 percentage points (acquisition-group bootstrap 95% interval −0.225 to −0.013). On the same 393 manual signal references, distance MAE was effectively unchanged (paired change −0.000057 µm, 95% interval −0.001385 to +0.001144). Thus this is a small volume improvement on shared references, not a demonstrated distance improvement or a substantial overall accuracy gain. The option remains experimental and off by default.

[Complete results](../reports/annotation-boundary/README.md), [training curves](../reports/annotation-boundary/training.png), and [development examples](../reports/annotation-boundary/development-examples.png).
