# Training on the full FISH collection

This is the completed baseline. For the revised exclusions, conservative acceptance and review workflow, see [conservative training](precision-training.md).

The baseline completed 150 epochs, selected epoch 146, and reached complete-test macro Dice 0.790 / IoU 0.664, detecting 445 of 505 complete annotations at IoU ≥ 0.5. It produced 22,009 test contours, of which 21,367 were unscored against partial references. Precision is unknown.

The collection audit covers every existing Imaris image under `FISH`, including nested folders, alternate scenes, and the older TIFF containers. The new model trains from scratch using the expanded annotation set. Original images and scenes remain unchanged.

## Audit findings

| Finding | Count |
| --- | ---: |
| Existing image files read at full source resolution | 392 |
| Referenced images missing from disk | 4 |
| Distinct normalized network inputs | 363 |
| Duplicate network inputs consolidated | 29 |
| Manual reference contours after consolidation | 3,339 |
| Complete reference annotations available for supervision | 2,592 |
| References cut by the image boundary | 747 |
| Complete annotations recovered by separate object crops | 310 |
| Images flagged for weak/ambiguous automatic channel evidence | 67 |

All 392 existing images have one time point. The audit reads every native-resolution channel, chooses a channel automatically, records calibration and source fingerprints, and rasterizes the saved meshes. Different scenes for the same image are combined; identical voxelized reference masks are deduplicated. Identical normalized inputs with the same calibrated grid are consolidated, preserving additional annotations from their aliases. Counts refer to saved annotations; overlapping annotations can describe closely related or alternative cell boundaries.

The four absent images are listed in the audit. No image-read or annotation-processing errors occurred for the existing files. Representative previews, including ambiguous selections and legacy scans, are saved in [channel-review.png](../artifacts/full-fish/channel-review.png); the confidence flags remain in the exported image table.

Audit artifacts:

- [audit.json](../artifacts/full-fish/audit/audit.json): sources, aliases, acquisition components, parameters, complete/cut counts, and preprocessing warnings.
- [source-files.csv](../artifacts/full-fish/audit/source-files.csv): one row per existing source image, including duplicates and companion scenes.
- [images.csv](../artifacts/full-fish/audit/images.csv): one row per distinct input, selected channel, split, and annotation counts.
- [annotations.csv](../artifacts/full-fish/audit/annotations.csv): one row per reference, physical shape measurements, cut status, and training eligibility.
- [folders.csv](../artifacts/full-fish/audit/folders.csv): counts by folder.
- [shape-audit.json](../artifacts/full-fish/audit/shape-audit.json): complete reference shapes relative to the prediction filters.
- [strongly-overlapping-references.csv](../artifacts/full-fish/audit/strongly-overlapping-references.csv): 40 annotation pairs with IoU above 0.5, including 22 above 0.9 that may be alternative boundaries for the same cell. These remain within their acquisition component; annotation counts do not imply that all biological cells are independent.

## Split and supervision

| Split | Acquisition components | Distinct scans | Complete annotations |
| --- | ---: | ---: | ---: |
| Training | 57 | 219 | 1,523 |
| Validation | 19 | 73 | 564 |
| Test | 19 | 71 | 505 |

Acquisition dates are parsed as DDMMYY. Images sharing a date are grouped across folders. Undated fields in the same folder stay together. Duplicate-input aliases join their acquisition components before the seeded 60/20/20 split. A component never crosses splits. Images without usable labels remain associated with their acquisition but supply no supervised training crop.

Each complete reference supplies its own crop. Its full mask is positive; other cells are ignored in that crop, including intersecting annotations. This retains the 310 complete annotations rejected by the earlier overlap rule without truncating their masks or inventing background. Cut references remain available for evaluation. Explicitly reviewed background masks, if supplied, retain their separate meaning. No reviewed background was supplied for this run, so precision cannot be established from these partial labels.

The model learns from all complete annotations, including elongated cells. Prediction filtering then selects the requested approximately spherical objects. Of the complete references, 1,886 pass the current physical shape filters; 701 exceed the axis-ratio limit, and five fall outside the diameter interval. Scores against all complete references therefore include shapes the final filter is deliberately designed to reject. Reports also retain evaluation before shape filtering.

## Changes required for the larger collection

The collection includes cells around 5 µm across and legacy cells around 25–30 µm. The full-run settings use a 2.5–32 µm diameter interval, minimum sphericity 0.62, maximum axis ratio 1.8, and an 80 × 80 × 80 maximum training crop. Edge exclusion is off for prediction.

The processing voxel is approximately 0.30 µm unless the source XY pixels are coarser; in that case the coarser XY sampling sets the target spacing. This preserves physical coordinates and avoids unnecessary lateral upsampling of the legacy scans. The selected spacing is recorded per image and used again when applying the saved model.

Crops are read lazily from memory-mapped arrays. Validation loss covers every usable validation crop each epoch, with no augmentation. For progress graphs, full-image contour measurements use a fixed eight-image validation preview every 15 epochs. After checkpoint selection, threshold calibration uses **all validation images**, followed by final prediction and evaluation of the test images. Preview scores and final full-validation scores have different image populations and are labeled accordingly.

Inference processes blocks with a 32-voxel context halo and disjoint central regions. It retains sparse centers and radial predictions, then performs global instance suppression. This bounds the dense network output memory for large fields. The 4,000-proposal limit remains explicit in prediction reports.

Training settings are [saved here](../artifacts/full-fish/settings.json): up to 150 epochs, 200 batches per epoch, batch size one, 64 rays, 16 initial U-Net filters, learning rate 0.0003, and early-stop patience 20 epochs. Loss and checkpoint selection use the independent validation component.

## Use in the viewer

Launch `./run-viewer.sh FISH` and open **Neural training…**.

- **Choose audit…** selects `artifacts/full-fish/audit/audit.json` and restores its settings and fixed input collection. Individual file selection is disabled for audited training.
- **Clear** returns to ordinary selected-file training.
- **Apply a saved StarDist model** uses the selected images and saved model settings; an audit selection does not force application to the whole collection.
- **Open saved run…** restores the graphs, results, model folder, and collection audit from a training report.

The full run is saved in [`artifacts/full-fish/training/neural-train-20260915-033619-354714/`](../artifacts/full-fish/training/neural-train-20260915-033619-354714/). Its [report.json](../artifacts/full-fish/training/neural-train-20260915-033619-354714/report.json) is updated after each epoch. Keep the audit cache as well as the model/run folders to reproduce training. Source image and scene file sizes/timestamps are checked before reusing an audit.

## Reproduce from the command line

```bash
# Use a new output directory when repeating the audit.
.venv/bin/python -m fish_viewer.neural FISH --audit-only \
  --settings artifacts/full-fish/settings.json --output /path/to/new-audit

.venv/bin/python -m fish_viewer.neural FISH \
  --audit /path/to/new-audit/audit.json \
  --settings artifacts/full-fish/settings.json --output neural_runs

.venv/bin/python scripts/summarize_collection.py \
  --audit /path/to/new-audit/audit.json --run /path/to/run/report.json
```

The held-out split is independent within this run. Two annotated test images from the `131212` acquisition were previously used for validation in the small pilot; supplementary test summaries exclude the prior-pilot folder to make that exposure visible. The larger test set is not directly comparable to the earlier single-image score. Unmatched predictions remain unscored throughout.
