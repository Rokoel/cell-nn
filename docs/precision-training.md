# Conservative cell selection

[Documentation](README.md) · [Models](models.md) · [Analysis](fish-analysis.md)

Generated arrays and historical predictions have been removed. Rebuild training
caches before running reproduction commands; [training data](training-data.md)
explains the retained metadata and dependency order.

The priority is to reject unsupported or incorrect contours, accepting that some real cells will be missed. Automatic acceptance does not establish certainty. The viewer can also export only contours that a person has confirmed.

## Completed run

The trained model and report (`training_metadata/precision-fish/training/neural-train-20260915-222552-439128/report.json`) completed 120 epochs, selected epoch 119, and produced predictions for all 359 revised inputs in 16.3 minutes. It generated 4,423 accepted candidates across the collection. No inference proposal limits or processing failures occurred.

On the same 71 test images, using the same 198 shape-eligible complete reference annotations:

| Measurement | Full baseline | Conservative model |
| --- | ---: | ---: |
| Predicted contours | 22,009 | 581 |
| Eligible references detected at IoU ≥ 0.5 | 196 / 198 | 68 / 198 |
| Mean Dice per eligible reference, including misses | 0.853 | 0.296 |
| Mean Dice among detected eligible references only | 0.856 | 0.863 |

This is **97.4% fewer contours with a steep recall cost**, not a measured 97.4% reduction in false positives. There are 475 predictions unmatched to any saved reference; their correctness remains unknown. Across all 505 complete test annotations, macro Dice is 0.167 and 104 are detected. The model deliberately rejects many annotated shapes or insufficiently supported membranes.

See the comparison data, validation illustrations, and test summary (`training_metadata/precision-fish/training/neural-train-20260915-222552-439128/test-summary.csv`). The six synthetic negative controls all yield zero accepted contours. The network still proposes one object on a one-sided arc; the membrane gate rejects it. Those controls do not establish real-image precision.

The completed training report is archived locally; historical prediction arrays were removed. Open newly generated runs through **Neural training… → Results → Open saved run…**. Use **Review selected cell contours…** and export confirmed contours when automatic acceptance is insufficient.

## Why the excessive contours?

The original annotations are partial. Unannotated voxels must remain unknown because many contain real unmarked cells. Without reviewed background or false detections, this gives almost no supervision for rejecting incorrect candidates. The original validation sweep selected probability threshold 0.35 by maximizing Dice against known cells. Extra predictions were unscored, so this objective could not penalize them.

The full baseline reached mean image Dice 0.790 on 505 complete test annotations, detecting 445 at IoU ≥ 0.5. It produced 22,009 test contours; 21,367 did not match a reference at the evaluation matching threshold. These include both real unannotated cells and errors and cannot automatically be labeled false positives.

Other plausible contributors are mixed acquisition scales, ambiguous channel choices, overlapping alternative boundaries, and the model inferring a cell from a bright arc. Shape filtering alone cannot reject a perfectly round hallucination.

## Revised dataset

The derived audit (`training_metadata/precision-fish/audit/audit.json`) excludes `18327antiLAD2.ims`, `18327antiLAD3.ims`, `18327antiLAD5.ims`, and `3AantiLADFITCTyrCy5_546.ims`, plus duplicate inputs: 11 source files representing four unique scans. Original files remain available to the viewer.

| Split | Unique scans | Acquisition groups | Eligible complete annotations |
| --- | ---: | ---: | ---: |
| Training | 215 | 56 | 690 |
| Validation | 73 | 19 | 279 |
| Test | 71 | 19 | 198 |

There are 359 unique inputs from 381 remaining source images. The original split is preserved; all excluded scans belonged to training. All 3,290 remaining references, including 739 cut annotations and shapes outside the training limits, remain available for evaluation. Shape eligibility is automatic curation, not a human judgment of annotation quality.

The model trains from scratch on the 690 manual targets, each isolated in its own crop, plus 173 synthetic blank/noise crops with explicit background labels. Unknown source voxels are never relabeled. Synthetic controls address blank inputs; they do not teach every way a cell fragment can create a false contour.

Settings: 48³ patches, 64 rays, 16 initial filters, batch size one, 200 batches per epoch, up to 120 epochs, learning rate 0.0003, patience 20. Validation loss covers all 279 eligible targets every epoch. Contour graphs sample eight fixed validation images every ten epochs. The best validation-loss checkpoint supplies final predictions for all 359 inputs. No probability threshold sweep is performed.

## Acceptance rules

- Probability threshold 0.75; overlap suppression threshold 0.15.
- Equivalent diameter 2.5–10 µm, sphericity ≥ 0.72, axis ratio ≤ 1.5; image-edge cells excluded.
- Membrane evidence along 128 directions: a bright local ridge, darker samples on both sides, and gradients aligned with the inward/outward direction.
- At least 65% supported directions, with at least 50% in each of the six physical hemispheres.

Saved preprocessing retains its 2.5–32 µm channel-search range and calibrated sampling. The separate 10 µm acceptance limit changes selection without changing cached inputs. The GUI exposes evidence and shape thresholds. These are empirical rules, not calibrated confidence. StarDist's probability output is not the probability that a whole contour is correct; see its [FAQ](https://github.com/stardist/stardist-docs/blob/main/docs/faq.md).

Controlled tests reject blank and constant-bright volumes, a one-sided arc, and a displaced sphere. A correctly aligned closed spherical membrane passes. On 279 eligible manual validation contours, the membrane rule accepts 176 before neural prediction. This documents a recall tradeoff; it does not measure precision.

## Viewer and progress

Launch `./run-viewer.sh FISH`, open **Neural training…**, and **Choose audit…** → `artifacts/precision-fish/audit/audit.json`. **Use conservative cell selection** reapplies this strict preset. New GUI configurations now [require visible boundaries without roundness limits](auto-contours.md); saved reports and models restore their own settings. Applying a saved model permits deliberate acceptance overrides while preserving preprocessing.

The progress panel stays above every tab and displays phase, epoch, completed epochs, batch/image counts, filename, volume tile, and recent losses. Final prediction identifies train/validation/test membership. **Open saved run…** can watch a running command-line job, refreshing progress and graphs every two seconds.

In **Results**, select an image and **Review selected cell contours…**. Three orthogonal views have independent slice sliders. Label candidates **Correct cell and contour**, **False / incorrect contour**, or **Unsure**. Decisions save in `contours.review.json`, tied to a hash of the exact predictions. They are not converted into voxel background masks: an incorrect contour may cover part of a real cell.

**Export confirmed contours** writes `contours.reviewed.npz` with only confirmed contours and opens it in the 2D/3D viewer. Unreviewed and uncertain contours are excluded. Original predictions and Imaris annotations are preserved. Retain the audit cache to review audited runs.

## Reproduce

```bash
.venv/bin/python -m fish_viewer.neural \
  --curate-audit artifacts/full-fish/audit/audit.json \
  --settings training_metadata/precision-fish/settings.json --output /path/to/new-curated-audit

.venv/bin/python -m fish_viewer.neural FISH \
  --audit artifacts/precision-fish/audit/audit.json \
  --settings training_metadata/precision-fish/settings.json --output neural_runs

.venv/bin/python -m fish_viewer.neural /path/to/images --mode apply \
  --model /path/to/model --precision --output neural_runs
```

## Next model experiments

| Option | Purpose and tradeoff |
| --- | --- |
| StarDist plus a small 3D candidate classifier | Recommended after review labels exist. Learn whether a cell and contour agree with the image, using correct contours and real incorrect proposals. Split reviews by acquisition and assess precision separately. |
| Cellpose-SAM v2 / Cellpose-DINO | Alternative instance segmentation candidates. The 3D workflow combines predictions in three planes: a different shape model, without a guarantee of fewer false detections. See [models](https://cellpose.readthedocs.io/en/latest/models.html) and [3D processing](https://cellpose.readthedocs.io/en/latest/do3d.html). |
| 3D U-Net predicting boundary/interior/background | Makes boundary evidence explicit, with watershed separating instances. An nnU-Net experiment needs reliable labeled regions/background and instance separation, or custom sparse-label handling. See [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). |

These alternatives are proposals, not models trained in this run. Reviewed incorrect proposals and independent validation examples are the most direct route to measuring precision. Later filtering for cells containing signals can supplement contour validation.
