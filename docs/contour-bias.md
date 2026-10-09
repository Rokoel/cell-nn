# Contour undersizing experiments

The October 9 experiments preserve the released cell center/box detector and
signal-center U-Net. They investigate where the contour is placed, using the
existing disjoint FISH acquisition split and the same four excluded scans.
Original models, source scans, annotations and prediction archives are preserved.

## Comparisons in order

1. Sweep probability cutoffs on the actual predicted boxes for all 441 complete
   baseline-matched validation cells. A cutoff of 0.45 improved voxel volume
   MAPE from 7.66% to 7.31%; mean symmetric voxel-boundary error fell from
   0.13658 to 0.13593 µm. Lower cutoffs over-expanded cells. These are validation
   grid measurements, not the historical 6.98% mesh-volume test cohort.
2. Compare 1/2/3 µm padding and 64³/96³ inputs. Keep 2 µm and 64³. The original
   crop contained the full manual bounding box in 438/441 cases. Changing size
   without retraining the model caused substantial errors; this does not establish
   that a model trained specifically at higher resolution would perform worse.
3. Fine-tune with a physical signed-distance boundary objective in addition to
   the existing Dice and boundary-weighted cross-entropy. Signed distances are
   derived after spatial augmentation, clipped at 2 µm and normalized. Unknown
   padding contributes no loss. Surface-loss weight is 0.1.
4. Try an asymmetric target-membership loss, assigning false-negative weight
   0.6 and false-positive weight 0.4, with the same surface objective.
5. Try a small residual 3D U-Net conditioned on the image, target seed and the
   frozen previous model's probability map. The previous network cannot change.
   The correction starts at zero and is bounded to ±2 logits. This allows local
   changes without an unconstrained replacement of the previous mask. Because
   the asymmetric penalty did not improve its parent, an additional balanced-loss
   correction control separates the architecture from that loss setting; it is
   registered before the control and before any test evaluation.

All complete 1,493 marked training cells are visited once per epoch; 224
synthetic negative crops supplement them. The 971 matched marked training cells
use frozen detector boxes and seeds; the other 522 retain manual crops. No
unmarked cell is sampled as a standalone negative training example. A different
cell in a target-conditioned crop is not a member of the selected target.

Validation still evaluates all 574 complete marked cells for loss; checkpoint
surface/volume scoring uses the fixed 441 baseline matches in production-like
crops. Each stage has at most eight epochs and three-epoch patience, with a
learning rate of 0.00005. The initial checkpoint is scored and retained if no
trained checkpoint improves boundary error without worsening volume MAPE.

The trained candidate is then checked on the original validation image grid.
Selection requires at least 99% of the original cell matches remain at IoU ≥0.5
and no increase in baseline volume MAPE. Unsupported changes retain the original
mask. Existing neighboring cells retain voxel ownership. Fixed cell identities
prevent detecting additional cells from being mistaken for a boundary gain.

## Selected model and test results

The production-crop surface-loss fine-tune wins (epoch 5 of 8). Asymmetric-loss
and both residual-correction trials do not improve their initial checkpoints.
After training, validation-only recalibration selects cutoff 0.55, with 2 µm
padding and 64³ inputs. The initial old-model calibration of 0.45 does not apply
to this new checkpoint. Selection is frozen before all 71 test scans are processed.

For the fixed 314 manually matched test cells, closed-mesh volume MAPE decreases
from 9.63% to 7.98%; symmetric surface error decreases from 0.159 to 0.146 µm.
For the fixed 150 signal-containing matched cells, volume MAPE decreases from
7.88% to 5.71%. At the same 140 U-Net predicted signal centers, distance MAE
decreases from 0.129 to 0.112 µm and RMSE from 0.177 to 0.153 µm. One center
moves outside its new mask and remains included and flagged. All 314 original
cell matches remain. Undersizing is reduced, not eliminated. The paired evaluation retains an original
mask if an edit fails; a fresh app run uses its normal proposal rejection rules
and does not automatically fall back to the old model. These are guarded
fixed-cell comparisons, not a fresh-run detection benchmark.

Choose `models/refinement-bias-candidate` in Analysis under **Models and device →
Boundaries**; keep the existing Cells and Signals models. The candidate is the bundled default since v0.4.1; previous
checkpoints are preserved. The candidate uses the unchanged contour architecture
and an inference manifest compatible with v0.4.0. Training-only loss settings are
stored separately in its manifest's `training_config`.

## Reports and reproduction

See [reports and graphs](../reports/contour-bias/README.md), including the
registered protocol, all validation configurations, training curves, fixed-cell
mesh comparisons and fixed signal-center distances. No outliers are removed.
The historical test split has been reused in earlier experiments; these results
remain exploratory agreement against manual drawings.

From the project root:

```bash
.venv/bin/python scripts/experiment_contour_bias.py --audit artifacts/refinement-fish/audit/audit.json --baseline <validation-contour-report> --mode threshold --output <new-threshold-folder>
.venv/bin/python scripts/experiment_contour_bias.py --audit artifacts/refinement-fish/audit/audit.json --baseline <validation-contour-report> --mode geometry --threshold .45 --output <new-geometry-folder>
.venv/bin/python scripts/build_contour_bias_patches.py --output <new-patch-folder>
.venv/bin/python scripts/train_contour_bias.py --initial-model <model-folder> --patches <new-patch-folder> --output <new-training-folder> --threshold .45
```

Add `--beta .6` for asymmetric training or `--beta .6 --residual` for the
correction network. All scripts preserve existing experiment directories.
The orchestration and reporting scripts document the actual completed local run.
