# Automatic contours and saved reports

Restart the viewer with `./run-viewer.sh FISH` to load these changes.

1. Open a scan from the dataset list.
2. Click **Auto-select contours…** immediately below the scan name.
3. Choose a saved model. **Balanced collection** is listed first; **Browse…** accepts another model folder containing `fish-model.json` and its checkpoint.
4. Click **Auto-select contours**. The neural panel shows preprocessing, inference, and evaluation progress. When complete, it closes and the viewer loads the generated contours automatically: green borders in 2D and surface objects in 3D.

Predictions and a new report are saved under the neural panel's output folder, initially `neural_runs/`. The source Imaris files remain read-only. Like the existing pipelines, this processes the first time point at full source resolution, regardless of the displayed preview resolution. It does not train a new model each time.

**Open saved run…** is available below the scan name, above the neural panel's tabs, and near the top of **Results**. Select a run's `report.json` to see loss graphs, per-image results, and saved contours. The [model list](models.md) links the main reports. Select a result row and click **Show selected contours in viewer** to load an existing prediction without recomputing it. **Review selected cell contours…** provides individual acceptance/rejection and confirmed-only export.

The viewer and both processing panels have vertical scrollbars when their contents exceed the window height. Horizontal scrolling is also available when needed. Use the outer scrollbar to reach the bottom of a processing panel; the Setup settings and channel controls also have their own scrollbars. Scrolling over the image keeps its slice/zoom behavior.

## Visible boundaries without roundness limits

**Auto-select contours…** disables both sphericity and elongation limits. It preserves the chosen model's preprocessing and its existing visible-boundary thresholds. For older models without boundary checks, it enables the balanced boundary preset. Cell probability, overlap suppression, physical size limits, image-edge exclusion, membrane intensity/contrast, and boundary-gradient checks still apply. Uniform bright or empty regions cannot pass the boundary-contrast check.

New training configurations also default to **Use visible boundaries · ignore roundness**, with no shape-based curation of complete manual training annotations. In the advanced panel, **Limit roundness and elongation (optional)** can restore explicit limits. Loading an old report or choosing a model in the advanced Setup panel restores its historical settings for reproducibility; use the visible-boundaries preset afterward to change those settings. The per-scan Auto-select workflow always disables the explicit shape limits.

This does not change a saved network's learned shape preferences. The completed balanced model was trained with moderate shape curation, and StarDist still represents cells as star-convex objects. Strongly concave contours may remain difficult. Boundary checks reduce unsupported detections, but cannot establish certainty on all real images; incomplete manual labels do not provide a real-image precision score.

## Collection prepared for future training without shape curation

Choose `artifacts/evidence-fish/audit/audit.json` in **Neural training… → Setup → Choose audit…**. This prepared audit retains all complete annotations after the four requested outlier exclusions and their aliases:

| Split | Scans | Complete cells |
| --- | ---: | ---: |
| Training | 215 | 1,482 |
| Validation | 73 | 564 |
| Test | 71 | 505 |

The acquisition split and preprocessing cache are unchanged. Cut annotations remain evaluation references; unknown image regions remain ignored during training. The audit and settings are ready; a new network has **not** been trained from this audit. The latest completed model is the [balanced run](balanced-training.md).

```bash
# Apply a model with visible-boundary checks and no roundness limits.
.venv/bin/python -m fish_viewer.neural /path/to/scan.ims --mode apply \
  --model /path/to/run/model --evidence-only

# Recheck the actual checkpoint on synthetic empty/partial-cell controls.
.venv/bin/python -m fish_viewer.neural --check-controls \
  --model /path/to/run/model --evidence-only --output /path/to/control-results

# Start a new training run using every complete cell in the prepared audit.
.venv/bin/python -m fish_viewer.neural FISH \
  --audit artifacts/evidence-fish/audit/audit.json \
  --settings artifacts/evidence-fish/audit/settings.json --output neural_runs
```
