# Classical 3D contour pipeline

[Documentation](README.md) · [Models](models.md) · [Analysis](fish-analysis.md)

The viewer includes a non-ML baseline that finds membrane-like rings, creates 3D cell regions, and keeps approximately spherical cells with moderate elongation. “Training” is a reproducible search over classical image-processing parameters. No neural network or pretrained image recognizer is used.

## GUI workflow

1. Start `./run-viewer.sh FISH` on Linux, or `run-viewer.bat` on Windows. Click **Contour pipeline…** in the toolbar.
2. The six images from `FISH/3abx3ad_101212` are initially selected when that folder exists. Use **Add files…**, **Add folder…** (recursive), or **Viewer library** to change the input. **Current image** selects the open image and switches to application mode.
3. Keep **Tune parameters + evaluate reserved images** and the default 12 trials for the reference experiment. Adjust the physical diameter and shape limits if another experiment contains differently sized cells. Channel selection is always automatic.
4. Click **Run pipeline**. **Live scores** shows tuning loss, best loss so far, Dice/IoU, boundary RMSE, and annotated-cell detection rate. **Preprocessing** shows a central slice before and after background correction, candidate interiors, accepted labels, channel evidence, and stage durations. The status bar names the current file and stage. Previews describe the current file, not a batch average.
5. In **Results**, select an image and click **Show selected contours in viewer**, or double-click its row. Predictions appear in green as mesh/plane intersections in 2D and triangle surfaces in 3D. Switch views with **Tab**. **Manual contours** and **Predicted contours** independently control visibility. Standard channel, slice, opacity, and wireframe controls remain available.
6. The chosen settings populate the setup controls. Use **Apply current settings to selected images** for new files; annotations are optional. Use **Load settings…** to reuse a run's `settings.json`, and **Open saved run…** to reopen its `report.json` and overlays.

The panel can be resized, floated, or docked. Processing runs on a worker thread. **Cancel** stops at the next checkpoint, keeps finished outputs, and records `cancelled`; it may need to finish a file read or compiled image-processing operation first. Closing the viewer requests cancellation and waits for the worker.

## Algorithm

- Reads **time point 1**, at full stored source resolution, using the viewer's calibrated reader. Other time points are not yet processed by this pipeline.
- Scores channels from sampled slices using circular Hough candidates and a bright-annulus/darker-interior test. A 5% bonus for membrane-related metadata breaks close ties; display colors do not drive the choice. Solid puncta fail the interior test. Weak or ambiguous evidence generates a warning; the best-scoring channel is still selected automatically. The evidence score is a heuristic, not a probability or cell count.
- Resamples the selected image onto an approximately isotropic **0.30 µm** grid. A processing-volume target of about 3 million voxels increases spacing for larger fields; actual calibration is saved. This reduces computation at the cost of fine boundary detail. Source arrays for one file still need to fit in RAM.
- Subtracts a broad background estimate independently within each XY section, normalizes intensity, and applies Gaussian smoothing.
- Uses 3D distance peaks in candidate interiors as seeds for a watershed over boundary intensity. A second classical option closes/fills XY shells and applies a distance watershed. The seed search and final labeling are volumetric.
- Filters candidates by equivalent spherical diameter, physical covariance axis ratio, and surface-based sphericity. Defaults are **2.5–8.0 µm** diameter, **axis ratio ≤ 1.8**, and **sphericity ≥ 0.62**. These are adjustable biological assumptions. Cells touching the volume frame are excluded by default, including cells adjacent to the artificial background frame used by boundary watershed.
- Produces separate integer object labels and calibrated triangle meshes. Sphericity is `π^(1/3) × (6V)^(2/3) / A`; voxelized shape estimates are approximate. Axis ratio uses square roots of the largest/smallest covariance eigenvalues in physical coordinates.

The circular channel heuristic assumes roughly similar X/Y pixel spacing and membrane-like contrast. It can select an unsuitable channel on unfamiliar stains or poor images; inspect its scores and preview. Reference contours are not used to choose the channel.

## Tuning and evaluation

Tuning needs annotated images from **at least two acquisition groups**. Groups use the containing directory and a leading 6–8 digit acquisition prefix, such as `101212_…`; otherwise each image name defines a group. Confirm this convention matches your experiment. Exact source-pixel duplicates are skipped. Images from the final sorted group(s), approximately 20% of groups by default, are reserved.

Each trial segments only the tuning images. The best trial minimizes `1 − mean Dice`, with equal weight per annotated image. The first trial uses the setup values, the second tries the filled-interior alternative, and subsequent trials explore bounded boundary thresholds, smoothing, and seed separations using seed 42. Geometry limits stay fixed. Reserved-image scores appear **once, after parameter selection**. Unannotated images receive predictions but no scores.

This is one acquisition split, not cross-validation. An acquisition date is only a proxy for independence; images of the same specimen may be related across dates. Preprocessing is independently computed per image without fitting cross-image statistics. It can be cached before the split without using reserved annotations to select parameters. Further algorithm development after inspecting evaluation results requires a fresh evaluation set for an unbiased final estimate.

Manual annotations are incomplete. Saved surface meshes are voxelized in calibrated coordinates as reference objects. Unmeshed editing curves and A/B measurement lines do not define closed reference volumes. Missing or too-small reference volumes are reported. Absence of an annotation does not imply empty background.

| Score | Interpretation |
| --- | --- |
| **Dice / IoU** | Objects are assigned one-to-one by maximum total full-object IoU; assignments below IoU 0.1 are discarded. Missed references contribute **zero**. |
| **Loss** | `1 − mean Dice`; lower is better. Overall scores weight annotated images equally. |
| **Detected refs** | Fraction of annotated cells with a match at **IoU ≥ 0.5**. This is annotation recall, not overall accuracy. |
| **Boundary RMSE, µm** | Symmetric root mean square nearest boundary-voxel distance in physical units. Reported **only for matched cells**, then averaged. Missed cells have no fabricated distance. |
| **Boundary HD95, µm** | 95th percentile of symmetric nearest-boundary distances for matched cells; saved in JSON and per-cell CSV. |
| **Pair mask MSE** | Binary disagreement within a padded bounding box containing both matched objects. Saved in JSON/CSV; conditional on a match. Whole-image MSE would be dominated by empty background. |
| **Unscored predictions** | Objects without a reference match. Their correctness is unknown. Partial annotations cannot establish precision or a false-positive rate. |

A low conditional boundary error with low recall is not a good overall result. Predictions remain candidates for review and are never promoted to manual truth.

## Outputs and command line

Every run creates a timestamped directory under `pipeline_runs/`, or the chosen folder:

- `report.json`: status, split, trial scores, selected settings, per-cell evaluation, failures, warnings, paths, and timing.
- `settings.json`: selected parameters, reusable in the GUI and CLI.
- `trials.csv`, `cells-evaluation.csv`: trial and individual annotated-cell scores.
- `scores.png`, `scores.svg`: standalone graphs for export.
- Per-image `preprocessing.json` and `preprocessed.npy`: channel evidence, calibration, source fingerprint, warnings, and a memory-mapped image cache.
- Per-image `contours.npz`: integer ZYX labels and JSON metadata with calibration, settings, source identity, shape measurements, and scores. Load using `numpy.load(..., allow_pickle=False)`.
- Per-image `contours.cells.csv`: volume, equivalent diameter, sphericity, elongation, and XYZ center of every predicted cell.

Predictions are separate viewer-compatible archives; they are **not written into `.ims` or `.imx` files**. Source files remain read-only. Overlay loading checks source identity and full-resolution image pixels. Reports store absolute paths: update those paths if moving a run or dataset to another computer. Completed overlays and reports do not need `preprocessed.npy`; the cache can be removed after processing.

```bash
# Tune and reserve acquisition groups for final scoring.
.venv/bin/python -m fish_viewer.pipeline FISH/3abx3ad_101212 --trials 12

# Apply saved settings recursively to another folder.
.venv/bin/python -m fish_viewer.pipeline /path/to/images --mode apply --settings /path/to/run/settings.json

# Process one image or companion scene without tuning.
.venv/bin/python -m fish_viewer.pipeline /path/to/image.imx --mode apply
```

On Windows substitute `.venv\Scripts\python.exe`. No GPU or Imaris installation is required. Native Windows execution has not been tested in this Linux environment.

## Reference experiment

The reference experiment uses four annotated tuning images from `101212` and `131212`; `131212…CG488c.ims` is unannotated; `141212…CG488b.ims` supplies ten evaluation cells. All six images automatically selected channel 3, the 633 nm channel.

The historical 12-trial run and its generated graphs/settings were removed. The following numbers document that experiment, rather than a currently bundled result. With conservative edge exclusion enabled, the evaluation image has **mean Dice 0.072**, **IoU 0.057**, and **1/10 annotated cells detected**. Matched-cell boundary RMSE is **0.611 µm**, based on that one match. There are 143 predicted candidates; 142 are unscored because no matching reference exists. This is poor annotation recall and is not ready for unattended contour production.

All saved reference cells are scored, including cells near the stack boundary that the conservative filter deliberately excludes. Several training annotations touch or approach the first/last optical section. Disabling **Exclude cells cut by image edges** admits more candidates, including partial cells. Inspect that tradeoff before interpreting overall recall. The earlier exploratory implementation had a less strict edge check; the linked run uses the corrected implementation.

This is an initial classical baseline. The saved scores document its limitations; broader validation needs additional independently annotated images. Examine missed cells and improve image-processing assumptions before comparing a future ML model with the same evaluation rules.

Algorithm references: [scikit-image watershed](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html), [circular Hough transform](https://scikit-image.org/docs/stable/auto_examples/edges/plot_circular_elliptical_hough_transform.html), and [SciPy calibrated distance transform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html).
