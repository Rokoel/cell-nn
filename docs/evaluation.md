# Reading the evaluation reports

[Documentation](README.md) · [Model inventory](models.md)

Reports are grouped by experiment under `reports/`. Each experiment has its own
checkpoint, acceptance settings, matching rules and denominator. A score from
one experiment does not describe every current Analysis preset.

## Where to start

| Question | Report |
|---|---|
| How did the signal detectors compare with fixed cell masks? | [Signal centers](../reports/signal-centers/README.md) |
| What changed with corrected channel roles and consensus? | [Correct-channel consensus](../reports/consensus-correct-channels/README.md) |
| How did the former center/box default perform across held-out scans? | [All-held-out comparison](../reports/all-heldout-comparison/README.md) |
| What did crop contours and boundary refinement change? | [Center/box](../reports/center-box-contours/README.md), [boundary refinement](../reports/boundary-refinement/README.md) |
| Did surface losses reduce volume/boundary bias? | [Contour bias](../reports/contour-bias/README.md) |
| Did preprocessing or linked 2D contours help? | [Preprocessing](../reports/preprocessed-stardist/README.md), [StarDist 2D](../reports/stardist2d/README.md) |
| What are the historical physical measurement errors? | [Measurement accuracy](measurement-accuracy.md) |

## Signal comparison

![Fixed-mask signal model comparison](images/signal-models.png)

This plot uses the **test** rows from
[`comparison.json`](../reports/signal-centers/comparison.json): 228 manual signal
references, fixed cell masks, and the experiment's cell/channel matching rules.
Precision and recall describe eligible annotated regions. Unmarked regions are
unknown and cannot establish whole-scan precision. This is a detector comparison,
not an end-to-end score for the current default consensus pipeline.

## Interpret the denominators

- **Cell overlap:** complete manual references are matched one-to-one. Cut cells
  and unmatched detections have different scoring roles; inspect the protocol.
- **Volume and boundary error:** often reported only for matched cells. Better
  error on fewer matches may reflect a changed cohort.
- **Signal localization:** requires the same predicted cell and inferred channel,
  within the declared matching radius.
- **A–B distance:** the shortest physical distance from a signal center to its
  cell's triangle surface. It depends on both signal location and cell geometry.
- **Detection totals:** incomplete annotation means extra cells are often unscored,
  rather than proven false positives.

Synthetic empty controls test specific failure modes. They do not replace
reviewing real-image predictions. Training curves document optimization and
checkpoint selection; use separate held-out comparisons to assess generalization.

The JSON/CSV tables, Excel workbooks and figures bundled with the release are
preserved. Historical generated run folders were removed; see
[training-data reconstruction](training-data.md) for the retained metadata.
