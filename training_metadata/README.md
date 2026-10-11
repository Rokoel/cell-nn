# Local training reconstruction metadata

This archive retains audit metadata, acquisition splits, exclusions, preprocessing
recipes, patch indexes, label-correction records and training provenance after
generated arrays and historical predictions were removed.

`inventory.json` maps original artifact paths to archived files and records their
SHA-256 hashes. Files remain verbatim; historical absolute paths inside them do
not indicate that caches or old runs still exist. The data files are local and
excluded from Git. Original scans and painted annotations remain in `FISH/` and
`training_annotations/`.

See [training data and reconstruction](../docs/training-data.md) for rebuild order
and commands. Inference uses the retained `models/` and working environments;
it does not need these training arrays. The annotator can use the archived
acquisition metadata directly.
