# Imaris export compatibility

[Documentation](README.md) · [Models](models.md) · [Analysis](fish-analysis.md)

Use **Scan workflow / Export Imaris… → Export current result as Imaris…** after
filtering cells without signals. Keep the exported `.ims` and `.imx` together and
open the **`.imx` in Imaris** to load the image, contours, and signal centers.
Opening the `.ims` alone loads the image. The CSV tables retain counts and scores.

The exporter now uses a complete Imaris 6.3.1 scene template taken from
`FISH/3abx3ad_101212/141212_3ab3ad_Lam633m_dsRed543_CG488b.imx`. The template is
packaged with the application; exports do not require access to the FISH folder.

## What was corrected

The previous exporter generated a minimal XML scene that our own reader could
open, but omitted native surface tracking properties, materials, creation and
filter settings, `timeInfo`, frame settings, and application state. Its numeric
mesh blocks were larger than those in the native examples. It also used a
`LineMode` value absent from every measurement object in this collection.
These were compatibility defects; the exact cause of the reported Imaris crash
has not been reproduced in Imaris here.

The replacement preserves the native field hierarchy and ordering, including
the per-cell component groups. Surface timestamps come from the source image.
Meshes use at most 400 records per XML block, 40 quoted lines per block, and
10 records per line, matching the native files. Each record is six numbers for
a vertex/normal or three triangle indices. Each signal has its own measurement
object using the native `Polygon` mode, so unrelated signals are not connected.
Signal-only reports contain one point. Measured results from **FISH Analysis**
contain `A` and `B` with distance labels enabled. Component IDs are unique. The image filename remains relative for
portability. XML has native-style CRLF line endings and no added declaration or
comments. A provenance string is stored in the existing surface history field.

The `.ims` is copied byte for byte and its SHA-256 is verified. All image channels,
time points, resolutions, metadata, compression, and calibration remain unchanged.
The scene thumbnail comes from the source image, avoiding the original scene's
annotated screenshot. Original files are never modified and existing outputs are
never overwritten. Re-export old results to obtain the corrected scene format.

## Create an annotation-free test copy

Historical test exports were removed during cleanup. The original source scan
and packaged scene template remain. Generate a new pair in a fresh output folder:

```bash
.venv/bin/python -m fish_viewer.workflow unannotated \
  --image 'FISH/3abx3ad_101212/141212_3ab3ad_Lam633m_dsRed543_CG488b.ims' \
  --output 'artifacts/my-test/unannotated.ims'
```

For a processed export, use the Analysis app or Viewer workflow and retain the
`.ims`/`.imx` pair. Source-file and template regression checks live in
[`test_imaris_compatibility.py`](../tests/test_imaris_compatibility.py).

## Validation scope

The collection audit found 381 HDF5 images containing only `DataSet`, `DataSetInfo`,
and `Thumbnail`, plus 11 legacy TIFF images. None has an embedded scene. The 396
scene companions comprise 385 Imaris 6.3.1 scenes and 11 Imaris 5.0.1 scenes.
All 3,455 Imaris 6 measurement objects use `Polygon` line mode.

Regression tests compare the generated hierarchy, attribute names, and field order
directly to the original Imaris-written example, verify mesh block boundaries,
coordinates, normals, triangle indices, timestamps, image hashes, relocation,
and annotation removal. They also reject malformed scenes that our permissive
display reader could otherwise accept. The packaged wheel includes the template.

This is an example-derived compatibility profile, not an official vendor schema.
Imaris itself is not available in this environment, so opening the files in Imaris
has **not** been verified. Current exports use the FISH 6.3.1 scene format; opening
them in Imaris 5 is not a supported target. Embedded `Scene`/`Scene8` XML and recognized HDF5 mesh/spot layouts are read.
Export removes stale scene groups from a copied image and writes the current
annotations to the companion `.imx`. Image datasets remain unchanged, although
the whole-file checksum changes when embedded scenes are removed. Unsupported
scene objects are reported and block export to avoid silent annotation loss.
