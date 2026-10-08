# Releases

`.github/workflows/release.yml` builds and publishes releases automatically.
The repository contains the inference checkpoints under `models/`, so builds
do not depend on the original FISH scans or local training artifact folders.

1. Set the version in `pyproject.toml` and update `docs/release-notes.md`.
2. Commit the changes, tag that commit with `v<version>`, and push the commit/tag.
3. GitHub Actions installs the application and a separate CPU PyTorch runtime on Linux and Windows, runs synthetic
   data/export/geometry/Excel tests, loads the bundled checkpoints and executes CPU
   inference, then builds source/model ZIP bundles for each platform. It verifies
   checksums and CPU inference again from each packaged copy, requires the latest
   center/box, crop and default signal U-Net models, and tests that unknown detector voxels have zero
   loss gradient.
4. Only after both builds succeed does the publish job create a GitHub release
   with both archives and `SHA256SUMS.txt`.

For an existing tag, use **Actions → Build and release FISH Analysis → Run
workflow**, entering that tag. Re-running replaces the assets for that tag after
verification; it does not retrain or pick different models. The release token is
GitHub's built-in `GITHUB_TOKEN`; only the publish job has `contents: write`.

These are Python source/model distributions, not standalone executables. They
include setup and launch scripts for both operating systems. Python 3.12 and an
internet connection for dependency installation are required on first setup.
Native Windows uses CPU inference. Linux can install the optional GPU extra.

`scripts/package_analysis.py --output <new-directory>` creates the same bundle
locally using the installed runtime. `scripts/verify_release.py` verifies all
bundled model checksums and CPU inference without access to FISH or training
caches. Pass `--require-latest-models` to reject a bundle missing the center/box
detector or crop U-Net. Both Viewer and Analysis launchers, release notes, and the
latest evaluation reports are included in each archive.

The workflow runs isolated U-Net/Spotiflow tests and verifies both direct-center
checkpoints with `--with-direct-signals`. Default U-Net verification is mandatory
even without that flag. No FISH scans are distributed.
