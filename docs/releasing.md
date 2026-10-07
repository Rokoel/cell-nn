# Releases

`.github/workflows/release.yml` builds and publishes releases automatically.
The repository contains the two inference checkpoints under `models/`, so builds
do not depend on the original FISH scans or local training artifact folders.

1. Set the version in `pyproject.toml` and update `docs/release-notes.md`.
2. Commit the changes, tag that commit with `v<version>`, and push the commit/tag.
3. GitHub Actions installs the application on Linux and Windows, runs synthetic
   data/export/geometry/Excel tests, loads both checkpoints and executes CPU
   inference, then builds source/model ZIP bundles for each platform.
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
locally using the installed runtime. `scripts/verify_release.py` verifies both
models without access to FISH or training caches.
