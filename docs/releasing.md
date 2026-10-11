# Releases

`.github/workflows/release.yml` builds Windows x64 and Linux x64 on their native
GitHub runners, then publishes only after both packaged applications pass.
No FISH scans or local training caches are required or distributed.

The release contains `FISH-Analysis-v<version>-windows-x64.exe` and
`FISH-Analysis-v<version>-linux-x64.AppImage`. Each file contains the GUI,
Python, Qt/VTK, TensorFlow/PyTorch runtimes, Imaris export templates and
all current analysis preset models. Both-detector consensus and the signal-center
3D U-Net remain the defaults. There is no first-run dependency download.
Windows includes the Visual Studio x64 redistributable DLLs app-locally, including
`msvcp140.dll`, `vcruntime140.dll` and `vcruntime140_1.dll`; users do not need the
separate runtime installer. Build manifests record dependency versions and model
hashes, and `SHA256SUMS.txt` covers the assets.

## Running a release

On Windows 10/11 x64, double-click the EXE. On Linux x64 (glibc 2.35 or newer,
including Ubuntu 22.04+), mark the AppImage executable and launch it:

```sh
chmod +x FISH-Analysis-*-linux-x64.AppImage
./FISH-Analysis-v<version>-linux-x64.AppImage
```

Linux still needs a working desktop/OpenGL driver. If FUSE is unavailable, use
`APPIMAGE_EXTRACT_AND_RUN=1 ./FISH-Analysis-v<version>-linux-x64.AppImage`.
The Windows single-file application extracts its bundled libraries to a temporary
folder on launch, so the first startup can take longer. Both formats need free
space for their bundled libraries and analysis outputs. The Linux AppImage bundles
CUDA 12 libraries for both neural backends and supports auto/CPU/GPU inference.
A compatible NVIDIA driver must be installed on the host; no separate CUDA toolkit
or Python installation is needed. Windows uses CPU inference and does not offer
an unsupported GPU choice. Results and settings are saved outside the executable. On its first launch, the
Linux AppImage verifies and unpacks its embedded neural libraries into
`~/.cache/fish-analysis/runtimes/` (or `$XDG_CACHE_HOME/fish-analysis/runtimes/`).
Allow about 8 GB for this cache. Later launches reuse it; no network download is
performed. This compression keeps the complete GPU app below GitHub’s release
asset limit. Triton compilation is excluded because the bundled inference models
use normal TensorFlow/PyTorch execution.

To check actual GPU execution in both bundled backends:

```sh
./FISH-Analysis-v<version>-linux-x64.AppImage --check-runtime --device gpu
```

This executes a TensorFlow 3D convolution and the bundled signal-center U-Net in
separate processes and fails if GPU mode cannot be used. `--device cpu` checks CPU mode.

## Building and publishing

1. Update `pyproject.toml`, `fish_viewer/__init__.py` and `docs/release-notes.md`.
2. Commit, tag with `v<version>`, and push the commit/tag.
3. Actions tests readers, geometry, annotations, Excel and neural workers, loads
   every checkpoint on CPU, and creates a verified source staging bundle.
4. `scripts/build_standalone.py` freezes that bundle using PyInstaller. Windows
   uses one-file mode; Linux wraps a directory build in an AppImage using the
   checksum-pinned upstream appimagetool 1.9.1. Linux first compresses large neural
   libraries into a verified embedded archive; the launcher unpacks it once into
   a versioned cache before starting the neural workers.
5. Actions runs the actual executable from a temporary directory outside the
   checkout, with external Python environment overrides removed. It verifies all
   checkpoints and U-Net/Spotiflow empty-signal controls, then runs the full default
   consensus pipeline on synthetic Imaris data. A non-empty contour/signal case
   checks volumes, A/B distances, Imaris round-trip and single-sheet Excel export.
6. Only after both platforms pass are the binaries and checksums published.

For a trial build, use **Actions → Build and release FISH Analysis → Run workflow**
on the selected branch, enter its version label, and leave **publish** disabled.
An optional **ref** selects a commit/tag instead. Enabling **publish** requires
an existing version tag and publishes the selected build. For a release rebuild,
select that tag as the ref. Only the publish job has `contents: write`.

Local builds require Python 3.12, the application neural dependencies,
`requirements-spots.txt`, `requirements-build.txt`, CPU torch 2.9.1 and torchvision
0.24.1. For Linux CUDA packaging, then install `requirements-cuda-bundle.txt` and
replace torch/torchvision with the exact `2.9.1+cu128` / `0.24.1+cu128` wheels
using `--no-deps` as shown in the
workflow. Pass `--cuda` to `build_standalone.py`. The shared CUDA versions are
pinned: Torch’s exact NCCL dependency conflicts with TensorFlow’s minimum, so
compatibility is checked with actual worker inference rather than pip metadata.
Windows also requires Visual Studio C++ tools with its x64 redistributable
folder. Linux requires the Qt xcb/OpenGL development-host runtime libraries listed
in the workflow. Run `scripts/package_analysis.py --output <new-directory>`, verify
that directory using `scripts/verify_release.py --require-latest-models
--with-direct-signals`, then run `scripts/build_standalone.py --bundle <directory>
--tag v<version>`. The source bundle script remains available for developers.

The frozen executable recognizes the same restricted `-m fish_viewer.<worker>`
commands as the source app's QProcess workers. Workers run separately so the GUI
never loads TensorFlow and PyTorch together. `--verify-models` and `--self-test
--output <new-directory>` expose the packaged checks for troubleshooting.
