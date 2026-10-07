FISH Analysis now exports an Excel workbook automatically for every processed
scan. **Export Excel…** saves a workbook from the displayed completed analysis,
including older saved analyses. It includes cell volumes (µm³), per-channel
signal counts, signal centers A, closest border points B, confidence scores,
and 3D signal-to-border distances (µm). Cell volumes are also shown in the app.

Both existing trained neural models are bundled. All non-membrane channels are
searched independently; cells with at least one accepted signal in any channel
are retained. Roundness filtering remains disabled.

Run folders are created atomically with unique names, preventing collisions
when jobs start within the same clock tick on Windows.

Download the ZIP for your operating system, extract it, install 64-bit Python
3.12, then run `setup-analysis.bat` (Windows) or `bash setup-analysis.sh` (Linux).
Start with `run-analysis.bat` or `bash run-analysis.sh`. Setup requires internet
to install dependencies. These bundles contain Python source and checkpoints,
not standalone executables. Windows inference uses CPU; Linux supports the
optional CUDA extra. The original FISH dataset is not needed to run the app.

GitHub Actions tests both platform builds and verifies both bundled checkpoints
before publishing. The held-out measurement benchmark and its limitations are
documented in `docs/measurement-accuracy.md`.
