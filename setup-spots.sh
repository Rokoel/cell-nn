#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv-spots
.venv-spots/bin/python -m pip install --upgrade pip
# CPU works on all supported machines. For a compatible NVIDIA GPU, set
# FISH_TORCH_INDEX=https://download.pytorch.org/whl/cu128.
.venv-spots/bin/python -m pip install 'torch==2.9.1' 'torchvision==0.24.1' --index-url "${FISH_TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
.venv-spots/bin/python -m pip install -r requirements-spots.txt
