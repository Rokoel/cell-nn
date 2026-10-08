#!/usr/bin/env bash
set -euo pipefail
analysis_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 -m venv "$analysis_root/.venv"
"$analysis_root/.venv/bin/python" -m pip install --upgrade pip
"$analysis_root/.venv/bin/python" -m pip install -e "$analysis_root[neural]"
bash "$analysis_root/setup-spots.sh"
echo "Setup complete. Start run-analysis.sh."
