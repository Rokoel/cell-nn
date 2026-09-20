#!/usr/bin/env bash
set -euo pipefail
analysis_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$analysis_root/.venv/bin/python" ]]; then
    echo "Run setup-analysis.sh first." >&2
    exit 1
fi
exec "$analysis_root/.venv/bin/python" "$analysis_root/run_analysis.py" "$@"
