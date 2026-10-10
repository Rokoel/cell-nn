#!/usr/bin/env bash
set -euo pipefail
annotator_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$annotator_root/.venv/bin/python" ]]; then
    echo "Run setup-analysis.sh first." >&2
    exit 1
fi
exec "$annotator_root/.venv/bin/python" "$annotator_root/run_annotator.py" "$@"
