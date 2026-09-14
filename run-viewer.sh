#!/usr/bin/env bash
set -euo pipefail
viewer_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$viewer_root/.venv/bin/python" ]]; then
    echo "Set up the viewer first: python3 -m venv .venv && .venv/bin/python -m pip install -e ." >&2
    exit 1
fi
exec "$viewer_root/.venv/bin/python" "$viewer_root/run_viewer.py" "$@"
