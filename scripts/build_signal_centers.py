"""Build a candidate-independent cell-crop corpus from the frozen FISH split."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fish_viewer.spots.data import build

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--audit', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
build(args.audit, args.output, emit=lambda e: print(json.dumps(e), flush=True))
