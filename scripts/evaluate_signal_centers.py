"""Benchmark frozen direct-center models against the existing candidate CNN."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fish_viewer.spots.benchmark import evaluate, controls

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--index', type=Path, required=True)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--split', choices=['validation', 'test'], required=True)
parser.add_argument('--baseline-report', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--device', default='auto', choices=['auto', 'gpu', 'cpu'])
args = parser.parse_args()
evaluate(args.index, args.model, args.split, args.output, args.baseline_report, args.device,
         lambda e: print(json.dumps(e), flush=True))
if args.split == 'test':
    print(json.dumps(controls(args.model, args.output / 'controls', args.device)), flush=True)
