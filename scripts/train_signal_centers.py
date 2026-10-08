"""Train direct-center U-Net or upstream Spotiflow on the same masked corpus."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fish_viewer.spots.config import CenterConfig
from fish_viewer.spots.training import train

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--index', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--engine', choices=['signal-center-unet3d', 'signal-spotiflow3d'], required=True)
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--initial-model', type=Path, help='Recover a finite selected checkpoint, restarting the optimizer.')
parser.add_argument('--learning-rate', type=float, default=.0005)
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
args = parser.parse_args()
try:
    train(args.index, args.output, args.engine, replace(CenterConfig(), epochs=args.epochs, learning_rate=args.learning_rate), args.device,
          lambda e: print(json.dumps(e, allow_nan=False), flush=True), initial_model=args.initial_model)
except Exception:
    path = args.output / 'report.json'
    if path.is_file():
        from fish_viewer.spots.training import write
        report = json.loads(path.read_text())
        report.update(status='failed', error=traceback.format_exc())
        write(path, report)
    raise
