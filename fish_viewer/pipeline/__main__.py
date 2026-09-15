import argparse
from dataclasses import replace
import json
from pathlib import Path
import signal
import threading

from ..reader import entry_for_path, scan_library
from .config import PipelineConfig
from .runner import run_pipeline


def main():
    parser = argparse.ArgumentParser(description='Classical 3D cell contour pipeline; no ML model is fitted.')
    parser.add_argument('paths', nargs='+')
    parser.add_argument('--mode', choices=['tune', 'apply'], default='tune')
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--output', default='pipeline_runs')
    parser.add_argument('--settings', type=Path)
    parser.add_argument('--voxel', type=float)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    config = PipelineConfig()
    if args.settings:
        settings = json.loads(args.settings.read_text(encoding='utf-8'))
        if settings.get('version') != 1:
            parser.error('Unsupported settings version.')
        config = PipelineConfig.from_dict(settings['config'])
    if args.voxel is not None:
        config = replace(config, voxel_um=args.voxel).validate()
    entries = []
    for path in args.paths:
        if Path(path).is_dir():
            found, warnings = scan_library(path)
            entries.extend(found)
            for warning in warnings:
                print(warning, flush=True)
        else:
            entries.append(entry_for_path(path))
    cancel = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: cancel.set())
    def emit(event):
        if event['kind'] == 'file':
            print(f'{event["phase"]}: {Path(event["file"]).name} ({event["current"]}/{event["total"]})', flush=True)
        elif event['kind'] == 'trial':
            print(f'Trial {event["trial"]}: Dice={event["scores"]["mean_dice"]:.4f}, loss={event["scores"]["objective"]:.4f}', flush=True)
        elif event['kind'] in ('warning', 'evaluation'):
            print(json.dumps(event), flush=True)
    report = run_pipeline(entries, config, args.output, args.mode, args.trials, seed=args.seed, emit=emit, cancel=cancel)
    print(json.dumps({k:report.get(k) for k in ('status','directory','best_config','evaluation','error')}, indent=2))
    return 0 if report['status'] == 'complete' else 2 if report['status'] == 'cancelled' else 1


if __name__ == '__main__':
    raise SystemExit(main())
