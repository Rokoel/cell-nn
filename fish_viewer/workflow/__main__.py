"""Lightweight filtering/export worker; neural stages use their existing workers."""
import argparse
import json
import signal
import traceback

from ..neural.__main__ import Cancellation
from ..pipeline.segmentation import Cancelled


def main():
    parser = argparse.ArgumentParser(description='Filter and export cell/signal predictions.')
    parser.add_argument('operation', choices=['filter', 'export'])
    parser.add_argument('--report', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cancel-file')
    args = parser.parse_args()
    cancel = Cancellation(args.cancel_file)
    signal.signal(signal.SIGINT, lambda *_: cancel.event.set())
    def emit(event):
        print(json.dumps(event, allow_nan=False), flush=True)
    try:
        if args.operation == 'filter':
            from .results import filter_result
            report = filter_result(args.report, args.output, emit, cancel)
        else:
            from .export import export_result
            report = export_result(args.report, args.output, emit, cancel)
    except Cancelled:
        report = dict(status='cancelled')
    except Exception:
        report = dict(status='failed', error=traceback.format_exc())
    emit(dict(kind='finished', report=report))
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
