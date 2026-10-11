"""Signal worker entry point; stdout contains JSON progress events."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import signal
import sys

from ..worker_runtime import Cancellation, restart_with_cuda
from .config import SignalConfig


def main():
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument('--model', type=Path)
    early.add_argument('--mode')
    known, _ = early.parse_known_args()
    if known.model and known.mode in ('apply', 'controls', 'scout'):
        from ..spots.runtime import DIRECT_ENGINES, dispatch
        manifest = json.loads((known.model / 'signal-model.json').read_text())
        if manifest.get('engine') in DIRECT_ENGINES:
            dispatch(sys.argv[1:])
    restart_with_cuda('fish_viewer.signals')
    parser=argparse.ArgumentParser(description='Train or apply a cell-restricted 3D signal detector.')
    parser.add_argument('--mode',choices=['audit','train','apply','controls'],required=True)
    parser.add_argument('--cell-report',type=Path)
    parser.add_argument('--collection-audit',type=Path)
    parser.add_argument('--audit',type=Path)
    parser.add_argument('--output',type=Path,default=Path('signal_runs'))
    parser.add_argument('--model',type=Path)
    parser.add_argument('--image',type=Path)
    parser.add_argument('--cell-predictions',type=Path)
    parser.add_argument('--signal-channels',nargs='+',type=int)
    parser.add_argument('--scene',type=Path)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--steps',type=int)
    parser.add_argument('--device',choices=['auto','cpu','gpu'],default='auto')
    parser.add_argument('--cancel-file',type=Path)
    parser.add_argument('--config',type=Path,help='Signal configuration JSON (for auditing/training).')
    parser.add_argument('--initial-model',type=Path,help='Fine-tune a compatible saved signal checkpoint.')
    parser.add_argument('--split',choices=['training','validation','test'],help='Audit only this acquisition split.')
    args=parser.parse_args()
    config=SignalConfig(**json.loads(args.config.read_text())) if args.config else SignalConfig()
    config=replace(config,device=args.device)
    if args.epochs is not None:config=replace(config,epochs=args.epochs)
    if args.steps is not None:config=replace(config,steps_per_epoch=args.steps)
    config.validate()
    cancel=Cancellation(args.cancel_file);signal.signal(signal.SIGINT,lambda *_:cancel.event.set())
    stream=sys.stdout;sys.stdout=sys.stderr
    def emit(event):print(json.dumps(event,allow_nan=False),file=stream,flush=True)
    if args.mode=='audit':
        if not args.cell_report or not args.collection_audit:parser.error('Audit requires --cell-report and --collection-audit.')
        from .data import build_audit
        report=build_audit(args.cell_report,args.collection_audit,args.output,config,emit,cancel,split=args.split)
    elif args.mode=='controls':
        if not args.model:parser.error('Controls require --model.')
        from .controls import check_controls
        report=check_controls(args.model,args.output,args.device);emit(dict(kind='finished',report=report))
    else:
        from .engine import train,apply
        if args.mode=='train':
            if not args.audit:parser.error('Training requires --audit.')
            report=train(args.audit,args.output,config,emit,cancel,initial_model=args.initial_model)
        else:
            if not args.model or not args.image or not args.cell_predictions:parser.error('Application requires --model, --image and --cell-predictions.')
            report=apply(args.image,args.scene,args.cell_predictions,args.model,args.output,config.device,emit,cancel,args.signal_channels)
    return 0 if report['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
