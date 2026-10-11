"""Fine-tune contour objectives on the unchanged complete FISH training split."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

from fish_viewer.worker_runtime import restart_with_cuda

def main():
    restart_with_cuda()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial-model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--patches',type=Path,default=ROOT/'artifacts/refinement-fish/patches')
    p.add_argument('--surface-weight',type=float,default=.1);p.add_argument('--beta',type=float,default=.5)
    p.add_argument('--epochs',type=int,default=8);p.add_argument('--size',type=int,default=64)
    p.add_argument('--margin',type=float,default=2);p.add_argument('--threshold',type=float,default=.5)
    p.add_argument('--residual',action='store_true')
    args=p.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    from fish_viewer.refinement.config import RefinementConfig,read_model
    from fish_viewer.refinement.training import train
    config=replace(RefinementConfig(**read_model(args.initial_model)['config']),epochs=args.epochs,patience=3,
        learning_rate=.00005,surface_loss_weight=args.surface_weight,tversky_beta=args.beta,
        residual_correction=args.residual,patch_size=args.size,margin_um=args.margin,threshold=args.threshold)
    parent_audit=Path(json.loads((ROOT/'artifacts/refinement-fish/audit/audit.json').read_text())['parent_audit'])
    report=train(parent_audit,ROOT/'artifacts/refinement-fish/audit',
        args.patches,args.output,config,'gpu',lambda e:print(json.dumps(e,allow_nan=False),flush=True),initial_model=args.initial_model)
    if report['status']!='complete':raise RuntimeError(report.get('error',report['status']))
    print(json.dumps(dict(model=report['model_directory'],report=report['directory']+'/report.json')),flush=True)
if __name__=='__main__':main()
