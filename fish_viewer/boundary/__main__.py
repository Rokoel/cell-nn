from ..worker_runtime import restart_with_cuda
from ..project_paths import annotation_audit

import argparse
from pathlib import Path


def main():
    restart_with_cuda('fish_viewer.boundary')
    p=argparse.ArgumentParser(description='Train a sparse-label boundary U-Net.')
    p.add_argument('--annotations',type=Path,default=Path('training_annotations'))
    p.add_argument('--audit',type=Path,default=annotation_audit())
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--device',choices=['cpu','gpu','auto'],default='auto')
    args=p.parse_args()
    from .training import train
    train(args.annotations,args.audit,args.output,args.epochs,args.device)


if __name__=='__main__': main()
