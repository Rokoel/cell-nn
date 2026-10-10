import argparse
import os
from pathlib import Path
import sys
import sysconfig


def main():
    if sys.platform.startswith('linux') and not os.environ.get('FISH_CUDA_PATH_READY'):
        libraries=sorted((Path(sysconfig.get_paths()['purelib'])/'nvidia').glob('*/lib'))
        if libraries:
            env=os.environ.copy();env['FISH_CUDA_PATH_READY']='1'
            env['LD_LIBRARY_PATH']=os.pathsep.join(map(str,libraries))+os.pathsep+env.get('LD_LIBRARY_PATH','')
            os.execve(sys.executable,[sys.executable,'-m','fish_viewer.boundary',*sys.argv[1:]],env)
    p=argparse.ArgumentParser(description='Train a sparse-label boundary U-Net.')
    p.add_argument('--annotations',type=Path,default=Path('training_annotations'))
    p.add_argument('--audit',type=Path,default=Path('artifacts/refinement-fish/audit/audit.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--device',choices=['cpu','gpu','auto'],default='auto')
    args=p.parse_args()
    from .training import train
    train(args.annotations,args.audit,args.output,args.epochs,args.device)


if __name__=='__main__': main()
