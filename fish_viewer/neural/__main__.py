import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import signal
import sys
import sysconfig
import threading

import numpy as np

from ..models import LibraryEntry
from ..reader import entry_for_path, scan_library
from .config import NeuralConfig


class Cancellation:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.event = threading.Event()
    def is_set(self):
        return self.event.is_set() or bool(self.path and self.path.exists())


def main():
    # pip CUDA libraries live outside the system loader path. Set the path
    # before importing TensorFlow by restarting only this worker, once.
    if sys.platform.startswith('linux') and not os.environ.get('FISH_CUDA_PATH_READY'):
        libraries=sorted((Path(sysconfig.get_paths()['purelib'])/'nvidia').glob('*/lib'))
        if libraries:
            environment=os.environ.copy()
            environment['FISH_CUDA_PATH_READY']='1'
            environment['LD_LIBRARY_PATH']=os.pathsep.join(map(str,libraries))+os.pathsep+environment.get('LD_LIBRARY_PATH','')
            os.execve(sys.executable,[sys.executable,'-m','fish_viewer.neural',*sys.argv[1:]],environment)
    parser=argparse.ArgumentParser(description='Train or apply StarDist 3D to Imaris stacks.')
    parser.add_argument('paths',nargs='*')
    parser.add_argument('--request',type=Path,help='JSON request written by the GUI')
    parser.add_argument('--output',default='neural_runs')
    parser.add_argument('--mode',choices=['train','apply'],default='train')
    parser.add_argument('--model',type=Path)
    parser.add_argument('--settings',type=Path)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--steps',type=int)
    parser.add_argument('--device',choices=['auto','cpu','gpu'])
    parser.add_argument('--cancel-file',type=Path)
    parser.add_argument('--check-device',action='store_true',help='Check TensorFlow convolution and gradient computation')
    args=parser.parse_args()
    if args.check_device:
        from .engine import runtime
        tf, _, _, info = runtime(args.device or 'auto')
        layer = tf.keras.layers.Conv3D(4,3,padding='same')
        x = tf.ones((1,8,8,8,1))
        with tf.GradientTape() as tape:
            loss=tf.reduce_mean(layer(x)**2)
        gradients=tape.gradient(loss,layer.trainable_variables)
        if not all(bool(tf.reduce_all(tf.math.is_finite(g))) for g in gradients):
            raise RuntimeError('Non-finite convolution gradients.')
        print(json.dumps(dict(info,convolution_device=loss.device,gradient_check='passed')))
        return 0
    config=NeuralConfig()
    entries=[]
    if args.request:
        request=json.loads(args.request.read_text())
        config=NeuralConfig.from_dict(request['config'])
        entries=[LibraryEntry(Path(e['image']),Path(e['scene']) if e.get('scene') else None) for e in request['entries']]
        args.output=request['output'];args.mode=request['mode'];args.model=request.get('model')
        args.cancel_file=request.get('cancel_file')
    else:
        if args.settings:
            settings=json.loads(args.settings.read_text())
            if settings.get('engine')!='stardist3d':parser.error('Not StarDist settings.')
            config=NeuralConfig.from_dict(settings['config'])
        updates={k:v for k,v in dict(epochs=args.epochs,steps_per_epoch=args.steps,device=args.device).items() if v is not None}
        config=replace(config,**updates).validate()
        for path in args.paths:
            entries.extend(scan_library(path)[0] if Path(path).is_dir() else [entry_for_path(path)])
    if not entries:parser.error('Select Imaris files or a folder.')
    os.environ.setdefault('OMP_NUM_THREADS','4')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    cancel=Cancellation(args.cancel_file)
    signal.signal(signal.SIGINT,lambda *_:cancel.event.set())
    # Library progress goes to stderr; stdout is exclusively JSON events.
    output=sys.stdout
    sys.stdout=sys.stderr
    def serialize(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    def emit(event):
        print(json.dumps(event,default=serialize,allow_nan=False),file=output,flush=True)
    from .engine import run_neural
    report=run_neural(entries,config,args.output,args.mode,args.model,emit,cancel)
    return 0 if report['status']=='complete' else 2 if report['status']=='cancelled' else 1


if __name__=='__main__':
    raise SystemExit(main())
