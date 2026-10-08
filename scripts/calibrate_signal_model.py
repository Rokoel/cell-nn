"""Freeze an exact validation score cutoff without changing signal CNN weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import sysconfig

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from fish_viewer.neural.engine import runtime
from fish_viewer.signals.config import SignalConfig
from fish_viewer.signals.engine import network,predict_patches
from fish_viewer.signals.evaluation import select_threshold
from fish_viewer.pipeline.runner import _json


def main():
    if sys.platform.startswith('linux') and not os.environ.get('FISH_CUDA_PATH_READY'):
        libraries=sorted((Path(sysconfig.get_paths()['purelib'])/'nvidia').glob('*/lib'))
        if libraries:
            env=os.environ.copy();env['FISH_CUDA_PATH_READY']='1'
            env['LD_LIBRARY_PATH']=os.pathsep.join(map(str,libraries))+os.pathsep+env.get('LD_LIBRARY_PATH','')
            os.execve(sys.executable,[sys.executable,__file__,*sys.argv[1:]],env)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['auto','cpu','gpu'],default='auto')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    audit=json.loads(args.audit.read_text());manifest=json.loads((args.model/'signal-model.json').read_text())
    if audit['status']!='complete' or audit['engine']!='signal-audit':raise ValueError('Completed signal audit required.')
    digest=hashlib.sha256(args.audit.read_bytes()).hexdigest()
    expected=manifest.get('training_provenance',{}).get('audit_sha256')
    if expected and expected!=digest:raise ValueError('Calibration audit differs from the training audit.')
    config=SignalConfig(**manifest['config']).validate()
    cases=[c for c in audit['cases'] if c['split']=='validation']
    tf,_,_,info=runtime(args.device,config.seed);model=network(tf,config)
    model.load_weights(args.model/manifest['checkpoint']);probabilities=[]
    report=dict(status='running',split='validation',runtime=info,source_model=str(args.model.resolve()),
                audit=str(args.audit.resolve()),audit_sha256=digest)
    for i,case in enumerate(cases):
        probabilities.append(predict_patches(model,np.load(Path(case['directory'])/'patches.npy',mmap_mode='r')))
        report['progress']=dict(phase='Calibrating signal cutoff on validation scans',current=i+1,total=len(cases))
        _json(args.output/'report.json',report)
    selection=select_threshold(cases,probabilities,config)
    target=args.output/'model';target.mkdir()
    shutil.copy2(args.model/manifest['checkpoint'],target/manifest['checkpoint'])
    manifest['threshold']=selection['selected']['threshold'];manifest['selection']=selection
    manifest.setdefault('training_provenance',{})['cutoff_calibration']=dict(
        source_model=str(args.model.resolve()),audit_sha256=digest,split='validation',
        policy=selection['policy'],weights_unchanged=True)
    _json(target/'signal-model.json',manifest)
    report.update(status='complete',selection=selection,model_directory=str(target.resolve()),
                  checkpoint_sha256=hashlib.sha256((target/manifest['checkpoint']).read_bytes()).hexdigest())
    _json(args.output/'report.json',report)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig=Figure(figsize=(7,5),layout='constrained');FigureCanvasAgg(fig);ax=fig.subplots()
    rows=selection['candidates']
    ax.plot([r['metrics']['recall'] for r in rows],[r['metrics']['precision'] for r in rows],label='Validation scores')
    selected=selection['selected']['metrics'];ax.scatter([selected['recall']],[selected['precision']],color='red',label='Selected cutoff')
    ax.axhline(config.target_precision,color='gray',linestyle='--',label='Precision target')
    ax.set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1),title='Signal cutoff calibration');ax.grid(alpha=.2);ax.legend()
    fig.savefig(args.output/'calibration.png',dpi=150)
    print(json.dumps(dict(status='complete',model=str(target.resolve()),target_met=selection['target_met'],selected=selection['selected'])),flush=True)


if __name__=='__main__':main()
