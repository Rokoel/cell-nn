"""Apply an unchanged baseline checkpoint to the same stain-corrected test cache."""
from dataclasses import asdict,replace
from pathlib import Path
import argparse
import copy
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fish_viewer.neural.config import NeuralConfig
from fish_viewer.neural.precision import evidence_preset
from fish_viewer.pipeline.runner import _json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--model',type=Path,default=ROOT/'models/cells-conservative')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--split',default='test',choices=['test','validation'])
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    audit=copy.deepcopy(json.loads(args.audit.read_text()))
    manifest=json.loads((args.model/'fish-model.json').read_text())
    config=replace(evidence_preset(NeuralConfig.from_dict(manifest['config'])),device='cpu',preprocessing='none')
    audit['cases']=[c for c in audit['cases'] if c['audit']['split']==args.split]
    audit['config']=asdict(config);audit['directory']=str(args.output.resolve())
    for i,c in enumerate(audit['cases']):
        source=Path(c['directory']);dest=args.output/f'{i+1:04d}-{Path(c["image"]).stem}';dest.mkdir()
        for old,new in [('evidence.npy','image.npy'),('labels.npy','labels.npy'),('references.npz','references.npz')]:os.link(source/old,dest/new)
        c['directory']=str(dest.resolve())
    path=args.output/'audit.json';_json(path,audit)
    request=args.output/'request.json';_json(request,dict(mode='apply',config=asdict(config),model=str(args.model.resolve()),
        entries=[dict(image=audit['cases'][0]['image'],scene=audit['cases'][0]['scene'])],output=str((args.output/'inference').resolve()),audit=str(path.resolve()),cancel_file=str((args.output/'cancel').resolve())))
    os.execv(sys.executable,[sys.executable,'-m','fish_viewer.neural','--request',str(request)])

if __name__=='__main__':raise SystemExit(main())
