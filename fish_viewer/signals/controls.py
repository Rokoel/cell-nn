"""Saved-checkpoint controls with a cell mask present and no signal puncta."""
import json
from pathlib import Path

import numpy as np

from ..neural.engine import runtime
from ..pipeline.runner import _json
from .config import SignalConfig
from .data import candidates,extract_patches,normalized
from .engine import network,predict_patches


def check_controls(model_directory,output,device='auto'):
    model_directory=Path(model_directory).resolve()
    manifest=json.loads((model_directory/'signal-model.json').read_text())
    config=SignalConfig(**manifest['config']);tf,_,_,info=runtime(device,config.seed)
    model=network(tf,config);model.load_weights(model_directory/manifest['checkpoint'])
    spacing=np.array([.15]*3);origin=np.zeros(3)
    radius=np.linalg.norm((np.indices((48,)*3)-23.5)*.15,axis=0)
    cells=(radius<2.5).astype(np.uint32);membrane=np.exp(-((radius-2.5)/.18)**2).astype(np.float32)
    rng=np.random.default_rng(3141)
    images=dict(blank=np.zeros(cells.shape,np.float32),constant=np.full(cells.shape,.5,np.float32),
        noise=np.clip(rng.normal(.1,.025,cells.shape),0,1).astype(np.float32),
        membrane_only=membrane.copy())
    rows=[]
    for name,image in images.items():
        image=normalized(image)
        proposed=[dict(row,channel=1) for row in candidates(image,cells,spacing,origin,config)]
        scores=[]
        for start in range(0,len(proposed),64):
            patches=extract_patches([membrane,image],0,cells,proposed[start:start+64],origin,spacing,config)
            scores.extend(predict_patches(model,patches).tolist())
        rows.append(dict(name=name,candidates=len(proposed),accepted=sum(score>=manifest['threshold'] for score in scores),max_score=max(scores,default=None)))
    passed=all(row['accepted']==0 for row in rows)
    report=dict(version=1,engine='signal-cnn3d-controls',status='complete' if passed else 'failed',passed=passed,
        model_directory=str(model_directory),runtime=info,threshold=manifest['threshold'],controls=rows,
        note='Synthetic controls with cell masks. This does not measure real-image precision.')
    output=Path(output);output.mkdir(parents=True,exist_ok=True);_json(output/'controls.json',report)
    return report
