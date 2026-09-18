"""Synthetic negative controls for a trained model and its acceptance policy."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ..models import LibraryEntry
from ..pipeline.segmentation import Prepared
from ..pipeline.runner import _json
from .config import NeuralConfig


def control_images():
    grid=np.indices((64,)*3);radius=np.linalg.norm((grid-31.5)*.3,axis=0)
    rng=np.random.default_rng(1842)
    images=dict(blank=np.zeros((64,)*3,np.float32),constant_dim=np.full((64,)*3,.15,np.float32),
        constant_bright=np.ones((64,)*3,np.float32),
        low_noise=np.maximum(0,rng.normal(0,.01,(64,)*3)).astype(np.float32),
        stronger_noise=np.maximum(0,rng.normal(.05,.05,(64,)*3)).astype(np.float32),
        one_sided_arc=(np.exp(-((radius-3)/.25)**2)*(grid[0]>31.5)).astype(np.float32))
    return images


def evaluate_controls(model,config):
    from .engine import predict
    rows=[]
    for name,image in control_images().items():
        prepared=Prepared(image,(.3,)*3,(0.,)*3,0,[],name,{}, {})
        case=SimpleNamespace(entry=LibraryEntry(Path(name+'.ims')),prepared=prepared,references=[],names=[],reference_cut=[])
        prediction,metrics=predict(model,case,config)
        rows.append(dict(control=name,accepted=len(prediction.cells),proposals=metrics['proposal_count'],
            before_filter=metrics['before_shape_filter']['predicted_count'],rejected=prediction.rejected))
    return dict(controls=rows,
        passed=all(row['accepted']==0 for row in rows),
        note='Synthetic inputs in normalized network units contain no complete cells. This checks only these controlled cases, not real-image precision.')


def check_controls(model_directory,output,device='auto',evidence_only=False):
    from .engine import runtime
    model_directory=Path(model_directory).resolve()
    manifest=json.loads((model_directory/'fish-model.json').read_text())
    config=replace(NeuralConfig.from_dict(manifest['config']),device=device)
    if evidence_only:
        from .precision import evidence_preset
        config=evidence_preset(config)
    _,StarDist3D,_,info=runtime(config.device,config.seed)
    model=StarDist3D(None,name=model_directory.name,basedir=str(model_directory.parent))
    model.load_weights(manifest['checkpoint'])
    report=dict(version=1,model_directory=str(model_directory),runtime=info,config=asdict(config),**evaluate_controls(model,config))
    _json(output,report);return report
