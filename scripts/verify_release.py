"""Load all bundled checkpoints and exercise CPU inference without training data."""
import argparse
import json
import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fish_viewer.analysis.controller import default_models, default_refinement_model, default_proposal_model, check_models
from fish_viewer.neural.engine import runtime
from fish_viewer.signals.engine import network
from fish_viewer.signals.config import SignalConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-latest-models', action='store_true',
                        help='Require the current center/box and cropped contour models.')
    args = parser.parse_args()
    cells,signals=default_models()
    check_models(cells,signals)
    boundary = default_refinement_model()
    proposals = default_proposal_model()
    if args.require_latest_models and not (boundary and proposals):
        parser.error('The release must include models/proposals and models/refinement.')
    provenance=ROOT/'models/origins.json'
    if args.require_latest_models and not provenance.is_file():
        parser.error('The release must include models/origins.json with model checksums.')
    if provenance.exists():
        origins = json.loads(provenance.read_text())
        if args.require_latest_models and not {'cells', 'signals', 'proposals', 'refinement'}.issubset(origins):
            parser.error('Missing model provenance for a required checkpoint.')
        for label,info in origins.items():
            for name,digest in info['sha256'].items():
                with (ROOT/'models'/label/name).open('rb') as stream:
                    assert hashlib.file_digest(stream,'sha256').hexdigest()==digest, f'Model checksum mismatch: {label}/{name}'
    tf,StarDist3D,_,info=runtime('cpu')
    cell_model=StarDist3D(None,name=cells.name,basedir=str(cells.parent))
    manifest=json.loads((cells/'fish-model.json').read_text())
    cell_model.load_weights(manifest['checkpoint'])
    probability,distances=cell_model.predict(np.zeros((16,16,16),np.float32),axes='ZYX',verbose=0)
    assert np.isfinite(probability).all() and np.isfinite(distances).all()
    signal_manifest=json.loads((signals/'signal-model.json').read_text())
    config=SignalConfig(**signal_manifest['config'])
    signal_model=network(tf,config)
    signal_model.load_weights(signals/signal_manifest['checkpoint'])
    score=signal_model(np.zeros((1,config.patch_size,config.patch_size,config.patch_size,3),np.float32),training=False).numpy()
    assert score.shape==(1,1) and np.isfinite(score).all()
    boundary_shapes = {}
    boundary_models = [('refinement', Path(boundary))] if boundary else []
    baseline = ROOT / 'models/refinement-base'
    if (baseline / 'refinement-model.json').is_file():
        boundary_models.append(('refinement-base', baseline))
    for label, folder in boundary_models:
        from fish_viewer.refinement.config import read_model
        from fish_viewer.refinement.config import RefinementConfig
        from fish_viewer.refinement.network import network as boundary_network
        manifest = read_model(folder); config = RefinementConfig(**manifest['config'])
        boundary_model = boundary_network(tf, config)
        boundary_model.load_weights(folder / manifest['checkpoint'])
        output = boundary_model(np.zeros((1,)+(config.patch_size,)*3+(2,),np.float32),training=False).numpy()
        assert output.shape==(1,)+(config.patch_size,)*3+(1,) and np.isfinite(output).all()
        boundary_shapes[label] = list(output.shape)
    proposal_shape = None
    if proposals:
        from fish_viewer.proposals.config import ProposalConfig, read_model as read_proposal_model
        from fish_viewer.proposals.network import network as proposal_network
        manifest=read_proposal_model(proposals);config=ProposalConfig(**manifest['config'])
        detector=proposal_network(tf,config);detector.load_weights(Path(proposals)/manifest['checkpoint'])
        output=detector(np.zeros((1,)+(config.patch_size,)*3+(1,),np.float32),training=False).numpy()
        assert output.shape==(1,)+(config.patch_size//config.stride,)*3+(7,) and np.isfinite(output).all()
        proposal_shape=list(output.shape)
    print(json.dumps(dict(status='complete',runtime=info,cell_probability_shape=list(probability.shape),
                         signal_score=score.tolist(),boundary_logit_shape=boundary_shapes.get('refinement'),
                         boundary_models=boundary_shapes,proposal_shape=proposal_shape)))


if __name__=='__main__':main()
