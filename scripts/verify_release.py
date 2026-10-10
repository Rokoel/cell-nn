"""Load all bundled checkpoints and exercise CPU inference without training data."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import sys
import subprocess

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
    parser.add_argument('--with-direct-signals', action='store_true',
                        help='Also require the optional PyTorch runtime and test direct-center checkpoints.')
    args = parser.parse_args()
    cells,signals=default_models()
    check_models(cells,signals)
    boundary = default_refinement_model()
    proposals = default_proposal_model()
    if args.require_latest_models and not (boundary and proposals):
        parser.error('The release must include the current cell and contour models.')
    if args.require_latest_models and Path(boundary).name != 'refinement-bias-candidate':
        parser.error('The release must include models/refinement-bias-candidate for optional crop contours.')
    provenance=ROOT/'models/origins.json'
    if args.require_latest_models and not provenance.is_file():
        parser.error('The release must include models/origins.json with model checksums.')
    if provenance.exists():
        origins = json.loads(provenance.read_text())
        if args.require_latest_models and not {'cells', 'cells-conservative', 'cells-sensitive', 'cells-preprocessed', 'cells-stardist2d', 'signals', 'proposals', 'refinement', 'refinement-bias-candidate', 'signals-center-unet', 'boundary-annotations'}.issubset(origins):
            parser.error('Missing model provenance for a required checkpoint.')
        for label,info in origins.items():
            for name,digest in info['sha256'].items():
                with (ROOT/'models'/label/name).open('rb') as stream:
                    assert hashlib.file_digest(stream,'sha256').hexdigest()==digest, f'Model checksum mismatch: {label}/{name}'
    tf,StarDist3D,_,info=runtime('cpu')
    manifest=json.loads((cells/'fish-model.json').read_text())
    cell_model=StarDist3D(None,name=cells.name,basedir=str(cells.parent))
    cell_model.load_weights(manifest['checkpoint'])
    probability,distances=cell_model.predict(np.zeros((16,16,16),np.float32),axes='ZYX',verbose=0)
    assert np.isfinite(probability).all() and np.isfinite(distances).all()
    conservative=ROOT/'models/cells-conservative'
    old=StarDist3D(None,name=conservative.name,basedir=str(conservative.parent))
    old.load_weights('best.weights.h5')
    p,d=old.predict(np.zeros((16,16,16),np.float32),axes='ZYX',verbose=0)
    assert np.isfinite(p).all() and np.isfinite(d).all()
    balanced = ROOT / 'models/cells'
    balanced_model = StarDist3D(None, name=balanced.name, basedir=str(balanced.parent))
    balanced_manifest = json.loads((balanced/'fish-model.json').read_text())
    balanced_model.load_weights(balanced/balanced_manifest['checkpoint'])
    balanced_probability, balanced_distances = balanced_model.predict(np.zeros((16,16,16),np.float32),axes='ZYX',verbose=0)
    assert np.isfinite(balanced_probability).all() and np.isfinite(balanced_distances).all()
    sensitive = ROOT / 'models/cells-sensitive'
    sensitive_model = StarDist3D(None, name=sensitive.name, basedir=str(sensitive.parent))
    sensitive_manifest = json.loads((sensitive/'fish-model.json').read_text())
    sensitive_model.load_weights(sensitive/sensitive_manifest['checkpoint'])
    sensitive_probability, sensitive_distances = sensitive_model.predict(np.zeros((16,16,16),np.float32),axes='ZYX',verbose=0)
    assert np.isfinite(sensitive_probability).all() and np.isfinite(sensitive_distances).all()
    enhanced = ROOT / 'models/cells-preprocessed'
    if (enhanced / 'fish-model.json').is_file():
        check_models(enhanced,signals)
        enhanced_manifest=json.loads((enhanced/'fish-model.json').read_text())
        enhanced_model=StarDist3D(None,name=enhanced.name,basedir=str(enhanced.parent))
        enhanced_model.load_weights(enhanced/enhanced_manifest['checkpoint'])
        from fish_viewer.neural.preprocessing import membrane_connected
        enhanced_probability,enhanced_distances=enhanced_model.predict(membrane_connected(np.zeros((16,)*3,np.float32),(.3,)*3),axes='ZYX',verbose=0)
        assert np.isfinite(enhanced_probability).all() and np.isfinite(enhanced_distances).all()
    sliced=ROOT/'models/cells-stardist2d'
    if (sliced/'fish-model.json').is_file():
        manifest=check_models(sliced,signals)
        from stardist.models import StarDist2D
        sliced_model=StarDist2D(None,name=sliced.name,basedir=str(sliced.parent))
        sliced_model.load_weights(sliced/manifest['checkpoint'])
        outputs=sliced_model.keras_model.predict_on_batch(np.zeros((1,32,32,1),np.float32))
        assert all(np.isfinite(a).all() for a in outputs)
    from fish_viewer.boundary.model import network as learned_network, read_model as read_learned_model
    learned_folder = ROOT/'models/boundary-annotations'
    learned_manifest = read_learned_model(learned_folder)
    learned = learned_network(tf)
    learned.load_weights(learned_folder/learned_manifest['checkpoint'])
    learned_output = learned(np.zeros((1,32,40,1),np.float32),training=False).numpy()
    assert learned_output.shape == (1,32,40,1) and np.isfinite(learned_output).all()
    default_signal_manifest=json.loads((signals/'signal-model.json').read_text())
    legacy_signals=ROOT/'models/signals'
    signal_manifest=json.loads((legacy_signals/'signal-model.json').read_text())
    config=SignalConfig(**signal_manifest['config'])
    signal_model=network(tf,config)
    signal_model.load_weights(legacy_signals/signal_manifest['checkpoint'])
    score=signal_model(np.zeros((1,config.patch_size,config.patch_size,config.patch_size,3),np.float32),training=False).numpy()
    assert score.shape==(1,1) and np.isfinite(score).all()
    signal_shapes={'signals':list(score.shape)}
    for label in ('signals-base','signals-refreshed'):
        baseline_signals=ROOT/'models'/label
        if not (baseline_signals/'signal-model.json').is_file():continue
        manifest=json.loads((baseline_signals/'signal-model.json').read_text())
        baseline_config=SignalConfig(**manifest['config']).validate()
        baseline_network=network(tf,baseline_config)
        baseline_network.load_weights(baseline_signals/manifest['checkpoint'])
        output=baseline_network(np.zeros((1,)+(baseline_config.patch_size,)*3+(3,),np.float32),training=False).numpy()
        assert output.shape==(1,1) and np.isfinite(output).all()
        signal_shapes[label]=list(output.shape)
    boundary_shapes = {}
    boundary_models = [(Path(boundary).name, Path(boundary))] if boundary else []
    for label in ('refinement', 'refinement-base', 'refinement-bias-candidate'):
        baseline = ROOT / 'models' / label
        if (baseline / 'refinement-model.json').is_file() and label not in {name for name, _ in boundary_models}:
            boundary_models.append((label, baseline))
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
    direct_controls = {}
    if args.with_direct_signals or default_signal_manifest['engine'] in ('signal-center-unet3d', 'signal-spotiflow3d'):
        import tempfile
        for label in (('signals-center-unet', 'signals-spotiflow') if args.with_direct_signals else (signals.name,)):
            folder = ROOT / 'models' / label
            if not (folder / 'signal-model.json').is_file():
                parser.error(f'Missing direct-center model: {label}')
            with tempfile.TemporaryDirectory(prefix='fish-center-check-') as output:
                process = subprocess.run([sys.executable, '-m', 'fish_viewer.signals', '--mode', 'controls',
                    '--model', str(folder), '--output', output, '--device', 'cpu'],
                    cwd=ROOT, text=True, capture_output=True,
                    env=dict(os.environ, PYTHONFAULTHANDLER='1'))
                if process.returncode:
                    print(process.stdout, file=sys.stderr)
                    print(process.stderr, file=sys.stderr)
                    process.check_returncode()
                report = json.loads(process.stdout.splitlines()[-1])['report']
                assert report['passed'], f'Empty-signal controls failed for {label}'
                direct_controls[label] = report
    print(json.dumps(dict(status='complete',default_cell_model=cells.name,default_signal_model=signals.name,default_signal_engine=default_signal_manifest['engine'],runtime=info,cell_probability_shape=list(probability.shape),
                         default_pipeline="consensus", default_boundary_model=Path(boundary).name if boundary else None, optional_boundary_model=Path(boundary).name if boundary else None,
                         signal_score=score.tolist(),boundary_logit_shape=boundary_shapes.get(Path(boundary).name if boundary else ''),
                         learned_boundary_shape=list(learned_output.shape),
                         sensitive_cell_probability_shape=list(sensitive_probability.shape),
                         boundary_models=boundary_shapes,proposal_shape=proposal_shape,signal_models=signal_shapes,direct_signal_controls=direct_controls)))


if __name__=='__main__':main()
