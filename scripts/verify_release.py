"""Load both bundled checkpoints and exercise CPU inference without training data."""
import json
import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fish_viewer.analysis.controller import default_models, check_models
from fish_viewer.neural.engine import runtime
from fish_viewer.signals.engine import network
from fish_viewer.signals.config import SignalConfig


def main():
    cells,signals=default_models()
    check_models(cells,signals)
    provenance=ROOT/'models/origins.json'
    if provenance.exists():
        for label,info in json.loads(provenance.read_text()).items():
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
    print(json.dumps(dict(status='complete',runtime=info,cell_probability_shape=list(probability.shape),signal_score=score.tolist())))


if __name__=='__main__':main()
