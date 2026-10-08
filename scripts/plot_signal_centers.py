"""Show fixed-order validation cell crops and their learned center heatmaps."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from fish_viewer.spots.inference import load
from fish_viewer.spots.training import predict_volume

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--index', type=Path, required=True)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--benchmark', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
index = json.loads(args.index.read_text()); report = json.loads(args.benchmark.read_text())
if report['split'] != 'validation' or report['status'] != 'complete':
    raise ValueError('Use the frozen validation comparison for illustrations.')
model, manifest, config, device = load(args.model)
sha = hashlib.sha256((args.model / manifest['checkpoint']).read_bytes()).hexdigest()
if sha != report['checkpoint_sha256']:
    raise ValueError('The illustration checkpoint differs from the scored checkpoint.')
lookup = {r['image']: r for r in report['scans']}
selected = []; used = set()
for record in index['records']:
    if record['split'] == 'validation' and record['points'] and record['group'] not in used:
        selected.append(record); used.add(record['group'])
        if len(selected) == 6: break
figure = Figure(figsize=(10, 3 * len(selected)), layout='constrained'); FigureCanvasAgg(figure)
axes = figure.subplots(len(selected), 2, squeeze=False)
examples = []
for pair, record in zip(axes, selected):
    with np.load(record['path']) as archive:
        inputs = archive['inputs'].astype(np.float32); manual = archive['points']
    heat, _ = predict_volume(model, inputs, device)
    image = index['cases'][record['case']]['image']
    rows = [p for p in lookup[image]['points'] if p['cell_id'] == record['cell_id'] and p['channel'] == record['channel']]
    predicted = np.asarray([(np.asarray(p['xyz']) - record['origin_xyz'])[::-1] / config.spacing_um for p in rows]).reshape(-1, 3)
    for axis, volume, title in zip(pair, (inputs[0], heat), ('Signal channel (maximum projection)', 'Learned center heatmap (maximum projection)')):
        axis.imshow(volume.max(0), cmap='gray' if title.startswith('Signal') else 'magma', vmin=0, vmax=1)
        axis.contour(inputs[2].max(0), levels=[.5], colors=['cyan'], linewidths=.5)
        axis.scatter(manual[:, 2], manual[:, 1], marker='+', c='#3fa9ff', s=55, label='Manual A')
        if len(predicted): axis.scatter(predicted[:, 2], predicted[:, 1], facecolors='none', edgecolors='#3dff7c', s=70, label='Accepted center')
        axis.set_title(title, fontsize=10); axis.set_xlabel(f'{Path(image).name[:44]} · cell {record["cell_id"]} · channel {record["channel"] + 1}', fontsize=8)
        axis.set_xticks([]); axis.set_yticks([]); axis.legend(loc='upper right', fontsize=7)
    examples.append(dict(image=image, cell_id=record['cell_id'], channel=record['channel'], manual=len(manual), predicted=len(predicted)))
args.output.mkdir(parents=True, exist_ok=True)
figure.savefig(args.output / 'validation-examples.png', dpi=130)
(args.output / 'validation-examples.json').write_text(json.dumps(dict(checkpoint_sha256=sha, examples=examples,
    policy='First eligible record in each of six distinct validation acquisition groups, in corpus order. Projection hides axial position.'), indent=2))
