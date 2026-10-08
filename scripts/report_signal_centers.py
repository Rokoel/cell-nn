"""Summarize the frozen direct-center comparison with plots and CSV tables."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reports', type=Path, nargs='+', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
reports = [json.loads(p.read_text()) for p in args.reports]
if any(r['status'] != 'complete' for r in reports):
    raise ValueError('Only completed benchmarks can be summarized.')
rows = []
for split in ('validation', 'test'):
    selected = [r for r in reports if r['split'] == split]
    if not selected:
        continue
    rows.append(dict(split=split, model='Current candidate CNN', **selected[0]['baseline']))
    refreshed = selected[0].get('candidate_baselines', {}).get('new_finder_new_cnn')
    if refreshed:
        rows.append(dict(split=split, model='Refreshed candidate CNN', **refreshed))
    for report in selected:
        name = '3D center U-Net' if report['engine'] == 'signal-center-unet3d' else 'Spotiflow (masked training)'
        rows.append(dict(split=split, model=name, **report['summary']))
keys = ['split', 'model', 'tp', 'fp', 'fn', 'references', 'predictions', 'precision', 'recall', 'f1', 'f05', 'localization_mean_um', 'localization_rmse_um']
keys = [k for k in keys if k in rows[0]]
with (args.output / 'comparison.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, keys, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
(args.output / 'comparison.json').write_text(json.dumps(rows, indent=2))
figure = Figure(figsize=(11, 4.5), layout='constrained'); FigureCanvasAgg(figure)
axes = figure.subplots(1, 2)
for axis, split in zip(axes, ('validation', 'test')):
    selected = [r for r in rows if r['split'] == split]
    x = np.arange(len(selected))
    axis.bar(x - .18, [100 * (r['precision'] or 0) for r in selected], .36, label='Precision')
    axis.bar(x + .18, [100 * (r['recall'] or 0) for r in selected], .36, label='Recall')
    axis.set_xticks(x, [r['model'].replace(' candidate', '\ncandidate').replace(' (masked training)', '\n(masked training)') for r in selected], fontsize=9)
    for i, row in enumerate(selected):
        axis.text(i, 3, f"{row['tp']} matches\n{row['fp']} extra", ha='center', fontsize=9)
    axis.set_title(split.capitalize()); axis.set_ylim(0, 105); axis.set_ylabel('Percent'); axis.grid(axis='y', alpha=.2); axis.legend()
figure.savefig(args.output / 'precision-recall.png', dpi=160)
figure = Figure(figsize=(10, 4.5), layout='constrained'); FigureCanvasAgg(figure)
axis = figure.subplots()
reference = next((r for r in reports if r['split'] == 'test'), None)
if reference:
    baseline = json.loads(Path(reference['baseline_report']).read_text())
    values = sorted(d for scan in baseline['scans'] for d in scan['metrics']['old_finder_old_cnn']['distances_um'])
    if values:
        axis.plot(values, np.arange(1, len(values) + 1) / len(values), label='Current candidate CNN')
for report in reports:
    if report['split'] != 'test':
        continue
    values = sorted(d for scan in report['scans'] for d in scan['metrics'].get('distances_um', []))
    if values:
        axis.plot(values, np.arange(1, len(values) + 1) / len(values), label=report['engine'])
axis.set_xlabel('Matched signal-center error (µm)'); axis.set_ylabel('Cumulative fraction of matched signals')
axis.set_title('Localization conditional on a matched signal'); axis.grid(alpha=.2)
if axis.lines: axis.legend()
figure.savefig(args.output / 'localization.png', dpi=160)
