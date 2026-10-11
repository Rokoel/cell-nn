"""Plot the documented fixed-mask signal comparison from the bundled source table."""
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = [r for r in json.loads((ROOT / 'reports/signal-centers/comparison.json').read_text()) if r['split'] == 'test']
    figure = Figure(figsize=(10, 4.5), layout='constrained')
    FigureCanvasAgg(figure)
    precision, localization = figure.subplots(1, 2, gridspec_kw={'width_ratios': [1.5, 1]})
    labels = ['Original CNN', 'Refreshed CNN', 'Center U-Net', 'Spotiflow']
    expected = ['Current candidate CNN', 'Refreshed candidate CNN', '3D center U-Net', 'Spotiflow (masked training)']
    by_name = {row['model']: row for row in rows}
    rows = [by_name[name] for name in expected]
    positions = np.arange(len(rows))
    for offset, key, color in [(-.19, 'precision', '#287e8b'), (.19, 'recall', '#cf7542')]:
        bars = precision.bar(positions + offset, [100 * r[key] for r in rows], width=.38, label=key.title(), color=color)
        precision.bar_label(bars, fmt='%.1f', fontsize=8, padding=2)
    precision.set(xticks=positions, xticklabels=labels, ylim=(0, 105), ylabel='Percent', title='Precision and recall')
    precision.legend(frameon=False)
    bars = localization.bar(positions, [r['localization_rmse_um'] for r in rows], color='#287e8b', width=.65)
    localization.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)
    localization.set(xticks=positions, xticklabels=labels, ylim=(0, .23), ylabel='µm', title='Matched-signal localization RMSE')
    for axis in (precision, localization):
        axis.tick_params(axis='x', labelrotation=20)
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.15)
        axis.set_axisbelow(True)
    figure.suptitle('Historical fixed-mask test comparison · 228 reference signals', fontsize=12)
    destination = ROOT / 'docs/images/signal-models.png'
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    print(destination)


if __name__ == '__main__':
    main()
