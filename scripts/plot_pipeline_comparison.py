"""Render current-preset measurement evidence from the retained JSON reports."""
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    consensus = json.loads((ROOT / 'reports/consensus-correct-channels/summary.json').read_text())
    stardist = json.loads((ROOT / 'reports/stardist2d/distances/report.json').read_text())
    assert consensus['status'] == stardist['status'] == 'complete'
    rows = [consensus['summary']['test'], stardist['summary']]
    assert all(row['scans'] == 71 for row in rows)
    labels = ['Both-detector\nconsensus', 'Fine-tuned\nStarDist 2D']
    colors = ['#168b87', '#7163bb']
    figure = Figure(figsize=(12, 5.8), facecolor='#f7f9fc')
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    figure.subplots_adjust(left=.08, right=.97, bottom=.25, top=.73, wspace=.34)
    for axis, (metric, key, title, unit, limit) in zip(axes, [
        ('volume', 'mape_percent', 'Cell volume error', 'Mean absolute percentage error (%)', 14),
        ('distance', 'mae', 'Signal-to-border distance error', 'Mean absolute error (µm)', .25),
    ]):
        values = np.array([row[metric][key] for row in rows])
        interval_key = 'mape_95ci_acquisition_bootstrap' if metric == 'volume' else 'mae_95ci_acquisition_bootstrap'
        intervals = np.array([row[metric][interval_key] for row in rows])
        axis.set_facecolor('#f7f9fc')
        axis.bar(range(2), values, width=.48, color=colors, zorder=3)
        axis.errorbar(range(2), values, yerr=np.stack([values-intervals[:, 0], intervals[:, 1]-values]),
                      fmt='none', color='#25364d', capsize=5, zorder=4)
        for index, (value, interval, row) in enumerate(zip(values, intervals, rows)):
            label = f'{value:.2f}%' if metric == 'volume' else f'{value:.3f} µm'
            axis.text(index, interval[1]+limit*.04, label, ha='center', fontsize=13, weight='bold', color='#25364d')
            axis.text(index, -limit*.23, f"{row[metric]['n']} matched {'cells' if metric == 'volume' else 'signals'}",
                      ha='center', fontsize=10, color='#596b80')
        axis.set(xticks=range(2), xticklabels=labels, ylim=(0, limit), ylabel=unit)
        axis.set_title(title, loc='left', fontsize=14, weight='bold', pad=16, color='#25364d')
        axis.spines[['top', 'right', 'left']].set_visible(False)
        axis.spines['bottom'].set_color('#cbd5e1')
        axis.tick_params(length=0, labelcolor='#25364d', pad=8)
        axis.grid(axis='y', color='#e1e7ef', zorder=0)
        axis.set_axisbelow(True)
    figure.text(.08, .94, 'Two cell pipelines. One measurement workflow.', fontsize=21, weight='bold', color='#192c43')
    figure.text(.08, .87, '71 historical test scans · roundness ≥ 0.72 · signal-center 3D U-Net', fontsize=12, color='#596b80')
    figure.text(.08, .06, 'Whiskers: 95% acquisition-bootstrap intervals. Different matched subsets and channel protocols;\n'
                'these report summaries are not a paired ranking or a measure of whole-field detection accuracy.',
                fontsize=10, color='#596b80', linespacing=1.6)
    destination = ROOT / 'docs/images/pipeline-comparison.png'
    figure.savefig(destination, dpi=180)
    print(destination)


if __name__ == '__main__':
    main()
