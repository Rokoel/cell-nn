"""Plot paired measurements and their errors from the held-out benchmark CSVs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import MaxNLocator, PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
MANUAL, PREDICTED, ERROR = '#64748b', '#0d9488', '#7c3aed'


def read_pairs(directory):
    with (directory / 'volumes.csv').open(newline='', encoding='utf-8') as stream:
        cells = [r for r in csv.DictReader(stream)
                 if float(r['iou']) >= .5 and r['complete'].lower() == 'true'
                 and r['retained_after_signal_filter'].lower() == 'true'
                 and r['manual_volume_um3'] and r['predicted_volume_um3']]
    with (directory / 'end_to_end_distances.csv').open(newline='', encoding='utf-8') as stream:
        signals = [r for r in csv.DictReader(stream)
                   if r['manual_ab_distance_um'] and r['predicted_distance_um']]
    volumes = np.array([[float(r['manual_volume_um3']), float(r['predicted_volume_um3'])] for r in cells])
    distances = np.array([[float(r['manual_ab_distance_um']), float(r['predicted_distance_um'])] for r in signals])
    for values in (volumes, distances):
        if values.ndim != 2 or values.shape[1] != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError('Finite, paired manual and predicted measurements are required.')
    if np.any(volumes[:, 0] <= 0):
        raise ValueError('Reference volumes must be positive for percentage errors.')
    cohort = dict(scans=len({r['image'] for r in signals}), groups=len({r['group'] for r in signals}))
    return volumes, distances, cohort


def bins(values, step, zero=False):
    low = 0. if zero else np.floor(np.min(values) / step) * step
    high = np.ceil(np.max(values) / step) * step
    return np.arange(low, max(low + step, high) + step * .5, step)


def style(ax, xlabel, ylabel='Count'):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    if ylabel == 'Count':
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(labelsize=10)


def box(ax, text, x=.97, y=.95, align='right'):
    ax.text(x, y, text, transform=ax.transAxes, va='top', ha=align, fontsize=10,
            bbox=dict(boxstyle='round,pad=.5', facecolor='white', edgecolor='#e2e8f0', alpha=.95))


def ecdf(ax, values, label, thresholds=()):
    ordered = np.sort(values)
    upper = float(ordered[-1]) * 1.08
    x = np.r_[0., ordered, upper]
    y = np.r_[0., np.arange(1, len(ordered)+1) / len(ordered) * 100, 100.]
    ax.step(x, y, where='post', color=ERROR, linewidth=2.3)
    ax.axhline(50, color='#94a3b8', linestyle=':', linewidth=1)
    ax.axhline(95, color='#94a3b8', linestyle=':', linewidth=1)
    for threshold in thresholds:
        ax.axvline(threshold, color='#cbd5e1', linestyle=':', linewidth=1)
        ax.scatter([threshold], [np.mean(values <= threshold) * 100], color=ERROR, s=30, zorder=3)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, 103)
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    style(ax, label, 'Observations within this error')


def save(fig, output, name):
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(output / f'{name}.{suffix}', dpi=200, facecolor='white')


def plot_volumes(values, output):
    manual, predicted = values.T
    error = predicted-manual
    absolute = np.abs(error)
    relative = error/manual*100
    fig = Figure(figsize=(12.8, 9.2))
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(top=.86, bottom=.10, hspace=.38, wspace=.25)
    fig.suptitle(f'Cell volume distributions · {len(values)} matched cells', fontsize=20, x=.125, ha='left', y=.97)
    fig.text(.125, .915, 'Complete cells retained after signal filtering; one-to-one cell IoU ≥ 0.5.', fontsize=11, color='#475569')
    ax = axes[0, 0]
    edges = bins(values.ravel(), 20, zero=True)
    ax.hist(manual, bins=edges, histtype='step', color=MANUAL, linewidth=2.3, label='Manual')
    ax.hist(predicted, bins=edges, histtype='step', color=PREDICTED, linewidth=2.3, label='Neural pipeline')
    ax.set_title('Measured volumes', loc='left', fontsize=13)
    ax.legend(frameon=False)
    style(ax, 'Cell volume (µm³)')
    ax = axes[0, 1]
    ax.hist(relative, bins=bins(relative, 5), color=ERROR, alpha=.8, edgecolor='white')
    ax.axvline(0, color='#334155', linewidth=1.3)
    ax.set_title('Signed relative volume error', loc='left', fontsize=13)
    style(ax, '100 × (predicted − manual) / manual (%)')
    box(ax, f'Mean |error|: {np.mean(np.abs(relative)):.2f}%\nMedian |error|: {np.median(np.abs(relative)):.2f}%', x=.03, align='left')
    ax = axes[1, 0]
    ax.hist(absolute, bins=bins(absolute, 2, zero=True), color=ERROR, alpha=.8, edgecolor='white')
    ax.axvline(absolute.mean(), color='#b45309', linestyle='--', linewidth=1.8, label=f'Mean {absolute.mean():.2f} µm³')
    ax.axvline(np.median(absolute), color='#334155', linestyle=':', linewidth=1.8, label=f'Median {np.median(absolute):.2f} µm³')
    ax.legend(frameon=False, fontsize=10)
    ax.set_title('Absolute volume error', loc='left', fontsize=13)
    style(ax, '|Predicted − manual| (µm³)')
    ax = axes[1, 1]
    ecdf(ax, np.abs(relative), 'Absolute relative volume error (%)', thresholds=(10, 20))
    ax.set_title('Cumulative absolute relative error', loc='left', fontsize=13)
    box(ax, f'Within 10%: {np.mean(np.abs(relative)<=10)*100:.1f}%\nWithin 20%: {np.mean(np.abs(relative)<=20)*100:.1f}%', y=.28)
    fig.text(.125, .025, 'All matched observations included. These distributions exclude missed/unmatched cells; errors measure agreement with manual meshes.',
             fontsize=9, color='#475569')
    save(fig, output, 'volume-distributions')


def plot_distances(values, output, cohort):
    manual, predicted = values.T
    error = predicted-manual
    absolute = np.abs(error)
    fig = Figure(figsize=(12.8, 9.2))
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    fig.subplots_adjust(top=.86, bottom=.10, hspace=.38, wspace=.25)
    fig.suptitle(f'Signal-to-border distance distributions · {len(values)} matched signals', fontsize=19, x=.125, ha='left', y=.97)
    fig.text(.125, .915, 'Manual A–B lengths versus the full pipeline: predicted signal center → predicted cell border.', fontsize=11, color='#475569')
    ax = axes[0, 0]
    edges = bins(values.ravel(), .1, zero=True)
    ax.hist(manual, bins=edges, histtype='step', color=MANUAL, linewidth=2.3, label='Manual')
    ax.hist(predicted, bins=edges, histtype='step', color=PREDICTED, linewidth=2.3, label='Neural pipeline')
    ax.set_title('Measured distances', loc='left', fontsize=13)
    ax.legend(frameon=False)
    style(ax, 'Signal-to-border distance (µm)')
    ax = axes[0, 1]
    ax.hist(error, bins=bins(error, .1), color=ERROR, alpha=.8, edgecolor='white')
    ax.axvline(0, color='#334155', linewidth=1.3)
    ax.set_title('Signed distance error', loc='left', fontsize=13)
    style(ax, 'Predicted − manual distance (µm)')
    box(ax, f'Bias: {np.mean(error):+.3f} µm\nRMSE: {np.sqrt(np.mean(error**2)):.3f} µm')
    ax = axes[1, 0]
    ax.hist(absolute, bins=bins(absolute, .05, zero=True), color=ERROR, alpha=.8, edgecolor='white')
    ax.axvline(absolute.mean(), color='#b45309', linestyle='--', linewidth=1.8, label=f'Mean {absolute.mean():.3f} µm')
    ax.axvline(np.median(absolute), color='#334155', linestyle=':', linewidth=1.8, label=f'Median {np.median(absolute):.3f} µm')
    ax.set_title('Absolute distance error · including outliers', loc='left', fontsize=13)
    ax.legend(frameon=False, fontsize=10)
    style(ax, '|Predicted − manual distance| (µm)')
    ax = axes[1, 1]
    ecdf(ax, absolute, 'Absolute distance error (µm)', thresholds=(.1, .2, .3))
    ax.set_title('Cumulative absolute error', loc='left', fontsize=13)
    box(ax, '\n'.join(f'Within {limit:.1f} µm: {np.mean(absolute<=limit)*100:.1f}%' for limit in (.1, .2, .3)), y=.28)
    fig.text(.125, .025, f'All {len(values)} matches included, including the largest error ({absolute.max():.3f} µm). Matched subset: {cohort["scans"]} scans, {cohort["groups"]} acquisition groups; missed signals excluded.',
             fontsize=9, color='#475569')
    save(fig, output, 'distance-distributions')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'reports/measurement-accuracy')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/measurement-accuracy/distributions')
    args = parser.parse_args()
    volumes, distances, cohort = read_pairs(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    plot_volumes(volumes, args.output)
    plot_distances(distances, args.output, cohort)
    (args.output/'README.md').write_text(
        '# Measurement distributions\n\n'
        f'Volume: {len(volumes)} complete matched cells with IoU >= 0.5, retained after signal filtering. '
        f'Distance: {len(distances)} matched signals, comparing stored manual A–B lengths with full-pipeline distances. '
        'Histograms show counts using shared bins for manual and predicted values. '
        'Signed error is predicted minus manual; cumulative plots show the proportion within each absolute error. '
        'All observations, including outliers, are included. These are conditional distributions, not estimates of detection recall.\n\n'
        'PNG previews, scalable SVGs and PDF copies are supplied. Rerun with '
        '`python scripts/plot_measurement_distributions.py`.\n', encoding='utf-8')
    print(json.dumps(dict(volume_pairs=len(volumes), distance_pairs=len(distances), output=str(args.output.resolve())), indent=2))


if __name__ == '__main__':
    main()
