"""Validate and filter saved predictions without changing the source scans."""
from collections import Counter
from datetime import datetime
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from ..models import LibraryEntry
from ..pipeline.runner import _json
from ..pipeline.segmentation import checkpoint
from ..reader import load_dataset
from ..signals.data import at_labels
from ..signals.engine import image_fingerprint
from ..signals.overlays import read_signals


def validated_result(path, cancel=None):
    """Verify pixels, calibration and every point's actual cell membership."""
    _, report = read_signals(path)
    if report.get('time', 0) != 0 or report.get('channel_index_base') != 0:
        raise ValueError('This workflow requires a time-point 1, zero-based-channel signal report.')
    with np.load(report['cell_predictions'], allow_pickle=False) as archive:
        labels = archive['labels'].copy()
        metadata = json.loads(str(archive['metadata']))
    if (metadata.get('version') != 1 or 'rays' not in metadata.get('config', {})
            or labels.ndim != 3 or not labels.size or not np.issubdtype(labels.dtype, np.integer)
            or labels.min() < 0 or labels.max() > 100_000 or metadata.get('time', 0) != 0):
        raise ValueError('A valid neural cell prediction archive is required.')
    spacing, origin = np.asarray(metadata['spacing_zyx']), np.asarray(metadata['origin_xyz'])
    if (spacing.shape != (3,) or origin.shape != (3,) or not np.isfinite(spacing).all()
            or not np.isfinite(origin).all() or np.any(spacing <= 0)):
        raise ValueError('Invalid cell prediction calibration.')
    if report['fingerprint'] != metadata['fingerprint']:
        raise ValueError('Signals and contours belong to different image pixels.')
    checkpoint(cancel)
    dataset = load_dataset(LibraryEntry(Path(report['image'])))
    if image_fingerprint(dataset) != metadata['fingerprint']:
        raise ValueError('The source scan has changed since these detections were made.')
    if (not np.allclose(origin - spacing[::-1] / 2, dataset.metadata.origin, rtol=1e-6, atol=1e-6)
            or not np.allclose(np.asarray(labels.shape) * spacing, dataset.metadata.extent[::-1], rtol=1e-6, atol=1e-6)):
        raise ValueError('The scan calibration differs from the cell prediction archive.')
    if len(report['channels']) != len(dataset.metadata.channels):
        raise ValueError('The signal channel list differs from the scan.')
    membrane = report['membrane_channel']
    if not isinstance(membrane, int) or not 0 <= membrane < len(report['channels']) or membrane != metadata['channel']:
        raise ValueError('The membrane channel differs from the cell prediction archive.')
    points = report['points']
    if any(p['channel'] == membrane for p in points):
        raise ValueError('Signals cannot be assigned to the cell membrane channel.')
    if len({p['signal_id'] for p in points}) != len(points):
        raise ValueError('Duplicate signal IDs in report.')
    if points:
        membership = at_labels(labels, [p['xyz'] for p in points], origin, spacing)
        if any(int(found) <= 0 or found != p['cell_id'] for found, p in zip(membership, points)):
            raise ValueError('A signal center is outside its assigned cell. Rerun signal detection on these contours.')
    checkpoint(cancel)
    return report, labels, metadata, dataset.metadata


def write_tables(directory, report, prefix=''):
    """CSV channel numbers are one-based, as displayed in the viewer."""
    directory = Path(directory)
    for name, rows, fields in [
        ('signals', report['points'], ['signal_id', 'cell_id', 'channel', 'x_um', 'y_um', 'z_um', 'score', 'contrast']),
        ('cell-counts', report['counts'], ['cell_id', 'channel', 'channel_name', 'count']),
    ]:
        with (directory / f'{prefix}{name}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                values = dict(row, channel=row['channel'] + 1)
                if name == 'signals':
                    values.update(zip(('x_um', 'y_um', 'z_um'), values.pop('xyz')))
                writer.writerow({key: values.get(key, '') for key in fields})


def filter_result(path, output, emit=lambda event: None, cancel=None):
    emit(dict(kind='progress', phase='Checking source pixels and signal membership', current=0, total=3))
    report, labels, metadata, _ = validated_result(path, cancel)
    kept = sorted({int(point['cell_id']) for point in report['points']})
    original = [int(value) for value in np.unique(labels) if value > 0]
    removed = sorted(set(original) - set(kept))
    emit(dict(kind='progress', phase='Removing contours of cells without signals', current=1, total=3))
    labels[~np.isin(labels, kept)] = 0
    # Keep label IDs stable: signals, meshes and tables continue to identify the same cell.
    filtering = dict(kept_cell_ids=kept, removed_cell_ids=removed, original_cell_count=len(original),
                     kept_cell_count=len(kept), removed_cell_count=len(removed), rule='at_least_one_signal_center_inside')
    metadata['signal_filter'] = filtering
    metadata['cells'] = [cell for cell in metadata.get('cells', []) if cell['label'] in kept]
    metadata.pop('metrics', None)  # Evaluation of the unfiltered prediction no longer applies.
    counts = Counter((p['cell_id'], p['channel']) for p in report['points'])
    report = dict(report, counts=[dict(cell_id=cell, channel=channel, channel_name=info['name'],
                                      count=counts[cell, channel])
        for cell in kept for channel, info in enumerate(report['channels']) if channel != report['membrane_channel']],
        signal_filter=filtering, parent_signal_report=str(Path(path).resolve()),
        unfiltered_cell_predictions=report['cell_predictions'])
    directory = Path(output).resolve() / f'filtered-scan-{datetime.now():%Y%m%d-%H%M%S-%f}'
    directory.mkdir(parents=True)
    try:
        checkpoint(cancel)
        np.savez_compressed(directory / 'contours.npz', labels=labels, metadata=np.array(json.dumps(metadata, allow_nan=False)))
        report.update(directory=str(directory), cell_predictions=str(directory / 'contours.npz'),
                      progress=dict(phase='Complete', current=3, total=3))
        write_tables(directory, report)
        _json(directory / 'report.json', report)
        checkpoint(cancel)
    except BaseException:
        shutil.rmtree(directory)
        raise
    emit(dict(kind='progress', **report['progress']))
    return report
