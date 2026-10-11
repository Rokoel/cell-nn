from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import json
from pathlib import Path
import re
import time

import numpy as np

from ..models import LibraryEntry
from ..reader import load_dataset
from ..run_paths import new_run_directory
from .evaluation import aggregate, evaluate, rasterize_reference
from .segmentation import Cancelled, Prediction, Prepared, checkpoint, prediction_surfaces, prepare, segment


@dataclass(eq=False)
class Case:
    entry: LibraryEntry
    prepared: Prepared
    references: list
    names: list[str]
    skipped: list[str]
    group: str
    directory: Path


def acquisition_group(entry):
    match = re.match(r'^(\d{6,8})[_ -]', entry.image.name)
    return str(entry.image.parent) + '/' + (match.group(1) if match else entry.image.stem)


def split_cases(cases, fraction=0.2):
    """Group by acquisition prefix and exact pixel fingerprint before splitting."""
    if not 0 < fraction < 1:
        raise ValueError('Evaluation fraction must be between 0 and 1.')
    # Union groups when identical pixels were saved under different filenames.
    parent = list(range(len(cases)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    groups, fingerprints = {}, {}
    for i, case in enumerate(cases):
        for key, mapping in ((case.group, groups), (case.prepared.fingerprint, fingerprints)):
            if key in mapping:
                parent[find(i)] = find(mapping[key])
            else:
                mapping[key] = i
    buckets = {}
    for i, case in enumerate(cases):
        buckets.setdefault(find(i), []).append(case)
    grouped = sorted(buckets.values(), key=lambda bucket: min(c.group for c in bucket))
    if len(grouped) < 2:
        raise ValueError('Tuning with independent evaluation needs at least two acquisition groups. '
            'Add images from another acquisition, or use Apply settings for a single group.')
    count = max(1, min(len(grouped)-1, int(round(len(grouped) * fraction))))
    # Stable chronological/name order, independent of annotation scores.
    return [c for group in grouped[:-count] for c in group], [c for group in grouped[-count:] for c in group]


def trial_configs(base, count, seed=42):
    if not 1 <= count <= 200:
        raise ValueError('Choose between 1 and 200 parameter trials.')
    result = [base.validate()]
    # Include a classical alternative, then bounded local parameter exploration.
    if count > 1:
        result.append(replace(base, method='distance_watershed', closing_um=max(0.3, base.closing_um)))
    rng = np.random.default_rng(seed)
    grid = [replace(base, method='boundary_watershed', threshold=float(np.clip(base.threshold * factor, 0.01, 0.9)),
        smooth_um=float(np.clip(base.smooth_um * smooth, 0, 5)),
        seed_distance_um=float(np.clip(base.seed_distance_um * separation, 0.2, 100)))
        for factor in (0.45, 0.65, 0.85, 1.15, 1.5, 2.0)
        for smooth in (0.65, 1.0, 1.4) for separation in (0.8, 1.0, 1.3)]
    rng.shuffle(grid)
    result.extend(grid[:max(0, count-len(result))])
    while len(result) < count:
        result.append(replace(base, threshold=float(rng.uniform(0.02, 0.16)), smooth_um=float(rng.uniform(0.08, 0.35)),
            seed_distance_um=float(rng.uniform(0.7, 1.8)), closing_um=float(rng.choice([0, 0.3]))))
    return result


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_prediction(path, prediction, prepared, entry, config, metrics):
    metadata = dict(version=1, image=str(entry.image), image_name=entry.image.name,
        scene=str(entry.scene) if entry.scene else None, time=0, channel=prepared.channel,
        channel_scores=prepared.channel_scores, fingerprint=prepared.fingerprint, origin_xyz=prepared.origin,
        spacing_zyx=prepared.spacing, unit='µm', config=asdict(config), metrics=metrics, cells=prediction.cells)
    np.savez_compressed(path, labels=prediction.labels, metadata=np.array(json.dumps(metadata, allow_nan=False)))
    with Path(path).with_suffix('.cells.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['label', 'volume_um3', 'diameter_um', 'sphericity', 'elongation', 'center_xyz']
        fields += sorted({key for cell in prediction.cells for key in cell}-set(fields))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(prediction.cells)
    return metadata


def read_prediction(path):
    with np.load(path, allow_pickle=False) as archive:
        labels = archive['labels'].copy()
        metadata = json.loads(str(archive['metadata']))
    if metadata.get('version') != 1 or labels.ndim != 3 or not labels.size or not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0:
        raise ValueError('Invalid contour prediction archive.')
    spacing = tuple(metadata['spacing_zyx'])
    origin = tuple(metadata['origin_xyz'])
    if len(spacing) != 3 or len(origin) != 3 or not all(np.isfinite(spacing)) or not all(s > 0 for s in spacing) or not all(np.isfinite(origin)):
        raise ValueError('Invalid prediction calibration.')
    if labels.max() > 100_000:
        raise ValueError('Prediction archive has too many object labels.')
    prepared = Prepared(np.empty((0, 0, 0), np.float32), spacing, origin,
        metadata['channel'], metadata['channel_scores'], metadata['fingerprint'], {}, {})
    prediction = Prediction(labels, metadata.get('cells', []), {}, {}, {})
    return prediction_surfaces(prediction, prepared, metadata.get('time', 0)), metadata


def write_charts(directory, report):
    # Figure API avoids pyplot's global state when called by a GUI worker.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(11, 7), layout='constrained')
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2)
    trials = report.get('trials', [])
    x = [r['trial'] for r in trials]
    for ax, key, title in [(axes[0,0], 'objective', 'Tuning loss: 1 − mean Dice'),
                          (axes[0,1], 'mean_dice', 'Annotated-cell overlap'),
                          (axes[1,0], 'boundary_rmse_um', 'Matched-cell boundary RMSE (µm)'),
                          (axes[1,1], 'detection_rate', 'Annotated-cell detection rate (IoU ≥ 0.5)')]:
        y = [r['scores'].get(key) for r in trials]
        ax.plot(x, [np.nan if v is None else v for v in y], 'o-', label='Tuning images', markersize=3)
        if key == 'objective' and y:
            ax.plot(x, np.minimum.accumulate(y), label='Best so far')
        if key == 'mean_dice':
            ax.plot(x, [r['scores'].get('mean_iou', np.nan) for r in trials], label='IoU')
        final = report.get('evaluation', {}).get(key)
        if final is not None:
            ax.scatter([max(x, default=0)+1], [final], marker='*', s=120, color='#dd9222', label='Reserved images, final only')
        ax.set_title(title)
        ax.set_xlabel('Parameter trial')
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(directory / 'scores.png', dpi=150)
    fig.savefig(directory / 'scores.svg')


def run_pipeline(entries, config, output_root, mode='tune', trials=12, holdout=0.2, seed=42, emit=None, cancel=None):
    config.validate()
    if mode not in ('tune', 'apply'):
        raise ValueError('Mode must be tune or apply.')
    if not entries:
        raise ValueError('Select at least one Imaris dataset.')
    unique = {}
    for entry in entries:
        key = entry.image.resolve()
        if key not in unique or (entry.scene and not unique[key].scene) or (entry.scene and entry.scene.stem == entry.image.stem):
            unique[key] = entry
    entries = sorted(unique.values(), key=lambda e: str(e.image))
    directory = new_run_directory(output_root, mode)
    report = dict(version=1, mode=mode, status='running', directory=str(directory), seed=seed,
        initial_config=asdict(config), best_config=None, trials=[], files=[], failures=[], warnings=[], split={},
        metric_notes='Partial references: unmatched predictions are unscored. Missed references have zero Dice/IoU. '
        'Boundary RMSE and mask MSE are conditional on matched cells. Objective is macro mean over images. '
        'Evaluation images are used only once, after selecting parameters on tuning images.')
    _json(directory / 'report.json', report)
    start = time.perf_counter()
    def notify(event):
        checkpoint(cancel)
        if emit:
            emit(event)
    def announce_case(case, phase, current, total):
        notify(dict(kind='file', phase=phase, file=str(case.entry.image), current=current, total=total,
            previews=case.prepared.previews, channel_scores=case.prepared.channel_scores,
            timings=case.prepared.timings))
    cases = []
    seen = {}
    try:
        for number, entry in enumerate(entries, 1):
            checkpoint(cancel)
            notify(dict(kind='file', phase='Preprocessing', file=str(entry.image), current=number, total=len(entries)))
            def file_event(event, entry=entry):
                notify(dict(event, file=str(entry.image)))
            try:
                dataset = load_dataset(entry, time=0, level=0)
                prepared = prepare(dataset, config, file_event, cancel)
                if prepared.fingerprint in seen:
                    report['warnings'].append(f'Exact duplicate skipped: {entry.image}; same pixels as {seen[prepared.fingerprint]}')
                    continue
                seen[prepared.fingerprint] = str(entry.image)
                refs, names, skipped = rasterize_reference(dataset.scene.surfaces, prepared, 0, cancel)
                file_dir = directory / f'{number:04d}-{entry.image.stem}'
                file_dir.mkdir()
                # Memory-mapped cache bounds RAM when many datasets are tuned.
                np.save(file_dir / 'preprocessed.npy', prepared.image)
                prepared.image = np.load(file_dir / 'preprocessed.npy', mmap_mode='r')
                _json(file_dir / 'preprocessing.json', dict(channel_scores=prepared.channel_scores,
                    selected_channel=prepared.channel, spacing_zyx=prepared.spacing, origin_xyz=prepared.origin,
                    fingerprint=prepared.fingerprint, warnings=prepared.warnings + dataset.scene.warnings + skipped))
                report['warnings'].extend(f'{entry.image.name}: {w}' for w in prepared.warnings + dataset.scene.warnings + skipped)
                case = Case(entry, prepared, refs, names, skipped, acquisition_group(entry), file_dir)
                cases.append(case)
                del dataset
                if mode == 'apply':
                    _finish_case(case, config, 'apply', report, file_event, cancel)
                    cases.clear()  # Batch application streams one image at a time.
            except Cancelled:
                raise
            except Exception as exc:
                report['failures'].append(dict(file=str(entry.image), error=str(exc)))
                notify(dict(kind='warning', message=f'{entry.image.name}: {exc}'))
            _json(directory / 'report.json', report)
        if mode == 'tune':
            eligible = [c for c in cases if c.references]
            training, evaluation = split_cases(eligible, holdout)
            report['split'] = dict(tuning=[str(c.entry.image) for c in training], evaluation=[str(c.entry.image) for c in evaluation],
                unannotated=[str(c.entry.image) for c in cases if not c.references], policy='Acquisition-prefix groups; exact pixel duplicates removed; final group(s) reserved.')
            notify(dict(kind='split', split=report['split']))
            best_score = float('inf')
            best = config
            for index, candidate in enumerate(trial_configs(config, trials, seed), 1):
                scores = []
                for number, case in enumerate(training, 1):
                    announce_case(case, f'Trial {index}/{trials}', number, len(training))
                    prediction = segment(case.prepared, candidate, lambda e: notify(dict(e, file=str(case.entry.image))), cancel)
                    scores.append(evaluate(prediction.labels, case.references, case.names, case.prepared.spacing, cancel))
                score = aggregate(scores)
                if score['objective'] < best_score:
                    best_score, best = score['objective'], candidate
                row = dict(trial=index, config=asdict(candidate), scores=score, best_objective=best_score, elapsed_s=time.perf_counter()-start)
                report['trials'].append(row)
                report['best_config'] = asdict(best)
                notify(dict(kind='trial', **row))
                _json(directory / 'report.json', report)
            report['best_config'] = asdict(best)
            # Reserved images are first scored here, after the winner is fixed.
            reserved_scores = []
            for number, case in enumerate(cases, 1):
                split = 'evaluation' if case in evaluation else 'tuning' if case in training else 'unannotated'
                announce_case(case, f'Final {split}', number, len(cases))
                metrics = _finish_case(case, best, split, report, lambda e: notify(dict(e, file=str(case.entry.image))), cancel)
                if split == 'evaluation':
                    reserved_scores.append(metrics)
            report['evaluation'] = aggregate(reserved_scores)
            notify(dict(kind='evaluation', scores=report['evaluation']))
        else:
            if not report['files']:
                raise ValueError('No dataset could be processed; see the failures in the report.')
            report['best_config'] = asdict(config)
        report['status'] = 'complete_with_errors' if report['failures'] else 'complete'
        report['elapsed_s'] = time.perf_counter()-start
        _json(directory / 'settings.json', dict(version=1, config=report['best_config']))
        write_charts(directory, report)
    except Cancelled:
        report['status'] = 'cancelled'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = str(exc)
    finally:
        report['elapsed_s'] = time.perf_counter()-start
        _json(directory / 'report.json', report)
        _write_tables(directory, report)
    if emit:
        emit(dict(kind='finished', report=report))
    return report


def _finish_case(case, config, split, report, emit, cancel):
    prediction = segment(case.prepared, config, emit, cancel)
    metrics = evaluate(prediction.labels, case.references, case.names, case.prepared.spacing, cancel)
    path = case.directory / 'contours.npz'
    save_prediction(path, prediction, case.prepared, case.entry, config, metrics)
    row = dict(image=str(case.entry.image), scene=str(case.entry.scene) if case.entry.scene else None,
        split=split, channel=case.prepared.channel, channel_scores=case.prepared.channel_scores,
        predictions=str(path), metrics=metrics, rejected=prediction.rejected, timings=prediction.timings)
    report['files'].append(row)
    if emit:
        emit(dict(kind='result', row=row))
    return metrics


def _write_tables(directory, report):
    with (directory / 'trials.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['trial', 'method', 'threshold', 'smooth_um', 'seed_distance_um', 'objective', 'mean_dice', 'mean_iou', 'detection_rate', 'boundary_rmse_um', 'best_objective']
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        for row in report['trials']:
            writer.writerow(dict(row, **row['config'], **row['scores']))
    with (directory / 'cells-evaluation.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['image', 'split', 'name', 'predicted_label', 'dice', 'iou', 'detected', 'boundary_rmse_um', 'boundary_hd95_um', 'pair_mask_mse']
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        for row in report['files']:
            for cell in row['metrics']['per_cell']:
                writer.writerow(dict(cell, image=row['image'], split=row['split']))
