"""Frozen-checkpoint evaluation and synthetic empty-signal controls."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..signals.evaluation import suppress, evaluate_points, summarize
from .inference import load
from .training import predict_records, predict_volume, write
from .detection import local_centers


def evaluate(index_path, model_directory, split, output, baseline_report, device='auto', emit=print):
    index = json.loads(Path(index_path).read_text())
    baseline = json.loads(Path(baseline_report).read_text())
    parent = json.loads(Path(index['parent_audit']).read_text())
    old_audit = json.loads(Path(baseline['audit']).read_text())
    expected = {c['image']: c for c in parent['cases'] if c['split'] == split}
    old_cases = {c['image']: c for c in old_audit['cases'] if c['split'] == split}
    if set(expected) != set(old_cases) or baseline['status'] != 'complete' or baseline['split'] != split:
        raise ValueError('Baseline and dense models must score identical acquisitions.')
    for image, case in expected.items():
        previous = old_cases[image]
        key = lambda c: {(p['cell_id'], p['channel'], tuple(p['xyz'])) for p in c['truth']}
        if key(case) != key(previous) or case['fingerprint'] != previous['fingerprint'] or case['cell_predictions'] != previous['cell_predictions']:
            raise ValueError('Different frozen signals or cell masks in the baseline comparison.')
    started = time.monotonic()
    model, manifest, config, runtime_device = load(model_directory, device)
    cases, scores = predict_records(model, index, split, config, runtime_device, emit)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    scans = []; all_metrics = []; by_channel = {}; counts = []
    for case, probabilities in zip(cases, scores):
        points = suppress(case['candidates'], probabilities, manifest['threshold'], config.detection_separation_um)
        metric = evaluate_points(points, case['truth'], config.match_radius_um)
        all_metrics.append(metric)
        case_id = next(i for i, c in enumerate(index['cases']) if c['image'] == case['image'])
        for cell, channel in {(r['cell_id'], r['channel']) for r in index['records'] if r['case'] == case_id}:
            found = sum(p['cell_id'] == cell and p['channel'] == channel for p in points)
            truth = sum(p['cell_id'] == cell and p['channel'] == channel for p in case['truth'])
            counts.append(dict(image=case['image'], cell_id=cell, channel=channel, predicted=found, manual=truth, absolute_error=abs(found - truth)))
        for channel in {p['channel'] for p in case['truth'] + points}:
            by_channel.setdefault(str(channel), []).append(evaluate_points(
                [p for p in points if p['channel'] == channel], [p for p in case['truth'] if p['channel'] == channel], config.match_radius_um))
        scans.append(dict(image=case['image'], group=case['group'], metrics=metric, points=points))
    old_scans = {s['image']: s for s in baseline['scans']}
    groups = sorted({c['group'] for c in cases}); rng = np.random.default_rng(42)
    deltas = []; intervals = {k: [] for k in ('precision', 'recall')}
    for _ in range(2000):
        selected = [i for g in rng.choice(groups, len(groups), replace=True) for i, c in enumerate(cases) if c['group'] == g]
        new = summarize([all_metrics[i] for i in selected])
        old = summarize([old_scans[cases[i]['image']]['metrics']['old_finder_old_cnn'] for i in selected])
        for name in intervals:
            if new[name] is not None: intervals[name].append(new[name])
        if new['recall'] is not None and old['recall'] is not None: deltas.append(new['recall'] - old['recall'])
    result = dict(version=1, status='complete', engine=manifest['engine'], split=split,
        model_directory=str(Path(model_directory).resolve()), threshold=manifest['threshold'], best_epoch=manifest['best_epoch'],
        checkpoint_sha256=hashlib.sha256((Path(model_directory) / manifest['checkpoint']).read_bytes()).hexdigest(),
        summary=summarize(all_metrics), baseline=baseline['summary']['old_finder_old_cnn'],
        candidate_baselines=baseline['summary'],
        by_channel_index={k: summarize(v) for k, v in by_channel.items()},
        channel_note='Zero-based file-local channel indices; an index does not define a fluorophore across scans.',
        count_mae_per_scored_cell_channel=float(np.mean([r['absolute_error'] for r in counts])) if counts else None,
        acquisition_bootstrap_95ci={k: np.quantile(v, [.025, .975]).tolist() if v else None for k, v in intervals.items()},
        paired_recall_improvement_95ci=np.quantile(deltas, [.025, .975]).tolist() if deltas else None,
        elapsed_s=time.monotonic() - started, scans=scans, counts=counts,
        index=str(Path(index_path).resolve()), baseline_report=str(Path(baseline_report).resolve()),
        policy='Identical frozen predicted-cell/manual-cell intersections, original inferred channels, one-to-one matching '
        'within the same cell and channel at 0.6 µm. Unknown regions unscored. No test selection. All eligible references '
        'included, even if no network peak. Threshold/checkpoint selected on validation at 90% target precision. '
        'Historically reused test fold is exploratory, not an untouched prospective assessment.')
    write(output / 'report.json', result)
    write(output / 'raw-predictions.json', dict(cases=cases))
    emit(dict(kind='finished', summary=result['summary']))
    return result


def controls(model_directory, output, device='auto'):
    model, manifest, config, runtime_device = load(model_directory, device)
    shape = (48,) * 3
    radius = np.linalg.norm((np.indices(shape) - 23.5) * config.spacing_um, axis=0)
    mask = radius < 2.5
    membrane = np.exp(-((radius - 2.5) / .18) ** 2).astype(np.float32)
    rng = np.random.default_rng(3141)
    images = dict(blank=np.zeros(shape, np.float32), constant=np.zeros(shape, np.float32),
        noise=np.clip(rng.normal(.1, .025, shape), 0, 1).astype(np.float32), membrane_only=membrane.copy())
    rows = []
    for name, image in images.items():
        heat, flow = predict_volume(model, np.stack([image, membrane, mask]).astype(np.float32), runtime_device)
        proposed = local_centers(heat, mask, np.zeros(3), config, 1, 1, flow=flow)
        accepted = suppress(proposed, [p['score'] for p in proposed], manifest['threshold'], config.detection_separation_um)
        rows.append(dict(name=name, learned_peaks=len(proposed), accepted=len(accepted), max_score=float(heat[mask].max())))
    result = dict(status='complete', engine=manifest['engine'], threshold=manifest['threshold'], controls=rows,
                  passed=all(r['accepted'] == 0 for r in rows), note='Synthetic controls with a cell mask and membrane channel; not field precision.')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    write(output / 'controls.json', result)
    return result
