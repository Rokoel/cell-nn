"""Compare bounded membrane attraction on fixed, complete validation matches."""
import argparse
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy import ndimage as ndi
from fish_viewer.neural.audit import read_references
from fish_viewer.refinement.membrane import MembraneConfig, snap_cell


def scores(found, truth, spacing):
    a = truth & ~ndi.binary_erosion(truth)
    b = found & ~ndi.binary_erosion(found)
    distance = np.r_[ndi.distance_transform_edt(~a, sampling=spacing)[b],
                     ndi.distance_transform_edt(~b, sampling=spacing)[a]]
    return dict(boundary_mean_error_um=float(distance.mean()),
                volume_bias_percent=float((found.sum() / truth.sum() - 1) * 100),
                absolute_volume_error_percent=float(abs(found.sum() / truth.sum() - 1) * 100),
                iou=float((found & truth).sum() / (found | truth).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    source = json.loads(args.report.read_text())
    if audit['status'] != 'complete' or source['status'] != 'complete':
        raise ValueError('Completed cache and inference required.')
    cases = {c['image']: c for c in audit['cases'] if c['audit']['split'] == 'validation'}
    if set(cases) != set(r['image'] for r in source['files']):
        raise ValueError('Calibration requires all and only validation scans.')
    configs = {f'radius-{r}-strength-{s}': MembraneConfig(max_shift_um=r, strength=s)
               for r, s in ((.2, .5), (.3, .5), (.3, 1.))}
    rows = []
    for number, row in enumerate(source['files'], 1):
        print(f'[{number}/{len(cases)}] Validation membrane peaks: {Path(row["image"]).name}', flush=True)
        case = cases[row['image']]
        with np.load(row['predictions'], allow_pickle=False) as f:
            labels = f['labels']
            metadata = json.loads(str(f['metadata']))
        a = case['audit']
        if metadata['fingerprint'] != a['fingerprint'] or metadata['channel'] != a['channel'] or not np.allclose(metadata['spacing_zyx'], a['spacing_zyx']) or not np.allclose(metadata['origin_xyz'], a['origin_xyz']):
            raise ValueError('Contour/cache provenance differs.')
        image = np.load(Path(case['directory']) / 'image.npy', mmap_mode='r')
        spacing = np.asarray(a['spacing_zyx'])
        refs, names = read_references(Path(case['directory']) / 'references.npz')
        boxes = ndi.find_objects(labels)
        padding = np.ceil(1. / spacing).astype(int) + 2
        for ref, match in enumerate(row['metrics']['per_cell']):
            cell = match['predicted_label']
            if cell is None or match['iou'] < .5 or a['reference_cut'][ref]:
                continue
            box = boxes[cell - 1]
            ref_box, ref_mask = refs[ref]
            low = np.maximum(0, np.minimum([s.start for s in box], [s.start for s in ref_box]) - padding)
            high = np.minimum(labels.shape, np.maximum([s.stop for s in box], [s.stop for s in ref_box]) + padding)
            region = tuple(slice(int(lo), int(hi)) for lo, hi in zip(low, high))
            original = labels[region] == cell
            truth = np.zeros(original.shape, bool)
            insert = tuple(slice(s.start - int(lo), s.stop - int(lo)) for s, lo in zip(ref_box, low))
            truth[insert] = ref_mask
            values = {'baseline': scores(original, truth, spacing)}
            for name, config in configs.items():
                found, diagnostic = snap_cell(original, image[region], spacing,
                                               (labels[region] > 0) & ~original, config)
                values[name] = dict(scores(found, truth, spacing), accepted=diagnostic['accepted'])
            rows.append(dict(image=row['image'], group=a['group'], cell=cell, manual_cell=names[ref], results=values))
    if not rows:
        raise ValueError('No complete validation matches.')
    summaries = {}
    for name in ['baseline', *configs]:
        by_scan = []
        for image in sorted({r['image'] for r in rows}):
            cells = [r['results'][name] for r in rows if r['image'] == image]
            by_scan.append({k: float(np.mean([c[k] for c in cells])) for k in ('boundary_mean_error_um', 'absolute_volume_error_percent', 'volume_bias_percent', 'iou')})
        summaries[name] = {k: float(np.mean([r[k] for r in by_scan])) for k in by_scan[0]}
        summaries[name]['accepted'] = sum(r['results'][name].get('accepted', False) for r in rows)
    selected = min(summaries, key=lambda n: (summaries[n]['boundary_mean_error_um'], summaries[n]['absolute_volume_error_percent']))
    result = dict(status='complete', audit=str(args.audit.resolve()), baseline_report=str(args.report.resolve()),
        audit_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        baseline_report_sha256=hashlib.sha256(args.report.read_bytes()).hexdigest(),
        validation_scans=len(cases), matched_complete_cells=len(rows),
        configs={k: asdict(c) for k, c in configs.items()}, results=summaries, selected=selected,
        selected_config=asdict(configs[selected]) if selected != 'baseline' else None, cells=rows,
        policy='Fixed baseline complete matches, IoU>=.5. Macro average over scans; voxel targets on the unchanged fine grid. '
               'All outliers and rejected/no-change refinements included. Select by mean symmetric boundary error, then volume MAPE. '
               'Baseline participates: refinement need not win. No test scans used for selection. '
               'Neighbor protection uses original contours; production also protects previously refined neighbors.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'cells'}, indent=2))


if __name__ == '__main__':
    main()
