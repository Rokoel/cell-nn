"""Bounded attraction of an existing contour to local cell-channel peaks.

This is an optional image-based step, not another detector. Instances, channel,
calibration and model weights stay fixed. The result is rasterized back onto the
original grid so display, signal membership, volumes and A–B distances agree.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes, euler_number

from ..models import LibraryEntry, Surface
from ..pipeline.evaluation import rasterize_reference
from ..pipeline.runner import _json
from ..pipeline.segmentation import Cancelled, Prepared, checkpoint, mask_properties
from ..run_paths import new_run_directory
from .inference import surface_evidence


@dataclass(frozen=True)
class MembraneConfig:
    max_shift_um: float = .3
    strength: float = .5
    blur_um: float = .1
    smooth_um: float = .25
    min_gain: float = .01
    min_peak: float = .08
    min_prominence: float = .025
    min_volume_ratio: float = .85
    max_volume_ratio: float = 1.25

    def validate(self):
        if any(not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Membrane refinement settings must be finite.')
        if not .05 <= self.max_shift_um <= .6 or not 0 < self.strength <= 1:
            raise ValueError('Membrane shift must be 0.05–0.6 µm; strength must be in (0, 1].')
        if not .02 <= self.blur_um <= .3 or not .05 <= self.smooth_um <= .6:
            raise ValueError('Invalid membrane smoothing scale.')
        if any(not 0 <= v <= 1 for v in (self.min_gain, self.min_peak, self.min_prominence)):
            raise ValueError('Invalid membrane brightness requirement.')
        if not 0 < self.min_volume_ratio <= 1 <= self.max_volume_ratio <= 1.5:
            raise ValueError('Invalid membrane volume limits.')
        return self


def snap_cell(mask, image, spacing, occupied, config):
    """Find nearby peaks along outward normals and move a closed 3D mesh.

    Both inward and outward corrections are possible. Weak/flat profiles and
    peaks at the search limit are ignored; they do not establish a local rim.
    Local averaging regularizes displacement without imposing a convex shape.
    """
    spacing = np.asarray(spacing, dtype=float)
    vertices, faces, _, _ = marching_cubes(np.pad(mask, 1).astype(np.float32), .5,
                                          spacing=tuple(spacing))
    points = vertices / spacing - 1
    signed = ndi.distance_transform_edt(~mask, sampling=spacing) - ndi.distance_transform_edt(mask, sampling=spacing)
    signed = ndi.gaussian_filter(signed, sigma=.6)
    gradient = np.gradient(signed, *spacing)
    normals = np.column_stack([ndi.map_coordinates(g, points.T, order=1, mode='nearest') for g in gradient])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    visible = ndi.gaussian_filter(image.astype(np.float32), config.blur_um / spacing)
    # Include extra context beyond the allowed search, used only for prominence.
    count = max(3, int(np.ceil(config.max_shift_um / .05)))
    offsets = np.linspace(-config.max_shift_um, config.max_shift_um, 2 * count + 1)
    profiles = np.array([ndi.map_coordinates(visible, (points + normals * d / spacing).T,
                                              order=1, mode='constant', cval=0, prefilter=False)
                         for d in offsets])
    context = np.array([ndi.map_coordinates(visible, (points + normals * d / spacing).T,
                                            order=1, mode='constant', cval=0, prefilter=False)
                        for d in (-config.max_shift_um - .2, config.max_shift_um + .2)])
    # A small distance penalty favors the nearer rim over another bright feature.
    score = profiles - .02 * (offsets[:, None] / config.max_shift_um) ** 2
    local_peak = (profiles[1:-1] > profiles[:-2]) & (profiles[1:-1] >= profiles[2:])
    candidates = np.full_like(score, -np.inf)
    candidates[1:-1] = np.where(local_peak, score[1:-1], -np.inf)
    selected = np.argmax(candidates, axis=0)
    peak = profiles[selected, np.arange(len(points))]
    valid = (np.isfinite(candidates[selected, np.arange(len(points))])
             & (peak >= config.min_peak) & (peak - profiles[count] >= config.min_gain)
             & (peak - context.mean(0) >= config.min_prominence))
    displacement = np.where(valid, offsets[selected] * config.strength, 0.)
    # Sparse peaks cannot drag unsupported points across the surface.
    tree = cKDTree(vertices)
    distances, neighbors = tree.query(vertices, k=min(16, len(vertices)))
    weights = np.exp(-.5 * (distances / config.smooth_um) ** 2)
    weights[distances > 2 * config.smooth_um] = 0
    displacement = (weights * displacement[neighbors]).sum(1) / weights.sum(1)
    displacement[~valid] = 0
    moved = np.abs(displacement) > 1e-5
    diagnostic = dict(peak_supported_fraction=float(valid.mean()), moved_vertex_fraction=float(moved.mean()),
                      mean_shift_um=float(np.mean(np.abs(displacement))),
                      mean_signed_shift_um=float(displacement.mean()),
                      max_vertex_shift_um=float(np.max(np.abs(displacement), initial=0)))
    if not moved.any():
        return mask, dict(diagnostic, accepted=False, fallback_reason='No brighter supported local membrane peak')
    moved_vertices = vertices - spacing + normals * displacement[:, None]
    surface = Surface('Membrane refinement', moved_vertices[:, ::-1], faces)
    prepared = Prepared(np.empty(mask.shape, np.uint8), tuple(spacing), (0., 0., 0.), 0, [], '', {}, {})
    refs, _, _ = rasterize_reference([surface], prepared)
    proposed = np.zeros_like(mask)
    if refs:
        box, raster = refs[0]
        proposed[box] = raster
    volume_ratio = float(proposed.sum() / mask.sum())
    # Match the detector's evidence scale. A narrow peak-search image would
    # alias against the evidence sampler's 0.3 µm profile spacing.
    evidence_image = ndi.gaussian_filter(image.astype(np.float32), .3 / spacing)
    baseline_evidence = surface_evidence(mask, evidence_image, spacing)
    evidence = surface_evidence(proposed, evidence_image, spacing)
    components, n = ndi.label(proposed)
    core = np.unravel_index(np.argmax(ndi.distance_transform_edt(mask, sampling=spacing)), mask.shape)
    reason = None
    if not proposed.any() or n != 1 or not components[core]:
        reason = 'Refinement disconnected the cell or lost its core'
    elif np.any(proposed & occupied):
        reason = 'Refinement would overlap another detected cell'
    elif any(np.any(proposed.take(i, axis=a)) for a in range(3) for i in (0, -1)):
        reason = 'Refinement touches the scan or search boundary'
    elif not config.min_volume_ratio <= volume_ratio <= config.max_volume_ratio:
        reason = 'Large unsupported volume change'
    elif euler_number(proposed, connectivity=3) != euler_number(mask, connectivity=3):
        reason = 'Refinement changes cell topology'
    elif evidence['surface_support'] < baseline_evidence['surface_support'] - .03 or evidence['surface_contrast'] < baseline_evidence['surface_contrast'] - .005:
        reason = 'Refinement reduced visible membrane evidence'
    elif np.array_equal(proposed, mask):
        reason = 'Peak shifts are smaller than the stored voxel grid'
    diagnostic.update(accepted=reason is None, fallback_reason=reason, proposed_volume_ratio=volume_ratio,
                      baseline_evidence=baseline_evidence, refined_evidence=evidence,
                      changed_voxels=int(np.count_nonzero(proposed != mask)))
    return (proposed if reason is None else mask), diagnostic


def snap(labels, prepared, config=None, emit=lambda e: None, cancel=None):
    config = (config or MembraneConfig()).validate()
    if labels.shape != prepared.image.shape:
        raise ValueError('Membrane image and contours must use the same voxel grid.')
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0) or labels.max(initial=0) > 100000:
        raise ValueError('Expected nonnegative 3D instance labels, at most 100000.')
    spacing = np.asarray(prepared.spacing)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or not (spacing > 0).all():
        raise ValueError('Positive, finite ZYX voxel spacing is required.')
    if not np.isfinite(prepared.image).all():
        raise ValueError('Non-finite membrane pixels.')
    result = labels.copy()
    boxes = ndi.find_objects(labels)
    ids = [i for i, box in enumerate(boxes, 1) if box is not None]
    padding = np.ceil((config.max_shift_um + .7) / np.asarray(prepared.spacing)).astype(int) + 2
    diagnostics = []
    for index, cell in enumerate(ids):
        checkpoint(cancel)
        emit(dict(kind='progress', phase='Snapping contours to membrane peaks', current=index, total=len(ids), cell=cell))
        box = boxes[cell - 1]
        region = tuple(slice(max(0, s.start - int(p)), min(n, s.stop + int(p)))
                       for s, p, n in zip(box, padding, labels.shape))
        original = labels[region] == cell
        # Protect all original neighbors as well as refinements already accepted.
        occupied = ((labels[region] != 0) & ~original) | ((result[region] != 0) & (result[region] != cell))
        proposed, diagnostic = snap_cell(original, prepared.image[region], prepared.spacing, occupied, config)
        if diagnostic['accepted']:
            result[region][original] = 0
            result[region][proposed] = cell
        diagnostics.append(dict(cell_id=cell, **diagnostic))
    emit(dict(kind='progress', phase='Membrane contours ready', current=len(ids), total=len(ids)))
    return result, diagnostics


def apply(cells_report, output, config=None, emit=lambda e: None, cancel=None, audit=None):
    """Process saved contours; retain the entire baseline run for comparison."""
    config = (config or MembraneConfig()).validate()
    directory = new_run_directory(output, 'membrane-snap')
    report = dict(version=1, engine='membrane-snap', status='running', directory=str(directory),
                  parent_cells_report=str(Path(cells_report).resolve()), config=asdict(config), files=[], failures=[])
    started = time.monotonic()
    def notify(event):
        if event['kind'] == 'progress':
            report['progress'] = {k: v for k, v in event.items() if k != 'kind'}
            if event.get('current', 0) % 10 == 0:
                _json(directory / 'report.json', report)
        emit(event)
    emit(dict(kind='run_started', directory=str(directory)))
    _json(directory / 'report.json', report)
    try:
        source = json.loads(Path(cells_report).read_text())
        if source['status'] != 'complete':
            raise ValueError('Completed cell inference is required.')
        cache = {}
        if audit:
            collection = json.loads(Path(audit).read_text())
            if collection['status'] != 'complete':
                raise ValueError('A completed membrane cache is required.')
            cache = {c['image']: c for c in collection['cases']}
        for index, row in enumerate(source['files']):
            checkpoint(cancel)
            notify(dict(kind='progress', phase='Loading cell-channel pixels', current=index, total=len(source['files']), file=row['image']))
            with np.load(row['predictions'], allow_pickle=False) as archive:
                labels = archive['labels']
                metadata = json.loads(str(archive['metadata']))
            if metadata.get('membrane_snap'):
                raise ValueError('These contours have already been snapped; choose the original run to avoid accumulating shifts.')
            if row['image'] in cache:
                case = cache[row['image']]
                a = case['audit']
                if a['fingerprint'] != metadata['fingerprint'] or a['channel'] != metadata['channel']:
                    raise ValueError('Cached cell channel differs from the detected cells.')
                image = np.load(Path(case['directory']) / 'image.npy', mmap_mode='r')
                if not np.allclose(a['spacing_zyx'], metadata['spacing_zyx']) or not np.allclose(a['origin_xyz'], metadata['origin_xyz']):
                    raise ValueError('Cached membrane uses a different physical grid.')
                prepared = Prepared(image, tuple(a['spacing_zyx']), tuple(a['origin_xyz']), a['channel'],
                                    metadata['channel_scores'], a['fingerprint'], {}, {})
            else:
                from ..reader import load_dataset
                from ..signals.engine import image_fingerprint
                from .data import prepare_membrane
                from .config import RefinementConfig
                dataset = load_dataset(LibraryEntry(Path(row['image'])))
                fingerprint = image_fingerprint(dataset)
                if fingerprint != metadata['fingerprint']:
                    raise ValueError('Source pixels changed since cell detection.')
                membrane = prepare_membrane(dataset, metadata['channel'], RefinementConfig(), fingerprint)
                # Preserve the contour grid even for older/coarser models.
                scale = np.asarray(metadata['spacing_zyx']) / membrane.spacing
                offset = (np.asarray(metadata['origin_xyz']) - membrane.origin)[::-1] / membrane.spacing
                image = ndi.affine_transform(membrane.image, np.diag(scale), offset, output_shape=labels.shape,
                                             order=1, mode='nearest', prefilter=False)
                prepared = Prepared(image, tuple(metadata['spacing_zyx']), tuple(metadata['origin_xyz']), metadata['channel'],
                                    metadata['channel_scores'], fingerprint, {}, {})
            result, diagnostics = snap(labels, prepared, config, notify, cancel)
            metadata['membrane_snap'] = dict(config=asdict(config), cells=diagnostics,
                policy='Bounded local normal-direction membrane attraction. Failed refinements retain original contours. No new cells; no roundness filter.')
            props = {c['label']: c for c in metadata.get('cells', [])}
            metadata['cells'] = []
            metadata.pop('metrics', None)
            for cell, box in enumerate(ndi.find_objects(result), 1):
                if box is None:
                    continue
                p = dict(props.get(cell, {}), **mask_properties(result[box] == cell, np.asarray(prepared.spacing)))
                p['label'] = cell
                metadata['cells'].append(p)
            folder = directory / f'{index + 1:04d}-{Path(row["image"]).stem}'
            folder.mkdir()
            path = folder / 'contours.npz'
            np.savez_compressed(path, labels=result, metadata=np.array(json.dumps(metadata, allow_nan=False)))
            updated = dict(row, predictions=str(path), membrane_snap=diagnostics)
            if 'metrics' in updated:
                updated['baseline_metrics'] = updated.pop('metrics')
            report['files'].append(updated)
            _json(directory / 'report.json', report)
        all_cells = [c for row in report['files'] for c in row['membrane_snap']]
        report.update(status='complete', summary=dict(scans=len(report['files']), cells=len(all_cells),
                                                       snapped=sum(c['accepted'] for c in all_cells)))
    except Cancelled:
        report['status'] = 'cancelled'
    except Exception:
        report.update(status='failed', error=traceback.format_exc())
    report['elapsed_s'] = time.monotonic() - started
    _json(directory / 'report.json', report)
    emit(dict(kind='finished', report=report))
    return report


def main():
    import argparse
    import signal
    from ..neural.__main__ import Cancellation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--max-shift', type=float, default=MembraneConfig.max_shift_um)
    parser.add_argument('--strength', type=float, default=MembraneConfig.strength)
    parser.add_argument('--config-file', type=Path, help='JSON object with MembraneConfig settings')
    parser.add_argument('--cancel-file', type=Path)
    args = parser.parse_args()
    cancel = Cancellation(args.cancel_file)
    signal.signal(signal.SIGINT, lambda *_: cancel.event.set())
    config = MembraneConfig(**json.loads(args.config_file.read_text())) if args.config_file else MembraneConfig(max_shift_um=args.max_shift, strength=args.strength)
    report = apply(args.cells_report, args.output, config,
                   lambda e: print(json.dumps(e, allow_nan=False), flush=True), cancel, args.audit)
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
