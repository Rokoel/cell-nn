"""Conservative image-supported recovery; no manual annotations or shape templates."""
from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage as ndi

from ..pipeline.segmentation import checkpoint
from ..refinement.data import coordinates, seed_image
from ..refinement.inference import seeded_component


@dataclass(frozen=True)
class RecoveryConfig:
    local_evidence: bool = True
    retry: bool = True
    signal_guided: bool = True
    local_support: float = .75
    relative_contrast: float = .15
    minimum_contrast: float = .006
    noise_multiplier: float = 2.5
    slice_continuity: float = .7
    retry_margin_um: float = 1.
    seed_shift_um: float = .4
    stability_dice: float = .85
    maximum_overlap: float = .1
    signal_score: float = .9
    signal_search_um: float = .6

    def validate(self):
        for key in ('local_evidence', 'retry', 'signal_guided'):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f'Invalid recovery {key}; expected true or false.')
        for key in ('local_support', 'relative_contrast', 'slice_continuity', 'stability_dice', 'maximum_overlap', 'signal_score'):
            if not 0 < getattr(self, key) <= 1:
                raise ValueError(f'Invalid recovery {key}.')
        for key in ('minimum_contrast', 'noise_multiplier', 'retry_margin_um', 'seed_shift_um', 'signal_search_um'):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'Invalid recovery {key}.')
        return self


def local_surface_evidence(mask, image, spacing, config, noise_image=None):
    """Contrast against nearby background/noise, and support across Z slices."""
    empty = dict(local_support=0., relative_contrast=0., local_contrast=0., noise=0., slice_continuity=0.)
    if not mask.any(): return empty
    spacing = np.asarray(spacing)
    inside = ndi.distance_transform_edt(np.pad(mask, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    outside = ndi.distance_transform_edt(~np.pad(mask, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    signed = ndi.gaussian_filter(outside-inside, .6)
    gradient = np.asarray(np.gradient(signed, *spacing))
    boundary = np.argwhere(mask & ~ndi.binary_erosion(mask))
    points = boundary[np.linspace(0, len(boundary)-1, min(1024, len(boundary)), dtype=int)].T
    normals = gradient[(slice(None), *points)]
    norms = np.linalg.norm(normals, axis=0)
    normals /= np.maximum(norms, 1e-6)
    profiles = np.array([ndi.map_coordinates(image, points+normals*d/spacing[:, None], order=1,
                                            mode='nearest', prefilter=False)
                         for d in (-.9, -.3, 0, .3, .9)])
    rim = profiles[1:4].max(axis=0)
    contrast = rim-(profiles[0]+profiles[-1])/2
    shell = (outside <= 1.2) & (inside <= .6)
    values = image[shell]
    scale = max(float(np.percentile(values, 95)-np.percentile(values, 20)), .001)
    # Neighbour differences in external context estimate noise without using
    # the bright rim itself as a noise estimate. Constant images cannot pass.
    background = (~mask) & (outside <= 1.2) & (outside >= .6)
    differences = []
    noise_source = image if noise_image is None else noise_image
    for axis in range(3):
        left = [slice(None)]*3; right = left.copy()
        left[axis] = slice(0, -1); right[axis] = slice(1, None)
        valid = background[tuple(left)] & background[tuple(right)]
        differences.extend((noise_source[tuple(left)]-noise_source[tuple(right)])[valid].tolist())
    noise = float(np.median(np.abs(differences))/.954) if differences else .001
    required = max(config.minimum_contrast, config.noise_multiplier*noise, config.relative_contrast*scale)
    supported = (contrast >= required) & (norms > 1e-5)
    counts = np.bincount(points[0], minlength=mask.shape[0])
    good = np.bincount(points[0], weights=supported, minlength=mask.shape[0])
    substantive = counts >= max(3, counts.max()*.2)
    continuity = float(np.mean((good/np.maximum(counts, 1))[substantive] >= .55)) if substantive.any() else 0.
    return dict(local_support=float(supported.mean()), relative_contrast=float(contrast.mean()/scale),
                local_contrast=float(contrast.mean()), noise=noise, slice_continuity=continuity)


def evidence_passes(evidence, config):
    return (evidence['local_support'] >= config.local_support
            and evidence['relative_contrast'] >= config.relative_contrast
            and evidence['local_contrast'] >= max(config.minimum_contrast, config.noise_multiplier*evidence['noise'])
            and evidence['slice_continuity'] >= config.slice_continuity)


def crop_prediction(prepared, proposal, model, config):
    spacing = np.asarray(prepared.spacing); origin = np.asarray(prepared.origin)
    low = (np.asarray(proposal['box_low_xyz'])-origin)[::-1]/spacing
    high = (np.asarray(proposal['box_high_xyz'])-origin)[::-1]/spacing
    center = (low+high)/2
    seed = (np.asarray(proposal['center_xyz'])-origin)[::-1]/spacing
    width = float(np.max((high-low)*spacing)+2*config.margin_um); step = width/config.patch_size
    coords = coordinates(center, width, spacing, config.patch_size)
    sample = ndi.map_coordinates(prepared.image, coords, order=1, mode='constant', cval=0, prefilter=False)
    seed_patch = (seed-center)*spacing/step+(config.patch_size-1)/2
    inputs = np.stack([sample, seed_image(seed_patch, step, config)], axis=-1)[None]
    logits = np.asarray(model(inputs, training=False))[0, ..., 0]
    if not np.isfinite(logits).all(): raise ValueError('Non-finite recovery contour prediction.')
    probability = 1/(1+np.exp(-np.clip(logits, -60, 60)))
    lower = np.maximum(0, np.floor(center-width/2/spacing).astype(int))
    upper = np.minimum(prepared.image.shape, np.ceil(center+width/2/spacing).astype(int)+1)
    if (upper <= lower).any(): return None
    region = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
    grid = np.indices(tuple(upper-lower), dtype=np.float32)+lower[:, None, None, None]
    at = (grid-center[:, None, None, None])*spacing[:, None, None, None]/step+(config.patch_size-1)/2
    scores = ndi.map_coordinates(probability, at, order=1, mode='constant', cval=0, prefilter=False)
    return region, seeded_component(scores >= config.threshold, seed-lower), seed-lower


def geometry_valid(mask, spacing):
    return (mask.any() and mask.sum()*np.prod(spacing) >= 5.
            and not any(mask.take(i, axis=a).any() for a in range(3) for i in (0, -1)))


def point_inside(candidate, xyz, prepared):
    region, mask, _ = candidate
    at = np.rint((np.asarray(xyz)-prepared.origin)[::-1]/prepared.spacing).astype(int)
    at -= np.array([s.start for s in region])
    return bool(np.all(at >= 0) and np.all(at < mask.shape) and mask[tuple(at)])


def dice(left, right):
    lr, lm, _ = left; rr, rm, _ = right
    low = np.maximum([s.start for s in lr], [s.start for s in rr])
    high = np.minimum([s.stop for s in lr], [s.stop for s in rr])
    if (high <= low).any(): return 0.
    a = tuple(slice(int(lo-s.start), int(hi-s.start)) for lo, hi, s in zip(low, high, lr))
    b = tuple(slice(int(lo-s.start), int(hi-s.start)) for lo, hi, s in zip(low, high, rr))
    return float(2*np.count_nonzero(lm[a] & rm[b])/max(1, lm.sum()+rm.sum()))


def recover(prepared, proposals, model, boundary_config, labels, diagnostics, candidates,
            config=None, emit=lambda e: None, cancel=None, signals=None, local=True, retries=True):
    config = (config or RecoveryConfig()).validate()
    spacing = np.asarray(prepared.spacing)
    visible = ndi.gaussian_filter(prepared.image, .3/spacing)
    rows = {d['proposal_id']: d for d in diagnostics}
    summary = dict(local=0, retry=0, signal_guided=0, attempted=0)
    for index, proposal in enumerate(proposals):
        checkpoint(cancel)
        phase = 'Recovering signal-supported cell contours' if signals is not None else 'Checking local membranes and retrying uncertain contours'
        emit(dict(kind='progress', phase=phase, current=index, total=len(proposals)))
        row = rows[proposal['proposal_id']]
        if row['accepted']: continue
        row.setdefault('original_rejection_reason', row['rejection_reason'])
        guidance = [s for s in (signals or []) if s['cell_id'] == proposal['proposal_id'] and s['score'] >= config.signal_score]
        if signals is not None and not guidance: continue
        summary['attempted'] += 1
        attempt_log = []
        possible = []
        original = candidates.get(proposal['proposal_id'])
        if original is not None: possible.append(('original', original))
        if retries:
            expanded = replace(boundary_config, margin_um=boundary_config.margin_um+config.retry_margin_um)
            variants = [('padded', proposal)]
            widths = np.asarray(proposal['box_high_xyz'])-proposal['box_low_xyz']
            axis = int(np.argmax(widths))
            for sign in (-1, 1):
                shifted = np.asarray(proposal['center_xyz'], dtype=float).copy(); shifted[axis] += sign*config.seed_shift_um
                variants.append((f'seed_{sign:+}', dict(proposal, center_xyz=shifted.tolist())))
            if guidance:
                # Signals may lie on a border, so never substitute their position
                # for the cell seed. Move the original seed by at most 0.4 µm.
                shift = np.asarray(max(guidance, key=lambda s:s['score'])['xyz'])-proposal['center_xyz']
                shift *= min(.25, config.seed_shift_um/max(np.linalg.norm(shift), 1e-6))
                guided_seed = np.asarray(proposal['center_xyz'])+shift
                variants.append(('signal_seed', dict(proposal, center_xyz=guided_seed.tolist())))
                for sign in (-1, 1):
                    perturbed = guided_seed.copy(); perturbed[axis] += sign*config.seed_shift_um/4
                    variants.append((f'signal_seed_{sign:+}', dict(proposal, center_xyz=perturbed.tolist())))
            for name, variant in variants:
                checkpoint(cancel)
                candidate = crop_prediction(prepared, variant, model, expanded)
                if candidate is not None: possible.append((name, candidate))
        qualified = []
        for name, candidate in possible:
            region, mask, seed = candidate
            evidence = local_surface_evidence(mask, visible[region], spacing, config, prepared.image[region])
            valid = geometry_valid(mask, spacing) and evidence_passes(evidence, config)
            if guidance: valid = valid and any(point_inside(candidate, s['xyz'], prepared) for s in guidance)
            attempt_log.append(dict(variant=name, volume_um3=float(mask.sum()*np.prod(spacing)), qualified=bool(valid), **evidence))
            if valid: qualified.append((name, candidate, evidence))
        choice = None
        if local and signals is None:
            choice = next((q for q in qualified if q[0]=='original'), None)
        stable = []
        if retries:
            for q in qualified:
                agreement = max((dice(q[1], other[1]) for other in qualified if q is not other), default=0.)
                if agreement >= config.stability_dice: stable.append((agreement, q))
            if choice is None and stable:
                _, choice = max(stable, key=lambda v:(v[1][2]['local_support'], v[0]))
        accepted = False; method = None
        if choice:
            name, (region, mask, seed), evidence = choice
            overlap = float(np.mean(labels[region][mask] > 0))
            if overlap <= config.maximum_overlap:
                mask = seeded_component(mask & (labels[region]==0), seed)
                evidence = local_surface_evidence(mask, visible[region], spacing, config, prepared.image[region])
                candidate = (region, mask, seed)
                supported_signal = not guidance or any(point_inside(candidate, s['xyz'], prepared) for s in guidance)
                if geometry_valid(mask, spacing) and evidence_passes(evidence, config) and supported_signal:
                    cell = int(labels.max())+1; labels[region][mask] = cell
                    method = 'signal_guided' if signals is not None else ('local' if name=='original' else 'retry')
                    row.update(accepted=True, cell_id=cell, rejection_reason=None,
                               volume_um3=float(mask.sum()*np.prod(spacing)), recovery_method=method, local_evidence=evidence)
                    summary[method] += 1; accepted = True
        row['recovery'] = dict(accepted=accepted, method=method, attempts=attempt_log,
                               stable_pairs=len(stable), signal_ids=[s.get('signal_id') for s in guidance])
    emit(dict(kind='progress', phase='Cell recovery checks complete', current=len(proposals), total=len(proposals)))
    return labels, diagnostics, summary


def candidate_labels(labels, candidates, diagnostics, spacing=None, search_um=0.):
    """Provisional NN masks for signal searches, never publishable cell contours."""
    output = np.zeros_like(labels)
    for row in diagnostics:
        if row['accepted'] or row['proposal_id'] not in candidates: continue
        region, mask, seed = candidates[row['proposal_id']]
        if search_um:
            mask = ndi.distance_transform_edt(~mask, sampling=spacing) <= search_um
        mask = seeded_component(mask & (labels[region]==0) & (output[region]==0), seed)
        if mask.any(): output[region][mask] = row['proposal_id']
    return output
