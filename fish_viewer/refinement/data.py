"""Calibrated membrane data, exact manual-mesh targets and target-cell seeds."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.transform import resize

from ..models import LibraryEntry
from ..reader import load_dataset
from ..scenes import read_scene
from ..neural.audit import read_references, save_references, unique_references
from ..neural.data import is_cut
from ..pipeline.evaluation import rasterize_reference
from ..pipeline.runner import _json
from ..pipeline.segmentation import Prepared, checkpoint, normalize
from ..signals.engine import image_fingerprint
from ..workflow.volumes import surface_volume


def prepare_membrane(dataset, channel, config, fingerprint=None):
    # Do not invent axial resolution: retain native spacing where it is coarser.
    physical = np.asarray(dataset.metadata.extent[::-1])
    requested = np.maximum(config.voxel_um, np.asarray(dataset.spacing[::-1]))
    shape = np.maximum(2, np.rint(physical/requested).astype(int))
    spacing = physical/shape
    image = resize(normalize(dataset.volumes[channel]), tuple(shape), order=1,
                   preserve_range=True, anti_aliasing=True).astype(np.float32)
    origin = tuple(np.asarray(dataset.metadata.origin)+spacing[::-1]/2)
    return Prepared(image, tuple(spacing), origin, channel, [],
                    fingerprint or image_fingerprint(dataset), {}, {})


def cache_collection(audit_path, output, config, emit=lambda e: None, cancel=None):
    """Reuse acquisition split/channel choices, re-rasterize original closed meshes."""
    source = json.loads(Path(audit_path).read_text(encoding='utf-8'))
    if source['status'] != 'complete':
        raise ValueError('A complete acquisition audit is required.')
    for item in source['sources']:
        stat=Path(item['image']).stat()
        if stat.st_size!=item['size'] or stat.st_mtime_ns!=item['mtime_ns']:
            raise ValueError('Source changed since the audit; regenerate it: '+item['image'])
        for scene in item.get('scene_stats', []):
            stat=Path(scene['path']).stat()
            if stat.st_size!=scene['size'] or stat.st_mtime_ns!=scene['mtime_ns']:
                raise ValueError('Manual annotations changed since the audit: '+scene['path'])
    excluded = {'18327antiLAD2.ims','18327antiLAD3.ims','18327antiLAD5.ims','3AantiLADFITCTyrCy5_546.ims'}
    if any(Path(alias).name in excluded for c in source['cases'] for alias in c['audit']['aliases']):
        raise ValueError('The four user-excluded scans/aliases must be excluded from the audit.')
    groups = {split: {c['audit']['group'] for c in source['cases'] if c['audit']['split']==split}
              for split in ('training','validation','test')}
    if any(groups[a]&groups[b] for a,b in [('training','validation'),('training','test'),('validation','test')]):
        raise ValueError('Acquisition groups overlap between splits.')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    target = output/'audit.json'
    settings = dict(config=dict(voxel_um=config.voxel_um),
                    source_sha256=hashlib.sha256(Path(audit_path).read_bytes()).hexdigest())
    if target.exists():
        report = json.loads(target.read_text())
        identity = dict(report['cache_identity'])
        identity['config'] = {k:identity['config'][k] for k in settings['config']}
        if identity != settings:
            raise ValueError('This cache belongs to another audit/config; choose a new cache folder.')
    else:
        report = dict(version=1, engine='seeded-unet3d', status='running', config=source['config'],
                      cache_identity=settings, parent_audit=str(Path(audit_path).resolve()),
                      sources=source['sources'], cases=[], invalid_meshes=[],
                      policy='Original disjoint acquisition split and channel choices; all complete closed '
                             'manual cells, no roundness or elongation selection.')
    done = {c['image'] for c in report['cases']}
    sources = {s['image']: s for s in source['sources']}
    for number, original in enumerate(source['cases'], 1):
        checkpoint(cancel)
        if original['image'] in done:
            continue
        emit(dict(kind='progress', phase='Preparing finer membrane images and manual boundaries',
                  current=number-1, total=len(source['cases']), file=original['image']))
        dataset = load_dataset(LibraryEntry(Path(original['image'])))
        fingerprint = image_fingerprint(dataset)
        if fingerprint != original['audit']['fingerprint']:
            raise ValueError('Source pixels differ from the acquisition audit: '+original['image'])
        prepared = prepare_membrane(dataset, original['audit']['channel'], config, fingerprint)
        scenes = sorted({p for alias in original['audit']['aliases'] for p in sources[alias]['scenes']})
        surfaces = []
        for path in scenes:
            for surface in read_scene(Path(path)).surfaces:
                if surface.time != 0:
                    continue
                if surface_volume(surface) is None:
                    report['invalid_meshes'].append(dict(image=original['image'], scene=path, name=surface.name))
                else:
                    surfaces.append(surface)
        refs, names, skipped = rasterize_reference(surfaces, prepared, cancel=cancel)
        refs, names = unique_references(refs, names)
        cuts = [is_cut(box, mask, prepared.image.shape) for box, mask in refs]
        folder = output/f'{number:04d}-{Path(original["image"]).stem}'
        folder.mkdir(exist_ok=True)
        np.save(folder/'image.npy', prepared.image)
        save_references(folder/'references.npz', refs, names)
        a = dict(original['audit'], shape_zyx=list(prepared.image.shape), spacing_zyx=list(prepared.spacing),
                 origin_xyz=list(prepared.origin), reference_cut=cuts, reference_count=len(refs),
                 complete_references=sum(not c for c in cuts), training_objects=sum(not c for c in cuts),
                 training_reference_ids=[i+1 for i,c in enumerate(cuts) if not c], shape_exclusions=[],
                 warnings=skipped, background_voxels=0)
        report['cases'].append(dict(original, directory=str(folder), audit=a))
        _json(target, report)
    report['status'] = 'complete'
    report['summary'] = {split: dict(scans=sum(c['audit']['split']==split for c in report['cases']),
                                    complete_cells=sum(c['audit']['complete_references'] for c in report['cases']
                                                       if c['audit']['split']==split))
                         for split in ('training','validation','test')}
    _json(target, report)
    return report


def coordinates(center, width_um, spacing, size):
    return np.meshgrid(*[center[a]+(np.arange(size)-(size-1)/2)*width_um/size/spacing[a]
                         for a in range(3)], indexing='ij')


def crop_geometry(box, spacing, margin_um):
    low = np.array([s.start for s in box]); high = np.array([s.stop for s in box])
    return (low+high-1)/2, float(np.max((high-low)*spacing)+2*margin_um)


def seed_center(mask, spacing):
    distance = ndi.distance_transform_edt(np.pad(mask, 1), sampling=spacing)[1:-1,1:-1,1:-1]
    return np.array(np.unravel_index(distance.argmax(), mask.shape), dtype=float)


def seed_image(center, spacing_um, config):
    axes = [(np.arange(config.patch_size)-center[a])*spacing_um for a in range(3)]
    return np.exp(-sum(v*v for v in np.meshgrid(*axes, indexing='ij')) /
                  (2*config.seed_sigma_um**2)).astype(np.float32)


def sample_target(image, box, mask, spacing, config):
    center, width = crop_geometry(box, np.asarray(spacing), config.margin_um)
    coords = coordinates(center, width, spacing, config.patch_size)
    sample = ndi.map_coordinates(image, coords, order=1, mode='constant', cval=0, prefilter=False)
    local = [v-s.start for v,s in zip(coords,box)]
    truth = ndi.map_coordinates(mask.astype(np.uint8), local, order=0, mode='constant', cval=0)>0
    valid = np.logical_and.reduce([(v>=0)&(v<=n-1) for v,n in zip(coords,image.shape)])
    step = width/config.patch_size
    inside = ndi.distance_transform_edt(truth, sampling=step)
    outside = ndi.distance_transform_edt(~truth, sampling=step)
    weight = (1+(config.boundary_weight-1)*np.exp(-((inside+outside)/config.boundary_band_um)**2))*valid
    seed = seed_center(mask, spacing)+np.array([s.start for s in box])
    seed_patch = (seed-center)*np.asarray(spacing)/step+(config.patch_size-1)/2
    return sample, truth, weight.astype(np.float32), valid, seed_patch, step


def cache_patches(collection, output, config, emit=lambda e: None, cancel=None):
    """One target-conditioned crop for every complete training/validation cell."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    target = output/'index.json'
    geometry_keys = ('patch_size','margin_um','boundary_band_um','boundary_weight')
    identity = dict(config={k:getattr(config,k) for k in geometry_keys}, cases_sha256=hashlib.sha256(
        json.dumps(collection['cases'], sort_keys=True).encode()).hexdigest())
    if target.exists():
        old = json.loads(target.read_text())
        previous = dict(old['identity'])
        previous['config'] = {k:previous['config'][k] for k in geometry_keys}
        if previous != identity:
            raise ValueError('Patch cache belongs to another collection/config.')
        return old
    items = []
    for case in collection['cases']:
        if case['audit']['split'] not in ('training','validation'):
            continue
        refs, names = read_references(Path(case['directory'])/'references.npz')
        for i, (box, mask) in enumerate(refs):
            if not case['audit']['reference_cut'][i]:
                items.append((case, i, box, mask, names[i]))
    size = config.patch_size; n = len(items)
    arrays = {name: np.lib.format.open_memmap(output/(name+'.npy'), mode='w+', dtype=dtype,
                                             shape=(n,size,size,size))
              for name,dtype in [('images',np.float16),('masks',np.uint8),('weights',np.float16),('valid',np.uint8)]}
    records = []
    for index, (case, ref, box, mask, name) in enumerate(items):
        checkpoint(cancel)
        emit(dict(kind='progress', phase='Preparing target-cell crops and boundary weights',
                  current=index, total=n, file=case['image']))
        image = np.load(Path(case['directory'])/'image.npy', mmap_mode='r')
        sample, truth, weight, valid, seed, step = sample_target(image,box,mask,case['audit']['spacing_zyx'],config)
        for key,value in [('images',sample),('masks',truth),('weights',weight),('valid',valid)]:
            arrays[key][index] = value
        records.append(dict(index=index, image=case['image'], reference_id=ref+1, name=name,
                            split=case['audit']['split'], group=case['audit']['group'],
                            seed_patch=seed.tolist(), patch_spacing_um=step))
    for array in arrays.values():
        array.flush()
    result = dict(identity=identity, records=records, policy='Complete closed target-cell membership is '
                  'supervised throughout its crop. Other cells are not this target, not global background. '
                  'Padding is ignored. No test cell is used in training or checkpoint selection.')
    _json(target, result)
    return result
