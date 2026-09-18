from dataclasses import dataclass, replace
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from ..pipeline.evaluation import rasterize_reference
from ..pipeline.runner import acquisition_group, split_cases, _json
from ..pipeline.segmentation import Prepared, prepare, checkpoint, mask_properties
from ..reader import load_dataset
from ..models import LibraryEntry


@dataclass(eq=False)
class NeuralCase:
    entry: LibraryEntry
    prepared: Prepared
    references: list
    names: list
    group: str
    directory: Path
    labels: np.ndarray
    reference_cut: list[bool]
    audit: dict


def is_cut(box, mask, shape):
    points = np.argwhere(mask)
    if not len(points):
        return False
    low = points.min(0) + [s.start for s in box]
    high = points.max(0) + [s.start for s in box]
    return bool(np.any(low == 0) or np.any(high == np.asarray(shape)-1))


def sparse_labels(shape, references, names, overlap_policy='discard'):
    """Unknown=-1, complete disjoint manual objects=positive integer IDs.

    An overlapping object is discarded in its entirety, so no partially labeled
    object is sent to StarDist. Cut-off references remain available for scoring.
    No unknown pixel is inferred to be background (0).
    """
    labels = np.full(shape, -1, np.int16 if len(references)<32767 else np.int32)
    cuts = [is_cut(box, mask, shape) for box, mask in references]
    overlap_ids = set()
    for i, ((box, mask), cut) in enumerate(zip(references, cuts), 1):
        if cut:
            continue
        occupied = np.unique(labels[box][mask])
        occupied = occupied[occupied > 0]
        if len(occupied):
            overlap_ids.update(int(v) for v in occupied)
            overlap_ids.add(i)
        labels[box][mask] = i
    for i in overlap_ids:
        box, mask = references[i-1]
        labels[box][mask] = -1
    used = [int(i) for i in np.unique(labels) if i > 0]
    audit = dict(reference_count=len(references), complete_references=int(sum(not c for c in cuts)),
        cut_references=int(sum(cuts)), training_objects=len(used),
        overlapping_objects_ignored=[names[i-1] for i in sorted(overlap_ids)],
        foreground_voxels=int((labels>0).sum()), background_voxels=0,
        ignored_voxels=int((labels<0).sum()), label_policy='Complete disjoint manual objects; all other voxels ignored.')
    audit['training_reference_ids'] = used if overlap_policy=='discard' else [i for i,cut in enumerate(cuts,1) if not cut]
    audit['training_objects'] = len(audit['training_reference_ids'])
    audit['overlap_policy'] = overlap_policy
    if overlap_policy=='isolate':
        audit['overlapping_objects_retained_in_crops'] = audit.pop('overlapping_objects_ignored')
        audit['label_policy'] = 'Each complete reference is supervised in its own crop; other cells are unknown. Dense preview shows disjoint cells only.'
    return labels, cuts, audit


def prepare_neural(dataset, config, emit=None, cancel=None):
    geometry=config.geometry()
    if config.native_sampling:
        geometry=replace(geometry,voxel_um=max(config.voxel_um,*dataset.spacing[:2]))
    return prepare(dataset,geometry,emit,cancel,background_correct=False,max_voxels=None)


def three_way_split(cases):
    eligible = [c for c in cases if c.audit['training_objects']]
    development, testing = split_cases(eligible, 0.2)
    try:
        training, validation = split_cases(development, 0.2)
    except ValueError as exc:
        raise ValueError('Neural training requires at least three independent acquisition groups with complete manual cells '
                         '(training, validation, and final test).') from exc
    return training, validation, testing


def load_cases(entries, config, directory, emit, cancel=None):
    unique = {}
    for entry in entries:
        key = entry.image.resolve()
        old = unique.get(key)
        if old is None or (entry.scene and not old.scene) or (entry.scene and entry.scene.stem == entry.image.stem):
            unique[key] = entry
    cases, seen = [], {}
    for index, entry in enumerate(sorted(unique.values(), key=lambda e:str(e.image)), 1):
        checkpoint(cancel)
        emit(dict(kind='file', file=str(entry.image), phase='Preparing neural labels', current=index, total=len(unique)))
        dataset = load_dataset(entry, time=0, level=0)
        # Keep the trained physical scale for every field of view. Neural
        # inference uses tiles rather than coarsening large image volumes.
        prepared = prepare_neural(dataset, config, emit, cancel)
        refs, names, skipped = rasterize_reference(dataset.scene.surfaces, prepared, 0, cancel)
        if prepared.fingerprint in seen:
            emit(dict(kind='warning', message=f'Exact pixel duplicate skipped: {entry.image}; same as {seen[prepared.fingerprint]}'))
            continue
        seen[prepared.fingerprint] = str(entry.image)
        labels, cuts, audit = sparse_labels(prepared.image.shape, refs, names, config.overlap_policy)
        # Optional, explicitly reviewed background on the same processing grid.
        background_file = entry.image.with_suffix('.background.npz')
        if background_file.exists():
            with np.load(background_file, allow_pickle=False) as archive:
                bg = archive['background']
                if str(archive['fingerprint']) != prepared.fingerprint or bg.shape != labels.shape or bg.dtype != bool:
                    raise ValueError(f'Background mask does not match the source image/grid: {background_file}')
                if not np.allclose(archive['spacing_zyx'], prepared.spacing) or not np.allclose(archive['origin_xyz'], prepared.origin):
                    raise ValueError(f'Background mask calibration differs: {background_file}')
            annotated_union = np.zeros(labels.shape, bool)
            for box, mask in refs:
                annotated_union[box] |= mask
            if np.any(bg & annotated_union):
                raise ValueError(f'Reviewed background overlaps a reference cell: {background_file}')
            labels[bg] = 0
            audit['background_voxels'] = int(bg.sum())
            audit['ignored_voxels'] = int((labels<0).sum())
            audit['background_file'] = str(background_file)
        case_dir = directory / f'{index:04d}-{entry.image.stem}'
        case_dir.mkdir()
        np.save(case_dir/'image.npy', prepared.image)
        np.save(case_dir/'labels.npy', labels)
        prepared.image = np.load(case_dir/'image.npy', mmap_mode='r')
        labels = np.load(case_dir/'labels.npy', mmap_mode='r')
        audit.update(image=str(entry.image), channel=prepared.channel, channel_scores=prepared.channel_scores,
            shape_zyx=list(labels.shape), spacing_zyx=list(prepared.spacing), origin_xyz=list(prepared.origin),
            fingerprint=prepared.fingerprint, warnings=skipped+prepared.warnings+dataset.scene.warnings)
        _json(case_dir/'labels.json', audit)
        emit(dict(kind='audit', audit=audit, file=str(entry.image), preview=np.asarray(labels[len(labels)//2])))
        cases.append(NeuralCase(entry, prepared, refs, names, acquisition_group(entry), case_dir, labels, cuts, audit))
    return cases


class ReferencePatches:
    """Read one complete object crop at a time from calibrated memory maps."""
    def __init__(self,cases,config):
        self.size=np.array((config.patch_z,config.patch_xy,config.patch_xy));self.items=[]
        for case in cases:
            for number in case.audit['training_reference_ids']:
                box,mask=case.references[number-1]
                if config.curate_training_shapes:
                    from .precision import shape_eligible
                    if not shape_eligible(mask_properties(mask,np.asarray(case.prepared.spacing)),config):continue
                points=np.argwhere(mask)+[s.start for s in box]
                low,high=points.min(0),points.max(0)+1
                if np.any(high-low>self.size-2):
                    raise ValueError(f'{case.entry.image.name}: reference {number} needs a larger patch than {tuple(self.size)}.')
                start=np.clip((low+high)//2-self.size//2,0,np.maximum(0,np.array(case.labels.shape)-self.size))
                stop=np.minimum(np.array(case.labels.shape),start+self.size)
                self.items.append((case,number,start,stop))

    def __len__(self):return len(self.items)

    @lru_cache(maxsize=8)
    def pair(self,index):
        case,number,start,stop=self.items[index]
        crop=tuple(slice(int(a),int(b)) for a,b in zip(start,stop))
        y=np.full(tuple(stop-start),-1,np.int16)
        y[case.labels[crop]==0]=0
        box,mask=case.references[number-1]
        lower=np.maximum(start,[s.start for s in box]);upper=np.minimum(stop,[s.stop for s in box])
        target=tuple(slice(int(a-b),int(c-b)) for a,b,c in zip(lower,start,upper))
        source=tuple(slice(int(a-s.start),int(b-s.start)) for a,b,s in zip(lower,upper,box))
        y[target][mask[source]]=1
        if int((y>0).sum())!=int(mask.sum()):raise ValueError('Training crop truncated its target reference.')
        pad=[(0,int(n-s)) for n,s in zip(self.size,y.shape)]
        x=np.pad(case.prepared.image[crop],pad,mode='reflect').astype(np.float32)
        y=np.pad(y,pad,mode='constant',constant_values=-1)
        if not (y[:,::2,::2]>0).any():raise ValueError('No supervised foreground survives the network output grid.')
        return x,y


class PatchView:
    def __init__(self,patches,which):self.patches,self.which=patches,which
    def __len__(self):return len(self.patches)
    def __getitem__(self,index):return self.patches.pair(int(index))[self.which]


class WithSyntheticBackground:
    """Explicit synthetic blank controls; never relabel unknown source voxels."""
    def __init__(self,images,labels,config):
        self.images,self.labels,self.config=images,labels,config
        self.count=int(np.ceil(len(images)*config.synthetic_background_fraction))
    def __len__(self):return len(self.images)+self.count
    @lru_cache(maxsize=4)
    def pair(self,index):
        if index<len(self.images):return self.images[index],self.labels[index]
        if index>=len(self):raise IndexError(index)
        rng=np.random.default_rng(self.config.seed+index)
        shape=(self.config.patch_z,self.config.patch_xy,self.config.patch_xy)
        image=np.maximum(0,rng.normal(0,.002,shape)).astype(np.float32)
        return image,np.zeros(shape,np.int16)


def add_synthetic_background(images,labels,config):
    if not config.synthetic_background_fraction:return images,labels,0
    data=WithSyntheticBackground(images,labels,config)
    return PatchView(data,0),PatchView(data,1),data.count


def training_arrays(cases, config):
    """Object-centered patches keep supervised voxels on the output grid.

    Random full-volume crops can contain only a tiny annotated sliver that
    disappears on StarDist's subsampled grid, causing an empty masked loss.
    Every crop here retains its entire target cell; clipped neighbors and
    padding are ignored. Photometric and spatial augmentation stays stochastic.
    """
    if config.overlap_policy=='isolate' or config.curate_training_shapes:
        patches=ReferencePatches(cases,config)
        return PatchView(patches,0),PatchView(patches,1)
    images, labels = [], []
    requested = np.array((config.patch_z, config.patch_xy, config.patch_xy))
    for case in cases:
        objects=ndi.find_objects(np.maximum(case.labels,0))
        shape=np.array(case.labels.shape)
        for number, box in enumerate(objects,1):
            if box is None:continue
            size=np.array([s.stop-s.start for s in box])
            if np.any(size>requested-2):
                raise ValueError(f'{case.entry.image.name}: cell {number} needs a larger training patch than {tuple(requested)}.')
            center=np.array([(s.start+s.stop)//2 for s in box])
            low=np.clip(center-requested//2,0,np.maximum(0,shape-requested))
            high=np.minimum(shape,low+requested)
            crop=tuple(slice(int(a),int(b)) for a,b in zip(low,high))
            y=np.array(case.labels[crop])
            for other in np.unique(y[y>0]):
                bounds=objects[other-1]
                if any(s.start<a or s.stop>b for s,a,b in zip(bounds,low,high)):
                    y[y==other]=-1
            pad=[(0,int(want-have)) for want,have in zip(requested,y.shape)]
            x=np.pad(case.prepared.image[crop],pad,mode='reflect').astype(np.float32)
            y=np.pad(y,pad,mode='constant',constant_values=-1)
            if not (y[:,::2,::2]>0).any():
                raise ValueError('No supervised foreground survives on the network output grid; use a finer voxel size.')
            images.append(x);labels.append(y)
    return images, labels


def subset_metrics(predicted, case, evaluate, cancel=None):
    def score(selection):
        refs = [r for r,keep in zip(case.references,selection) if keep]
        names = [n for n,keep in zip(case.names,selection) if keep]
        return evaluate(predicted, refs, names, case.prepared.spacing, cancel)
    metrics = score([True]*len(case.references))
    metrics['complete_cells'] = score([not cut for cut in case.reference_cut])
    metrics['cut_cells'] = score(case.reference_cut)
    return metrics
