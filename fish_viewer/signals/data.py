"""Physical-coordinate crops and supervision confined to fully marked cells."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.distance import cdist

from ..models import LibraryEntry
from ..reader import load_dataset
from ..scenes import read_scene
from ..neural.audit import read_references
from ..pipeline.runner import _json
from ..pipeline.segmentation import checkpoint
from .config import SignalConfig


def normalized(volume):
    sample=volume.ravel()[::max(1,volume.size//1_000_000)]
    lo,hi=np.percentile(sample,(1,99.9))
    if hi<=lo:lo,hi=float(volume.min()),float(volume.max())
    if hi<=lo:return np.zeros(volume.shape,np.float32)
    return np.clip((volume.astype(np.float32)-lo)/(hi-lo),0,1)


def xyz_to_grid(points,origin,spacing):
    return (np.asarray(points).reshape(-1,3)-np.asarray(origin))[:,::-1]/np.asarray(spacing)


def at_labels(labels,points,origin,spacing):
    coords=xyz_to_grid(points,origin,spacing)
    return ndi.map_coordinates(labels,coords.T,order=0,mode='constant',cval=0).astype(np.int32)


def native_labels(labels,metadata,dataset):
    # Separable affine mapping retains calibrated voxel centers and discrete IDs.
    spacing=np.asarray(dataset.spacing[::-1])
    scale=spacing/metadata['spacing_zyx']
    offset=(np.asarray(dataset.center_origin)-metadata['origin_xyz'])[::-1]/metadata['spacing_zyx']
    return ndi.affine_transform(labels,np.diag(scale),offset=offset,
        output_shape=dataset.volumes[0].shape,order=0,mode='constant',cval=0,prefilter=False)


def marker_points(scenes):
    unique={}
    for path in scenes:
        for group in read_scene(Path(path)).measurements:
            for point,label,time_index in zip(group.points,group.labels,group.times):
                if label.strip().upper()=='A' and time_index==0:
                    key=tuple(np.round(point,4))
                    unique.setdefault(key,dict(xyz=point.tolist(),sources=[]))['sources'].append(str(path)+'::'+group.name)
    return list(unique.values())


def reference_memberships(points,references,origin,spacing,tolerance=0):
    coords=xyz_to_grid(points,origin,spacing)
    members=[[] for _ in coords]
    distances=np.full((len(coords),len(references)),np.inf)
    for index,(box,mask) in enumerate(references):
        local=coords-np.array([s.start for s in box])
        if not len(coords):continue
        inside=ndi.map_coordinates(mask.astype(np.uint8),local.T,order=0,mode='constant',cval=0)>0
        for p in np.flatnonzero(inside):members[p].append(index)
        if tolerance:
            distance=ndi.distance_transform_edt(~mask,sampling=spacing)
            distances[:,index]=ndi.map_coordinates(distance,local.T,order=1,mode='constant',cval=1e6)
    for i,found in enumerate(members):
        if not found and tolerance and len(references):
            near=int(distances[i].argmin())
            if distances[i,near]<=tolerance:found.append(near)
    return members


def channel_evidence(image,point,origin,spacing):
    center=xyz_to_grid([point],origin,spacing)[0]
    margin=np.ceil(.9/np.asarray(spacing)).astype(int)+1
    low=np.maximum(0,np.floor(center).astype(int)-margin)
    high=np.minimum(image.shape,np.ceil(center).astype(int)+margin+1)
    if np.any(high<=low):return dict(score=0.,contrast=0.)
    box=tuple(slice(int(a),int(b)) for a,b in zip(low,high))
    radius=np.linalg.norm((np.moveaxis(np.indices(tuple(high-low)),0,-1)+low-center)*spacing,axis=-1)
    patch=image[box];core=patch[radius<=.35];shell=patch[(radius>=.55)&(radius<=.9)]
    if not core.size or shell.size<8:return dict(score=0.,contrast=0.)
    baseline=float(np.median(shell));contrast=float(core.max()-baseline)
    noise=max(.04,float(np.median(np.abs(shell-baseline)))*1.4826)
    return dict(score=max(0.,contrast/noise),contrast=contrast)


def infer_channels(images,points,membrane,origin,spacing,config):
    rows=[]
    for point in points:
        scores=[channel_evidence(image,point['xyz'],origin,spacing) for image in images]
        order=sorted((c for c in range(len(images)) if c!=membrane),key=lambda c:scores[c]['score'],reverse=True)
        channel=order[0] if order else None
        contrast_min=config.channel_contrast_min if config.channel_contrast_min is not None else config.candidate_contrast
        good=channel is not None and scores[channel]['score']>=config.channel_score_min and scores[channel]['contrast']>=contrast_min
        if len(order)>1:good=good and scores[channel]['score']>=config.channel_margin*max(.01,scores[order[1]]['score'])
        rows.append(dict(point,channel=channel if good else None,channel_evidence=scores))
    return rows


def candidates(image,cells,spacing,origin,config):
    spacing=np.asarray(spacing)
    if config.candidate_scales_um:
        return multiscale_candidates(image,cells,spacing,origin,config)
    smooth=ndi.gaussian_filter(image,np.maximum(.55,config.candidate_sigma_um/spacing))
    background=ndi.gaussian_filter(image,np.maximum(.8,config.candidate_background_um/spacing))
    response=np.maximum(0,smooth-background)
    # No candidate is manufactured in a uniform image. Candidate centers, not
    # their whole patches, must be inside a predicted cell (no erosion).
    size=tuple((2*np.maximum(1,np.floor(config.candidate_separation_um/spacing).astype(int))+1).tolist())
    peaks=(response==ndi.maximum_filter(response,size=size,mode='nearest'))&(response>=config.candidate_contrast)&(cells>0)
    points=np.argwhere(peaks)
    if not len(points):return []
    values=response[tuple(points.T)];order=np.argsort(-values,kind='stable')
    # Spatial bins enforce a physical separation without a quadratic all-pairs matrix.
    accepted=[];bins={};distance=config.candidate_separation_um
    from itertools import product
    for index in order:
        p=points[index];physical=p*spacing;key=tuple(np.floor(physical/distance).astype(int))
        neighbours=[old for offset in product((-1,0,1),repeat=3) for old in bins.get(tuple(np.array(key)+offset),[])]
        if any(np.linalg.norm(physical-old)<distance for old in neighbours):continue
        bins.setdefault(key,[]).append(physical)
        xyz=(physical[::-1]+origin).tolist()
        accepted.append(dict(xyz=xyz,cell_id=int(cells[tuple(p)]),contrast=float(values[index])))
    return accepted


def multiscale_candidates(image,cells,spacing,origin,config):
    """Merge 3D peaks across physical spot sizes; the CNN makes acceptance decisions.

    A native-voxel neighbourhood finds peaks before physical-distance suppression.
    Unlike the legacy large rectangular maximum filter, it does not hide a nearby
    dim peak merely because a brighter peak lies elsewhere in that rectangle.
    Old checkpoints without candidate_scales_um retain their original algorithm.
    """
    from itertools import product
    proposed=[]
    for scale in sorted(set(config.candidate_scales_um)):
        smooth=ndi.gaussian_filter(image,np.maximum(.55,scale/spacing))
        broad=max(config.candidate_background_um,scale*config.candidate_background_ratio)
        background=ndi.gaussian_filter(image,np.maximum(.8,broad/spacing))
        response=np.maximum(0,smooth-background)
        peaks=(response==ndi.maximum_filter(response,size=3,mode='nearest')) & (response>=config.candidate_contrast) & (cells>0)
        positions=np.argwhere(peaks)
        proposed.extend((float(response[tuple(p)]),tuple(p),float(scale)) for p in positions)
    proposed.sort(key=lambda p:-p[0])
    accepted=[];bins={};distance=config.candidate_separation_um
    neighbours=list(product((-1,0,1),repeat=3))
    for value,position,scale in proposed:
        physical=np.asarray(position)*spacing
        cell=int(cells[position]);key=np.floor(physical/distance).astype(int)
        near=[old for delta in neighbours for old in bins.get((cell,*tuple(key+delta)),[])]
        if any(np.linalg.norm(physical-old)<distance for old in near):continue
        bins.setdefault((cell,*tuple(key)),[]).append(physical)
        accepted.append(dict(xyz=(physical[::-1]+origin).tolist(),cell_id=cell,
                             contrast=value,candidate_scale_um=scale))
    return accepted


def extract_patches(images,membrane,cells,rows,origin,spacing,config):
    size=config.patch_size
    offsets=(np.moveaxis(np.indices((size,)*3),0,-1)-(size-1)/2)*config.patch_spacing_um
    result=np.empty((len(rows),size,size,size,3),np.float16)
    for index,row in enumerate(rows):
        coords=xyz_to_grid([row['xyz']],origin,spacing)[0]+offsets/spacing
        result[index,...,0]=ndi.map_coordinates(images[row['channel']],np.moveaxis(coords,-1,0),order=1,mode='reflect')
        result[index,...,1]=ndi.map_coordinates(images[membrane],np.moveaxis(coords,-1,0),order=1,mode='reflect')
        ids=ndi.map_coordinates(cells,np.moveaxis(coords,-1,0),order=0,mode='constant',cval=0)
        result[index,...,2]=ids==row['cell_id']
    return result


def candidate_targets(rows,truth,supervised,config,exclusion_points=None):
    targets=np.full(len(rows),-1,np.int8)
    for index,row in enumerate(rows):
        if not supervised[index]:continue
        points=[p['xyz'] for p in truth if p['channel']==row['channel'] and p['cell_id']==row['cell_id']]
        distance=min((np.linalg.norm(np.array(row['xyz'])-p) for p in points),default=np.inf)
        if distance<=config.match_radius_um and not config.one_to_one_positives:targets[index]=1
        elif distance>=config.negative_radius_um:targets[index]=0
    if config.one_to_one_positives and rows and truth:
        from scipy.optimize import linear_sum_assignment
        distances=cdist([p['xyz'] for p in rows],[p['xyz'] for p in truth])
        compatible=np.array([[bool(supervised[i]) and p['channel']==r['channel'] and p['cell_id']==r['cell_id']
                              for r in truth] for i,p in enumerate(rows)])
        cost=np.where(compatible & (distances<=config.match_radius_um),distances,1e6)
        left,right=linear_sum_assignment(cost)
        targets[left[cost[left,right]<1e6]]=1
    # A visible annotated signal remains an unsafe negative even if its center
    # falls outside this predicted cell or its channel cannot be resolved.
    if exclusion_points:
        from scipy.spatial import cKDTree
        for channel in {p['channel'] for p in rows}:
            points=[p['xyz'] for p in exclusion_points if p['channel'] in (channel,None)]
            indices=[i for i,p in enumerate(rows) if p['channel']==channel and targets[i]==0]
            if points and indices:
                distances=cKDTree(points).query([rows[i]['xyz'] for i in indices])[0]
                targets[np.asarray(indices)[distances<config.negative_radius_um]]=-1
    return targets


def prepare_supervision(case,sources,row,config):
    """Load one scan with checked pixels, grid and sparse manual supervision."""
    source=sources[case['image']]
    for alias in case['audit']['aliases']:
        s=sources[alias];stat=Path(alias).stat()
        if stat.st_size!=s['size'] or stat.st_mtime_ns!=s['mtime_ns']:
            raise ValueError('Source changed since cell audit.')
        if s['fingerprint']!=source['fingerprint']:
            raise ValueError('Aliases differ in raw channels; a new split-safe signal audit is required.')
        for scene in s['scene_stats']:
            st=Path(scene['path']).stat()
            if st.st_size!=scene['size'] or st.st_mtime_ns!=scene['mtime_ns']:
                raise ValueError('Annotations changed since cell audit.')
    with np.load(row['predictions'],allow_pickle=False) as archive:
        labels=archive['labels'];metadata=json.loads(str(archive['metadata']))
    if metadata['fingerprint']!=source['fingerprint']:
        raise ValueError('Cell predictions do not match source pixels.')
    for field in ('spacing_zyx','origin_xyz'):
        if not np.allclose(metadata[field],case['audit'][field],rtol=1e-6,atol=1e-8):
            raise ValueError('Reference and prediction grids differ; use the matching calibrated collection audit.')
    scenes=sorted({scene for alias in case['audit']['aliases'] for scene in sources[alias]['scenes']})
    points=marker_points(scenes)
    references,_=read_references(Path(case['directory'])/'references.npz')
    memberships=reference_memberships([p['xyz'] for p in points],references,metadata['origin_xyz'],metadata['spacing_zyx'],config.assignment_tolerance_um)
    dataset=load_dataset(LibraryEntry(Path(case['image']),Path(case['scene']) if case['scene'] else None))
    digest=hashlib.sha256();digest.update(str(dataset.volumes[0].shape).encode())
    for volume in dataset.volumes:digest.update(memoryview(np.ascontiguousarray(volume)))
    if digest.hexdigest()!=source['fingerprint']:
        raise ValueError('Decoded source pixels differ from the audited fingerprint.')
    images=[normalized(v) for v in dataset.volumes];membrane=int(metadata['channel'])
    origin=np.asarray(dataset.center_origin);spacing=np.asarray(dataset.spacing[::-1])
    points=infer_channels(images,points,membrane,origin,spacing,config)
    marked={m for group in memberships for m in group}
    ambiguous={m for point,group in zip(points,memberships) if point['channel'] is None for m in group}
    usable=marked-ambiguous
    roi=np.zeros(labels.shape,np.uint8)
    for ref in usable:
        box,mask=references[ref];roi[box][mask]=1
    for ref in ambiguous:
        box,mask=references[ref];roi[box][mask]=0
    roi*=labels>0
    truth=[]
    for point,group in zip(points,memberships):
        point['manual_cells']=group
        point['cell_id']=int(at_labels(labels,[point['xyz']],metadata['origin_xyz'],metadata['spacing_zyx'])[0])
        point['usable']=point['channel'] is not None and bool(set(group)&usable) and not bool(set(group)&ambiguous) and point['cell_id']>0
        if point['usable']:truth.append(point)
    return dict(source=source,labels=labels,metadata=metadata,roi=roi,dataset=dataset,images=images,
                membrane=membrane,origin=origin,spacing=spacing,points=points,truth=truth,
                marked=marked,ambiguous=ambiguous,usable=usable,
                native=native_labels(labels,metadata,dataset))


def supervised_candidates(prepared,config):
    rows=[]
    for channel,image in enumerate(prepared['images']):
        if channel==prepared['membrane']:continue
        rows.extend(dict(p,channel=channel) for p in candidates(image,prepared['native'],prepared['spacing'],prepared['origin'],config))
    metadata=prepared['metadata']
    supervised=at_labels(prepared['roi'],[p['xyz'] for p in rows],metadata['origin_xyz'],metadata['spacing_zyx'])>0
    return [p for p,valid in zip(rows,supervised) if valid],len(rows)


def build_audit(cell_report,audit_path,output,config=SignalConfig(),emit=print,cancel=None,split=None):
    """Use all retained acquisitions; never label unmarked cells as negative."""
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'audit.json').exists():raise ValueError('Choose a new output folder to preserve the existing audit.')
    cells_run=json.loads(Path(cell_report).read_text());collection=json.loads(Path(audit_path).read_text())
    if cells_run['status']!='complete' or collection['status']!='complete':raise ValueError('Completed cell run and audit required.')
    sources={s['image']:s for s in collection['sources']}
    cell_rows={r['image']:r for r in cells_run['files']}
    cases=[c for c in collection['cases'] if split is None or c['audit']['split']==split]
    report=dict(version=1,engine='signal-audit',status='running',directory=str(output),config=asdict(config),
        cell_report=str(Path(cell_report).resolve()),cell_model=cells_run['model_directory'],
        collection_audit=str(Path(audit_path).resolve()),cases=[],errors=[],
        supervision='A = signal center. All signals in marked cells annotated (user confirmed). Only predicted-cell interiors intersecting unambiguous marked manual cells are supervised. Other regions unknown.',
        split_policy=collection.get('split_policy',collection.get('policy')))
    started=time.monotonic()
    for number,case in enumerate(cases,1):
        checkpoint(cancel)
        emit(dict(kind='progress',phase='Signal annotation audit',current=number-1,total=len(cases),file=case['image']))
        try:
            row=cell_rows[case['image']]
            prepared=prepare_supervision(case,sources,row,config)
            source=prepared['source'];metadata=prepared['metadata'];dataset=prepared['dataset']
            images=prepared['images'];membrane=prepared['membrane'];origin=prepared['origin'];spacing=prepared['spacing']
            points=prepared['points'];truth=prepared['truth'];native=prepared['native'];labels=prepared['labels']
            marked=prepared['marked'];ambiguous=prepared['ambiguous'];usable=prepared['usable']
            proposed,all_candidate_count=supervised_candidates(prepared,config)
            targets=candidate_targets(proposed,truth,np.ones(len(proposed),bool),config,exclusion_points=points)
            # Keep ambiguous-distance candidates for honest detector evaluation;
            # their -1 targets are ignored only by the training/loss sampler.
            chosen=np.arange(len(proposed))
            directory=output/f'{number:04d}-{Path(case["image"]).stem}';directory.mkdir()
            patches=extract_patches(images,membrane,native,[proposed[i] for i in chosen],origin,spacing,config)
            np.save(directory/'patches.npy',patches);np.save(directory/'targets.npy',targets[chosen])
            selected=[]
            for patch_index,index in enumerate(chosen):selected.append(dict(proposed[index],target=int(targets[index]),patch_index=patch_index))
            # Retain every reference and its eligibility so upstream misses remain visible.
            record=dict(image=case['image'],scene=case['scene'],aliases=case['audit']['aliases'],split=case['audit']['split'],group=case['audit']['group'],
                directory=str(directory),cell_predictions=row['predictions'],fingerprint=source['fingerprint'],
                channels=[dict(name=c.name,color=list(c.color)) for c in dataset.metadata.channels],membrane_channel=membrane,
                origin_xyz=origin.tolist(),spacing_zyx=spacing.tolist(),shape=list(dataset.volumes[0].shape),
                predicted_cells=int(labels.max()),marked_cells=len(marked),ambiguous_cells=len(ambiguous),supervised_cells=len(usable),
                markers=points,truth=truth,candidates=selected,all_candidate_count=all_candidate_count,
                positive_patches=int((targets==1).sum()),negative_patches=int((targets==0).sum()),
                supervised_candidate_count=len(proposed))
            _json(directory/'case.json',record);report['cases'].append(record)
            del patches,images,dataset,native,prepared
        except Exception as exc:
            report['errors'].append(dict(image=case['image'],error=str(exc)))
            emit(dict(kind='warning',message=f'{case["image"]}: {exc}'))
        report['elapsed_s']=time.monotonic()-started
        _json(output/'audit.json',report)
    report['status']='complete' if not report['errors'] else 'failed'
    report['summary']={split:dict(scans=len(cs),markers=sum(len(c['markers']) for c in cs),
        assigned_markers=sum(sum(p['channel'] is not None for p in c['markers']) for c in cs),
        eligible_signals=sum(len(c['truth']) for c in cs),positive_patches=sum(c['positive_patches'] for c in cs),
        negative_patches=sum(c['negative_patches'] for c in cs),marked_cells=sum(c['marked_cells'] for c in cs),
        ambiguous_cells=sum(c['ambiguous_cells'] for c in cs),predicted_cells=sum(c['predicted_cells'] for c in cs))
        for split in ('training','validation','test') for cs in [[c for c in report['cases'] if c['split']==split]]}
    _json(output/'audit.json',report)
    emit(dict(kind='finished',report=dict(status=report['status'],directory=str(output),summary=report['summary'])))
    return report
