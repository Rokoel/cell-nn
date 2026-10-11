"""Every complete marked cell is a training target; unmarked regions are ignored."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from ..neural.audit import read_references
from ..pipeline.runner import _json
from ..pipeline.segmentation import checkpoint
from ..refinement.data import seed_center


def sparse_target(references,cuts,spacing,coords,config):
    shape=coords[0].shape
    heat=np.zeros(shape,np.float32);known=np.zeros(shape,bool)
    faces=np.zeros(shape+(6,),np.float32);box_weight=np.zeros(shape,np.float32)
    grid=np.asarray(coords)*np.asarray(spacing)[:,None,None,None]
    for (box,mask),cut in zip(references,cuts):
        if cut:continue
        low=np.array([s.start for s in box]);high=np.array([s.stop for s in box])
        if any(high[a]<coords[a].min() or low[a]>coords[a].max() for a in range(3)):continue
        local=[v-s.start for v,s in zip(coords,box)]
        inside=ndi.map_coordinates(mask.astype(np.uint8),local,order=0,mode='constant',cval=0)>0
        known|=inside
        seed=(seed_center(mask,spacing)+low)*spacing
        gaussian=np.exp(-np.sum((grid-seed[:,None,None,None])**2,axis=0)/(2*config.center_sigma_um**2))
        # Normalize quantized center peaks only when the center lies in this tile.
        if all(grid[a].min()<=seed[a]<=grid[a].max() for a in range(3)):
            gaussian/=max(float(gaussian.max()),1e-6)
        strongest=gaussian>heat;heat=np.maximum(heat,gaussian)
        bounds_low=(low-.5)*spacing;bounds_high=(high-.5)*spacing
        distance=np.moveaxis(np.concatenate([grid-bounds_low[:,None,None,None],
                                             bounds_high[:,None,None,None]-grid],axis=0),0,-1)
        core=strongest & inside & (gaussian>=.6)
        faces[core]=np.maximum(0,distance[core])/config.box_scale_um
        box_weight[core]=gaussian[core]
    # Gaussian tails outside marked cells are deliberately ignored, too.
    return np.concatenate([heat[...,None],known[...,None],faces,box_weight[...,None]],axis=-1)


def cache_patches(audit,output,config,emit=lambda e:None,cancel=None):
    audit=Path(audit);collection=json.loads(audit.read_text())
    if collection['status']!='complete':raise ValueError('Complete calibrated manual-cell cache is required.')
    excluded={'18327antiLAD2.ims','18327antiLAD3.ims','18327antiLAD5.ims','3AantiLADFITCTyrCy5_546.ims'}
    if any(Path(alias).name in excluded for c in collection['cases'] for alias in c['audit'].get('aliases',[c['image']])):
        raise ValueError('The four excluded scans must not appear in this collection.')
    groups={s:{c['audit']['group'] for c in collection['cases'] if c['audit']['split']==s} for s in ('training','validation','test')}
    if any(groups[a]&groups[b] for a,b in [('training','validation'),('training','test'),('validation','test')]):
        raise ValueError('Acquisition groups overlap between splits.')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    target=output/'index.json'
    keys=('patch_size','voxel_um','stride','center_sigma_um','box_scale_um','seed')
    identity=dict(audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),
                  config={k:getattr(config,k) for k in keys})
    if target.exists():
        index=json.loads(target.read_text())
        if index['identity']!=identity:raise ValueError('Choose a new proposal patch cache.')
        return index
    items=[]
    for case in collection['cases']:
        if case['audit']['split'] not in ('training','validation'):continue
        refs,names=read_references(Path(case['directory'])/'references.npz')
        spacing=np.array(case['audit']['spacing_zyx'])
        for i,(box,mask) in enumerate(refs):
            if case['audit']['reference_cut'][i]:continue
            seed=seed_center(mask,spacing)+[s.start for s in box]
            items.append((case,i,seed,names[i]))
    n=len(items);size=config.patch_size;out=size//config.stride
    images=np.lib.format.open_memmap(output/'images.npy',mode='w+',dtype=np.float16,shape=(n,size,size,size,1))
    targets=np.lib.format.open_memmap(output/'targets.npy',mode='w+',dtype=np.float16,shape=(n,out,out,out,9))
    records=[];rng=np.random.default_rng(config.seed);previous=None
    for number,(case,ref,seed,name) in enumerate(items):
        checkpoint(cancel)
        emit(dict(kind='progress',phase='Preparing marked-cell centers and padded boxes',
                  current=number,total=n,file=case['image']))
        if previous!=case['image']:
            image=np.load(Path(case['directory'])/'image.npy',mmap_mode='r')
            refs,_=read_references(Path(case['directory'])/'references.npz');previous=case['image']
        spacing=np.array(case['audit']['spacing_zyx']);box,mask=refs[ref]
        bbox_center=(np.array([s.start+s.stop-1 for s in box]))/2
        bbox_size=np.array([s.stop-s.start for s in box])*spacing
        room=(size*config.voxel_um-bbox_size)/2-config.voxel_um
        if (room<=0).any():raise ValueError('A marked cell needs a larger proposal training tile.')
        center=bbox_center+rng.uniform(-1,1,3)*np.minimum(room,3.)/spacing
        axes=[center[a]+(np.arange(size)-(size-1)/2)*config.voxel_um/spacing[a] for a in range(3)]
        coords=np.meshgrid(*axes,indexing='ij')
        images[number,...,0]=ndi.map_coordinates(image,coords,order=1,mode='constant',cval=0,prefilter=False)
        sparse=sparse_target(refs,case['audit']['reference_cut'],spacing,
                             [c[::config.stride,::config.stride,::config.stride] for c in coords],config)
        targets[number]=sparse
        records.append(dict(index=number,image=case['image'],reference_id=ref+1,name=name,
                            split=case['audit']['split'],group=case['audit']['group']))
    images.flush();targets.flush()
    index=dict(identity=identity,records=records,policy='Center loss only inside complete marked cells; '
        'unmarked cells and all regions outside manual target masks are ignored. Box loss only near '
        'marked centers. No shape filtering; no test crops; no synthetic or source background negatives.')
    _json(target,index);return index


def batch(images,targets,ids,rng,augment=False):
    xs=[];ys=[]
    for i in ids:
        x=np.array(images[i],np.float32);y=np.array(targets[i],np.float32)
        if augment:
            # Axis reflections also exchange the appropriate low/high box faces.
            for axis in range(3):
                if rng.random()<.5:
                    x=np.flip(x,axis);y=np.flip(y,axis).copy()
                    y[..., [2+axis,5+axis]]=y[..., [5+axis,2+axis]]
            x=np.array(x,copy=True)
            x=np.clip(x*rng.uniform(.8,1.2)+rng.uniform(-.03,.03)+rng.normal(0,.008,x.shape),0,1)
        xs.append(x);ys.append(y)
    return np.ascontiguousarray(xs,np.float32),np.ascontiguousarray(ys,np.float32)
