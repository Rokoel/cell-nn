"""Sparse XY training sections and one-to-one adjacent-section instance linking."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from .audit import read_references


class SlicePatches:
    """Up to seven equally spaced usable sections per complete annotated cell.

    Outside the one target cell is unknown (-1), never inferred background.
    Image and mask coordinates are enlarged identically, preserving full targets.
    """
    def __init__(self, cases, size=128, sections=7, synthetic_fraction=.25):
        self.size=size; self.items=[]; self.cells=0
        for case in cases:
            refs,_=read_references(Path(case['directory'])/'references.npz')
            for number in case['audit']['training_reference_ids']:
                box,mask=refs[number-1]
                zs=np.flatnonzero(mask.sum(axis=(1,2))>=16)
                if not len(zs): continue
                self.cells+=1
                chosen=zs[np.unique(np.round(np.linspace(0,len(zs)-1,min(sections,len(zs)))).astype(int))]
                for z in chosen:
                    plane=mask[z]; points=np.argwhere(plane)
                    if np.any(np.ptp(points,axis=0)+1>size-2):
                        raise ValueError(f'Target too large for XY patch: {case["image"]}')
                    center=(points.min(0)+points.max(0)+1)//2+np.array([s.start for s in box[1:]])
                    shape=np.array(case['audit']['shape_zyx'][1:])
                    start=np.clip(center-size//2,0,np.maximum(0,shape-size))
                    self.items.append((case,box,plane,int(z+box[0].start),start))
        self.real=len(self.items); self.synthetic=int(np.ceil(self.real*synthetic_fraction))

    def __len__(self): return self.real+self.synthetic

    @lru_cache(maxsize=16)
    def image(self, directory):
        return np.load(Path(directory)/'image.npy',mmap_mode='r')

    @lru_cache(maxsize=16)
    def pair(self, index):
        if index>=self.real:
            rng=np.random.default_rng(42+index)
            return (np.abs(rng.normal(0,.002,(self.size*2,self.size*2))).astype('float32'),
                    np.zeros((self.size*2,self.size*2),np.int16))
        case,box,mask,z,start=self.items[index]
        stop=np.minimum(start+self.size,case['audit']['shape_zyx'][1:])
        crop=tuple(slice(int(a),int(b)) for a,b in zip(start,stop))
        x=np.asarray(self.image(case['directory'])[z][crop])
        y=np.full(x.shape,-1,np.int16)
        lower=np.maximum(start,[s.start for s in box[1:]])
        upper=np.minimum(stop,[s.stop for s in box[1:]])
        target=tuple(slice(int(a-b),int(c-b)) for a,b,c in zip(lower,start,upper))
        source=tuple(slice(int(a-s.start),int(b-s.start)) for a,b,s in zip(lower,upper,box[1:]))
        y[target][mask[source]]=1
        if np.count_nonzero(y>0)!=np.count_nonzero(mask): raise ValueError('Truncated target')
        pad=[(0,self.size-n) for n in x.shape]
        x=np.pad(x,pad,mode='reflect'); y=np.pad(y,pad,constant_values=-1)
        x=ndi.zoom(x,2,order=1,prefilter=False).astype('float32')
        y=np.repeat(np.repeat(y,2,axis=0),2,axis=1)
        return x,y


class SliceView:
    def __init__(self, patches, which): self.patches=patches; self.which=which
    def __len__(self): return len(self.patches)
    def __getitem__(self,index): return self.patches.pair(int(index))[self.which]


def link_sections(sections, minimum_iou=.2):
    """Hungarian one-to-one links; never merge touching independent instances.

    Only adjacent slices may link. A missing section starts a new track. This
    intentionally exposes fragmented 2D predictions rather than filling gaps.
    """
    sections=np.asarray(sections)
    if sections.ndim!=3 or not np.issubdtype(sections.dtype,np.integer) or np.any(sections<0):
        raise ValueError('Expected nonnegative integer ZYX instances')
    if not 0<minimum_iou<=1: raise ValueError('Invalid linking IoU')
    out=np.zeros(sections.shape,np.uint32); next_id=1
    for z,plane in enumerate(sections):
        ids=np.unique(plane); ids=ids[ids>0]; matches={}
        if z and len(ids):
            previous=out[z-1]; old=np.unique(previous);old=old[old>0]
            if len(old):
                old_sizes=np.array([np.count_nonzero(previous==i) for i in old])
                sizes=np.array([np.count_nonzero(plane==i) for i in ids])
                intersections=np.zeros((len(old),len(ids)))
                for j,i in enumerate(ids):
                    overlaps=np.bincount(previous[plane==i],minlength=int(previous.max())+1)
                    intersections[:,j]=overlaps[old]
                iou=intersections/(old_sizes[:,None]+sizes[None,:]-intersections)
                rows,cols=linear_sum_assignment(-iou)
                matches={int(ids[c]):int(old[r]) for r,c in zip(rows,cols) if iou[r,c]>=minimum_iou}
        for i in ids:
            number=matches.get(int(i))
            if number is None:number=next_id;next_id+=1
            out[z][plane==i]=number
    return out


def predict_sections(model, image, probability, nms=.3, batch_size=4, cancel=None, progress=None):
    """Predict XY polygons at 2x scale, batching slices without changing weights.

    Match StarDist's end-only reflect padding, output-grid crop and radial clamp.
    Very large sections use the official tiled API to bound memory.
    """
    from ..pipeline.segmentation import checkpoint
    checkpoint(cancel)
    image=np.asarray(image)
    if image.ndim!=3 or not np.isfinite(image).all():raise ValueError('Expected finite ZYX intensity')
    out=np.zeros(image.shape,np.uint32)
    enlarged_shape=tuple(2*n for n in image.shape[1:])
    if max(enlarged_shape)>768:
        for z,plane in enumerate(image):
            checkpoint(cancel)
            large=ndi.zoom(plane,2,order=1,prefilter=False)
            labels,_=model.predict_instances(large,axes='YX',prob_thresh=probability,nms_thresh=nms,
                sparse=False,show_tile_progress=False,verbose=False,
                n_tiles=tuple(max(1,int(np.ceil(n/512))) for n in large.shape))
            out[z]=labels[::2,::2]
            if progress:progress(z+1,len(image))
        return out
    divisors=model._axes_div_by('YX')
    pad=[(0,(-n)%d) for n,d in zip(enlarged_shape,divisors)]
    grid=model.config.grid
    result_shape=tuple(int(np.ceil(n/g)) for n,g in zip(enlarged_shape,grid))
    for start in range(0,len(image),batch_size):
        checkpoint(cancel)
        x=np.stack([np.pad(ndi.zoom(p,2,order=1,prefilter=False),pad,mode='reflect') for p in image[start:start+batch_size]])[...,None]
        probs,dists=model.keras_model.predict_on_batch(x)
        for offset in range(len(x)):
            crop=tuple(slice(0,n) for n in result_shape)
            prob=probs[offset][crop][...,0]
            dist=np.maximum(1e-3,dists[offset][crop])
            labels,_=model._instances_from_prediction(enlarged_shape,prob,dist,prob_thresh=probability,nms_thresh=nms,verbose=False)
            out[start+offset]=labels[::2,::2]
        if progress:progress(min(start+batch_size,len(image)),len(image))
    return out
