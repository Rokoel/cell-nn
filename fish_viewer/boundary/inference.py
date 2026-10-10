"""Calibrated slice-wise boundary probabilities for an existing 3D field."""
from pathlib import Path
import hashlib
import numpy as np
from scipy import ndimage as ndi
from .model import read_model, network, normalize, tensorflow


class BoundaryPredictor:
    def __init__(self, directory, device='auto'):
        self.directory=Path(directory).resolve()
        self.manifest=read_model(self.directory)
        self.sha256=hashlib.sha256((self.directory/self.manifest['checkpoint']).read_bytes()).hexdigest()
        if self.sha256!=self.manifest['checkpoint_sha256']: raise ValueError('Boundary checkpoint checksum mismatch.')
        tf=tensorflow(device);self.model=network(tf)
        self.model.load_weights(self.directory/self.manifest['checkpoint'])

    def volume(self, image, spacing, emit=lambda e:None, cancel=None):
        from ..pipeline.segmentation import checkpoint
        spacing=np.asarray(spacing)
        if image.ndim!=3 or not np.isfinite(image).all() or spacing.shape!=(3,) or np.any(spacing<=0):
            raise ValueError('Finite 3D cell image and positive ZYX spacing required.')
        pixel=self.manifest['training_xy_um'];patch=self.manifest['patch_size']
        shape=np.maximum(1,np.rint(np.asarray(image.shape[1:])*spacing[1:]/pixel).astype(int))
        result=np.empty(image.shape,np.float32)
        window=np.maximum(.05,np.outer(np.hanning(patch),np.hanning(patch))).astype(np.float32)
        for z, raw in enumerate(image):
            checkpoint(cancel)
            emit(dict(kind='progress',phase='Predicting learned membrane boundaries',current=z,total=len(image)))
            plane=ndi.zoom(raw,shape/np.asarray(raw.shape),order=1,prefilter=False)
            padded=np.pad(plane,[(0,max(0,patch-n)) for n in plane.shape],mode='edge')
            ys=sorted(set([*range(0,max(1,padded.shape[0]-patch+1),96),padded.shape[0]-patch]))
            xs=sorted(set([*range(0,max(1,padded.shape[1]-patch+1),96),padded.shape[1]-patch]))
            positions=[(y,x) for y in ys for x in xs];total=np.zeros_like(padded,np.float32);weight=np.zeros_like(total)
            for start in range(0,len(positions),8):
                checkpoint(cancel)
                batch=positions[start:start+8]
                samples=np.stack([normalize(padded[y:y+patch,x:x+patch]) for y,x in batch])[...,None]
                logits=np.asarray(self.model(samples,training=False))[...,0]
                if not np.isfinite(logits).all():raise ValueError('Non-finite boundary predictions.')
                probability=1/(1+np.exp(-np.clip(logits,-60,60)))
                for (y,x),value in zip(batch,probability):
                    total[y:y+patch,x:x+patch]+=value*window;weight[y:y+patch,x:x+patch]+=window
            plane=(total/weight)[:shape[0],:shape[1]]
            result[z]=ndi.zoom(plane,np.asarray(raw.shape)/np.asarray(plane.shape),order=1,prefilter=False)
        emit(dict(kind='progress',phase='Learned membrane boundaries ready',current=len(image),total=len(image)))
        return result
