"""Fixed physical-scale membrane enhancement; never used as acceptance evidence."""
import numpy as np
from scipy import ndimage as ndi
from skimage.exposure import equalize_adapthist
from ..pipeline.segmentation import checkpoint

RECIPE = dict(version='membrane-connected-v1', denoise_um=.18, background_um=6.,
              background_fraction=.6, clahe_tile_um=4., clahe_clip=.01,
              clahe_fraction=.35, closing_um=.35, closing_fraction=.15)


def membrane_connected(image, spacing, cancel=None):
    image=np.asarray(image,dtype=np.float32)
    spacing=np.asarray(spacing,dtype=float)
    if image.ndim != 3 or spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing<=0) or not np.isfinite(image).all():
        raise ValueError('Preprocessing requires a finite 3D image and positive physical spacing.')
    if float(image.max()) <= float(image.min()):return np.zeros(image.shape,np.float32)
    checkpoint(cancel)
    smooth=ndi.gaussian_filter(image,sigma=RECIPE['denoise_um']/spacing,mode='reflect')
    background=ndi.gaussian_filter(smooth,sigma=RECIPE['background_um']/spacing,mode='reflect')
    corrected=np.maximum(smooth-RECIPE['background_fraction']*background,0)
    lo,hi=np.percentile(corrected,(1,99.5))
    corrected=np.clip((corrected-lo)/max(float(hi-lo),1e-6),0,1).astype(np.float32)
    checkpoint(cancel)
    kernel=tuple(np.maximum(2,np.rint(RECIPE['clahe_tile_um']/spacing).astype(int)))
    local=equalize_adapthist(corrected,kernel_size=kernel,clip_limit=RECIPE['clahe_clip']).astype(np.float32)
    enhanced=(1-RECIPE['clahe_fraction'])*corrected+RECIPE['clahe_fraction']*local
    checkpoint(cancel)
    radius=np.ceil(RECIPE['closing_um']/spacing).astype(int)
    grid=np.indices(tuple(2*radius+1))-radius[:,None,None,None]
    footprint=np.sum((grid*spacing[:,None,None,None])**2,axis=0)<=RECIPE['closing_um']**2
    closed=ndi.grey_closing(enhanced,footprint=footprint,mode='reflect')
    return np.clip(enhanced+RECIPE['closing_fraction']*(closed-enhanced),0,1).astype(np.float32)


def validate_manifest(manifest):
    if manifest.get('config',{}).get('preprocessing','none')!='none' and manifest.get('preprocessing_recipe')!=RECIPE:
        raise ValueError('Saved preprocessing recipe does not match this application version.')
