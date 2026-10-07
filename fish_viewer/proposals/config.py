from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ProposalConfig:
    patch_size: int = 64
    voxel_um: float = .3
    stride: int = 2
    filters: int = 4
    epochs: int = 35
    batch_size: int = 4
    patience: int = 7
    learning_rate: float = .0005
    center_sigma_um: float = .65
    box_scale_um: float = 10.
    threshold: float = .5
    separation_um: float = 2.
    box_margin_um: float = 2.
    seed: int = 43

    def validate(self):
        if any(not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('Proposal settings must be finite.')
        for name,lo,hi in [('patch_size',32,96),('stride',2,2),('filters',4,16),
                           ('epochs',1,200),('batch_size',1,8),('patience',1,30),('seed',0,2**31-1)]:
            v=getattr(self,name)
            if type(v) is not int or not lo<=v<=hi:raise ValueError('Invalid '+name)
        if self.patch_size%8 or not .2<=self.voxel_um<=.5 or not .2<=self.center_sigma_um<=2:
            raise ValueError('Invalid detector sampling.')
        if not .01<=self.threshold<=.99 or not .5<=self.separation_um<=5 or not .5<=self.box_margin_um<=5:
            raise ValueError('Invalid proposal acceptance.')
        if not 1e-6<=self.learning_rate<=.01 or not 1<=self.box_scale_um<=30:
            raise ValueError('Invalid detector training settings.')
        return self


def read_model(directory):
    directory=Path(directory).expanduser().resolve()
    manifest=json.loads((directory/'proposal-model.json').read_text(encoding='utf-8'))
    if manifest.get('version')!=1 or manifest.get('engine')!='cell-center-box3d':
        raise ValueError('Choose a trained FISH center-and-box model folder.')
    ProposalConfig(**manifest['config']).validate()
    if manifest.get('checkpoint')!='best.weights.h5' or not (directory/manifest['checkpoint']).is_file():
        raise ValueError('Missing proposal checkpoint.')
    return manifest
