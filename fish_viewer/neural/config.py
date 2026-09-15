from dataclasses import asdict, dataclass
import math

from ..pipeline.config import PipelineConfig


@dataclass(frozen=True)
class NeuralConfig:
    voxel_um: float = 0.3
    min_diameter_um: float = 2.5
    max_diameter_um: float = 8.0
    min_sphericity: float = 0.62
    max_elongation: float = 1.8
    exclude_border: bool = False
    epochs: int = 150
    steps_per_epoch: int = 40
    batch_size: int = 1
    patch_z: int = 32
    patch_xy: int = 64
    rays: int = 64
    filters: int = 16
    learning_rate: float = 0.0003
    patience: int = 10
    validate_every: int = 5
    probability_threshold: float = 0.5
    nms_threshold: float = 0.4
    seed: int = 42
    device: str = 'auto'
    calibrate_probability: bool = True

    def geometry(self):
        keys = ('voxel_um', 'min_diameter_um', 'max_diameter_um', 'min_sphericity', 'max_elongation', 'exclude_border')
        return PipelineConfig(**{k:getattr(self, k) for k in keys}).validate()

    def validate(self):
        self.geometry()
        for key, value in asdict(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f'{key} must be finite.')
        for key, low, high in [('epochs',1,10000), ('steps_per_epoch',1,10000), ('batch_size',1,16),
                              ('patch_z',16,128), ('patch_xy',16,256), ('rays',16,256), ('filters',8,64),
                              ('patience',1,1000), ('validate_every',1,1000), ('seed',0,2**31-1)]:
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{key} must be an integer in {low}–{high}.')
        if self.patch_z % 16 or self.patch_xy % 16:
            raise ValueError('Patch dimensions must be multiples of 16.')
        if not 1e-7 <= self.learning_rate <= 0.01:
            raise ValueError('Learning rate must be between 1e-7 and 0.01.')
        if not 0.05 <= self.probability_threshold <= 0.95 or not 0.05 <= self.nms_threshold <= 0.9:
            raise ValueError('Invalid detection or overlap threshold.')
        if self.device not in ('auto', 'cpu', 'gpu'):
            raise ValueError('Device must be auto, cpu, or gpu.')
        return self

    @classmethod
    def from_dict(cls, values):
        return cls(**values).validate()
