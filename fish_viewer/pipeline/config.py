from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class PipelineConfig:
    voxel_um: float = 0.30
    min_diameter_um: float = 2.5
    max_diameter_um: float = 8.0
    smooth_um: float = 0.18
    threshold: float = 0.06
    closing_um: float = 0.0
    seed_distance_um: float = 1.2
    min_sphericity: float = 0.62
    max_elongation: float = 1.8
    exclude_border: bool = True
    method: str = 'boundary_watershed'

    def validate(self):
        values = asdict(self)
        for key, value in values.items():
            if isinstance(value, (float, int)) and not isinstance(value, bool) and not math.isfinite(value):
                raise ValueError(f'{key} must be finite.')
        if not 0.05 <= self.voxel_um <= 5:
            raise ValueError('Processing voxel size must be 0.05–5 µm.')
        if not 0 < self.min_diameter_um < self.max_diameter_um <= 500:
            raise ValueError('Cell diameter limits must satisfy 0 < minimum < maximum ≤ 500 µm.')
        if not 0 <= self.smooth_um <= 5 or not 0 <= self.closing_um <= 5:
            raise ValueError('Smoothing and gap closing must be between 0 and 5 µm.')
        if not 0.01 <= self.threshold <= 0.9 or not 0 <= self.min_sphericity <= 1:
            raise ValueError('Invalid boundary threshold or sphericity.')
        if not 1 <= self.max_elongation <= 5 or not 0.2 <= self.seed_distance_um <= 100:
            raise ValueError('Invalid elongation or seed separation.')
        if self.method not in ('boundary_watershed', 'distance_watershed'):
            raise ValueError('Unknown segmentation method.')
        return self

    @classmethod
    def from_dict(cls, values):
        return cls(**values).validate()
