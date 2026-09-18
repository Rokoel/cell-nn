from dataclasses import dataclass


@dataclass(frozen=True)
class SignalConfig:
    patch_size: int = 24
    patch_spacing_um: float = .15
    candidate_sigma_um: float = .18
    candidate_background_um: float = .50
    candidate_separation_um: float = .35
    candidate_contrast: float = .025
    match_radius_um: float = .6
    negative_radius_um: float = .9
    assignment_tolerance_um: float = .35
    channel_score_min: float = 1.5
    channel_margin: float = 1.25
    epochs: int = 100
    steps_per_epoch: int = 80
    batch_size: int = 32
    patience: int = 15
    learning_rate: float = .0003
    target_precision: float = .90
    seed: int = 42
    device: str = 'auto'

    def validate(self):
        import math
        from dataclasses import asdict
        for key,value in asdict(self).items():
            if isinstance(value,(int,float)) and not math.isfinite(value):raise ValueError(f'{key} must be finite')
        if self.patch_size < 8 or self.patch_size % 8:raise ValueError('Patch size must be a positive multiple of 8.')
        if not 0 < self.match_radius_um < self.negative_radius_um:raise ValueError('Invalid matching radii.')
        if not 0 < self.target_precision <= 1:raise ValueError('Invalid target precision.')
        if self.device not in ('auto','cpu','gpu'):raise ValueError('Invalid device.')
        if min(self.epochs,self.steps_per_epoch,self.batch_size,self.patience)<1:raise ValueError('Invalid training budget.')
        return self
