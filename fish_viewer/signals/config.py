from dataclasses import dataclass


@dataclass(frozen=True)
class SignalConfig:
    patch_size: int = 24
    patch_spacing_um: float = .15
    candidate_sigma_um: float = .18
    candidate_background_um: float = .50
    candidate_separation_um: float = .35
    detection_separation_um: float | None = None
    candidate_contrast: float = .025
    candidate_scales_um: tuple[float, ...] = ()
    candidate_background_ratio: float = 2.8
    channel_contrast_min: float | None = None
    one_to_one_positives: bool = False
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
    hard_negative_fraction: float = 0.
    hard_negative_interval: int = 5
    checkpoint_policy: str = 'balanced_loss'
    target_precision: float = .90
    seed: int = 42
    device: str = 'auto'

    def validate(self):
        import math
        from dataclasses import asdict
        for key,value in asdict(self).items():
            if isinstance(value,(int,float)) and not math.isfinite(value):raise ValueError(f'{key} must be finite')
        if any(not math.isfinite(v) or v <= 0 for v in self.candidate_scales_um):
            raise ValueError('Candidate scales must be positive finite physical sizes.')
        if min(self.patch_spacing_um,self.candidate_sigma_um,self.candidate_background_um,
               self.candidate_separation_um,self.candidate_contrast) <= 0:
            raise ValueError('Candidate sizes, separation and contrast must be positive.')
        if self.candidate_background_ratio <= 1:
            raise ValueError('Candidate background ratio must exceed one.')
        if self.channel_contrast_min is not None and self.channel_contrast_min <= 0:
            raise ValueError('Channel evidence contrast must be positive.')
        if self.detection_separation_um is not None and self.detection_separation_um <= 0:
            raise ValueError('Detection separation must be positive.')
        if not 0 <= self.hard_negative_fraction <= 1 or self.hard_negative_interval < 1:
            raise ValueError('Invalid hard-negative sampling settings.')
        if self.checkpoint_policy not in ('balanced_loss','validation_precision'):
            raise ValueError('Invalid checkpoint policy.')
        if self.patch_size < 8 or self.patch_size % 8:raise ValueError('Patch size must be a positive multiple of 8.')
        if not 0 < self.match_radius_um < self.negative_radius_um:raise ValueError('Invalid matching radii.')
        if not 0 < self.target_precision <= 1:raise ValueError('Invalid target precision.')
        if self.device not in ('auto','cpu','gpu'):raise ValueError('Invalid device.')
        if min(self.epochs,self.steps_per_epoch,self.batch_size,self.patience)<1:raise ValueError('Invalid training budget.')
        if self.batch_size<2:raise ValueError('Balanced signal training needs batch size at least two.')
        return self
