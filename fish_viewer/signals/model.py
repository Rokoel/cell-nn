"""Validate classifier and direct-center model folders without loading a runtime."""
import json
from pathlib import Path

from .config import SignalConfig
from ..spots.runtime import DIRECT_ENGINES


def read_model(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'signal-model.json').read_text())
    if manifest.get('engine') in DIRECT_ENGINES:
        from ..spots.config import read_model as read_centers
        return read_centers(directory)
    if manifest.get('version') != 1 or manifest.get('engine') != 'signal-cnn3d':
        raise ValueError('Choose a compatible trained FISH signal model folder.')
    SignalConfig(**manifest['config']).validate()
    if not (directory / manifest['checkpoint']).is_file():
        raise ValueError('The signal model checkpoint is missing.')
    return manifest
