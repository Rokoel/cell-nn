"""Capabilities of a packaged runtime, without importing either neural library."""
import json
import sys
from pathlib import Path


def packaged_profile():
    if not getattr(sys, 'frozen', False):
        return None
    path = Path(sys._MEIPASS) / 'runtime-profile.json'
    # Older standalone releases bundled CPU runtimes.
    return json.loads(path.read_text()) if path.exists() else {'cuda': False}


def configure_device_selector(selector, saved):
    profile = packaged_profile()
    if profile is not None and not profile['cuda']:
        selector.addItem('CPU (bundled runtime)', 'cpu')
        selector.setToolTip('This executable contains CPU runtimes. GPU inference is available in the CUDA-enabled Linux AppImage.')
    else:
        for name in ('auto', 'cpu', 'gpu'):
            selector.addItem(name, name)
        selector.setCurrentIndex(max(0, selector.findData(saved)))
        if profile is not None:
            selector.setToolTip('CUDA libraries are installed automatically on first launch. Auto uses an available NVIDIA GPU and otherwise uses CPU. GPU requires a compatible system NVIDIA driver.')
