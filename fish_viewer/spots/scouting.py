"""Experimental full-scan signal proposals, before cell masks exist."""
import hashlib
import itertools
import json
from pathlib import Path
import time
import traceback

import numpy as np

from ..models import LibraryEntry
from ..reader import load_dataset
from ..pipeline.config import PipelineConfig
from ..pipeline.segmentation import prepare, checkpoint, Cancelled
from ..pipeline.runner import _json
from ..run_paths import new_run_directory
from ..signals.data import normalized
from ..signals.evaluation import suppress
from .data import resample_crop
from .detection import local_centers
from .inference import load
from .training import predict_volume


def tile_starts(shape, size=96, stride=64):
    return itertools.product(*(sorted(set([*range(0, max(1, int(n)-size+1), stride),
                                          max(0, int(n)-size)])) for n in shape))


def scout(image, model_directory, output, cell_model, device='auto', emit=lambda e: None, cancel=None, cell_channel=None, signal_channels=None):
    directory = new_run_directory(output, 'signal-scout')
    report = dict(version=1, status='running', directory=str(directory), image=str(Path(image).resolve()),
                  points=[], candidate_only=True, policy='Full-scan heatmaps with an all-valid mask; unvalidated outside training cell-mask context. Only gate boxes; final signals are rechecked inside accepted cells.')
    started = time.monotonic()
    emit(dict(kind='run_started', directory=str(directory)))
    try:
        model, manifest, config, runtime_device = load(model_directory, device)
        dataset = load_dataset(LibraryEntry(Path(image)))
        channel_config = json.loads((Path(cell_model)/'proposal-model.json').read_text()).get('channel_selection')
        from dataclasses import replace
        geometry=PipelineConfig(**(channel_config or dict(min_diameter_um=2.5, max_diameter_um=32.)))
        if cell_channel is not None:geometry=replace(geometry,cell_channel=cell_channel)
        selected = prepare(dataset, geometry,
                           lambda e: emit({k:v for k,v in e.items() if k != 'previews'}), cancel,
                           background_correct=False, max_voxels=None)
        report.update(fingerprint=selected.fingerprint, membrane_channel=selected.channel,
                      model_directory=str(Path(model_directory).resolve()), threshold=manifest['threshold'],
                      checkpoint_sha256=hashlib.sha256((Path(model_directory)/manifest['checkpoint']).read_bytes()).hexdigest())
        spacing = np.asarray(dataset.spacing[::-1]); origin = np.asarray(dataset.center_origin)
        images = [normalized(v) for v in dataset.volumes]
        shape = np.maximum(2, np.ceil((np.asarray(images[0].shape)-1)*spacing/config.spacing_um).astype(int)+1)
        from ..channel_roles import signal_indices
        starts = list(tile_starts(shape)); channels = signal_indices(len(images),selected.channel,signal_channels)
        report['signal_channels']=channels
        proposed = []; current = 0; size = 96
        for start in starts:
            checkpoint(cancel)
            crop_origin = origin + (np.asarray(start)*config.spacing_um)[::-1]
            cell_image = resample_crop(images[selected.channel], spacing, origin, crop_origin, (size,)*3, config.spacing_um)
            valid = np.all(np.indices((size,)*3)+np.asarray(start)[:,None,None,None] < shape[:,None,None,None], axis=0)
            for channel in channels:
                checkpoint(cancel)
                emit(dict(kind='progress', phase='Searching whole scan for signal centers', current=current, total=len(starts)*len(channels)))
                signal = resample_crop(images[channel], spacing, origin, crop_origin, (size,)*3, config.spacing_um)
                heatmap, flow = predict_volume(model, np.stack([signal, cell_image, valid]).astype(np.float32), runtime_device)
                # Overlapping tiles supply context; discard internal-edge peaks.
                keep = valid.copy()
                for axis, offset in enumerate(start):
                    for edge, discard in ((slice(0, 8), offset > 0),
                                          (slice(-8, None), offset+size < shape[axis])):
                        if discard:
                            box=[slice(None)]*3; box[axis]=edge; keep[tuple(box)]=False
                proposed.extend(local_centers(heatmap, keep, crop_origin, config, 0, channel,
                                              floor=manifest['threshold'], flow=flow))
                current += 1
        report['points'] = suppress(proposed, [p['score'] for p in proposed], manifest['threshold'], config.detection_separation_um)
        report.update(status='complete', channels=[dict(name=c.name, color=list(c.color)) for c in dataset.metadata.channels],
                      candidate_count=len(report['points']))
        emit(dict(kind='progress', phase='Signal candidates ready', current=current, total=current))
    except Cancelled:
        report['status']='cancelled'
    except Exception:
        report.update(status='failed', error=traceback.format_exc())
    report['elapsed_s']=time.monotonic()-started
    _json(directory/'report.json', report); emit(dict(kind='finished', report=report))
    return report
