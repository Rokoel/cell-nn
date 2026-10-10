"""Training-only sampling and portable, atomic annotation records."""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import random
import shutil
import uuid

import h5py
import numpy as np

from fish_viewer.reader import ImarisReader

CATEGORIES = {0: 'unlabeled', 1: 'likely not cell border', 3: 'likely cell border'}
EXCLUDED = {'18327antiLAD2.ims', '18327antiLAD3.ims', '18327antiLAD5.ims', '3AantiLADFITCTyrCy5_546.ims'}


@dataclass
class Crop:
    image: np.ndarray  # YX, original cell-channel intensity values
    labels: np.ndarray  # YX, zero is UNKNOWN, never background
    metadata: dict


class TrainingSampler:
    def __init__(self, audit, crop_shape=(128, 128), seed=None, fish_root=None):
        self.audit_path = Path(audit).resolve()
        document = json.loads(self.audit_path.read_text())
        cases = document['cases']
        heldout = set()
        heldout_groups = set()
        heldout_fingerprints = set()
        for case in cases:
            if case['audit']['split'] != 'training':
                heldout.update(str(Path(p).resolve()) for p in [case['image'], *case['audit'].get('aliases', [])])
                heldout_groups.add(case['audit'].get('group'))
                if case['audit'].get('fingerprint'):
                    heldout_fingerprints.add(case['audit']['fingerprint'])
        self.cases = []
        seen = set()
        for case in cases:
            case = dict(case)
            info = case['audit']
            paths = [case['image'], *info.get('aliases', [])]
            if info['split'] != 'training' or any(Path(p).name in EXCLUDED for p in paths):
                continue
            if (any(str(Path(p).resolve()) in heldout for p in paths)
                    or (info.get('group') is not None and info['group'] in heldout_groups)
                    or info.get('fingerprint') in heldout_fingerprints):
                raise ValueError('Audit has training/held-out leakage; no crops will be sampled.')
            key = info.get('fingerprint', str(Path(case['image']).resolve()))
            if fish_root is not None:
                # Audits retain their original absolute paths; relocate the FISH tree
                # without changing split identities or acquisition groups.
                parts = Path(case['image'].replace('\\', '/')).parts
                if 'FISH' not in parts:
                    raise ValueError('Cannot relocate source: audit image path has no FISH directory.')
                case['original_image'] = case['image']
                case['image'] = str(Path(fish_root).joinpath(*parts[parts.index('FISH') + 1:]))
            if key not in seen and Path(case['image']).is_file():
                seen.add(key)
                self.cases.append(case)
        if not self.cases:
            raise ValueError('No available training scans in the audit. Use --audit with a training-split audit whose image paths exist.')
        self.crop_shape = tuple(int(v) for v in crop_shape)
        if len(self.crop_shape) != 2 or min(self.crop_shape) < 1:
            raise ValueError('Crop dimensions must be positive Y, X sizes.')
        self.seed = seed if seed is not None else random.SystemRandom().randrange(2**63)
        self.rng = random.Random(self.seed)
        self.previous = None
        self.sequence = 0

    def next_crop(self):
        choices = [i for i in range(len(self.cases)) if i != self.previous] or [0]
        index = self.rng.choice(choices)
        case = self.cases[index]
        reader = ImarisReader(case['image'])
        meta = reader.metadata
        size = (1, *(min(n, c) for n, c in zip(meta.shape[1:], self.crop_shape)))
        offset = tuple(self.rng.randrange(n - c + 1) for n, c in zip(meta.shape, size))
        region = tuple(slice(o, o + c) for o, c in zip(offset, size))
        time = self.rng.choice(meta.times)
        channel = case['audit'].get('channel')
        if not isinstance(channel, int) or not 0 <= channel < len(meta.channels):
            raise ValueError('Audit must identify the cell channel; signal channels will not be guessed.')
        if reader.hdf:
            with h5py.File(reader.path, 'r') as f:
                data = f[f'{reader.dataset_name}/ResolutionLevel 0/TimePoint {time}/Channel {reader.channel_ids[channel]}/Data']
                image = data[region][0]
        else:
            image = reader.read(time=time)[channel][region][0]
        self.previous = index
        self.sequence += 1
        stat = reader.path.stat()
        metadata = {
            'schema_version': 2, 'categories': CATEGORIES,
            'label_policy': 'Only painted pixels are supervised; zero is unknown, not background. Categories are likely judgments, not confirmed truth.',
            'source': str(reader.path), 'source_size_bytes': stat.st_size, 'source_mtime_ns': stat.st_mtime_ns,
            'audited_source_path': case.get('original_image', case['image']),
            'audit': str(self.audit_path), 'split': 'training', 'acquisition_group': case['audit'].get('group'),
            'audited_source_fingerprint': case['audit'].get('fingerprint'),
            'sampler_seed': self.seed, 'sample_sequence': self.sequence,
            'time_point': time, 'resolution_level': 0, 'source_shape_zyx': list(meta.shape),
            'offset_zyx': list(offset), 'shape_zyx': list(size), 'spacing_xyz': list(meta.spacing),
            'origin_xyz': [o + s * i for o, s, i in zip(meta.origin, meta.spacing, offset[::-1])],
            'unit': meta.unit, 'image_axes': 'YX', 'label_axes': 'YX', 'sampling': 'single native slice, no projection',
            'cell_channel_index': channel,
            'cell_channel': {'name': meta.channels[channel].name, 'color': list(meta.channels[channel].color)},
        }
        return Crop(image, np.zeros(image.shape, np.uint8), metadata)


def save_crop(crop, output):
    """Commit image + labels + provenance together; never save empty supervision."""
    if (crop.metadata.get('schema_version') != 2 or crop.image.ndim != 2
            or crop.labels.shape != crop.image.shape or not np.isin(crop.labels, list(CATEGORIES)).all()):
        raise ValueError('Invalid annotation labels.')
    if not np.any(crop.labels):
        raise ValueError('No marks to save. Unpainted crops are unknown; use Skip.')
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = uuid.uuid4().hex
    staging = root / ('.pending-' + identity)
    target = root / ('crop-' + identity)
    staging.mkdir()
    try:
        metadata = dict(crop.metadata)
        metadata.update(saved_at_utc=datetime.now(timezone.utc).isoformat(),
                        annotation_id=identity,
                        category_pixel_counts={str(k): int(np.count_nonzero(crop.labels == k)) for k in CATEGORIES})
        np.savez_compressed(staging / 'annotation.npz', image=crop.image, labels=crop.labels)
        (staging / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def paint(labels, start, end, diameter, category):
    """Interpolated circular brush in native image pixels, independent of zoom."""
    if category is None:
        return
    radius = max(1., diameter) / 2
    length = np.hypot(end[0] - start[0], end[1] - start[1])
    for fraction in np.linspace(0, 1, max(2, int(np.ceil(length / max(.5, radius / 2))) + 1)):
        x, y = np.array(start) + fraction * (np.array(end) - start)
        x0, x1 = max(0, int(np.floor(x - radius))), min(labels.shape[1], int(np.ceil(x + radius)))
        y0, y1 = max(0, int(np.floor(y - radius))), min(labels.shape[0], int(np.ceil(y + radius)))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask = (xx + .5 - x)**2 + (yy + .5 - y)**2 <= radius**2
        labels[y0:y1, x0:x1][mask] = category
        if not mask.any() and 0 <= x < labels.shape[1] and 0 <= y < labels.shape[0]:
            labels[int(y), int(x)] = category


def fill(labels, point, category):
    """Fill the four-connected region of the clicked annotation label.

    Painted contours stop a fill; raw intensity does not determine membership.
    """
    if category is None:
        return
    x, y = int(np.floor(point[0])), int(np.floor(point[1]))
    if not (0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]):
        return
    current = labels[y, x]
    if current == category:
        return
    from scipy.ndimage import label
    components, _ = label(labels == current)
    labels[components == components[y, x]] = category
