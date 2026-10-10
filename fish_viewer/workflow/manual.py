"""Original manual volumes and A-B distances for the combined Excel measurement export."""
from pathlib import Path

import numpy as np

from ..models import LibraryEntry
from ..reader import entry_for_path, load_dataset
from ..pipeline.evaluation import rasterize_reference
from ..pipeline.segmentation import Prepared, checkpoint
from ..signals.config import SignalConfig
from ..signals.data import normalized, infer_channels, reference_memberships
from .volumes import surface_volume


def manual_measurements(report, cancel=None):
    if 'manual_measurements' in report:
        return report['manual_measurements']
    source = Path(report.get('image', ''))
    if not source.is_file():
        return dict(cells=[], points=[])
    entry = entry_for_path(source)
    scene_path = report.get('scene')
    if scene_path and Path(scene_path).is_file():
        entry = LibraryEntry(source, Path(scene_path))
    dataset = load_dataset(entry, time=report.get('time', 0))
    if report.get('fingerprint'):
        from ..signals.engine import image_fingerprint
        if image_fingerprint(dataset) != report['fingerprint']:
            raise ValueError('Original scan pixels differ from the analysis.')
    time = report.get('time', 0)
    surfaces = [s for s in dataset.scene.surfaces if s.time == time and s.source == 'manual']
    groups = [g for g in dataset.scene.measurements if g.source == 'manual']
    if not surfaces and not groups:
        return dict(cells=[], points=[])
    prepared = Prepared(dataset.volumes[0], dataset.spacing[::-1], dataset.center_origin,
                        report.get('membrane_channel', 0), [], '', {}, {})
    cells, references, names = [], [], []
    for surface in surfaces:
        checkpoint(cancel)
        cells.append(dict(cell_id=surface.name, volume_um3=surface_volume(surface)))
        refs, _, _ = rasterize_reference([surface], prepared, time_index=time)
        if refs:
            references.append(refs[0]); names.append(surface.name)
    points = []
    for group in groups:
        linked = set()
        for left, right in group.edges:
            if group.times[left] != time or group.times[right] != time:
                continue
            if {group.labels[left].strip().upper(), group.labels[right].strip().upper()} != {'A', 'B'}:
                continue
            a, b = (left, right) if group.labels[left].strip().upper() == 'A' else (right, left)
            linked.add(a)
            points.append(dict(signal_id=group.name, xyz=group.points[a].tolist(),
                               distance_um=float(np.linalg.norm(group.points[a] - group.points[b]))))
        for index, (point, label, point_time) in enumerate(zip(group.points, group.labels, group.times)):
            if label.strip().upper() == 'A' and point_time == time and index not in linked:
                points.append(dict(signal_id=group.name, xyz=point.tolist(), distance_um=None))
    if points:
        spacing, origin = dataset.spacing[::-1], dataset.center_origin
        memberships = reference_memberships([p['xyz'] for p in points], references, origin, spacing)
        points = infer_channels([normalized(v) for v in dataset.volumes], points,
                                report.get('membrane_channel', 0), origin, spacing, SignalConfig())
        locators = {}
        for point, membership in zip(points, memberships):
            checkpoint(cancel)
            point['cell_id'] = ' | '.join(names[i] for i in membership) or None
            if point['distance_um'] is None and len(membership) == 1:
                from .distances import SurfaceLocator
                name = names[membership[0]]
                if name not in locators:
                    locators[name] = SurfaceLocator(next(s for s in surfaces if s.name == name))
                point['distance_um'] = float(locators[name].closest(point['xyz'])[1])
            channel = point['channel']
            point['channel_name'] = dataset.metadata.channels[channel].name if channel is not None else None
    return dict(cells=cells, points=points, source_scene=str(dataset.scene.source) if dataset.scene.source else None)
