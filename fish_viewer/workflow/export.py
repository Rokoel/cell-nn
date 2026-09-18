"""Export an unchanged IMS image and an Imaris 6 XML scene companion.

The XML hierarchy and object fields follow the real Imaris 6.3 scenes in FISH.
We do not write undocumented binary Scene/Scene8 structures into the image.
"""
from collections import Counter
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from ..models import LibraryEntry
from ..pipeline.runner import read_prediction
from ..pipeline.segmentation import checkpoint
from ..reader import load_dataset
from ..signals.overlays import read_signals
from .results import validated_result, write_tables


def _text(parent, tag, value):
    element = ET.SubElement(parent, tag)
    element.text = str(value)
    return element


def _component(parent, name, identifier, color=(.8, .8, .8), opacity=1., group=False):
    component = ET.SubElement(parent, 'bpSurfaceComponent')
    for tag, value in dict(name=name, id=identifier, isHidden='false',
                           isNode=str(group).lower(), isExpanded=str(group).lower()).items():
        _text(component, tag, value)
    material = ET.SubElement(component, 'material')
    for tag, value in dict(ambientColor='((0.2,0.2,0.2))',
            diffuseColor='((' + ','.join(f'{c:.9g}' for c in color) + '))',
            specularColor='((0,0,0))', emissiveColor='((0,0,0))',
            shininess='(0.2)', transparency=f'({1-opacity:.9g})').items():
        _text(material, tag, value)
    return component


def _array(parent, tag, values, integer=False):
    # Imaris serializes numeric vectors as quoted, whitespace-separated blocks.
    flat = np.asarray(values).ravel()
    for start in range(0, len(flat), 12000):
        block = flat[start:start + 12000]
        tokens = [str(int(value)) if integer else f'{value:.12g}' for value in block]
        _text(parent, tag, '\n' + '\n'.join('"' + ' '.join(tokens[i:i+24]) + '"' for i in range(0, len(tokens), 24)) + '\n')


def scene_xml(image_name, surfaces, measurements, metadata, report):
    root = ET.Element('bpImarisApplication', Version='Imaris 6.3.1')
    root.append(ET.Comment('FISH Viewer exported detections; image pixels are preserved separately in the IMS file.'))
    ET.SubElement(root, 'bpCommandLoadDataSet', actorname='bpImarisApplication', actorid='1',
                  file=image_name, reader='All Formats', timereader='1',
                  croplimitsmin='', croplimitsmax='', resample='')
    application = ET.SubElement(root, 'bpSurfaceApplication')
    group = ET.SubElement(application, 'bpComponentGroup')
    scene = _component(group, 'Surpass Scene', 25, group=True)
    light = ET.SubElement(scene, 'bpLightSource')
    _component(light, 'Light Source 1', 26)
    _text(light, 'intensity', '1.000')
    ET.SubElement(light, 'color', red='1', green='1', blue='1')
    _text(light, 'direction', '0.408 -0.408 -0.816')
    _text(light, 'mFixToDataSet', 'false')
    volume = ET.SubElement(scene, 'bpVolumeRenderer')
    _component(volume, 'Volume', 27)
    _text(volume, 'renderingMode', 'MaxIntensityProjection')
    _text(volume, 'cutHighOrLow', '1')
    _text(volume, 'alphaValue', '0.000')
    counts = Counter(point['cell_id'] for point in report['points'])
    identifier = 28
    for surface in surfaces:
        cell_id = int(surface.name.rsplit(' ', 1)[-1])
        viewer = ET.SubElement(scene, 'bpSurfacesViewer')
        name = f'Predicted cell {cell_id} · {counts[cell_id]} signals'
        _component(viewer, name, identifier, surface.color, surface.opacity)
        identifier += 1
        ET.SubElement(viewer, 'bpSurfacesProperties', mColorCoding='eOff',
                      mTransparency=f'{1-surface.opacity:.9g}', mStyle='eTriangle', mCenterPointSize='5')
        objects = ET.SubElement(viewer, 'bpSurfaces')
        _text(objects, 'name', name)
        _text(objects, 'description', f'Neural cell {cell_id}; {counts[cell_id]} detected signal centers inside.')
        _text(objects, 'historyLog', 'Exported by FISH Viewer')
        _text(objects, 'unit', 'um' if metadata.unit in ('µm', 'um') else metadata.unit)
        mesh = ET.SubElement(objects, 'surface')
        vertices, faces = surface.vertices, surface.triangles.copy()
        centered = vertices - vertices.mean(axis=0)
        a, b, c = (centered[faces[:, i]] for i in range(3))
        if np.einsum('ij,ij->', a, np.cross(b, c)) < 0:
            faces = faces[:, [0, 2, 1]]
        normals = np.zeros_like(vertices)
        face_normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
        for i in range(3):
            np.add.at(normals, faces[:, i], face_normals)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
        _array(mesh, 'vertices', np.column_stack((vertices, normals)))
        _array(mesh, 'triangles', faces, integer=True)
        _text(mesh, 'timeIndex', surface.time)
    grouped = {}
    for point in report['points']:
        grouped.setdefault((point['cell_id'], point['channel']), []).append(point)
    for measurement, ((cell, channel), points) in zip(measurements, sorted(grouped.items())):
        obj = ET.SubElement(scene, 'bpMeasurementPoints')
        _component(obj, measurement.name + ' · ' + report['channels'][channel]['name'], identifier, measurement.color)
        identifier += 1
        for point in points:
            ET.SubElement(obj, 'Point', Name=f'Signal {point["signal_id"]}; cell {cell}; channel {channel+1}; score {point["score"]:.12g}',
                          Position=' '.join(f'{value:.12g}' for value in point['xyz']), TimeIndex=str(report.get('time', 0)))
        ET.SubElement(obj, 'PointShape', Shape='Sphere', Filled='true')
        _text(obj, 'PointDiameter', '0.15')
        ET.SubElement(obj, 'PointLabels', Name='true', Intensity='false')
        _text(obj, 'LineMode', 'None')
        _text(obj, 'VisibleForAllTimepoints', 'false')
        _text(obj, 'LineWidth', '1')
        ET.SubElement(obj, 'LineLabels', Distance='false', Velocity='false')
        _text(obj, 'LabelsColor', '1 1 1')
        ET.SubElement(obj, 'LabelsFont', Name='Arial', Size='12', Bold='false', Italic='false', Underline='false')
        ET.SubElement(obj, 'AutodepthMode', Intersect='SolidObjects')
    camera = ET.SubElement(application, 'SoPerspectiveCamera')
    size = np.asarray(metadata.shape[::-1]) * metadata.spacing
    center = np.asarray(metadata.origin) + size / 2
    distance = float(max(size) * 1.8)
    _text(camera, 'viewportMapping', '3')
    _text(camera, 'position', '(' + ', '.join(f'{v:.9g}' for v in center + [0, 0, distance]) + ')')
    _text(camera, 'orientation', '(0, 0, 1, 0)')
    _text(camera, 'aspectRatio', '1.000')
    _text(camera, 'focalDistance', f'{distance:.9g}')
    _text(camera, 'heightAngle', '0.785')
    ET.indent(root)
    return ET.tostring(root, encoding='utf-8', xml_declaration=True)


def export_result(report_path, target, emit=lambda event: None, cancel=None):
    """Publish a new IMS/IMX pair and tables; refuse to overwrite any existing file."""
    target = Path(target).expanduser().resolve()
    if target.suffix.lower() != '.ims':
        raise ValueError('Choose an export filename ending in .ims.')
    suffixes = ['.ims', '.imx', '.signals.csv', '.cell-counts.csv', '.annotations.json']
    destinations = [target.with_suffix(suffix) for suffix in suffixes]
    if any(path.exists() for path in destinations):
        raise FileExistsError('The export would overwrite existing files. Choose a new name or folder.')
    emit(dict(kind='progress', phase='Validating detections for export', current=0, total=1))
    report, _, cell_metadata, metadata = validated_result(report_path, cancel)
    if not report.get('signal_filter') or report['signal_filter'] != cell_metadata.get('signal_filter'):
        raise ValueError('Remove cells without signals before exporting this workflow result.')
    surfaces, _ = read_prediction(report['cell_predictions'])
    measurements, _ = read_signals(report_path)
    if len(surfaces) != len({point['cell_id'] for point in report['points']}):
        raise ValueError('The filtered contour set does not match the detected signals.')
    target.parent.mkdir(parents=True, exist_ok=True)
    created = []
    with tempfile.TemporaryDirectory(prefix='.fish-export-', dir=target.parent) as temporary:
        staging = Path(temporary)
        source = Path(report['image'])
        before = source.stat()
        total, copied = before.st_size, 0
        with source.open('rb') as src, (staging / target.name).open('wb') as dst:
            while block := src.read(8 * 1024 * 1024):
                checkpoint(cancel)
                dst.write(block)
                copied += len(block)
                emit(dict(kind='progress', phase='Copying Imaris image (all channels, times and resolutions)',
                          current=copied, total=total))
        if source.stat().st_mtime_ns != before.st_mtime_ns or copied != total:
            raise ValueError('The source scan changed during export. Please retry.')
        (staging / target.with_suffix('.imx').name).write_bytes(scene_xml(target.name, surfaces, measurements, metadata, report))
        write_tables(staging, report, target.stem + '.')
        annotation_report = dict(report, engine='fish-scan-export', mode='export', image=target.name, scene=target.with_suffix('.imx').name,
                                 directory='.', cell_predictions=None, export_format='Imaris IMS + XML IMX',
                                 source_signal_report=str(Path(report_path).resolve()))
        (staging / target.with_suffix('.annotations.json').name).write_text(json.dumps(annotation_report, indent=2, allow_nan=False), encoding='utf-8')
        emit(dict(kind='progress', phase='Verifying exported scene', current=0, total=1))
        reopened = load_dataset(LibraryEntry(staging / target.name, staging / target.with_suffix('.imx').name))
        if (len(reopened.scene.surfaces) != len(surfaces) or reopened.scene.warnings
                or sum(len(group.points) for group in reopened.scene.measurements) != len(report['points'])):
            raise ValueError('Exported annotations failed the reload check.')
        checkpoint(cancel)
        try:
            # Exclusive reservations avoid clobbering either user files or concurrent exports.
            for destination in destinations:
                with destination.open('xb'):
                    pass
                created.append(destination)
            for destination in destinations:
                (staging / destination.name).replace(destination)
        except BaseException:
            for destination in created:
                destination.unlink(missing_ok=True)
            raise
    emit(dict(kind='progress', phase='Export complete', current=1, total=1))
    return dict(status='complete', image=str(target), scene=str(target.with_suffix('.imx')),
                cells=len(surfaces), signals=len(report['points']), files=[str(path) for path in destinations])
