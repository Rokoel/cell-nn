"""Write the native XML layout observed in the FISH Imaris 6.3.1 scenes.

The bundled template retains the complete application and object settings. Mesh
blocks follow the native 400-record / 40-line layout, including quoted records.
This is a compatibility profile, not an official Imaris schema or SDK.
"""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta
import base64
import bz2
from importlib.resources import files
import re
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from ..reader import ImarisReader, attributes, decode
from ..scenes import numbers


PROVENANCE = 'FISH Viewer exported detections;'


def template():
    return ET.fromstring(files(__package__).joinpath('templates/imaris-6.3.1.xml').read_bytes())


def _set(parent, path, value):
    element = parent.find(path)
    if element is None:
        raise ValueError(f'Incomplete Imaris template: {path}')
    element.text = str(value)


def _component(obj, name, color=None, opacity=1.):
    component = obj.find('bpSurfaceComponent')
    _set(component, 'name', name)
    _set(component, 'isHidden', 'false')
    if color is not None:
        _set(component, 'material/diffuseColor', '((' + ','.join(f'{c:.9g}' for c in color) + '))')
    _set(component, 'material/transparency', f'({1-opacity:.9g})')
    return component


def _array(parent, tag, rows, integer=False):
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != (3 if integer else 6) or not np.isfinite(rows).all():
        raise ValueError(f'Invalid Imaris {tag} records.')
    for start in range(0, len(rows), 400):
        block = rows[start:start + 400]
        lines = []
        for line in range(0, len(block), 10):
            tokens = [str(int(v)) if integer else f'{v:.9g}' for v in block[line:line + 10].ravel()]
            lines.append('"' + ' '.join(tokens) + ' "')
        ET.SubElement(parent, tag).text = '\n' + '\n'.join(lines) + '\n'


def _mesh(obj, surface):
    vertices = np.asarray(surface.vertices, dtype=float)
    faces = np.asarray(surface.triangles).copy()
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3
            or not len(faces) or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('Invalid cell mesh for Imaris export.')
    centered = vertices - vertices.mean(axis=0)
    a, b, c = (centered[faces[:, i]] for i in range(3))
    if np.einsum('ij,ij->', a, np.cross(b, c)) < 0:
        faces = faces[:, [0, 2, 1]]
    normals = np.zeros_like(vertices)
    face_normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                           vertices[faces[:, 2]] - vertices[faces[:, 0]])
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    mesh = obj.find('surface')
    mesh.clear()
    _array(mesh, 'vertices', np.column_stack((vertices, normals)))
    _array(mesh, 'triangles', faces, integer=True)
    ET.SubElement(mesh, 'timeIndex').text = str(surface.time)


def _image_context(metadata):
    """Use the image's acquisition times and image-only thumbnail, never a scene screenshot."""
    times = {}
    thumbnail = np.zeros((32, 32, 4), np.uint8)
    reader = ImarisReader(metadata.path)
    if reader.hdf:
        with h5py.File(metadata.path, 'r') as image:
            path = f'{reader.info_name}/TimeInfo'
            if path in image:
                info = attributes(image[path])
                times = {t: info[f'TimePoint{t+1}'] for t in metadata.times if f'TimePoint{t+1}' in info}
            path = decode(image.attrs.get('ThumbnailDirectoryName', 'Thumbnail')) + '/Data'
            if path in image:
                raw = image[path][()]
                if raw.dtype == np.uint8 and raw.ndim == 2 and raw.shape[1] % 4 == 0:
                    thumbnail = raw.reshape(raw.shape[0], raw.shape[1] // 4, 4)
    else:
        from configparser import ConfigParser
        import tifffile
        with tifffile.TiffFile(metadata.path) as image:
            info = ConfigParser(interpolation=None, strict=False)
            info.read_string(decode(image.pages[0].tags['ImageDescription'].value))
            for t in metadata.times:
                value = info.get('TimeInfo', f'TimePoint{t+1}', fallback=None)
                if value:
                    times[t] = value
    for t in metadata.times:
        # Some valid images lack timestamps. Use deterministic placeholders only
        # for those images; never copy acquisition times from the template scan.
        times.setdefault(t, (datetime(1970, 1, 1) + timedelta(seconds=t)).strftime('%Y-%m-%d %H:%M:%S.000'))
    return times, thumbnail


def scene_xml(image_name, surfaces, metadata, report):
    root = template()
    root.find('bpCommandLoadDataSet').set('file', image_name)
    application = root.find('bpSurfaceApplication')
    scene = application.find('bpComponentGroup/bpSurfaceComponent')
    prototype = scene.find('bpComponentGroup')
    scene.remove(prototype)
    _set(scene.find('bpVolumeRenderer'), 'bpSurfaceComponent/isHidden', 'false')
    times, thumbnail = _image_context(metadata)
    field = root.find('Thumbnail/bpImageField')
    for i, size in enumerate((thumbnail.shape[1], thumbnail.shape[0], 1)):
        field.set(f'mSize{i}', str(size))
        field.set(f'mExtendMin{i}', '0.000')
        field.set(f'mExtendMax{i}', f'{size / 100 if i < 2 else 1:.3f}')
    encoded = base64.b64encode(bz2.compress(thumbnail.tobytes())).decode('ascii')
    field.text = '\n' + '\n'.join(encoded[i:i+76] for i in range(0, len(encoded), 76)) + '\n'
    counts = Counter(p['cell_id'] for p in report['points'])
    grouped = {}
    for point in report['points']:
        grouped.setdefault(point['cell_id'], []).append(point)
    for surface in surfaces:
        cell = int(surface.name.rsplit(' ', 1)[-1])
        group = deepcopy(prototype)
        name = f'Predicted cell {cell} - {counts[cell]} signals'
        component = _component(group, name)
        _set(component, 'isExpanded', 'true')
        viewer = component.find('bpSurfacesViewer')
        _component(viewer, name, surface.color, surface.opacity)
        viewer.find('bpSurfacesProperties').set('mTransparency', f'{1-surface.opacity:.9g}')
        unit = 'um' if metadata.unit in ('µm', 'um') else metadata.unit
        viewer.find('bpTracksProperties').set('mUnit', unit)
        objects = viewer.find('bpSurfaces')
        _set(objects, 'name', name)
        _set(objects, 'description', f'Neural cell {cell}; {counts[cell]} detected signal centers inside.')
        _set(objects, 'historyLog', PROVENANCE + ' neural contours and signal centers.')
        _set(objects, 'unit', unit)
        objects.find('bpMaterial').set('mDiffusion', ' '.join(f'{v:.9g}' for v in surface.color))
        objects.find('bpMaterial').set('mTransparency', f'{1-surface.opacity:.9g}')
        creation = objects.find('bpSurfacesCreationParameters')
        creation.set('mSourceChannelIndex', str(report.get('membrane_channel', 0)))
        roi = creation.find('bpRegionOfInterestContainer/bpRegionOfInterest')
        roi.set('mMinXYZ', '0 0 0')
        roi.set('mMaxXYZ', ' '.join(str(n-1) for n in metadata.shape[::-1]))
        roi.set('mMaxT', str(max(metadata.times)))
        for element in objects.findall('timeInfo'):
            objects.remove(element)
        for index, value in sorted(times.items()):
            objects.insert(list(objects).index(objects.find('surface')),
                           ET.Element('timeInfo', index=str(index), time=value))
        _mesh(objects, surface)
        marker_prototype = component.find('bpMeasurementPoints')
        component.remove(marker_prototype)
        # All 3,455 native measurement objects use Polygon; one object per center
        # avoids both connecting unrelated signals and inventing an enum value.
        for point in grouped.get(cell, []):
            marker = deepcopy(marker_prototype)
            channel = point['channel']
            channel_info = report['channels'][channel]
            label = (f'Signal {point["signal_id"]}; cell {cell}; channel {channel+1}; '
                     f'score {point["score"]:.12g}')
            _component(marker, label + ' - ' + channel_info['name'], channel_info['color'])
            marker.insert(1, ET.Element('Point', Name=label,
                          Position=' '.join(f'{v:.12g}' for v in point['xyz']),
                          TimeIndex=str(report.get('time', 0))))
            _set(marker, 'PointDiameter', '0.15')
            _set(marker, 'VisibleForAllTimepoints', 'false')
            marker.find('LineLabels').set('Distance', 'false')
            component.append(marker)
        scene.append(group)
    for identifier, component in enumerate(root.iter('bpSurfaceComponent'), 25):
        _set(component, 'id', identifier)
    camera = application.find('SoPerspectiveCamera')
    size = np.asarray(metadata.shape[::-1]) * metadata.spacing
    center = np.asarray(metadata.origin) + size / 2
    distance = float(max(size) * 1.8)
    _set(camera, 'position', '(' + ', '.join(f'{v:.9g}' for v in center + [0, 0, distance]) + ')')
    _set(camera, 'orientation', '(0, 0, 1, 0)')
    _set(camera, 'focalDistance', f'{distance:.9g}')
    validate_scene(root, metadata)
    ET.indent(root)
    # Native examples start with the application element, have CRLF line endings,
    # and have no XML declaration or comments for a legacy parser to interpret.
    return ET.tostring(root, encoding='utf-8', short_empty_elements=False).replace(b'\n', b'\r\n') + b'\r\n'


def validate_scene(root, metadata):
    """Validate native field structure in addition to our permissive scene reader."""
    native = template()
    optional = {'bpComponentGroup'}  # A scene may have no annotation groups.

    def check(actual, expected):
        if actual.tag != expected.tag or set(actual.attrib) != set(expected.attrib):
            raise ValueError(f'Non-native Imaris structure at {actual.tag}.')
        prototypes = {e.tag: e for e in expected}
        if actual.tag == 'surface':
            prototypes.update(vertices=ET.Element('vertices'), triangles=ET.Element('triangles'))
            order = ['vertices', 'triangles', 'timeIndex']
        elif actual.tag == 'bpMeasurementPoints':
            prototypes['Point'] = ET.Element('Point', Name='', Position='', TimeIndex='')
            order = list(prototypes)
            order.remove('Point')
            order.insert(1, 'Point')
        else:
            order = list(prototypes)
        actual_order = list(dict.fromkeys(e.tag for e in actual))
        required = [tag for tag in order if tag not in optional or actual.tag != 'bpSurfaceComponent']
        if any(tag not in actual_order for tag in required) or actual_order != [tag for tag in order if tag in actual_order]:
            raise ValueError(f'Missing or reordered native Imaris fields at {actual.tag}.')
        for child in actual:
            if child.tag not in prototypes:
                raise ValueError(f'Unknown Imaris field {child.tag}.')
            check(child, prototypes[child.tag])

    check(root, native)
    ids = [int(c.findtext('id')) for c in root.iter('bpSurfaceComponent')]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate Imaris component identifiers.')
    for objects in root.iter('bpSurfaces'):
        time_indices = {int(e.get('index')) for e in objects.findall('timeInfo')}
        if time_indices != set(metadata.times):
            raise ValueError('Surface timestamps differ from the scan time points.')
        for mesh in objects.findall('surface'):
            arrays = {}
            for tag, width in [('vertices', 6), ('triangles', 3)]:
                records = []
                for block in mesh.findall(tag):
                    text = block.text or ''
                    lines = re.findall(r'"([^"\n]*)"', text)
                    if re.sub(r'"[^"\n]*"', '', text).strip() or not 1 <= len(lines) <= 40:
                        raise ValueError(f'Invalid native Imaris {tag} block.')
                    for line in lines:
                        values = numbers(line)
                        if not 1 <= len(values) // width <= 10 or len(values) % width or not np.isfinite(values).all():
                            raise ValueError(f'Invalid native Imaris {tag} row.')
                        records.extend(values)
                arrays[tag] = np.asarray(records).reshape(-1, width)
            vertices, faces = arrays['vertices'], arrays['triangles']
            if (not len(vertices) or not len(faces) or not np.equal(faces, np.floor(faces)).all()
                    or faces.min() < 0 or faces.max() >= len(vertices)
                    or int(mesh.findtext('timeIndex')) not in time_indices):
                raise ValueError('Invalid Imaris mesh indices or time index.')
    for marker in root.iter('bpMeasurementPoints'):
        if marker.findtext('LineMode') != 'Polygon' or len(marker.findall('Point')) != 1:
            raise ValueError('Exported signal centers require separate native measurement objects.')
        point = marker.find('Point')
        values = numbers(point.get('Position'))
        if len(values) != 3 or not np.isfinite(values).all() or int(point.get('TimeIndex')) not in metadata.times:
            raise ValueError('Invalid Imaris signal position or time.')
