"""Export byte-identical IMS images with complete native Imaris 6.3.1 scenes."""
import hashlib
import json
from pathlib import Path
import tempfile

import h5py

from ..models import LibraryEntry
from ..pipeline.runner import read_prediction
from ..pipeline.segmentation import checkpoint
from ..reader import ImarisReader, load_dataset, entry_for_path
from .imaris_scene import scene_xml
from .results import validated_result, write_tables


def _copy_image(source, destination, emit, cancel):
    before = source.stat()
    total, copied = before.st_size, 0
    digest = hashlib.sha256()
    with source.open('rb') as src, destination.open('wb') as dst:
        while block := src.read(8 * 1024 * 1024):
            checkpoint(cancel)
            dst.write(block)
            digest.update(block)
            copied += len(block)
            emit(dict(kind='progress', phase='Copying Imaris image (all channels, times and resolutions)',
                      current=copied, total=total))
    if source.stat().st_mtime_ns != before.st_mtime_ns or copied != total:
        raise ValueError('The source scan changed during export. Please retry.')
    with destination.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != digest.hexdigest():
            raise ValueError('The exported image differs from its source.')
    # Replace embedded scene objects with the companion's current annotations.
    # Copying first preserves all image datasets, acquisition metadata and pyramids.
    embedded = False
    if h5py.is_hdf5(destination):
        with h5py.File(destination, 'r') as image:
            embedded = any(k.startswith('Scene') for k in image) or any(k.startswith('Scene') for k in image.attrs)
    if embedded:
        with h5py.File(destination, 'r+') as image:
            for key in list(image):
                if key.startswith('Scene'):
                    del image[key]
            for key in list(image.attrs):
                if key.startswith('Scene'):
                    del image.attrs[key]
    with destination.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _publish(staging, destinations):
    created = []
    try:
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


def export_result(report_path, target, emit=lambda event: None, cancel=None, source_scene=None):
    """Publish a new IMS/IMX pair and tables; refuse to overwrite any existing file."""
    target = Path(target).expanduser().resolve()
    if target.suffix.lower() != '.ims':
        raise ValueError('Choose an export filename ending in .ims.')
    suffixes = ['.ims', '.imx', '.signals.csv', '.cell-counts.csv', '.annotations.json', '.xlsx']
    destinations = [target.with_suffix(suffix) for suffix in suffixes]
    if any(path.exists() for path in destinations):
        raise FileExistsError('The export would overwrite existing files. Choose a new name or folder.')
    emit(dict(kind='progress', phase='Validating detections for export', current=0, total=1))
    report, labels, cell_metadata, metadata = validated_result(report_path, cancel)
    if not report.get('signal_filter') or report['signal_filter'] != cell_metadata.get('signal_filter'):
        raise ValueError('Remove cells without signals before exporting this workflow result.')
    surfaces, _ = read_prediction(report['cell_predictions'])
    from .volumes import add_cell_volumes
    from .excel import write_excel
    add_cell_volumes(report, surfaces, labels, cell_metadata['spacing_zyx'])
    from .distances import validate_measurements
    validate_measurements(report, surfaces, metadata)
    if len(surfaces) != len({point['cell_id'] for point in report['points']}):
        raise ValueError('The filtered contour set does not match the detected signals.')
    target.parent.mkdir(parents=True, exist_ok=True)
    from .manual import manual_measurements
    emit(dict(kind='progress', phase='Reading manual cell volumes and signal distances', current=0, total=1))
    if source_scene: report['scene'] = str(source_scene)
    report['manual_measurements'] = manual_measurements(report, cancel)
    source_entry=LibraryEntry(Path(report['image']),Path(source_scene)) if source_scene else entry_for_path(report['image'])
    original_scene=load_dataset(source_entry).scene
    combined=None
    if original_scene.surfaces or original_scene.measurements or original_scene.contours:
        from ..scenes import parse_scene
        from copy import deepcopy
        generated=parse_scene(scene_xml(target.name,surfaces,metadata,report))
        combined=deepcopy(original_scene)
        combined.surfaces.extend(generated.surfaces)
        combined.measurements.extend(generated.measurements)
    if original_scene.warnings and any('unsupported' in w.lower() or 'not supported' in w.lower() or 'cannot read' in w.lower() for w in original_scene.warnings):
        raise ValueError('Some imported annotations could not be read; export cannot safely discard them: '+'; '.join(original_scene.warnings))
    with tempfile.TemporaryDirectory(prefix='.fish-export-', dir=target.parent) as temporary:
        staging = Path(temporary)
        source = Path(report['image'])
        digest = _copy_image(source, staging / target.name, emit, cancel)
        if combined is not None:
            from .edit_export import editable_scene_xml
            scene_data=editable_scene_xml(target.name,combined,metadata)
        else: scene_data=scene_xml(target.name,surfaces,metadata,report)
        (staging / target.with_suffix('.imx').name).write_bytes(scene_data)
        write_tables(staging, report, target.stem + '.')
        write_excel(staging / target.with_suffix('.xlsx').name, report)
        annotation_report = dict(report, engine='fish-scan-export', mode='export', image=target.name, scene=target.with_suffix('.imx').name,
                                 directory='.', cell_predictions=None, export_format='Imaris IMS + XML IMX',
                                 source_signal_report=str(Path(report_path).resolve()), image_sha256=digest,
                                 scene_profile='FISH Imaris 6.3.1 native XML')
        (staging / target.with_suffix('.annotations.json').name).write_text(json.dumps(annotation_report, indent=2, allow_nan=False), encoding='utf-8')
        emit(dict(kind='progress', phase='Verifying exported scene', current=0, total=1))
        reopened = load_dataset(LibraryEntry(staging / target.name, staging / target.with_suffix('.imx').name))
        expected_points = len(report['points']) * (2 if report.get('measurement') else 1)
        expected_lines = len(report['points']) if report.get('measurement') else 0
        expected_surfaces=len(surfaces)
        if combined is not None:
            expected_surfaces=len(combined.surfaces)
            expected_points=sum(len(m.points) for m in combined.measurements)
            expected_lines=sum(len(m.edges) for m in combined.measurements)
        if (len(reopened.scene.surfaces) != expected_surfaces or reopened.scene.warnings
                or sum(len(group.points) for group in reopened.scene.measurements) != expected_points
                or sum(len(group.edges) for group in reopened.scene.measurements) != expected_lines):
            raise ValueError('Exported annotations failed the reload check.')
        checkpoint(cancel)
        _publish(staging, destinations)
    emit(dict(kind='progress', phase='Export complete', current=1, total=1))
    return dict(status='complete', image=str(target), scene=str(target.with_suffix('.imx')),
                cells=len(surfaces), signals=len(report['points']), measurements=expected_lines,
                image_sha256=digest, files=[str(path) for path in destinations])


def export_unannotated(source, target, emit=lambda event: None, cancel=None):
    """Make a new image/empty-scene pair, without copying any source annotations."""
    source, target = Path(source).expanduser().resolve(), Path(target).expanduser().resolve()
    if target.suffix.lower() != '.ims':
        raise ValueError('Choose an export filename ending in .ims.')
    destinations = [target, target.with_suffix('.imx')]
    if any(path.exists() for path in destinations):
        raise FileExistsError('The export would overwrite existing files. Choose a new name or folder.')
    metadata = ImarisReader(source).metadata
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fish-export-', dir=target.parent) as temporary:
        staging = Path(temporary)
        digest = _copy_image(source, staging / target.name, emit, cancel)
        scene = staging / target.with_suffix('.imx').name
        scene.write_bytes(scene_xml(target.name, [], metadata, dict(points=[])))
        reopened = load_dataset(LibraryEntry(staging / target.name, scene))
        if (reopened.scene.surfaces or reopened.scene.measurements or reopened.scene.contours
                or reopened.scene.warnings):
            raise ValueError('The unannotated copy failed the reload check.')
        checkpoint(cancel)
        _publish(staging, destinations)
    return dict(status='complete', image=str(target), scene=str(target.with_suffix('.imx')),
                cells=0, signals=0, source=str(source), image_sha256=digest,
                files=[str(path) for path in destinations])
