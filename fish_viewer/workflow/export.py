"""Export byte-identical IMS images with complete native Imaris 6.3.1 scenes."""
import hashlib
import json
from pathlib import Path
import tempfile

import h5py

from ..models import LibraryEntry
from ..pipeline.runner import read_prediction
from ..pipeline.segmentation import checkpoint
from ..reader import ImarisReader, load_dataset
from .imaris_scene import scene_xml
from .results import validated_result, write_tables


def _copy_image(source, destination, emit, cancel):
    # The FISH dataset stores annotations externally. Refuse embedded scenes
    # instead of quietly exporting stale annotations inside a supposedly new image.
    if h5py.is_hdf5(source):
        with h5py.File(source, 'r') as image:
            if any(key.startswith('Scene') for key in image):
                raise ValueError('This exporter requires an image with external .imx annotations, '
                                 'as in FISH. Embedded Imaris scenes are not supported.')
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
    return digest.hexdigest()


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
    if len(surfaces) != len({point['cell_id'] for point in report['points']}):
        raise ValueError('The filtered contour set does not match the detected signals.')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.fish-export-', dir=target.parent) as temporary:
        staging = Path(temporary)
        source = Path(report['image'])
        digest = _copy_image(source, staging / target.name, emit, cancel)
        (staging / target.with_suffix('.imx').name).write_bytes(scene_xml(target.name, surfaces, metadata, report))
        write_tables(staging, report, target.stem + '.')
        annotation_report = dict(report, engine='fish-scan-export', mode='export', image=target.name, scene=target.with_suffix('.imx').name,
                                 directory='.', cell_predictions=None, export_format='Imaris IMS + XML IMX',
                                 source_signal_report=str(Path(report_path).resolve()), image_sha256=digest,
                                 scene_profile='FISH Imaris 6.3.1 native XML')
        (staging / target.with_suffix('.annotations.json').name).write_text(json.dumps(annotation_report, indent=2, allow_nan=False), encoding='utf-8')
        emit(dict(kind='progress', phase='Verifying exported scene', current=0, total=1))
        reopened = load_dataset(LibraryEntry(staging / target.name, staging / target.with_suffix('.imx').name))
        if (len(reopened.scene.surfaces) != len(surfaces) or reopened.scene.warnings
                or sum(len(group.points) for group in reopened.scene.measurements) != len(report['points'])):
            raise ValueError('Exported annotations failed the reload check.')
        checkpoint(cancel)
        _publish(staging, destinations)
    emit(dict(kind='progress', phase='Export complete', current=1, total=1))
    return dict(status='complete', image=str(target), scene=str(target.with_suffix('.imx')),
                cells=len(surfaces), signals=len(report['points']), image_sha256=digest, files=[str(path) for path in destinations])


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
