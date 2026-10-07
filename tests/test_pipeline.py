from dataclasses import replace
import json
from pathlib import Path
import threading
import time

import h5py
import numpy as np
import pytest

from fish_viewer.models import Channel, Dataset, LibraryEntry, Metadata, Scene
from fish_viewer.pipeline.config import PipelineConfig
from fish_viewer.pipeline.evaluation import aggregate, evaluate, rasterize_reference
from fish_viewer.pipeline import runner
from fish_viewer.pipeline.segmentation import (
    Cancelled, Prediction, Prepared, mask_properties, prediction_surfaces, prepare, segment,
)


@pytest.fixture
def synthetic(tmp_path):
    """Two hollow spheres and a misleadingly named punctum channel, in µm."""
    shape = (48, 80, 80)
    z, y, x = np.indices(shape) * 0.3
    r = np.sqrt((z-7.2)**2 + (y-7.2)**2 + (x-7.2)**2)
    s = np.sqrt((z-7.2)**2 + (y-16)**2 + (x-16)**2)
    rings = np.maximum(np.exp(-0.5*((r-2)/0.18)**2), np.exp(-0.5*((s-2.2)/0.18)**2))
    spots = np.maximum(np.exp(-r*r/0.12), np.exp(-s*s/0.12))
    channels = [Channel('Lamin 633', (1, 0, 0), excitation_nm=633), Channel('Unknown', (0, 0, 1))]
    metadata = Metadata(tmp_path / '101212_a.ims', shape, (10, 20, 30), (0.3, 0.3, 0.3), channels, [0], {0:shape})
    volumes = [(v*60000).astype(np.uint16) for v in (spots, rings)]
    truth = (r < 2).astype(np.uint32) + (s < 2.2).astype(np.uint32)*2
    prepared = Prepared(rings, (0.3,)*3, metadata.center_origin, 1, [], 'synthetic', {}, {})
    surfaces = prediction_surfaces(Prediction(truth, [], {}, {}, {}), prepared)
    for surface in surfaces:
        surface.source = 'manual'
    return Dataset(metadata, Scene(surfaces=surfaces), volumes), truth


def references(labels):
    box = tuple(slice(0, n) for n in labels.shape)
    return [(box, labels == i) for i in np.unique(labels) if i], ['Cell 1', 'Cell 2'][:int(labels.max())]


def write_ims(dataset, path):
    metadata = dataset.metadata
    with h5py.File(path, 'w') as f:
        group = f.create_group('DataSetInfo/Image')
        for axis in range(3):
            group.attrs[f'ExtMin{axis}'] = str(metadata.origin[axis]).encode()
            group.attrs[f'ExtMax{axis}'] = str(metadata.origin[axis] + metadata.extent[axis]).encode()
        group.attrs['Unit'] = b'um'
        for i, (channel, volume) in enumerate(zip(metadata.channels, dataset.volumes)):
            group = f.create_group(f'DataSetInfo/Channel {i}')
            group.attrs['Name'] = channel.name.encode()
            group.attrs['Color'] = ' '.join(map(str, channel.color)).encode()
            group.attrs['ColorRange'] = b'0 60000'
            group = f.create_group(f'DataSet/ResolutionLevel 0/TimePoint 0/Channel {i}')
            for axis, n in zip('ZYX', volume.shape):
                group.attrs[f'ImageSize{axis}'] = str(n).encode()
            group.create_dataset('Data', data=volume)


def test_automatic_channel_and_spherical_segmentation(synthetic):
    dataset, truth = synthetic
    prepared = prepare(dataset, PipelineConfig())
    # Display colors, a lamin name, and a wavelength hint cannot override rings.
    assert prepared.channel == 1
    assert prepared.channel_scores[0]['score'] == 0
    prediction = segment(prepared, PipelineConfig())
    assert len(prediction.cells) == 2
    refs, names = references(truth)
    scores = evaluate(prediction.labels, refs, names, prepared.spacing)
    assert scores['mean_dice'] > 0.95
    assert scores['detection_rate'] == 1
    np.testing.assert_allclose(prepared.origin, dataset.center_origin)
    np.testing.assert_allclose(prepared.spacing, (0.3,)*3)
    assert all(c['sphericity'] > 0.9 and c['elongation'] < 1.1 for c in prediction.cells)


def test_shape_metrics_use_physical_units_and_reject_elongation():
    z, y, x = np.indices((25, 65, 65))
    ellipsoid = ((z-12)/5)**2 + ((y-32)/18)**2 + ((x-32)/18)**2 <= 1
    physical_sphere = mask_properties(ellipsoid, np.array([0.72, 0.2, 0.2]))
    voxel_ellipsoid = mask_properties(ellipsoid, np.array([0.2, 0.2, 0.2]))
    assert physical_sphere['elongation'] < 1.1
    assert voxel_ellipsoid['elongation'] > 3
    # A round XY section alone is insufficient for acceptance in 3D.
    radius = np.sqrt(((z-12)*0.3/0.55)**2 + ((y-32)*0.3)**2 + ((x-32)*0.3)**2)
    shell = np.exp(-0.5*((radius-2)/0.15)**2).astype(np.float32)
    prepared = Prepared(shell, (0.3,)*3, (0, 0, 0), 0, [], 'ellipse', {}, {})
    broad = segment(prepared, replace(PipelineConfig(), max_elongation=3, min_sphericity=0.4))
    conservative = segment(prepared, replace(PipelineConfig(), max_elongation=1.3, min_sphericity=0.4))
    assert len(broad.cells) == 1
    assert not conservative.cells
    assert conservative.rejected['elongation'] == 1


def test_evaluation_identical_missing_unknown_and_merged_cells():
    truth = np.zeros((16, 16, 24), np.uint32)
    truth[4:10, 4:10, 3:9] = 1
    truth[4:10, 4:10, 13:19] = 2
    refs, names = references(truth)
    exact = evaluate(truth, refs, names, (1, 1, 1))
    assert exact['mean_dice'] == exact['mean_iou'] == exact['detection_rate'] == 1
    assert exact['boundary_rmse_um'] == exact['pair_mask_mse'] == 0
    extra = truth.copy()
    extra[12:14, 12:14, 12:14] = 3
    unknown = evaluate(extra, refs, names, (1, 1, 1))
    assert unknown['mean_dice'] == 1
    assert unknown['unscored_predictions'] == 1
    missing = evaluate(np.zeros_like(truth), refs, names, (1, 1, 1))
    assert missing['mean_dice'] == missing['detection_rate'] == 0
    assert missing['objective'] == 1
    assert missing['boundary_rmse_um'] is None
    merged = evaluate((truth > 0).astype(np.uint32), refs, names, (1, 1, 1))
    assert merged['matched_count'] == 1  # One predicted object cannot explain two cells.
    assert merged['mean_dice'] == pytest.approx(1/3)
    assert merged['detection_rate'] == 0.5
    empty = evaluate(extra, [], [], (1, 1, 1))
    assert empty['mean_dice'] is None and empty['unscored_predictions'] == 3
    assert aggregate([exact, missing, empty])['mean_dice'] == 0.5


def test_boundary_error_scales_in_micrometers():
    truth = np.zeros((12, 12, 12), np.uint32)
    truth[3:8, 3:8, 3:8] = 1
    refs, names = references(truth)
    shifted = np.roll(truth, 1, axis=0)
    small = evaluate(shifted, refs, names, (0.3,)*3)
    large = evaluate(shifted, refs, names, (0.6,)*3)
    assert 0 < small['boundary_rmse_um'] < 0.3
    assert large['boundary_rmse_um'] == pytest.approx(small['boundary_rmse_um']*2)
    assert small['pair_mask_mse'] == large['pair_mask_mse'] > 0


def test_edge_exclusion_accounts_for_artificial_background_frame():
    z, y, x = np.indices((32, 48, 48))*0.3
    distance = np.sqrt((z-0.9)**2 + (y-7.2)**2 + (x-7.2)**2)
    shell = np.exp(-0.5*((distance-2)/0.18)**2).astype(np.float32)
    prepared = Prepared(shell, (0.3,)*3, (0,)*3, 0, [], 'cut-cell', {}, {})
    allowed = segment(prepared, replace(PipelineConfig(), exclude_border=False))
    excluded = segment(prepared, PipelineConfig())
    assert len(allowed.cells) == 1
    assert not excluded.cells and excluded.rejected['border'] == 1


def test_mesh_rasterization_and_archive_preserve_calibration(synthetic, tmp_path):
    dataset, truth = synthetic
    prepared = prepare(dataset, PipelineConfig())
    refs, names, skipped = rasterize_reference(dataset.scene.surfaces, prepared)
    assert not skipped and len(refs) == 2
    assert evaluate(truth, refs, names, prepared.spacing)['mean_dice'] == 1
    prediction = segment(prepared, PipelineConfig())
    path = tmp_path / 'contours.npz'
    runner.save_prediction(path, prediction, prepared, LibraryEntry(dataset.metadata.path), PipelineConfig(), {})
    surfaces, metadata = runner.read_prediction(path)
    assert all(s.source == 'prediction' for s in surfaces)
    roundtrip, names, skipped = rasterize_reference(surfaces, prepared)
    assert not skipped
    assert evaluate(prediction.labels, roundtrip, names, prepared.spacing)['mean_dice'] == 1
    assert metadata['spacing_zyx'] == list(prepared.spacing)
    assert path.with_suffix('.cells.csv').is_file()
    assert metadata['fingerprint'] == prepared.fingerprint


def test_split_keeps_acquisitions_and_duplicate_pixels_together(tmp_path):
    cases = []
    for group, fingerprint in [('101212', 'a'), ('101212', 'b'), ('131212', 'c'), ('141212', 'c'), ('151212', 'd')]:
        entry = LibraryEntry(tmp_path / f'{group}_{len(cases)}.ims')
        prepared = Prepared(np.zeros((2,)*3), (1,)*3, (0,)*3, 0, [], fingerprint, {}, {})
        cases.append(runner.Case(entry, prepared, [], [], [], runner.acquisition_group(entry), tmp_path))
    tuning, evaluation = runner.split_cases(cases, 0.5)
    assert set(c.group for c in tuning).isdisjoint(c.group for c in evaluation)
    assert set(c.prepared.fingerprint for c in tuning).isdisjoint(c.prepared.fingerprint for c in evaluation)
    with pytest.raises(ValueError, match='two acquisition groups'):
        runner.split_cases(cases[:2])


def test_trial_bounds_and_reproducibility():
    config = replace(PipelineConfig(), smooth_um=5, seed_distance_um=100, threshold=0.9)
    trials = runner.trial_configs(config, 200)
    assert len(trials) == 200
    assert all(t.validate() for t in trials)
    assert trials == runner.trial_configs(config, 200)


def test_tuning_reserved_scores_only_after_parameter_selection(synthetic, tmp_path, monkeypatch):
    dataset, _ = synthetic
    entries = [LibraryEntry(tmp_path / f'{prefix}_{i}.ims') for i, prefix in enumerate(('101212', '101212', '131212'))]
    datasets = {}
    for i, entry in enumerate(entries):
        volumes = [v.copy() for v in dataset.volumes]
        volumes[0][0, 0, 0] = i  # Distinct source pixels with identical cell geometry.
        datasets[entry.image] = replace(dataset, metadata=replace(dataset.metadata, path=entry.image), volumes=volumes)
    monkeypatch.setattr(runner, 'load_dataset', lambda entry, **kwargs:datasets[entry.image])
    events = []
    report = runner.run_pipeline(entries, PipelineConfig(), tmp_path/'runs', trials=2, emit=events.append)
    assert report['status'] == 'complete', report
    assert len(report['trials']) == 2
    assert report['split']['evaluation'] == [str(entries[-1].image)]
    last_trial = max(i for i, e in enumerate(events) if e['kind'] == 'trial')
    reserved_results = [i for i, e in enumerate(events) if e['kind'] == 'result' and e['row']['split'] == 'evaluation']
    assert reserved_results and min(reserved_results) > last_trial
    trial_files = [e['file'] for e in events if e['kind'] == 'file' and e['phase'].startswith('Trial')]
    assert str(entries[-1].image) not in trial_files
    assert report['evaluation']['scored_images'] == 1
    assert report['evaluation']['mean_dice'] > 0.95
    output = Path(report['directory'])
    for name in ('report.json', 'settings.json', 'scores.png', 'scores.svg', 'trials.csv', 'cells-evaluation.csv'):
        assert (output/name).stat().st_size > 0
    persisted = json.loads((output/'report.json').read_text())
    assert persisted['best_config'] == report['best_config']
    assert len(persisted['files']) == 3


def test_cancellation_and_file_errors_are_reported(ims_file, tmp_path, monkeypatch):
    from datetime import datetime
    from types import SimpleNamespace
    from fish_viewer import run_paths
    instant = datetime(2026, 10, 7, 0, 0, 0)
    monkeypatch.setattr(run_paths, 'datetime', SimpleNamespace(now=lambda: instant))
    event = threading.Event()
    event.set()
    entries = [LibraryEntry(ims_file)]
    report = runner.run_pipeline(entries, PipelineConfig(), tmp_path/'runs', mode='apply', cancel=event)
    assert report['status'] == 'cancelled' and not report['files']
    assert json.loads((Path(report['directory'])/'report.json').read_text())['status'] == 'cancelled'
    bad = runner.run_pipeline([LibraryEntry(tmp_path/'absent.ims')], PipelineConfig(), tmp_path/'runs', mode='apply')
    assert bad['status'] == 'failed' and len(bad['failures']) == 1
    assert bad['directory'] != report['directory']
    assert json.loads((Path(report['directory'])/'report.json').read_text())['status'] == 'cancelled'
    with pytest.raises(Cancelled):
        segment(Prepared(np.zeros((8,)*3), (0.3,)*3, (0,)*3, 0, [], '', {}, {}), PipelineConfig(), cancel=event)


@pytest.mark.gui
def test_gui_apply_background_progress_and_prediction_overlay(qapp, synthetic, tmp_path):
    from fish_viewer.app import MainWindow
    dataset, _ = synthetic
    path = dataset.metadata.path
    write_ims(dataset, path)
    window = MainWindow(off_screen=True)
    errors = []
    window.error = errors.append
    try:
        window.show()
        window.show_pipeline()
        panel = window.pipeline_panel
        panel.show_error = errors.append
        panel.set_entries([LibraryEntry(path)])
        panel.mode.setCurrentIndex(1)
        panel.output.setText(str(tmp_path/'gui-runs'))
        panel.set_config(replace(PipelineConfig(), smooth_um=0.117))
        assert panel.config().smooth_um == 0.117
        panel.start()
        assert panel.job is not None and panel.cancel_button.isEnabled()
        deadline = time.monotonic() + 30
        while panel.job is not None and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert not errors, errors
        assert panel.job is None and panel.report['status'] == 'complete'
        assert panel.results.rowCount() == 1 and panel.channel_scores[1]['score'] > 0
        assert 'Accepted spherical cells' in panel.previews
        assert panel.progress.value() == 100
        panel.open_overlay()
        deadline = time.monotonic() + 15
        while window._pending_load and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert not errors, errors
        assert len(window.dataset.scene.surfaces) == 2
        assert all(s.source == 'prediction' for s in window.dataset.scene.surfaces)
        assert window.slice_view._overlays
        panel.predicted_visible.setChecked(False)
        assert not window.slice_view._overlays
        panel.predicted_visible.setChecked(True)
        assert window.slice_view._overlays
        # Reopening checks the archive against original full-resolution pixels.
        changed = replace(dataset, volumes=[v+1 for v in dataset.volumes])
        write_ims(changed, path)
        panel.open_overlay()
        deadline = time.monotonic() + 15
        while window._pending_load and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert not window.dataset.scene.surfaces
        assert any('pixels differ' in w for w in window.dataset.scene.warnings)
    finally:
        window.close()
        qapp.processEvents()
