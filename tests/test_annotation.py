import json
from pathlib import Path

import numpy as np
import pytest

from fish_viewer.annotation.data import TrainingSampler, fill, paint, save_crop


def sampler_for(path, tmp_path):
    document = {'cases': [
        {'image': str(path), 'audit': {'split': 'training', 'group': 'train', 'channel': 1}},
        {'image': str(tmp_path / 'heldout.ims'), 'audit': {'split': 'test', 'group': 'test'}},
    ]}
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps(document))
    return TrainingSampler(audit, (2, 3), seed=12)


def test_training_crop_native_coordinates_and_save(ims_file, tmp_path):
    sampler = sampler_for(ims_file, tmp_path)
    assert len(sampler.cases) == 1
    crop = sampler.next_crop()
    assert crop.image.shape == (2, 3)
    assert crop.image.max() < 65535  # Imaris padding never enters the crop
    assert crop.metadata['split'] == 'training'
    offset = crop.metadata['offset_zyx']
    np.testing.assert_allclose(crop.metadata['origin_xyz'], np.array([10, 20, 30]) + 2 * np.array(offset[::-1]))
    output = tmp_path / 'annotations'
    with pytest.raises(ValueError, match='No marks'):
        save_crop(crop, output)
    assert not output.exists()
    paint(crop.labels, (.5, .5), (2.5, .5), 1, 3)
    assert np.all(crop.labels[0] == 3)
    path = save_crop(crop, output)
    data = np.load(path / 'annotation.npz')
    np.testing.assert_array_equal(data['image'], crop.image)
    np.testing.assert_array_equal(data['labels'], crop.labels)
    metadata = json.loads((path / 'metadata.json').read_text())
    assert metadata['categories']['0'] == 'unlabeled'
    assert metadata['category_pixel_counts']['3'] == 3
    assert not list(output.glob('.pending-*'))


def test_leakage_rejected(ims_file, tmp_path):
    sampler = sampler_for(ims_file, tmp_path)
    doc = json.loads(sampler.audit_path.read_text())
    doc['cases'][1]['audit']['aliases'] = [str(ims_file)]
    sampler.audit_path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='leakage'):
        TrainingSampler(sampler.audit_path)


def test_brush_unknown_overwrite_eraser_and_interpolation():
    labels = np.zeros((25, 30), np.uint8)
    paint(labels, (2.5, 2.5), (20.5, 2.5), 3, None)
    assert not labels.any()
    paint(labels, (2.5, 2.5), (20.5, 2.5), 3, 1)
    assert np.all(labels[2, 2:21] == 1)
    paint(labels, (10.5, 2.5), (10.5, 2.5), 3, 3)
    assert labels[2, 10] == 3
    paint(labels, (10.5, 2.5), (10.5, 2.5), 3, 0)
    assert labels[2, 10] == 0
    assert not labels[20].any()
    paint(labels, (-100, -100), (-80, -80), 3, 2)
    assert not labels[20:].any()


def wait_idle(qapp, window):
    from PySide6.QtCore import QElapsedTimer
    from PySide6.QtTest import QTest
    timer = QElapsedTimer()
    timer.start()
    while window.busy and timer.elapsed() < 10000:
        qapp.processEvents()
        QTest.qWait(10)
    assert not window.busy
    qapp.processEvents()


@pytest.mark.gui
def test_gui_no_category_zoom_undo_fill_save_skip(qapp, ims_file, tmp_path):
    from PySide6 import QtCore
    from PySide6.QtTest import QTest
    from fish_viewer.annotation.app import AnnotationWindow
    sampler = sampler_for(ims_file, tmp_path)
    output = tmp_path / 'saved'
    window = AnnotationWindow(sampler, output, autoload=False)
    window.resize(1000, 720)
    window.show()
    window.next_crop()
    wait_idle(qapp, window)
    canvas = window.canvas
    # Click at a native pixel after zooming; unset category never paints.
    canvas.scale(1.5, 1.5)
    point = canvas.mapFromScene(QtCore.QPointF(1.5, .5))
    QTest.mouseClick(canvas.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=point)
    assert not canvas.crop.labels.any()
    assert not window.save.isEnabled()
    window.group.button(3).click()
    window.size.setValue(1)
    QTest.mouseClick(canvas.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=point)
    assert canvas.crop.labels[0, 1] == 3
    assert window.save.isEnabled()
    window.activateWindow()
    qapp.processEvents()
    QTest.keyClick(window, QtCore.Qt.Key.Key_Z, QtCore.Qt.KeyboardModifier.ControlModifier)
    assert not canvas.crop.labels.any()
    assert not window.save.isEnabled()
    # Fill all unknown pixels as one edit, and clear as a second edit.
    window.group.button(1).click()
    window.fill_button.click()
    QTest.mouseClick(canvas.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=point)
    assert np.all(canvas.crop.labels == 1)
    window.clear_all()
    assert not canvas.crop.labels.any()
    canvas.undo()
    assert np.all(canvas.crop.labels == 1)
    # Display controls cannot alter labels.
    saved_labels = canvas.crop.labels.copy()
    window.set_gain(200)
    np.testing.assert_array_equal(saved_labels, canvas.crop.labels)
    assert not hasattr(window, 'layer')
    assert not hasattr(window, 'channel_row')
    window.save_next()
    wait_idle(qapp, window)
    assert len(list(output.glob('crop-*'))) == 1
    assert not canvas.crop.labels.any()
    # A painted, skipped crop must not be persisted.
    paint(canvas.crop.labels, (.5, .5), (.5, .5), 1, 1)
    window.skip_crop()
    wait_idle(qapp, window)
    assert len(list(output.glob('crop-*'))) == 1
    assert not canvas.crop.labels.any()
    canvas.crop.labels.fill(3)
    window.clear_all()
    assert not canvas.crop.labels.any()
    assert not window.save.isEnabled()
    window.close()


def test_failed_save_leaves_no_partial_record(ims_file, tmp_path, monkeypatch):
    crop = sampler_for(ims_file, tmp_path).next_crop()
    crop.labels[0, 0] = 1
    output = tmp_path / 'failed-save'
    def fail(*args, **kwargs):
        raise OSError('Disk full')
    monkeypatch.setattr(np, 'savez_compressed', fail)
    with pytest.raises(OSError, match='Disk full'):
        save_crop(crop, output)
    assert not list(output.iterdir())
    assert crop.labels[0, 0] == 1


def test_relocation_preserves_training_split(ims_file, tmp_path):
    audit = tmp_path / 'relocated.json'
    audit.write_text(json.dumps({'cases': [
        {'image': '/old/machine/FISH/sample.ims', 'audit': {'split': 'training', 'group': 'train', 'channel': 1}}]}))
    sampler = TrainingSampler(audit, fish_root=tmp_path)
    crop = sampler.next_crop()
    assert crop.metadata['source'] == str(ims_file.resolve())
    assert crop.metadata['audited_source_path'] == '/old/machine/FISH/sample.ims'
    assert crop.metadata['split'] == 'training'


def test_fill_respects_closed_contour_and_unset_category():
    labels = np.zeros((20, 20), np.uint8)
    labels[4, 4:16] = labels[15, 4:16] = 3
    labels[4:16, 4] = labels[4:16, 15] = 3
    original = labels.copy()
    fill(labels, (8, 8), None)
    np.testing.assert_array_equal(original, labels)
    fill(labels, (8, 8), 1)
    assert np.all(labels[5:15, 5:15] == 1)
    assert labels[0, 0] == 0
    assert labels[4, 8] == 3
    fill(labels, (-.2, 3), 1)
    assert labels[0, 0] == 0


def test_single_stroke_is_one_undo_and_resets_between_images(qapp, ims_file, tmp_path):
    from PySide6 import QtCore
    from PySide6.QtTest import QTest
    from fish_viewer.annotation.app import Canvas
    sampler = sampler_for(ims_file, tmp_path)
    canvas = Canvas()
    canvas.resize(500, 500)
    canvas.show()
    canvas.set_crop(sampler.next_crop())
    canvas.category = 3
    canvas.diameter = 1
    start = canvas.mapFromScene(QtCore.QPointF(.5, .5))
    end = canvas.mapFromScene(QtCore.QPointF(2.5, .5))
    QTest.mousePress(canvas.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(canvas.viewport(), end)
    QTest.mouseRelease(canvas.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=end)
    assert np.all(canvas.crop.labels[0] == 3)
    assert len(canvas.history) == 1
    canvas.undo()
    assert not canvas.crop.labels.any()
    canvas.category = 0
    QTest.mouseClick(canvas.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=start)
    assert not canvas.history  # erasing unknown is not an edit
    canvas.category = 3
    QTest.mouseClick(canvas.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=start)
    assert canvas.history
    canvas.set_crop(sampler.next_crop())
    assert not canvas.history
    canvas.close()
