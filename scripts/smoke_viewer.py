"""Exercise the real desktop renderer and save review screenshots, then exit."""
from pathlib import Path
from dataclasses import replace
import argparse
import json
import os
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['QT_API'] = 'pyside6'
if sys.platform.startswith('linux'):
    os.environ['QT_QPA_PLATFORM'] = 'xcb'

import numpy as np
from PySide6 import QtCore, QtWidgets
from fish_viewer.app import MainWindow
from fish_viewer.reader import entry_for_path, load_dataset, scan_library


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--pipeline', action='store_true', help='Also run two GUI tuning trials and inspect prediction overlays')
parser.add_argument('--neural-report', type=Path, help='Open a trained neural run, apply its model through the GUI, and inspect overlays')
parser.add_argument('--neural-train', action='store_true', help='Also exercise one live training epoch through the neural GUI')
args = parser.parse_args()
if args.neural_train and not args.neural_report:
    parser.error('--neural-train requires --neural-report')
app = QtWidgets.QApplication([])
print('Qt platform:', app.platformName(), flush=True)
app.setStyle('Fusion')
window = MainWindow()
window.show()
output = Path('artifacts')
output.mkdir(exist_ok=True)


def exercise_pipeline():
    window.show_pipeline()
    panel = window.pipeline_panel
    failures = []
    panel.show_error = failures.append
    panel.set_entries(scan_library('FISH/3abx3ad_101212')[0])
    panel.trials.setValue(2)
    panel.output.setText(str(output / 'gui-pipeline-smoke'))
    panel.start()
    deadline = time.monotonic() + 120
    while panel.job is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert not failures, failures
    assert panel.job is None and panel.report['status'] == 'complete', panel.report
    assert len(panel.history) == 2 and panel.final_scores['reference_count'] == 10
    assert panel.results.rowCount() == 6
    for tab, name in [(0, 'setup'), (1, 'scores'), (2, 'preprocessing'), (3, 'results')]:
        panel.tabs.setCurrentIndex(tab)
        app.processEvents()
        panel.grab().save(str(output / f'pipeline-{name}.png'))
    row = next(i for i, row in enumerate(panel.report['files']) if row['split'] == 'evaluation')
    panel.results.selectRow(row)
    panel.open_overlay()
    deadline = time.monotonic() + 30
    while window._pending_load and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert not window._pending_load
    assert any(s.source == 'prediction' for s in window.dataset.scene.surfaces)
    window.pipeline_dock.hide()
    app.processEvents()
    window.fit_view()
    window.slice_view.viewport().grab().save(str(output / 'pipeline-overlay-2d.png'))
    print('PASS: background GUI preprocessing, tuning graphs, reserved evaluation, results and 2D prediction overlays.', flush=True)


def exercise_neural(path):
    window.show_neural()
    panel=window.neural_panel
    failures=[];panel.show_error=failures.append
    report=json.loads(path.read_text())
    panel.display_report(report)
    if args.neural_train:
        panel.mode.setCurrentIndex(0)
        panel.set_entries(scan_library('FISH/3abx3ad_101212')[0])
        panel.set_config(replace(panel.config(),epochs=1,steps_per_epoch=1,calibrate_probability=False))
        panel.output.setText(str(output/'neural-gui-training-smoke'))
        received=[]
        handle_event=panel.handle_event
        def record_event(event):
            handle_event(event)
            if event['kind'] in ('batch','epoch'):
                received.append(event['kind'])
                assert panel.job is not None
            if event['kind']=='epoch':
                panel.grab().save(str(output/'neural-live-training.png'))
        panel.handle_event=record_event
        panel.start()
        deadline=time.monotonic()+180
        while panel.job is not None and time.monotonic()<deadline:
            app.processEvents();time.sleep(.01)
        assert not failures,failures
        assert panel.job is None and panel.report['status']=='complete',panel.report
        assert 'batch' in received and 'epoch' in received
        assert len(panel.history)==1 and panel.results.rowCount()==6
        panel.handle_event=handle_event
        panel.display_report(report)
        print('PASS: live neural training subprocess, batch/epoch events, graphs, checkpoint and final evaluation.',flush=True)
    assert len(panel.history)==len(report['epochs']) and panel.history
    for tab,name in [(0,'setup'),(1,'scores'),(3,'results')]:
        panel.tabs.setCurrentIndex(tab);app.processEvents()
        panel.grab().save(str(output/f'neural-{name}.png'))
    panel.current_image()
    panel.output.setText(str(output/'neural-gui-smoke'))
    panel.start()
    deadline=time.monotonic()+120
    while panel.job is not None and time.monotonic()<deadline:
        app.processEvents();time.sleep(.01)
    assert not failures,failures
    assert panel.job is None and panel.report['status']=='complete',panel.report
    assert panel.results.rowCount()==1
    panel.tabs.setCurrentIndex(2);app.processEvents()
    panel.grab().save(str(output/'neural-preprocessing.png'))
    panel.results.selectRow(0);panel.open_overlay()
    deadline=time.monotonic()+30
    while window._pending_load and time.monotonic()<deadline:
        app.processEvents();time.sleep(.01)
    assert not window._pending_load
    assert any(s.source=='prediction' for s in window.dataset.scene.surfaces)
    window.neural_dock.hide();app.processEvents();window.fit_view()
    window.slice_view.viewport().grab().save(str(output/'neural-overlay-2d.png'))
    print('PASS: neural report/graphs, saved checkpoint application in a subprocess, and 2D prediction overlays.',flush=True)


def exercise():
    try:
        entry = entry_for_path('FISH/3abx3ad_101212/141212_3ab3ad_Lam633m_dsRed543_CG488b.ims')
        dataset = load_dataset(entry)
        window.add_entries(scan_library('FISH/3abx3ad_101212')[0])
        window.entry = entry
        window.set_dataset(dataset)
        app.processEvents()
        window.fit_view()
        app.processEvents()
        window.grab().save(str(output / 'viewer-2d.png'))
        window.slice_view.viewport().grab().save(str(output / 'slice-2d.png'))
        assert window.slice_view._overlays
        window.orientation.setCurrentIndex(1)
        window.orientation.setCurrentIndex(2)
        window.orientation.setCurrentIndex(0)
        if args.pipeline:
            exercise_pipeline()
        if args.neural_report:
            exercise_neural(args.neural_report)
        window.switch_mode(True)
        app.processEvents()
        plotter = window.volume_view.plotter
        image = plotter.screenshot(str(output / 'volume-3d.png'))
        # Capture 3D through VTK above: Qt's backing-store grabs cannot reliably
        # capture native OpenGL child windows, and Wayland can block screen grabs.
        assert image.std() > 15, '3D render appears blank'
        if args.pipeline or args.neural_report:
            plotter.screenshot(str(output / ('neural-overlay-3d.png' if args.neural_report else 'pipeline-overlay-3d.png')))
            window.set_surface_source_visible('prediction', False)
            assert all(not s.visible for s in window.dataset.scene.surfaces if s.source == 'prediction')
            manual = plotter.screenshot()
            assert np.mean(np.abs(image.astype(float) - manual)) > 0.05
            window.set_surface_source_visible('prediction', True)
            window.set_surface_source_visible('manual', False)
            assert all(s.visible for s in window.dataset.scene.surfaces if s.source == 'prediction')
            assert all(not s.visible for s in window.dataset.scene.surfaces if s.source == 'manual')
            window.set_surface_source_visible('manual', True)
            print('PASS: predicted 3D objects and independent manual/predicted visibility.', flush=True)
        for panel in window.channel_panels:
            panel.visible.setChecked(False)
        window.refresh_timer.stop()
        window.refresh_images()
        without_channels = plotter.screenshot(str(output / 'surfaces-3d.png'))
        assert np.mean(np.abs(image.astype(float) - without_channels)) > 1
        assert np.max(window.volume_view._grid['rgba']) == 0
        for panel in window.channel_panels:
            panel.visible.setChecked(True)
        window.refresh_timer.stop()
        window.full_detail.setChecked(True)
        window.refresh_timer.stop()
        window.refresh_images()
        assert tuple(window.volume_view._grid.dimensions) == dataset.volumes[0].shape[::-1]
        window.full_detail.setChecked(False)
        window.refresh_timer.stop()
        window.refresh_images()
        window.show_surfaces.setChecked(False)
        assert all(not actor.GetVisibility() for actor in window.volume_view._mesh_actors)
        window.show_surfaces.setChecked(True)
        window.wireframe.setChecked(True)
        window.wireframe.setChecked(False)
        measurement_item = window.objects.topLevelItem(1).child(0)
        window.objects.setCurrentItem(measurement_item)
        window.focus_object()
        assert 'distance' in window.object_info.text().lower()
        state = window.view_state()
        window.channel_panels[0].brightness.setValue(40)
        window.apply_state(state)
        assert window.dataset.metadata.channels[0].brightness == 0
        window.switch_mode(False)
        window.focus_object()
        app.processEvents()
        window.grab().save(str(output / 'measurement-2d.png'))
        print('PASS: actual 2D/3D renderer, RGB channels, resolution, visibility, selection, and saved settings.', flush=True)
        print('Render backend:', plotter.render_window.GetClassName(), flush=True)
        window.close()
        app.exit(0)
    except Exception:
        import traceback
        traceback.print_exc()
        window.close()
        app.exit(1)


QtCore.QTimer.singleShot(500, exercise)
sys.exit(app.exec())
