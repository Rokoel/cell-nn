"""Exercise the real desktop renderer and save review screenshots, then exit."""
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['QT_API'] = 'pyside6'
if sys.platform.startswith('linux'):
    os.environ['QT_QPA_PLATFORM'] = 'xcb'

import numpy as np
from PySide6 import QtCore, QtWidgets
from fish_viewer.app import MainWindow
from fish_viewer.reader import entry_for_path, load_dataset, scan_library


app = QtWidgets.QApplication([])
print('Qt platform:', app.platformName(), flush=True)
app.setStyle('Fusion')
window = MainWindow()
window.show()
output = Path('artifacts')
output.mkdir(exist_ok=True)


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
        window.switch_mode(True)
        app.processEvents()
        plotter = window.volume_view.plotter
        image = plotter.screenshot(str(output / 'volume-3d.png'))
        # Capture 3D through VTK above: Qt's backing-store grabs cannot reliably
        # capture native OpenGL child windows, and Wayland can block screen grabs.
        assert image.std() > 15, '3D render appears blank'
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
