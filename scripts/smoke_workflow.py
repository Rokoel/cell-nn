"""Exercise both trained models, filtering, Imaris export and standalone 2D/3D reopening."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['QT_API'] = 'pyside6'
if sys.platform.startswith('linux'):
    os.environ['QT_QPA_PLATFORM'] = 'xcb'

import numpy as np
from PySide6 import QtCore, QtWidgets
from fish_viewer.app import MainWindow
from fish_viewer.reader import entry_for_path
from fish_viewer.signals.data import at_labels

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--image', type=Path, default=Path('FISH/S2_dsHDAC3_22A/050512_dsHDAC3_Lam633m_22A546b.ims'))
args = parser.parse_args()
output = Path('artifacts/workflow-smoke') / datetime.now().strftime('%Y%m%d-%H%M%S')
output.mkdir(parents=True)
app = QtWidgets.QApplication([])
app.setStyle('Fusion')
window = MainWindow()
window.resize(1100, 700)
window.show()
errors = []
window.error = errors.append


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def wait(predicate, seconds=600):
    deadline = time.monotonic() + seconds
    last = None
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        assert not errors, errors
        panel = window.workflow_panel
        if panel:
            text = panel.stage_label.text() + ': ' + panel.phase.text()
            if text != last:
                print(text, flush=True)
                last = text
        time.sleep(.02)
    assert predicate(), 'Timed out waiting for worker'
    assert not errors, errors


def exercise():
    standalone = None
    try:
        source_hash = digest(args.image)
        window.load_entry(entry_for_path(args.image))
        wait(lambda: not window._pending_load)
        window.show_workflow()
        panel = window.workflow_panel
        panel.error = errors.append
        panel.output.setText(str(output / 'runs'))
        panel.start()
        wait(lambda: not panel.busy and not window._pending_load)
        assert panel.result_path is not None
        report = json.loads(panel.result_path.read_text())
        assert report['points'] and report['signal_filter']['removed_cell_count'] > 0
        assert len(window.dataset.scene.surfaces) == report['signal_filter']['kept_cell_count']
        assert not window.dataset.scene.contours
        assert all(obj.source == 'prediction' for obj in window.dataset.scene.surfaces)
        with np.load(report['cell_predictions']) as archive:
            metadata = json.loads(str(archive['metadata']))
            assigned = at_labels(archive['labels'], [p['xyz'] for p in report['points']], metadata['origin_xyz'], metadata['spacing_zyx'])
        np.testing.assert_array_equal(assigned, [p['cell_id'] for p in report['points']])
        assert sum(row['count'] for row in report['counts']) == len(report['points'])
        panel.grab().save(str(output / 'workflow-results.png'))
        target = output / 'export' / 'cells-with-signals.ims'
        panel.export_current(target=target)
        wait(lambda: not panel.busy and not window._pending_load)
        assert panel.exported is not None
        assert window.entry.image == target.resolve()
        assert digest(target) == source_hash == digest(args.image)
        window.close()
        standalone = MainWindow()
        standalone.error = errors.append
        standalone.resize(1100, 700)
        standalone.show()
        standalone.load_entry(entry_for_path(target))
        wait(lambda: not standalone._pending_load)
        assert not standalone._pipeline_archives and not standalone._signal_archives
        assert len(standalone.dataset.scene.surfaces) == report['signal_filter']['kept_cell_count']
        assert sum(len(group.points) for group in standalone.dataset.scene.measurements) == len(report['points'])
        for panel_control in standalone.channel_panels:
            panel_control.auto()
        group = standalone.objects.topLevelItem(1)
        standalone.objects.setCurrentItem(group.child(0))
        standalone.focus_object()
        app.processEvents()
        standalone.slice_view.viewport().grab().save(str(output / 'reopened-2d.png'))
        standalone.switch_mode(True)
        app.processEvents()
        standalone.volume_view.plotter.screenshot(str(output / 'reopened-3d.png'))
        assert standalone.volume_view._annotation_actors
        summary = dict(status='passed', report=str(panel.result_path), export=panel.exported,
                       source_sha256=source_hash, filter=report['signal_filter'], signals=len(report['points']))
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        standalone.close()
        app.exit(0)
    except Exception:
        import traceback
        traceback.print_exc()
        if standalone:standalone.close()
        window.close()
        app.exit(1)


QtCore.QTimer.singleShot(100, exercise)
raise SystemExit(app.exec())
