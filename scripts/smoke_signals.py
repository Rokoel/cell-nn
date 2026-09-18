"""Exercise the trained signal model through the real desktop viewer."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['QT_API']='pyside6'
if sys.platform.startswith('linux'):os.environ['QT_QPA_PLATFORM']='xcb'

import numpy as np
from PySide6 import QtCore,QtWidgets
from fish_viewer.app import MainWindow
from fish_viewer.signals.data import at_labels

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--report',type=Path,required=True)
args=parser.parse_args()
trained=json.loads(args.report.read_text())
audit=json.loads(Path(trained['audit_path']).read_text())
test_rows=sorted(trained['files'],key=lambda r:r['count'],reverse=True)
case=next(c for c in audit['cases'] if c['image']==test_rows[0]['image'])
app=QtWidgets.QApplication([]);app.setStyle('Fusion');window=MainWindow();window.resize(1100,700);window.show()
output=Path('artifacts/signal-fish/gui-smoke');output.mkdir(parents=True,exist_ok=True)


def wait_until(predicate,seconds=120):
    deadline=time.monotonic()+seconds
    while not predicate() and time.monotonic()<deadline:app.processEvents();time.sleep(.01)
    assert predicate(),'Timed out waiting for the GUI worker'


def exercise():
    try:
        window.open_pipeline_prediction(dict(image=case['image'],scene=case['scene'],predictions=case['cell_predictions']))
        wait_until(lambda:not window._pending_load)
        window.show_signals();panel=window.signal_panel
        panel.display_report(trained);panel.tabs.setCurrentIndex(0);app.processEvents()
        panel.grab().save(str(output/'training-panel.png'))
        panel.mode.setCurrentIndex(0);panel.output.setText(str(output/'runs'))
        errors=[];panel.error=errors.append;panel.start();wait_until(lambda:panel.job is None)
        assert not errors,errors
        report=panel.report
        assert report['status']=='complete',report.get('error')
        assert len(report['points'])>0
        wait_until(lambda:not window._pending_load)
        assert any(m.source=='signal_prediction' for m in window.dataset.scene.measurements)
        with np.load(report['cell_predictions']) as archive:
            labels=archive['labels'];metadata=json.loads(str(archive['metadata']))
        assignments=at_labels(labels,[p['xyz'] for p in report['points']],metadata['origin_xyz'],metadata['spacing_zyx'])
        assert np.array_equal(assignments,[p['cell_id'] for p in report['points']])
        assert sum(row['count'] for row in report['counts'])==len(report['points'])
        window.show_signals();panel.tabs.setCurrentIndex(1);app.processEvents()
        panel.grab().save(str(output/'results-panel.png'));window.signal_dock.hide()
        group=window.objects.topLevelItem(1)
        index=next(i for i,m in enumerate(window.dataset.scene.measurements) if m.source=='signal_prediction')
        window.objects.setCurrentItem(group.child(index));window.focus_object();app.processEvents()
        window.slice_view.viewport().grab().save(str(output/'signals-2d.png'))
        window.switch_mode(True);app.processEvents();plotter=window.volume_view.plotter
        plotter.screenshot(str(output/'signals-3d.png'))
        before=len(window.volume_view._annotation_actors)
        window.show_predicted_signals.setChecked(False);app.processEvents()
        assert len(window.volume_view._annotation_actors)<before
        window.show_predicted_signals.setChecked(True);app.processEvents()
        print('PASS: saved signal model, live worker, per-cell/channel counts, calibrated inside-cell centers, 2D/3D overlays and visibility.',flush=True)
        print('Signals:',len(report['points']),'Report:',Path(report['directory'])/'report.json',flush=True)
        window.close();app.exit(0)
    except Exception:
        import traceback
        traceback.print_exc();window.close();app.exit(1)


QtCore.QTimer.singleShot(100,exercise)
raise SystemExit(app.exec())
