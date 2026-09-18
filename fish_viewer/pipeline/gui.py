from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import threading
import time
import traceback

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets

from ..reader import entry_for_path, scan_library
from .config import PipelineConfig
from .runner import run_pipeline


class Signals(QtCore.QObject):
    event = QtCore.Signal(object)
    done = QtCore.Signal(object)
    error = QtCore.Signal(str)


class PipelineJob(QtCore.QRunnable):
    def __init__(self, entries, config, output, mode, trials, holdout):
        super().__init__()
        self.signals = Signals()
        self.cancel = threading.Event()
        self.arguments = entries, config, output, mode, trials, holdout

    def run(self):
        try:
            report = run_pipeline(*self.arguments, emit=self.signals.event.emit, cancel=self.cancel)
            self.signals.done.emit(report)
        except Exception:
            self.signals.error.emit(traceback.format_exc())


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, rows=2, columns=2):
        self.figure = Figure(figsize=(11, 4), layout='constrained', facecolor='#111821')
        self.axes = self.figure.subplots(rows, columns).ravel()
        super().__init__(self.figure)
        self.setMinimumSize(620, 360)
        self.style()

    def style(self):
        for ax in self.axes:
            ax.set_facecolor('#17232f')
            ax.tick_params(colors='#a9becf', labelsize=8)
            for spine in ax.spines.values():
                spine.set_color('#405366')
            ax.xaxis.label.set_color('#bbcedd')
            ax.yaxis.label.set_color('#bbcedd')
            ax.title.set_color('#e1eaf4')
            ax.title.set_fontsize(10)


class PipelinePanel(QtWidgets.QWidget):
    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self.entries = []
        self.report = None
        self.job = None
        self.history = []
        self.final_scores = None
        self.previews = {}
        self.channel_scores = []
        self.timings = {}
        self.current_file = ''
        self._shutdown = False
        self.started = 0
        layout = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(['Tune parameters + evaluate reserved images', 'Apply current settings to selected images'])
        self.trials = QtWidgets.QSpinBox()
        self.trials.setRange(1, 200)
        self.trials.setValue(12)
        self.run_button = QtWidgets.QPushButton('Run pipeline')
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.cancel_button.setEnabled(False)
        self.run_button.clicked.connect(self.start)
        self.cancel_button.clicked.connect(self.cancel)
        top.addWidget(self.mode, 1)
        top.addWidget(QtWidgets.QLabel('Trials'))
        top.addWidget(self.trials)
        top.addWidget(self.run_button)
        top.addWidget(self.cancel_button)
        layout.addLayout(top)
        self.open_report_button=QtWidgets.QPushButton('Open saved run…')
        self.open_report_button.clicked.connect(self.load_report)
        report_row=QtWidgets.QHBoxLayout();report_row.addWidget(self.open_report_button);report_row.addStretch()
        layout.addLayout(report_row)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)
        self._setup_page()
        self.score_canvas = PlotCanvas()
        self.tabs.addTab(self.score_canvas, 'Live scores')
        self.preprocess_canvas = PlotCanvas(2, 3)
        self.tabs.addTab(self.preprocess_canvas, 'Preprocessing')
        self._results_page()
        progress_row = QtWidgets.QHBoxLayout()
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setMaximumWidth(220)
        self.status = QtWidgets.QLabel('Ready. Classical parameter search; no neural network is fitted.')
        self.status.setWordWrap(True)
        self.elapsed = QtWidgets.QLabel('')
        progress_row.addWidget(self.status, 1)
        progress_row.addWidget(self.progress)
        progress_row.addWidget(self.elapsed)
        layout.addLayout(progress_row)
        note = QtWidgets.QLabel('Processes time point 1. Partial references: unknown cells are unscored. Misses count as zero overlap. '
            'Boundary error uses matched cells only. Reserved-image scores appear after tuning finishes.')
        note.setWordWrap(True)
        note.setObjectName('muted')
        layout.addWidget(note)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(lambda: self.elapsed.setText(f'{int(time.monotonic()-self.started)} s'))
        self.draw_scores()
        self.draw_preprocessing()
        reference = Path(__file__).resolve().parents[2] / 'FISH/3abx3ad_101212'
        if reference.is_dir():
            self.set_entries(scan_library(reference)[0])

    def _setup_page(self):
        page = QtWidgets.QWidget()
        self.setup_page = page
        layout = QtWidgets.QHBoxLayout(page)
        files_layout = QtWidgets.QVBoxLayout()
        buttons = QtWidgets.QHBoxLayout()
        for text, callback in [('Add files…', self.add_files), ('Add folder…', self.add_folder),
                               ('Current image', self.current_image), ('Viewer library', self.viewer_library)]:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        files_layout.addLayout(buttons)
        self.files = QtWidgets.QListWidget()
        self.files.setMinimumWidth(290)
        files_layout.addWidget(self.files, 1)
        actions = QtWidgets.QHBoxLayout()
        for text, checked in [('Select all', True), ('Select none', False)]:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(lambda _, checked=checked: self.check_all(checked))
            actions.addWidget(button)
        self.file_count = QtWidgets.QLabel('0 images')
        actions.addWidget(self.file_count)
        files_layout.addLayout(actions)
        layout.addLayout(files_layout, 1)
        settings_scroll = QtWidgets.QScrollArea()
        settings_scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(body)
        self.controls = {}
        settings = [('voxel_um', 'Processing voxel (µm)', 0.05, 5, 0.05),
                    ('min_diameter_um', 'Minimum cell diameter (µm)', 0.1, 499, 0.5),
                    ('max_diameter_um', 'Maximum cell diameter (µm)', 0.2, 500, 0.5),
                    ('max_elongation', 'Maximum 3D axis ratio', 1, 5, 0.1),
                    ('min_sphericity', 'Minimum sphericity', 0, 1, 0.05),
                    ('threshold', 'Boundary threshold', 0.01, 0.9, 0.01),
                    ('smooth_um', 'Smoothing (µm)', 0, 5, 0.05),
                    ('seed_distance_um', 'Seed separation (µm)', 0.2, 100, 0.1),
                    ('closing_um', 'Close boundary gaps (µm)', 0, 5, 0.1)]
        defaults = asdict(PipelineConfig())
        for key, title, low, high, step in settings:
            control = QtWidgets.QDoubleSpinBox()
            control.setRange(low, high)
            control.setSingleStep(step)
            control.setDecimals(6)
            control.setValue(defaults[key])
            control.setKeyboardTracking(False)
            self.controls[key] = control
            form.addRow(title, control)
        self.method = QtWidgets.QComboBox()
        self.method.addItem('Boundary watershed', 'boundary_watershed')
        self.method.addItem('Filled-interior distance watershed', 'distance_watershed')
        form.addRow('Classical method', self.method)
        self.exclude_border = QtWidgets.QCheckBox('Exclude cells cut by image edges')
        self.exclude_border.setChecked(True)
        form.addRow(self.exclude_border)
        self.holdout = QtWidgets.QSpinBox()
        self.holdout.setRange(10, 50)
        self.holdout.setValue(20)
        self.holdout.setSuffix('% of acquisition groups')
        form.addRow('Reserve for evaluation', self.holdout)
        self.output = QtWidgets.QLineEdit(str(Path.cwd() / 'pipeline_runs'))
        output_row = QtWidgets.QHBoxLayout()
        output_row.addWidget(self.output)
        choose = QtWidgets.QPushButton('…')
        choose.clicked.connect(self.choose_output)
        output_row.addWidget(choose)
        form.addRow('Output folder', output_row)
        settings_row = QtWidgets.QHBoxLayout()
        for text, callback in [('Load settings…', self.load_settings), ('Save settings…', self.save_settings), ('Defaults', lambda: self.set_config(PipelineConfig()))]:
            button = QtWidgets.QPushButton(text)
            button.clicked.connect(callback)
            settings_row.addWidget(button)
        form.addRow(settings_row)
        settings_scroll.setWidget(body)
        layout.addWidget(settings_scroll, 1)
        self.tabs.addTab(page, 'Setup')

    def _results_page(self):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.summary = QtWidgets.QLabel('No completed run yet.')
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary)
        self.results_open_report=QtWidgets.QPushButton('Open saved run…')
        self.results_open_report.clicked.connect(self.load_report)
        layout.addWidget(self.results_open_report)
        self.results = QtWidgets.QTableWidget(0, 8)
        self.results.setHorizontalHeaderLabels(['Image', 'Split', 'Channel', 'Cells', 'Dice', 'IoU', 'Detected refs', 'Boundary RMSE µm'])
        self.results.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.results.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.results.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.results.cellDoubleClicked.connect(lambda *_: self.open_overlay())
        layout.addWidget(self.results, 1)
        row = QtWidgets.QHBoxLayout()
        self.overlay_button = QtWidgets.QPushButton('Show selected contours in viewer')
        self.overlay_button.clicked.connect(self.open_overlay)
        row.addWidget(self.overlay_button)
        self.manual_visible = QtWidgets.QCheckBox('Manual contours')
        self.predicted_visible = QtWidgets.QCheckBox('Predicted contours')
        for check in (self.manual_visible, self.predicted_visible):
            check.setChecked(True)
        self.manual_visible.toggled.connect(lambda checked: self.host.set_surface_source_visible('manual', checked))
        self.predicted_visible.toggled.connect(lambda checked: self.host.set_surface_source_visible('prediction', checked))
        row.addWidget(self.manual_visible)
        row.addWidget(self.predicted_visible)
        row.addWidget(QtWidgets.QLabel('Predictions: green'))
        layout.addLayout(row)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(90)
        layout.addWidget(self.log)
        self.tabs.addTab(page, 'Results')

    def config(self):
        return PipelineConfig(**{key:control.value() for key,control in self.controls.items()},
            method=self.method.currentData(), exclude_border=self.exclude_border.isChecked()).validate()

    def set_config(self, config):
        for key, control in self.controls.items():
            control.setValue(getattr(config, key))
        self.method.setCurrentIndex(self.method.findData(config.method))
        self.exclude_border.setChecked(config.exclude_border)

    def set_entries(self, entries):
        by_image = {}
        for entry in entries:
            if entry.image not in by_image or entry.scene:
                by_image[entry.image] = entry
        self.entries = sorted(by_image.values(), key=lambda e:str(e.image))
        self.files.clear()
        for entry in self.entries:
            item = QtWidgets.QListWidgetItem(entry.image.name)
            item.setToolTip(str(entry.image))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, entry)
            item.setCheckState(QtCore.Qt.CheckState.Checked if entry.image.exists() else QtCore.Qt.CheckState.Unchecked)
            self.files.addItem(item)
        self.file_count.setText(f'{len(self.entries)} images')

    def checked_entries(self):
        return [self.files.item(i).data(QtCore.Qt.ItemDataRole.UserRole) for i in range(self.files.count())
            if self.files.item(i).checkState() == QtCore.Qt.CheckState.Checked]

    def check_all(self, checked):
        for i in range(self.files.count()):
            self.files.item(i).setCheckState(QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)

    def add_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, 'Add Imaris datasets', '', 'Imaris (*.ims *.imx *.IMS *.IMX)')
        try:
            self.set_entries(self.entries + [entry_for_path(p) for p in paths])
        except Exception as exc:
            self.show_error(str(exc))

    def add_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Add folder recursively')
        if path:
            def done(result):
                entries, warnings = result
                self.set_entries(self.entries + entries)
                for warning in warnings:
                    self.log.appendPlainText(warning)
            self.host.run_job(lambda _:scan_library(path), done)

    def current_image(self):
        if self.host.entry:
            self.set_entries([self.host.entry])
            self.mode.setCurrentIndex(1)
        else:
            self.show_error('Open an image in the viewer first.')

    def viewer_library(self):
        self.set_entries(self.host.entries)

    def choose_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose output folder', self.output.text())
        if path:
            self.output.setText(path)

    def load_settings(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load pipeline settings', '', 'JSON (*.json)')
        if path:
            try:
                values = json.loads(Path(path).read_text(encoding='utf-8'))
                if values.get('version') != 1:
                    raise ValueError('Unsupported settings version.')
                self.set_config(PipelineConfig.from_dict(values['config']))
            except Exception as exc:
                self.show_error(str(exc))

    def save_settings(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save pipeline settings', 'contour-settings.json', 'JSON (*.json)')
        if path:
            try:
                Path(path).write_text(json.dumps(dict(version=1, config=asdict(self.config())), indent=2), encoding='utf-8')
            except Exception as exc:
                self.show_error(str(exc))

    def show_error(self, text):
        QtWidgets.QMessageBox.warning(self, 'Contour pipeline', text)

    def start(self):
        if self.job:
            return
        try:
            config = self.config()
            entries = self.checked_entries()
            if not entries:
                raise ValueError('Select at least one image.')
            if not self.output.text().strip():
                raise ValueError('Choose an output folder.')
            mode = 'tune' if self.mode.currentIndex() == 0 else 'apply'
            if mode == 'tune' and len(entries) < 2:
                raise ValueError('Tuning needs multiple acquisition groups. Use Apply settings for one image.')
        except Exception as exc:
            self.show_error(str(exc))
            return
        self.history = []
        self.final_scores = None
        self.previews = {}
        self.channel_scores = []
        self.timings = {}
        self.report = None
        self.results.setRowCount(0)
        self.log.clear()
        self.summary.setText('Run in progress. Final evaluation has not been performed.')
        self.draw_scores()
        self.draw_preprocessing()
        self.job = PipelineJob(entries, config, self.output.text(), mode, self.trials.value(), self.holdout.value()/100)
        self.job.signals.event.connect(self.handle_event)
        self.job.signals.done.connect(self.completed)
        self.job.signals.error.connect(self.failed)
        self.set_running(True)
        self.started = time.monotonic()
        self.timer.start()
        self.tabs.setCurrentIndex(1 if mode == 'tune' else 2)
        self.host.thread_pool.start(self.job)

    def set_running(self, running):
        self.setup_page.setEnabled(not running)
        self.run_button.setEnabled(not running)
        self.mode.setEnabled(not running)
        self.trials.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    def cancel(self):
        if self.job:
            self.job.cancel.set()
            self.cancel_button.setEnabled(False)
            self.status.setText('Cancellation requested; finishing the current operation…')

    @QtCore.Slot(object)
    def handle_event(self, event):
        if self._shutdown:
            return
        kind = event['kind']
        if kind == 'file':
            self.current_file = Path(event['file']).name
            self.previews = event.get('previews', {})
            self.channel_scores = event.get('channel_scores', [])
            self.timings = event.get('timings', {})
            self.draw_preprocessing()
            self.phase = event['phase']
            self.status.setText(f'{self.phase} · {event["current"]}/{event["total"]} · {self.current_file}')
            self.progress.setValue(0)
        elif kind == 'stage':
            self.progress.setValue(int(event.get('progress', 0)))
            self.status.setText(f'{self.current_file} · {event["stage"]}')
            redraw = False
            if 'channel_scores' in event:
                self.channel_scores = event['channel_scores']
                redraw = True
            if 'previews' in event:
                self.previews.update(event['previews'])
                redraw = True
            if 'timings' in event:
                self.timings = event['timings']
                redraw = True
            if redraw:
                self.draw_preprocessing()
        elif kind == 'trial':
            self.history.append(event)
            self.draw_scores()
        elif kind == 'evaluation':
            self.final_scores = event['scores']
            self.draw_scores()
        elif kind == 'result':
            self.add_result(event['row'])
        elif kind == 'split':
            self.log.appendPlainText('Tuning images:\n' + '\n'.join(event['split']['tuning']))
            self.log.appendPlainText('Reserved images, scored after tuning:\n' + '\n'.join(event['split']['evaluation']))
        elif kind == 'warning':
            self.log.appendPlainText(event['message'])

    def draw_scores(self):
        axes = self.score_canvas.axes
        x = [row['trial'] for row in self.history]
        specs = [('objective', 'Tuning loss: 1 − mean Dice'), ('mean_dice', 'Dice and IoU overlap'),
                 ('boundary_rmse_um', 'Matched-cell boundary RMSE (µm)'), ('detection_rate', 'Annotated cells detected (IoU ≥ 0.5)')]
        for ax, (key, title) in zip(axes, specs):
            ax.clear()
            ax.set_title(title)
            ax.set_xlabel('Parameter trial')
            if x:
                y = [row['scores'].get(key) for row in self.history]
                ax.plot(x, [np.nan if v is None else v for v in y], 'o-', color='#56dcc1', markersize=3, label='Tuning')
                if key == 'objective':
                    ax.plot(x, [row['best_objective'] for row in self.history], color='#84bafa', label='Best so far')
                elif key == 'mean_dice':
                    ax.plot(x, [row['scores']['mean_iou'] for row in self.history], color='#84bafa', label='IoU')
            if self.final_scores and self.final_scores.get(key) is not None:
                ax.scatter([max(x, default=0)+1], [self.final_scores[key]], color='#edb867', marker='*', s=90, label='Reserved: final only')
            if not x:
                ax.text(0.5, 0.5, 'Waiting for completed parameter trials', ha='center', va='center', transform=ax.transAxes, color='#a9becf', fontsize=9)
            if key != 'boundary_rmse_um':
                ax.set_ylim(-0.02, 1.02)
            if x or self.final_scores:
                ax.legend(fontsize=7, facecolor='#17232f', labelcolor='#e0e8f1', edgecolor='#405366')
            ax.grid(alpha=0.15)
        self.score_canvas.style()
        self.score_canvas.draw_idle()

    def draw_preprocessing(self):
        axes = self.preprocess_canvas.axes
        for ax, key in zip(axes[:4], ['Raw selected channel', 'Background corrected', 'Candidate interiors', 'Accepted spherical cells']):
            ax.clear()
            ax.set_title(key)
            if key in self.previews:
                data = self.previews[key]
                ax.imshow(data, cmap='nipy_spectral' if 'cells' in key else 'gray', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
        ax = axes[4]
        ax.clear()
        ax.set_title('Automatic channel: ring evidence')
        if self.channel_scores:
            names = [f'{r["channel"]+1}: {r["name"]}' for r in self.channel_scores]
            ax.bar(names, [r['score'] for r in self.channel_scores], color='#56dcc1')
            ax.tick_params(axis='x', labelsize=7)
        ax = axes[5]
        ax.clear()
        ax.set_title('Latest segmentation stage time (s)')
        rows = [(k.replace('_', ' '), v) for k,v in self.timings.items() if k != 'segmentation']
        if rows:
            ax.barh([k for k,v in rows], [v for k,v in rows], color='#84bafa')
        self.preprocess_canvas.style()
        self.preprocess_canvas.draw_idle()

    def add_result(self, row):
        index = self.results.rowCount()
        self.results.insertRow(index)
        m = row['metrics']
        def number(value):
            return '—' if value is None else f'{value:.3f}'
        values = [Path(row['image']).name, row['split'], str(row['channel']+1), str(m['predicted_count']),
            number(m['mean_dice']), number(m['mean_iou']), number(m['detection_rate']), number(m['boundary_rmse_um'])]
        for column, value in enumerate(values):
            item = QtWidgets.QTableWidgetItem(value)
            item.setToolTip(row['image'])
            if column == 0:
                item.setData(QtCore.Qt.ItemDataRole.UserRole, row)
            self.results.setItem(index, column, item)

    @QtCore.Slot(object)
    def completed(self, report):
        if self._shutdown:
            return
        self.timer.stop()
        self.job = None
        self.set_running(False)
        self.report = report
        self.final_scores = report.get('evaluation')
        if report.get('best_config'):
            self.set_config(PipelineConfig.from_dict(report['best_config']))
        self.status.setText(f'{report["status"].replace("_", " ").capitalize()} · {report["elapsed_s"]:.1f} s')
        self.elapsed.setText(f'{report["elapsed_s"]:.1f} s')
        if report['status'].startswith('complete'):
            self.progress.setValue(100)
        evaluation = report.get('evaluation')
        extra = ''
        if evaluation and evaluation.get('mean_dice') is not None:
            extra = f'\nReserved images: Dice {evaluation["mean_dice"]:.3f}, IoU {evaluation["mean_iou"]:.3f}, detected {evaluation["detection_rate"]:.1%} of annotated cells.'
        self.summary.setText(f'{report["status"]} · {len(report["files"])} output datasets\nSaved to {report["directory"]}{extra}')
        for warning in report.get('warnings', []):
            self.log.appendPlainText(warning)
        for failure in report.get('failures', []):
            self.log.appendPlainText(str(failure))
        if report.get('error'):
            self.log.appendPlainText(report['error'])
            self.summary.setText(self.summary.text() + '\n' + report['error'])
        self.draw_scores()
        self.tabs.setCurrentIndex(3)
        if self.results.rowCount():
            self.results.selectRow(0)

    def failed(self, error):
        self.timer.stop()
        self.job = None
        self.set_running(False)
        self.status.setText('Run failed.')
        self.log.appendPlainText(error)
        self.show_error(error.splitlines()[-1])

    def open_overlay(self):
        index = self.results.currentRow()
        if index >= 0:
            row = self.results.item(index, 0).data(QtCore.Qt.ItemDataRole.UserRole)
            self.host.open_pipeline_prediction(row)

    def load_report(self):
        if self.job:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open saved pipeline run', self.output.text(), 'JSON report (*.json)')
        if path:
            try:
                report = json.loads(Path(path).read_text(encoding='utf-8'))
                if report.get('version') != 1 or 'files' not in report or 'trials' not in report:
                    raise ValueError('Not a supported pipeline run report.')
                self.results.setRowCount(0)
                self.log.clear()
                self.history = report['trials']
                for row in report['files']:
                    self.add_result(row)
                self.completed(report)
            except Exception as exc:
                self.show_error(str(exc))

    def shutdown(self):
        self._shutdown = True
        self.timer.stop()
        if self.job:
            self.job.cancel.set()
