"""A focused scan → cells → signals → A/B distances → Imaris desktop app."""
import json
from pathlib import Path
import re
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..app import ChannelPanel, Job, STYLE
from ..models import LibraryEntry, Scene
from ..reader import entry_for_path, load_dataset, scan_library
from ..signals.overlays import read_signals
from ..views import SliceView, VolumeView
from .controller import AnalysisController, default_models, default_refinement_model, default_proposal_model, migrate_bundled_defaults, sensitive_stardist_model, STAGES


class AnalysisWindow(QtWidgets.QMainWindow):
    def __init__(self, off_screen=False, settings=None):
        super().__init__()
        self.setWindowTitle('FISH Analysis — cells, signals and distances')
        self.resize(1180, 780)
        self.setStyleSheet(STYLE)
        self.off_screen = off_screen
        self.settings = settings or QtCore.QSettings('Cell NN', 'FISH Analysis')
        try:
            self.channel_roles=json.loads(self.settings.value('channel_roles','{}'))
            if not isinstance(self.channel_roles,dict):self.channel_roles={}
        except (TypeError,ValueError):
            self.channel_roles={}
        self._source_image=None
        self.controller = AnalysisController(self)
        self.controller.progress.connect(self.processing_progress)
        self.controller.completed.connect(self.completed)
        self.controller.failed.connect(self.failed)
        self.controller.cancelled.connect(self.cancelled)
        self.dataset = None
        self.result_path = None
        self.result_report = None
        self.volume_view = None
        self.channel_dialog = None
        self.full_detail = False
        self.loading = False
        self.running = False
        self._generation = 0
        self._jobs = set()
        self.queue = []
        self.scan_number = 0
        self._build()
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tick)
        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.setInterval(100)
        self.refresh_timer.timeout.connect(self.refresh_images)
        self.sync_buttons()

    def _button(self, text, callback, layout):
        button = QtWidgets.QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def _build(self):
        splitter = QtWidgets.QSplitter()
        self.setCentralWidget(splitter)
        self.sidebar = QtWidgets.QScrollArea()
        self.sidebar.setWidgetResizable(True)
        self.sidebar.setMinimumWidth(285)
        self.sidebar.setMaximumWidth(410)
        body = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(body)
        title = QtWidgets.QLabel('FISH Analysis')
        title.setStyleSheet('font-size:22px;font-weight:600;padding:5px;')
        left.addWidget(title)
        description = QtWidgets.QLabel('Cell contours, signals and shortest 3D distances to the cell border.')
        description.setWordWrap(True)
        left.addWidget(description)
        row = QtWidgets.QHBoxLayout()
        self.add_button = self._button('Add scans…', self.open_files, row)
        self.folder_button = self._button('Add folder…', self.open_folder, row)
        left.addLayout(row)
        self.scans = QtWidgets.QListWidget()
        self.scans.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.scans.setFixedHeight(135)
        self.scans.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scans.currentItemChanged.connect(self.scan_selected)
        self.scans.setToolTip('Select one scan, or use Ctrl/Shift to select several for sequential processing.')
        left.addWidget(self.scans)
        row = QtWidgets.QHBoxLayout()
        self.remove_button = self._button('Remove selected', self.remove_selected, row)
        self.select_button = self._button('Select all', self.scans.selectAll, row)
        left.addLayout(row)
        left.addWidget(QtWidgets.QLabel('Save results in'))
        row = QtWidgets.QHBoxLayout()
        default_output = Path(QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.DocumentsLocation)) / 'FISH Analysis'
        self.output = QtWidgets.QLineEdit(self.settings.value('output', str(default_output)))
        row.addWidget(self.output, 1)
        self.output_button = self._button('Browse…', lambda: self.choose_folder(self.output), row)
        left.addLayout(row)
        self.options_button = QtWidgets.QToolButton()
        self.options_button.setText('Pipeline and settings')
        self.options_button.setCheckable(True)
        self.options_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.options_button.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        left.addWidget(self.options_button)
        self.options = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.options)
        migrate_bundled_defaults(self.settings)
        cells, signals = default_models()
        # Fixed model folders belong to presets; they are not model pickers.
        self.cell_model = QtWidgets.QLineEdit(str(cells))
        self.signal_model = QtWidgets.QLineEdit(str(signals))
        self.boundary_model = QtWidgets.QLineEdit('')
        self.boundary_model.setPlaceholderText('Optional 3D U-Net model folder')
        self.membrane_snap = QtWidgets.QCheckBox()
        self.learned_boundary = QtWidgets.QCheckBox()
        self.cell_recovery = QtWidgets.QCheckBox()
        self.stardist_proposals = QtWidgets.QCheckBox()
        self.consensus_cells = QtWidgets.QCheckBox()
        self.learned_boundary_model = Path(__file__).resolve().parents[2] / 'models/boundary-annotations'
        self.membrane_shift = QtWidgets.QDoubleSpinBox()
        self.membrane_shift.setValue(.3)
        self.pipeline = QtWidgets.QComboBox()
        self.pipeline.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.pipeline.setMinimumContentsLength(12)
        self.pipeline.setMinimumWidth(0)
        self.pipeline.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,QtWidgets.QSizePolicy.Policy.Fixed)
        for label, key in (
            ('Fine-tuned StarDist 2D (default)', 'stardist2d'),
            ('Conservative StarDist', 'conservative'),
            ('Balanced StarDist (first release cells)', 'balanced'),
            ('Preprocessed StarDist (experimental)', 'preprocessed'),
            ('Center/box + surface-loss contours', 'crop'),
            ('Sensitive StarDist + surface-loss contours', 'hybrid'),
            ('Both-detector consensus', 'consensus'),
            ('Signals first + center/box + contours (experimental)', 'signal-first'),
        ):
            self.pipeline.addItem(label, key)
        self.pipeline.setToolTip('Each pipeline uses its current bundled models and the signal-center U-Net. No training is performed.')
        form.addRow('Pipeline', self.pipeline)
        self.pipeline.currentIndexChanged.connect(self.select_pipeline)
        selected = self.pipeline.findData(self.settings.value('pipeline', 'stardist2d'))
        self.pipeline.setCurrentIndex(max(0, selected))
        self.select_pipeline()
        self.roundness = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.roundness.setRange(0, 100)
        self.roundness.setValue(int(self.settings.value('minimum_roundness_percent', 72)))
        self.roundness.setToolTip('Minimum physical 3D sphericity. Higher values reject less round cells; 0 disables this check. Accepted contours are not reshaped.')
        self.roundness_value = QtWidgets.QLabel()
        self.roundness.valueChanged.connect(lambda v: self.roundness_value.setText('Off' if v == 0 else f'{v/100:.2f}'))
        self.roundness_value.setText('Off' if self.roundness.value() == 0 else f'{self.roundness.value()/100:.2f}')
        row = QtWidgets.QHBoxLayout(); row.addWidget(self.roundness, 1); row.addWidget(self.roundness_value)
        form.addRow('Minimum roundness', row)
        self.device = QtWidgets.QComboBox()
        self.device.addItems(['auto', 'cpu', 'gpu'])
        self.device.setCurrentText(self.settings.value('device', 'auto'))
        form.addRow('Device', self.device)
        left.addWidget(self.options)
        self.options.hide()
        self.options_button.toggled.connect(lambda visible: (
            self.options.setVisible(visible), self.options_button.setArrowType(
                QtCore.Qt.ArrowType.DownArrow if visible else QtCore.Qt.ArrowType.RightArrow)))
        row = QtWidgets.QHBoxLayout()
        self.run_button = self._button('Analyze selected', self.start, row)
        self.run_button.setStyleSheet('background:#205e58;font-weight:bold;padding:10px;')
        self.cancel_button = self._button('Cancel', self.cancel, row)
        left.addLayout(row)
        self.phase = QtWidgets.QLabel('Add a scan to begin.')
        self.phase.setWordWrap(True)
        self.phase.setStyleSheet('font-weight:600;')
        left.addWidget(self.phase)
        self.overall = QtWidgets.QProgressBar()
        self.overall.setRange(0, 5000)
        self.overall.setValue(0)
        self.overall.setFormat('Ready')
        left.addWidget(self.overall)
        self.stage_progress = QtWidgets.QProgressBar()
        self.stage_progress.setRange(0, 1000)
        self.stage_progress.setValue(0)
        left.addWidget(self.stage_progress)
        self.elapsed = QtWidgets.QLabel('')
        left.addWidget(self.elapsed)
        self.summary = QtWidgets.QLabel('Results are saved automatically as Imaris files, CSV tables and an Excel workbook with cell volumes and distances.')
        self.summary.setWordWrap(True)
        left.addWidget(self.summary)
        self.saved_button = self._button('Open saved analysis…', self.open_saved, left)
        self.results_button = self._button('Open result folder', self.open_result_folder, left)
        self.excel_button = self._button('Export Excel…', self.export_excel, left)
        note = QtWidgets.QLabel('Processes time point 1. Cells without signals are removed. A = signal; B = nearest cell border.')
        note.setWordWrap(True)
        note.setObjectName('muted')
        left.addWidget(note)
        left.addStretch()
        for label in body.findChildren(QtWidgets.QLabel):
            if label.wordWrap():
                label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        self.sidebar.setWidget(body)
        splitter.addWidget(self.sidebar)

        right = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(right)
        self.heading = QtWidgets.QLabel('No scan loaded')
        self.heading.setWordWrap(True)
        self.heading.setObjectName('heading')
        self.heading.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        layout.addWidget(self.heading)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)
        image_page = QtWidgets.QWidget()
        image_layout = QtWidgets.QVBoxLayout(image_page)
        row = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(['2D layers', '3D volume'])
        self.mode.currentIndexChanged.connect(self.change_mode)
        row.addWidget(self.mode)
        self.fit_button = self._button('Fit', self.fit_view, row)
        self.channels_button = self._button('Channels…', self.show_channels, row)
        row.addStretch()
        self.show_cells, self.show_points, self.show_lines = [QtWidgets.QCheckBox(name) for name in ('Cells', 'A/B', 'Lines')]
        for box in (self.show_cells, self.show_points, self.show_lines):
            box.setChecked(True)
            box.toggled.connect(self.refresh_annotations)
            row.addWidget(box)
        image_layout.addLayout(row)
        self.views = QtWidgets.QStackedWidget()
        self.slice_view = SliceView()
        self.slice_view.stepped.connect(lambda delta: self.slice.setValue(self.slice.value()+delta))
        self.views.addWidget(self.slice_view)
        image_layout.addWidget(self.views, 1)
        self.slice_row = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self.slice_row)
        row.setContentsMargins(0, 0, 0, 0)
        self.axis = QtWidgets.QComboBox()
        self.axis.addItems(['XY / Z', 'XZ / Y', 'YZ / X'])
        self.axis.currentIndexChanged.connect(self.change_axis)
        row.addWidget(self.axis)
        self.slice = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slice.valueChanged.connect(self.change_slice)
        row.addWidget(self.slice, 1)
        self.slice_label = QtWidgets.QLabel('')
        row.addWidget(self.slice_label)
        image_layout.addWidget(self.slice_row)
        self.selection_info = QtWidgets.QLabel('Select a row in Distances to locate its A/B pair.')
        self.selection_info.setWordWrap(True)
        image_layout.addWidget(self.selection_info)
        self.tabs.addTab(image_page, 'Image')
        self.distances = self._table(['Signal', 'Cell', 'Channel', 'Distance (µm)', 'Score',
                                      'A X', 'A Y', 'A Z', 'B X', 'B Y', 'B Z', 'Note'])
        self.distances.itemSelectionChanged.connect(self.focus_measurement)
        self.distances.itemDoubleClicked.connect(lambda _: self.tabs.setCurrentIndex(0))
        self.tabs.addTab(self.distances, 'Distances')
        self.counts = self._table(['Cell', 'Channel', 'Channel name', 'Signals', 'Volume (µm³)'])
        self.tabs.addTab(self.counts, 'Cell counts')
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 840])
        self.statusBar().showMessage('Original scans are read-only.')

    @staticmethod
    def _table(columns):
        table = QtWidgets.QTableWidget(0, len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        table.verticalHeader().hide()
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        return table

    def sync_buttons(self):
        for widget in (self.add_button, self.folder_button, self.remove_button, self.select_button,
                       self.scans, self.output, self.output_button, self.options, self.saved_button):
            widget.setEnabled(not self.running)
        self.run_button.setEnabled(not self.running and not self.loading and self.scans.count() > 0)
        self.cancel_button.setEnabled(self.running)
        self.results_button.setEnabled(self.result_path is not None)
        self.excel_button.setEnabled(self.result_report is not None and not self.loading)
        for widget in (self.channels_button, self.fit_button, self.mode, self.axis, self.slice):
            widget.setEnabled(self.dataset is not None)

    def choose_folder(self, field):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose folder', field.text())
        if path:
            field.setText(path)

    def open_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, 'Add Imaris scans', '', 'Imaris (*.ims *.imx *.IMS *.IMX)')
        if paths:
            self.add_paths(paths)

    def open_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Add scans from folder and subfolders')
        if path:
            self.add_paths([path])

    def background(self, function, completed, failed=None):
        job = Job(function)
        self._jobs.add(job)
        job.signals.result.connect(completed)
        job.signals.error.connect(failed or self.error)
        job.signals.finished.connect(lambda: self._jobs.discard(job))
        QtCore.QThreadPool.globalInstance().start(job)

    def add_paths(self, paths):
        if self.running:
            return
        self.phase.setText('Reading scan list…')
        def read(_):
            entries = []
            for path in paths:
                entries.extend(scan_library(path)[0] if Path(path).is_dir() else [entry_for_path(path)])
            return sorted({str(e.image.resolve()) for e in entries})
        def done(paths):
            known = {self.scans.item(i).data(QtCore.Qt.ItemDataRole.UserRole)['image'] for i in range(self.scans.count())}
            first = None
            for path in paths:
                if path in known:
                    continue
                item = QtWidgets.QListWidgetItem(Path(path).name)
                item.setToolTip(path)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, dict(image=path))
                self.scans.addItem(item)
                if first is None:
                    first = item
            if first:
                self.scans.setCurrentItem(first)
            self.phase.setText('Select scans, then click Analyze selected.')
            self.sync_buttons()
        self.background(read, done)

    def remove_selected(self):
        for item in self.scans.selectedItems():
            self.scans.takeItem(self.scans.row(item))
        self.sync_buttons()

    def scan_selected(self, item, previous=None):
        if item is None or self.running:
            return
        record = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if record.get('result'):
            self.load_result(record['result'])
        else:
            self._load(record['image'])

    def _load(self, image, result_path=None):
        self._source_image=str(Path(image).resolve())
        if result_path:
            saved=json.loads(Path(result_path).read_text())
            self._source_image=saved.get('source',self._source_image)
        self._generation += 1
        generation = self._generation
        self.loading = True
        self.sync_buttons()
        self.heading.setText('Loading ' + Path(image).name + '…')
        def read(_):
            if result_path:
                manifest = json.loads(Path(result_path).read_text(encoding='utf-8'))
                root = Path(result_path).parent
                dataset = load_dataset(LibraryEntry(root / manifest['image'], root / manifest['scene']))
                _, report = read_signals(root / manifest['signal_report'])
                annotations = (root / manifest['image']).with_suffix('.annotations.json')
                if annotations.is_file():
                    saved = json.loads(annotations.read_text(encoding='utf-8'))
                    if 'manual_measurements' in saved:
                        report['manual_measurements'] = saved['manual_measurements']
            else:
                dataset = load_dataset(LibraryEntry(Path(image)))
                dataset.scene = Scene()  # Raw preview; manual contours never become pipeline output.
                report = None
            for channel, volume in zip(dataset.metadata.channels, dataset.volumes):
                sample = volume.ravel()[::max(1, volume.size // 1_000_000)]
                low, high = np.percentile(sample, [1, 99.8])
                if high > low:
                    channel.low, channel.high = float(low), float(high)
            return dataset, report
        def done(value):
            if generation != self._generation:
                return
            self.loading = False
            self.result_path = Path(result_path) if result_path else None
            self.set_dataset(*value)
            if not self.running:
                self.phase.setText('Saved analysis ready' if result_path else 'Select scans, then click Analyze selected.')
                self.overall.setValue(self.overall.maximum() if result_path else 0)
                self.overall.setFormat('Complete' if result_path else 'Ready')
            self.sync_buttons()
        def failed(message):
            if generation == self._generation:
                self.loading = False
                self.sync_buttons()
                self.error(message)
        self.background(read, done, failed)

    def set_dataset(self, dataset, report=None):
        if report and 'cell_volumes' not in report:
            from ..workflow.volumes import add_cell_volumes
            add_cell_volumes(report, dataset.scene.surfaces)
        self.dataset, self.result_report = dataset, report
        if self.channel_dialog:
            self.channel_dialog.close()
            self.channel_dialog.deleteLater()
            self.channel_dialog = None
        self.heading.setText(dataset.metadata.path.name)
        self.slice_view.set_dataset(dataset)
        self._volume_dataset = None
        self.change_axis(self.axis.currentIndex())
        self.change_mode(self.mode.currentIndex())
        self.distances.blockSignals(True)
        self.distances.setRowCount(0)
        self.counts.setRowCount(0)
        self.selection_info.setText('Select a row in Distances to locate its A/B pair.')
        if report:
            for i, point in enumerate(report['points']):
                self.distances.insertRow(i)
                values = [point['signal_id'], point['cell_id'], point['channel']+1,
                          f'{point["distance_um"]:.4f}', f'{point["score"]:.4f}',
                          *[f'{v:.4f}' for v in point['xyz']], *[f'{v:.4f}' for v in point['border_xyz']],
                          'Image-edge contact' if point.get('border_at_image_edge') else '']
                for j, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(str(value))
                    if j == 0:
                        item.setData(QtCore.Qt.ItemDataRole.UserRole, point)
                    self.distances.setItem(i, j, item)
            for i, row in enumerate(report['counts']):
                self.counts.insertRow(i)
                volume = next((v['volume_um3'] for v in report.get('cell_volumes', []) if v['cell_id'] == row['cell_id']), None)
                for j, value in enumerate([row['cell_id'], row['channel']+1, row['channel_name'], row['count'],
                                          f'{volume:.3f}' if volume is not None else 'Unavailable']):
                    self.counts.setItem(i, j, QtWidgets.QTableWidgetItem(str(value)))
            filtering = report['signal_filter']
            self.summary.setText(f'{filtering["kept_cell_count"]} cells kept · {len(report["points"])} signals and distances\n'
                                 f'{filtering["removed_cell_count"]} cells without signals removed.')
        else:
            self.summary.setText('Ready for automatic cell contours, signals and A/B measurements.')
        self.distances.blockSignals(False)
        self.refresh_annotations()

    def start(self):
        if self.running or self.loading:
            return
        if not self.output.text().strip():
            self.error('Choose a folder for analysis results.')
            return
        items = self.scans.selectedItems()
        if not items:
            self.error('Select at least one scan.')
            return
        self.queue = [self.scans.row(item) for item in items]
        self.total_scans = len(self.queue)
        self.scan_number = 0
        self.running = True
        self.started = time.monotonic()
        self.cancel_button.setText('Cancel')
        self.timer.start()
        self.overall.setRange(0, self.total_scans*5000)
        self.overall.setValue(0)
        for key, widget in [('output', self.output), ('cell_model', self.cell_model),
                            ('boundary_model', self.boundary_model), ('signal_model', self.signal_model)]:
            self.settings.setValue(key, widget.text())
        self.settings.setValue('pipeline', self.pipeline.currentData())
        self.settings.setValue('minimum_roundness_percent', self.roundness.value())
        self.settings.setValue('device', self.device.currentText())
        self.settings.setValue('membrane_snap', self.membrane_snap.isChecked())
        self.settings.setValue('learned_boundary', self.learned_boundary.isChecked())
        self.settings.setValue('membrane_shift_um', self.membrane_shift.value())
        self.settings.setValue('cell_recovery', self.cell_recovery.isChecked())
        self.settings.setValue('stardist_proposals', self.stardist_proposals.isChecked())
        self.settings.setValue('consensus_cells', self.consensus_cells.isChecked())
        self.sync_buttons()
        self.next_scan()

    def select_pipeline(self, *_):
        self.use_conservative_cells()
        self.signal_model.setText(str(default_models()[1]))
        key = self.pipeline.currentData()
        root = Path(__file__).resolve().parents[2] / 'models'
        if key == 'preprocessed':
            self.cell_model.setText(str(root / 'cells-preprocessed'))
        elif key == 'stardist2d':
            self.cell_model.setText(str(root / 'cells-stardist2d'))
        elif key == 'balanced':
            self.cell_model.setText(str(root / 'cells'))
        elif key in ('crop', 'signal-first'):
            self.cell_model.setText(default_proposal_model())
            self.boundary_model.setText(default_refinement_model())
        elif key in ('hybrid', 'consensus'):
            self.cell_model.setText(sensitive_stardist_model())
            self.boundary_model.setText(default_refinement_model())
            self.stardist_proposals.setChecked(True)
            self.consensus_cells.setChecked(key == 'consensus')

    def use_conservative_cells(self):
        self.cell_model.setText(str(Path(__file__).resolve().parents[2] / 'models/cells-conservative'))
        self.boundary_model.clear()
        for checkbox in (self.membrane_snap, self.learned_boundary, self.cell_recovery,
                         self.stardist_proposals, self.consensus_cells):
            checkbox.setChecked(False)

    def next_scan(self):
        if not self.queue:
            self.running = False
            self.timer.stop()
            self.phase.setText('Analysis complete')
            self.overall.setValue(self.overall.maximum())
            self.overall.setFormat('Complete')
            self.sync_buttons()
            return
        self.active_row = self.queue.pop(0)
        item = self.scans.item(self.active_row)
        self.scan_number += 1
        self.active_image = item.data(QtCore.Qt.ItemDataRole.UserRole)['image']
        item.setText('Processing · ' + Path(self.active_image).name)
        try:
            roles=self.channel_roles.get(str(Path(self.active_image).resolve()),{})
            self.controller.start(self.active_image, self.output.text(), self.cell_model.text(),
                                  self.signal_model.text(), self.device.currentText(), self.boundary_model.text().strip() or None,
                                  membrane_snap=dict(max_shift_um=self.membrane_shift.value()) if self.membrane_snap.isChecked() or self.learned_boundary.isChecked() else None,
                                  boundary_guidance=self.learned_boundary_model if self.learned_boundary.isChecked() else None,
                                  recovery={} if self.cell_recovery.isChecked() else None,
                                  stardist_proposals=self.stardist_proposals.isChecked(),
                                  consensus_model=default_proposal_model() if self.consensus_cells.isChecked() else None,
                                  roundness=self.roundness.value()/100,
                                  signal_first=self.pipeline.currentData() == 'signal-first',
                                  cell_channel=roles.get('cell'),signal_channels=roles.get('signals'))
        except Exception as exc:
            self.failed(str(exc))

    def processing_progress(self, event):
        stage = event['stage']
        stages = event.get('stages', 5)
        total, current = event.get('total', 0), event.get('current', 0)
        name = event.get('stage_name', STAGES[min(stage-1,len(STAGES)-1)][1])
        self.phase.setText(f'{stage} / {stages} · {name}\n{event["phase"]}')
        self.overall.setFormat(f'Scan {self.scan_number} / {self.total_scans} · Step {stage} / {stages}')
        fraction = min(1., max(0., current/total)) if total > 0 else 0.
        self.overall.setValue((self.scan_number-1)*5000 + round((stage-1+fraction)/stages*5000))
        self.stage_progress.setRange(0, 1000 if total > 0 else 0)
        if total > 0:
            self.stage_progress.setValue(round(fraction*1000))
            self.stage_progress.setFormat(f'{current} / {total}')
        self.statusBar().showMessage(f'{Path(self.active_image).name} · {event["phase"]}')

    def completed(self, report):
        item = self.scans.item(self.active_row)
        record = item.data(QtCore.Qt.ItemDataRole.UserRole)
        record['result'] = report['report_path']
        item.setData(QtCore.Qt.ItemDataRole.UserRole, record)
        item.setText('✓ ' + Path(record['image']).name)
        self.stage_progress.setRange(0, 1000)
        self.stage_progress.setValue(1000)
        self.stage_progress.setFormat('Saved')
        self.load_result(report['report_path'])
        QtCore.QTimer.singleShot(0, self.next_scan)

    def failed(self, message):
        self.queue.clear()
        self.running = False
        self.timer.stop()
        self.phase.setText('Analysis failed')
        if hasattr(self, 'active_row'):
            self.scans.item(self.active_row).setText('Failed · ' + Path(self.active_image).name)
        self.stage_progress.setRange(0, 1000)
        self.sync_buttons()
        self.error(message)

    def cancelled(self):
        self.queue.clear()
        self.running = False
        self.timer.stop()
        self.phase.setText('Cancelled — completed results were kept.')
        self.stage_progress.setRange(0, 1000)
        if hasattr(self, 'active_row'):
            self.scans.item(self.active_row).setText('Cancelled · ' + Path(self.active_image).name)
        self.sync_buttons()

    def cancel(self):
        self.queue.clear()
        self.controller.cancel()
        self.cancel_button.setText('Stop now')
        self.phase.setText('Stopping after the current operation…')

    def tick(self):
        seconds = int(time.monotonic()-self.started)
        self.elapsed.setText(f'Elapsed {seconds//60:02d}:{seconds%60:02d}')

    def open_saved(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open saved analysis', self.output.text(), 'Analysis result (analysis-result.json)')
        if path:
            try:
                self.load_result(path)
            except Exception as exc:
                self.error(str(exc))

    def load_result(self, path):
        path = Path(path).resolve()
        report = json.loads(path.read_text(encoding='utf-8'))
        if report.get('engine') != 'fish-analysis' or report.get('status') != 'complete':
            raise ValueError('Choose an analysis-result.json from a completed analysis.')
        self._load(path.parent / report['image'], path)

    def export_excel(self):
        if self.result_report is None:
            return
        default = str(self.result_path.parent / (self.dataset.metadata.path.stem + '.xlsx')) if self.result_path else 'analysis.xlsx'
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export volumes and distances', default, 'Excel workbook (*.xlsx)')
        if path:
            try:
                from ..workflow.excel import write_excel
                write_excel(Path(path).with_suffix('.xlsx'), self.result_report)
                self.phase.setText('Excel workbook saved: ' + str(path))
            except Exception as exc:
                self.error(str(exc))

    def open_result_folder(self):
        if self.result_path:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.result_path.parent / 'export')))

    def change_axis(self, axis):
        if self.dataset is None:
            return
        self.slice_view.axis = axis
        self.slice.blockSignals(True)
        self.slice.setRange(0, self.dataset.volumes[0].shape[axis]-1)
        self.slice.setValue(self.slice.maximum()//2)
        self.slice.blockSignals(False)
        self.change_slice(self.slice.value())
        self.slice_view.fit()

    def change_slice(self, index):
        if self.dataset is not None:
            self.slice_view.index = index
            self.slice_view.update_image()
            self.slice_label.setText(f'{index+1} / {self.slice.maximum()+1}')

    def change_mode(self, index):
        if self.dataset is None:
            return
        if index:
            if self.volume_view is None:
                self.volume_view = VolumeView(off_screen=self.off_screen)
                self.volume_view.full_detail = self.full_detail
                self.views.addWidget(self.volume_view)
            if self._volume_dataset is not self.dataset:
                self.volume_view.set_dataset(self.dataset)
                self._volume_dataset = self.dataset
            self.views.setCurrentWidget(self.volume_view)
        else:
            self.views.setCurrentWidget(self.slice_view)
        self.slice_row.setVisible(index == 0)
        self.refresh_annotations()

    def show_channels(self):
        if not self.dataset:
            return
        if self.channel_dialog:
            self.channel_dialog.show()
            self.channel_dialog.raise_()
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Channel roles and display')
        dialog.resize(370, 620)
        layout = QtWidgets.QVBoxLayout(dialog)
        roles_box=QtWidgets.QGroupBox('Processing roles for this scan')
        roles_layout=QtWidgets.QVBoxLayout(roles_box)
        note=QtWidgets.QLabel('Choose one cell channel and the signal channels. Other channels are ignored for signal detection. These choices do not change display colors.')
        note.setWordWrap(True);roles_layout.addWidget(note)
        self.cell_channel_choice=QtWidgets.QComboBox()
        self.cell_channel_choice.addItem('Cell channel: automatic',None)
        labels=[f'{c.name}' + (f' · {c.excitation_nm:g} nm' if c.excitation_nm is not None else '') for c in self.dataset.metadata.channels]
        for i,c in enumerate(self.dataset.metadata.channels):
            swatch=QtGui.QPixmap(12,12);swatch.fill(QtGui.QColor.fromRgbF(*c.color))
            self.cell_channel_choice.addItem(QtGui.QIcon(swatch),f'Cell channel: {i+1} · {labels[i]}',i)
        source=self._source_image or str(self.dataset.metadata.path.resolve())
        saved=self.channel_roles.get(source,{})
        self.cell_channel_choice.setCurrentIndex(max(0,self.cell_channel_choice.findData(saved.get('cell'))))
        roles_layout.addWidget(self.cell_channel_choice)
        self.signal_channel_checks=[]
        names=[c.name for c in self.dataset.metadata.channels]
        chosen=saved.get('signals')
        if chosen is None:chosen=list(range(len(names)))
        for i,name in enumerate(names):
            check=QtWidgets.QCheckBox(f'Signal channel: {i+1} · {labels[i]}')
            swatch=QtGui.QPixmap(12,12);swatch.fill(QtGui.QColor.fromRgbF(*self.dataset.metadata.channels[i].color))
            check.setIcon(QtGui.QIcon(swatch))
            check.setChecked(i in chosen);roles_layout.addWidget(check)
            self.signal_channel_checks.append(check)
        def save_roles(*_):
            cell=self.cell_channel_choice.currentData()
            for i,check in enumerate(self.signal_channel_checks):
                check.setEnabled(i!=cell)
                if i==cell:check.setChecked(False)
            signals=[i for i,c in enumerate(self.signal_channel_checks) if c.isChecked() and i!=cell]
            self.channel_roles[source]=dict(cell=cell,signals=signals,names=names)
            self.settings.setValue('channel_roles',json.dumps(self.channel_roles))
        self.cell_channel_choice.currentIndexChanged.connect(save_roles)
        for check in self.signal_channel_checks:check.toggled.connect(save_roles)
        for i,check in enumerate(self.signal_channel_checks):
            if i==self.cell_channel_choice.currentData():check.setChecked(False);check.setEnabled(False)
        roles_box.setEnabled(not self.running)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        channels = QtWidgets.QVBoxLayout(body)
        channels.addWidget(roles_box)
        for i, (channel, volume) in enumerate(zip(self.dataset.metadata.channels, self.dataset.volumes)):
            panel = ChannelPanel(channel, volume, i)
            panel.changed.connect(lambda: self.refresh_timer.start())
            channels.addWidget(panel)
        scroll.setWidget(body)
        layout.addWidget(scroll)
        detail = QtWidgets.QCheckBox('Full volume detail in 3D')
        detail.setChecked(self.full_detail)
        detail.setToolTip('Use all voxels instead of a downsampled 3D preview. Measurement coordinates are always full precision.')
        def changed(value):
            self.full_detail = value
            if self.volume_view:
                self.volume_view.full_detail = value
                self.refresh_images()
        detail.toggled.connect(changed)
        layout.addWidget(detail)
        self.channel_dialog = dialog
        dialog.show()

    def refresh_images(self):
        if self.dataset is not None:
            self.slice_view.update_image()
            if self.volume_view and self._volume_dataset is self.dataset:
                self.volume_view.update_volume()

    def refresh_annotations(self):
        for view in (self.slice_view, self.volume_view):
            if view is None or view.dataset is not self.dataset:
                continue
            view.surfaces_visible = self.show_cells.isChecked()
            view.markers_visible = self.show_points.isChecked()
            view.lines_visible = self.show_lines.isChecked()
            if view is self.slice_view:
                view.update_overlays()
            else:
                view.update_annotations()

    def fit_view(self):
        if self.dataset is not None:
            (self.volume_view if self.mode.currentIndex() else self.slice_view).fit()

    def focus_measurement(self):
        row = self.distances.currentRow()
        if row < 0 or self.dataset is None:
            return
        point = self.distances.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole)
        index = next((i for i, m in enumerate(self.dataset.scene.measurements)
                      if re.match(rf'Signal {point["signal_id"]}(?:;|\s)', m.name)), None)
        if index is None:
            return
        self.slice_view.selected = ('measurement', index)
        if self.volume_view:
            self.volume_view.selected = ('measurement', index)
        xyz = np.asarray(point['xyz'])
        axis = 2-self.slice_view.axis
        self.slice.setValue(int(round((xyz[axis]-self.dataset.center_origin[axis])/self.dataset.spacing[axis])))
        surface = next((s for s in self.dataset.scene.surfaces if s.name.startswith(f'Predicted cell {point["cell_id"]} -')), None)
        coordinates = surface.vertices if surface is not None else np.asarray([point['xyz'], point['border_xyz']])
        low, high = coordinates.min(0)-.8, coordinates.max(0)+.8
        u, v = self.slice_view.projection_axes
        self.slice_view.fitInView(QtCore.QRectF(low[u], low[v], high[u]-low[u], high[v]-low[v]), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        if self.volume_view and self._volume_dataset is self.dataset:
            self.volume_view.plotter.reset_camera(bounds=tuple(v for lo, hi in zip(low, high) for v in (lo, hi)))
        self.selection_info.setText(f'Signal {point["signal_id"]} · cell {point["cell_id"]} · channel {point["channel"]+1} · A → B: {point["distance_um"]:.4f} µm')
        self.refresh_annotations()

    def error(self, message):
        QtWidgets.QMessageBox.warning(self, 'FISH Analysis', str(message)[-3000:])

    def closeEvent(self, event):
        self._generation += 1
        self.timer.stop()
        self.refresh_timer.stop()
        self.controller.shutdown()
        if self.volume_view:
            self.volume_view.close()
        super().closeEvent(event)
