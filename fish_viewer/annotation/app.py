"""Single-image border-painting UI; no models or training frameworks are loaded."""
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .data import fill, paint, save_crop

COLORS = {1: (80, 170, 255), 3: (255, 220, 65)}


def pixmap(array, rgba=False):
    array = np.ascontiguousarray(array, dtype=np.uint8)
    fmt = QtGui.QImage.Format.Format_RGBA8888 if rgba else QtGui.QImage.Format.Format_RGB888
    return QtGui.QPixmap.fromImage(QtGui.QImage(array.data, array.shape[1], array.shape[0], array.strides[0], fmt).copy())


class Canvas(QtWidgets.QGraphicsView):
    changed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.base = self.scene().addPixmap(QtGui.QPixmap())
        self.overlay = self.scene().addPixmap(QtGui.QPixmap())
        self.overlay.setZValue(1)
        pen = QtGui.QPen(QtGui.QColor('white'))
        pen.setCosmetic(True)
        self.brush_preview = self.scene().addEllipse(0, 0, 1, 1, pen)
        self.brush_preview.setZValue(2)
        self.brush_preview.hide()
        self.setBackgroundBrush(QtGui.QColor('#141414'))
        self.setTransformationAnchor(self.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(self.ViewportAnchor.AnchorViewCenter)
        self.setMouseTracking(True)
        self.crop = None
        self.category = None
        self.diameter = 8
        self.last = None
        self.pan = None
        self.fill_mode = False
        self.history = []
        self.pending = None
        self.limits = (0, 1)
        self.gain = 1.

    def set_crop(self, crop):
        self.crop = crop
        self.last = None
        self.pending = None
        self.history.clear()
        self.brush_preview.hide()
        low, high = np.percentile(crop.image, (1, 99.5))
        if high <= low:
            low, high = 0, max(float(crop.image.max()), 1)
        self.limits = (low, high)
        self.scene().setSceneRect(0, 0, crop.labels.shape[1], crop.labels.shape[0])
        self.refresh()
        self.fit()

    def fit(self):
        self.resetTransform()
        self.fitInView(self.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def refresh(self):
        if self.crop is None:
            return
        low, high = self.limits
        intensity = np.clip((self.crop.image.astype(np.float32) - low) / max(high - low, 1e-6) * self.gain, 0, 1)
        self.base.setPixmap(pixmap(np.repeat((intensity * 255)[..., None], 3, axis=2)))
        self.refresh_marks()

    def refresh_marks(self):
        rgba = np.zeros((*self.crop.labels.shape, 4), np.uint8)
        for category, color in COLORS.items():
            rgba[self.crop.labels == category] = (*color, 110)
        self.overlay.setPixmap(pixmap(rgba, True))

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        scale = self.transform().m11() * factor
        if .05 <= scale <= 100:
            self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event):
        self.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
        if event.button() in (QtCore.Qt.MouseButton.MiddleButton, QtCore.Qt.MouseButton.RightButton):
            self.pan = event.position().toPoint()
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
        elif event.button() == QtCore.Qt.MouseButton.LeftButton and self.crop is not None and self.category is not None:
            p = self.mapToScene(event.position().toPoint())
            point = (p.x(), p.y())
            self.pending = self.crop.labels.copy()
            if self.fill_mode:
                fill(self.crop.labels, point, self.category)
                self.commit_edit()
                self.refresh_marks()
            else:
                self.last = point
                self.stroke(point)
        event.accept()

    def mouseMoveEvent(self, event):
        p = self.mapToScene(event.position().toPoint())
        radius = self.diameter / 2
        self.brush_preview.setRect(p.x() - radius, p.y() - radius, self.diameter, self.diameter)
        self.brush_preview.setVisible(self.crop is not None and self.category is not None and self.pan is None and not self.fill_mode)
        if self.pan is not None:
            delta = event.position().toPoint() - self.pan
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.pan = event.position().toPoint()
        elif self.last is not None:
            p = self.mapToScene(event.position().toPoint())
            self.stroke((p.x(), p.y()))
        event.accept()

    def leaveEvent(self, event):
        self.brush_preview.hide()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.last is not None:
            p = self.mapToScene(event.position().toPoint())
            self.stroke((p.x(), p.y()))
            self.commit_edit()
        self.last = self.pan = None
        self.unsetCursor()
        event.accept()

    def stroke(self, point):
        paint(self.crop.labels, self.last, point, self.diameter, self.category)
        self.last = point
        self.refresh_marks()
        self.changed.emit()

    def commit_edit(self):
        if self.pending is not None and not np.array_equal(self.pending, self.crop.labels):
            self.history.append(self.pending)
            self.history = self.history[-100:]
        self.pending = None
        self.changed.emit()

    def undo(self):
        if not self.isEnabled() or self.crop is None:
            return
        self.commit_edit()
        self.last = None
        if self.history:
            self.crop.labels[:] = self.history.pop()
            self.refresh_marks()
        self.changed.emit()

    def clear_all(self):
        if self.crop is not None:
            self.commit_edit()
            self.last = None
            self.pending = self.crop.labels.copy()
            self.crop.labels.fill(0)
            self.commit_edit()
            self.refresh_marks()


class Job(QtCore.QThread):
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, function, parent):
        super().__init__(parent)
        self.function = function

    def run(self):
        try:
            self.done.emit(self.function())
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')


class AnnotationWindow(QtWidgets.QMainWindow):
    def __init__(self, sampler, output, autoload=True):
        super().__init__()
        self.sampler = sampler
        self.output = output
        self.job = None
        self.busy = False
        self.saved = 0
        self.setWindowTitle('FISH training annotations')
        self.resize(1000, 720)
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        self.title = QtWidgets.QLabel('Random crops from training scans only')
        self.title.setWordWrap(True)
        layout.addWidget(self.title)
        tools = QtWidgets.QHBoxLayout()
        layout.addLayout(tools)
        self.group = QtWidgets.QButtonGroup(self)
        for category, text in [(3, 'Likely cell border'), (1, 'Likely not cell border'), (0, 'Eraser')]:
            button = QtWidgets.QPushButton(text)
            button.setCheckable(True)
            if category:
                color = '#%02x%02x%02x' % COLORS[category]
                button.setStyleSheet(f'QPushButton {{border-bottom: 3px solid {color}; padding: 5px;}} QPushButton:checked {{background: #555; color: white;}}')
            self.group.addButton(button, category)
            tools.addWidget(button)
        self.group.idClicked.connect(self.select_category)
        tools.addWidget(QtWidgets.QLabel('Brush (px)'))
        self.size = QtWidgets.QSpinBox()
        self.size.setRange(1, 100)
        self.size.setValue(8)
        tools.addWidget(self.size)
        self.clear = QtWidgets.QPushButton('Clear all')
        self.clear.setToolTip('Erase all annotations on this image; Ctrl+Z restores them.')
        tools.addWidget(self.clear)
        self.canvas = Canvas()
        layout.addWidget(self.canvas, 1)
        self.size.valueChanged.connect(lambda value: setattr(self.canvas, 'diameter', value))
        self.clear.clicked.connect(self.clear_all)
        self.canvas.changed.connect(self.update_save)
        self.undo_button = QtWidgets.QPushButton('Undo')
        self.undo_button.setToolTip('Undo the last stroke, fill, erasure or clear (Ctrl+Z).')
        self.undo_shortcut = QtGui.QShortcut(QtGui.QKeySequence('Ctrl+Z'), self)
        self.undo_shortcut.activated.connect(self.canvas.undo)
        self.undo_button.clicked.connect(self.canvas.undo)
        tools.addWidget(self.undo_button)
        self.fill_button = QtWidgets.QPushButton('Fill')
        self.fill_button.setCheckable(True)
        self.fill_button.setToolTip('Fill the connected annotation region under the cursor with the selected category. A closed painted border stops the fill.')
        self.fill_button.toggled.connect(lambda active: setattr(self.canvas, 'fill_mode', active))
        tools.insertWidget(3, self.fill_button)
        display = QtWidgets.QHBoxLayout()
        layout.addLayout(display)
        display.addWidget(QtWidgets.QLabel('Cell channel only · single image'))
        display.addStretch()
        display.addWidget(QtWidgets.QLabel('Brightness'))
        self.brightness = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.brightness.setRange(25, 400)
        self.brightness.setValue(100)
        self.brightness.setMaximumWidth(120)
        self.brightness.valueChanged.connect(self.set_gain)
        display.addWidget(self.brightness)
        fit = QtWidgets.QPushButton('Fit view')
        fit.clicked.connect(self.canvas.fit)
        display.addWidget(fit)
        actions = QtWidgets.QHBoxLayout()
        layout.addLayout(actions)
        self.hint = QtWidgets.QLabel('Choose a category before painting. Unpainted = unknown.\nWheel: zoom · right/middle drag: pan · Ctrl+Z: undo · Fill: connected region.')
        self.hint.setWordWrap(True)
        actions.addWidget(self.hint, 1)
        self.skip = QtWidgets.QPushButton('Skip')
        self.skip.setToolTip('Discard every mark on this crop and load another. Nothing is stored.')
        self.skip.clicked.connect(self.skip_crop)
        actions.addWidget(self.skip)
        self.save = QtWidgets.QPushButton('Save && next')
        self.save.setShortcut('Ctrl+Return')
        self.save.clicked.connect(self.save_next)
        actions.addWidget(self.save)
        self.statusBar().showMessage(f'{len(sampler.cases)} training scans · Output: {output}')
        self.save.setEnabled(False)
        if autoload:
            QtCore.QTimer.singleShot(0, self.next_crop)

    def select_category(self, category):
        self.canvas.category = category
        self.canvas.setCursor(QtCore.Qt.CursorShape.CrossCursor)

    def set_busy(self, value):
        self.busy = value
        self.canvas.setEnabled(not value)
        self.skip.setEnabled(not value)
        self.clear.setEnabled(not value)
        self.undo_button.setEnabled(not value and bool(self.canvas.history))
        self.save.setEnabled(not value and self.canvas.crop is not None and bool(np.any(self.canvas.crop.labels)))

    def start_job(self, function, on_done):
        if self.busy:
            return
        self.set_busy(True)
        job = Job(function, self)
        self.job = job
        job.done.connect(on_done)
        job.failed.connect(self.failed)
        # Keep each QThread alive until finished, including while a following job starts.
        job.finished.connect(job.deleteLater)
        job.start()

    def next_crop(self):
        self.statusBar().showMessage('Loading a random training crop…')
        self.start_job(self.sampler.next_crop, self.loaded)

    def loaded(self, crop):
        self.canvas.set_crop(crop)
        self.brightness.blockSignals(True)
        self.brightness.setValue(100)
        self.brightness.blockSignals(False)
        self.canvas.gain = 1.
        self.canvas.refresh()
        offset = crop.metadata['offset_zyx']
        self.title.setText(f'{crop.metadata["source"]}\nCrop offset Z/Y/X: {offset} · cell channel: {crop.metadata["cell_channel_index"] + 1} · time: {crop.metadata["time_point"]}')
        self.set_busy(False)
        self.statusBar().showMessage(f'{self.saved} crops saved · Unpainted pixels remain unknown · Output: {self.output}')
        # Fit after the changed title row have received their layout sizes.
        QtCore.QTimer.singleShot(0, self.canvas.fit)

    def set_gain(self, value):
        self.canvas.gain = value / 100
        self.canvas.refresh()

    def clear_all(self):
        self.canvas.clear_all()

    def update_save(self):
        self.undo_button.setEnabled(not self.busy and bool(self.canvas.history))
        self.save.setEnabled(not self.busy and self.canvas.crop is not None and bool(np.any(self.canvas.crop.labels)))

    def skip_crop(self):
        if not self.busy:
            self.next_crop()

    def save_next(self):
        if self.busy or self.canvas.crop is None or not np.any(self.canvas.crop.labels):
            return
        self.canvas.commit_edit()
        self.canvas.last = None
        self.statusBar().showMessage('Saving image, marks and crop coordinates…')
        self.start_job(lambda: save_crop(self.canvas.crop, self.output), self.saved_crop)

    def saved_crop(self, path):
        self.saved += 1
        self.set_busy(False)
        # The completed crop is no longer editable; load immediately.
        self.canvas.crop = None
        self.next_crop()

    def failed(self, message):
        self.set_busy(False)
        self.statusBar().showMessage(message)
        QtWidgets.QMessageBox.critical(self, 'Annotation operation failed', message + '\nCurrent marks were not discarded. Retry Save, or use Skip to load another crop.')

    def closeEvent(self, event):
        if self.busy:
            self.statusBar().showMessage('Please wait for the current load/save to finish before closing.')
            event.ignore()
            return
        if self.canvas.crop is not None and np.any(self.canvas.crop.labels):
            reply = QtWidgets.QMessageBox.question(self, 'Unsaved marks', 'Discard the current unsaved marks and close?', QtWidgets.QMessageBox.StandardButton.Discard | QtWidgets.QMessageBox.StandardButton.Cancel, QtWidgets.QMessageBox.StandardButton.Cancel)
            if reply != QtWidgets.QMessageBox.StandardButton.Discard:
                event.ignore()
                return
        # Completion signals can arrive just before QThread.run() returns.
        for job in self.findChildren(Job):
            job.wait()
        event.accept()
