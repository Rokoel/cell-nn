"""First-launch runtime setup with a responsive progress window."""
from PySide6 import QtCore, QtWidgets
from .runtime_cache import Cancelled, prepare


class SetupWorker(QtCore.QThread):
    progress = QtCore.Signal(object)

    def __init__(self, root):
        super().__init__()
        self.root = root
        self.result = None
        self.error = None

    def run(self):
        try:
            self.result = prepare(self.root, lambda *values: self.progress.emit(values),
                                  self.isInterruptionRequested)
        except Exception as error:
            self.error = error


def prepare_with_progress(root):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(False)
    dialog = QtWidgets.QProgressDialog('Checking neural runtime…', 'Cancel', 0, 0)
    dialog.setWindowTitle('FISH Analysis — first-launch setup')
    dialog.setMinimumWidth(480)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    worker = SetupWorker(root)

    def update(values):
        phase, completed, total = values
        dialog.setRange(0, 1000)
        dialog.setValue(round(completed / total * 1000) if total else 0)
        dialog.setLabelText(f'{phase}\n{completed / 1024**2:,.0f} / {total / 1024**2:,.0f} MB\nThis is required only once. Models are already included.')

    def finish():
        if worker.error and not isinstance(worker.error, Cancelled):
            QtWidgets.QMessageBox.critical(dialog, 'Runtime setup failed',
                f'{worker.error}\n\nCheck your internet connection and available disk space, then restart the app. Completed download chunks are retained.')
        app.quit()

    dialog.canceled.connect(worker.requestInterruption)
    worker.progress.connect(update)
    worker.finished.connect(finish)
    dialog.show()
    worker.start()
    app.exec()
    worker.wait()
    if worker.error:
        raise worker.error
    return worker.result
