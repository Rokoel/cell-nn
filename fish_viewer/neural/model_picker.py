"""Choose a local checkpoint without importing the neural runtime."""
import json
from pathlib import Path

from PySide6 import QtWidgets

from .config import NeuralConfig


def read_model(path):
    path = Path(path).expanduser().resolve()
    manifest = json.loads((path / 'fish-model.json').read_text())
    if manifest.get('version') != 1 or manifest.get('engine') != 'stardist3d':
        raise ValueError('Choose a FISH Viewer StarDist model folder.')
    NeuralConfig.from_dict(manifest['config'])
    checkpoint = path / manifest['checkpoint']
    if not checkpoint.is_file():
        raise ValueError(f'The model checkpoint is missing: {checkpoint.name}')
    return manifest


def discover_models(roots=None):
    if roots is None:
        project = Path(__file__).resolve().parents[2]
        roots = [project / 'artifacts', project / 'neural_runs', Path.cwd() / 'neural_runs']
    found = {}
    for root in roots:
        for manifest in Path(root).glob('**/fish-model.json'):
            path = manifest.parent.resolve()
            if any(word in str(path.relative_to(Path(root).resolve())).lower() for word in ('smoke', 'gpu-check')):
                continue
            try:
                read_model(path)
                report = json.loads((path.parent / 'report.json').read_text())
                if report.get('status') != 'complete' or report.get('mode') != 'train':
                    continue
                family = next((label for key, label in (
                    ('balanced-fish', 'Balanced collection'), ('precision-fish', 'Conservative collection'),
                    ('full-fish', 'Original full collection')) if key in path.parts), 'Saved model')
                count = len(report.get('epochs', []))
                found[path] = f'{family} · {count} epochs · {path.parent.name}'
            except (OSError, ValueError, KeyError, TypeError):
                continue
    # Put the completed balanced collection first; list newer runs first within families.
    return sorted(found.items(), key=lambda pair: (
        0 if 'balanced-fish' in pair[0].parts else 1 if 'precision-fish' in pair[0].parts else 2,
        -pair[0].stat().st_mtime))


class ModelPicker(QtWidgets.QDialog):
    def __init__(self, scan, parent=None, preferred=None, roots=None):
        super().__init__(parent)
        self.setWindowTitle('Auto-select contours')
        self.resize(690, 250)
        self.model_path = None
        layout = QtWidgets.QVBoxLayout(self)
        scan_label = QtWidgets.QLabel(f'Scan: {Path(scan).name}')
        scan_label.setWordWrap(True)
        layout.addWidget(scan_label)
        layout.addWidget(QtWidgets.QLabel('Choose a trained model'))
        self.models = QtWidgets.QComboBox()
        self.models.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.models.setMinimumContentsLength(30)
        for path, title in discover_models(roots):
            self.models.addItem(title, str(path))
        if preferred and (Path(preferred) / 'fish-model.json').is_file():
            index = self.models.findData(str(Path(preferred).resolve()))
            if index < 0:
                try:
                    read_model(preferred)
                    self.models.addItem(f'{Path(preferred).parent.name} · {Path(preferred).name}', str(Path(preferred).resolve()))
                    index = self.models.count() - 1
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            if index >= 0:
                self.models.setCurrentIndex(index)
        self.models.currentIndexChanged.connect(self.update_path)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.models, 1)
        browse = QtWidgets.QPushButton('Browse…')
        browse.clicked.connect(self.browse)
        row.addWidget(browse)
        layout.addLayout(row)
        self.path = QtWidgets.QLineEdit()
        self.path.setReadOnly(True)
        layout.addWidget(self.path)
        note = QtWidgets.QLabel('Roundness and elongation limits are off. Contours must have visible boundary support. '
            'Results are saved separately and displayed on this scan when processing finishes.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self.error = QtWidgets.QLabel()
        self.error.setWordWrap(True)
        layout.addWidget(self.error)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.apply_button = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        self.apply_button.setText('Auto-select contours')
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.update_path()

    def update_path(self):
        self.path.setText(self.models.currentData() or '')
        self.apply_button.setEnabled(bool(self.path.text()))
        self.error.clear()

    def browse(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Choose model folder', self.path.text())
        if path:
            try:
                read_model(path)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.error.setText(str(exc))
                return
            self.models.addItem(f'{Path(path).parent.name} · {Path(path).name}', str(Path(path).resolve()))
            self.models.setCurrentIndex(self.models.count() - 1)

    def accept(self):
        try:
            read_model(self.path.text())
            self.model_path = Path(self.path.text()).resolve()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.error.setText(str(exc))
            return
        super().accept()
