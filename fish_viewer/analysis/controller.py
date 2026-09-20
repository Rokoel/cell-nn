"""Run the existing inference workers in order without loading TensorFlow into Qt."""
from dataclasses import asdict, replace
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import uuid

from PySide6 import QtCore

from ..neural.config import NeuralConfig
from ..neural.model_picker import read_model
from ..neural.precision import evidence_preset
from ..signals.config import SignalConfig


STAGES = [('cells', 'Detecting cell contours'), ('signals', 'Finding signals inside cells'),
          ('filter', 'Removing cells without signals'), ('measure', 'Creating B points and distance lines'),
          ('export', 'Saving Imaris files and tables')]


def default_models():
    root = Path(__file__).resolve().parents[2]
    cells, signals = root / 'models/cells', root / 'models/signals'
    if not (cells / 'fish-model.json').exists():
        cells = root / 'artifacts/balanced-fish/training/neural-train-20260918-151950-694838/model'
    if not (signals / 'signal-model.json').exists():
        signals = root / 'artifacts/signal-fish/training/signal-train-20260919-020312-268747/model'
    return cells, signals


def check_models(cells, signals):
    manifest = read_model(cells)
    info = json.loads((Path(signals) / 'signal-model.json').read_text(encoding='utf-8'))
    if info.get('engine') != 'signal-cnn3d' or info.get('version') != 1:
        raise ValueError('Choose a trained FISH signal model folder.')
    SignalConfig(**info['config']).validate()
    if not (Path(signals) / info['checkpoint']).is_file():
        raise ValueError('The signal model checkpoint is missing.')
    return manifest


class AnalysisController(QtCore.QObject):
    progress = QtCore.Signal(object)
    completed = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()
    busy_changed = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.job = None
        self.busy = False
        self.stopping = False
        self.closing = False

    def start(self, image, output, cells, signals, device='auto'):
        if self.busy:
            raise ValueError('A scan is already being processed.')
        manifest = check_models(cells, signals)
        self.image = Path(image).expanduser().resolve()
        if not self.image.is_file():
            raise FileNotFoundError(self.image)
        if device not in ('auto', 'cpu', 'gpu'):
            raise ValueError('Unknown processing device.')
        self.device = device
        self.signal_model = str(Path(signals).expanduser().resolve())
        self.directory = Path(output).expanduser().resolve() / f'analysis-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}'
        self.directory.mkdir(parents=True)
        self.cancel_file = self.directory / 'cancel'
        self.stopping = False
        self.state = dict(version=1, engine='fish-analysis', status='running', source=str(self.image),
                          cell_model=str(Path(cells).resolve()), signal_model=self.signal_model, device=device,
                          processed_time=0, stages={})
        self._persist()
        config = replace(evidence_preset(NeuralConfig.from_dict(manifest['config'])), device=device)
        request = dict(config=asdict(config), mode='apply', model=str(Path(cells).resolve()),
                       output=str(self.directory / 'work'), cancel_file=str(self.cancel_file),
                       entries=[dict(image=str(self.image), scene=None)])
        request_path = self.directory / 'cells-request.json'
        request_path.write_text(json.dumps(request, indent=2), encoding='utf-8')
        self.busy = True
        self.busy_changed.emit(True)
        self._launch(0, ['-m', 'fish_viewer.neural', '--request', str(request_path)])

    def _persist(self):
        target = self.directory / 'analysis-result.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.state, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(target)

    def _launch(self, index, arguments):
        self.stage_index = index
        self.stage = STAGES[index][0]
        self.buffer, self.errors, self.finished_report = b'', '', None
        self.state['current_stage'] = self.stage
        self._persist()
        self.progress.emit(dict(stage=index+1, stages=5, phase=STAGES[index][1], current=0, total=0))
        process = QtCore.QProcess(self)
        self.job = process
        environment = QtCore.QProcessEnvironment.systemEnvironment()
        environment.insert('PYTHONUNBUFFERED', '1')
        environment.insert('PYTHONPATH', str(Path(__file__).resolve().parents[2]) + os.pathsep + os.environ.get('PYTHONPATH', ''))
        environment.insert('OMP_NUM_THREADS', '4')
        environment.insert('TF_CPP_MIN_LOG_LEVEL', '2')
        environment.insert('MPLCONFIGDIR', str(self.directory / 'matplotlib'))
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self._read)
        process.readyReadStandardError.connect(self._read_errors)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(lambda error: self._finished(-1) if error == QtCore.QProcess.ProcessError.FailedToStart else None)
        process.start(sys.executable, arguments)

    def _read(self):
        if self.job is None:
            return
        self.buffer += bytes(self.job.readAllStandardOutput())
        while b'\n' in self.buffer:
            line, self.buffer = self.buffer.split(b'\n', 1)
            try:
                event = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if event.get('kind') == 'finished':
                self.finished_report = event['report']
            elif event.get('kind') in ('progress', 'tile', 'file', 'stage', 'warning'):
                phase = event.get('phase') or event.get('message') or (
                    'Predicting image tiles' if event['kind'] == 'tile' else STAGES[self.stage_index][1])
                self.progress.emit(dict(stage=self.stage_index+1, stages=5, phase=phase,
                                        current=event.get('current', 0), total=event.get('total', 0)))

    def _read_errors(self):
        if self.job is not None:
            text = bytes(self.job.readAllStandardError()).decode(errors='replace')
            self.errors = (self.errors + text)[-8000:]
            with (self.directory / f'{self.stage}.log').open('a', encoding='utf-8') as stream:
                stream.write(text)

    def _finished(self, code=0, *_):
        if self.job is None:
            return
        self._read()
        self._read_errors()
        process, self.job = self.job, None
        process.deleteLater()
        report = self.finished_report or {}
        if self.closing:
            return
        # If publication finished just before cancellation, preserve and display
        # that complete result instead of falsely reporting it as cancelled.
        published = self.stage == 'export' and code == 0 and report.get('status') == 'complete'
        if (self.stopping and not published) or report.get('status') == 'cancelled':
            self.state['status'] = 'cancelled'
            self._stop()
            self.cancelled.emit()
            return
        if code != 0 or report.get('status') != 'complete':
            self.state.update(status='failed', error=report.get('error') or self.errors or 'The processing worker failed to start or exited unexpectedly.')
            self._stop()
            self.failed.emit(self.state['error'])
            return
        try:
            self.state['stages'][self.stage] = report.get('directory', report.get('image'))
            if self.stage == 'cells':
                if len(report.get('files', [])) != 1 or report.get('failures'):
                    raise ValueError('Cell detection did not produce a complete scan result.')
                args = ['-m', 'fish_viewer.signals', '--mode', 'apply', '--image', str(self.image),
                        '--cell-predictions', report['files'][0]['predictions'], '--model', self.signal_model,
                        '--output', str(self.directory / 'work'), '--device', self.device,
                        '--cancel-file', str(self.cancel_file)]
                self._launch(1, args)
            elif self.stage in ('signals', 'filter'):
                operation = 'filter' if self.stage == 'signals' else 'measure'
                self._launch(self.stage_index+1, ['-m', 'fish_viewer.workflow', operation,
                    '--report', str(Path(report['directory']) / 'report.json'),
                    '--output', str(self.directory / 'work'), '--cancel-file', str(self.cancel_file)])
            elif self.stage == 'measure':
                self.state['signal_report'] = str((Path(report['directory']) / 'report.json').relative_to(self.directory))
                target = self.directory / 'export' / (self.image.stem + '_measured.ims')
                self._launch(4, ['-m', 'fish_viewer.workflow', 'export', '--report', str(Path(report['directory']) / 'report.json'),
                    '--output', str(target), '--cancel-file', str(self.cancel_file)])
            else:
                self.state.update(status='complete', cells=report['cells'], signals=report['signals'],
                                  measurements=report['measurements'], image_sha256=report['image_sha256'],
                                  image=str(Path(report['image']).relative_to(self.directory)),
                                  scene=str(Path(report['scene']).relative_to(self.directory)),
                                  files=[str(Path(p).relative_to(self.directory)) for p in report['files']])
                self._stop()
                self.completed.emit(dict(self.state, report_path=str(self.directory / 'analysis-result.json')))
        except Exception as exc:
            self.state.update(status='failed', error=str(exc))
            self._stop()
            self.failed.emit(str(exc))

    def _stop(self):
        self._persist()
        self.busy = False
        self.busy_changed.emit(False)

    def cancel(self):
        if self.busy:
            if self.stopping and self.job is not None:
                self.job.kill()
            else:
                self.stopping = True
                self.cancel_file.touch()

    def shutdown(self):
        if self.job is not None:
            self.closing = True
            self.cancel_file.touch()
            process = self.job
            if not process.waitForFinished(5000):
                process.kill()
                process.waitForFinished(1500)
            self.state['status'] = 'cancelled'
            self._persist()
            self.busy = False
