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


STAGES = [('cells', 'Detecting cell contours'), ('signals', 'Finding signals inside cells'),
          ('filter', 'Removing cells without signals'), ('measure', 'Creating B points and distance lines'),
          ('export', 'Saving Imaris files and tables')]


def default_models():
    root = Path(__file__).resolve().parents[2]
    cells, signals = root / 'models/cells', root / 'models/signals-center-unet'
    if not (cells / 'fish-model.json').exists():
        cells = root / 'artifacts/balanced-fish/training/neural-train-20260918-151950-694838/model'
    if not (signals / 'signal-model.json').exists():
        signals = root / 'artifacts/signal-fish/training/signal-train-20260919-020312-268747/model'
    return cells, signals


def default_refinement_model():
    root = Path(__file__).resolve().parents[2] / 'models'
    for name in ('refinement-bias-candidate', 'refinement'):
        path = root / name
        if (path / 'refinement-model.json').is_file():
            return str(path)
    return ''


def default_proposal_model():
    path = Path(__file__).resolve().parents[2] / 'models/proposals'
    return str(path) if (path / 'proposal-model.json').is_file() else ''


def migrate_bundled_defaults(settings):
    """Upgrade former bundled defaults once, preserving custom model choices."""
    if settings.value('default_pipeline_version', '') == '0.4.1':
        return
    root = Path(__file__).resolve().parents[2]
    for key, old, new in (
        ('signal_model', root / 'models/signals', default_models()[1]),
        ('cell_model', root / 'models/cells', default_proposal_model()),
        ('boundary_model', root / 'models/refinement', default_refinement_model()),
    ):
        value = settings.value(key, '')
        if value and new and Path(value).resolve() == old.resolve():
            settings.setValue(key, str(new))
            if key == 'cell_model' and not settings.value('boundary_model', ''):
                settings.setValue('boundary_model', default_refinement_model())
    settings.setValue('default_pipeline_version', '0.4.1')


def check_models(cells, signals):
    if (Path(cells) / 'proposal-model.json').is_file():
        from ..proposals.config import read_model as read_proposal_model
        manifest = read_proposal_model(cells)
    else:
        manifest = read_model(cells)
    from ..signals.model import read_model as read_signal_model
    read_signal_model(signals)
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

    def start(self, image, output, cells, signals, device='auto', refinement=None, membrane_snap=None, recovery=None):
        if self.busy:
            raise ValueError('A scan is already being processed.')
        manifest = check_models(cells, signals)
        self.refinement_model = str(Path(refinement).expanduser().resolve()) if refinement else None
        self.center_box_detector = manifest.get('engine') == 'cell-center-box3d'
        if self.center_box_detector and not self.refinement_model:
            raise ValueError('The center-and-box detector requires a Boundaries model to draw crop contours.')
        self.recovery_config = None
        if recovery is not None:
            from ..proposals.recovery import RecoveryConfig
            self.recovery_config = asdict(RecoveryConfig(**recovery).validate())
            if not self.center_box_detector:
                raise ValueError('Cell recovery requires the center and crop models.')
            from ..spots.runtime import DIRECT_ENGINES
            from ..signals.model import read_model as read_signal_model
            if self.recovery_config['signal_guided'] and read_signal_model(signals)['engine'] not in DIRECT_ENGINES:
                raise ValueError('Signal-guided recovery requires a center U-Net or Spotiflow signal model.')
        self.stages = list(STAGES)
        if self.recovery_config and self.recovery_config['signal_guided']:
            self.stages[1:1] = [('guide_signals', 'Searching uncertain cells for confident signals'),
                               ('recover', 'Recovering signal-supported contours')]
        self.membrane_snap = None
        if membrane_snap is not None:
            from ..refinement.membrane import MembraneConfig
            self.membrane_snap = asdict(MembraneConfig(**membrane_snap).validate())
        if self.refinement_model:
            from ..refinement.config import read_model as read_refinement_model
            read_refinement_model(self.refinement_model)
            if not self.center_box_detector:
                self.stages.insert(1, ('refine', 'Refining cell boundaries'))
            else:
                self.stages[0] = ('cells', 'Finding cell centers and drawing crop contours')
        if self.membrane_snap:
            self.stages.insert(next(i for i, s in enumerate(self.stages) if s[0] == 'signals'),
                               ('snap', 'Snapping contours to membrane peaks'))
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
        self.state['refinement_model'] = self.refinement_model
        self.state['cell_detector_engine'] = manifest.get('engine', 'stardist3d')
        self.state['membrane_snap'] = self.membrane_snap
        self.state['cell_recovery'] = self.recovery_config
        self._persist()
        if self.center_box_detector:
            self.busy = True
            self.busy_changed.emit(True)
            arguments = ['-m', 'fish_viewer.proposals', '--mode', 'apply',
                '--image', str(self.image), '--model', str(Path(cells).resolve()),
                '--boundary-model', self.refinement_model, '--output', str(self.directory / 'work'),
                '--device', self.device, '--cancel-file', str(self.cancel_file)]
            if self.recovery_config:
                self.recovery_path = self.directory / 'recovery-config.json'
                self.recovery_path.write_text(json.dumps(self.recovery_config), encoding='utf-8')
                arguments += ['--recovery-config', str(self.recovery_path)]
            self._launch(0, arguments)
            return
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
        self.stage = self.stages[index][0]
        self.buffer, self.errors, self.finished_report = b'', '', None
        self.state['current_stage'] = self.stage
        self._persist()
        self.progress.emit(dict(stage=index+1, stages=len(self.stages), stage_name=self.stages[index][1],
                                phase=self.stages[index][1], current=0, total=0))
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
                    'Predicting image tiles' if event['kind'] == 'tile' else self.stages[self.stage_index][1])
                self.progress.emit(dict(stage=self.stage_index+1, stages=len(self.stages), phase=phase,
                                        stage_name=self.stages[self.stage_index][1],
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
            if self.stage == 'guide_signals':
                self._launch(next(i for i,s in enumerate(self.stages) if s[0]=='recover'),
                    ['-m','fish_viewer.proposals','--mode','recover','--cells-report',
                     str(Path(self.state['stages']['cells'])/'report.json'),'--signal-report',
                     str(Path(report['directory'])/'report.json'),'--output',str(self.directory/'work'),
                     '--device',self.device,'--cancel-file',str(self.cancel_file)])
                return
            if self.stage in ('cells', 'refine', 'snap', 'recover'):
                if len(report.get('files', [])) != 1 or report.get('failures'):
                    raise ValueError('Cell detection did not produce a complete scan result.')
                if self.stage == 'cells' and self.refinement_model and not self.center_box_detector:
                    self._launch(1, ['-m', 'fish_viewer.refinement', '--mode', 'apply',
                        '--cells-report', str(Path(report['directory']) / 'report.json'),
                        '--model', self.refinement_model, '--output', str(self.directory / 'work'),
                        '--device', self.device, '--cancel-file', str(self.cancel_file)])
                    return
                if self.stage == 'refine':
                    self.state['refinement_summary'] = report['summary']
                if self.stage == 'cells' and self.recovery_config and self.recovery_config['signal_guided']:
                    self._launch(next(i for i,s in enumerate(self.stages) if s[0]=='guide_signals'),
                        ['-m','fish_viewer.signals','--mode','apply','--image',str(self.image),
                         '--cell-predictions',report['files'][0]['recovery_candidates'],'--model',self.signal_model,
                         '--output',str(self.directory/'work'),'--device',self.device,
                         '--cancel-file',str(self.cancel_file)])
                    return
                if self.stage == 'recover':
                    self.state['recovery_summary'] = report['summary']
                if self.stage != 'snap' and self.membrane_snap:
                    snap_config = self.directory / 'membrane-config.json'
                    snap_config.write_text(json.dumps(self.membrane_snap, indent=2), encoding='utf-8')
                    self._launch(next(i for i, s in enumerate(self.stages) if s[0] == 'snap'),
                        ['-m', 'fish_viewer.refinement.membrane',
                         '--cells-report', str(Path(report['directory']) / 'report.json'),
                         '--output', str(self.directory / 'work'),
                         '--config-file', str(snap_config),
                         '--cancel-file', str(self.cancel_file)])
                    return
                if self.stage == 'snap':
                    self.state['membrane_snap_summary'] = report['summary']
                args = ['-m', 'fish_viewer.signals', '--mode', 'apply', '--image', str(self.image),
                        '--cell-predictions', report['files'][0]['predictions'], '--model', self.signal_model,
                        '--output', str(self.directory / 'work'), '--device', self.device,
                        '--cancel-file', str(self.cancel_file)]
                self._launch(next(i for i,s in enumerate(self.stages) if s[0]=='signals'), args)
            elif self.stage in ('signals', 'filter'):
                operation = 'filter' if self.stage == 'signals' else 'measure'
                self._launch(self.stage_index+1, ['-m', 'fish_viewer.workflow', operation,
                    '--report', str(Path(report['directory']) / 'report.json'),
                    '--output', str(self.directory / 'work'), '--cancel-file', str(self.cancel_file)])
            elif self.stage == 'measure':
                self.state['signal_report'] = str((Path(report['directory']) / 'report.json').relative_to(self.directory))
                target = self.directory / 'export' / (self.image.stem + '_measured.ims')
                self._launch(len(self.stages)-1, ['-m', 'fish_viewer.workflow', 'export', '--report', str(Path(report['directory']) / 'report.json'),
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
