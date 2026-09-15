"""Qt controls and JSON event transport; no TensorFlow import in the viewer."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..pipeline.gui import PipelinePanel
from ..pipeline.runner import _json
from .config import NeuralConfig


class NeuralPanel(PipelinePanel):
    def __init__(self, host):
        self.neural_controls = {}
        self._base_config = NeuralConfig()
        self._buffer = b''
        self._finished_report = None
        self._run_directory = None
        self._cancel_requested = False
        super().__init__(host)
        self.mode.clear()
        self.mode.addItems(['Train StarDist 3D + validate + test', 'Apply a saved StarDist model'])
        self.layout().itemAt(0).layout().itemAt(1).widget().setText('Epochs')
        self.trials.setRange(1,10000)
        self.trials.setValue(self._base_config.epochs)
        self.output.setText(str(Path.cwd()/'neural_runs'))
        self.status.setText('Ready. StarDist 3D learns from manual cells; unannotated voxels are ignored.')
        self.results.horizontalHeaderItem(4).setText('Dice: all refs')
        self.set_config(NeuralConfig())

    def _setup_page(self):
        super()._setup_page()
        form = self.controls['voxel_um'].parentWidget().layout()
        for key in ('threshold','smooth_um','seed_distance_um','closing_um'):
            form.setRowVisible(self.controls[key],False)
        form.setRowVisible(self.method,False)
        form.setRowVisible(self.holdout,False)
        self.exclude_border.setChecked(False)
        self.model_path = QtWidgets.QLineEdit()
        choose = QtWidgets.QPushButton('Choose model…')
        choose.clicked.connect(self.choose_model)
        row = QtWidgets.QHBoxLayout();row.addWidget(self.model_path);row.addWidget(choose)
        form.insertRow(0,'Saved model for application',row)
        self.device = QtWidgets.QComboBox();self.device.addItems(['auto','gpu','cpu'])
        form.insertRow(1,'Training / inference device',self.device)
        for key,label,low,high,step in [
            ('steps_per_epoch','Batches per epoch',1,10000,1),('batch_size','Batch size',1,16,1),
            ('patch_z','Patch depth (voxels)',16,128,16),('patch_xy','Patch width / height',16,256,16),
            ('patience','Early-stop patience (epochs)',1,1000,1),
            ('validate_every','Measure contours every N epochs',1,1000,1)]:
            control=QtWidgets.QSpinBox();control.setRange(low,high);control.setSingleStep(step)
            self.neural_controls[key]=control;form.addRow(label,control)
        for key,label,low,high,step in [
            ('learning_rate','Learning rate',1e-7,.01,.0001),
            ('probability_threshold','Cell probability threshold',.05,.95,.05),
            ('nms_threshold','Object overlap threshold',.05,.9,.05)]:
            control=QtWidgets.QDoubleSpinBox();control.setDecimals(7);control.setRange(low,high);control.setSingleStep(step)
            self.neural_controls[key]=control;form.addRow(label,control)
        info=QtWidgets.QLabel('Three acquisition groups minimum. Complete disjoint manual cells supervise training. '
            'Unknown voxels are ignored. A saved model supplies its preprocessing and contour settings in Apply mode.')
        info.setWordWrap(True);form.addRow(info)

    def config(self):
        keys=('voxel_um','min_diameter_um','max_diameter_um','min_sphericity','max_elongation')
        values={k:self.controls[k].value() for k in keys}
        values.update({k:v.value() for k,v in self.neural_controls.items()})
        return replace(self._base_config,**values,exclude_border=self.exclude_border.isChecked(),
            epochs=self.trials.value(),device=self.device.currentText()).validate()

    def set_config(self, config):
        if not isinstance(config,NeuralConfig):
            config=NeuralConfig()
        self._base_config=config
        for key,control in self.controls.items():
            if hasattr(config,key):control.setValue(getattr(config,key))
        for key,control in self.neural_controls.items():control.setValue(getattr(config,key))
        self.exclude_border.setChecked(config.exclude_border)
        self.trials.setValue(config.epochs);self.device.setCurrentText(config.device)

    def choose_model(self):
        path=QtWidgets.QFileDialog.getExistingDirectory(self,'Select StarDist model folder',self.model_path.text())
        if path:self.model_path.setText(path)

    def load_settings(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Load StarDist settings','','JSON (*.json)')
        if path:
            try:
                settings=json.loads(Path(path).read_text())
                if settings.get('engine')!='stardist3d':raise ValueError('Not StarDist settings.')
                self.set_config(NeuralConfig.from_dict(settings['config']))
            except Exception as exc:self.show_error(str(exc))

    def save_settings(self):
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,'Save StarDist settings','neural-settings.json','JSON (*.json)')
        if path:
            try:_json(Path(path),dict(version=1,engine='stardist3d',config=asdict(self.config())))
            except Exception as exc:self.show_error(str(exc))

    def start(self):
        if self.job:return
        try:
            config=self.config();entries=self.checked_entries()
            if not entries:raise ValueError('Select input images.')
            if not self.output.text().strip():raise ValueError('Choose an output folder.')
            mode='train' if self.mode.currentIndex()==0 else 'apply'
            if mode=='apply' and not (Path(self.model_path.text())/'fish-model.json').is_file():
                raise ValueError('Select a trained StarDist model folder containing fish-model.json.')
            job_dir=Path(self.output.text()).resolve()/'.jobs'/uuid.uuid4().hex
            job_dir.mkdir(parents=True)
            self.cancel_file=job_dir/'cancel'
            request=dict(config=asdict(config),mode=mode,output=str(Path(self.output.text()).resolve()),
                model=self.model_path.text() or None,cancel_file=str(self.cancel_file),
                entries=[dict(image=str(e.image.resolve()),scene=str(e.scene.resolve()) if e.scene else None) for e in entries])
            _json(job_dir/'request.json',request)
        except Exception as exc:
            self.show_error(str(exc));return
        self.history=[];self.final_scores=None;self.report=None;self.previews={};self.channel_scores=[];self.timings={}
        self.results.setRowCount(0);self.log.clear();self.summary.setText('Neural run in progress.')
        self._buffer=b'';self._finished_report=None;self._run_directory=None;self._cancel_requested=False
        self._last_stderr=''
        self.draw_scores();self.draw_preprocessing()
        self.job=QtCore.QProcess(self)
        environment=QtCore.QProcessEnvironment.systemEnvironment()
        environment.insert('PYTHONUNBUFFERED','1');environment.insert('OMP_NUM_THREADS','4')
        environment.insert('MPLCONFIGDIR',str(job_dir/'matplotlib'))
        root=Path(__file__).resolve().parents[2]
        environment.insert('PYTHONPATH',str(root))
        self.job.setProcessEnvironment(environment)
        self.job.readyReadStandardOutput.connect(self.read_events)
        self.job.readyReadStandardError.connect(self.read_errors)
        self.job.finished.connect(self.process_finished)
        self.job.errorOccurred.connect(self.process_error)
        self.set_running(True);self.cancel_button.setText('Cancel')
        self.started=time.monotonic();self.timer.start();self.tabs.setCurrentIndex(1)
        self.status.setText('Starting neural worker…')
        self.job.start(sys.executable,['-m','fish_viewer.neural','--request',str(job_dir/'request.json')])

    def read_events(self):
        if not self.job:return
        self._buffer+=bytes(self.job.readAllStandardOutput())
        while b'\n' in self._buffer:
            line,self._buffer=self._buffer.split(b'\n',1)
            try:
                event=json.loads(line)
            except (ValueError,UnicodeDecodeError):
                if line.strip():self.log.appendPlainText(line.decode(errors='replace')[-500:])
                continue
            if event['kind']=='finished':self._finished_report=event['report']
            else:self.handle_event(event)

    def read_errors(self):
        if self.job:
            text=bytes(self.job.readAllStandardError()).decode(errors='replace')
            self._last_stderr=(self._last_stderr+text)[-6000:]
            # Keep full library diagnostics on disk without flooding live graphs.
            if hasattr(self,'cancel_file'):
                with self.cancel_file.with_name('worker.log').open('a',encoding='utf-8') as stream:stream.write(text)

    def process_error(self, error):
        if error==QtCore.QProcess.ProcessError.FailedToStart:
            self.process_finished(-1,QtCore.QProcess.ExitStatus.CrashExit)

    def process_finished(self, exit_code, status):
        if not self.job:return
        self.read_events();self.read_errors()
        report=self._finished_report
        if report is None and self._run_directory:
            path=Path(self._run_directory)/'report.json'
            if path.exists():
                report=json.loads(path.read_text())
                report['status']='cancelled' if self._cancel_requested else 'failed'
                report['elapsed_s']=time.monotonic()-self.started
                report['error']=f'Worker exited before completion (code {exit_code}). '+self._last_stderr[-1200:]
                _json(path,report)
        old=self.job;self.job=None;old.deleteLater()
        self.cancel_button.setText('Cancel')
        if self._shutdown:return
        if report is not None:self.completed(report)
        else:self.failed('Neural worker could not start or finish. '+self._last_stderr[-1200:])

    def cancel(self):
        if self.job:
            if self._cancel_requested:
                self.job.kill()
            else:
                self._cancel_requested=True;self.cancel_file.touch()
                self.status.setText('Stopping after the current operation; saved checkpoints will remain.')
                self.cancel_button.setText('Stop now')

    def handle_event(self,event):
        if self._shutdown:return
        kind=event['kind']
        if kind=='run_started':self._run_directory=event['directory']
        elif kind=='runtime':self.log.appendPlainText('TensorFlow device: '+event['device'])
        elif kind=='neural_split':
            for name,paths in event['split'].items():self.log.appendPlainText(name.capitalize()+':\n'+'\n'.join(paths))
        elif kind=='audit':
            a=event['audit']
            self.log.appendPlainText(f'{Path(event["file"]).name}: {a["training_objects"]} training cells, '
                f'{a["cut_references"]} cut references, {a["background_voxels"]} reviewed background voxels.')
            self.previews['Candidate interiors']=np.maximum(np.asarray(event['preview']),0)
            self.draw_preprocessing()
        elif kind=='batch':
            self.progress.setValue(round(100*event['batch']/event['total']))
            self.status.setText(f'Epoch {len(self.history)+1}: batch {event["batch"]}/{event["total"]}, loss {event["loss"]:.4f}')
        elif kind=='epoch':
            self.history.append(event);self.draw_scores()
            self.status.setText(f'Epoch {event["epoch"]}: training loss {event["losses"]["loss"]:.4f}, validation {event["losses"]["val_loss"]:.4f}')
        elif kind=='calibration':
            m=event['validation']['complete_cells']
            self.log.appendPlainText(f'Validation threshold {event["probability_threshold"]:.2f}: complete-cell Dice {m["mean_dice"]:.3f}, '
                f'{m["predicted_count"]} candidates. Unmatched predictions remain unscored.')
        else:super().handle_event(event)

    def draw_scores(self):
        axes=self.score_canvas.axes
        for ax in axes:ax.clear();ax.set_xlabel('Epoch');ax.grid(alpha=.15)
        rows=self.history
        for key,label,color in [('loss','Training','#56dcc1'),('val_loss','Validation','#84bafa')]:
            axes[0].plot([r['epoch'] for r in rows],[r['losses'].get(key,np.nan) for r in rows],label=label,color=color)
        axes[0].set_title('StarDist total loss: probability + shape')
        specs=[('mean_dice','Validation overlap: complete cells'),('detection_rate','Detected complete cells (IoU ≥ 0.5)'),
               ('boundary_rmse_um','Matched complete-cell boundary RMSE (µm)')]
        scored=[r for r in rows if 'validation' in r]
        for ax,(key,title) in zip(axes[1:],specs):
            ax.set_title(title)
            ax.plot([r['epoch'] for r in scored],[r['validation']['complete_cells'].get(key) for r in scored],
                'o-',color='#56dcc1',label='Validation',markersize=3)
            if key=='mean_dice':
                ax.plot([r['epoch'] for r in scored],[r['validation']['complete_cells'].get('mean_iou') for r in scored],
                    color='#84bafa',label='IoU')
            selected=(self.report or {}).get('calibrated_validation',{}).get('complete_cells',{}).get(key)
            if selected is not None:
                ax.scatter([(self.report or {}).get('best_epoch',0)],[selected],marker='D',s=45,
                    color='#d4a0ee',label='Selected model / threshold')
            final=(self.final_scores or {}).get('complete_cells',{}).get(key)
            if final is not None:ax.scatter([max([r['epoch'] for r in rows],default=0)+1],[final],
                marker='*',s=90,color='#edb867',label='Test: final only')
            if key!='boundary_rmse_um':ax.set_ylim(-.02,1.02)
        for ax in axes:ax.legend(fontsize=7,facecolor='#17232f',labelcolor='#e0e8f1',edgecolor='#405366')
        self.score_canvas.style();self.score_canvas.draw_idle()

    def draw_preprocessing(self):
        super().draw_preprocessing()
        self.preprocess_canvas.axes[1].set_title('Normalized network input')
        self.preprocess_canvas.axes[2].set_title('Manual training cells; unknown ignored')
        self.preprocess_canvas.draw_idle()

    def completed(self,report):
        self.history=report.get('epochs',[])
        super().completed(report)
        if report.get('model_directory'):self.model_path.setText(report['model_directory'])
        if report.get('config'):self.set_config(NeuralConfig.from_dict(report['config']))
        complete=report.get('evaluation',{}).get('complete_cells',{})
        if complete.get('mean_dice') is not None:
            self.summary.setText(self.summary.text()+f'\nComplete test cells: Dice {complete["mean_dice"]:.3f}, '
                f'detected {complete["detection_rate"]:.1%}; best epoch {report.get("best_epoch")}.')

    def load_report(self):
        if self.job:return
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,'Open StarDist run',self.output.text(),'JSON (*.json)')
        if path:
            try:self.display_report(json.loads(Path(path).read_text()))
            except Exception as exc:self.show_error(str(exc))

    def display_report(self,report):
        if report.get('version')!=1 or report.get('engine')!='stardist3d':raise ValueError('Not a supported StarDist run.')
        self.results.setRowCount(0);self.log.clear()
        for row in report['files']:self.add_result(row)
        self.completed(report)

    def shutdown(self):
        self._shutdown=True;self.timer.stop()
        if self.job:
            process=self.job
            self._cancel_requested=True;self.cancel_file.touch()
            if not process.waitForFinished(5000):
                process.kill();process.waitForFinished(1500)
