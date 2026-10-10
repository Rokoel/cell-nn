"""StarDist runner. Only this optional worker imports TensorFlow."""
from dataclasses import asdict, replace
import csv
import hashlib
import importlib.metadata
from itertools import product
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

from ..pipeline.evaluation import aggregate, evaluate
from ..pipeline.runner import _json, save_prediction
from ..pipeline.segmentation import Cancelled, Prediction, checkpoint, mask_properties
from ..run_paths import new_run_directory
from .config import NeuralConfig
from .data import load_cases, three_way_split, training_arrays, subset_metrics, add_synthetic_background
from .precision import membrane_evidence,evidence_passes


def runtime(device='auto', seed=42):
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    try:
        import tensorflow as tf
        from stardist.models import StarDist3D, Config3D
    except ImportError as exc:
        raise RuntimeError('StarDist training dependencies are missing. Install the project with the [neural] extra; '
                           'see docs/neural-training.md.') from exc
    devices = tf.config.list_physical_devices('GPU')
    if device == 'gpu' and not devices:
        raise RuntimeError('GPU requested but TensorFlow cannot access CUDA. Check the driver/runtime or select CPU.')
    for gpu in devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.utils.set_random_seed(seed)
    return tf, StarDist3D, Config3D, dict(device='GPU' if devices else 'CPU',
        devices=[str(d) for d in devices], versions={p:importlib.metadata.version(p)
        for p in ('stardist','tensorflow','keras','numpy','scipy')})


def filter_instances(labels, prepared, config):
    result = np.zeros(labels.shape, np.uint32)
    rejected = dict(size=0, border=0, elongation=0, sphericity=0, membrane_evidence=0)
    cells = []
    for i, box in enumerate(ndi.find_objects(labels), 1):
        if box is None:
            continue
        mask = labels[box] == i
        volume = int(mask.sum())*float(np.prod(prepared.spacing))
        diameter = (6*volume/np.pi)**(1/3)
        upper=min(config.max_diameter_um,config.precision_max_diameter_um) if config.precision_mode else config.max_diameter_um
        if not config.min_diameter_um <= diameter <= upper:
            rejected['size'] += 1
            continue
        cut = any(s.start==0 or s.stop==n for s,n in zip(box, labels.shape))
        if config.exclude_border and cut:
            rejected['border'] += 1
            continue
        props = mask_properties(mask, np.asarray(prepared.spacing))
        if config.use_shape_filter and props['elongation'] > config.max_elongation:
            rejected['elongation'] += 1
            continue
        if config.use_shape_filter and props['sphericity'] < config.min_sphericity:
            rejected['sphericity'] += 1
            continue
        if config.precision_mode:
            evidence=membrane_evidence(mask,box,prepared,config)
            if not evidence_passes(evidence,config):
                rejected['membrane_evidence']+=1
                continue
            props.update(evidence)
        number = len(cells)+1
        result[box][mask] = number
        center = (np.argwhere(mask).mean(0)+[s.start for s in box])*prepared.spacing
        props.update(label=number, center_xyz=(center[::-1]+prepared.origin).tolist())
        cells.append(props)
    return Prediction(result, cells, {}, {}, rejected)


def proposals(model,case,config,cancel=None,emit=None):
    """Keep sparse centers/rays; never assemble a whole-field dense ray tensor."""
    shape=np.asarray(case.prepared.image.shape)
    grid=np.asarray(model.config.grid)
    # A 32-voxel halo supplies context; disjoint cores own each proposal once.
    core=np.array((64,128,128));halo=np.array((32,32,32))
    chunks=[];block_capped=False
    blocks=list(product(*(range(0,int(n),int(c)) for n,c in zip(shape,core))))
    for block_number,start in enumerate(blocks,1):
        checkpoint(cancel)
        if emit:emit(dict(kind='tile',current=block_number,total=len(blocks),file=str(case.entry.image)))
        start=np.asarray(start);stop=np.minimum(shape,start+core)
        low=np.maximum(0,start-halo);high=np.minimum(shape,stop+halo)
        block=tuple(slice(int(a),int(b)) for a,b in zip(low,high))
        probability,distances=model.predict(np.asarray(case.prepared.image[block]),axes='ZYX',
            n_tiles=(1,1,1),show_tile_progress=False,verbose=0)
        if not np.isfinite(probability).all() or not np.isfinite(distances).all():
            raise ValueError('Network produced non-finite predictions.')
        peaks=peak_local_max(probability,min_distance=2,threshold_abs=config.probability_threshold,
            exclude_border=False,num_peaks=4000)
        block_capped=block_capped or len(peaks)==4000
        if not len(peaks):continue
        points=peaks*grid+low
        radial=distances[tuple(peaks.T)]
        valid=np.all((points>=start)&(points<stop),axis=1)&(radial.min(1)>=.25)&(radial.max(1)<=2*config.max_diameter_um/min(case.prepared.spacing))
        chunks.append((points[valid],probability[tuple(peaks[valid].T)],radial[valid]))
    if not chunks:return np.empty((0,3),int),np.empty(0),np.empty((0,config.rays)),False
    points,probability,distances=[np.concatenate([chunk[i] for chunk in chunks]) for i in range(3)]
    capped=block_capped or len(points)>=4000
    if len(points)>4000:
        order=np.argsort(-probability,kind='stable')[:4000]
        points,probability,distances=points[order],probability[order],distances[order]
    return points,probability,distances,capped


def predict(model, case, config, cancel=None,emit=None):
    checkpoint(cancel)
    started = time.monotonic()
    peaks,probability,radial,capped=proposals(model,case,config,cancel,emit)
    if len(peaks):
        raw,_=model._instances_from_prediction(case.prepared.image.shape,
            probability,radial,points=peaks,
            nms_thresh=config.nms_threshold)
    else:
        raw=np.zeros(case.prepared.image.shape,np.uint32)
    checkpoint(cancel)
    prediction = filter_instances(raw, case.prepared, config)
    prediction.timings['neural_inference'] = time.monotonic()-started
    metrics = subset_metrics(prediction.labels, case, evaluate, cancel)
    metrics['before_shape_filter'] = subset_metrics(raw, case, evaluate, cancel)
    metrics['proposal_count']=int(len(peaks))
    metrics['proposal_limit_reached']=capped
    return prediction, metrics


def augment(x, y):
    turns = np.random.randint(4)
    x, y = np.rot90(x, turns, axes=(1,2)), np.rot90(y, turns, axes=(1,2))
    for axis in range(3):
        if np.random.random() < 0.5:
            x, y = np.flip(x,axis), np.flip(y,axis)
    x = x*np.random.uniform(0.8,1.2)+np.random.uniform(-0.04,0.04)
    x = np.clip(x + np.random.normal(0, 0.015, x.shape), 0, 1).astype(np.float32)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def aggregate_subsets(rows):
    result = aggregate(rows)
    for subset in ('complete_cells', 'cut_cells'):
        result[subset] = aggregate([r[subset] for r in rows])
    return result


def run_neural(entries, config, output_root, mode='train', model_directory=None, emit=None, cancel=None, audit_path=None, initial_model_directory=None):
    config.validate()
    if not entries or mode not in ('train','apply'):
        raise ValueError('Choose input files and train or apply mode.')
    directory = new_run_directory(output_root, f'neural-{mode}')
    os.environ.setdefault('MPLCONFIGDIR', str(directory/'matplotlib-cache'))
    started = time.monotonic()
    report = dict(version=1, engine='stardist3d', directory=str(directory), mode=mode, status='running',
        config=asdict(config), initial_config=asdict(config), epochs=[], files=[], warnings=[], failures=[], split={},
        model_directory=None, best_epoch=None, elapsed_s=0,
        metric_notes='Incomplete references: unknown predictions unscored. Complete and cut cells reported separately. '
        'Validation loss selects the checkpoint; test predictions run only after selection. '
        'This reference test acquisition was inspected in earlier classical experiments; results are exploratory.')
    last_progress_write=0.
    def notify(event):
        nonlocal last_progress_write
        checkpoint(cancel)
        if event.get('kind')=='progress':
            previous=report.get('progress',{})
            report['progress']={k:v for k,v in event.items() if k!='kind'}
            if time.monotonic()-last_progress_write>=1 or previous.get('phase')!=event['phase']:
                persist();last_progress_write=time.monotonic()
        if event.get('kind') == 'warning':
            report['warnings'].append(event['message'])
        if emit:
            emit(event)
        if event.get('kind')=='file' and (event['phase'].startswith('Preparing') or event['phase']=='Preprocessing'):
            progress('Preprocessing',event['current']-1,event['total'],file=event['file'])
        if event.get('kind')=='tile':
            report.setdefault('progress',{}).update(tile=event['current'],tiles=event['total'])
    def progress(phase,current,total,**extra):
        notify(dict(kind='progress',phase=phase,current=current,total=total,**extra))
    def persist():
        report['elapsed_s'] = time.monotonic()-started
        _json(directory/'report.json', report)
    persist()
    if emit:
        emit(dict(kind='run_started', directory=str(directory)))
    model = None
    try:
        if mode == 'apply':
            if not model_directory:
                raise ValueError('Select a trained model folder for prediction.')
            model_directory = Path(model_directory).resolve()
            manifest = json.loads((model_directory/'fish-model.json').read_text())
            if manifest.get('version') != 1 or manifest.get('engine') != 'stardist3d':
                raise ValueError('Unsupported neural model manifest.')
            # Use exactly the preprocessing and geometry policy saved with training.
            from .preprocessing import validate_manifest
            validate_manifest(manifest)
            requested=config
            config = replace(NeuralConfig.from_dict(manifest['config']), device=requested.device, cell_channel=requested.cell_channel)
            if requested.precision_mode:
                fields=('precision_mode','precision_max_diameter_um','membrane_support','membrane_contrast','membrane_intensity',
                    'probability_threshold','nms_threshold','min_sphericity','max_elongation','exclude_border','use_shape_filter')
                config=replace(config,**{key:getattr(requested,key) for key in fields},calibrate_probability=False)
            report['config'] = asdict(config)
        if audit_path:
            from .audit import load_audit
            cases=load_audit(audit_path,config,directory,notify,cancel,training=mode=='train')
            report['audit_path']=str(Path(audit_path).resolve())
        else:
            cases = load_cases(entries, config, directory, notify, cancel)
        if not cases:
            raise ValueError('No usable image datasets were found.')
        report['data_audit'] = [c.audit for c in cases]
        if mode == 'train':
            if audit_path:
                training,validation,testing=([c for c in cases if c.audit['split']==split] for split in ('training','validation','test'))
            else:
                training, validation, testing = three_way_split(cases)
            validation_preview=[c for c in validation if c.references]
            if config.validation_images and len(validation_preview)>config.validation_images:
                validation_preview=[validation_preview[i] for i in np.linspace(0,len(validation_preview)-1,config.validation_images,dtype=int)]
            report['validation_preview_images']=[str(c.entry.image) for c in validation_preview]
            report['split'] = {k:[str(c.entry.image) for c in v] for k,v in
                [('training',training),('validation',validation),('test',testing)]}
            notify(dict(kind='neural_split', split=report['split']))
            if not sum(c.audit['background_voxels'] for c in training):
                notify(dict(kind='warning', message='No reviewed background examples: pilot uses sparse object-only supervision. '
                    'Unannotated cells are ignored, not treated as background; false-positive control is not established.'))
            if sum(c.audit['training_objects'] for c in training) < 50:
                notify(dict(kind='warning', message='Fewer than 50 complete training cells: this is a small exploratory pilot.'))
        notify(dict(kind='stage', stage='Initializing StarDist / TensorFlow', progress=0))
        progress('Initializing neural network',0,1)
        tf, StarDist3D, Config3D, device_info = runtime(config.device, config.seed)
        report['runtime'] = device_info
        notify(dict(kind='runtime', **device_info))
        checkpoint(cancel)
        if mode == 'train':
            model_directory = directory/'model'
            # Setting background_reg=0 is essential: StarDist's distance
            # regularizer would otherwise penalize predictions on ignored voxels.
            architecture = Config3D(n_rays=config.rays, grid=(1,2,2), n_channel_in=1,
                unet_n_depth=2, unet_n_filter_base=config.filters, net_conv_after_unet=config.filters*2,
                train_patch_size=(config.patch_z,config.patch_xy,config.patch_xy),
                train_batch_size=config.batch_size, train_learning_rate=config.learning_rate,
                train_background_reg=0, train_foreground_only=1, train_tensorboard=False,
                train_n_val_patches=8, train_sample_cache=False, use_gpu=False,
                train_checkpoint='best.weights.h5', train_checkpoint_last='last.weights.h5')
            model = StarDist3D(architecture, name='model', basedir=str(directory))
            initialization=None
            if initial_model_directory:
                initial=Path(initial_model_directory).resolve()
                initial_manifest=json.loads((initial/'fish-model.json').read_text())
                initial_config=NeuralConfig.from_dict(initial_manifest['config'])
                if initial_manifest.get('engine')!='stardist3d' or any(getattr(initial_config,k)!=getattr(config,k) for k in ('rays','filters')):
                    raise ValueError('Initial StarDist checkpoint architecture differs from the new model.')
                weights=initial/initial_manifest['checkpoint']
                model.load_weights(weights)
                with weights.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
                initialization=dict(model=str(initial),checkpoint_sha256=digest)
                report['initialization']=initialization
            model.prepare_for_training()
            # Eager kernels avoid an unnecessary XLA requirement on new GPUs.
            model.keras_model.jit_compile = False
            model_manifest = dict(version=1, engine='stardist3d', config=asdict(config),
                checkpoint='best.weights.h5', supervision='Sparse manual objects; unknown=-1; reviewed background=0',
                runtime=device_info)
            if initialization:model_manifest['initialization']=initialization
            if config.preprocessing != 'none':
                from .preprocessing import RECIPE
                model_manifest['preprocessing_recipe']=RECIPE
            _json(model_directory/'fish-model.json', model_manifest)
            report['model_directory'] = str(model_directory)
            best_loss = float('inf')
            stale = 0
            outer_model = model
            class Progress(tf.keras.callbacks.Callback):
                def on_epoch_begin(self,epoch,logs=None):
                    self.epoch=epoch+1
                    progress('Training',0,config.steps_per_epoch,epoch=self.epoch,epochs=config.epochs)

                def on_test_begin(self,logs=None):
                    self.validation_total=int(np.ceil((len(xv) if config.streaming_validation else 8)/config.batch_size))
                    progress('Validation loss',0,self.validation_total,epoch=self.epoch,epochs=config.epochs)

                def on_test_batch_end(self,batch,logs=None):
                    checkpoint(cancel)
                    if (batch+1)%5==0 or batch+1==self.validation_total:
                        notify(dict(kind='validation_batch',batch=batch+1,total=self.validation_total,epoch=self.epoch,epochs=config.epochs))
                        progress('Validation loss',batch+1,self.validation_total,epoch=self.epoch,epochs=config.epochs)

                def on_train_batch_end(self, batch, logs=None):
                    checkpoint(cancel)
                    if not np.isfinite(float(logs.get('loss',0))):
                        raise ValueError('Non-finite batch loss. Check patch supervision and learning rate; last complete checkpoints were retained.')
                    if (batch+1) % 5 == 0 or batch+1 == config.steps_per_epoch:
                        notify(dict(kind='batch', batch=batch+1, total=config.steps_per_epoch,
                            loss=float(logs.get('loss',0))))
                        progress('Training',batch+1,config.steps_per_epoch,epoch=self.epoch,epochs=config.epochs,loss=float(logs.get('loss',0)))

                def on_epoch_end(self, epoch, logs=None):
                    nonlocal best_loss, stale
                    checkpoint(cancel)
                    logs = {k:float(v) for k,v in (logs or {}).items()}
                    logs['learning_rate']=float(tf.keras.backend.get_value(outer_model.keras_model.optimizer.learning_rate))
                    if not all(np.isfinite(v) for v in logs.values()):
                        raise ValueError('Training produced a non-finite loss; inspect the supervision and reduce the learning rate.')
                    current = logs['val_loss']
                    if current < best_loss:
                        best_loss, stale = current, 0
                        report['best_epoch'] = epoch+1
                        outer_model.keras_model.save_weights(model_directory/'best.weights.h5')
                    else:
                        stale += 1
                    outer_model.keras_model.save_weights(model_directory/'last.weights.h5')
                    row = dict(epoch=epoch+1, losses=logs, best_val_loss=best_loss,
                        elapsed_s=time.monotonic()-started)
                    if (epoch+1) % config.validate_every == 0 or epoch == 0:
                        scores=[]
                        for number,c in enumerate(validation_preview,1):
                            progress('Validation contours',number-1,len(validation_preview),epoch=epoch+1,epochs=config.epochs,file=str(c.entry.image))
                            scores.append(predict(outer_model,c,config,cancel,notify)[1])
                        row['validation'] = aggregate_subsets(scores)
                    report['epochs'].append(row)
                    persist()
                    notify(dict(kind='epoch', **row))
                    if stale >= config.patience:
                        outer_model.keras_model.stop_training = True
                        report['stop_reason']='Validation loss did not improve within patience.'
            # Own checkpoints have a stable name across supported Keras versions.
            model.callbacks = [Progress(), tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', patience=max(2,config.patience//2), factor=0.5, min_lr=1e-6)]
            x, y = training_arrays(training, config)
            xv, yv = training_arrays(validation, config)
            if not len(x) or not len(xv):
                raise ValueError('Training and validation each need at least one complete cell within the selected shape limits.')
            x,y,synthetic_count=add_synthetic_background(x,y,config)
            report['patch_counts']=dict(training=len(x),validation=len(xv))
            report['patch_counts']['synthetic_background']=synthetic_count
            if synthetic_count:
                notify(dict(kind='warning',message=f'{synthetic_count} synthetic blank/noise crops supplement manual targets. '
                    'These test empty-image behavior; they are not reviewed real false-positive examples.'))
            persist()
            if config.streaming_validation:
                from stardist.models.model3d import StarDistData3D
                from stardist.rays3d import rays_from_json
                kwargs=dict(rays=rays_from_json(model.config.rays_json),grid=model.config.grid,
                    patch_size=model.config.train_patch_size,anisotropy=model.config.anisotropy,
                    use_gpu=False,foreground_prob=1,sample_ind_cache=False,
                    keras_kwargs=dict(workers=1,use_multiprocessing=False,max_queue_size=2))
                train_data=StarDistData3D(x,y,batch_size=config.batch_size,length=config.epochs*config.steps_per_epoch,
                    augmenter=augment,**kwargs)
                class OrderedValidation(StarDistData3D):
                    def batch(self,i):return np.arange(i*self.batch_size,min((i+1)*self.batch_size,self.data_size))
                val_data=OrderedValidation(xv,yv,batch_size=config.batch_size,
                    length=int(np.ceil(len(xv)/config.batch_size)),**kwargs)
                model.keras_model.fit(iter(train_data),validation_data=val_data,epochs=config.epochs,
                    steps_per_epoch=config.steps_per_epoch,callbacks=model.callbacks,verbose=0,shuffle=False)
            else:
                model.train(x, y, validation_data=(xv,yv), epochs=config.epochs,
                    steps_per_epoch=config.steps_per_epoch, augmenter=augment, seed=config.seed, workers=1)
            checkpoint(cancel)
            model.load_weights('best.weights.h5')
            report.setdefault('stop_reason','Maximum epochs reached.')
            if config.select_balanced_acceptance:
                from .selection import select_acceptance
                config,selection=select_acceptance(model,validation,config,notify,cancel)
                report['acceptance_selection']=selection
                report['config']=asdict(config)
                model_manifest.update(config=asdict(config),acceptance_selection=selection)
                _json(model_directory/'fish-model.json',model_manifest)
                _json(directory/'acceptance-selection.json',selection)
                persist()
            if config.calibrate_probability:
                # Standard dense-label threshold optimization would count all
                # unknown cells as false positives. Use our partial-reference
                # evaluator on validation images only, with a bounded sweep.
                report['calibration']=[]
                best_dice=-1.0
                selected=config
                thresholds=sorted(set([config.probability_threshold,.15,.25,.35,.45]),reverse=True)
                for index,threshold in enumerate(thresholds,1):
                    candidate=replace(config,probability_threshold=threshold)
                    notify(dict(kind='stage',stage=f'Validation probability threshold {threshold:.2f}',progress=100*index/len(thresholds)))
                    values=[]
                    for number,c in enumerate(validation,1):
                        progress('Threshold calibration',number-1,len(validation),threshold=threshold,file=str(c.entry.image))
                        values.append(predict(model,c,candidate,cancel,notify)[1])
                    scores=aggregate_subsets(values)
                    value=scores['complete_cells']['mean_dice']
                    row=dict(probability_threshold=threshold,validation=scores)
                    report['calibration'].append(row)
                    if value is not None and value>best_dice:
                        best_dice,selected=value,candidate
                    notify(dict(kind='calibration',**row))
                    persist()
                config=selected
                report['config']=asdict(config)
                model_manifest['config']=asdict(config)
                model_manifest['calibration']='Maximum complete-reference Dice on validation acquisitions; unknown objects unscored.'
                _json(model_directory/'fish-model.json',model_manifest)
                report['calibrated_validation']=next(r['validation'] for r in report['calibration'] if r['probability_threshold']==config.probability_threshold)
            model.thresholds = dict(prob=config.probability_threshold, nms=config.nms_threshold)
            _json(model_directory/'thresholds.json', model.thresholds._asdict())
        else:
            model = StarDist3D(None, name=model_directory.name, basedir=str(model_directory.parent))
            model.load_weights(manifest['checkpoint'])
            report['model_directory'] = str(model_directory)
            training, validation, testing = [], [], []
        test_scores = [];validation_scores=[]
        for index, case in enumerate(cases, 1):
            split = 'test' if case in testing else 'validation' if case in validation else 'training' if case in training else 'apply' if mode=='apply' else 'unscored'
            notify(dict(kind='file', file=str(case.entry.image), phase=f'Final {split}', current=index, total=len(cases),
                previews=case.prepared.previews, channel_scores=case.prepared.channel_scores))
            progress('Final prediction and evaluation',index-1,len(cases),split=split,file=str(case.entry.image))
            prediction, metrics = predict(model, case, config, cancel,notify)
            if metrics['proposal_limit_reached']:
                notify(dict(kind='warning', message=f'{case.entry.image.name}: reached the 4,000-proposal limit; '
                    'some candidate cells may be omitted. Review this field before using its contours.'))
            path = case.directory/'contours.npz'
            save_prediction(path, prediction, case.prepared, case.entry, config, metrics)
            row = dict(image=str(case.entry.image), scene=str(case.entry.scene) if case.entry.scene else None,
                predictions=str(path), split=split, channel=case.prepared.channel, metrics=metrics,
                rejected=prediction.rejected, timings=prediction.timings)
            report['files'].append(row)
            persist()
            notify(dict(kind='result', row=row))
            if split == 'test':
                test_scores.append(metrics)
            if split == 'validation':validation_scores.append(metrics)
            notify(dict(kind='stage', stage='Neural contours ready', progress=100,
                previews={'Accepted spherical cells':prediction.labels[len(prediction.labels)//2]}, timings=prediction.timings))
        if validation_scores:report['selected_validation']=aggregate_subsets(validation_scores)
        if test_scores:
            report['evaluation'] = aggregate_subsets(test_scores)
            notify(dict(kind='evaluation', scores=report['evaluation']))
        report['status'] = 'complete'
        progress('Complete',len(cases),len(cases),completed_epochs=len(report['epochs']),best_epoch=report.get('best_epoch'))
        _json(directory/'settings.json', dict(version=1, engine='stardist3d', config=asdict(config)))
    except Cancelled:
        report['status'] = 'cancelled'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = str(exc)
        report['traceback'] = traceback.format_exc()
    finally:
        persist()
        write_outputs(directory, report)
    if emit:
        emit(dict(kind='finished', report=report))
    return report


def write_outputs(directory, report):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    rows = report['epochs']
    fields = sorted(set(k for r in rows for k in r['losses']))
    with (directory/'epochs.csv').open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, ['epoch','elapsed_s','best_val_loss']+fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(dict(epoch=r['epoch'],elapsed_s=r['elapsed_s'],best_val_loss=r['best_val_loss'],**r['losses']))
    with (directory/'cells-evaluation.csv').open('w',newline='',encoding='utf-8') as stream:
        fields = ['image','split','name','predicted_label','dice','iou','detected','boundary_rmse_um','boundary_hd95_um','pair_mask_mse']
        writer = csv.DictWriter(stream,fields,extrasaction='ignore');writer.writeheader()
        for row in report['files']:
            for cell in row['metrics']['per_cell']:
                writer.writerow(dict(cell,image=row['image'],split=row['split']))
    fig = Figure(figsize=(11,7),layout='constrained');FigureCanvasAgg(fig)
    axes=fig.subplots(2,2).ravel()
    for key, label in [('loss','Training'),('val_loss','Validation')]:
        axes[0].plot([r['epoch'] for r in rows],[r['losses'].get(key,np.nan) for r in rows],label=label)
    axes[0].set_title('StarDist total loss')
    for ax,key,title in zip(axes[1:], ['mean_dice','detection_rate','boundary_rmse_um'],
            ['Validation Dice: complete cells','Validation detection: complete cells','Boundary RMSE (µm), matched complete cells']):
        scored=[r for r in rows if 'validation' in r]
        ax.plot([r['epoch'] for r in scored],[r['validation']['complete_cells'].get(key) for r in scored],'o-',
            label='Validation preview' if report['config'].get('validation_images',0) else 'Validation')
        selected=(report.get('calibrated_validation') or report.get('selected_validation',{})).get('complete_cells',{}).get(key)
        if selected is not None:
            ax.scatter([report['best_epoch']],[selected],marker='D',s=45,label='Selected: full validation')
        final=report.get('evaluation',{}).get('complete_cells',{}).get(key)
        if final is not None:
            ax.scatter([max([r['epoch'] for r in rows],default=0)+1],[final],marker='*',s=100,label='Test: final only')
        ax.set_title(title)
    for ax in axes:
        ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(directory/'training.png',dpi=150);fig.savefig(directory/'training.svg')
