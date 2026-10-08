"""Train a compact 3D candidate classifier; predict only inside neural cells."""
from dataclasses import asdict,replace
import csv
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from ..models import LibraryEntry
from ..reader import load_dataset
from ..run_paths import new_run_directory
from ..neural.engine import runtime
from ..pipeline.runner import _json
from ..pipeline.segmentation import checkpoint,Cancelled
from .config import SignalConfig
from .data import normalized,native_labels,candidates,extract_patches,candidate_targets
from .evaluation import evaluate_cases,select_threshold,suppress,detection_separation,evaluate_points,summarize


def network(tf,config):
    inputs=tf.keras.Input((config.patch_size,)*3+(3,))
    x=inputs
    for filters in (16,32,48):
        x=tf.keras.layers.Conv3D(filters,3,padding='same',activation='relu')(x)
        x=tf.keras.layers.MaxPool3D(2)(x)
    x=tf.keras.layers.Flatten()(x)
    x=tf.keras.layers.Dense(64,activation='relu')(x)
    x=tf.keras.layers.Dropout(.25)(x)
    outputs=tf.keras.layers.Dense(1,activation='sigmoid')(x)
    model=tf.keras.Model(inputs,outputs,name='signal_candidate_3d')
    model.compile(optimizer=tf.keras.optimizers.Adam(config.learning_rate),loss='binary_crossentropy')
    return model


def predict_patches(model,patches,batch=64):
    if not len(patches):return np.empty(0,np.float32)
    return np.concatenate([np.asarray(model(np.asarray(patches[i:i+batch],np.float32),training=False)).ravel()
        for i in range(0,len(patches),batch)])


class Pool:
    def __init__(self,cases):
        self.arrays=[np.load(Path(c['directory'])/'patches.npy',mmap_mode='r') for c in cases]
        self.targets=[np.array([r['target'] for r in c['candidates']],np.int8) for c in cases]
        self.positive=[(i,j) for i,y in enumerate(self.targets) for j in np.flatnonzero(y==1)]
        self.negative=[(i,j) for i,y in enumerate(self.targets) for j in np.flatnonzero(y==0)]
        if not self.positive or not self.negative:raise ValueError('Both signal and non-signal candidate examples are required.')

    def begin_epoch(self,rng):
        self.positive_order=rng.permutation(len(self.positive));self.positive_cursor=0

    def mine(self,model,emit=lambda *_:None,cancel=None):
        scored=[]
        for i,(array,targets) in enumerate(zip(self.arrays,self.targets)):
            indices=np.flatnonzero(targets==0)
            for start in range(0,len(indices),64):
                checkpoint(cancel)
                batch=indices[start:start+64]
                scores=predict_patches(model,array[batch])
                scored.extend((float(score),i,int(j)) for score,j in zip(scores,batch))
            emit(i+1,len(self.arrays))
        scored.sort(reverse=True)
        selected=scored[:max(32,int(np.ceil(len(scored)*.05)))]
        self.hard_negative=[(i,j) for score,i,j in selected]
        return dict(candidates=len(scored),hard_candidates=len(selected),
                    score_range=[selected[-1][0],selected[0][0]],
                    policy='Highest-scoring 5% of reviewed training negatives only. No validation/test or unknown candidates.',
                    examples=[dict(case_index=i,patch_index=j,score=score) for score,i,j in selected])

    def batch(self,size,rng,hard_fraction=0.):
        if not hasattr(self,'positive_order'):self.begin_epoch(rng)
        positive=[]
        while len(positive)<size//2:
            if self.positive_cursor==len(self.positive_order):self.begin_epoch(rng)
            count=min(size//2-len(positive),len(self.positive_order)-self.positive_cursor)
            positive.extend(self.positive_order[self.positive_cursor:self.positive_cursor+count])
            self.positive_cursor+=count
        pairs=[self.positive[i] for i in positive]
        hard_count=int((size-size//2)*hard_fraction) if getattr(self,'hard_negative',None) else 0
        pairs += [self.hard_negative[i] for i in rng.integers(0,len(self.hard_negative),size=hard_count)] if hard_count else []
        pairs += [self.negative[i] for i in rng.integers(0,len(self.negative),size=size-size//2-hard_count)]
        rng.shuffle(pairs)
        x=np.stack([self.arrays[i][j] for i,j in pairs]).astype(np.float32)
        y=np.array([self.targets[i][j] for i,j in pairs],np.float32)
        for i in range(len(x)):
            x[i]=np.rot90(x[i],int(rng.integers(4)),axes=(1,2))
            for axis in range(3):
                if rng.random()<.5:x[i]=np.flip(x[i],axis)
            x[i,...,:2]=np.clip(x[i,...,:2]*rng.uniform(.8,1.2,size=(1,1,1,2))+
                rng.normal(0,.008,x[i,...,:2].shape),0,1)
        return x,y


def balanced_loss(cases,probabilities):
    losses=[[],[]]
    for case,p in zip(cases,probabilities):
        y=np.array([r['target'] for r in case['candidates']]);p=np.clip(p,1e-6,1-1e-6)
        losses[1].extend(-np.log(p[y==1]));losses[0].extend(-np.log(1-p[y==0]))
    if not all(losses):raise ValueError('Validation requires both positive and negative supervised candidates.')
    return float(np.mean([np.mean(part) for part in losses]))


def charts(report,directory):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig=Figure(figsize=(11,7),layout='constrained');FigureCanvasAgg(fig)
    axes=fig.subplots(2,2);epochs=report['epochs'];x=[e['epoch'] for e in epochs]
    for key,title in [('loss','Training'),('val_loss','Validation (class balanced)')]:
        axes[0,0].plot(x,[e[key] for e in epochs],label=title)
    axes[0,0].set_title('Binary cross-entropy');axes[0,0].legend()
    for ax,key,title in [(axes[0,1],'precision','Validation precision'),(axes[1,0],'recall','Validation recall'),(axes[1,1],'f05','Validation F0.5')]:
        ax.plot(x,[e.get('validation_selected',e['validation'])[key] for e in epochs]);ax.set_title(title+(' at validation-selected score' if any('validation_selected' in e for e in epochs) else ' at score 0.5'));ax.set_ylim(0,1)
    for ax in axes.flat:ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    fig.savefig(directory/'training.png',dpi=150);fig.savefig(directory/'training.svg')


def train(audit_path,output,config,emit=lambda event:None,cancel=None,initial_model=None):
    audit=json.loads(Path(audit_path).read_text())
    if audit.get('engine')!='signal-audit' or audit['status']!='complete':raise ValueError('A completed signal audit is required.')
    # Sampling and matching parameters belong to the audit, not to an unrelated UI default.
    base=SignalConfig(**audit['config'])
    config=replace(base,**{key:getattr(config,key) for key in ('epochs','steps_per_epoch','batch_size','patience','learning_rate','device','seed',
        'hard_negative_fraction','hard_negative_interval','checkpoint_policy','detection_separation_um')}).validate()
    directory=new_run_directory(output, 'signal-train')
    model_dir=directory/'model';model_dir.mkdir()
    started=time.monotonic()
    last_progress_write=0.
    report=dict(version=1,engine='signal-cnn3d',mode='train',status='running',directory=str(directory),model_directory=str(model_dir),
        config=asdict(config),audit_path=str(Path(audit_path).resolve()),epochs=[],best_epoch=None,files=[],
        metric_notes='Precision/recall are conditional on annotated-cell intersections with predicted cells and inferred unambiguous channels. Unknown cells are unscored. Scores are not calibrated probabilities.')
    def persist():
        report['elapsed_s']=time.monotonic()-started;_json(directory/'report.json',report)
    def progress(phase,current,total,**kwargs):
        nonlocal last_progress_write
        event=dict(kind='progress',phase=phase,current=current,total=total,**kwargs)
        report['progress']={k:v for k,v in event.items() if k!='kind'};emit(event)
        if time.monotonic()-last_progress_write>=1:
            persist();last_progress_write=time.monotonic()
    persist();emit(dict(kind='run_started',directory=str(directory)))
    try:
        splits={name:[c for c in audit['cases'] if c['split']==name] for name in ('training','validation','test')}
        groups={name:{c['group'] for c in cs} for name,cs in splits.items()}
        if any(groups[a]&groups[b] for a,b in [('training','validation'),('training','test'),('validation','test')]):
            raise ValueError('Acquisition leakage between dataset splits.')
        report['data_summary']=audit['summary']
        training=Pool(splits['training'])
        val_arrays=[np.load(Path(c['directory'])/'patches.npy',mmap_mode='r') for c in splits['validation']]
        report['training_candidates']=dict(positive=len(training.positive),negative=len(training.negative))
        progress('Initializing signal network',0,0);persist()
        tf,_,_,info=runtime(config.device,config.seed);report['runtime']=info
        model=network(tf,config);report['parameters']=model.count_params()
        if initial_model:
            parent=Path(initial_model).resolve();parent_manifest=json.loads((parent/'signal-model.json').read_text())
            parent_config=SignalConfig(**parent_manifest['config'])
            if parent_manifest.get('engine')!='signal-cnn3d' or (parent_config.patch_size,parent_config.patch_spacing_um)!=(config.patch_size,config.patch_spacing_um):
                raise ValueError('Fine-tuning requires a compatible signal crop model.')
            checkpoint_path=parent/parent_manifest['checkpoint'];model.load_weights(checkpoint_path)
            report['initial_checkpoint']=dict(model=str(parent),sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest())
            # Retain fresh optimizer settings rather than the parent's stored slots.
            model.compile(optimizer=tf.keras.optimizers.Adam(config.learning_rate),loss='binary_crossentropy')
        steps=max(config.steps_per_epoch,int(np.ceil(len(training.positive)/(config.batch_size//2))))
        report['steps_per_epoch_actual']=steps
        report['sampling_policy']='Every positive training candidate is visited each epoch, then reshuffled if more batches remain. Balanced reviewed negatives; optional training-only hard-negative mining. Unknown candidates never sampled.'
        report['hard_negative_mining']=[]
        best=np.inf;best_key=None;waiting=0;rng=np.random.default_rng(config.seed)
        for epoch in range(1,config.epochs+1):
            if config.hard_negative_fraction and (epoch-1)%config.hard_negative_interval==0:
                mined=training.mine(model,lambda current,total:progress('Mining reviewed training negatives',current,total,epoch=epoch,epochs=config.epochs),cancel)
                _json(directory/f'hard-negatives-epoch-{epoch:04d}.json',mined)
                report['hard_negative_mining'].append(dict(epoch=epoch,**{k:v for k,v in mined.items() if k!='examples'}))
            model.reset_metrics();losses=[];training.begin_epoch(rng)
            for step in range(steps):
                checkpoint(cancel)
                x,y=training.batch(config.batch_size,rng,config.hard_negative_fraction)
                loss=float(model.train_on_batch(x,y));losses.append(loss)
                if step%5==0:
                    progress('Training signal classifier',step+1,steps,epoch=epoch,epochs=config.epochs,loss=loss)
            progress('Validating signal candidates',0,len(val_arrays),epoch=epoch,epochs=config.epochs)
            probabilities=[]
            for i,patches in enumerate(val_arrays):
                checkpoint(cancel);probabilities.append(predict_patches(model,patches))
                if i%10==0:progress('Validating signal candidates',i+1,len(val_arrays),epoch=epoch,epochs=config.epochs)
            val_loss=balanced_loss(splits['validation'],probabilities)
            metrics=evaluate_cases(splits['validation'],probabilities,.5,config)
            row=dict(epoch=epoch,loss=float(np.mean(losses)),val_loss=val_loss,validation=metrics)
            if config.checkpoint_policy=='validation_precision':
                epoch_selection=select_threshold(splits['validation'],probabilities,config)
                row['validation_selected']=epoch_selection['selected']['metrics']
                row['selected_threshold']=epoch_selection['selected']['threshold']
                row['precision_target_met']=epoch_selection['target_met']
                key=(epoch_selection['target_met'],row['validation_selected']['recall'] if epoch_selection['target_met'] else row['validation_selected']['f05'],
                     row['validation_selected']['precision'] or 0,-val_loss)
                improved=best_key is None or key>best_key
            else:
                key=None;improved=val_loss<best-1e-5
            report['epochs'].append(row)
            if improved:
                best=val_loss;best_key=key;report['best_epoch']=epoch;waiting=0;model.save_weights(model_dir/'best.weights.h5')
            else:waiting+=1
            model.save_weights(model_dir/'last.weights.h5')
            persist();emit(dict(kind='epoch',**row))
            if epoch%5==0:charts(report,directory)
            if waiting>=config.patience:break
        model.load_weights(model_dir/'best.weights.h5')
        progress('Selecting validation threshold',0,1);persist()
        probabilities=[predict_patches(model,p) for p in val_arrays]
        selection=select_threshold(splits['validation'],probabilities,config)
        threshold=selection['selected']['threshold'];report['selection']=selection
        report['validation']=selection['selected']['metrics']
        manifest=dict(version=1,engine='signal-cnn3d',config=asdict(config),checkpoint='best.weights.h5',threshold=threshold,
            cell_model=audit['cell_model'],cell_report=audit['cell_report'],
            input_channels=['candidate fluorescence','membrane fluorescence','predicted cell mask'],
            score_note='Sigmoid classifier score, not calibrated correctness probability.',selection=selection)
        manifest['training_provenance']=dict(
            audit_sha256=hashlib.sha256(Path(audit_path).read_bytes()).hexdigest(),
            data_summary=audit['summary'],best_epoch=report['best_epoch'],
            cell_checkpoint_sha256=audit.get('cell_checkpoint_sha256'),
            boundary_checkpoint_sha256=audit.get('boundary_checkpoint_sha256'),
            candidate_selection=audit.get('candidate_selection'),
            sampling_policy=report['sampling_policy'],
            initial_checkpoint=report.get('initial_checkpoint'),hard_negative_mining=report['hard_negative_mining'],
            selection_policy=('Maximize validation recall at the declared precision target; precision and balanced loss break ties.' if config.checkpoint_policy=='validation_precision' else 'Best class-balanced validation loss; threshold maximizes validation recall at the declared precision target.')+' Test does not select weights or cutoffs.')
        _json(model_dir/'signal-model.json',manifest)
        # The test fold is first accessed for network evaluation after checkpoint/threshold selection.
        test_probabilities=[]
        for index,case in enumerate(splits['test']):
            checkpoint(cancel);progress('Testing annotated signal regions',index+1,len(splits['test']),file=case['image']);persist()
            patches=np.load(Path(case['directory'])/'patches.npy',mmap_mode='r')
            scores=predict_patches(model,patches);test_probabilities.append(scores)
            points=suppress(case['candidates'],scores,threshold,detection_separation(config))
            path=directory/f'test-{index+1:04d}.json'
            _json(path,dict(image=case['image'],points=points,truth=case['truth'],scope='Scored annotated-cell intersections only'))
            report['files'].append(dict(image=case['image'],scene=case['scene'],signals=str(path),split='test',count=len(points),cell_predictions=case['cell_predictions']))
        report['test']=evaluate_cases(splits['test'],test_probabilities,threshold,config)
        report['candidate_ceiling']=summarize([evaluate_points(c['candidates'],c['truth'],config.match_radius_um) for c in splits['test']])
        # Same candidates without the network, calibrated on validation contrast only.
        baseline=select_threshold(splits['validation'],[np.array([p['contrast'] for p in c['candidates']]) for c in splits['validation']],config)
        report['classical_baseline']=dict(selection=baseline,test=evaluate_cases(splits['test'],
            [np.array([p['contrast'] for p in c['candidates']]) for c in splits['test']],baseline['selected']['threshold'],config))
        report['status']='complete';progress('Complete',len(report['epochs']),len(report['epochs']))
        charts(report,directory)
        with (directory/'epochs.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['epoch','loss','val_loss','precision','recall','f05']);writer.writeheader()
            for row in report['epochs']:writer.writerow({k:row[k] for k in ('epoch','loss','val_loss')}|{k:row['validation'][k] for k in ('precision','recall','f05')})
    except Cancelled:report['status']='cancelled'
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc()
    persist();emit(dict(kind='finished',report=report));return report


def image_fingerprint(dataset):
    digest=hashlib.sha256();digest.update(str(dataset.volumes[0].shape).encode())
    for volume in dataset.volumes:digest.update(memoryview(np.ascontiguousarray(volume)))
    return digest.hexdigest()


def apply(image,scene,cell_predictions,model_directory,output,device='auto',emit=lambda event:None,cancel=None):
    directory=new_run_directory(output, 'signal-apply')
    report=dict(version=1,engine='signal-cnn3d',mode='apply',status='running',directory=str(directory),image=str(Path(image).resolve()),
        scene=str(scene) if scene else None,cell_predictions=str(Path(cell_predictions).resolve()),model_directory=str(Path(model_directory).resolve()),
        channel_index_base=0,unit='µm',time=0,points=[],counts=[])
    started=time.monotonic()
    def progress(phase,current,total):
        report['progress']=dict(phase=phase,current=current,total=total)
        _json(directory/'report.json',report);emit(dict(kind='progress',**report['progress']))
    emit(dict(kind='run_started',directory=str(directory)))
    try:
        manifest=json.loads((Path(model_directory)/'signal-model.json').read_text())
        if manifest.get('engine')!='signal-cnn3d' or manifest.get('version')!=1:raise ValueError('Choose a compatible signal model.')
        config=SignalConfig(**manifest['config']).validate()
        with np.load(cell_predictions,allow_pickle=False) as archive:
            labels=archive['labels'];metadata=json.loads(str(archive['metadata']))
        if 'rays' not in metadata.get('config',{}) and metadata.get('detector',{}).get('engine')!='cell-center-box3d':
            raise ValueError('Select neural-network cell contours, not classical contours.')
        if metadata.get('detector'):
            report['cell_detector']=metadata['detector']
        if metadata.get('refinement'):
            report['boundary_refinement']=metadata['refinement']
        if metadata.get('membrane_snap'):
            report['membrane_snap']=metadata['membrane_snap']
        progress('Loading signal channels',0,1)
        dataset=load_dataset(LibraryEntry(Path(image),Path(scene) if scene else None))
        if image_fingerprint(dataset)!=metadata['fingerprint']:raise ValueError('Cell contours belong to different image pixels.')
        report['fingerprint']=metadata['fingerprint'];report['channels']=[dict(name=c.name,color=list(c.color)) for c in dataset.metadata.channels]
        images=[normalized(volume) for volume in dataset.volumes];membrane=int(metadata['channel'])
        cells=native_labels(labels,metadata,dataset);spacing=np.array(dataset.spacing[::-1]);origin=np.array(dataset.center_origin)
        proposed=[]
        for channel,volume in enumerate(images):
            checkpoint(cancel);progress('Finding candidates inside neural cells',channel+1,len(images))
            if channel==membrane:continue
            proposed.extend(dict(row,channel=channel) for row in candidates(volume,cells,spacing,origin,config))
        report['candidate_count']=len(proposed);report['threshold']=manifest['threshold'];report['membrane_channel']=membrane
        progress('Loading signal network',0,1)
        tf,_,_,info=runtime(device,config.seed);report['runtime']=info
        model=network(tf,config);model.load_weights(Path(model_directory)/manifest['checkpoint'])
        scores=[]
        for start in range(0,len(proposed),64):
            checkpoint(cancel);progress('Scoring 3D signal candidates',start,len(proposed))
            patches=extract_patches(images,membrane,cells,proposed[start:start+64],origin,spacing,config)
            scores.extend(predict_patches(model,patches).tolist())
        points=suppress(proposed,scores,manifest['threshold'],detection_separation(config))
        report['points']=[dict(signal_id=i+1,**row) for i,row in enumerate(points)]
        report['counts']=[dict(cell_id=int(cell_id),channel=channel,channel_name=dataset.metadata.channels[channel].name,
            count=int(sum(p['cell_id']==cell_id and p['channel']==channel for p in points)))
            for cell_id in np.unique(labels) if cell_id>0 for channel in range(len(images)) if channel!=membrane]
        for name,rows,fields in [('signals',report['points'],['signal_id','cell_id','channel','x_um','y_um','z_um','score','contrast','candidate_scale_um']),
                                 ('cell-counts',report['counts'],['cell_id','channel','channel_name','count'])]:
            with (directory/f'{name}.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                for row in rows:
                    if name=='signals':row={k:v for k,v in row.items() if k!='xyz'}|dict(zip(('x_um','y_um','z_um'),row['xyz']))
                    writer.writerow(dict(row,channel=row['channel']+1))
        report['status']='complete';progress('Complete',len(points),len(points))
    except Cancelled:report['status']='cancelled'
    except Exception:report['status']='failed';report['error']=traceback.format_exc()
    report['elapsed_s']=time.monotonic()-started;_json(directory/'report.json',report)
    emit(dict(kind='finished',report=report));return report
