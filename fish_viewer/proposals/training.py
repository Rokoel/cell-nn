from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from ..neural.engine import runtime
from ..pipeline.runner import _json
from ..pipeline.segmentation import Cancelled,checkpoint
from ..run_paths import new_run_directory
from .data import cache_patches,batch
from .network import network,loss


def charts(directory,rows):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure=Figure(figsize=(10,4),layout='constrained');FigureCanvasAgg(figure)
    axes=figure.subplots(1,2)
    for ax,key,title in zip(axes,('loss','box_mae_um'),('Masked center + box loss','Box face MAE (µm)')):
        for split in ('training','validation'):
            ax.plot([r['epoch'] for r in rows],[r[split][key] for r in rows],label=split)
        ax.set(xlabel='Epoch',title=title);ax.grid(alpha=.2);ax.legend()
    figure.savefig(directory/'training-progress.png',dpi=160)
    with (directory/'epochs.csv').open('w',newline='',encoding='utf-8') as stream:
        fields=['epoch']+[s+'_'+k for s in ('training','validation') for k in ('loss','center_loss','box_mae_um')]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in rows:writer.writerow(dict(epoch=r['epoch'],**{s+'_'+k:r[s][k] for s in ('training','validation') for k in ('loss','center_loss','box_mae_um')}))


def train(audit,patches,output,config,device='auto',emit=lambda e:None,cancel=None):
    config.validate();directory=new_run_directory(output,'proposal-train')
    model_dir=directory/'model';model_dir.mkdir()
    report=dict(version=1,engine='cell-center-box3d',status='running',directory=str(directory),
                model_directory=str(model_dir),config=asdict(config),audit=str(Path(audit).resolve()),epochs=[])
    started=time.monotonic()
    def notify(event):
        if event['kind']=='progress':
            report['progress']={k:v for k,v in event.items() if k!='kind'};_json(directory/'report.json',report)
        emit(event)
    _json(directory/'report.json',report);emit(dict(kind='run_started',directory=str(directory)))
    try:
        index=cache_patches(audit,patches,config,notify,cancel)
        records=index['records'];images=np.load(Path(patches)/'images.npy',mmap_mode='r')
        targets=np.load(Path(patches)/'targets.npy',mmap_mode='r')
        train_ids=[r['index'] for r in records if r['split']=='training']
        val_ids=[r['index'] for r in records if r['split']=='validation']
        if not train_ids or not val_ids:raise ValueError('Independent training and validation targets are required.')
        report['samples']=dict(training_cells=len(train_ids),validation_cells=len(val_ids),background_crops=0)
        notify(dict(kind='progress',phase='Initializing tiny point-and-box network',current=0,total=1))
        tf,_,_,info=runtime(device,config.seed);report['runtime']=info
        if info['device']=='GPU':tf.keras.mixed_precision.set_global_policy('mixed_float16')
        model=network(tf,config);report['parameters']=model.count_params()
        optimizer=tf.keras.optimizers.Adam(config.learning_rate,clipnorm=5.)
        if info['device']=='GPU':optimizer=tf.keras.mixed_precision.LossScaleOptimizer(optimizer,initial_scale=128.)
        @tf.function(jit_compile=False,reduce_retracing=True)
        def train_step(x,y):
            with tf.GradientTape() as tape:
                result=loss(tf,y,model(x,training=True))
                scaled=optimizer.scale_loss(result[0]) if hasattr(optimizer,'scale_loss') else result[0]
            optimizer.apply_gradients(zip(tape.gradient(scaled,model.trainable_variables),model.trainable_variables))
            return result
        @tf.function(jit_compile=False,reduce_retracing=True)
        def val_step(x,y):return loss(tf,y,model(x,training=False))
        rng=np.random.default_rng(config.seed);best=float('inf');stale=0
        manifest=dict(version=1,engine='cell-center-box3d',checkpoint='best.weights.h5',config=asdict(config),
            supervision=index['policy'],samples=report['samples'],parameters=report['parameters'],
            architecture='Two-level tiny 3D encoder/decoder; heatmap + six positive box-face distances; output stride 2.',
            audit_sha256=index['identity']['audit_sha256'],runtime=info)
        source_config=json.loads(Path(audit).read_text())['config']
        manifest['channel_selection']={k:source_config[k] for k in ('min_diameter_um','max_diameter_um')}
        for epoch in range(1,config.epochs+1):
            checkpoint(cancel);row=dict(epoch=epoch)
            for split,ids,step in [('training',rng.permutation(train_ids),train_step),('validation',val_ids,val_step)]:
                values=[];counts=[];total=int(np.ceil(len(ids)/config.batch_size))
                for start in range(0,len(ids),config.batch_size):
                    checkpoint(cancel);selected=ids[start:start+config.batch_size]
                    x,y=batch(images,targets,selected,rng,split=='training')
                    result=[float(v.numpy()) for v in step(x,y)]
                    if not np.isfinite(result).all():raise ValueError('Non-finite detector loss.')
                    values.append(result);counts.append(len(selected))
                    b=start//config.batch_size+1
                    if b%10==1 or b==total:
                        notify(dict(kind='progress',phase=split.title()+' cell centers and boxes',epoch=epoch,
                            epochs=config.epochs,current=b,total=total,loss=result[0],box_mae_um=result[2]*config.box_scale_um))
                mean=np.average(values,axis=0,weights=counts)
                row[split]=dict(loss=float(mean[0]),center_loss=float(mean[1]),box_mae_um=float(mean[2]*config.box_scale_um))
            report['epochs'].append(row)
            if row['validation']['loss']<best-1e-4:
                best=row['validation']['loss'];stale=0
                temporary=model_dir/'checkpoint.tmp.weights.h5';model.save_weights(temporary)
                temporary.replace(model_dir/'best.weights.h5');report['best_epoch']=epoch
                manifest.update(best_epoch=epoch,validation_loss=best);_json(model_dir/'proposal-model.json',manifest)
            else:stale+=1
            _json(directory/'report.json',report);charts(directory,report['epochs']);emit(dict(kind='epoch',**row))
            if stale>=config.patience:report['stop_reason']='Validation loss stopped improving.';break
        report['status']='complete'
        report['checkpoint_sha256']=hashlib.sha256((model_dir/'best.weights.h5').read_bytes()).hexdigest()
    except Cancelled:report['status']='cancelled'
    except Exception:report.update(status='failed',error=traceback.format_exc())
    report['elapsed_s']=time.monotonic()-started;_json(directory/'report.json',report)
    emit(dict(kind='finished',report=report));return report
