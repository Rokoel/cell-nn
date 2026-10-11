import hashlib
import json
from pathlib import Path
import numpy as np
from .model import network, normalize, loss, tensorflow


def metrics(model, rows, threshold=.5):
    tp=fp=fn=tn=0; negative_rates=[]
    for row in rows:
        image, labels = row['image'], row['labels']
        probability=1/(1+np.exp(-np.clip(np.asarray(model(image[None,...,None], training=False))[0,...,0],-60,60)))
        found=probability>=threshold; truth=labels==3; valid=labels!=0
        tp+=int((found&truth&valid).sum());fp+=int((found&~truth&valid).sum())
        fn+=int((~found&truth&valid).sum());tn+=int((~found&~truth&valid).sum())
        if not truth.any(): negative_rates.append(float(found[valid].mean()))
    precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
    return dict(crops=len(rows),tp=tp,fp=fp,fn=fn,tn=tn,precision=precision,recall=recall,
                f1=2*tp/max(2*tp+fp+fn,1),iou=tp/max(tp+fp+fn,1),
                negative_only_false_positive_fraction=float(np.mean(negative_rates)) if negative_rates else None)


def train(annotations, audit, output, epochs=40, device='auto', seed=42):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',engine='sparse-boundary-unet2d',epochs=[],seed=seed,failures=[])
    def write(): (out/'training-report.json').write_text(json.dumps(report,indent=2))
    collection=json.loads(Path(audit).read_text())
    by_path={str(Path(p).resolve()):c for c in collection['cases']
             for p in [c['image'], *c['audit'].get('aliases', [])]}
    rows=[]
    for folder in sorted(Path(annotations).glob('crop-*')):
        meta=json.loads((folder/'metadata.json').read_text())
        if meta.get('schema_version')!=2: raise ValueError('Only schema-2 binary boundary annotations are supported.')
        case=by_path.get(str(Path(meta['source']).resolve()))
        if case is None and meta.get('audited_source_path'):
            case=by_path.get(str(Path(meta['audited_source_path']).resolve()))
        if not case or case['audit']['split']!='training' or case['audit']['channel']!=meta['cell_channel_index']:
            raise ValueError('Annotation source/channel must match the audited training split.')
        with np.load(folder/'annotation.npz') as data: image=data['image'];labels=data['labels']
        if image.shape!=(128,128) or labels.shape!=image.shape or not np.isin(labels,[0,1,3]).all():
            raise ValueError('This pilot expects valid 128×128 schema-2 annotations.')
        if not np.any(labels): continue
        rows.append(dict(image=normalize(image),labels=labels,group=case['audit']['group'],
                         folder=str(folder.resolve()),sha256=hashlib.sha256((folder/'annotation.npz').read_bytes()).hexdigest(),
                         spacing_xy=meta['spacing_xyz'][:2]))
    if len(rows)<8: raise ValueError('At least eight annotated crops are needed for a grouped pilot.')
    rng=np.random.default_rng(seed);groups=sorted({r['group'] for r in rows});rng.shuffle(groups)
    # Fix group-disjoint development split with both positive and negative-only examples.
    for _ in range(100):
        held=set(groups[:max(2,round(len(groups)*.2))]);val=[r for r in rows if r['group'] in held];training=[r for r in rows if r['group'] not in held]
        if all(any((r['labels']==3).any() for r in subset) and any(not (r['labels']==3).any() for r in subset) for subset in (training,val)):break
        rng.shuffle(groups)
    else: raise ValueError('Cannot form a useful group-disjoint development split.')
    report.update(annotation_crops=len(rows),development_train=len(training),development_validation=len(val),
                  train_groups=sorted(set(r['group'] for r in training)),validation_groups=sorted(held),
                  annotation_sources=[{k:r[k] for k in ('folder','group','sha256')} for r in rows],
                  policy='Unknown ignored. Per-image class-balanced BCE + positive-only masked Dice. 75% positive-containing crop sampling. Acquisition-group-disjoint development; final refit on all annotations.')
    write();tf=tensorflow(device,seed);model=network(tf);optimizer=tf.keras.optimizers.Adam(1e-3)
    def training_step(model, optimizer):
        # Each model/optimizer pair needs an independent TensorFlow trace.
        @tf.function
        def step(images, labels):
            with tf.GradientTape() as tape:
                value = loss(tf, labels, model(images, training=True))
            optimizer.apply_gradients(zip(tape.gradient(value, model.trainable_variables), model.trainable_variables))
            return value
        return step
    step = training_step(model, optimizer)
    def epoch(subset):
        positive=[r for r in subset if (r['labels']==3).any()];negative=[r for r in subset if not (r['labels']==3).any()]
        values=[]
        for _ in range(max(12,len(subset)//4)):
            images=[];labels=[]
            for _ in range(4):
                pool=positive if rng.random()<.75 or not negative else negative
                r=pool[int(rng.integers(len(pool)))];x,y=r['image'].copy(),r['labels'].copy()
                k=int(rng.integers(4));x,y=np.rot90(x,k),np.rot90(y,k)
                if rng.random()<.5:x,y=np.fliplr(x),np.fliplr(y)
                x=np.clip(x*rng.uniform(.7,1.3)+rng.uniform(-.05,.05),0,1)
                images.append(x[...,None]);labels.append(y[...,None])
            values.append(float(step(np.asarray(images,np.float32),np.asarray(labels,np.uint8))))
        return float(np.mean(values))
    best=-1;best_epoch=0
    for e in range(1,epochs+1):
        value=epoch(training);scores=metrics(model,val);record=dict(epoch=e,loss=value,validation=scores)
        report['epochs'].append(record)
        if scores['f1']>best:
            best=scores['f1'];best_epoch=e;model.save_weights(out/'development.weights.h5');report['development_metrics']=scores
        write();print(json.dumps(dict(phase='development',epoch=e,epochs=epochs,loss=value,validation_f1=scores['f1'])),flush=True)
    model.load_weights(out/'development.weights.h5')
    # Select operating threshold from development validation only, before final refit.
    candidates=[(t,metrics(model,val,float(t))) for t in np.linspace(.3,.85,12)]
    threshold,scores=max(candidates,key=lambda p:p[1]['f1'])
    report.update(best_epoch=best_epoch,development_threshold=float(threshold),development_metrics=scores)
    tf.keras.backend.clear_session();model=network(tf);optimizer=tf.keras.optimizers.Adam(1e-3)
    step = training_step(model, optimizer)
    report['refit_epochs']=[]
    for e in range(1,best_epoch+1):
        value=epoch(rows);report['refit_epochs'].append(dict(epoch=e,loss=value));write()
        print(json.dumps(dict(phase='all_annotation_refit',epoch=e,epochs=best_epoch,loss=value)),flush=True)
    model.save_weights(out/'boundary.weights.h5')
    manifest=dict(version=1,engine='sparse-boundary-unet2d',checkpoint='boundary.weights.h5',
                  checkpoint_sha256=hashlib.sha256((out/'boundary.weights.h5').read_bytes()).hexdigest(),
                  threshold=float(threshold),training_xy_um=float(np.median([r['spacing_xy'] for r in rows])),
                  patch_size=128,parameters=model.count_params(),annotation_crops=len(rows),
                  development_metrics=report['development_metrics'],best_epoch=best_epoch,
                  metrics_policy='Metrics are from the separate development checkpoint, not the final all-annotation refit.',
                  annotation_sha256=[r['sha256'] for r in rows])
    (out/'boundary-model.json').write_text(json.dumps(manifest,indent=2))
    report['status']='complete';write();return report
