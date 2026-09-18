from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from fish_viewer.models import LibraryEntry
from fish_viewer.neural.config import NeuralConfig
from fish_viewer.neural.data import NeuralCase, sparse_labels, three_way_split, training_arrays, subset_metrics
from fish_viewer.neural.engine import augment, filter_instances, run_neural
from fish_viewer.pipeline.evaluation import evaluate
from fish_viewer.pipeline.segmentation import Prepared


def reference(mask):
    return (tuple(slice(0,n) for n in mask.shape),mask)


def test_unknown_voxels_are_never_implicit_background():
    a=np.zeros((24,)*3,bool);a[5:12,5:12,5:12]=True
    cut=np.zeros_like(a);cut[:5,15:20,15:20]=True
    labels,cuts,audit=sparse_labels(a.shape,[reference(a),reference(cut)],['complete','cut'])
    assert set(np.unique(labels))=={-1,1}
    assert np.all(labels[a]==1) and np.all(labels[~a]==-1)
    assert cuts==[False,True]
    assert audit['training_objects']==1 and audit['cut_references']==1
    assert audit['background_voxels']==0


def test_overlapping_reference_objects_are_ignored_wholly():
    a=np.zeros((24,)*3,bool);a[5:12,5:12,5:12]=True
    b=np.roll(a,3,axis=0)
    c=np.zeros_like(a);c[16:20,16:20,16:20]=True
    labels,cuts,audit=sparse_labels(a.shape,list(map(reference,[a,b,c])),['A','B','C'])
    assert np.all(labels[a|b]==-1)
    assert np.all(labels[c]>0)
    assert audit['training_objects']==1
    assert audit['overlapping_objects_ignored']==['A','B']


def make_case(tmp_path,prefix,index=0):
    shape=(32,)*3
    z,y,x=np.indices(shape)
    r=np.sqrt((z-16)**2+(y-16)**2+(x-16)**2)
    image=np.exp(-.5*((r-6)/.8)**2).astype(np.float32)
    mask=r<6
    refs=[reference(mask)]
    labels,cuts,audit=sparse_labels(shape,refs,['manual'])
    prepared=Prepared(image,(.3,)*3,(.15,)*3,0,[],f'{prefix}-{index}',{}, {})
    path=tmp_path/f'{prefix}_{index}.ims'
    directory=tmp_path/f'case-{prefix}-{index}';directory.mkdir()
    return NeuralCase(LibraryEntry(path),prepared,refs,['manual'],prefix,directory,labels,cuts,audit)


def test_three_way_split_and_padding_preserve_unknowns(tmp_path):
    cases=[make_case(tmp_path,g,i) for g,i in [('101212',0),('101212',1),('131212',0),('141212',0)]]
    train,val,test=three_way_split(cases)
    assert len(train)==2 and len(val)==len(test)==1
    assert {c.group for c in train}.isdisjoint(c.group for c in val+test)
    with pytest.raises(ValueError):three_way_split(cases[:3])
    x,y=training_arrays(train,NeuralConfig(patch_z=48,patch_xy=48))
    assert x[0].shape==(48,)*3
    assert np.all(y[0][32:]==-1)
    assert 0 not in np.unique(y[0])


def test_augmentation_moves_labels_without_changing_ids(tmp_path):
    case=make_case(tmp_path,'101212')
    x,y=augment(case.prepared.image,case.labels)
    assert x.shape==y.shape==case.labels.shape
    assert set(np.unique(y))==set(np.unique(case.labels))
    assert int((y>0).sum())==int((case.labels>0).sum())
    assert np.isfinite(x).all() and 0<=x.min()<=x.max()<=1


def test_neural_shape_filter_and_complete_cut_scores(tmp_path):
    case=make_case(tmp_path,'101212')
    raw=np.maximum(case.labels,0).astype(np.uint32)
    output=filter_instances(raw,case.prepared,NeuralConfig())
    assert len(output.cells)==1
    metrics=subset_metrics(output.labels,case,evaluate)
    assert metrics['mean_dice']==metrics['complete_cells']['mean_dice']==1
    assert metrics['cut_cells']['mean_dice'] is None


def test_cancelled_run_persists_status_without_loading_tensorflow(tmp_path):
    cancel=threading.Event();cancel.set()
    report=run_neural([LibraryEntry(tmp_path/'missing.ims')],NeuralConfig(),tmp_path/'runs',cancel=cancel)
    assert report['status']=='cancelled'
    assert not report['epochs']
    assert json.loads((Path(report['directory'])/'report.json').read_text())['status']=='cancelled'


@pytest.mark.gui
def test_neural_gui_controls_and_failed_process(qapp,ims_file,tmp_path):
    from fish_viewer.app import MainWindow
    window=MainWindow(off_screen=True)
    errors=[]
    try:
        window.show_neural();panel=window.neural_panel;panel.show_error=errors.append
        config=replace(NeuralConfig(),epochs=7,steps_per_epoch=3,device='cpu',learning_rate=.000123,rays=96,filters=24,seed=77)
        panel.set_config(config)
        assert panel.config()==config
        panel.set_entries([LibraryEntry(ims_file)])
        panel.output.setText(str(tmp_path/'runs'))
        panel.start()
        deadline=time.monotonic()+20
        while panel.job is not None and time.monotonic()<deadline:
            qapp.processEvents();time.sleep(.01)
        # One unannotated acquisition cannot be silently used for all splits.
        assert panel.job is None
        assert panel.report['status']=='failed'
        assert panel.run_button.isEnabled()
        assert not errors
    finally:
        window.close();qapp.processEvents()


@pytest.mark.neural
@pytest.mark.parametrize('streaming',[False,True])
def test_real_stardist_training_checkpoint_and_prediction(tmp_path,monkeypatch,streaming):
    if not importlib.util.find_spec('stardist'):pytest.skip('Optional StarDist dependencies not installed')
    from fish_viewer.neural import engine
    cases=[make_case(tmp_path,g) for g in ('101212','131212','141212')]
    monkeypatch.setattr(engine,'load_cases',lambda *args,**kwargs:cases)
    config=NeuralConfig(epochs=1,steps_per_epoch=1,patch_z=32,patch_xy=32,rays=16,filters=8,device='cpu',calibrate_probability=False,
        streaming_validation=streaming,overlap_policy='isolate' if streaming else 'discard',
        synthetic_background_fraction=1 if streaming else 0)
    events=[]
    report=engine.run_neural([c.entry for c in cases],config,tmp_path/'runs',emit=events.append)
    assert report['status']=='complete',report.get('traceback')
    assert len(report['epochs'])==1 and np.isfinite(report['epochs'][0]['losses']['loss'])
    model=Path(report['model_directory'])
    assert (model/'best.weights.h5').stat().st_size>1000
    assert (model/'last.weights.h5').is_file()
    assert (model/'fish-model.json').is_file()
    assert len(report['files'])==3
    epoch_event=next(i for i,e in enumerate(events) if e['kind']=='epoch')
    test_event=next(i for i,e in enumerate(events) if e['kind']=='result' and e['row']['split']=='test')
    assert test_event>epoch_event
    phases={e['phase'] for e in events if e['kind']=='progress'}
    assert {'Training','Validation loss','Final prediction and evaluation','Complete'}<=phases
    from stardist.models import StarDist3D
    loaded=StarDist3D(None,name=model.name,basedir=str(model.parent))
    loaded.load_weights('best.weights.h5')
    # Unknown pixels also have no distance regularizer during sparse training.
    assert loaded.config.train_background_reg==0
    import tensorflow as tf
    loaded.prepare_for_training()
    probability=tf.Variable([.4,.4,.4],dtype=tf.float32)
    with tf.GradientTape() as tape:
        loss=loaded.keras_model.loss[0](tf.constant([-1.,0.,1.]),probability)
    gradients=tape.gradient(loss,probability).numpy()
    assert gradients[0]==0 and gradients[1]>0 and gradients[2]<0
    distances=tf.Variable(tf.ones((1,1,1,1,config.rays)))
    with tf.GradientTape() as tape:
        loss=loaded.keras_model.loss[1](tf.zeros((1,1,1,1,config.rays+1)),distances)
    assert np.all(tape.gradient(loss,distances).numpy()==0)
