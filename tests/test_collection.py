from types import SimpleNamespace
from pathlib import Path
import numpy as np
import json
from dataclasses import asdict
import pytest

from fish_viewer.neural.config import NeuralConfig
from fish_viewer.neural.data import NeuralCase,sparse_labels,training_arrays
from fish_viewer.neural.audit import acquisition_key,split_audit_cases,unique_references,save_references,read_references
from fish_viewer.pipeline.segmentation import Prepared
from fish_viewer.models import LibraryEntry


def test_overlapping_cells_keep_complete_independent_training_targets(tmp_path):
    a=np.zeros((32,)*3,bool);a[6:16,6:16,6:16]=True
    b=np.roll(a,4,axis=0);box=(slice(0,32),)*3
    refs=[(box,a),(box,b)]
    labels,cuts,audit=sparse_labels(a.shape,refs,['a','b'],'isolate')
    assert audit['training_objects']==2 and (labels==-1).all()
    prepared=Prepared(a.astype(np.float32),(.3,)*3,(0.,)*3,0,[],'unique',{}, {})
    case=NeuralCase(LibraryEntry(tmp_path/'a.ims'),prepared,refs,['a','b'],'a',tmp_path,labels,cuts,audit)
    x,y=training_arrays([case],NeuralConfig(overlap_policy='isolate',patch_z=32,patch_xy=32))
    assert len(x)==len(y)==2
    assert np.array_equal(y[0]>0,a) and np.array_equal(y[1]>0,b)
    assert set(np.unique(y[0]))=={-1,1}


def test_acquisition_components_hold_dates_aliases_and_pixels_together(tmp_path):
    def case(i,keys,fp=None):
        return SimpleNamespace(entry=LibraryEntry(tmp_path/f'{i}.ims'),audit=dict(acquisition_keys=keys,training_objects=1),
            prepared=SimpleNamespace(fingerprint=fp or str(i)),group='')
    cases=[case(0,['date:2013-01-01']),case(1,['date:2013-01-01','date:2013-01-02']),
        case(2,['date:2013-01-02']),case(3,['date:2013-01-03'],'same'),case(4,['date:2013-01-04'],'same'),
        case(5,['date:2013-01-05']),case(6,['date:2013-01-06']),case(7,['date:2013-01-07'])]
    split_audit_cases(cases,42)
    assert len({c.group for c in cases[:3]})==1
    assert cases[3].group==cases[4].group
    splits={name:{c.group for c in cases if c.audit['split']==name} for name in ['training','validation','test']}
    assert all(splits.values())
    assert splits['training'].isdisjoint(splits['validation']|splits['test'])
    assert splits['validation'].isdisjoint(splits['test'])
    assert acquisition_key(tmp_path/'101212_a.ims')=='date:2012-12-10'
    assert acquisition_key(tmp_path/'unknown_a.ims')==acquisition_key(tmp_path/'unknown_b.ims')


def test_reference_deduplication_and_calibrated_cache_roundtrip(tmp_path):
    refs=[((slice(2,6),slice(3,8),slice(1,7)),np.ones((4,5,6),bool))]
    unique,names=unique_references(refs+refs,['original','copy'])
    assert len(unique)==1 and names==['original']
    path=tmp_path/'references.npz';save_references(path,unique,names)
    read,loaded_names=read_references(path)
    assert loaded_names==names and read[0][0]==refs[0][0]
    assert np.array_equal(read[0][1],refs[0][1])


def test_tiled_proposals_keep_physical_grid_and_unique_core_ownership(tmp_path):
    from fish_viewer.neural.engine import proposals
    # Maxima on either side of a block boundary must each be owned once.
    shape=(32,160,160);data=np.zeros(shape,np.float32)
    data[12,124,124]=.9;data[12,132,132]=.8
    class Model:
        config=SimpleNamespace(grid=(1,2,2))
        def predict(self,x,**kwargs):
            p=x[:,::2,::2];return p,np.full((*p.shape,16),3,np.float32)
    case=SimpleNamespace(prepared=Prepared(data,(.3,)*3,(0.,)*3,0,[],'x',{},{}))
    points,prob,dist,capped=proposals(Model(),case,NeuralConfig(rays=16))
    assert sorted(points.tolist())==[[12,124,124],[12,132,132]]
    assert len(prob)==2 and not capped


@pytest.mark.gui
def test_audited_gui_inputs_and_settings(qapp,ims_file,tmp_path):
    from fish_viewer.app import MainWindow
    config=NeuralConfig(native_sampling=True,overlap_policy='isolate',streaming_validation=True,validation_images=8)
    audit=tmp_path/'audit.json'
    audit.write_text(json.dumps(dict(engine='stardist3d-audit',status='complete',config=asdict(config),
        cases=[dict(image=str(ims_file),scene=None)])))
    window=MainWindow(off_screen=True)
    try:
        window.show_neural();panel=window.neural_panel
        panel.set_audit(audit,restore_config=True)
        assert panel.config()==config
        assert not panel.files.isEnabled() and panel.files.count()==1
        panel.mode.setCurrentIndex(1)
        assert panel.files.isEnabled()
        panel.mode.setCurrentIndex(0)
        assert not panel.files.isEnabled()
        panel.set_audit(None)
        assert panel.files.isEnabled() and not panel.audit_path.text()
    finally:window.close();qapp.processEvents()
