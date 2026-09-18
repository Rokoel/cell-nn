from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fish_viewer.signals.config import SignalConfig
from fish_viewer.signals.data import (at_labels,native_labels,marker_points,infer_channels,candidates,
    candidate_targets,extract_patches)
from fish_viewer.signals.evaluation import evaluate_points,summarize,select_threshold


def test_a_is_signal_and_duplicate_scenes_do_not_duplicate_truth(tmp_path):
    scene=tmp_path/'a.imx'
    scene.write_text('<root><bpMeasurementPoints><Point Name="A" Position="1 2 3"/>'
        '<Point Name="B" Position="4 5 6"/><Point Name="C" Position="7 8 9"/></bpMeasurementPoints></root>')
    points=marker_points([scene,scene])
    assert len(points)==1 and points[0]['xyz']==[1.,2.,3.]


def test_native_label_mapping_preserves_calibrated_voxel_centers():
    labels=np.zeros((4,4,4),np.uint32);labels[1,1,1]=3
    metadata=dict(spacing_zyx=[.3,.3,.3],origin_xyz=[.15,.15,.15])
    dataset=SimpleNamespace(spacing=(.15,.15,.3),center_origin=(.075,.075,.15),volumes=[np.zeros((4,8,8))])
    native=native_labels(labels,metadata,dataset)
    assert native[1,2:4,2:4].min()==3
    assert at_labels(labels,[[.45,.45,.45]],metadata['origin_xyz'],metadata['spacing_zyx']).tolist()==[3]


def test_spots_require_image_evidence_and_center_membership_without_erosion():
    config=SignalConfig();grid=np.indices((32,)*3)
    image=np.exp(-sum((grid[a]-16)**2 for a in range(3))/2).astype(np.float32)
    labels=np.zeros(image.shape,np.uint32);labels[16:24,10:24,10:24]=1
    found=candidates(image,labels,(.15,)*3,np.zeros(3),config)
    assert any(np.allclose(p['xyz'],[2.4]*3) for p in found)
    assert not candidates(image,np.zeros_like(labels),(.15,)*3,np.zeros(3),config)
    for intensity in (0.,.1,1.):
        assert not candidates(np.full_like(image,intensity),labels,(.15,)*3,np.zeros(3),config)


def test_channel_inference_marks_ambiguous_channels_unknown():
    grid=np.indices((24,)*3);spot=np.exp(-sum((grid[a]-12)**2 for a in range(3))/2).astype(np.float32)
    points=[dict(xyz=[1.8]*3)]
    rows=infer_channels([np.zeros_like(spot),spot,np.zeros_like(spot)],points,0,np.zeros(3),(.15,)*3,SignalConfig())
    assert rows[0]['channel']==1
    rows=infer_channels([np.zeros_like(spot),spot,spot.copy()],points,0,np.zeros(3),(.15,)*3,SignalConfig())
    assert rows[0]['channel'] is None


def test_unknown_cells_wrong_channels_and_other_cells_are_not_positive():
    truth=[dict(xyz=[1,1,1],channel=1,cell_id=3)]
    rows=[dict(xyz=[1,1,1],channel=1,cell_id=3),dict(xyz=[1,1,1],channel=2,cell_id=3),
        dict(xyz=[1,1,1],channel=1,cell_id=4),dict(xyz=[9,9,9],channel=1,cell_id=8)]
    assert candidate_targets(rows,truth,[True,True,True,False],SignalConfig()).tolist()==[1,0,0,-1]
    assert summarize([evaluate_points(rows[:3],truth,.6)])['tp']==1


def test_center_matching_is_one_to_one_and_reports_false_detections():
    refs=[dict(xyz=[0,0,0],channel=1,cell_id=1),dict(xyz=[3,0,0],channel=1,cell_id=1)]
    predictions=[dict(xyz=[.1,0,0],channel=1,cell_id=1),dict(xyz=[.2,0,0],channel=1,cell_id=1)]
    metric=summarize([evaluate_points(predictions,refs,.6)])
    assert metric['tp']==1 and metric['fp']==1 and metric['fn']==1
    assert metric['precision']==metric['recall']==.5


def test_threshold_selection_uses_precision_constraint():
    refs=[dict(xyz=[i*3,0,0],channel=1,cell_id=1) for i in range(5)]
    rows=refs+[dict(xyz=[20,0,0],channel=1,cell_id=1)]
    selected=select_threshold([dict(candidates=rows,truth=refs)],[np.array([.9]*5+[.2])],SignalConfig())
    assert selected['target_met'] and selected['selected']['metrics']['precision']==1
    assert selected['selected']['metrics']['recall']==1


def test_physical_patch_is_bounded_and_includes_only_selected_cell_mask():
    image=np.ones((16,)*3,np.float32);cells=np.full(image.shape,2,np.uint32);cells[8:]=3
    patches=extract_patches([image,image*.5],0,cells,[dict(xyz=[.8]*3,channel=1,cell_id=3)],np.zeros(3),np.array([.1]*3),SignalConfig())
    assert patches.shape==(1,24,24,24,3) and np.isfinite(patches).all()
    assert set(np.unique(patches[...,2]))=={0,1}


def test_normalization_preserves_a_rare_bright_punctum():
    from fish_viewer.signals.data import normalized
    volume=np.ones((64,64,64),np.uint16);volume[32,32,32]=500
    result=normalized(volume)
    assert result[32,32,32]==1 and np.count_nonzero(result)==1


@pytest.mark.gui
def test_signal_report_counts_and_viewer_overlays(qapp,ims_file,tmp_path):
    from fish_viewer.app import MainWindow
    from fish_viewer.reader import load_dataset
    from fish_viewer.models import LibraryEntry
    from fish_viewer.signals.overlays import read_signals
    window=MainWindow(off_screen=True)
    report=dict(version=1,engine='signal-cnn3d',mode='apply',status='complete',directory=str(tmp_path),image=str(ims_file),
        fingerprint='expected',cell_predictions='cells.npz',channels=[dict(name='green',color=[0,1,0])],
        counts=[dict(cell_id=2,channel=0,channel_name='green',count=1)],
        points=[dict(signal_id=1,cell_id=2,channel=0,xyz=[11,21,31],score=.95)])
    path=tmp_path/'report.json';path.write_text(json.dumps(report))
    try:
        with pytest.raises(ValueError,match='different image'):read_signals(path,'wrong')
        measurements,_=read_signals(path,'expected')
        assert measurements[0].source=='signal_prediction' and not measurements[0].edges
        dataset=load_dataset(LibraryEntry(ims_file));dataset.scene.measurements.extend(measurements)
        window.entry=LibraryEntry(ims_file);window.set_dataset(dataset)
        window.slice_slider.setValue(0)
        assert window.slice_view._overlays
        window.show_predicted_signals.setChecked(False)
        assert not measurements[0].visible
        window.show_predicted_signals.setChecked(True)
        assert measurements[0].visible
        window.show_signals();panel=window.signal_panel
        panel.display_report(report)
        assert panel.points.rowCount()==1 and panel.counts.rowCount()==1
        assert panel.counts.item(0,1).text()=='1' and panel.points.item(0,6).text()=='0.950'
        panel.update_progress(dict(phase='Training signal classifier',current=40,total=80,epoch=7,epochs=100,loss=.13))
        assert panel.epoch_label.text()=='Epoch 7 / 100' and panel.progress.value()==40
    finally:window.close();qapp.processEvents()


def test_application_serializes_native_cell_ids_and_zero_counts(ims_file,tmp_path,monkeypatch):
    from dataclasses import asdict
    from fish_viewer.signals import engine
    from fish_viewer.reader import load_dataset
    from fish_viewer.models import LibraryEntry
    dataset=load_dataset(LibraryEntry(ims_file))
    labels=np.ones(dataset.volumes[0].shape,np.uint32);labels[-1]=2
    metadata=dict(channel=0,fingerprint=engine.image_fingerprint(dataset),spacing_zyx=list(dataset.spacing[::-1]),
        origin_xyz=list(dataset.center_origin),config=dict(rays=64))
    cells=tmp_path/'cells.npz';np.savez_compressed(cells,labels=labels,metadata=np.array(json.dumps(metadata)))
    model_dir=tmp_path/'model';model_dir.mkdir()
    (model_dir/'signal-model.json').write_text(json.dumps(dict(version=1,engine='signal-cnn3d',config=asdict(SignalConfig()),threshold=.9,checkpoint='placeholder')))
    monkeypatch.setattr(engine,'runtime',lambda *args:(None,None,None,dict(device='test')))
    monkeypatch.setattr(engine,'network',lambda *args:SimpleNamespace(load_weights=lambda path:None))
    monkeypatch.setattr(engine,'candidates',lambda *args:[dict(xyz=[11.,21.,31.],cell_id=1,contrast=.5)])
    monkeypatch.setattr(engine,'predict_patches',lambda model,patches:np.ones(len(patches)))
    report=engine.apply(ims_file,None,cells,model_dir,tmp_path/'runs')
    assert report['status']=='complete',report.get('error')
    loaded=json.loads((Path(report['directory'])/'report.json').read_text())
    assert [row['count'] for row in loaded['counts']]==[1,0]
    assert len(loaded['points'])==1
