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


def test_multiscale_candidates_reach_dim_nearby_spots_and_deduplicate_scales():
    from dataclasses import replace
    grid=np.indices((40,)*3)
    image=np.exp(-sum((grid[a]-18)**2 for a in range(3))/(2*.6**2))
    image+=.5*np.exp(-((grid[0]-18)**2+(grid[1]-21)**2+(grid[2]-21)**2)/(2*.6**2))
    cells=np.ones(image.shape,np.uint32);spacing=np.array([.1]*3)
    legacy=candidates(image,cells,spacing,np.zeros(3),SignalConfig())
    config=replace(SignalConfig(),candidate_scales_um=(.10,.18,.28),candidate_contrast=.01,candidate_separation_um=.25)
    found=candidates(image,cells,spacing,np.zeros(3),config)
    assert len(legacy)==1
    assert len(found)==2
    assert any(np.linalg.norm(np.array(p['xyz'])-[2.1,2.1,1.8])<.15 for p in found)
    assert all(p['candidate_scale_um'] in config.candidate_scales_um for p in found)
    assert not candidates(np.zeros_like(image),cells,spacing,np.zeros(3),config)
    assert not candidates(image,np.zeros_like(cells),spacing,np.zeros(3),config)


def test_more_permissive_candidates_do_not_change_manual_channel_assignment():
    from dataclasses import replace
    grid=np.indices((24,)*3)
    image=.02*np.exp(-sum((grid[a]-12)**2 for a in range(3))/2).astype(np.float32)
    config=replace(SignalConfig(),candidate_contrast=.005,channel_contrast_min=.025,channel_score_min=.1)
    result=infer_channels([np.zeros_like(image),image],[dict(xyz=[1.8]*3)],0,np.zeros(3),(.15,)*3,config)
    assert result[0]['channel'] is None


def test_one_to_one_positive_targets_ignore_duplicate_nearby_candidates():
    from dataclasses import replace
    config=replace(SignalConfig(),one_to_one_positives=True)
    truth=[dict(xyz=[1.,1.,1.],channel=1,cell_id=3)]
    rows=[dict(xyz=[1.05,1.,1.],channel=1,cell_id=3),dict(xyz=[1.4,1.,1.],channel=1,cell_id=3),
          dict(xyz=[4.,1.,1.],channel=1,cell_id=3),dict(xyz=[1.,1.,1.],channel=1,cell_id=9)]
    assert candidate_targets(rows,truth,[True,True,True,False],config).tolist()==[1,-1,0,-1]


def test_signal_supervision_rejects_a_mismatched_reference_grid(tmp_path):
    from fish_viewer.signals.data import prepare_supervision
    prediction=tmp_path/'contours.npz'
    metadata=dict(fingerprint='same',spacing_zyx=[.15]*3,origin_xyz=[.075]*3)
    np.savez_compressed(prediction,labels=np.ones((4,)*3,np.uint32),metadata=np.array(json.dumps(metadata)))
    case=dict(image='scan.ims',audit=dict(aliases=[],spacing_zyx=[.3]*3,origin_xyz=[.15]*3))
    with pytest.raises(ValueError,match='grids differ'):
        prepare_supervision(case,{'scan.ims':dict(fingerprint='same')},dict(predictions=str(prediction)),SignalConfig())


def test_signal_configuration_rejects_invalid_physical_candidate_scales():
    from dataclasses import replace
    for scales in ((0.,),(-.1,),(float('nan'),)):
        with pytest.raises(ValueError,match='scales'):
            replace(SignalConfig(),candidate_scales_um=scales).validate()


def test_signal_suppression_matches_physical_greedy_rule():
    from fish_viewer.signals.evaluation import suppress
    rng=np.random.default_rng(42)
    rows=[dict(xyz=xyz.tolist(),channel=int(rng.integers(2)),cell_id=int(rng.integers(3)))
          for xyz in rng.uniform(-1,1,(350,3))]
    rows.extend([dict(xyz=[0,0,0],channel=0,cell_id=1),
                 dict(xyz=[.25,0,0],channel=0,cell_id=1)])
    scores=np.r_[rng.choice([.1,.5,.9],350),1.,1.]
    for separation in (0,.25,.6):
        expected=[]
        for index in np.argsort(-scores,kind='stable'):
            if scores[index]<.5:continue
            row=rows[index]
            if any(old['channel']==row['channel'] and old['cell_id']==row['cell_id'] and
                   np.linalg.norm(np.asarray(old['xyz'])-row['xyz'])<separation for old in expected):continue
            expected.append(dict(row,score=float(scores[index])))
        assert suppress(rows,scores,.5,separation)==expected


def test_signal_training_visits_all_positive_crops_each_epoch(tmp_path):
    from fish_viewer.signals.engine import Pool
    patches=np.zeros((6,8,8,8,3),np.float16)
    for i in range(5):patches[i,...,2]=i+1
    np.save(tmp_path/'patches.npy',patches)
    pool=Pool([dict(directory=str(tmp_path),candidates=[dict(target=1)]*5+[dict(target=0)])])
    rng=np.random.default_rng(42);pool.begin_epoch(rng)
    visited=set()
    for _ in range(2):
        x,y=pool.batch(6,rng)
        visited.update(int(row[...,2].mean()) for row,positive in zip(x,y) if positive)
    assert visited=={1,2,3,4,5}


def test_hard_negative_mining_never_samples_unknown_or_positive_crops(tmp_path):
    from fish_viewer.signals.engine import Pool
    patches=np.zeros((4,8,8,8,3),np.float16)
    for i in range(4):patches[i,...,2]=i+1
    np.save(tmp_path/'patches.npy',patches)
    pool=Pool([dict(directory=str(tmp_path),candidates=[dict(target=y) for y in (1,0,0,-1)])])
    def score(batch,training=False):return batch[...,2].mean(axis=(1,2,3))[:,None]/4
    mined=pool.mine(score)
    assert mined['candidates']==2 and set(pool.hard_negative)=={(0,1),(0,2)}
    x,y=pool.batch(8,np.random.default_rng(42),hard_fraction=1.)
    assert {int(a[...,2].mean()) for a,b in zip(x,y) if b==0}<={2,3}
    assert all(int(a[...,2].mean())==1 for a,b in zip(x,y) if b==1)


def test_negatives_near_excluded_manual_signals_remain_unknown():
    from dataclasses import replace
    rows=[dict(xyz=[x,0,0],channel=1,cell_id=1) for x in (0,2,4,7)]
    truth=[dict(xyz=[0,0,0],channel=1,cell_id=1)]
    exclusions=truth+[dict(xyz=[2,0,0],channel=1,cell_id=0),dict(xyz=[4,0,0],channel=None,cell_id=0)]
    config=replace(SignalConfig(),one_to_one_positives=True)
    assert candidate_targets(rows,truth,[True]*4,config,exclusion_points=exclusions).tolist()==[1,-1,-1,0]


def test_candidate_and_final_detection_separations_are_independent():
    from dataclasses import replace
    from fish_viewer.signals.evaluation import evaluate_cases
    rows=[dict(xyz=[x,0,0],channel=1,cell_id=1) for x in (0,.3)]
    cases=[dict(candidates=rows,truth=rows)]
    config=replace(SignalConfig(),candidate_separation_um=.25)
    assert evaluate_cases(cases,[np.array([.9,.8])],.5,config)['tp']==2
    final=replace(config,detection_separation_um=.35)
    assert final.validate().candidate_separation_um==.25
    assert evaluate_cases(cases,[np.array([.9,.8])],.5,final)['tp']==1


def test_exact_signal_cutoff_matches_brute_force_one_to_one_scores():
    from dataclasses import replace
    from fish_viewer.signals.evaluation import evaluate_cases
    rng=np.random.default_rng(27)
    cases=[];probabilities=[]
    for _ in range(3):
        refs=[dict(xyz=xyz.tolist(),channel=int(rng.integers(2)),cell_id=1) for xyz in rng.normal(0,.4,(8,3))]
        rows=[dict(xyz=xyz.tolist(),channel=int(rng.integers(2)),cell_id=1) for xyz in rng.normal(0,.4,(35,3))]
        cases.append(dict(candidates=rows,truth=refs));probabilities.append(rng.choice([.151,.463,.733,.912,.973],35))
    for target in (.3,.7,.9):
        config=replace(SignalConfig(),target_precision=target,candidate_separation_um=.1,detection_separation_um=.2)
        table=[dict(threshold=float(t),metrics=evaluate_cases(cases,probabilities,float(t),config)) for t in np.unique(np.concatenate(probabilities))]
        good=[r for r in table if r['metrics']['predictions']>=5 and (r['metrics']['precision'] or 0)>=target]
        expected=max(good,key=lambda r:(r['metrics']['recall'],r['metrics']['precision'],r['threshold'])) if good else max(table,key=lambda r:(r['metrics']['f05'],r['metrics']['precision'] or 0,r['threshold']))
        actual=select_threshold(cases,probabilities,config)
        assert actual['target_met']==bool(good)
        assert actual['selected']==expected


@pytest.mark.gui
@pytest.mark.parametrize('path_style',['native','windows'])
def test_viewer_loads_current_cell_archives_with_signal_overlays(qapp,ims_file,tmp_path,path_style):
    from fish_viewer.app import MainWindow
    from fish_viewer.models import LibraryEntry
    from fish_viewer.reader import load_dataset
    from fish_viewer.signals.engine import image_fingerprint
    dataset=load_dataset(LibraryEntry(ims_file));labels=np.ones(dataset.volumes[0].shape,np.uint32)
    source_image=str(ims_file) if path_style=='native' else 'C:\\FISH\\'+ims_file.name
    metadata=dict(version=1,image=source_image,channel=0,channel_scores=[],fingerprint=image_fingerprint(dataset),
                  spacing_zyx=list(dataset.spacing[::-1]),origin_xyz=list(dataset.center_origin),time=0,
                  detector=dict(engine='cell-center-box3d'))
    cells=tmp_path/'cells.npz';np.savez_compressed(cells,labels=labels,metadata=np.array(json.dumps(metadata)))
    report=dict(version=1,engine='signal-cnn3d',mode='apply',status='complete',image=str(ims_file),
                fingerprint=metadata['fingerprint'],cell_predictions=str(cells),
                channels=[dict(name='green',color=[0,1,0])],
                points=[dict(signal_id=1,cell_id=1,channel=0,xyz=list(dataset.center_origin),score=.95)])
    path=tmp_path/'signals.json';path.write_text(json.dumps(report))
    window=MainWindow(off_screen=True)
    try:
        window.run_job=lambda function,result,error=None,progress=None:result(function(lambda *_:None))
        window.open_signal_prediction(path)
        assert any(s.source=='prediction' for s in window.dataset.scene.surfaces)
        assert any(m.source=='signal_prediction' for m in window.dataset.scene.measurements)
        assert not any('Cannot load pipeline contours' in w for w in window.dataset.scene.warnings)
    finally:window.close();qapp.processEvents()


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
