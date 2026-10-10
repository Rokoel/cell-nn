import json
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6 import QtCore

from fish_viewer.analysis.controller import AnalysisController, default_models, default_proposal_model, default_refinement_model
from fish_viewer.analysis.shape import filter_labels, apply
from fish_viewer.pipeline.segmentation import mask_properties
from fish_viewer.proposals.inference import signal_supported_boxes


def test_signal_gates_padded_boxes_without_moving_seed_or_assigning_signal_to_center():
    boxes=[dict(proposal_id=1, center_xyz=[10,20,30], box_low_xyz=[8,18,28], box_high_xyz=[12,22,32]),
           dict(proposal_id=2, center_xyz=[20,20,30], box_low_xyz=[18,18,28], box_high_xyz=[22,22,32])]
    assert signal_supported_boxes(boxes, [], 1) == []
    assert signal_supported_boxes(boxes, [dict(xyz=[12.5,20,30])], 1) == boxes[:1]
    assert signal_supported_boxes(boxes, [dict(xyz=[12.5,20,30])], 0) == []
    assert signal_supported_boxes(boxes, [dict(xyz=[10,20,30]),dict(xyz=[20,20,30])], 0) == boxes


def test_roundness_uses_physical_surface_and_does_not_change_accepted_cell():
    z,y,x=np.indices((48,48,96)); labels=np.zeros(z.shape,np.uint32)
    labels[(z-24)**2+(y-24)**2+(x-24)**2 < 9**2]=1
    labels[((z-24)/4)**2+((y-24)/4)**2+((x-68)/16)**2 < 1]=2
    a=mask_properties(labels==1,np.ones(3))['sphericity']
    b=mask_properties(labels==2,np.ones(3))['sphericity']
    assert a>b
    result, rows=filter_labels(labels,[1,1,1],(a+b)/2)
    assert np.array_equal(result==1, labels==1)
    assert not (result==2).any()
    assert rows[0]['accepted'] and not rows[1]['accepted']
    off,_=filter_labels(labels,[1,1,1],0)
    assert np.array_equal(off,labels)


def test_roundness_report_retains_calibration_and_signal_first_provenance(tmp_path):
    labels=np.zeros((12,)*3,np.uint32); labels[3:9,3:9,3:9]=7
    metadata=dict(cells=[dict(label=7)],spacing_zyx=[.3,.2,.1],origin_xyz=[10,20,30],
                  signal_first=dict(boxes_kept=1),fingerprint='same')
    path=tmp_path/'cells.npz';np.savez_compressed(path,labels=labels,metadata=np.array(json.dumps(metadata)))
    report=tmp_path/'report.json';report.write_text(json.dumps(dict(files=[dict(image='scan.ims',predictions=str(path))])))
    result=apply(report,tmp_path/'runs',0)
    assert result['status']=='complete'
    with np.load(result['files'][0]['predictions']) as f:
        after=json.loads(str(f['metadata']))
        assert np.array_equal(labels,f['labels'])
    assert after['origin_xyz']==metadata['origin_xyz']
    assert after['signal_first']==metadata['signal_first']


def test_signal_first_launch_order_and_roundness_stage(tmp_path,monkeypatch):
    controller=AnalysisController();launches=[]
    monkeypatch.setattr(controller,'_launch',lambda i,args: launches.append((i,args)))
    image=tmp_path/'scan.ims';image.touch()
    controller.start(image,tmp_path/'runs',default_proposal_model(),default_models()[1],
                     refinement=default_refinement_model(),device='cpu',signal_first=True,roundness=.72)
    assert [s[0] for s in controller.stages]==['scout','cells','shape','signals','filter','measure','export']
    assert '--mode' in launches[0][1] and 'scout' in launches[0][1]
    assert controller.state['minimum_roundness']==.72
    controller.stage='scout';controller.stage_index=0
    controller.finished_report=dict(status='complete',directory=str(tmp_path/'scout'))
    controller.job=SimpleNamespace(deleteLater=lambda:None)
    monkeypatch.setattr(controller,'_read',lambda:None)
    monkeypatch.setattr(controller,'_read_errors',lambda:None)
    controller._finished(0)
    assert launches[1][0]==1 and '--signal-candidates' in launches[1][1]
    assert launches[1][1][-1]==str(tmp_path/'scout/report.json')


@pytest.mark.gui
def test_analysis_presets_remove_training_and_model_pickers(qapp,tmp_path):
    from PySide6 import QtWidgets
    from fish_viewer.analysis.app import AnalysisWindow
    settings=QtCore.QSettings(str(tmp_path/'settings.ini'),QtCore.QSettings.Format.IniFormat)
    window=AnalysisWindow(settings=settings)
    try:
        assert window.roundness.value()==72
        assert not hasattr(window,'boundary_training_button')
        assert not hasattr(window,'proposal_training_button')
        assert all('Train ' not in b.text() for b in window.findChildren(QtWidgets.QPushButton))
        assert window.cell_model.parent() is None  # No editable folder picker in Analysis.
        window.pipeline.setCurrentIndex(window.pipeline.findData('signal-first'))
        assert window.cell_model.text()==default_proposal_model()
        assert window.boundary_model.text()==default_refinement_model()
        window.pipeline.setCurrentIndex(window.pipeline.findData('balanced'))
        assert window.cell_model.text().endswith('/models/cells')
        assert window.boundary_model.text()==''
        assert not window.stardist_proposals.isChecked()
    finally:
        window.close()


def test_roundness_accounts_for_anisotropic_spacing():
    z,y,x=np.indices((40,)*3)
    labels=(((z-20)/4)**2+((y-20)/12)**2+((x-20)/12)**2<1).astype(np.uint32)
    physical=mask_properties(labels>0,np.array([3,1,1]))['sphericity']
    voxel=mask_properties(labels>0,np.ones(3))['sphericity']
    assert physical>voxel
    threshold=(physical+voxel)/2
    accepted,_=filter_labels(labels,[3,1,1],threshold)
    rejected,_=filter_labels(labels,[1,1,1],threshold)
    assert np.array_equal(accepted,labels) and not rejected.any()
