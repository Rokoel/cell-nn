from dataclasses import asdict
import json
from pathlib import Path

import pytest
from PySide6 import QtCore
from fish_viewer.analysis import controller as module
from fish_viewer.neural.config import NeuralConfig


@pytest.mark.parametrize('hybrid', [False, True])
def test_hybrid_preserves_saved_detector_sensitivity_and_marks_run(tmp_path,monkeypatch,hybrid):
    application=QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    saved=NeuralConfig(probability_threshold=.35,precision_mode=False)
    monkeypatch.setattr(module,'check_models',lambda *_:dict(engine='stardist3d',config=asdict(saved)))
    from fish_viewer.refinement import config
    monkeypatch.setattr(config,'read_model',lambda _: {})
    image=tmp_path/'image.ims';image.touch()
    controller=module.AnalysisController();calls=[]
    monkeypatch.setattr(controller,'_launch',lambda index,args:calls.append((index,args)))
    controller.start(image,tmp_path/'runs',tmp_path/'cells',tmp_path/'signals',device='cpu',
                     refinement=tmp_path/'boundaries',stardist_proposals=hybrid)
    request=json.loads((controller.directory/'cells-request.json').read_text())
    assert request['config']['probability_threshold']==(.35 if hybrid else .55)
    assert request['config']['precision_mode'] is (not hybrid)
    assert controller.state['stardist_proposals'] is hybrid
    assert calls[0][0]==0


def test_hybrid_requires_crop_boundary_model(tmp_path,monkeypatch):
    application=QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    monkeypatch.setattr(module,'check_models',lambda *_:dict(engine='stardist3d',config=asdict(NeuralConfig())))
    with pytest.raises(ValueError,match='Boundaries model'):
        module.AnalysisController().start('image.ims',tmp_path,'cells','signals',stardist_proposals=True)
