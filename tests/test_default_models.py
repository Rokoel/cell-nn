from pathlib import Path
from PySide6 import QtCore
from fish_viewer.analysis.controller import default_models, default_proposal_model, default_refinement_model, migrate_bundled_defaults


def test_bundled_defaults_match_requested_pipeline():
    assert default_models()[0].name == 'cells-sensitive'
    assert default_models()[1].name == 'signals-center-unet'
    assert Path(default_proposal_model()).name == 'proposals'
    assert Path(default_refinement_model()).name == 'refinement-bias-candidate'


def test_upgrade_old_defaults_once_and_preserve_later_selection(tmp_path):
    settings = QtCore.QSettings(str(tmp_path / 'settings.ini'), QtCore.QSettings.Format.IniFormat)
    root = default_models()[1].parent
    settings.setValue('signal_model', str(root / 'signals'))
    settings.setValue('cell_model', str(root / 'cells'))
    migrate_bundled_defaults(settings)
    assert settings.value('signal_model') == str(default_models()[1])
    assert settings.value('cell_model') == str(default_models()[0])
    assert settings.value('boundary_model') == ''
    settings.setValue('signal_model', str(root / 'signals'))
    migrate_bundled_defaults(settings)
    assert settings.value('signal_model') == str(root / 'signals')


def test_upgrade_preserves_custom_models(tmp_path):
    settings = QtCore.QSettings(str(tmp_path / 'settings.ini'), QtCore.QSettings.Format.IniFormat)
    for key in ('cell_model', 'boundary_model', 'signal_model'):
        settings.setValue(key, str(tmp_path / key))
    migrate_bundled_defaults(settings)
    for key in ('cell_model', 'boundary_model', 'signal_model'):
        assert settings.value(key) == str(tmp_path / key)


def test_upgrade_v040_contour_default_once(tmp_path):
    settings = QtCore.QSettings(str(tmp_path / 'settings.ini'), QtCore.QSettings.Format.IniFormat)
    root = default_models()[1].parent
    settings.setValue('default_pipeline_version', '0.4.0')
    settings.setValue('cell_model', default_proposal_model())
    settings.setValue('signal_model', str(default_models()[1]))
    settings.setValue('boundary_model', str(root / 'refinement'))
    migrate_bundled_defaults(settings)
    assert settings.value('boundary_model') == ''
    assert settings.value('default_pipeline_version') == 'consensus-editable-1'
    settings.setValue('boundary_model', str(root / 'refinement'))
    migrate_bundled_defaults(settings)
    assert settings.value('boundary_model') == str(root / 'refinement')


def test_upgrade_v041_disables_all_contour_modifications(tmp_path):
    settings = QtCore.QSettings(str(tmp_path / 'settings.ini'), QtCore.QSettings.Format.IniFormat)
    settings.setValue('default_pipeline_version', '0.4.1')
    settings.setValue('cell_model', default_proposal_model())
    settings.setValue('boundary_model', default_refinement_model())
    flags = ('membrane_snap', 'learned_boundary', 'cell_recovery', 'stardist_proposals', 'consensus_cells')
    for key in flags:
        settings.setValue(key, True)
    migrate_bundled_defaults(settings)
    assert settings.value('cell_model') == str(default_models()[0])
    assert settings.value('boundary_model') == ''
    assert all(settings.value(key, type=bool) is False for key in flags)
    settings.setValue('cell_model', default_proposal_model())
    migrate_bundled_defaults(settings)
    assert settings.value('cell_model') == default_proposal_model()


def test_conservative_worker_uses_saved_thresholds_and_original_surfaces(tmp_path, monkeypatch):
    import json
    from fish_viewer.analysis.controller import AnalysisController
    controller = AnalysisController()
    launches = []
    monkeypatch.setattr(controller, '_launch', lambda stage, args: launches.append((stage, args)))
    image = tmp_path / 'scan.ims'; image.touch()
    controller.start(image, tmp_path / 'runs', default_models()[0].parent/'cells-conservative', default_models()[1], device='cpu')
    request = json.loads((controller.directory / 'cells-request.json').read_text())
    assert request['config']['probability_threshold'] == .75
    assert request['config']['nms_threshold'] == .15
    assert request['config']['membrane_support'] == .65
    assert request['config']['use_shape_filter'] is False
    assert [name for name, _ in controller.stages] == ['cells', 'signals', 'filter', 'measure', 'export']
    assert controller.state['refinement_model'] is None
    assert controller.state['boundary_guidance'] is None
    assert launches[0][0] == 0


def test_conservative_default_migrates_to_2d_but_explicit_other_preset_survives(tmp_path):
    for selected,expected in [('conservative','consensus'),('hybrid','hybrid')]:
        settings=QtCore.QSettings(str(tmp_path/(selected+'.ini')),QtCore.QSettings.Format.IniFormat)
        settings.setValue('default_pipeline_version','conservative-stardist-1')
        settings.setValue('pipeline',selected)
        migrate_bundled_defaults(settings)
        assert settings.value('pipeline')==expected
        assert settings.value('default_pipeline_version')=='consensus-editable-1'
