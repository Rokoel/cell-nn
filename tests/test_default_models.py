from pathlib import Path
from PySide6 import QtCore
from fish_viewer.analysis.controller import default_models, default_proposal_model, default_refinement_model, migrate_bundled_defaults


def test_bundled_defaults_match_requested_pipeline():
    assert default_models()[1].name == 'signals-center-unet'
    assert Path(default_proposal_model()).name == 'proposals'
    assert Path(default_refinement_model()).name == 'refinement'


def test_upgrade_old_defaults_once_and_preserve_later_selection(tmp_path):
    settings = QtCore.QSettings(str(tmp_path / 'settings.ini'), QtCore.QSettings.Format.IniFormat)
    root = default_models()[1].parent
    settings.setValue('signal_model', str(root / 'signals'))
    settings.setValue('cell_model', str(root / 'cells'))
    migrate_bundled_defaults(settings)
    assert settings.value('signal_model') == str(default_models()[1])
    assert settings.value('cell_model') == default_proposal_model()
    assert settings.value('boundary_model') == default_refinement_model()
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
