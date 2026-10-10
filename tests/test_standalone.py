import sys

import pytest

from fish_viewer import standalone


def test_frozen_signal_runtime_ignores_external_python(monkeypatch):
    from fish_viewer.spots.runtime import interpreter

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv("FISH_SPOTS_PYTHON", "/does/not/exist")
    assert interpreter() == sys.executable


def test_worker_dispatch_preserves_arguments_and_exit(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["FISH-Analysis", "-m", "fish_viewer.signals", "--mode", "controls"],
    )
    calls = []

    def run(module, **options):
        calls.append((module, sys.argv[:], options))
        raise SystemExit(7)

    monkeypatch.setattr(standalone.runpy, "run_module", run)
    with pytest.raises(SystemExit) as error:
        standalone.main()
    assert error.value.code == 7
    assert calls == [
        (
            "fish_viewer.signals",
            ["fish_viewer.signals", "--mode", "controls"],
            {"run_name": "__main__", "alter_sys": True},
        )
    ]


@pytest.mark.parametrize("args", [["-m"], ["-m", "os"], ["-m", "pip"]])
def test_arbitrary_python_modules_cannot_be_launched(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["FISH-Analysis", *args])
    with pytest.raises(SystemExit, match="Unsupported FISH worker"):
        standalone.main()


@pytest.mark.parametrize('cuda,saved,expected', [(False, 'gpu', 'cpu'), (True, 'gpu', 'gpu'), (True, 'auto', 'auto')])
def test_packaged_device_selection_matches_runtime(monkeypatch, tmp_path, cuda, saved, expected):
    import json
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6 import QtWidgets
    from fish_viewer.runtime_profile import configure_device_selector
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, '_MEIPASS', str(tmp_path), raising=False)
    (tmp_path / 'runtime-profile.json').write_text(json.dumps({'cuda': cuda}))
    selector = QtWidgets.QComboBox()
    configure_device_selector(selector, saved)
    assert selector.currentData() == expected
    assert (selector.findData('gpu') >= 0) == cuda
    selector.close()


def test_runtime_checks_use_separate_workers(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['FISH-Analysis', '--check-runtime', '--device', 'gpu'])
    calls = []
    monkeypatch.setattr(standalone.subprocess, 'run', lambda command, **kwargs: calls.append((command, kwargs)))
    assert standalone.main() == 0
    assert [call[0][-3:] for call in calls] == [['tensorflow', '--device', 'gpu'], ['torch', '--device', 'gpu']]
    assert all(call[1]['check'] for call in calls)
