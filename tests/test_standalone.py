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
