import os
import sys

import pytest

from fish_viewer import worker_runtime


def test_cuda_restart_preserves_arguments_and_happens_once(monkeypatch, tmp_path):
    library = tmp_path / 'nvidia/cudnn/lib'
    library.mkdir(parents=True)
    monkeypatch.setattr(sys, 'platform', 'linux')
    monkeypatch.setattr(sys, 'argv', ['worker', '--device', 'cpu'])
    monkeypatch.setattr(worker_runtime.sysconfig, 'get_paths', lambda: {'purelib': str(tmp_path)})
    monkeypatch.delenv('FISH_CUDA_PATH_READY', raising=False)
    monkeypatch.setenv('LD_LIBRARY_PATH', '/existing')
    calls = []
    monkeypatch.setattr(worker_runtime.os, 'execve', lambda *args: calls.append(args))
    worker_runtime.restart_with_cuda('fish_viewer.neural')
    executable, arguments, environment = calls[0]
    assert executable == sys.executable
    assert arguments == [sys.executable, '-m', 'fish_viewer.neural', '--device', 'cpu']
    assert environment['LD_LIBRARY_PATH'] == f'{library}{os.pathsep}/existing'
    assert environment['FISH_CUDA_PATH_READY'] == '1'
    monkeypatch.setenv('FISH_CUDA_PATH_READY', '1')
    worker_runtime.restart_with_cuda('fish_viewer.neural')
    assert len(calls) == 1


@pytest.mark.parametrize('platform', ['linux', 'win32'])
def test_worker_without_cuda_libraries_does_not_restart(monkeypatch, tmp_path, platform):
    monkeypatch.setattr(sys, 'platform', platform)
    monkeypatch.setattr(worker_runtime.sysconfig, 'get_paths', lambda: {'purelib': str(tmp_path)})
    monkeypatch.delenv('FISH_CUDA_PATH_READY', raising=False)
    monkeypatch.setattr(worker_runtime.os, 'execve', lambda *args: pytest.fail('Unexpected restart'))
    worker_runtime.restart_with_cuda()


def test_cancellation_handles_event_and_file(tmp_path):
    path = tmp_path / 'cancel'
    cancellation = worker_runtime.Cancellation(path)
    assert not cancellation.is_set()
    path.touch()
    assert cancellation.is_set()
    path.unlink()
    cancellation.event.set()
    assert cancellation.is_set()


def test_worker_environment_keeps_runtime_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv('PYTHONPATH', '/custom/modules')
    monkeypatch.setenv('FISH_SPOTS_PYTHON', '/custom/python')
    environment = worker_runtime.process_environment(tmp_path, quiet_tensorflow=True, preserve_pythonpath=True)
    assert environment.value('FISH_SPOTS_PYTHON') == '/custom/python'
    assert environment.value('PYTHONPATH').endswith('/custom/modules')
    assert environment.value('MPLCONFIGDIR') == str(tmp_path / 'matplotlib')
    assert environment.value('TF_CPP_MIN_LOG_LEVEL') == '2'
