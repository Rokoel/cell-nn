"""Shared worker startup without importing either neural backend into the GUI."""
import os
from pathlib import Path
import sys
import sysconfig
import threading


class Cancellation:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.event = threading.Event()

    def is_set(self):
        return self.event.is_set() or bool(self.path and self.path.exists())


def restart_with_cuda(module=None):
    """Expose pip's CUDA libraries before TensorFlow loads; restart at most once."""
    if not sys.platform.startswith('linux') or os.environ.get('FISH_CUDA_PATH_READY'):
        return
    libraries = sorted((Path(sysconfig.get_paths()['purelib']) / 'nvidia').glob('*/lib'))
    if not libraries:
        return
    environment = os.environ.copy()
    environment['FISH_CUDA_PATH_READY'] = '1'
    environment['LD_LIBRARY_PATH'] = os.pathsep.join(map(str, libraries)) + os.pathsep + environment.get('LD_LIBRARY_PATH', '')
    arguments = ['-m', module, *sys.argv[1:]] if module else sys.argv
    os.execve(sys.executable, [sys.executable, *arguments], environment)


def process_environment(directory=None, *, quiet_tensorflow=False, preserve_pythonpath=False):
    """Build the common QProcess environment for cancellable inference workers."""
    from PySide6 import QtCore

    environment = QtCore.QProcessEnvironment.systemEnvironment()
    root = str(Path(__file__).resolve().parents[1])
    if preserve_pythonpath:
        root += os.pathsep + environment.value('PYTHONPATH', '')
    environment.insert('PYTHONPATH', root)
    environment.insert('PYTHONUNBUFFERED', '1')
    environment.insert('OMP_NUM_THREADS', '4')
    if directory is not None:
        environment.insert('MPLCONFIGDIR', str(Path(directory) / 'matplotlib'))
    if quiet_tensorflow:
        environment.insert('TF_CPP_MIN_LOG_LEVEL', '2')
    return environment
