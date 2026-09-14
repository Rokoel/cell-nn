import os
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['QT_API'] = 'pyside6'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/fish-viewer-test-mpl' if os.name != 'nt' else str(Path(os.environ['TEMP']) / 'fish-viewer-test-mpl'))

import h5py
import numpy as np
import pytest


@pytest.fixture
def ims_file(tmp_path):
    """Two channels/times with padding that must never leak into the image."""
    path = tmp_path / 'sample.ims'
    with h5py.File(path, 'w') as f:
        image = f.create_group('DataSetInfo/Image')
        info = dict(ExtMin0='10', ExtMin1='20', ExtMin2='30', ExtMax0='18', ExtMax1='26', ExtMax2='34', Unit='um')
        for key, value in info.items():
            image.attrs[key] = np.frombuffer(value.encode(), dtype='S1')
        for channel, color in enumerate(['0 1 0', '1 0 0']):
            g = f.create_group(f'DataSetInfo/Channel {channel}')
            for key, value in dict(Name=f'Fluor {channel}', Color=color, ColorRange='0 100', GammaCorrection='1').items():
                g.attrs[key] = np.frombuffer(value.encode(), dtype='S1')
        for level, shape in [(0, (2, 3, 4)), (1, (2, 2, 2))]:
            for time in (0, 1):
                for channel in (0, 1):
                    g = f.create_group(f'DataSet/ResolutionLevel {level}/TimePoint {time}/Channel {channel}')
                    for axis, n in zip('ZYX', shape):
                        g.attrs[f'ImageSize{axis}'] = str(n).encode()
                    data = np.full((4, 4, 6), 65535, dtype=np.uint16)
                    data[:shape[0], :shape[1], :shape[2]] = 10 + time * 20 + channel * 5
                    g.create_dataset('Data', data=data)
    return path


@pytest.fixture(scope='session')
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    return app


@pytest.fixture
def reference():
    path = Path(__file__).parents[1] / 'FISH/3abx3ad_101212/141212_3ab3ad_Lam633m_dsRed543_CG488b.ims'
    if not path.exists():
        pytest.skip('Local FISH reference data is not present')
    return path
