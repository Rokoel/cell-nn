"""Capture the actual Analysis window with a read-only scan and isolated settings."""
import argparse
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', type=Path, default=ROOT / 'FISH/3abx3ad_101212/141212_3ab3ad_Lam633m_dsRed543_CG488b.imx')
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/images/analysis.png')
    args = parser.parse_args()
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6 import QtCore, QtWidgets
    from fish_viewer.analysis.app import AnalysisWindow

    app = QtWidgets.QApplication([])
    app.setStyle('Fusion')
    with tempfile.TemporaryDirectory(prefix='fish-docs-') as temporary:
        settings = QtCore.QSettings(str(Path(temporary) / 'settings.ini'), QtCore.QSettings.Format.IniFormat)
        window = AnalysisWindow(off_screen=True, settings=settings)
        window.resize(1280, 840)
        window.output.setText(str(Path(temporary) / 'results'))
        errors = []
        window.error = errors.append
        window.show()
        window.add_paths([args.scan.resolve()])
        deadline = time.monotonic() + 60
        while (window.dataset is None or window.loading or window._jobs) and not errors:
            app.processEvents()
            if time.monotonic() >= deadline:
                raise TimeoutError('The scan did not load within 60 seconds.')
            time.sleep(.01)
        if errors:
            raise RuntimeError('\n'.join(errors))
        window.slice_view.fit()
        app.processEvents()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not window.grab().save(str(args.output)):
            raise OSError(f'Could not save screenshot: {args.output}')
        window.close()
        app.processEvents()
    print(args.output)


if __name__ == '__main__':
    main()
