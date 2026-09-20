import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description='FISH Analysis: automatic cells, signals, A/B distances and Imaris export.')
    parser.add_argument('paths', nargs='*', help='Imaris scans or folders to add to the app')
    args = parser.parse_args()
    os.environ['QT_API'] = 'pyside6'
    if sys.platform.startswith('linux') and os.environ.get('QT_QPA_PLATFORM') not in ('offscreen', 'minimal'):
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
    from PySide6 import QtWidgets
    from .app import AnalysisWindow
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName('FISH Analysis')
    app.setOrganizationName('Cell NN')
    app.setStyle('Fusion')
    window = AnalysisWindow()
    window.show()
    if args.paths:
        window.add_paths(args.paths)
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
