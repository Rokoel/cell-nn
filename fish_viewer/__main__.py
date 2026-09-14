from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description='View Imaris images, cell surfaces, and measurement annotations.')
    parser.add_argument('paths', nargs='*', help='An .ims image, .imx scene, or folder to index recursively')
    parser.add_argument('--inspect', action='store_true', help='Print image and annotation metadata without starting the GUI')
    args = parser.parse_args()
    if args.inspect:
        import json
        from .reader import ImarisReader, entry_for_path
        from .scenes import read_scene
        for path in args.paths:
            entry = entry_for_path(path)
            reader = ImarisReader(entry.image)
            scene = read_scene(entry.scene) if entry.scene else reader.embedded_scene()
            m = reader.metadata
            print(json.dumps(dict(image=str(entry.image), scene=str(entry.scene) if entry.scene else None,
                shape_zyx=m.shape, spacing_xyz=m.spacing, unit=m.unit, channels=[c.name for c in m.channels],
                format=m.format, times=m.times, resolutions=m.levels, surfaces=len(scene.surfaces),
                measurement_groups=len(scene.measurements), unmeshed_contours=len(scene.contours),
                warnings=m.warnings + scene.warnings), indent=2))
        return
    os.environ['QT_API'] = 'pyside6'
    # VTK's Qt native window interactor requires an X11 window on Linux.
    # XWayland provides this when the desktop session itself uses Wayland.
    if sys.platform.startswith('linux') and os.environ.get('QT_QPA_PLATFORM') not in ('offscreen', 'minimal'):
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
    from PySide6 import QtWidgets
    from .app import MainWindow
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName('FISH Viewer')
    app.setOrganizationName('Cell NN')
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    if args.paths:
        window.open_paths([Path(p) for p in args.paths])
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
