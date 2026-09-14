"""Validate every image and XML scene without launching Qt or writing to source files."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fish_viewer.reader import ImarisReader, scan_library
from fish_viewer.scenes import read_scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--pixels', action='store_true', help='Also decode every channel at every time and resolution')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    entries, warnings = scan_library(args.folder)
    result = dict(entries=len(entries), images=0, scenes=0, surfaces=0, measurement_groups=0, unmeshed_contours=0,
        decoded_volumes=0, missing_images=[], warnings=warnings, errors=[], formats={})
    formats = Counter()
    for image in sorted({e.image for e in entries}):
        if not image.exists():
            result['missing_images'].append(str(image))
            continue
        try:
            reader = ImarisReader(image)
            formats[reader.metadata.format] += 1
            result['images'] += 1
            if args.pixels:
                for time in reader.metadata.times:
                    for level in reader.metadata.levels:
                        for volume in reader.read(time, level):
                            if volume.shape != reader.metadata.levels[level]:
                                raise ValueError('Unexpected decoded volume shape')
                            result['decoded_volumes'] += 1
            if result['images'] % 50 == 0:
                print(f'Checked {result["images"]} images', flush=True)
        except Exception as exc:
            result['errors'].append(dict(file=str(image), error=str(exc)))
    for path in sorted({e.scene for e in entries if e.scene}):
        try:
            scene = read_scene(path)
            result['scenes'] += 1
            result['surfaces'] += len(scene.surfaces)
            result['measurement_groups'] += len(scene.measurements)
            result['unmeshed_contours'] += len(scene.contours)
            result['warnings'].extend(f'{path}: {warning}' for warning in scene.warnings)
        except Exception as exc:
            result['errors'].append(dict(file=str(path), error=str(exc)))
    result['formats'] = dict(formats)
    output = json.dumps(result, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding='utf-8')
    return bool(result['errors'])


if __name__ == '__main__':
    sys.exit(main())
