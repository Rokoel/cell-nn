"""Build a portable source bundle with the existing inference checkpoints."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fish_viewer.analysis.controller import default_models, check_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/FISH-Analysis')
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists() or destination.with_suffix('.zip').exists():
        parser.error('Choose a new output path; existing bundles are never overwritten.')
    cells, signals = default_models()
    check_models(cells, signals)
    destination.mkdir(parents=True)
    shutil.copytree(ROOT / 'fish_viewer', destination / 'fish_viewer',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('pyproject.toml', 'run_analysis.py', 'run-analysis.sh', 'run-analysis.bat',
                 'setup-analysis.sh', 'setup-analysis.bat'):
        shutil.copy2(ROOT / name, destination / name)
    shutil.copy2(ROOT / 'docs/fish-analysis.md', destination / 'README.md')
    (destination / 'scripts').mkdir()
    for name in ('verify_release.py', 'package_analysis.py', 'benchmark_measurements.py'):
        shutil.copy2(ROOT / 'scripts' / name, destination / 'scripts' / name)
    (destination / 'docs').mkdir()
    for name in ('fish-analysis.md', 'imaris-export.md', 'signal-detection.md', 'neural-training.md'):
        shutil.copy2(ROOT / 'docs' / name, destination / 'docs' / name)
    for name in ('measurement-accuracy.md', 'releasing.md'):
        if (ROOT / 'docs' / name).exists():
            shutil.copy2(ROOT / 'docs' / name, destination / 'docs' / name)
    reports = ROOT / 'reports/measurement-accuracy'
    if reports.exists():
        shutil.copytree(reports, destination / 'reports/measurement-accuracy')
    provenance = {}
    for label, source in [('cells', cells), ('signals', signals)]:
        target = destination / 'models' / label
        target.mkdir(parents=True)
        copied = {}
        for path in source.iterdir():
            if path.is_file() and path.name != 'last.weights.h5':
                shutil.copy2(path, target / path.name)
                with (target / path.name).open('rb') as stream:
                    copied[path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        provenance[label] = dict(source=str(source), sha256=copied)
    (destination / 'models/origins.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    versions = {name: importlib.metadata.version(name) for name in (
        'PySide6', 'vtk', 'pyvista', 'pyvistaqt', 'numpy', 'scipy', 'scikit-image',
        'h5py', 'tensorflow', 'keras', 'stardist', 'openpyxl')}
    (destination / 'tested-runtime.json').write_text(json.dumps(dict(platform=sys.platform, versions=versions), indent=2), encoding='utf-8')
    archive = shutil.make_archive(str(destination), 'zip', destination.parent, destination.name)
    print(json.dumps(dict(bundle=str(destination), archive=archive, models=provenance), indent=2))


if __name__ == '__main__':
    main()
