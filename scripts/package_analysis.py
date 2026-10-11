"""Build a portable source bundle with the existing inference checkpoints."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fish_viewer.analysis.controller import default_models, default_refinement_model, default_proposal_model, check_models


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
                 'run_annotator.py', 'run-annotator.sh', 'run-annotator.bat',
                 'run_viewer.py', 'run-viewer.sh', 'run-viewer.bat',
                 'setup-analysis.sh', 'setup-analysis.bat', 'setup-windows-runtime.ps1', 'setup-spots.sh', 'setup-spots.bat',
                 'requirements-spots.txt', 'requirements-build.txt', 'requirements-cuda-bundle.txt'):
        shutil.copy2(ROOT / name, destination / name)
    guide=(ROOT / 'docs/fish-analysis.md').read_text(encoding='utf-8')
    # This guide normally lives under docs/; make its links work from the ZIP root.
    def guide_link(match):
        target=match[1]
        if target.startswith('../'):
            target=target[3:]
        elif not urlsplit(target).scheme and not target.startswith(('#', '/')):
            target='docs/'+target
        return ']('+target+')'
    (destination / 'README.md').write_text(re.sub(r'\]\(([^)]+)\)',guide_link,guide),encoding='utf-8')
    for name in ('scripts', 'docs', 'reports', 'tests', 'packaging'):
        shutil.copytree(ROOT / name, destination / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    provenance = {}
    model_sources = [('cells', ROOT / 'models/cells'), ('cells-conservative', ROOT / 'models/cells-conservative'),
                     ('signals', ROOT / 'models/signals')]
    from fish_viewer.boundary.model import read_model as read_boundary_model
    read_boundary_model(ROOT / 'models/boundary-annotations')
    model_sources.append(('boundary-annotations', ROOT / 'models/boundary-annotations'))
    check_models(ROOT / 'models/cells-sensitive', signals)
    model_sources.append(('cells-sensitive', ROOT / 'models/cells-sensitive'))
    baseline_signals = ROOT / 'models/signals-base'
    if (baseline_signals / 'signal-model.json').is_file():
        model_sources.append(('signals-base', baseline_signals))
    refreshed_signals = ROOT / 'models/signals-refreshed'
    if (refreshed_signals / 'signal-model.json').is_file():
        model_sources.append(('signals-refreshed', refreshed_signals))
    for label in ('signals-center-unet', 'signals-spotiflow'):
        source = ROOT / 'models' / label
        if (source / 'signal-model.json').is_file():
            from fish_viewer.signals.model import read_model as read_signal_model
            read_signal_model(source)
            model_sources.append((label, source))
    boundary = default_refinement_model()
    if boundary:
        from fish_viewer.refinement.config import read_model
        read_model(boundary)
        model_sources.append((Path(boundary).name, Path(boundary)))
    proposals = default_proposal_model()
    if proposals:
        from fish_viewer.proposals.config import read_model as read_proposal_model
        read_proposal_model(proposals)
        model_sources.append(('proposals', Path(proposals)))
    for label in ('refinement', 'refinement-base', 'refinement-bias-candidate'):
        source = ROOT / 'models' / label
        if (source / 'refinement-model.json').is_file() and label not in {name for name, _ in model_sources}:
            from fish_viewer.refinement.config import read_model
            read_model(source)
            model_sources.append((label, source))
    enhanced = ROOT / 'models/cells-preprocessed'
    if (enhanced / 'fish-model.json').is_file():
        check_models(enhanced, signals)
        model_sources.append(('cells-preprocessed', enhanced))
    sliced = ROOT / 'models/cells-stardist2d'
    if (sliced / 'fish-model.json').is_file():
        check_models(sliced, signals)
        model_sources.append(('cells-stardist2d', sliced))
    for label, source in model_sources:
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
