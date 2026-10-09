"""Build a portable source bundle with the existing inference checkpoints."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys

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
                 'run_viewer.py', 'run-viewer.sh', 'run-viewer.bat',
                 'setup-analysis.sh', 'setup-analysis.bat', 'setup-spots.sh', 'setup-spots.bat', 'requirements-spots.txt'):
        shutil.copy2(ROOT / name, destination / name)
    guide=(ROOT / 'docs/fish-analysis.md').read_text(encoding='utf-8')
    # This guide normally lives under docs/; make its links work from the ZIP root.
    def guide_link(match):
        target=match[1]
        if target.startswith('../'):
            target=target[3:]
        elif '/' not in target and target.endswith('.md'):
            target='docs/'+target
        return ']('+target+')'
    (destination / 'README.md').write_text(re.sub(r'\]\(([^)]+)\)',guide_link,guide),encoding='utf-8')
    (destination / 'scripts').mkdir()
    for name in ('verify_release.py', 'package_analysis.py', 'benchmark_measurements.py', 'smoke_refinement.py',
                 'calibrate_boundary_model.py', 'compare_refined_boundaries.py', 'audit_signal_candidates.py',
                 'refresh_signal_data.py', 'train_refreshed_signals.py', 'compare_signal_models.py', 'calibrate_signal_model.py',
                 'calibrate_membrane_snap.py', 'build_signal_centers.py', 'train_signal_centers.py',
                 'evaluate_signal_centers.py', 'report_signal_centers.py', 'plot_signal_centers.py',
                 'evaluate_center_measurements.py', 'experiment_contour_bias.py', 'train_contour_bias.py',
                 'build_contour_bias_patches.py', 'run_contour_bias_training.py', 'run_contour_correction_control.py',
                 'finalize_contour_bias.py', 'evaluate_contour_bias_signals.py', 'report_contour_bias.py',
                 'plot_contour_bias_overlays.py', 'check_contour_correction.py'):
        shutil.copy2(ROOT / 'scripts' / name, destination / 'scripts' / name)
    shutil.copy2(ROOT / 'scripts/calibrate_cell_proposals.py', destination / 'scripts/calibrate_cell_proposals.py')
    shutil.copy2(ROOT / 'scripts/evaluate_cell_centers.py', destination / 'scripts/evaluate_cell_centers.py')
    shutil.copy2(ROOT / 'scripts/compare_cell_pipelines.py', destination / 'scripts/compare_cell_pipelines.py')
    shutil.copy2(ROOT / 'scripts/evaluate_cell_contours.py', destination / 'scripts/evaluate_cell_contours.py')
    shutil.copy2(ROOT / 'scripts/check_cell_crop_controls.py', destination / 'scripts/check_cell_crop_controls.py')
    (destination / 'docs').mkdir()
    for name in ('fish-analysis.md', 'imaris-export.md', 'signal-detection.md', 'neural-training.md', 'models.md'):
        shutil.copy2(ROOT / 'docs' / name, destination / 'docs' / name)
    for name in ('measurement-accuracy.md', 'releasing.md', 'release-notes.md'):
        if (ROOT / 'docs' / name).exists():
            shutil.copy2(ROOT / 'docs' / name, destination / 'docs' / name)
    if (ROOT / 'docs/boundary-refinement.md').exists():
        shutil.copy2(ROOT / 'docs/boundary-refinement.md', destination / 'docs/boundary-refinement.md')
    if (ROOT / 'docs/center-box-contours.md').exists():
        shutil.copy2(ROOT / 'docs/center-box-contours.md', destination / 'docs/center-box-contours.md')
    if (ROOT / 'docs/membrane-snap.md').exists():
        shutil.copy2(ROOT / 'docs/membrane-snap.md', destination / 'docs/membrane-snap.md')
    if (ROOT / 'docs/contour-bias.md').exists():
        shutil.copy2(ROOT / 'docs/contour-bias.md', destination / 'docs/contour-bias.md')
    if (ROOT / 'reports/contour-bias').exists():
        shutil.copytree(ROOT / 'reports/contour-bias', destination / 'reports/contour-bias')
    reports = ROOT / 'reports/measurement-accuracy'
    if reports.exists():
        shutil.copytree(reports, destination / 'reports/measurement-accuracy')
    boundary_reports = ROOT / 'reports/boundary-refinement'
    if boundary_reports.exists():
        shutil.copytree(boundary_reports, destination / 'reports/boundary-refinement')
    proposal_reports = ROOT / 'reports/center-box-contours'
    if proposal_reports.exists():
        shutil.copytree(proposal_reports, destination / 'reports/center-box-contours')
    signal_reports = ROOT / 'reports/signal-refresh'
    if signal_reports.exists():
        shutil.copytree(signal_reports, destination / 'reports/signal-refresh')
    membrane_reports = ROOT / 'reports/membrane-snap'
    if membrane_reports.exists():
        shutil.copytree(membrane_reports, destination / 'reports/membrane-snap')
    if (ROOT / 'docs/signal-centers.md').exists():
        shutil.copy2(ROOT / 'docs/signal-centers.md', destination / 'docs/signal-centers.md')
    if (ROOT / 'reports/signal-centers').exists():
        shutil.copytree(ROOT / 'reports/signal-centers', destination / 'reports/signal-centers')
    provenance = {}
    model_sources = [('cells', cells), ('signals', ROOT / 'models/signals')]
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
