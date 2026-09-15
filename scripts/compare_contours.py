"""Compare a completed neural run with fixed classical settings on its test files."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from fish_viewer.neural.config import NeuralConfig
from fish_viewer.neural.data import is_cut, subset_metrics
from fish_viewer.neural.engine import aggregate_subsets
from fish_viewer.pipeline.config import PipelineConfig
from fish_viewer.pipeline.evaluation import evaluate, rasterize_reference
from fish_viewer.pipeline.runner import _json, save_prediction
from fish_viewer.pipeline.segmentation import prepare, segment
from fish_viewer.models import LibraryEntry
from fish_viewer.reader import load_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--neural-report',required=True,type=Path)
    parser.add_argument('--classical-settings',required=True,type=Path)
    args=parser.parse_args()
    report=json.loads(args.neural_report.read_text())
    if report.get('engine')!='stardist3d' or report.get('status')!='complete':
        parser.error('Select a completed StarDist training report.')
    tests=[row for row in report['files'] if row['split']=='test']
    if not tests:parser.error('The selected report has no test images.')
    neural=NeuralConfig.from_dict(report['config'])
    classical=PipelineConfig.from_dict(json.loads(args.classical_settings.read_text())['config'])
    keys=('voxel_um','min_diameter_um','max_diameter_um','min_sphericity','max_elongation','exclude_border')
    classical=replace(classical,**{key:getattr(neural,key) for key in keys})
    rows=[]
    for row in tests:
        entry=LibraryEntry(Path(row['image']),Path(row['scene']) if row.get('scene') else None)
        dataset=load_dataset(entry,time=0,level=0)
        prepared=prepare(dataset,classical,max_voxels=None)
        with np.load(row['predictions'],allow_pickle=False) as saved:
            metadata=json.loads(str(saved['metadata']))
            if (prepared.fingerprint!=metadata['fingerprint'] or prepared.image.shape!=saved['labels'].shape
                    or not np.allclose(prepared.spacing,metadata['spacing_zyx'])
                    or not np.allclose(prepared.origin,metadata['origin_xyz'])):
                raise ValueError('The source image or calibrated processing grid differs from the neural run.')
        refs,names,skipped=rasterize_reference(dataset.scene.surfaces,prepared,0)
        case=SimpleNamespace(prepared=prepared,references=refs,names=names,
            reference_cut=[is_cut(box,mask,prepared.image.shape) for box,mask in refs])
        predicted=segment(prepared,classical)
        metrics=subset_metrics(predicted.labels,case,evaluate)
        if metrics['reference_count']!=row['metrics']['reference_count']:
            raise ValueError('Reference annotation counts differ from the neural run.')
        path=Path(row['predictions']).with_name('classical-comparison.npz')
        save_prediction(path,predicted,prepared,entry,classical,metrics)
        rows.append(dict(image=row['image'],predictions=str(path),metrics=metrics,warnings=skipped))
    result=dict(neural_report=str(args.neural_report.resolve()),
        classical_settings=str(args.classical_settings.resolve()),
        common_geometry={key:getattr(neural,key) for key in keys},
        classical=aggregate_subsets([r['metrics'] for r in rows]),
        neural=aggregate_subsets([r['metrics'] for r in tests]),files=rows,
        notes='Fixed previously selected classical settings; no new tuning. Same test files, reference meshes, '
        'processing grid and physical filters. Each method keeps its own preprocessing. '
        'Unknown predictions are unscored; precision cannot be measured. Exploratory comparison on an acquisition '
        'already inspected during earlier development, not a fresh generalization test.')
    directory=args.neural_report.resolve().parent
    _json(directory/'comparison.json',result)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig=Figure(figsize=(10,4),layout='constrained');FigureCanvasAgg(fig)
    axes=fig.subplots(1,2)
    for ax,key,title in zip(axes,['mean_dice','detection_rate'],['Mean Dice','Detection rate (IoU ≥ 0.5)']):
        for i,(method,label) in enumerate([('classical','Classical'),('neural','StarDist 3D')]):
            scores=result[method]
            ax.bar(np.arange(2)+i*.35,[scores[key],scores['complete_cells'][key]],width=.35,label=label)
        ax.set_xticks([.175,1.175],['All references','Complete references'])
        ax.set_ylim(0,1.08);ax.set_title(title);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Exploratory comparison on reserved images; precision unknown')
    fig.savefig(directory/'comparison.png',dpi=150);fig.savefig(directory/'comparison.svg')
    for method in ('classical','neural'):
        scores=result[method]
        print(method, {key:scores[key] for key in ('mean_dice','detection_rate','predicted_count')},
            'complete Dice',scores['complete_cells']['mean_dice'])


if __name__=='__main__':main()
