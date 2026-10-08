"""Select a permissive candidate finder using validation acquisitions only."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
from fish_viewer.pipeline.runner import _json
from fish_viewer.signals.config import SignalConfig
from fish_viewer.signals.data import prepare_supervision,supervised_candidates
from fish_viewer.signals.evaluation import evaluate_points,summarize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-audit',type=Path,required=True)
    parser.add_argument('--cell-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    collection=json.loads(args.collection_audit.read_text())
    cell_run=json.loads(args.cell_report.read_text())
    if collection['status']!='complete' or cell_run['status']!='complete':
        raise ValueError('Completed collection and cell reports required.')
    cases=[c for c in collection['cases'] if c['audit']['split']=='validation']
    if not cases:raise ValueError('The collection has no validation cases.')
    sources={s['image']:s for s in collection['sources']}
    cells={r['image']:r for r in cell_run['files']}
    base=replace(SignalConfig(),channel_contrast_min=.025)
    multi=replace(base,candidate_scales_um=(.10,.18,.28),candidate_separation_um=.25,one_to_one_positives=True)
    variants={
        'legacy':base,
        'multiscale-025':replace(multi,candidate_contrast=.025),
        'multiscale-015':replace(multi,candidate_contrast=.015),
        'multiscale-010':replace(multi,candidate_contrast=.010),
        'multiscale-010-close':replace(multi,candidate_contrast=.010,candidate_separation_um=.20),
    }
    report=dict(version=1,status='running',split='validation',
                collection_audit=str(args.collection_audit.resolve()),cell_report=str(args.cell_report.resolve()),
                configs={key:asdict(value) for key,value in variants.items()},scans=[],
                policy='Validation only. Maximize candidate reference coverage; within one percentage point of best recall choose the smallest supervised candidate pool. Channel assignment is frozen at original evidence settings. Candidate precision is not final CNN precision.')
    started=time.monotonic();scores={key:[] for key in variants}
    for index,case in enumerate(cases):
        prepared=prepare_supervision(case,sources,cells[case['image']],base)
        metrics={}
        for key,config in variants.items():
            rows,all_count=supervised_candidates(prepared,config.validate())
            metric=evaluate_points(rows,prepared['truth'],config.match_radius_um)
            scores[key].append(metric)
            metrics[key]=dict(summarize([metric]),all_candidates=all_count)
        report['scans'].append(dict(image=case['image'],group=case['audit']['group'],metrics=metrics))
        report['progress']=dict(phase='Validating signal candidate finders',current=index+1,total=len(cases),file=case['image'])
        report['elapsed_s']=time.monotonic()-started
        _json(args.output/'report.json',report)
        print(json.dumps(report['progress']),flush=True)
        del prepared
    report['summary']={key:summarize(value) for key,value in scores.items()}
    # Original exact duplicates are suppressed by each finder, so extra peaks are
    # capacity for CNN rejection, not an assumed collection of real signals.
    best=max(value['recall'] or 0 for value in report['summary'].values())
    qualified=[key for key,value in report['summary'].items() if (value['recall'] or 0)>=best-.01]
    selected=min(qualified,key=lambda key:(report['summary'][key]['predictions'],-report['summary'][key]['recall']))
    report['selected']=selected;report['status']='complete'
    _json(args.output/'selected-config.json',asdict(variants[selected]))
    _json(args.output/'report.json',report)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig=Figure(figsize=(11,4),layout='constrained');FigureCanvasAgg(fig)
    axes=fig.subplots(1,2);names=list(variants)
    axes[0].bar(names,[100*report['summary'][key]['recall'] for key in names])
    axes[0].set(ylabel='Eligible references reached (%)',title='Validation candidate ceiling',ylim=(0,100))
    axes[1].bar(names,[report['summary'][key]['predictions'] for key in names])
    axes[1].set(ylabel='Candidates in annotated intersections',title='CNN scoring workload')
    for ax in axes:ax.tick_params(axis='x',labelrotation=30);ax.grid(axis='y',alpha=.2)
    fig.savefig(args.output/'candidate-comparison.png',dpi=150)
    print(json.dumps(dict(status='complete',selected=selected,summary=report['summary'])),flush=True)


if __name__=='__main__':main()
