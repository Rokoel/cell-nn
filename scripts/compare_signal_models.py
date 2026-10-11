"""Compare old/new signal finders and CNNs on identical frozen cell interiors."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from fish_viewer.worker_runtime import restart_with_cuda
from fish_viewer.neural.engine import runtime
from fish_viewer.pipeline.runner import _json
from fish_viewer.signals.config import SignalConfig
from fish_viewer.signals.data import prepare_supervision,supervised_candidates,extract_patches
from fish_viewer.signals.engine import network,predict_patches
from fish_viewer.signals.evaluation import suppress,evaluate_points,summarize,detection_separation


def main():
    restart_with_cuda()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--old-model',type=Path,required=True)
    parser.add_argument('--new-model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split',choices=['validation','test'],required=True)
    parser.add_argument('--device',choices=['auto','cpu','gpu'],default='auto')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    audit=json.loads(args.audit.read_text());collection=json.loads(Path(audit['collection_audit']).read_text())
    cases=[c for c in audit['cases'] if c['split']==args.split]
    collection_cases={c['image']:c for c in collection['cases']};sources={s['image']:s for s in collection['sources']}
    old=json.loads((args.old_model/'signal-model.json').read_text())
    new=json.loads((args.new_model/'signal-model.json').read_text())
    old_config=SignalConfig(**old['config']);new_config=SignalConfig(**new['config'])
    if new_config.match_radius_um!=old_config.match_radius_um:raise ValueError('Matching radii differ.')
    tf,_,_,info=runtime(args.device,new_config.seed)
    old_net=network(tf,old_config);old_net.load_weights(args.old_model/old['checkpoint'])
    new_net=network(tf,new_config);new_net.load_weights(args.new_model/new['checkpoint'])
    methods=('old_finder_old_cnn','new_finder_old_cnn','new_finder_new_cnn')
    report=dict(version=1,status='running',split=args.split,runtime=info,audit=str(args.audit.resolve()),
                old_model=str(args.old_model.resolve()),new_model=str(args.new_model.resolve()),scans=[],
                old_sha256=hashlib.sha256((args.old_model/old['checkpoint']).read_bytes()).hexdigest(),
                new_sha256=hashlib.sha256((args.new_model/new['checkpoint']).read_bytes()).hexdigest(),
                test_history_note='This project has evaluated the historical test fold before. The refresh freezes its candidate recipe and CNN on validation only, but these reused test results are exploratory rather than an untouched prospective assessment.',
                policy='Same scans, frozen latest cell masks, inferred channels and fully marked manual-cell intersections for all methods. One-to-one matching within the same cell/channel and 0.6 um. Old model retains its original threshold; new model threshold is validation-selected. No test tuning. Unknown cell regions unscored.')
    metrics={key:[] for key in methods};channels={key:{} for key in methods};ceilings={'old':[],'new':[]}
    started=time.monotonic()
    for index,case in enumerate(cases):
        prepared=prepare_supervision(collection_cases[case['image']],sources,dict(predictions=case['cell_predictions']),old_config)
        expected={(p['cell_id'],p['channel'],tuple(p['xyz'])) for p in case['truth']}
        if {(p['cell_id'],p['channel'],tuple(p['xyz'])) for p in prepared['truth']}!=expected:
            raise ValueError('Manual reference eligibility changed between finder configurations.')
        old_rows,_=supervised_candidates(prepared,old_config)
        old_scores=[]
        for start in range(0,len(old_rows),64):
            patches=extract_patches(prepared['images'],prepared['membrane'],prepared['native'],old_rows[start:start+64],prepared['origin'],prepared['spacing'],old_config)
            old_scores.extend(predict_patches(old_net,patches))
        patches=np.load(Path(case['directory'])/'patches.npy',mmap_mode='r')
        new_rows=case['candidates']
        shadow_scores=predict_patches(old_net,patches)
        new_scores=predict_patches(new_net,patches)
        points={
            'old_finder_old_cnn':suppress(old_rows,old_scores,old['threshold'],detection_separation(old_config)),
            'new_finder_old_cnn':suppress(new_rows,shadow_scores,old['threshold'],detection_separation(new_config)),
            'new_finder_new_cnn':suppress(new_rows,new_scores,new['threshold'],detection_separation(new_config)),
        }
        scan=dict(image=case['image'],group=case['group'],references=len(case['truth']),metrics={})
        for key,selected in points.items():
            metric=evaluate_points(selected,case['truth'],new_config.match_radius_um)
            metrics[key].append(metric);scan['metrics'][key]=metric
            for channel in {p['channel'] for p in selected+case['truth']}:
                channels[key].setdefault(str(channel),[]).append(evaluate_points(
                    [p for p in selected if p['channel']==channel],[p for p in case['truth'] if p['channel']==channel],new_config.match_radius_um))
        ceilings['old'].append(evaluate_points(old_rows,case['truth'],new_config.match_radius_um))
        ceilings['new'].append(evaluate_points(new_rows,case['truth'],new_config.match_radius_um))
        report['scans'].append(scan);report['elapsed_s']=time.monotonic()-started
        report['progress']=dict(phase='Comparing signal pipelines',current=index+1,total=len(cases),file=case['image'])
        _json(args.output/'report.json',report);print(json.dumps(report['progress']),flush=True)
        del prepared,patches
    report['summary']={key:summarize(rows) for key,rows in metrics.items()}
    report['by_channel_index']={key:{channel:summarize(rows) for channel,rows in values.items()} for key,values in channels.items()}
    report['channel_note']='Indices are zero-based and file-local; the same index need not represent the same fluorophore in every scan.'
    report['candidate_ceilings']={key:summarize(rows) for key,rows in ceilings.items()}
    groups=sorted({c['group'] for c in cases});rng=np.random.default_rng(42)
    samples={key:{metric:[] for metric in ('precision','recall')} for key in methods}
    improvements=[]
    for _ in range(2000):
        indices=[i for g in rng.choice(groups,len(groups),replace=True) for i,c in enumerate(cases) if c['group']==g]
        values={key:summarize([metrics[key][i] for i in indices]) for key in methods}
        for key in methods:
            for metric in ('precision','recall'):
                if values[key][metric] is not None:samples[key][metric].append(values[key][metric])
        a=values['old_finder_old_cnn']['recall'];b=values['new_finder_new_cnn']['recall']
        if a is not None and b is not None:improvements.append(b-a)
    report['acquisition_bootstrap_95ci']={key:{metric:np.quantile(values,[.025,.975]).tolist() if values else None
        for metric,values in sample.items()} for key,sample in samples.items()}
    report['paired_recall_improvement_95ci']=np.quantile(improvements,[.025,.975]).tolist()
    report['status']='complete';_json(args.output/'report.json',report)
    with (args.output/'scans.csv').open('w',newline='') as stream:
        fields=['image','group','method','tp','fp','fn','precision','recall','localization_rmse_um']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for case in report['scans']:
            for key,metric in case['metrics'].items():
                writer.writerow(dict(image=case['image'],group=case['group'],method=key,
                                     **{k:summarize([metric])[k] for k in fields[3:]}))
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig=Figure(figsize=(12,4),layout='constrained');FigureCanvasAgg(fig)
    axes=fig.subplots(1,3);labels=['Old finder\nold CNN','New finder\nold CNN','New finder\nnew CNN']
    for ax,metric,title in zip(axes,('recall','precision','localization_rmse_um'),('Signal recall (%)','Precision in scored regions (%)','Matched center RMSE (µm)')):
        values=[report['summary'][key][metric] for key in methods]
        ax.bar(labels,[v*100 if metric!='localization_rmse_um' else v for v in values])
        ax.set_title(title);ax.grid(axis='y',alpha=.2)
        if metric!='localization_rmse_um':ax.set_ylim(0,100)
    fig.savefig(args.output/'comparison.png',dpi=150)
    print(json.dumps(dict(status='complete',summary=report['summary'],candidate_ceilings=report['candidate_ceilings'])),flush=True)


if __name__=='__main__':main()
