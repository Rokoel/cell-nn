"""Merge disjoint held-out shards and report complete-cell and paired metrics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_annotation_boundaries import summarize
from scripts.benchmark_all_heldout import paired_stats


def aggregate(directory, audit, baseline):
    collection=json.loads(Path(audit).read_text());cases={c['image']:c for c in collection['cases'] if c['audit']['split'] in ('test','validation')}
    original=json.loads(Path(baseline).read_text());base={r['image']:r for r in original['results'] if r['mode']=='default'}
    rows=[]
    for path in sorted(Path(directory).glob('*/metrics.json')):
        guided=json.loads(path.read_text());image=guided['image'];case=cases[image]
        cuts=case['audit']['reference_cut']
        for row in [dict(base[image]),guided]:
            for phase in ('before_filter','final'):
                row['matched_complete_'+phase]=sum(not cuts[m['reference_id']] for m in row['matches_'+phase])
            for point in row['distance_pairs']:point['cut']=cuts[point['reference_id']]
            # Keep primary errors conditional on complete manual cells in both modes.
            row['distance_pairs']=[p for p in row['distance_pairs'] if not p['cut']]
            rows.append(row)
    if len(rows)!=len(cases)*2 or {r['image'] for r in rows}!=set(cases):
        raise ValueError(f'Incomplete held-out coverage: {len(rows)//2}/{len(cases)} scans.')
    summary=summarize(rows)
    for split,data in summary.items():
        selected=[r for r in rows if split=='all' or r['split']==split]
        for mode in ('default','boundary_guided'):
            subset=[r for r in selected if r['mode']==mode]
            data[mode].update(matched_complete_before_filter=sum(r['matched_complete_before_filter'] for r in subset),
                              matched_complete_final=sum(r['matched_complete_final'] for r in subset))
        common={}
        for field,key,a,b,pct in [('volume_pairs',lambda p:p['reference_id'],'manual_um3','predicted_um3',True),
                ('distance_pairs',lambda p:(p['reference_id'],p['channel'],tuple(p['manual_xyz'])),'manual_um','predicted_um',False)]:
            pairs={'default':[],'boundary_guided':[]}
            for image in {r['image'] for r in selected}:
                match={r['mode']:r for r in selected if r['image']==image}
                index={mode:{key(p):p for p in row[field] if not p.get('cut',False)} for mode,row in match.items()}
                shared=set(index['default'])&set(index['boundary_guided'])
                for mode in pairs:pairs[mode].extend(dict(index[mode][k],group=match[mode]['group']) for k in shared)
            common[field]={mode:paired_stats(values,a,b,pct) for mode,values in pairs.items()}
            changes=[dict(group=left['group'],
                          delta=(abs(right[b]-right[a])-abs(left[b]-left[a]))*(100/left[a] if pct else 1))
                     for left,right in zip(pairs['default'],pairs['boundary_guided'])]
            groups=sorted({p['group'] for p in changes})
            if len(groups)>1:
                sums=np.array([[sum(p['delta'] for p in changes if p['group']==g),
                                sum(p['group']==g for p in changes)] for g in groups])
                draws=np.random.default_rng(42).integers(len(groups),size=(2000,len(groups)))
                samples=sums[draws].sum(1)
                common[field]['paired_change']=dict(n=len(changes),
                    mean_error_change=float(np.mean([p['delta'] for p in changes])),
                    bootstrap_95ci=np.percentile(samples[:,0]/samples[:,1],[2.5,97.5]).tolist(),
                    units='percentage points' if pct else 'µm',negative_is_improvement=True)
        data['common_matches']=common
    identities=[json.loads(p.read_text()) for p in Path(directory).glob('identity-*.json')]
    if not identities or any(v!=identities[0] for v in identities):raise ValueError('Shard provenance differs.')
    for name,expected in identities[0]['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError('Inference code changed during benchmark.')
    checkpoint=ROOT/'models/boundary-annotations/boundary.weights.h5'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=identities[0]['checkpoint_sha256']:raise ValueError('Checkpoint changed during benchmark.')
    for model in original['identity']['models'].values():
        file=Path(model['directory'])/model['manifest']['checkpoint']
        if hashlib.sha256(file.read_bytes()).hexdigest()!=model['checkpoint_sha256']:
            raise ValueError('A baseline checkpoint changed during the comparison: '+str(file))
    report=dict(status='complete',scans=len(cases),summary=summary,results=rows,identity=identities[0],
        baseline_models=original['identity']['models'],
        guidance_accepted=sum(r['guidance_summary']['snapped'] for r in rows if r['mode']=='boundary_guided'),
        policy='All 144 held-out scans. Frozen baseline cell detections reused; learned contour guidance, signal-center NN, filtering and physical mesh measurements freshly executed. Complete manual cells only for primary errors, original manual A-B segments only, all outliers retained. No exported scan duplication.',
        limitations='Brush-band metrics do not imply exact surface accuracy. Measurement errors are conditional on matched cells/signals. Manual cell annotation is incomplete. Validation and test acquisitions have been previously inspected; this is not a new blind cohort.')
    (Path(directory)/'report.json').write_text(json.dumps(report,indent=2))
    return report


def publish(report, output):
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
    columns=['mode','scan','split','cells_before_filter','matched_complete_before_filter','final_cells','matched_complete_final','signals']
    with (output/'per-scan.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        writer.writerows({k:r[k] for k in columns} for r in report['results'])
    lines=['# Annotation-trained boundary guidance benchmark','',report['policy'],'']
    for split in ('test','validation','all'):
        data=report['summary'][split]
        lines += [f'## {split}', '', '| Method | Scans | Manual complete cells matched before signals | Volume MAPE | Distance MAE |', '|---|---:|---:|---:|---:|']
        for mode in ('default','boundary_guided'):
            d=data[mode];lines.append(f"| {mode} | {d['scans']} | {d['matched_complete_before_filter']} | {d['volume']['mape_percent']:.2f}% (n={d['volume']['n']}) | {d['distance']['mae']:.3f} µm (n={d['distance']['n']}) |")
        lines+=['', 'Same manual references matched by both methods:','']
        for mode in ('default','boundary_guided'):
            common=data['common_matches'];v=common['volume_pairs'][mode];d=common['distance_pairs'][mode]
            lines.append(f"- {mode}: volume MAPE {v['mape_percent']:.2f}% (n={v['n']}), distance MAE {d['mae']:.3f} µm (n={d['n']}).")
        lines.append('')
    lines += ['[Boundary training curves](training.png)', '',
              '[Group-held-out annotation examples](development-examples.png)', '',report['limitations']]
    (output/'README.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--audit',type=Path,default=ROOT/'artifacts/refinement-fish/audit/audit.json')
    p.add_argument('--baseline',type=Path,default=ROOT/'artifacts/all-heldout-20261009-213558/report.json')
    p.add_argument('--output',type=Path,default=ROOT/'reports/annotation-boundary')
    args=p.parse_args();publish(aggregate(args.directory,args.audit,args.baseline),args.output)
