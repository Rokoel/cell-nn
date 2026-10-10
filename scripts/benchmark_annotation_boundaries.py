"""Compare the actual boundary-guided pipeline against saved held-out defaults.

Only baseline cell detection is reused; learned guidance, signal inference,
cell filtering, mesh volumes and A-B distances are freshly executed.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import hashlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_all_heldout import evaluate_run, paired_stats
from fish_viewer.workflow.results import filter_result
from fish_viewer.workflow.distances import measure_result


def worker(arguments, folder, name):
    log=folder/(name+'.log')
    env=os.environ.copy();env['TF_CPP_MIN_LOG_LEVEL']='2';env['OMP_NUM_THREADS']='2'
    with log.open('w') as stream:
        completed=subprocess.run([str(ROOT/'.venv/bin/python'),'-m',*map(str,arguments)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if completed.returncode: raise RuntimeError('Worker failed: '+str(log))
    for line in reversed(log.read_text().splitlines()):
        try: event=json.loads(line)
        except ValueError:continue
        if event.get('kind')=='finished':
            report=event['report']
            if report['status']!='complete':raise RuntimeError('Incomplete stage: '+str(log))
            return report
    raise RuntimeError('No completion event: '+str(log))


def summarize(rows):
    result={}
    for split in ('validation','test','all'):
        selected=[r for r in rows if split=='all' or r['split']==split]
        output={}
        for mode in ('default','boundary_guided'):
            subset=[r for r in selected if r['mode']==mode]
            volume=[dict(p,group=r['group']) for r in subset for p in r['volume_pairs'] if not p['cut']]
            # Distance pairs on cut cells are retained consistently with baseline protocol.
            distance=[dict(p,group=r['group']) for r in subset for p in r['distance_pairs']]
            output[mode]=dict(scans=len(subset),matched_before_filter=sum(r['matched_before_filter'] for r in subset),
                matched_final=sum(r['matched_final'] for r in subset),cells_before_filter=sum(r['cells_before_filter'] for r in subset),
                final_cells=sum(r['final_cells'] for r in subset),
                volume=paired_stats(volume,'manual_um3','predicted_um3',True),
                distance=paired_stats(distance,'manual_um','predicted_um',False))
        result[split]=output
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,default=ROOT/'artifacts/all-heldout-20261009-213558/report.json')
    p.add_argument('--audit',type=Path,default=ROOT/'artifacts/refinement-fish/audit/audit.json')
    p.add_argument('--model',type=Path,default=ROOT/'models/boundary-annotations')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--split',choices=['all','validation','test'],default='all');p.add_argument('--limit',type=int)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    base=json.loads(args.baseline.read_text());collection=json.loads(args.audit.read_text())
    cases={c['image']:c for c in collection['cases']}
    selection=sorted([r for r in base['results'] if r['mode']=='default' and (args.split=='all' or r['split']==args.split)],key=lambda r:(r['split'],r['image']))
    if args.limit:selection=selection[:args.limit]
    manifest=json.loads((args.model/'boundary-model.json').read_text())
    identity=dict(checkpoint_sha256=manifest['checkpoint_sha256'],baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                  audit_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(),configuration=dict(max_shift_um=.3,strength=.5),
                  source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [ROOT/'fish_viewer/refinement/membrane.py',ROOT/'fish_viewer/boundary/inference.py',ROOT/'fish_viewer/boundary/model.py']})
    identity_path=out/f'identity-{args.shard_index}.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:raise ValueError('Cannot resume with changed inputs.')
    identity_path.write_text(json.dumps(identity,indent=2))
    results=[];started=time.monotonic()
    for index,original in enumerate(selection):
        if index%args.shards!=args.shard_index:continue
        case=cases[original['image']]
        if case['audit']['split'] not in ('validation','test'):raise ValueError('Held-out scans only.')
        folder=out/(f'{index+1:03d}-'+Path(original['image']).stem);folder.mkdir(exist_ok=True)
        target=folder/'metrics.json'
        if target.exists():row=json.loads(target.read_text())
        else:
            state=json.loads(Path(original['pipeline_report']).read_text())
            cell_report=Path(state['stages']['cells'])/'report.json'
            guided=worker(['fish_viewer.refinement.membrane','--cells-report',cell_report,'--output',folder/'work',
                           '--audit',args.audit,'--boundary-model',args.model,'--device','gpu'],folder,'boundary')
            signals=worker(['fish_viewer.signals','--mode','apply','--image',original['image'],
                           '--cell-predictions',guided['files'][0]['predictions'],'--model',ROOT/'models/signals-center-unet',
                           '--output',folder/'work','--device','gpu'],folder,'signals')
            filtered=filter_result(Path(signals['directory'])/'report.json',folder/'work')
            measured=measure_result(Path(filtered['directory'])/'report.json',folder/'work')
            if filtered['status']!='complete' or measured['status']!='complete':raise RuntimeError('Filter/measure failed.')
            newstate=dict(report_path=str(folder/'pipeline-result.json'),signal_report=str(Path(measured['directory'])/'report.json'))
            (folder/'pipeline-result.json').write_text(json.dumps(newstate,indent=2))
            row=evaluate_run(case,newstate,collection,Path(original['pipeline_report']).parents[3])
            row.update(mode='boundary_guided',guidance_summary=guided['summary'])
            target.write_text(json.dumps(row,indent=2))
        results.extend([original,row])
        report=dict(status='running',completed_scans=len(results)//2,total_scans=len(selection),elapsed_s=time.monotonic()-started,
                    identity=identity,results=results,summary=summarize(results))
        (out/f'report-{args.shard_index}.json').write_text(json.dumps(report,indent=2))
        print('COMPLETE',len(results)//2,row['scan'],row['guidance_summary'],flush=True)
    report['status']='complete';(out/f'report-{args.shard_index}.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
