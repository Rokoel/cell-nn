"""Choose center confidence on validation only; sparse labels cannot estimate field precision."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from fish_viewer.neural.audit import read_references
from fish_viewer.pipeline.evaluation import evaluate
from fish_viewer.pipeline.runner import _json
from fish_viewer.workflow.volumes import surface_volume
from fish_viewer.pipeline.runner import read_prediction
from fish_viewer.signals.data import reference_memberships


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    args=p.parse_args();audit=json.loads(args.audit.read_text());report=json.loads(args.report.read_text())
    if report['status']!='complete':raise ValueError('Completed validation proposals are required.')
    cases={c['image']:c for c in audit['cases'] if c['audit']['split']=='validation'}
    if set(cases)!=set(r['image'] for r in report['files']):raise ValueError('Every validation scan, and no test scan, is required.')
    thresholds=[.25,.35,.45,.5,.55,.6,.65,.7]
    summaries={t:dict(threshold=t,complete_references=0,matched_cells=0,cells=0,duplicate_centers_in_marked_cells=0,
                       per_cell_iou=[],scans=[]) for t in thresholds}
    for number,row in enumerate(report['files'],1):
        print(f'[{number}/{len(cases)}] Calibrate center confidence: {Path(row["image"]).name}',flush=True)
        case=cases[row['image']];refs,names=read_references(Path(case['directory'])/'references.npz')
        complete=[(r,n) for r,n,cut in zip(refs,names,case['audit']['reference_cut']) if not cut]
        refs=[r for r,n in complete];names=[n for r,n in complete]
        with np.load(row['predictions']) as a:labels=a['labels'];metadata=json.loads(str(a['metadata']))
        scores={d['cell_id']:d['score'] for d in row['proposals'] if d['accepted']}
        for threshold in thresholds:
            ids=[cell for cell,score in scores.items() if score>=threshold]
            selected=np.where(np.isin(labels,ids),labels,0)
            metrics=evaluate(selected,refs,names,metadata['spacing_zyx'])
            s=summaries[threshold];s['complete_references']+=len(refs);s['matched_cells']+=metrics['matched_count'];s['cells']+=len(ids)
            s['per_cell_iou'].extend(r['iou'] for r in metrics['per_cell'])
            # Multiple accepted centers inside an annotated cell are observed
            # duplicates, even though unannotated cells remain unscorable.
            centers=[d['center_xyz'] for d in row['proposals'] if d['accepted'] and d['score']>=threshold]
            members=reference_memberships(centers,refs,metadata['origin_xyz'],metadata['spacing_zyx'])
            s['duplicate_centers_in_marked_cells']+=sum(max(0,sum(i in m for m in members)-1) for i in range(len(refs)))
            s['scans'].append(dict(image=row['image'],cells=len(ids),matched=metrics['matched_count'],references=len(refs)))
    rows=[]
    for s in summaries.values():
        s['annotated_cell_recall']=s['matched_cells']/s['complete_references']
        s['mean_manual_cell_iou']=float(np.mean(s.pop('per_cell_iou')))
        rows.append(s)
    # Prefer the highest confidence preserving >=95% of the maximum observed
    # matched-cell count. This is a validation operating point, not precision.
    most=max(s['matched_cells'] for s in rows)
    chosen=max((s for s in rows if s['matched_cells']>=.95*most),key=lambda s:s['threshold'])
    model=args.model/'proposal-model.json';manifest=json.loads(model.read_text())
    manifest['config']['threshold']=chosen['threshold']
    result=dict(status='complete',selected_threshold=chosen['threshold'],thresholds=rows,
        policy='All validation scans only. Highest threshold preserving >=95% of maximum complete annotated-cell '
        'IoU>=.5 matches. Unknown/unmarked cells never count as false positives. No test selection. '
        'Predicted boxes are followed by the frozen contour model; failed crops have no fallback.',
        parent_report=str(args.report.resolve()),checkpoint_sha256=report['checkpoint_sha256'],boundary_sha256=report['boundary_sha256'])
    manifest['validation_operating_point']=result;_json(model,manifest);_json(args.output,result)
    print(json.dumps(dict(selected=chosen,policy=result['policy']),indent=2))


if __name__=='__main__':main()
