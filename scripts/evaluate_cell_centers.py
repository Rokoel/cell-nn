"""Evaluate and calibrate the point/box stage on sparse manual validation cells."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from fish_viewer.neural.audit import read_references
from fish_viewer.refinement.data import seed_center
from fish_viewer.signals.data import reference_memberships
from fish_viewer.pipeline.runner import _json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--model',type=Path)
    p.add_argument('--split',choices=['validation','test'],default='validation')
    args=p.parse_args();audit=json.loads(args.audit.read_text());report=json.loads(args.report.read_text())
    if report['status']!='complete':raise ValueError('Completed center predictions required.')
    cases={c['image']:c for c in audit['cases'] if c['audit']['split']==args.split}
    if set(cases)!=set(r['image'] for r in report['files']):raise ValueError('All and only the specified split must be present.')
    if args.model and args.split!='validation':raise ValueError('Model calibration cannot use test scans.')
    thresholds=[.25,.35,.45,.5,.55,.6,.65,.7,.75,.8]
    records=[];total=0
    for row in report['files']:
        case=cases[row['image']];refs,names=read_references(Path(case['directory'])/'references.npz')
        sp=np.array(case['audit']['spacing_zyx']);origin=np.array(case['audit']['origin_xyz'])
        points=[d['center_xyz'] for d in row['proposals']]
        memberships=reference_memberships(points,refs,origin,sp)
        for ref,(box,mask) in enumerate(refs):
            if case['audit']['reference_cut'][ref]:continue
            total+=1;low=np.array([s.start for s in box]);high=np.array([s.stop for s in box])
            seed=(seed_center(mask,sp)+low)[::-1]*sp[::-1]+origin
            expected_low=(low-.5)[::-1]*sp[::-1]+origin;expected_high=(high-.5)[::-1]*sp[::-1]+origin
            for proposal,members in zip(row['proposals'],memberships):
                if members!=[ref]:continue
                distance=float(np.linalg.norm(np.array(proposal['center_xyz'])-seed))
                margin=report['config']['box_margin_um']
                intersection=np.maximum(0,np.minimum(expected_high,np.array(proposal['box_high_xyz'])+margin)-
                    np.maximum(expected_low,np.array(proposal['box_low_xyz'])-margin))
                coverage=float(np.prod(intersection)/np.prod(expected_high-expected_low))
                records.append(dict(image=row['image'],group=case['audit']['group'],manual_cell=names[ref],
                    reference_id=ref+1,proposal_id=proposal['proposal_id'],score=proposal['score'],
                    center_error_um=distance,padded_aabb_coverage=coverage))
    summaries=[]
    for threshold in thresholds:
        eligible=[r for r in records if r['score']>=threshold and r['center_error_um']<=2.]
        unique={}
        for r in sorted(eligible,key=lambda r:r['center_error_um']):unique.setdefault((r['image'],r['reference_id']),r)
        values=list(unique.values());errors=[r['center_error_um'] for r in values]
        summaries.append(dict(threshold=threshold,manual_complete_cells=total,matched_center_cells=len(values),
            annotated_cell_center_recall=len(values)/total,center_mae_um=float(np.mean(errors)) if errors else None,
            center_median_error_um=float(np.median(errors)) if errors else None,
            padded_box_coverage_ge_095=sum(r['padded_aabb_coverage']>=.95 for r in values),
            duplicate_centers_in_matched_cells=len(eligible)-len(values),
            proposed_centers=sum(d['score']>=threshold for row in report['files'] for d in row['proposals'])))
    most=max(s['matched_center_cells'] for s in summaries)
    selected=max((s for s in summaries if s['matched_center_cells']>=.95*most),key=lambda s:s['threshold'])
    result=dict(status='complete',split=args.split,scans=len(cases),selected_threshold=selected['threshold'],
                summaries=summaries,records=records,checkpoint_sha256=report['checkpoint_sha256'],
                policy='Center must lie inside a complete manual cell and within 2 µm of its EDT interior seed. '
                'One-to-one per manual cell. Padded AABB coverage compares physical box extents. '
                'Unknown/unmarked cells are unscorable, not false positives. Validation selects highest '
                'confidence retaining >=95% of maximum matched marked centers. No test-based selection.')
    args.output.mkdir(parents=True,exist_ok=True);_json(args.output/'report.json',result)
    with (args.output/'center-matches.csv').open('w',newline='',encoding='utf-8') as stream:
        w=csv.DictWriter(stream,fieldnames=list(records[0]) if records else ['No observations']);w.writeheader();w.writerows(records)
    if args.model:
        path=args.model/'proposal-model.json';manifest=json.loads(path.read_text())
        manifest['config']['threshold']=selected['threshold'];manifest['center_calibration']={k:v for k,v in result.items() if k!='records'}
        _json(path,manifest)
    print(json.dumps(dict(selected=selected,policy=result['policy']),indent=2))


if __name__=='__main__':main()
