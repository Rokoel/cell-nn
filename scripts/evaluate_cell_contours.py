"""Physical contour agreement on complete manual cells; unmatched fields remain unscorable."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from openpyxl import Workbook
from benchmark_measurements import manual_meshes,ref_key,paired_stats
from compare_refined_boundaries import surface_samples,boundary_error
from fish_viewer.neural.audit import read_references
from fish_viewer.pipeline.runner import read_prediction,_json
from fish_viewer.pipeline.segmentation import Prepared
from fish_viewer.workflow.distances import SurfaceLocator
from fish_viewer.workflow.volumes import surface_volume


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--split',choices=['validation','test'],required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    audit=json.loads(args.audit.read_text());report=json.loads(args.report.read_text())
    if report['status']!='complete':raise ValueError('Completed contour inference required.')
    cases={c['image']:c for c in audit['cases'] if c['audit']['split']==args.split}
    if set(cases)!=set(r['image'] for r in report['files']):raise ValueError('All and only the chosen split required.')
    sources={s['image']:s for s in audit['sources']};cells=[];scores=[];total=0
    for number,row in enumerate(report['files'],1):
        print(f'[{number}/{len(cases)}] Evaluate {args.split} contours: {Path(row["image"]).name}',flush=True)
        case=cases[row['image']];refs,names=read_references(Path(case['directory'])/'references.npz')
        surfaces,info=read_prediction(row['predictions']);predicted={int(s.name.rsplit(' ',1)[-1]):s for s in surfaces}
        prepared=Prepared(np.empty(case['audit']['shape_zyx'],np.uint8),tuple(info['spacing_zyx']),tuple(info['origin_xyz']),info['channel'],[],info['fingerprint'],{},{})
        scenes=sorted({s for alias in case['audit']['aliases'] for s in sources[alias]['scenes']})
        manual,_=manual_meshes(scenes,prepared)
        for ref,match in enumerate(row['metrics']['per_cell']):
            if case['audit']['reference_cut'][ref]:continue
            total+=1;scores.append(match['iou'])
            cell=match['predicted_label']
            if cell is None or match['iou']<.5:continue
            truth=manual.get(ref_key(refs[ref]))
            if truth is None or surface_volume(truth) is None:raise ValueError('Complete closed manual reference missing.')
            mean,hd95=boundary_error(truth,predicted[cell],SurfaceLocator(truth),SurfaceLocator(predicted[cell]),surface_samples(truth))
            cells.append(dict(image=str(Path(case['image']).relative_to(ROOT)),group=case['audit']['group'],
                manual_cell=names[ref],predicted_cell=cell,iou=match['iou'],dice=match['dice'],
                manual_volume_um3=surface_volume(truth),predicted_volume_um3=surface_volume(predicted[cell]),
                surface_mean_error_um=mean,surface_hd95_um=hd95))
    summary=dict(scans=len(cases),complete_manual_cells=total,matched_complete_cells=len(cells),
        annotated_cell_recall_iou_05=len(cells)/total,predicted_cells=sum(r['cells'] for r in report['files']),
        mean_manual_cell_iou=float(np.mean(scores)),volume=paired_stats(cells,'manual_volume_um3','predicted_volume_um3'),
        mean_symmetric_surface_error_um=float(np.mean([r['surface_mean_error_um'] for r in cells])),
        mean_surface_hd95_um=float(np.mean([r['surface_hd95_um'] for r in cells])))
    result=dict(status='complete',split=args.split,summary=summary,cells=cells,
        checkpoint_sha256=report['checkpoint_sha256'],boundary_sha256=report['boundary_sha256'],
        policy='Complete closed manual-cell matches at whole-cell IoU>=.5, before signal retention. All outliers included. '
        'Volumes use closed physical triangle meshes. Symmetric surface distance uses 256 area-uniform points per mesh. '
        'Sparse labels cannot establish field precision. No threshold selection in this evaluation.')
    args.output.mkdir(parents=True,exist_ok=True);_json(args.output/'report.json',result)
    fields=list(cells[0]) if cells else ['No observations']
    with (args.output/'cells.csv').open('w',newline='',encoding='utf-8') as stream:
        w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(cells)
    book=Workbook();sheet=book.active;sheet.title='Cells';sheet.append(fields)
    for r in cells:sheet.append([r[k] for k in fields])
    sheet=book.create_sheet('Summary');sheet.append(['Metric','Value'])
    for k,v in summary.items():sheet.append([k,json.dumps(v)])
    sheet.append(['Policy',result['policy']]);book.save(args.output/'contour-evaluation.xlsx');book.close()
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
