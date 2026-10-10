"""Frozen 2D contours + signal-center U-Net measurement test on all held-out scans."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmark_all_heldout import evaluate_run,paired_stats
from fish_viewer.pipeline.runner import _json
from fish_viewer.analysis.shape import apply as shape_apply
from fish_viewer.workflow.results import filter_result
from fish_viewer.workflow.distances import measure_result


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--roundness',type=float,default=.72);args=p.parse_args()
 args.output.mkdir(parents=True,exist_ok=False)
 collection=json.loads((ROOT/'artifacts/preprocessed-stardist-stain-aware-20261010/audit/audit.json').read_text())
 cells=json.loads((ROOT/'artifacts/stardist2d-finetune-20261010/finetuned/report.json').read_text())
 cases=[c for c in collection['cases'] if c['audit']['split']=='test'];rows={r['image']:r for r in cells['files']}
 if cells['status']!='complete' or set(rows)!={c['image'] for c in cases}:raise ValueError('All held-out predictions required')
 report=dict(status='running',scans=[],minimum_roundness=args.roundness,errors=[],
  policy='Fixed fine-tuned StarDist2D contours, app roundness gate and unchanged signal-center U-Net. All 71 held-out scans; no test tuning. Any-signal cell retention. Complete manual cells matched at IoU>=0.5; same-cell/channel signal matches within 0.6 µm; distances compared with actual manual A-B lines. All outliers retained. Historically reused test fold is exploratory; sparse annotations cannot establish field precision.')
 for i,case in enumerate(cases,1):
  folder=args.output/f'{i:03d}';folder.mkdir();row=rows[case['image']]
  source=folder/'cells-report.json';_json(source,dict(status='complete',files=[row]))
  shaped=shape_apply(source,folder/'work',args.roundness)
  if shaped['status']!='complete':raise RuntimeError(shaped)
  log=folder/'signals.log'
  with log.open('w') as f:
   subprocess.run([sys.executable,'-m','fish_viewer.signals','--mode','apply','--image',case['image'],
     '--cell-predictions',shaped['files'][0]['predictions'],'--model',str(ROOT/'models/signals-center-unet'),
     '--output',str(folder/'work'),'--device','gpu'],stdout=f,stderr=subprocess.STDOUT,check=True)
  events=[]
  for line in log.read_text().splitlines():
   try:events.append(json.loads(line))
   except ValueError:pass
  signal=next(e['report'] for e in reversed(events) if e.get('kind')=='finished')
  if signal['status']!='complete':raise RuntimeError(signal)
  filtered=filter_result(Path(signal['directory'])/'report.json',folder/'work')
  measured=measure_result(Path(filtered['directory'])/'report.json',folder/'work')
  state=dict(report_path=str(folder/'pipeline.json'),signal_report=str(Path(measured['directory'])/'report.json'))
  result=evaluate_run(case,state,collection,folder)
  # Primary comparisons exclude manual objects touching scan boundaries.
  result['distance_pairs']=[v for v in result['distance_pairs'] if not case['audit']['reference_cut'][v['reference_id']]]
  result['volume_pairs']=[v for v in result['volume_pairs'] if not v['cut']]
  report['scans'].append(result);_json(args.output/'report.json',report)
  if i%50==0 or i==len(cases):print(f'Tested {i}/{len(cases)} scans',flush=True)
 distances=[dict(v,group=r['group'],image=r['image']) for r in report['scans'] for v in r['distance_pairs']]
 volumes=[dict(v,group=r['group'],image=r['image']) for r in report['scans'] for v in r['volume_pairs']]
 report.update(status='complete',summary=dict(scans=len(cases),signals=sum(r['signals'] for r in report['scans']),
  retained_cells=sum(r['final_cells'] for r in report['scans']),distance=paired_stats(distances,'manual_um','predicted_um',True),
  volume=paired_stats(volumes,'manual_um3','predicted_um3',True)),distances=distances,volumes=volumes)
 _json(args.output/'report.json',report)
 import csv
 for name,values in [('distances',distances),('volumes',volumes)]:
  with (args.output/f'{name}.csv').open('w',newline='') as f:
   fields=list(values[0]) if values else ['No observations'];w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(values)
 print(json.dumps(report['summary'],indent=2),flush=True)

if __name__=='__main__':main()
