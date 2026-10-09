"""Fixed validation sample: compare frozen defaults with conservative cell recovery.

Unmatched predictions are unscored because FISH annotations are incomplete.
This benchmark measures annotated recall and matched geometry, not precision.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from PySide6 import QtCore

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fish_viewer.analysis.controller import AnalysisController,default_proposal_model,default_refinement_model,default_models
from fish_viewer.models import LibraryEntry
from fish_viewer.reader import load_dataset
from fish_viewer.pipeline.segmentation import Prepared
from fish_viewer.pipeline.evaluation import rasterize_reference,evaluate
from fish_viewer.workflow.volumes import surface_volume

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--scans',type=int,default=4)
parser.add_argument('--seed',type=int,default=20261009)
parser.add_argument('--output',type=Path)
parser.add_argument('--resume',type=Path)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
output=args.resume or args.output or root/'artifacts'/('cell-recovery-validation-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
output.mkdir(parents=True,exist_ok=True)
audit=json.loads((root/'artifacts/refinement-fish/audit/audit.json').read_text())
if args.resume:
 selection=json.loads((output/'selection.json').read_text());selected=selection['cases']
else:
 groups={}
 for case in audit['cases']:
  if case['audit']['split']=='validation':groups.setdefault(case['audit']['group'],[]).append(case)
 rng=random.Random(args.seed)
 selected=[rng.choice(groups[g]) for g in rng.sample(sorted(groups),args.scans)]
 selection=dict(seed=args.seed,cases=selected,policy='One scan randomly sampled from each of distinct randomly sampled validation acquisition groups, before inference.',
                precision_note='Unmatched NN objects are unscored, not false positives. Cell precision requires reviewed detections.',status='selected')
 (output/'selection.json').write_text(json.dumps(selection,indent=2))
(root/'artifacts/current-cell-recovery-validation.txt').write_text(str(output)+'\n')
results=[]
queue=[(i,c,mode) for i,c in enumerate(selected) for mode in ('baseline','recovery')]
app=QtCore.QCoreApplication([]);controller=AnalysisController();active={}

def save(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def evaluate_run(case,state):
 run=Path(state['report_path']).parent
 annotation=json.loads((run/state['image']).with_suffix('.annotations.json').read_text())
 report=json.loads((run/state['signal_report']).read_text())
 report['manual_measurements']=annotation['manual_measurements']
 with np.load(report['unfiltered_cell_predictions']) as f:
  labels=f['labels'];metadata=json.loads(str(f['metadata']))
 dataset=load_dataset(LibraryEntry(Path(case['image']),Path(case['scene'])))
 prepared=Prepared(labels,tuple(metadata['spacing_zyx']),tuple(metadata['origin_xyz']),metadata['channel'],[],'',{},{})
 refs,names,skipped=rasterize_reference(dataset.scene.surfaces,prepared)
 metrics=evaluate(labels,refs,names,np.asarray(prepared.spacing))
 matched=[r for r in metrics['per_cell'] if r['iou']>=.5]
 kept=set(report['signal_filter']['kept_cell_ids'])
 with np.load(report['cell_predictions']) as f: final_labels=f['labels']
 final=evaluate(final_labels,refs,names,np.asarray(prepared.spacing))
 final_matches=[r for r in final['per_cell'] if r['iou']>=.5]
 by_name={s.name:s for s in dataset.scene.surfaces if s.time==0}
 manual=report['manual_measurements']
 volumes={c['cell_id']:c['volume_um3'] for c in report['cell_volumes']}
 volume_errors=[]
 for row in final_matches:
  original=surface_volume(by_name[row['name']]); predicted=volumes[row['predicted_label']]
  if original and predicted:volume_errors.append(100*abs(predicted-original)/original)
 distances=[]
 for row in final_matches:
  truth=[p for p in manual['points'] if p['cell_id']==row['name'] and p['channel'] is not None and p['distance_um'] is not None]
  predicted=[p for p in report['points'] if p['cell_id']==row['predicted_label']]
  if not truth or not predicted:continue
  costs=np.array([[np.linalg.norm(np.asarray(a['xyz'])-b['xyz']) if a['channel']==b['channel'] else 1e6 for b in predicted] for a in truth])
  aa,bb=linear_sum_assignment(costs)
  distances.extend(abs(truth[a]['distance_um']-predicted[b]['distance_um']) for a,b in zip(aa,bb) if costs[a,b]<=.6)
 return dict(scan=Path(case['image']).name,group=case['audit']['group'],split=case['audit']['split'],
             references=len(refs),cells_before_filter=report['signal_filter']['original_cell_count'],final_cells=len(kept),
             signals=len(report['points']),matched_before_filter=len(matched),matched_final=len(final_matches),
             unscored_before_filter=int(labels.max())-len(matched),unscored_final=len(kept)-len(final_matches),
             matched_volume_absolute_percent_errors=volume_errors,matched_distance_absolute_errors_um=distances,
             recovery=report.get('cell_recovery'),pipeline_report=state['report_path'])

def progress(e):
 print(f"{active.get('index',0)+1}/{len(queue)} {active.get('mode')} {e.get('stage')}/{e.get('stages')} {e.get('phase')} {e.get('current',0)}/{e.get('total',0)}",flush=True)

def completed(state):
 try:
  save(active['folder']/'pipeline-result.json',state)
  metrics=evaluate_run(active['case'],state);metrics['mode']=active['mode']
  save(active['folder']/'metrics.json',metrics);results.append(metrics)
  print('METRICS',json.dumps(metrics),flush=True)
  QtCore.QTimer.singleShot(0,start_next)
 except Exception:
  import traceback;traceback.print_exc();app.exit(1)

def start_next():
 if len(results)==len(queue):app.quit();return
 i,case,mode=queue[len(results)]
 folder=output/(f'{i+1:02d}-'+Path(case['image']).stem)/mode;folder.mkdir(parents=True,exist_ok=True)
 active.update(index=len(results),case=case,mode=mode,folder=folder)
 saved=folder/'pipeline-result.json'
 if saved.is_file():completed(json.loads(saved.read_text()));return
 controller.start(case['image'],folder/'runs',default_proposal_model(),default_models()[1],device='cpu',
                  refinement=default_refinement_model(),recovery={} if mode=='recovery' else None)

controller.progress.connect(progress);controller.completed.connect(completed)
controller.failed.connect(lambda message:(print(message,flush=True),app.exit(1)))
QtCore.QTimer.singleShot(0,start_next)
code=app.exec()
assert code==0 and len(results)==len(queue)
summary={}
for mode in ('baseline','recovery'):
 rows=[r for r in results if r['mode']==mode]
 values={k:sum(r[k] for r in rows) for k in ('references','cells_before_filter','final_cells','signals','matched_before_filter','matched_final','unscored_before_filter','unscored_final')}
 ve=[v for r in rows for v in r['matched_volume_absolute_percent_errors']]
 de=[v for r in rows for v in r['matched_distance_absolute_errors_um']]
 values.update(volume_pairs=len(ve),volume_mape_percent=float(np.mean(ve)) if ve else None,
               distance_pairs=len(de),distance_mae_um=float(np.mean(de)) if de else None)
 summary[mode]=values
preservation=[]
for case in selected:
 pair=[r for r in results if r['scan']==Path(case['image']).name and r['group']==case['audit']['group']]
 baseline=next(r for r in pair if r['mode']=='baseline')
 recovery=next(r for r in pair if r['mode']=='recovery')
 def measured(row):
  state=json.loads(Path(row['pipeline_report']).read_text())
  return json.loads((Path(row['pipeline_report']).parent/state['signal_report']).read_text())
 old,new=measured(baseline),measured(recovery)
 with np.load(old['unfiltered_cell_predictions']) as archive:old_labels=archive['labels']
 with np.load(new['unfiltered_cell_predictions']) as archive:new_labels=archive['labels']
 unchanged=old_labels.shape==new_labels.shape and np.array_equal(old_labels[old_labels>0],new_labels[old_labels>0])
 assert unchanged,'Recovery changed a baseline accepted contour.'
 def signature(point):return (point['cell_id'],point['channel'],tuple(point['xyz']),point['score'],point['distance_um'])
 new_signals={signature(p) for p in new['points']}
 assert all(signature(p) in new_signals for p in old['points']),'Recovery changed a baseline signal measurement.'
 preservation.append(dict(scan=baseline['scan'],baseline_contours_unchanged=unchanged,baseline_signals_and_distances_unchanged=True))
save(output/'report.json',dict(status='complete',selection=selection,results=results,summary=summary,
                            baseline_preservation=preservation,
                            limitation='Exploratory validation sample, not a precision estimate or full-dataset evaluation. No parameters were selected on test scans.'))
print(json.dumps(summary,indent=2),flush=True)
