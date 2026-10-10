"""Run frozen default, hybrid and consensus pipelines on EVERY held-out acquisition.

Inference uses normal app workers. Only image/scene export is omitted. Consensus
reuses the two freshly completed cell runs for that scan, without predicting them
again. Manual annotations enter evaluation only, with aliases deduplicated.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from PySide6 import QtCore

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fish_viewer.analysis.controller import AnalysisController, default_proposal_model, default_refinement_model, default_models, sensitive_stardist_model
from fish_viewer.models import LibraryEntry
from fish_viewer.reader import load_dataset
from fish_viewer.scenes import read_scene
from fish_viewer.neural.audit import read_references
from fish_viewer.pipeline.evaluation import rasterize_reference,evaluate
from fish_viewer.pipeline.segmentation import Prepared
from fish_viewer.signals.config import SignalConfig
from fish_viewer.signals.data import normalized,infer_channels,marker_points,reference_memberships
from fish_viewer.signals.engine import image_fingerprint
from fish_viewer.workflow.volumes import surface_volume

MODES=('default','hybrid','consensus')

def save(path,value):
 path=Path(path);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def reference_key(ref):
 box,mask=ref
 return hashlib.sha256(str([(s.start,s.stop) for s in box]).encode()+mask.tobytes()).hexdigest()

def frozen_models():
 models={}
 for kind,folder,file in [('default_detector',default_proposal_model(),'proposal-model.json'),
                          ('hybrid_detector',sensitive_stardist_model(),'fish-model.json'),
                          ('contours',default_refinement_model(),'refinement-model.json'),
                          ('signals',default_models()[1],'signal-model.json')]:
  folder=Path(folder);manifest=json.loads((folder/file).read_text())
  models[kind]=dict(directory=str(folder.resolve()),manifest=manifest,manifest_sha256=sha(folder/file),checkpoint_sha256=sha(folder/manifest['checkpoint']))
 return models

def collect_truth(case,metadata,collection,folder):
 cache=folder/f"manual-channel-{metadata['channel']}.json"
 refs,names=read_references(Path(case['directory'])/'references.npz')
 a=case['audit']
 if not (np.allclose(metadata['spacing_zyx'],a['spacing_zyx']) and np.allclose(metadata['origin_xyz'],a['origin_xyz'])):
  raise ValueError('Prediction grid differs from original calibrated reference cache.')
 if metadata['fingerprint']!=a['fingerprint']:raise ValueError('Prediction fingerprint differs from held-out audit.')
 if cache.is_file():return refs,json.loads(cache.read_text())
 sources={s['image']:s for s in collection['sources']}
 scenes=sorted({p for alias in a['aliases'] for p in sources[alias]['scenes']})
 prepared=Prepared(np.empty(tuple(a['shape_zyx']),np.uint8),tuple(a['spacing_zyx']),tuple(a['origin_xyz']),a['channel'],[],'',{}, {})
 mesh_by_key={};ab={}
 for scene_path in scenes:
  scene=read_scene(Path(scene_path))
  for surface in scene.surfaces:
   if surface.time!=0:continue
   candidates,_,_=rasterize_reference([surface],prepared)
   if candidates:mesh_by_key.setdefault(reference_key(candidates[0]),surface)
  for group in scene.measurements:
   for left,right in group.edges:
    if group.times[left]!=0 or group.times[right]!=0:continue
    if {group.labels[left].strip().upper(),group.labels[right].strip().upper()}!={'A','B'}:continue
    i,j=(left,right) if group.labels[left].strip().upper()=='A' else (right,left)
    key=tuple(np.round(group.points[i],4))
    ab.setdefault(key,[]).append(float(np.linalg.norm(group.points[i]-group.points[j])))
 cells=[]
 for index,(ref,name) in enumerate(zip(refs,names)):
  key=reference_key(ref);surface=mesh_by_key.get(key)
  if surface is None:raise ValueError('No original manual mesh found for reference '+name)
  cells.append(dict(reference_id=index,name=name,reference_key=key,volume_um3=surface_volume(surface),cut=a['reference_cut'][index]))
 dataset=load_dataset(LibraryEntry(Path(case['image']),Path(case['scene']) if case['scene'] else None))
 if image_fingerprint(dataset)!=a['fingerprint']:raise ValueError('Original source pixels changed.')
 points=marker_points(scenes)
 memberships=reference_memberships([p['xyz'] for p in points],refs,a['origin_xyz'],a['spacing_zyx'])
 points=infer_channels([normalized(v) for v in dataset.volumes],points,metadata['channel'],dataset.center_origin,dataset.spacing[::-1],SignalConfig())
 counts=Counter()
 for p,membership in zip(points,memberships):
  values=ab.get(tuple(np.round(p['xyz'],4)),[])
  p.update(reference_ids=membership,manual_distance_um=float(np.mean(values)) if values and np.ptp(values)<=.05 else None)
  if p['channel'] is None:counts['ambiguous_channel']+=1
  if len(membership)!=1:counts['ambiguous_or_missing_cell']+=1
  if not values:counts['no_manual_AB']+=1
  elif np.ptp(values)>.05:counts['conflicting_manual_AB']+=1
 truth=dict(cells=cells,points=points,source_scenes=scenes,manual_A_points=len(points),exclusions=dict(counts),
            distance_reference='Actual manual A-B segments only; no fabricated reference distance for A-only markers.')
 save(cache,truth);return refs,truth

def evaluate_run(case,state,collection,folder):
 run=Path(state['report_path']).parent;report=json.loads((run/state['signal_report']).read_text())
 with np.load(report['unfiltered_cell_predictions'],allow_pickle=False) as f:labels=f['labels'];metadata=json.loads(str(f['metadata']))
 refs,truth=collect_truth(case,metadata,collection,folder)
 names=[str(i) for i in range(len(refs))]
 metrics=evaluate(labels,refs,names,np.array(metadata['spacing_zyx']))
 before=[r for r in metrics['per_cell'] if r['iou']>=.5]
 with np.load(report['cell_predictions'],allow_pickle=False) as f:final_labels=f['labels']
 final=evaluate(final_labels,refs,names,np.array(metadata['spacing_zyx']))
 matched=[r for r in final['per_cell'] if r['iou']>=.5]
 volumes={r['cell_id']:r['volume_um3'] for r in report['cell_volumes']}
 volume_pairs=[];distance_pairs=[]
 for row in matched:
  ref=int(row['name']);cell=truth['cells'][ref];volume=volumes[row['predicted_label']]
  if cell['volume_um3'] and volume:
   volume_pairs.append(dict(reference_id=ref,cell=cell['name'],predicted_cell_id=row['predicted_label'],iou=row['iou'],
      manual_um3=cell['volume_um3'],predicted_um3=volume,cut=cell['cut']))
  manual=[p for p in truth['points'] if p['reference_ids']==[ref] and p['channel'] is not None and p['manual_distance_um'] is not None]
  predicted=[p for p in report['points'] if p['cell_id']==row['predicted_label']]
  if not manual or not predicted:continue
  costs=np.array([[np.linalg.norm(np.asarray(p['xyz'])-q['xyz']) if p['channel']==q['channel'] else 1e6 for q in predicted] for p in manual])
  valid=costs<=.6;weights=valid*(min(costs.shape)+1+np.maximum(0,1-costs/.6))
  aa,bb=linear_sum_assignment(-weights)
  for i,j in zip(aa,bb):
   if not valid[i,j]:continue
   p,q=manual[i],predicted[j]
   distance_pairs.append(dict(reference_id=ref,cell=cell['name'],predicted_cell_id=row['predicted_label'],channel=p['channel'],
       manual_xyz=p['xyz'],predicted_xyz=q['xyz'],center_error_um=float(costs[i,j]),manual_um=p['manual_distance_um'],predicted_um=q['distance_um']))
 count=report['signal_filter']['original_cell_count'];kept=report['signal_filter']['kept_cell_count']
 return dict(image=case['image'],scan=Path(case['image']).name,group=case['audit']['group'],split=case['audit']['split'],
   references=len(refs),complete_references=sum(not c['cut'] for c in truth['cells']),cells_before_filter=count,final_cells=kept,
   signals=len(report['points']),matched_before_filter=len(before),matched_final=len(matched),
   unscored_before_filter=count-len(before),unscored_final=kept-len(matched),
   matches_before_filter=[dict(reference_id=int(r['name']),cell_id=r['predicted_label'],iou=r['iou']) for r in before],
   matches_final=[dict(reference_id=int(r['name']),cell_id=r['predicted_label'],iou=r['iou']) for r in matched],
   volume_pairs=volume_pairs,distance_pairs=distance_pairs,manual_A_points=truth['manual_A_points'],manual_exclusions=truth['exclusions'],pipeline_report=state['report_path'])

def paired_stats(rows,expected,predicted,percentage=False):
 if not rows:return dict(n=0)
 truth=np.array([p[expected] for p in rows]);found=np.array([p[predicted] for p in rows]);error=found-truth;absolute=np.abs(error)
 result=dict(n=len(rows),bias=float(error.mean()),mae=float(absolute.mean()),rmse=float(np.sqrt(np.mean(error**2))),median_absolute_error=float(np.median(absolute)),p95_absolute_error=float(np.percentile(absolute,95)))
 if percentage:
  ape=absolute/truth*100;result.update(mape_percent=float(ape.mean()),median_ape_percent=float(np.median(ape)),p95_ape_percent=float(np.percentile(ape,95)))
 groups=sorted({p['group'] for p in rows})
 if len(groups)>1:
  sums=np.array([[sum(abs(p[predicted]-p[expected]) for p in rows if p['group']==g),sum(p['group']==g for p in rows)] for g in groups])
  draws=np.random.default_rng(42).integers(len(groups),size=(2000,len(groups)));sample=sums[draws].sum(1)
  result['mae_95ci_acquisition_bootstrap']=np.percentile(sample[:,0]/sample[:,1],[2.5,97.5]).tolist()
  if percentage:
   sums[:,0]=[sum(abs(p[predicted]-p[expected])/p[expected]*100 for p in rows if p['group']==g) for g in groups]
   sample=sums[draws].sum(1);result['mape_95ci_acquisition_bootstrap']=np.percentile(sample[:,0]/sample[:,1],[2.5,97.5]).tolist()
 return result

def summarize(results):
 output={}
 for split in ('test','validation','all_heldout'):
  subset=[r for r in results if split=='all_heldout' or r['split']==split];modes={}
  for mode in MODES:
   rows=[r for r in subset if r['mode']==mode]
   totals={key:sum(r[key] for r in rows) for key in ('references','complete_references','cells_before_filter','final_cells','signals','matched_before_filter','matched_final','unscored_before_filter','unscored_final')}
   totals['scans']=len(rows);totals['annotated_recall_before_filter']=totals['matched_before_filter']/totals['references'] if totals['references'] else None
   totals['annotated_recall_after_filter']=totals['matched_final']/totals['references'] if totals['references'] else None
   for field,a,b,pct in [('volume_pairs','manual_um3','predicted_um3',True),('distance_pairs','manual_um','predicted_um',False)]:
    pairs=[dict(p,group=r['group']) for r in rows for p in r[field]];totals[field]=paired_stats(pairs,a,b,pct)
   modes[mode]=totals
  shared={}
  by_image={r['image'] for r in subset}
  for field,key,a,b,pct in [('volume_pairs',lambda p:p['reference_id'],'manual_um3','predicted_um3',True),
                           ('distance_pairs',lambda p:(p['reference_id'],p['channel'],tuple(np.round(p['manual_xyz'],4))),'manual_um','predicted_um',False)]:
   pairs={m:[] for m in MODES}
   for image in by_image:
    rows={r['mode']:r for r in subset if r['image']==image}
    if set(rows)!=set(MODES):continue
    index={m:{key(p):p for p in rows[m][field]} for m in MODES};common=set.intersection(*(set(v) for v in index.values()))
    for m in MODES:pairs[m].extend(dict(index[m][k],group=rows[m]['group']) for k in common)
   shared[field]={m:paired_stats(v,a,b,pct) for m,v in pairs.items()}
  output[split]=dict(modes=modes,common_matches=shared)
 return output

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);p.add_argument('--resume',type=Path)
 p.add_argument('--shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
 p.add_argument('--device',choices=['cpu','gpu','auto'],default='gpu');p.add_argument('--audit',type=Path,default=ROOT/'artifacts/refinement-fish/audit/audit.json')
 args=p.parse_args();collection=json.loads(args.audit.read_text());models=frozen_models()
 if collection['status']!='complete':raise ValueError('Completed audit required.')
 cases=sorted([c for c in collection['cases'] if c['audit']['split'] in ('validation','test')],key=lambda c:(0 if c['audit']['split']=='test' else 1,c['image']))
 groups={s:{c['audit']['group'] for c in collection['cases'] if c['audit']['split']==s} for s in ('training','validation','test')}
 assert not any(groups[a]&groups[b] for a,b in [('training','test'),('training','validation'),('validation','test')])
 old=json.loads((ROOT/'artifacts/full-fish/audit/audit.json').read_text());old_split={c['image']:c['audit']['split'] for c in old['cases']}
 assert all(old_split[c['image']]==c['audit']['split'] for c in cases),'StarDist and crop training splits disagree.'
 signal_index=json.loads(Path(models['signals']['manifest']['data_index']).read_text())
 signal_training={c['image'] for c in signal_index['cases'] if c['split']=='training'}
 assert not signal_training & {c['image'] for c in cases},'Signal training overlaps held-out scans.'
 for s in collection['sources']:
  stat=Path(s['image']).stat()
  if stat.st_size!=s['size'] or stat.st_mtime_ns!=s['mtime_ns']:raise ValueError('Audited scan changed: '+s['image'])
  for scene in s.get('scene_stats',[]):
   stat=Path(scene['path']).stat()
   if stat.st_size!=scene['size'] or stat.st_mtime_ns!=scene['mtime_ns']:raise ValueError('Audited annotations changed: '+scene['path'])
 output=args.resume or args.output or ROOT/'artifacts'/('all-heldout-'+datetime.now().strftime('%Y%m%d-%H%M%S'));output.mkdir(parents=True,exist_ok=True)
 identity=dict(audit_sha256=sha(args.audit),models=models,device=args.device)
 if args.resume:
  if json.loads((output/'identity.json').read_text())!=identity:raise ValueError('Frozen models/audit/device differ from resumed run.')
 else:
  save(output/'identity.json',identity);save(output/'selection.json',dict(cases=cases,split_counts=dict(Counter(c['audit']['split'] for c in cases)),
      policy='Every unique held-out scan, no sampling. Acquisition splits checked across cell and signal training; manual aliases merged for evaluation only.'))
 (ROOT/'artifacts/current-all-heldout.txt').write_text(str(output.resolve())+'\n')
 if args.shards<1 or not 0<=args.shard_index<args.shards:raise ValueError('Invalid shard selection.')
 suffix=f'-shard-{args.shard_index}' if args.shards>1 else ''
 app=QtCore.QCoreApplication([]);controller=AnalysisController();queue=[(i,c,m) for i,c in enumerate(cases) if i%args.shards==args.shard_index for m in MODES];results=[];failures=[];active={};started=time.monotonic();last=[0.]
 def persist():save(output/f'progress{suffix}.json',dict(status='running',completed=len(results),failures=failures,total=len(queue),elapsed_s=time.monotonic()-started,active={k:v for k,v in active.items() if k in ('image','mode','index')}))
 def progress(e):
  if time.monotonic()-last[0]>=2:
   last[0]=time.monotonic();print(f"{active['index']+1}/{len(queue)} {active['mode']} {Path(active['image']).name}: {e['stage']}/{e['stages']} {e['phase']} {e.get('current',0)}/{e.get('total',0)}",flush=True);persist()
 def complete(state):
  try:
   save(active['folder']/'pipeline-result.json',state)
   metrics=evaluate_run(active['case'],state,collection,active['folder'].parent);metrics['mode']=active['mode']
   save(active['folder']/'metrics.json',metrics);results.append(metrics)
   print('COMPLETE',len(results),len(queue),active['mode'],metrics['scan'],metrics['matched_before_filter'],metrics['matched_final'],flush=True);persist()
   save(output/f'partial-summary{suffix}.json',summarize(results))
   QtCore.QTimer.singleShot(0,next_run)
  except Exception:
   import traceback;fail(traceback.format_exc())
 def fail(message):
  failures.append(dict(image=active['image'],mode=active['mode'],error=message));persist();print(message,flush=True);app.exit(1)
 original_launch=controller._launch
 def launch(index,arguments):
  if controller.stages[index][0]=='export':
   controller.state.update(status='complete',benchmark_only=True,export_omitted=True)
   controller._stop();complete(dict(controller.state,report_path=str(controller.directory/'analysis-result.json')))
  else:original_launch(index,arguments)
 controller._launch=launch
 def next_run():
  if len(results)==len(queue):app.quit();return
  i,case,mode=queue[len(results)];folder=output/(f'{i+1:03d}-'+Path(case['image']).stem)/mode;folder.mkdir(parents=True,exist_ok=True)
  active.update(index=len(results),case=case,image=case['image'],mode=mode,folder=folder)
  if (folder/'metrics.json').is_file():
   metrics=json.loads((folder/'metrics.json').read_text());results.append(metrics);QtCore.QTimer.singleShot(0,next_run);return
  if (folder/'pipeline-result.json').is_file():complete(json.loads((folder/'pipeline-result.json').read_text()));return
  if mode=='consensus':
   hybrid=json.loads((folder.parent/'hybrid/pipeline-result.json').read_text());default=json.loads((folder.parent/'default/pipeline-result.json').read_text())
   controller._launch=lambda *_:None
   try:controller.start(case['image'],folder/'runs',sensitive_stardist_model(),default_models()[1],device=args.device,refinement=default_refinement_model(),stardist_proposals=True,consensus_model=default_proposal_model())
   finally:controller._launch=launch
   controller.state['stages'].update(cells=hybrid['stages']['cells'],refine=hybrid['stages']['refine'],consensus_cells=default['stages']['cells'])
   controller.state['reused_cell_runs']=dict(hybrid=hybrid['report_path'],default=default['report_path']);controller._persist()
   launch(next(i for i,s in enumerate(controller.stages) if s[0]=='consensus'),['-m','fish_viewer.proposals.consensus',
      '--primary-report',str(Path(hybrid['stages']['refine'])/'report.json'),'--secondary-report',str(Path(default['stages']['cells'])/'report.json'),
      '--output',str(controller.directory/'work'),'--cancel-file',str(controller.cancel_file)])
  else:controller.start(case['image'],folder/'runs',sensitive_stardist_model() if mode=='hybrid' else default_proposal_model(),default_models()[1],device=args.device,refinement=default_refinement_model(),stardist_proposals=mode=='hybrid')
 controller.progress.connect(progress);controller.completed.connect(complete);controller.failed.connect(fail)
 QtCore.QTimer.singleShot(0,next_run);code=app.exec()
 if code or len(results)!=len(queue):return 1
 save(output/f'report{suffix}.json',dict(status='complete',selection=str(output/'selection.json'),identity=identity,results=results,summary=summarize(results),failures=failures,
     limitations='Validation influenced model selection; test scans were used in earlier exploratory comparisons. No weights/thresholds changed in this run. Sparse annotations cannot estimate precision of unmatched cells. Distances use explicit unambiguous manual A-B markers only. All outliers retained. Export alone omitted.'))
 save(output/f'progress{suffix}.json',dict(status='complete',completed=len(results),total=len(queue),failures=failures,elapsed_s=time.monotonic()-started))
 print('ALL HELD-OUT COMPLETE',output,flush=True);return 0

if __name__=='__main__':raise SystemExit(main())
