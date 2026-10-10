"""Check complete coverage and publish the three-mode held-out benchmark."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from benchmark_all_heldout import MODES,paired_stats,summarize,save,sha


def aggregate(root):
 selection=json.loads((root/'selection.json').read_text());cases=selection['cases'];identity=json.loads((root/'identity.json').read_text())
 for kind,model in identity.get('models',{}).items():
  directory=Path(model['directory'])
  manifest_name={'default_detector':'proposal-model.json','hybrid_detector':'fish-model.json','contours':'refinement-model.json','signals':'signal-model.json'}[kind]
  if sha(directory/model['manifest']['checkpoint'])!=model['checkpoint_sha256'] or sha(directory/manifest_name)!=model['manifest_sha256']:
   raise ValueError('Frozen model changed during the benchmark: '+kind)
 results=[];expected=set()
 for i,case in enumerate(cases):
  for mode in MODES:
   expected.add((case['image'],mode));folder=root/(f'{i+1:03d}-'+Path(case['image']).stem)/mode
   row=json.loads((folder/'metrics.json').read_text())
   if row['image']!=case['image'] or row['mode']!=mode or row['split']!=case['audit']['split']:raise ValueError('Incorrect scan/mode/split in '+str(folder))
   state=json.loads((folder/'pipeline-result.json').read_text())
   if state['status']!='complete':raise ValueError('Incomplete pipeline '+str(folder))
   cuts=case['audit']['reference_cut']
   for phase in ('before_filter','final'):
    matches=row['matches_'+phase]
    row['matched_complete_'+phase]=sum(not cuts[p['reference_id']] for p in matches)
    row['matched_cut_'+phase]=sum(cuts[p['reference_id']] for p in matches)
   for p in row['distance_pairs']:p['cut']=cuts[p['reference_id']]
   results.append(row)
 if len(results)!=len(expected) or {(r['image'],r['mode']) for r in results}!=expected:raise ValueError('Coverage does not match all held-out scans.')
 summary=summarize(results)
 for split,data in summary.items():
  subset=[r for r in results if split=='all_heldout' or r['split']==split]
  for mode,totals in data['modes'].items():
   rows=[r for r in subset if r['mode']==mode]
   totals.update({key:sum(r[key] for r in rows) for key in ('matched_complete_before_filter','matched_complete_final','matched_cut_before_filter','matched_cut_final')})
   totals['complete_cell_recall_before_filter']=totals['matched_complete_before_filter']/totals['complete_references'] if totals['complete_references'] else None
   totals['complete_cell_recall_after_filter']=totals['matched_complete_final']/totals['complete_references'] if totals['complete_references'] else None
   for field,a,b,pct in [('volume_pairs','manual_um3','predicted_um3',True),('distance_pairs','manual_um','predicted_um',False)]:
    pairs=[dict(p,group=r['group']) for r in rows for p in r[field] if not p['cut']]
    totals[field+'_complete_cells']=paired_stats(pairs,a,b,pct)
  common={}
  for field,key,a,b,pct in [('volume_pairs',lambda p:p['reference_id'],'manual_um3','predicted_um3',True),
                           ('distance_pairs',lambda p:(p['reference_id'],p['channel'],tuple(np.round(p['manual_xyz'],4))),'manual_um','predicted_um',False)]:
   pairs={m:[] for m in MODES}
   for image in {r['image'] for r in subset}:
    rows={r['mode']:r for r in subset if r['image']==image}
    index={m:{key(p):p for p in rows[m][field] if not p['cut']} for m in MODES}
    shared=set.intersection(*(set(v) for v in index.values()))
    for m in MODES:pairs[m].extend(dict(index[m][k],group=rows[m]['group']) for k in shared)
   common[field]={m:paired_stats(v,a,b,pct) for m,v in pairs.items()}
  data['common_matches_complete_cells']=common
 report=dict(status='complete',expected_scans=len(cases),expected_mode_runs=len(expected),completed_mode_runs=len(results),identity=identity,
   selection=selection,results=results,summary=summary,failures=[],
   policy='Every held-out unique scan; original acquisition split. Same frozen checkpoints and app processing functions. Consensus reuses the two freshly inferred cell sets but reruns signal inference and measurements. Only original-image/scene export omitted. All reference aliases/scenes merged and exact voxel masks deduplicated. Explicit manual AB distances only. All outliers retained. Partial cells reported separately.',
   limitations='Validation influenced checkpoint selection. Test acquisitions were previously used in exploratory model comparisons; this is a full held-out benchmark, not a new never-inspected test cohort. Incomplete annotations cannot estimate precision of unmatched cells. Error metrics are conditional on manual-cell and signal matches.')
 save(root/'report.json',report);save(root/'progress.json',dict(status='complete',completed=len(results),total=len(expected),failures=[]))
 return report


def publish(report,destination):
 destination.mkdir(parents=True,exist_ok=True)
 summary={k:report[k] for k in ('status','expected_scans','expected_mode_runs','completed_mode_runs','summary','policy','limitations')}
 summary['models']={k:{key:v[key] for key in ('directory','manifest_sha256','checkpoint_sha256')} for k,v in report['identity']['models'].items()}
 save(destination/'summary.json',summary)
 fields=('split','mode','scan','group','references','complete_references','cells_before_filter','matched_before_filter','matched_complete_before_filter','final_cells','matched_final','matched_complete_final','signals','unscored_final')
 with (destination/'per-scan.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r[k] for k in fields} for r in report['results'])
 for field in ('volume_pairs','distance_pairs'):
  rows=[dict(split=r['split'],mode=r['mode'],scan=r['scan'],group=r['group'],**p) for r in report['results'] for p in r[field]]
  with (destination/(field+'.csv')).open('w',newline='') as f:
   fields=list(dict.fromkeys(k for r in rows for k in r));writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
 lines=['# Full held-out comparison','','All 144 unique held-out scans were processed with all three modes (432 completed runs). Test and validation are separate. No weights or thresholds were adjusted.','']
 labels={'default':'Current default','hybrid':'Sensitive StarDist hybrid','consensus':'Both-detector consensus'}
 for split,title in [('test','Test scans'),('validation','Validation scans'),('all_heldout','All held-out scans')]:
  lines += ['## '+title,'','| Mode | Scans | Cells before / after signal filtering | Complete manual matches before / after | Volume MAPE (pairs) | Distance MAE, µm (pairs) |','|---|---:|---:|---:|---:|---:|']
  modes=report['summary'][split]['modes']
  for m in MODES:
   r=modes[m];v=r['volume_pairs_complete_cells'];d=r['distance_pairs_complete_cells']
   lines.append(f"| {labels[m]} | {r['scans']} | {r['cells_before_filter']} / {r['final_cells']} | {r['matched_complete_before_filter']} / {r['matched_complete_final']} of {r['complete_references']} | {v.get('mape_percent',0):.2f}% ({v['n']}) | {d.get('mae',0):.3f} ({d['n']}) |")
  lines+=['','Volumes and distances above use complete manual cells. All-object and cut-cell detection totals are retained in the JSON and CSV reports. Different modes have different matched subsets.','', '| Metric on the same complete-cell references | Default | Hybrid | Consensus |','|---|---:|---:|---:|']
  common=report['summary'][split]['common_matches_complete_cells']
  for field,label,value in [('volume_pairs','Volume MAPE (%)','mape_percent'),('distance_pairs','Distance MAE (µm)','mae')]:
   rows=common[field];n=rows['default']['n'];lines.append('| '+label+f' ({n} pairs) | '+' | '.join(f"{rows[m].get(value,0):.3f}" for m in MODES)+' |')
  lines.append('')
 lines += ['## Interpretation and reproducibility','','Cell matching requires one-to-one whole-object IoU ≥ 0.5. Distance matching requires the same cell and signal channel, with signal centers within 0.6 µm. Physical volumes are measured on closed triangle meshes; distances use nearest points on those meshes and explicit original manual A–B segments. All annotation aliases are considered, exact reference masks are deduplicated, and ambiguous manual channels/assignments or conflicting A–B segments are reported as exclusions. No manual annotations guide inference.','', 'The JSON includes biases, MAE/RMSE, median and 95th-percentile errors, and 95% acquisition-group bootstrap intervals. All outliers are retained. Unmatched cells remain unscored because manual annotations are incomplete; their number is not a false-positive count. Validation was used for model selection and test scans have been inspected in earlier experiments.','', 'Run `scripts/benchmark_all_heldout.py` with `FISH_SPOTS_PYTHON` pointing to the PyTorch environment. `--resume` preserves completed runs and checks model/audit hashes. Disjoint `--shards`/`--shard-index` jobs can share a run directory. `scripts/summarize_all_heldout.py --run RUN_DIRECTORY` refuses to publish if any held-out scan/mode is missing. Only Imaris image/scene export is omitted in this benchmark.']
 lines += ['', '[Test-set distribution graphs](test-comparison.png). Raw per-scan counts and matched volume/distance pairs are available in the CSV files beside this report.']
 (destination/'README.md').write_text('\n'.join(lines)+'\n')


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--publish',type=Path,default=ROOT/'reports/all-heldout-comparison');args=p.parse_args()
 report=aggregate(args.run);publish(report,args.publish);print(json.dumps(report['summary'],indent=2))

if __name__=='__main__':main()
