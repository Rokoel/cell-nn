"""Compare frozen direct-center border distances using verified cached manual geometry."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from benchmark_measurements import paired_stats
from fish_viewer.pipeline.runner import read_prediction
from fish_viewer.workflow.distances import SurfaceLocator
from fish_viewer.workflow.volumes import surface_volume

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'reports/signal-centers/measurements')
args=parser.parse_args()
output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
index=json.loads((ROOT/'artifacts/signal-centers/data/index.json').read_text())
parent=json.loads(Path(index['parent_audit']).read_text())
old=json.loads((ROOT/'reports/center-box-contours/test-measurements/report.json').read_text())
cell_run=json.loads(Path(old['cell_report']).read_text())
cell_paths={r['image']:r['predictions'] for r in cell_run['files']}
cases={c['image']:c for c in index['cases'] if c['split']=='test'}
if set(cases)!=set(cell_paths):raise ValueError('Cell measurement scans differ from signal test scans.')
for image,case in cases.items():
    if Path(case['cell_predictions']).resolve()!=Path(cell_paths[image]).resolve():raise ValueError('Cell contours differ from cached measurements.')
volumes=[r for r in old['volumes'] if r['complete'] and r['iou']>=.5 and r['manual_volume_um3'] is not None]
manual_lookup={(str((ROOT/r['image']).resolve()),r['predicted_cell'],r['channel'],tuple(np.round(r['manual_a_xyz'],4))):r
    for r in old['contour_distances'] if r['manual_ab_distance_um'] is not None and r['manual_ab_distance_um']>0 and r['channel'] is not None}
reports={name:json.loads((ROOT/f'reports/signal-centers/{name}-test/report.json').read_text()) for name in ('unet','spotiflow')}
scans={name:{r['image']:r for r in report['scans']} for name,report in reports.items()}
rows={name:[] for name in reports};retained={name:set() for name in reports};missing={name:0 for name in reports}
for number,(image,case) in enumerate(cases.items(),1):
    print(f'{number}/{len(cases)} {Path(image).name}',flush=True)
    surfaces,metadata=read_prediction(case['cell_predictions'])
    if metadata['fingerprint']!=case['fingerprint']:raise ValueError('Prediction fingerprint changed.')
    meshes={int(s.name.rsplit(' ',1)[-1]):s for s in surfaces}
    for row in volumes:
        if str((ROOT/row['image']).resolve())==image and not np.isclose(surface_volume(meshes[row['predicted_cell']]),row['predicted_volume_um3'],rtol=1e-8,atol=1e-8):
            raise ValueError('Cached volume geometry differs.')
    locators={}
    for name,report in reports.items():
        predictions=scans[name][image]['points']; truth=case['truth']
        retained[name].update((image,p['cell_id']) for p in predictions)
        if not predictions or not truth:continue
        distances=cdist([p['xyz'] for p in predictions],[p['xyz'] for p in truth])
        compatible=np.array([[p['cell_id']==r['cell_id'] and p['channel']==r['channel'] for r in truth] for p in predictions])
        cost=np.where(compatible&(distances<=.6),distances,1e6)
        left,right=linear_sum_assignment(cost)
        matched=[(a,b) for a,b in zip(left,right) if cost[a,b]<1e6]
        if len(matched)!=scans[name][image]['metrics']['tp']:raise ValueError('Matching differs from frozen detection benchmark.')
        for a,b in matched:
            prediction,manual=predictions[a],truth[b]
            key=(image,manual['cell_id'],manual['channel']+1,tuple(np.round(manual['xyz'],4)))
            reference=manual_lookup.get(key)
            if reference is None:missing[name]+=1;continue
            cell=prediction['cell_id']
            if cell not in locators:locators[cell]=SurfaceLocator(meshes[cell])
            border,distance,_=locators[cell].closest(prediction['xyz'])
            rows[name].append(dict(image=reference['image'],group=case['group'],manual_cell=reference['manual_cell'],
                cell_id=cell,channel=prediction['channel']+1,manual_a_xyz=manual['xyz'],predicted_a_xyz=prediction['xyz'],
                predicted_b_xyz=border.tolist(),manual_ab_distance_um=reference['manual_ab_distance_um'],
                manual_mesh_distance_um=reference['manual_mesh_distance_um'],predicted_distance_um=float(distance),
                localization_error_um=float(distances[a,b])))

def key(r):return (r['image'],r['channel'],tuple(np.round(r['manual_a_xyz'],4)))
lookup={name:{key(r):r for r in records} for name,records in rows.items()}
common=set.intersection(*(set(v) for v in lookup.values()))
baseline={key(r):r for r in old['end_to_end_distances'] if r['manual_ab_distance_um'] is not None and r['manual_ab_distance_um']>0}
shared_all=common&set(baseline)
summary=dict(volume_all_same_contours=paired_stats(volumes,'manual_volume_um3','predicted_volume_um3'),models={},
    shared_distance_manual_signals=len(common),shared_all_three_distance_manual_signals=len(shared_all),
    baseline_same_shared_all_three=paired_stats([baseline[k] for k in shared_all],'manual_ab_distance_um','predicted_distance_um'))
for name in reports:
    v=[r for r in volumes if (str((ROOT/r['image']).resolve()),r['predicted_cell']) in retained[name]]
    summary['models'][name]=dict(total_matched_centers=reports[name]['summary']['tp'],
        matched_centers_without_eligible_manual_distance=missing[name],
        distance_all_eligible_matches=paired_stats(rows[name],'manual_ab_distance_um','predicted_distance_um'),
        distance_against_manual_mesh=paired_stats(rows[name],'manual_mesh_distance_um','predicted_distance_um'),
        distance_shared_both_models=paired_stats([lookup[name][k] for k in common],'manual_ab_distance_um','predicted_distance_um'),
        distance_shared_all_three=paired_stats([lookup[name][k] for k in shared_all],'manual_ab_distance_um','predicted_distance_um'),
        volume_retained_in_scored_regions=paired_stats(v,'manual_volume_um3','predicted_volume_um3'))
    with (output/f'{name}-distances.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[name][0]));writer.writeheader();writer.writerows(rows[name])
paired=[]
for k in sorted(common):
    a,b=lookup['unet'][k],lookup['spotiflow'][k]
    paired.append(dict(group=a['group'],delta=abs(b['predicted_distance_um']-b['manual_ab_distance_um'])-abs(a['predicted_distance_um']-a['manual_ab_distance_um'])))
groups=sorted({r['group'] for r in paired})
if groups:
    sums=np.array([[sum(r['delta'] for r in paired if r['group']==g),sum(r['group']==g for r in paired)] for g in groups])
    ids=np.random.default_rng(42).integers(len(groups),size=(2000,len(groups)));samples=sums[ids].sum(axis=1)
    summary['paired_unet_distance_mae_improvement_over_spotiflow']=dict(n=len(paired),groups=len(groups),
        gain_um=float(np.mean([r['delta'] for r in paired])),acquisition_bootstrap_95ci_um=np.quantile(samples[:,0]/samples[:,1],[.025,.975]).tolist())
result=dict(status='complete',summary=summary,distances=rows,volumes=volumes,
    policy='Frozen test detector outputs; one-to-one matching reproduces the original 0.6 µm same-cell/channel benchmark. '
    'Distances require a complete manual cell matched at IoU>=0.5 and an unambiguous positive manual A/B line. '
    'All outliers included. Cell geometry and pixel fingerprints verified against the cached manual measurement benchmark. '
    'Shared-signal comparisons use identical manual A identities. Retained-volume subsets use only detections in the '
    'scored ROI, not all production detections in unknown regions. The shared full-cell volume result is independent '
    'of signal detection. Historically reused test fold remains exploratory.')
(output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps(summary,indent=2),flush=True)
