"""Freeze latest U-Net signal centers; compare only changed cell boundaries."""
import argparse,csv,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from fish_viewer.pipeline.runner import read_prediction,_json
from fish_viewer.signals.data import at_labels
from fish_viewer.workflow.distances import SurfaceLocator
from benchmark_measurements import paired_stats


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cell-comparison',type=Path);args=p.parse_args()
    if args.output.exists():p.error('Choose a new output directory')
    args.output.mkdir(parents=True)
    run=json.loads(args.report.read_text());files={str(Path(r['image']).relative_to(ROOT)):r for r in run['files']}
    original=json.loads((ROOT/'reports/signal-centers/measurements/report.json').read_text())
    with (ROOT/'reports/signal-centers/measurements/unet-distances.csv').open(newline='') as f:raw=list(csv.DictReader(f))
    grouped={}
    for r in raw:grouped.setdefault(r['image'],[]).append(r)
    rows=[]
    for number,(image,points) in enumerate(grouped.items(),1):
        print(f'Fixed U-Net distances {number}/{len(grouped)}: {Path(image).name}',flush=True)
        archive=files[image]['predictions'];surfaces,info=read_prediction(archive)
        meshes={int(s.name.rsplit(' ',1)[-1]):s for s in surfaces};locators={}
        with np.load(archive) as a:labels=a['labels']
        for r in points:
            cell=int(r['cell_id']);a=json.loads(r['predicted_a_xyz'])
            if cell not in locators:locators[cell]=SurfaceLocator(meshes[cell])
            b,distance,_=locators[cell].closest(a)
            rows.append(dict(image=image,group=r['group'],cell_id=cell,channel=int(r['channel']),
                manual_a_xyz=json.loads(r['manual_a_xyz']),predicted_a_xyz=a,refined_b_xyz=b.tolist(),
                manual_ab_distance_um=float(r['manual_ab_distance_um']),baseline_distance_um=float(r['predicted_distance_um']),
                refined_distance_um=float(distance),predicted_a_inside_refined=int(at_labels(labels,[a],info['origin_xyz'],info['spacing_zyx'])[0])==cell))
    summary=dict(fixed_signals=len(rows),outside_refined_cell=sum(not r['predicted_a_inside_refined'] for r in rows),
        baseline=paired_stats(rows,'manual_ab_distance_um','baseline_distance_um'),
        refined=paired_stats(rows,'manual_ab_distance_um','refined_distance_um'))
    if args.cell_comparison:
        cells=json.loads(args.cell_comparison.read_text())['cells']
        detection=json.loads((ROOT/'reports/signal-centers/unet-test/report.json').read_text())
        retained={(str(Path(r['image']).relative_to(ROOT)),p['cell_id']) for r in detection['scans'] for p in r['points']}
        fixed=[r for r in cells if (r['image'],r['predicted_cell']) in retained]
        summary['fixed_unet_retained_cells']=dict(cells=len(fixed),
            baseline=paired_stats(fixed,'manual_volume_um3','baseline_volume_um3'),
            refined=paired_stats(fixed,'manual_volume_um3','refined_volume_um3'),
            policy='Same original 150 matched cells retained by scored-ROI U-Net detections; no refiltering or new signal inference')
    # Resample complete acquisition groups, preserving all signal pairs.
    rng=np.random.default_rng(42);groups=sorted({r['group'] for r in rows});improvements=[]
    for _ in range(2000):
        sample=[r for group in rng.choice(groups,len(groups),replace=True) for r in rows if r['group']==group]
        improvements.append(np.mean([abs(r['baseline_distance_um']-r['manual_ab_distance_um'])-abs(r['refined_distance_um']-r['manual_ab_distance_um']) for r in sample]))
    summary['paired_mae_improvement_um']=float(np.mean([abs(r['baseline_distance_um']-r['manual_ab_distance_um'])-abs(r['refined_distance_um']-r['manual_ab_distance_um']) for r in rows]))
    summary['paired_mae_improvement_ci95_um']=np.percentile(improvements,[2.5,97.5]).tolist()
    result=dict(status='complete',summary=summary,rows=rows,policy='Same 140 eligible manual A/B identities and frozen U-Net predicted centers. No signal rerun or new matching. All outliers retained. Outside-center failures explicitly reported; full fixed cohort remains scored.')
    _json(args.output/'report.json',result)
    with (args.output/'distances.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()})
    print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
