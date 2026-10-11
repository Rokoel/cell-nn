"""Export per-scan signal scores and representative calibrated review crops."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from fish_viewer.models import LibraryEntry
from fish_viewer.reader import load_dataset
from fish_viewer.signals.data import normalized,xyz_to_grid
from fish_viewer.signals.evaluation import evaluate_points,summarize

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True)
args=parser.parse_args();report=json.loads(args.report.read_text());audit=json.loads(Path(report['audit_path']).read_text())
directory=args.report.parent;rows=[];examples=[]
for entry in report['files']:
    result=json.loads(Path(entry['signals']).read_text());metric=summarize([evaluate_points(result['points'],result['truth'],report['config']['match_radius_um'])])
    rows.append(dict(image=entry['image'],**{k:metric[k] for k in ('tp','fp','fn','precision','recall','localization_rmse_um')}))
    if result['points'] and len(examples)<4:examples.append((entry,result))
with (directory/'test-by-scan.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
summary={key:report[key] for key in ('config','best_epoch','training_candidates','validation','test','candidate_ceiling','metric_notes')}
summary['data_summary']=audit['summary'];summary['completed_epochs']=len(report['epochs']);summary['score_threshold']=report['selection']['selected']['threshold']
(directory/'summary.json').write_text(json.dumps(summary,indent=2))
fig=Figure(figsize=(12,3.5*len(examples)),layout='constrained');FigureCanvasAgg(fig);axes=fig.subplots(len(examples),3,squeeze=False)
for row,(entry,result) in enumerate(examples):
    case=next(c for c in audit['cases'] if c['image']==entry['image'])
    point=result['points'][0];center=np.array(point['xyz']);channel=point['channel'];spacing=np.array(case['spacing_zyx']);origin=np.array(case['origin_xyz'])
    matched=evaluate_points([point],result['truth'],report['config']['match_radius_um'])['tp']>0
    dataset=load_dataset(LibraryEntry(Path(entry['image']),Path(entry['scene']) if entry['scene'] else None));image=normalized(dataset.volumes[channel])
    central=xyz_to_grid([center],origin,spacing)[0];half=np.ceil(1.7/spacing).astype(int)
    low=np.maximum(0,np.rint(central).astype(int)-half);high=np.minimum(image.shape,np.rint(central).astype(int)+half+1)
    crop=tuple(slice(int(a),int(b)) for a,b in zip(low,high));image=image[crop]
    predictions=[p for p in result['points'] if p['channel']==channel and p['cell_id']==point['cell_id']]
    references=[p for p in result['truth'] if p['channel']==channel and p['cell_id']==point['cell_id']]
    for column,(normal,plane) in enumerate([(0,'XY'),(1,'XZ'),(2,'YZ')]):
        ax=axes[row,column];keep=[a for a in range(3) if a!=normal];index=int(np.clip(round(central[normal]-low[normal]),0,image.shape[normal]-1))
        sliced=np.take(image,index,axis=normal)
        extent=[(low[keep[1]]-.5)*spacing[keep[1]],(high[keep[1]]-.5)*spacing[keep[1]],(high[keep[0]]-.5)*spacing[keep[0]],(low[keep[0]]-.5)*spacing[keep[0]]]
        ax.imshow(sliced,cmap='gray',vmin=0,vmax=1,extent=extent)
        for points,color,marker,label in [(references,'#ffca42','o','A marker'),(predictions,'#2ac9b3','+','NN signal')]:
            coords=xyz_to_grid([p['xyz'] for p in points],origin,spacing)
            coords=coords[np.all((coords>=low)&(coords<high),axis=1)]*spacing
            if len(coords):ax.scatter(coords[:,keep[1]],coords[:,keep[0]],facecolors='none' if marker=='o' else color,edgecolors=color if marker=='o' else None,marker=marker,s=90,label=label)
        ax.set_xlim(extent[:2]);ax.set_ylim(extent[2:])
        ax.set_title(f'{Path(entry["image"]).name}\n{plane} · channel {channel+1} · cell {point["cell_id"]} · '+('matched' if matched else 'unmatched'),fontsize=9)
        ax.set_xlabel('µm from first voxel center');ax.legend(fontsize=7)
fig.suptitle('Signal review crops: markers projected within the same cell; image is one slice',fontsize=12)
fig.savefig(directory/'review-crops.png',dpi=140)
print(directory/'summary.json')
