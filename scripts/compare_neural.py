"""Compare saved runs on identical test inputs and a declared reference subset."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from fish_viewer.neural.audit import read_references
from fish_viewer.pipeline.evaluation import evaluate,aggregate
from fish_viewer.pipeline.runner import _json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--strict',type=Path,help='Include a separate conservative model in the comparison')
    parser.add_argument('--current-name',default='conservative',choices=['conservative','balanced'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit=json.loads(args.audit.read_text());baseline=json.loads(args.baseline.read_text());current=json.loads(args.run.read_text())
    if baseline['status']!='complete' or current['status']!='complete':raise ValueError('Both runs must be complete.')
    cases={c['image']:c for c in audit['cases']}
    reports={'baseline':baseline}
    if args.strict:reports['strict']=json.loads(args.strict.read_text())
    reports[args.current_name]=current;summary={};maps={}
    for name,report in reports.items():
        if report['status']!='complete':raise ValueError(f'{name} is not complete.')
        maps[name]={r['image']:r for r in report['files']}
        test=[maps[name][c['image']] for c in audit['cases'] if c['audit']['split']=='test']
        if any(r['split']!='test' for r in test):raise ValueError('Test split mismatch.')
        reference_scores=[]
        for row in test:
            case=cases[row['image']];a=case['audit']
            refs,names=read_references(Path(case['directory'])/'references.npz')
            ids=a['training_reference_ids']
            with np.load(row['predictions'],allow_pickle=False) as archive:labels=archive['labels']
            reference_scores.append(evaluate(labels,[refs[i-1] for i in ids],[names[i-1] for i in ids],a['spacing_zyx']))
        details=[c for m in reference_scores for c in m['per_cell']]
        matched=[c for c in details if c['detected']]
        rejected=Counter()
        for row in test:rejected.update(row['rejected'])
        summary[name]=dict(test_images=len(test),predictions=sum(r['metrics']['predicted_count'] for r in test),
            unscored=sum(r['metrics']['unscored_predictions'] for r in test),
            eligible_references=len(details),eligible_detected=sum(c['detected'] for c in details),
            eligible_macro=aggregate(reference_scores),
            eligible_pooled_dice=float(np.mean([c['dice'] for c in details])),
            eligible_detected_mean_dice=float(np.mean([c['dice'] for c in matched])) if matched else None,
            rejections=dict(rejected),best_epoch=report['best_epoch'])
    summary['prediction_reduction_fraction']=1-summary[args.current_name]['predictions']/summary['baseline']['predictions']
    count=summary[args.current_name]['eligible_references']
    summary['note']=f'Same test images and {count} shape-eligible complete references. Misses count as zero. Fewer contours is not measured false-positive reduction. Precision is unknown.'
    args.output.mkdir(parents=True,exist_ok=True);_json(args.output/'comparison.json',summary)
    # Fixed validation previews are illustrative, never used to retune thresholds.
    images=current['validation_preview_images'][::3]
    columns=['image',*reports]
    fig=Figure(figsize=(4.4*len(columns),4*len(images)),layout='constrained');FigureCanvasAgg(fig)
    axes=fig.subplots(len(images),len(columns),squeeze=False)
    for row_index,path in enumerate(images):
        c=cases[path];a=c['audit'];image=np.load(Path(c['directory'])/'image.npy',mmap_mode='r')
        refs,_=read_references(Path(c['directory'])/'references.npz')
        selected=a['training_reference_ids']
        box,mask=refs[(selected[0] if selected else 1)-1]
        center=np.rint(np.argwhere(mask).mean(0)+[s.start for s in box]).astype(int)
        z=int(center[0]);y,x=center[1:];ys=slice(max(0,y-55),min(image.shape[1],y+55));xs=slice(max(0,x-55),min(image.shape[2],x+55))
        for column,name in enumerate(columns):
            ax=axes[row_index,column];ax.imshow(image[z,ys,xs],cmap='gray',vmin=0,vmax=1)
            if name!='image':
                with np.load(maps[name][path]['predictions'],allow_pickle=False) as archive:plane=archive['labels'][z,ys,xs]
                for label in np.unique(plane):
                    if label and (plane!=label).any():ax.contour(plane==label,levels=[.5],colors=['#00d590'],linewidths=.8)
            title='Image' if name=='image' else name.capitalize()+' contours'
            ax.set_title(f'{Path(path).name}\n{title} · Z={z}',fontsize=9)
            ax.set_xticks([]);ax.set_yticks([])
    fig.savefig(args.output/'comparison.png',dpi=160)
    print(json.dumps({name:{k:v for k,v in values.items() if k!='eligible_macro'} if isinstance(values,dict) else values for name,values in summary.items()},indent=2))


if __name__=='__main__':main()
