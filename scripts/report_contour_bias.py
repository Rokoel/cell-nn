"""Publish graphs, paired uncertainty and tables for the contour-bias experiment."""
import argparse,csv,json,shutil
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from fish_viewer.pipeline.runner import _json


def bootstrap(rows,before,after,truth,relative=False):
    groups=sorted({r['group'] for r in rows});rng=np.random.default_rng(42)
    errors={g:[] for g in groups}
    for r in rows:
        factor=100/r[truth] if relative else 1
        errors[r['group']].append(factor*(abs(r[before]-r[truth])-abs(r[after]-r[truth])))
    draws=[np.mean([e for g in rng.choice(groups,len(groups),replace=True) for e in errors[g]]) for _ in range(2000)]
    return dict(mean_improvement=float(np.mean([e for values in errors.values() for e in values])),ci95=np.percentile(draws,[2.5,97.5]).tolist(),groups=len(groups),pairs=len(rows))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'reports/contour-bias');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);run=ROOT/'artifacts/contour-bias-20261009'
    phases={name:json.loads((run/f'{name}/report.json').read_text()) for name in ('thresholds-v2','geometry')}
    for name,r in phases.items():_json(args.output/(name+'-calibration.json'),r)
    fig=Figure(figsize=(11,8),layout='constrained');FigureCanvasAgg(fig);axes=fig.subplots(2,2)
    rows=list(phases['thresholds-v2']['summary'].items());x=[float(k.split('-')[-1]) for k,r in rows]
    for ax,key,title in [(axes[0,0],'boundary_error_um','Validation boundary error (µm)'),(axes[0,1],'volume_mape','Validation volume error (%)')]:
        ax.plot(x,[r[key] for k,r in rows],marker='o');ax.axvline(.45,color='green',linestyle='--');ax.axvline(.5,color='gray',linestyle=':')
        ax.set(xlabel='Contour probability cutoff',title=title);ax.grid(alpha=.2)
    rows=list(phases['geometry']['summary'].items());labels=[k.replace('margin-','').replace('-size-',' µm / ')+'³' for k,r in rows]
    for ax,key,title in [(axes[1,0],'boundary_error_um','Changing crop geometry without retraining'),(axes[1,1],'volume_bias_percent','Signed volume bias (%)')]:
        ax.bar(range(len(rows)),[r[key] for k,r in rows]);ax.set_xticks(range(len(rows)),labels,rotation=25,ha='right');ax.set(title=title);ax.grid(axis='y',alpha=.2)
    fig.savefig(args.output/'calibration.png',dpi=170)
    post=run/'post-training-thresholds/report.json'
    if post.exists():shutil.copy2(post,args.output/'post-training-thresholds.json')
    state=json.loads((run/'training-stages.json').read_text());_json(args.output/'training-stages.json',state)
    for stage in state['stages']:
        directory=args.output/stage['stage'];directory.mkdir(exist_ok=True)
        train=Path(stage['training_report']);shutil.copy2(train,directory/'training.json')
        shutil.copy2(train.parent/'training-progress.png',directory/'training-progress.png')
        shutil.copy2(train.parent/'epochs.csv',directory/'epochs.csv')
        shutil.copy2(run/(stage['stage']+'-validation')/'report.json',directory/'validation.json')
    test=run/'test-comparison/comparison.json'
    if test.exists():
        comparison=json.loads(test.read_text());cells=comparison['cells']
        applied=json.loads((run/'test-contours/report.json').read_text())
        comparison['summary']['total_preserved_cell_instances']=sum(r['cells'] for r in applied['files'])
        comparison['summary']['accepted_boundary_edits']=sum(d['accepted'] for r in applied['files'] for d in r['refinement'])
        comparison['summary']['retained_original_boundaries']=sum(not d['accepted'] for r in applied['files'] for d in r['refinement'])
        comparison['summary']['baseline_mean_signed_volume_percent']=float(np.mean([(r['baseline_volume_um3']/r['manual_volume_um3']-1)*100 for r in cells]))
        comparison['summary']['refined_mean_signed_volume_percent']=float(np.mean([(r['refined_volume_um3']/r['manual_volume_um3']-1)*100 for r in cells]))
        ci=bootstrap(cells,'baseline_volume_um3','refined_volume_um3','manual_volume_um3',True)
        _json(args.output/'paired-volume-uncertainty.json',ci)
        shutil.copytree(test.parent,args.output/'test-comparison',dirs_exist_ok=True)
        _json(args.output/'test-comparison/report.json',comparison)
        signals=json.loads((run/'test-signals/report.json').read_text());shutil.copytree(run/'test-signals',args.output/'test-signals',dirs_exist_ok=True)
        fig=Figure(figsize=(12,8),layout='constrained');FigureCanvasAgg(fig);axes=fig.subplots(2,2)
        for ax,source,before,after,truth,title,relative in [
            (axes[0,0],cells,'baseline_volume_um3','refined_volume_um3','manual_volume_um3','Volume error (%)',True),
            (axes[0,1],cells,'baseline_volume_um3','refined_volume_um3','manual_volume_um3','Signed volume bias (%)',True),
            (axes[1,0],signals['rows'],'baseline_distance_um','refined_distance_um','manual_ab_distance_um','Distance absolute error (µm)',False),
            (axes[1,1],cells,'baseline_surface_mean_error_um','refined_surface_mean_error_um',None,'Symmetric surface error (µm)',False)]:
            a=np.array([r[before] for r in source]);b=np.array([r[after] for r in source])
            if truth:
                t=np.array([r[truth] for r in source]);a-=t;b-=t
                if relative:a=a/t*100;b=b/t*100
                if 'Signed' not in title:a=abs(a);b=abs(b)
            bins=np.histogram_bin_edges(np.r_[a,b],bins=25)
            ax.hist(a,bins=bins,alpha=.5,label='Current default');ax.hist(b,bins=bins,alpha=.5,label='Validation-selected candidate')
            ax.set(xlabel=title,ylabel='Count',title=f'{len(source)} fixed pairs');ax.grid(alpha=.2);ax.legend()
        fig.savefig(args.output/'test-distributions.png',dpi=170)
        _json(args.output/'summary.json',dict(selection=state,test=comparison['summary'],fixed_unet_signals=signals['summary'],paired_volume_uncertainty=ci))
    print(str(args.output.resolve()))
if __name__=='__main__':main()
