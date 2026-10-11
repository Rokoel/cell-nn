"""Archive paired physical contour measurements for the official 2D experiment."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fish_viewer.pipeline.runner import _json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'reports/stardist2d');args=p.parse_args()
    training=json.loads((args.run/'training.json').read_text())
    if training['status']!='complete':raise ValueError('Completed training required')
    args.output.mkdir(parents=True,exist_ok=True)
    methods={}
    for name in ('pretrained','finetuned'):
        subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_cell_contours.py'),'--audit',training['audit'],
            '--report',str(args.run/name/'report.json'),'--split','test','--output',str(args.output/name)],check=True)
        methods[name]=json.loads((args.output/name/'report.json').read_text())
    # Prior 3D evaluations used this exact same corrected-channel cache and split.
    for name in ('conservative','preprocessed'):
        source=ROOT/'reports/preprocessed-stardist'/name/'report.json'
        methods[name]=json.loads(source.read_text())
    maps={k:{(c['image'],c['manual_reference_id']):c for c in v['cells']} for k,v in methods.items()}
    common=set.intersection(*(set(v) for v in maps.values()))
    import numpy as np
    common_stats={k:dict(cells=len(common),volume_mape_percent=float(np.mean([abs(v[c]['predicted_volume_um3']/v[c]['manual_volume_um3']-1)*100 for c in common])) if common else None,
        surface_mean_error_um=float(np.mean([v[c]['surface_mean_error_um'] for c in common])) if common else None) for k,v in maps.items()}
    from itertools import combinations
    pairwise={}
    for left,right in combinations(maps,2):
        keys=set(maps[left])&set(maps[right])
        pairwise[left+' vs '+right]=dict(cells=len(keys),methods={name:dict(
            volume_mape_percent=float(np.mean([abs(maps[name][c]['predicted_volume_um3']/maps[name][c]['manual_volume_um3']-1)*100 for c in keys])) if keys else None,
            surface_mean_error_um=float(np.mean([maps[name][c]['surface_mean_error_um'] for c in keys])) if keys else None) for name in (left,right)})
    _json(args.output/'comparison.json',dict(training=training,methods={k:v['summary'] for k,v in methods.items()},common=common_stats,pairwise_intersections=pairwise))
    shutil.copy2(args.run/'training.json',args.output/'training-history.json')
    if (args.run/'synthetic-controls.json').exists():shutil.copy2(args.run/'synthetic-controls.json',args.output/'synthetic-controls.json')
    destination=ROOT/'models/cells-stardist2d'
    if not destination.exists():shutil.copytree(args.run/'model',destination)
    elif (destination/'best.weights.h5').read_bytes()!=(args.run/'model/best.weights.h5').read_bytes():raise ValueError('Refusing to replace an existing experiment')
    shutil.copy2(ROOT/'artifacts/stardist2d-official/LICENSE.txt',destination/'PRETRAINED-LICENSE.txt')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    history=training['epochs'];axes[0].plot([e['epoch'] for e in history],[e['losses']['loss'] for e in history],label='training')
    axes[0].plot([e['epoch'] for e in history],[e['losses']['val_loss'] for e in history],label='validation');axes[0].set(title='Fine-tuning loss',xlabel='Epoch');axes[0].legend()
    for name,method in methods.items():
        for ax,key in [(axes[1],'volume'),(axes[2],'surface')]:
            values=sorted(abs(c['predicted_volume_um3']/c['manual_volume_um3']-1)*100 if key=='volume' else c['surface_mean_error_um'] for c in method['cells'])
            if values:ax.plot(values,np.arange(1,len(values)+1)/len(values),label=name)
    axes[1].set(title='Matched-cell volume error',xlabel='Absolute percentage error',ylabel='Cumulative fraction');axes[2].set(title='Matched-cell surface error',xlabel='Mean symmetric error (µm)')
    axes[1].legend();axes[2].legend();fig.tight_layout();fig.savefig(args.output/'comparison.png',dpi=150);plt.close(fig)
    # Fixed-seed held-out examples; section selected at the first complete manual cell center.
    from fish_viewer.neural.audit import read_references
    audit=json.loads(Path(training['audit']).read_text())
    test=[c for c in audit['cases'] if c['audit']['split']=='test']
    example_rows={name:{r['image']:r for r in json.loads((args.run/name/'report.json').read_text())['files']} for name in ('pretrained','finetuned')}
    fig,axes=plt.subplots(4,3,figsize=(12,14))
    for row,index in enumerate(np.random.default_rng(1010).choice(len(test),4,replace=False)):
        case=test[index];directory=Path(case['directory'])
        image=np.load(directory/'evidence.npy',mmap_mode='r')
        refs,_=read_references(directory/'references.npz')
        complete=[n for n,cut in enumerate(case['audit']['reference_cut']) if not cut]
        z=(refs[complete[0]][0][0].start+refs[complete[0]][0][0].stop)//2 if complete else len(image)//2
        truth=np.zeros(image.shape[1:],bool)
        for box,mask in refs:
            if box[0].start<=z<box[0].stop:truth[box[1:]]|=mask[z-box[0].start]
        lo,hi=np.percentile(image[z],(1,99.5));hi=max(hi,lo+1e-6)
        for col,name in enumerate(('raw cell channel','pretrained','finetuned')):
            ax=axes[row,col];ax.imshow(image[z],cmap='gray',vmin=lo,vmax=hi)
            if truth.any():ax.contour(truth,levels=[.5],colors=['orange'],linewidths=.8)
            if col:
                prediction=example_rows[name][case['image']]['predictions']
                with np.load(prediction) as archive:plane=archive['labels'][z]
                for label in np.unique(plane):
                    if label:ax.contour(plane==label,levels=[.5],colors=['cyan'],linewidths=.6)
            ax.set_title(name if not row else '',fontsize=10);ax.axis('off')
        axes[row,0].text(0,-.04,f"{Path(case['image']).name}\ncell channel {case['audit']['channel']}, z={z}",transform=axes[row,0].transAxes,fontsize=7)
    fig.suptitle('Held-out XY examples: orange manual contours, cyan predictions',fontsize=12)
    fig.tight_layout();fig.savefig(args.output/'heldout-examples.png',dpi=130);plt.close(fig)
    lines=['# Official StarDist2D fluorescence fine-tuning','',
        'Official model: [2D_versatile_fluo](https://github.com/stardist/stardist-models/releases/tag/v0.1). Archive checksum verified against the installed official StarDist registry.',
        '',f"Fine-tuned {len(training['epochs'])} epochs; best validation epoch {training['best_epoch']}. Training scans: {len(training['split']['training'])}; validation: {len(training['split']['validation'])}; held-out test: {len(training['split']['test'])}.",
        '',training['policy'],'',
        'Input is the corrected single cell intensity channel, enhanced with membrane-connected-v1 and enlarged 2x in XY. Up to seven sections per annotated complete cell; all 1,482 eligible training cells included. Neither signal channels nor RGB display colors are network inputs.',
        '', 'Network architecture and initial weights are the official 32-ray, depth-3 2D U-Net. All weights fine-tuned at 3e-5. Unknown real pixels have disabled probability loss; distance background regularization is zero. Synthetic blanks/noise provide limited negative supervision.',
        '', 'Adjacent XY polygons are joined by one-to-one Hungarian IoU matching (minimum 0.2). Missing slices break tracks; no gap filling or contour dilation. The 2D models use the preprocessed 3D model’s raw membrane-evidence and physical size acceptance settings; the conservative 3D baseline keeps its original settings. No roundness or signal-retention filtering. The 2D experiment uses the unchanged official probability threshold and NMS 0.3, with no held-out threshold tuning.',
        '', '| Method | All contours | Manual matches / 505 | Volume MAPE | Surface error µm |','|---|---:|---:|---:|---:|']
    for name,m in methods.items():
        s=m['summary'];v=s['volume'].get('mean_absolute_percentage_error')
        volume='—' if v is None else f'{v:.2f}%'
        error=s['mean_symmetric_surface_error_um']
        surface='—' if error is None else f'{error:.3f}'
        lines.append(f"| {name} | {s['predicted_cells']} | {s['matched_complete_cells']} | {volume} | {surface} |")
    lines.extend(['','Volume and surface errors are conditional on whole-cell IoU≥0.5. Different methods can match different cells; comparison.json includes pairwise intersections and the intersection across all four methods. Sparse annotations cannot establish full-field precision. All outliers included.',
        '', 'This is a 2D experiment with a separate 3D linking algorithm, not a pretrained 3D model. It can fragment cells where 2D sections disappear. StarDist polygons still require star-convex shapes per section. The model is trained originally for fluorescent nuclei, while FISH contains membrane-stained cells, so domain shift remains.',
        '', 'The signal-center U-Net is unchanged. Fine-tuned 2D is the current analysis default; default-setting distances are reported separately.',
        '', 'Checkpoint: models/cells-stardist2d/best.weights.h5. Select Fine-tuned StarDist 2D (default) in the analysis app to use it with the existing signal and measurement pipeline.',
        '', '![Comparison](comparison.png)', '', '![Held-out examples](heldout-examples.png)'])
    shared=pairwise['finetuned vs preprocessed']
    if shared['cells']:
        f=shared['methods']['finetuned'];v=shared['methods']['preprocessed']
        lines.extend(['',f"On the {shared['cells']} manual cells matched by both fine-tuned 2D and preprocessed 3D, volume MAPE is {f['volume_mape_percent']:.2f}% versus {v['volume_mape_percent']:.2f}%, and mean symmetric surface error is {f['surface_mean_error_um']:.3f} versus {v['surface_mean_error_um']:.3f} µm."])
    fine=methods['finetuned']['summary'];other=methods['preprocessed']['summary']
    if fine['mean_surface_hd95_um'] is not None:
        lines.extend(['',f"Fine-tuned 2D mean per-cell HD95 is {fine['mean_surface_hd95_um']:.3f} µm versus {other['mean_surface_hd95_um']:.3f} µm for preprocessed 3D. Average errors, larger local errors, and detection recall must be considered separately."])
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v['summary'] for k,v in methods.items()},indent=2))

if __name__=='__main__':main()
