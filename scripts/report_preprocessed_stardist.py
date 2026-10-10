"""Evaluate the completed membrane experiment and archive a selectable checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark_measurements import paired_stats
from fish_viewer.neural.model_picker import read_model


def write(path,value):
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/preprocessed-stardist')
    parser.add_argument('--bundle-model',action='store_true')
    args=parser.parse_args();args.run=args.run.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    report_path=next((args.run/'training').glob('*/report.json'))
    training=json.loads(report_path.read_text())
    if training['status']!='complete' or training['failures']:raise ValueError('Successful completed training and all predictions required.')
    subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_cell_contours.py'),'--audit',str(args.run/'audit/audit.json'),
        '--report',str(report_path),'--split','test','--output',str(out/'preprocessed')],check=True)
    baseline=json.loads((out/'conservative/report.json').read_text());new=json.loads((out/'preprocessed/report.json').read_text())
    key=lambda c:(c['image'],c['manual_reference_id'])
    left={key(c):c for c in baseline['cells']};right={key(c):c for c in new['cells']};common=sorted(left.keys()&right.keys())
    comparison={}
    for name,rows in [('conservative',[left[k] for k in common]),('preprocessed',[right[k] for k in common])]:
        comparison[name]=dict(volume=paired_stats(rows,'manual_volume_um3','predicted_volume_um3'),
            mean_signed_volume_percent=float(np.mean([100*(r['predicted_volume_um3']/r['manual_volume_um3']-1) for r in rows])) if rows else None,
            mean_symmetric_surface_error_um=float(np.mean([r['surface_mean_error_um'] for r in rows])) if rows else None)
    audit=json.loads((args.run/'audit/audit.json').read_text());counts={s:sum(c['audit']['split']==s for c in audit['cases']) for s in ('training','validation','test')}
    changes={s:sum(c['audit']['split']==s and c['audit']['channel_selection']['previous']!=c['audit']['channel'] for c in audit['cases']) for s in counts}
    manifest=read_model(training['model_directory'])
    result=dict(status='complete',scans=counts,channel_corrections=changes,epochs=len(training['epochs']),best_epoch=training['best_epoch'],
        training_cells=training['patch_counts']['training']-training['patch_counts']['synthetic_background'],
        synthetic_negatives=training['patch_counts']['synthetic_background'],recipe=manifest['preprocessing_recipe'],
        conservative=baseline['summary'],preprocessed=new['summary'],common_matched_cells=len(common),common_cells=comparison,
        selected_acceptance=training['acceptance_selection']['selected_index'],
        negative_controls=training['acceptance_selection']['candidates'][training['acceptance_selection']['selected_index']]['controls'],
        checkpoint_sha256=new['checkpoint_sha256'],training_report=str(report_path),initialization=training.get('initialization'),
        policy='All 71 original held-out acquisition-group scans, all outliers included, IoU>=0.5 complete manual-cell matches, no signal retention or app roundness filter. '
        'Both models use stain-corrected channels; sparse annotations cannot estimate precision. Training channels and preprocessing changed together; this is not an isolated preprocessing ablation.')
    write(out/'comparison.json',result)
    write(out/'training-history.json',dict(epochs=training['epochs'],config=training['config'],patch_counts=training['patch_counts'],stop_reason=training['stop_reason']))
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    epochs=[r['epoch'] for r in training['epochs']]
    for metric,label in [('loss','Training'),('val_loss','Validation')]:axes[0].plot(epochs,[r['losses'][metric] for r in training['epochs']],label=label)
    axes[0].axvline(training['best_epoch'],ls=':',color='black',label='Selected epoch');axes[0].set(xlabel='Epoch',ylabel='Loss',title='Full validation loss');axes[0].legend()
    for rows,label in [(baseline['cells'],'Conservative'),(new['cells'],'Preprocessed')]:
        error=np.array([100*(r['predicted_volume_um3']/r['manual_volume_um3']-1) for r in rows])
        if len(error):axes[1].hist(error,bins=np.linspace(-60,60,31),alpha=.5,label=f'{label} (n={len(error)})')
    axes[1].axvline(0,color='black',ls=':');axes[1].set(xlabel='Signed volume error (%)',ylabel='Matched cells',title='Own matched cells; plot range ±60%');axes[1].legend()
    for values,label in [(left,'Conservative'),(right,'Preprocessed')]:
        errors=np.sort([values[k]['surface_mean_error_um'] for k in common]);axes[2].plot(errors,np.arange(1,len(errors)+1)/max(1,len(errors)),label=label)
    axes[2].set(xlabel='Symmetric surface error (µm)',ylabel='Fraction of common matches',title=f'Same {len(common)} manual cells');axes[2].legend()
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    source_audit=Path(audit.get('parent_preprocessing_audit',args.run/'audit/audit.json'))
    example=source_audit.parent.parent/'channel-preprocessing-examples.png'
    if example.exists():shutil.copy2(example,out/'preprocessing-examples.png')
    test_cases=[c for c in audit['cases'] if c['audit']['split']=='test']
    original_rows={r['image']:r for r in json.loads(Path(baseline['source_report']).read_text())['files']}
    new_rows={r['image']:r for r in training['files']}
    chosen=np.random.default_rng(1010).choice(len(test_cases),size=min(4,len(test_cases)),replace=False)
    fig,axes=plt.subplots(len(chosen),3,figsize=(13,4.5*len(chosen)),squeeze=False)
    from fish_viewer.neural.audit import read_references
    for index,choice in enumerate(chosen):
        case=test_cases[int(choice)];directory=Path(case['directory']);raw=np.load(directory/'evidence.npy')
        refs,_=read_references(directory/'references.npz');coverage=np.zeros(raw.shape[0])
        for box,mask in refs:coverage[box[0]]+=mask.sum(axis=(1,2))
        z=int(coverage.argmax())
        for column,(title,rows,color) in enumerate([('Raw membrane / manual contours',None,None),('Conservative contours',original_rows,'cyan'),('Preprocessed contours',new_rows,'lime')]):
            ax=axes[index,column];ax.imshow(raw[z],cmap='gray',vmin=0,vmax=1)
            for box,mask in refs:
                if box[0].start<=z<box[0].stop:
                    plane=mask[z-box[0].start]
                    if min(plane.shape)>1 and plane.any() and not plane.all():
                        ax.contour(np.arange(box[2].start,box[2].stop),np.arange(box[1].start,box[1].stop),plane,levels=[.5],colors='orange',linewidths=.7)
            if rows:
                with np.load(rows[case['image']]['predictions'],allow_pickle=False) as data:labels=data['labels'][z]
                for cell in np.unique(labels):
                    if cell>0:ax.contour(labels==cell,levels=[.5],colors=color,linewidths=.7)
            ax.set_title(title+'\n'+Path(case['image']).name,fontsize=8);ax.set_axis_off()
    fig.suptitle('Four held-out scans sampled with seed 1010; orange = manual; cyan/green = predictions',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.98));fig.savefig(out/'heldout-examples.png',dpi=150);plt.close(fig)
    lines=['# Membrane-preprocessed StarDist comparison','',result['policy'],'',
        '| Model | Detected cells | Manual matches / 505 | Volume MAPE | Mean surface error |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name,summary in [('Conservative',baseline['summary']),('Preprocessed StarDist',new['summary'])]:
        vol=summary['volume'].get('mean_absolute_percentage_error',float('nan'))
        surface=summary['mean_symmetric_surface_error_um']
        surface_text=f'{surface:.3f} µm' if surface is not None else 'Not measurable'
        lines.append(f"| {name} | {summary['predicted_cells']} | {summary['matched_complete_cells']} | {vol:.2f}% | {surface_text} |")
    lines+=['',f"The common-match comparison contains {len(common)} cells; see comparison.json for paired volume and surface errors.",
        '',f"Training: {result['training_cells']} complete manual cells from 215 scans, {result['synthetic_negatives']} synthetic negatives, {result['epochs']} epochs; best epoch {result['best_epoch']}.",
        '', 'The conservative pipeline remains the app default. Select **Preprocessed StarDist (experimental)** to use the additional checkpoint. Its preprocessing is automatic.',
        '', 'Channel corrections affect 108 training, 19 validation, and 36 test scans. Display colors are not network inputs.',
        '', 'Signal-center weights are unchanged. This evaluation measures cell contours, not signal-distance accuracy.',
        '', '![Training and contour agreement](comparison.png)', '', '![Random held-out contour examples](heldout-examples.png)', '',
        'Training-channel correction and enhancement examples: [preprocessing-examples.png](preprocessing-examples.png).']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if args.bundle_model:
        target=ROOT/'models/cells-preprocessed'
        if target.exists():raise ValueError('Existing bundled checkpoint preserved; choose an explicit replacement process.')
        target.mkdir()
        for source in Path(training['model_directory']).iterdir():
            if source.is_file() and source.name!='last.weights.h5':shutil.copy2(source,target/source.name)
        read_model(target)
        origins=json.loads((ROOT/'models/origins.json').read_text());hashes={}
        for source in target.iterdir():
            with source.open('rb') as stream:hashes[source.name]=hashlib.file_digest(stream,'sha256').hexdigest()
        origins[target.name]=dict(source=str(Path(training['model_directory']).resolve()),sha256=hashes,report='reports/preprocessed-stardist/comparison.json')
        write(ROOT/'models/origins.json',origins)
    write(args.run/'finalized.json',dict(status='complete',report=str(out/'comparison.json'),model=str(ROOT/'models/cells-preprocessed') if args.bundle_model else None))
    print(json.dumps({k:v for k,v in result.items() if k not in ('negative_controls','recipe')},indent=2))


if __name__=='__main__':main()
