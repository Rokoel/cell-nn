"""Export a complete collection audit and optional trained-run results for review."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fish_viewer.neural.audit import read_references
from fish_viewer.neural.engine import aggregate_subsets,write_outputs
from fish_viewer.pipeline.runner import _json
from fish_viewer.pipeline.segmentation import mask_properties
from fish_viewer.pipeline.evaluation import evaluate,aggregate


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else [])
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--run',type=Path)
    args=parser.parse_args();audit=json.loads(args.audit.read_text());out=args.audit.resolve().parent
    sources={s['image']:s for s in audit['sources']};image_rows=[];cell_rows=[]
    for c in audit['cases']:
        a=c['audit'];source=sources[c['image']]
        image_rows.append(dict(image=c['image'],split=a['split'],group=a['group'],
            aliases=' | '.join(a['aliases']),format=source['format'],channel=a['channel']+1,
            references=a['reference_count'],complete=a['complete_references'],cut=a['cut_references'],
            usable=a['training_objects'],recovered_overlaps=len(a.get('overlapping_objects_retained_in_crops',[])),
            spacing_zyx=str(a['spacing_zyx']),warnings=' | '.join(a['warnings'])))
        refs,names=read_references(Path(c['directory'])/'references.npz')
        for number,((box,mask),name,cut) in enumerate(zip(refs,names,a['reference_cut']),1):
            props=mask_properties(mask,np.asarray(a['spacing_zyx']))
            cell_rows.append(dict(image=c['image'],name=name,reference_id=number,split=a['split'],group=a['group'],
                cut=cut,usable=number in a['training_reference_ids'],**props))
    write_csv(out/'images.csv',image_rows);write_csv(out/'annotations.csv',cell_rows)
    write_csv(out/'source-files.csv',[dict(image=s['image'],status=s['status'],duplicate_of=s.get('duplicate_of',''),
        manual_meshes=s['manual_meshes'],rasterized_references=s['rasterized_references'],
        scenes=' | '.join(s['scenes']),source_warnings=' | '.join(s['source_warnings']),
        reference_warnings=' | '.join(s['reference_warnings'])) for s in audit['sources']])
    folders=defaultdict(list)
    for r in image_rows:folders[str(Path(r['image']).parent.relative_to(Path(audit['root'])))].append(r)
    write_csv(out/'folders.csv',[dict(folder=f,images=len(rows),references=sum(r['references'] for r in rows),
        complete=sum(r['complete'] for r in rows),training=sum(r['usable'] for r in rows if r['split']=='training'),
        validation=sum(r['usable'] for r in rows if r['split']=='validation'),test=sum(r['usable'] for r in rows if r['split']=='test'))
        for f,rows in sorted(folders.items())])
    config=audit['config']
    upper=min(config['max_diameter_um'],config['precision_max_diameter_um']) if config.get('precision_mode') else config['max_diameter_um']
    eligibility=Counter()
    for r in cell_rows:
        if r['cut']:continue
        if not config['min_diameter_um']<=r['diameter_um']<=upper:eligibility['outside_size']+=1
        elif r['elongation']>config['max_elongation']:eligibility['elongated']+=1
        elif r['sphericity']<config['min_sphericity']:eligibility['low_sphericity']+=1
        else:eligibility['passes_shape_filters']+=1
    _json(out/'shape-audit.json',dict(complete_reference_shapes=dict(eligibility),
        note='Training targets follow the audit selection. All retained references remain available for evaluation.'))
    print('Audit tables saved:',out,'shape audit:',dict(eligibility))
    if args.run:
        report=json.loads(args.run.read_text());directory=args.run.resolve().parent
        test=[r for r in report['files'] if r['split']=='test']
        fresh=[r for r in test if '3abx3ad_101212' not in Path(r['image']).parts]
        rows=[]
        for subset,files in [('all_test',test),('test_excluding_prior_pilot',fresh)]:
            if not files:continue
            values=aggregate_subsets([r['metrics'] for r in files])
            for group in ('all','complete_cells','cut_cells'):
                cells=[cell for r in files for cell in (r['metrics'] if group=='all' else r['metrics'][group])['per_cell']]
                scores=values if group=='all' else values[group]
                rows.append(dict(subset=subset,cell_group=group,images=scores['scored_images'],references=len(cells),
                    detected=sum(c['detected'] for c in cells),macro_dice=scores['mean_dice'],macro_iou=scores['mean_iou'],
                    pooled_dice=float(np.mean([c['dice'] for c in cells])) if cells else None,boundary_rmse_um=scores['boundary_rmse_um']))
        # The user targets approximately spherical cells; score that reference
        # subset separately using the same declared physical filters.
        audited={c['image']:c for c in audit['cases']}
        shapes={(c['image'],c['reference_id']):c for c in cell_rows}
        round_scores=[];round_details=[]
        for file in test:
            case=audited[file['image']];a=case['audit'];cfg=report['config']
            upper=min(cfg['max_diameter_um'],cfg['precision_max_diameter_um']) if cfg.get('precision_mode') else cfg['max_diameter_um']
            refs,names=read_references(Path(case['directory'])/'references.npz')
            selected=[]
            for index in range(len(refs)):
                s=shapes[(file['image'],index+1)]
                selected.append(not s['cut'] and cfg['min_diameter_um']<=s['diameter_um']<=upper
                    and s['elongation']<=cfg['max_elongation'] and s['sphericity']>=cfg['min_sphericity'])
            with np.load(file['predictions'],allow_pickle=False) as archive:labels=archive['labels']
            score=evaluate(labels,[r for r,keep in zip(refs,selected) if keep],
                [name for name,keep in zip(names,selected) if keep],a['spacing_zyx'])
            round_scores.append(score);round_details.append(dict(image=file['image'],metrics=score))
        if round_scores:
            score=aggregate(round_scores);cells=[c for s in round_scores for c in s['per_cell']]
            rows.append(dict(subset='all_test',cell_group='round_complete_cells',images=score['scored_images'],
                references=len(cells),detected=sum(c['detected'] for c in cells),macro_dice=score['mean_dice'],macro_iou=score['mean_iou'],
                pooled_dice=float(np.mean([c['dice'] for c in cells])) if cells else None,boundary_rmse_um=score['boundary_rmse_um']))
            _json(directory/'round-cell-test.json',dict(aggregate=score,files=round_details))
        write_csv(directory/'test-summary.csv',rows)
        grouped=defaultdict(list)
        for row in test:grouped[str(Path(row['image']).parent.relative_to(Path(audit['root'])))].append(row['metrics'])
        _json(directory/'test-by-folder.json',{f:aggregate_subsets(ms) for f,ms in grouped.items()})
        _json(directory/'test-summary.json',dict(results=rows,proposal_limit_images=[r['image'] for r in report['files']
            if r['metrics']['proposal_limit_reached']],note='Macro Dice weights images equally. Pooled Dice weights references equally. '
            'Unknown candidates remain unscored; precision is not measurable. Prior-pilot exclusion is disclosed separately.'))
        write_outputs(directory,report)
        print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
