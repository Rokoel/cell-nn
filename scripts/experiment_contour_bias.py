"""Production-box contour calibration, preserving frozen cell identities."""
import argparse
from dataclasses import replace, asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import sysconfig
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy import ndimage as ndi
from fish_viewer.refinement.config import RefinementConfig, read_model
from fish_viewer.refinement.data import coordinates, seed_image
from fish_viewer.refinement.inference import seeded_component, surface_evidence
from fish_viewer.neural.audit import read_references
from fish_viewer.pipeline.runner import _json
from fish_viewer.pipeline.segmentation import mask_properties


def crop_prediction(image, proposal, spacing, origin, config, model):
    spacing, origin = np.asarray(spacing), np.asarray(origin)
    low = (np.asarray(proposal['box_low_xyz']) - origin)[::-1] / spacing
    high = (np.asarray(proposal['box_high_xyz']) - origin)[::-1] / spacing
    center = (low + high) / 2
    seed = (np.asarray(proposal['center_xyz']) - origin)[::-1] / spacing
    width = float(np.max((high - low) * spacing) + 2 * config.margin_um)
    step = width / config.patch_size
    coords = coordinates(center, width, spacing, config.patch_size)
    sample = ndi.map_coordinates(image, coords, order=1, mode='constant', cval=0, prefilter=False)
    seed_patch = (seed-center)*spacing/step + (config.patch_size-1)/2
    x = np.stack([sample, seed_image(seed_patch, step, config)], axis=-1)[None]
    logits = np.asarray(model(x, training=False))[0,...,0]
    if not np.isfinite(logits).all():
        raise ValueError('Non-finite contour prediction')
    probability = 1 / (1 + np.exp(-np.clip(logits, -60, 60)))
    lower = np.maximum(0, np.floor(center-width/2/spacing).astype(int))
    upper = np.minimum(image.shape, np.ceil(center+width/2/spacing).astype(int)+1)
    grid = np.indices(tuple(upper-lower), dtype=np.float32)+lower[:,None,None,None]
    at = (grid-center[:,None,None,None])*spacing[:,None,None,None]/step+(config.patch_size-1)/2
    scores = ndi.map_coordinates(probability, at, order=1, mode='constant', cval=0, prefilter=False)
    return scores, lower, upper, seed-lower, dict(width_um=width, crop_spacing_um=step,
        seed_patch=seed_patch.tolist(), center_zyx=center.tolist())


def guarded_mask(scores, threshold, seed, old, neighbors, visible, spacing, evidence_cutoff=.65):
    mask = seeded_component(scores >= threshold, seed)
    reason = None
    if not mask.any(): reason = 'Empty seeded component'
    elif mask.sum()*np.prod(spacing) < 5: reason = 'Too small'
    elif any(mask.take(i,axis=a).any() for a in range(3) for i in (0,-1)): reason = 'Touches crop/scan boundary'
    elif np.mean(neighbors[mask]) > .4: reason = 'Conflicting neighbor'
    if reason is None:
        mask &= ~neighbors
        mask = seeded_component(mask, seed)
        if not mask.any(): reason = 'Seed occupied by neighbor'
        elif mask.sum()*np.prod(spacing)<5:reason='Too small after neighbor protection'
        else:
            evidence = surface_evidence(mask, visible, spacing)
            if evidence['surface_support'] < evidence_cutoff or evidence['surface_contrast'] < .04:
                reason = 'Insufficient membrane evidence'
    return (old.copy() if reason else mask), reason


def mask_metrics(found, truth, spacing):
    n, t = int(found.sum()), int(truth.sum())
    intersection = int(np.sum(found & truth))
    a = found & ~ndi.binary_erosion(found)
    b = truth & ~ndi.binary_erosion(truth)
    if not a.any() or not b.any():
        assd = float(np.linalg.norm(np.array(truth.shape)*spacing))
    else:
        da = ndi.distance_transform_edt(~a, sampling=spacing)
        db = ndi.distance_transform_edt(~b, sampling=spacing)
        assd = float(np.mean(np.r_[db[a], da[b]]))
    bias = 100*(n-t)/max(t,1)
    return dict(iou=intersection/max(n+t-intersection,1), volume_bias_percent=bias,
                volume_mape=abs(bias), boundary_error_um=assd)


def embedded_metrics(mask, lower, upper, box, truth_local, spacing):
    a = np.minimum(lower, [s.start for s in box]); b = np.maximum(upper, [s.stop for s in box])
    found = np.zeros(tuple(b-a),bool); truth = np.zeros_like(found)
    found[tuple(slice(int(x),int(y)) for x,y in zip(lower-a,upper-a))] = mask
    truth[tuple(slice(s.start-int(k),s.stop-int(k)) for s,k in zip(box,a))] = truth_local
    return mask_metrics(found, truth, spacing)


def main():
    if sys.platform.startswith('linux') and not os.environ.get('FISH_CUDA_PATH_READY'):
        libs=sorted((Path(sysconfig.get_paths()['purelib'])/'nvidia').glob('*/lib'))
        if libs:
            env=os.environ.copy();env['FISH_CUDA_PATH_READY']='1'
            env['LD_LIBRARY_PATH']=os.pathsep.join(map(str,libs))+os.pathsep+env.get('LD_LIBRARY_PATH','')
            os.execve(sys.executable,[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],env)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--model',type=Path,default=ROOT/'models/refinement');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['threshold','geometry','score','apply'],required=True)
    p.add_argument('--threshold',type=float,default=.5);p.add_argument('--margin',type=float,default=2)
    p.add_argument('--size',type=int,default=64);p.add_argument('--device',default='gpu')
    args=p.parse_args()
    if args.output.exists():p.error('Output already exists; preserve earlier results')
    args.output.mkdir(parents=True)
    audit=json.loads(args.audit.read_text());baseline=json.loads(args.baseline.read_text())
    cases={c['image']:c for c in audit['cases']}
    if baseline['status']!='complete':raise ValueError('Complete baseline required')
    if args.mode!='apply' and any(cases[r['image']]['audit']['split']!='validation' for r in baseline['files']):
        raise ValueError('Calibration permits validation only')
    manifest=read_model(args.model);config=RefinementConfig(**manifest['config'])
    if args.mode=='threshold':variants=[(f'threshold-{t}',2,64,t) for t in (.3,.35,.4,.45,.5,.55)]
    elif args.mode=='geometry':variants=[(f'margin-{m}-size-{n}',m,n,args.threshold) for m,n in ((1,64),(2,64),(3,64),(2,96),(3,96))]
    else:variants=[('apply',args.margin,args.size,args.threshold)]
    from fish_viewer.neural.engine import runtime
    from fish_viewer.refinement.network import network
    tf,_,_,info=runtime(args.device)
    models={}
    for size in sorted({v[2] for v in variants}):
        net=network(tf,replace(config,patch_size=size));net.load_weights(args.model/manifest['checkpoint']);models[size]=net
    report=dict(status='running',mode=args.mode,baseline_report=str(args.baseline.resolve()),
        model=str(args.model.resolve()),checkpoint_sha256=hashlib.sha256((args.model/manifest['checkpoint']).read_bytes()).hexdigest(),
        runtime=info,rows=[],files=[],variants=variants,policy='Frozen baseline accepted cells and IDs; no new detections. Neighbor ownership fixed. Unsafe changes retain original mask. Calibration validation only.')
    start=time.monotonic()
    for number,row in enumerate(baseline['files'],1):
        print(f'{args.mode} scan {number}/{len(baseline["files"])}: {Path(row["image"]).name}',flush=True)
        case=cases[row['image']];image=np.load(Path(case['directory'])/'image.npy',mmap_mode='r')
        with np.load(row['predictions']) as a:
            old_labels=a['labels'];metadata=json.loads(str(a['metadata']))
        spacing=np.asarray(metadata['spacing_zyx']);origin=metadata['origin_xyz']
        visible=ndi.gaussian_filter(image,sigma=.3/spacing)
        refs,names=read_references(Path(case['directory'])/'references.npz')
        matches={m['predicted_label']:(i,refs[i]) for i,m in enumerate(row['metrics']['per_cell'])
                 if m['predicted_label'] is not None and m['iou']>=.5 and not case['audit']['reference_cut'][i]}
        selected=[d for d in row['proposals'] if d['accepted'] and (args.mode=='apply' or d['cell_id'] in matches)]
        new_labels=old_labels.copy();diagnostics=[]
        for index,proposal in enumerate(selected,1):
            cell=proposal['cell_id'];cache={}
            old_box=ndi.find_objects((old_labels==cell).astype(np.uint8),max_label=1)[0]
            for label,margin,size,threshold in variants:
                cfg=replace(config,margin_um=margin,patch_size=size,threshold=threshold)
                if (margin,size) not in cache:
                    cache[margin,size]=crop_prediction(image,proposal,spacing,origin,cfg,models[size])
                scores,lower,upper,seed,geometry=cache[margin,size]
                region=tuple(slice(int(a),int(b)) for a,b in zip(lower,upper))
                old=old_labels[region]==cell;neighbors=(old_labels[region]!=0)&~old
                if args.mode=='apply':
                    neighbors |= (new_labels[region]!=0)&(new_labels[region]!=cell)
                mask,reason=guarded_mask(scores,threshold,seed,old,neighbors,visible[region],spacing)
                geometry['original_cell_crop_coverage']=float(old.sum()/np.sum(old_labels[old_box]==cell))
                diagnostic=dict(variant=label,image=row['image'],group=case['audit']['group'],cell_id=cell,
                    accepted=reason is None,fallback_reason=reason,**geometry)
                if cell in matches:
                    ref,(box,truth)=matches[cell]
                    old_low=np.array([s.start for s in old_box]);old_high=np.array([s.stop for s in old_box])
                    baseline_metrics=embedded_metrics(old_labels[old_box]==cell,old_low,old_high,box,truth,spacing)
                    metrics=embedded_metrics(mask,lower,upper,box,truth,spacing)
                    if geometry['original_cell_crop_coverage']<1:
                        reason='Original cell clipped by crop'
                        diagnostic.update(accepted=False,fallback_reason=reason)
                        metrics=baseline_metrics.copy()
                    diagnostic.update(reference_id=ref+1,baseline=baseline_metrics,metrics=metrics)
                    lo=np.array([s.start for s in box]);hi=np.array([s.stop for s in box])-1
                    diagnostic['manual_max_extent_um']=float(np.max((hi-lo+1)*spacing))
                    diagnostic['box_max_extent_um']=geometry['width_um']-2*margin
                    diagnostic['manual_bbox_crop_coverage']=bool(np.all(lo>=lower)&np.all(hi<upper))
                    diagnostic['manual_crop_edge_margin_um']=float(np.min(np.r_[(lo-lower)*spacing,(upper-1-hi)*spacing]))
                    report['rows'].append(diagnostic)
                if args.mode=='apply':
                    # If original cell extends outside this crop, reject the edit;
                    # the entire original mask must remain visible and preserved.
                    if geometry['original_cell_crop_coverage']<1:
                        mask=old;reason='Original cell clipped by crop';diagnostic.update(accepted=False,fallback_reason=reason)
                    if reason is None:
                        new_labels[new_labels==cell]=0;new_labels[region][mask]=cell
                    diagnostics.append(diagnostic)
            if index%25==0:print(f'  cells {index}/{len(selected)}',flush=True)
        if args.mode=='apply':
            folder=args.output/f'{number:04d}-{Path(row["image"]).stem}';folder.mkdir()
            cells=[]
            for cell,box in enumerate(ndi.find_objects(new_labels),1):
                if box is None:continue
                local=new_labels[box]==cell;properties=mask_properties(local,spacing)
                center=(np.argwhere(local).mean(0)+[s.start for s in box])*spacing
                previous=next((c for c in metadata.get('cells',[]) if c.get('label')==cell),{})
                cells.append(dict(previous,**properties,label=cell,center_xyz=(center[::-1]+np.asarray(origin)).tolist()))
            metadata['cells']=cells
            metadata['refinement']=dict(metadata.get('refinement',{}),model=str(args.model.resolve()),
                config=asdict(replace(config,margin_um=args.margin,patch_size=args.size,threshold=args.threshold)),cells=diagnostics,
                checkpoint_sha256=report['checkpoint_sha256'])
            path=folder/'contours.npz';np.savez_compressed(path,labels=new_labels,metadata=np.array(json.dumps(metadata,allow_nan=False)))
            new_row=dict(row,predictions=str(path.resolve()),refinement=diagnostics)
            report['files'].append(new_row)
        report['progress']=dict(scan=number,scans=len(baseline['files']),elapsed_s=time.monotonic()-start)
        _json(args.output/'report.json',report)
    summary={}
    for label,_,_,_ in variants:
        rows=[r for r in report['rows'] if r['variant']==label]
        summary[label]=dict(cells=len(rows),accepted=sum(r['accepted'] for r in rows),
            all_manual_bbox_inside_crop=sum(r['manual_bbox_crop_coverage'] for r in rows),
            cells_still_matched=sum(r['metrics']['iou']>=.5 for r in rows),
            **{k:float(np.mean([r['metrics'][k] for r in rows])) for k in ('iou','volume_bias_percent','volume_mape','boundary_error_um')},
            baseline={k:float(np.mean([r['baseline'][k] for r in rows])) for k in ('iou','volume_bias_percent','volume_mape','boundary_error_um')})
    report.update(status='complete',summary=summary,elapsed_s=time.monotonic()-start)
    _json(args.output/'report.json',report);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
