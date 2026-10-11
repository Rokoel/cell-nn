"""Keep every manual target; use frozen predicted boxes where matched."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy import ndimage as ndi
from fish_viewer.refinement.config import RefinementConfig,read_model
from fish_viewer.refinement.data import coordinates,sample_target
from fish_viewer.neural.audit import read_references
from fish_viewer.pipeline.runner import _json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--audit', type=Path, default=ROOT/'artifacts/refinement-fish/audit/audit.json')
    p.add_argument('--patch-index', type=Path, default=ROOT/'training_metadata/refinement-fish/patches/index.json')
    p.add_argument('--baseline-reports', type=Path, nargs='+', default=[
        ROOT/'training_metadata/signal-refresh/training-contours/center-contours-20261007-070644-473469-tdz4ceh5/report.json',
        ROOT/'training_metadata/proposal-fish/validation-contours-final/center-contours-20261007-055507-430727-vlcnm8p3/report.json'])
    args=p.parse_args()
    if args.output.exists():p.error('Preserve existing caches; choose a new output')
    args.output.mkdir(parents=True)
    config=RefinementConfig(**read_model(ROOT/'models/refinement')['config'])
    collection=json.loads(args.audit.read_text())
    cases={c['image']:c for c in collection['cases']}
    original=json.loads(args.patch_index.read_text())
    baseline_files=[]
    for path in args.baseline_reports:
        baseline_files+=json.loads(path.read_text())['files']
    rows={r['image']:r for r in baseline_files}
    shape=(len(original['records']),)+(config.patch_size,)*3
    arrays={k:np.lib.format.open_memmap(args.output/(k+'.npy'),mode='w+',shape=shape,dtype=d)
            for k,d in [('images',np.float16),('masks',np.uint8),('weights',np.float16),('valid',np.uint8)]}
    records=[];counts={s:dict(predicted_box_targets=0,manual_box_targets=0) for s in ('training','validation')}
    loaded={}
    for number,record in enumerate(original['records']):
        if number%25==0:print(f'Production target crops {number}/{shape[0]}',flush=True)
        case=cases[record['image']];row=rows[record['image']];ref=record['reference_id']-1
        if record['image'] not in loaded:
            loaded.clear()
            loaded[record['image']]=(np.load(Path(case['directory'])/'image.npy',mmap_mode='r'),
                read_references(Path(case['directory'])/'references.npz')[0])
        image,references=loaded[record['image']];box,mask=references[ref]
        match=row['metrics']['per_cell'][ref];proposal=None
        if match['iou']>=.5 and match['predicted_label'] is not None:
            proposal=next(d for d in row['proposals'] if d['cell_id']==match['predicted_label'])
        updated=dict(record,validation_checkpoint_score=bool(proposal) if record['split']=='validation' else True)
        spacing=np.array(case['audit']['spacing_zyx']);origin=np.array(case['audit']['origin_xyz'])
        if proposal is None:
            sample,truth,weight,valid,seed,step=sample_target(image,box,mask,spacing,config)
            counts[record['split']]['manual_box_targets']+=1
            updated['crop_source']='manual box; no matched detector proposal'
        else:
            low=(np.array(proposal['box_low_xyz'])-origin)[::-1]/spacing
            high=(np.array(proposal['box_high_xyz'])-origin)[::-1]/spacing
            center=(low+high)/2;width=float(np.max((high-low)*spacing)+2*config.margin_um)
            coords=coordinates(center,width,spacing,config.patch_size)
            sample=ndi.map_coordinates(image,coords,order=1,mode='constant',cval=0,prefilter=False)
            truth=ndi.map_coordinates(mask.astype(np.uint8),[v-s.start for v,s in zip(coords,box)],order=0,mode='constant',cval=0)>0
            valid=np.logical_and.reduce([(v>=0)&(v<=n-1) for v,n in zip(coords,image.shape)])
            step=width/config.patch_size
            inside=ndi.distance_transform_edt(truth,sampling=step);outside=ndi.distance_transform_edt(~truth,sampling=step)
            weight=(1+(config.boundary_weight-1)*np.exp(-((inside+outside)/config.boundary_band_um)**2))*valid
            seed=(np.array(proposal['center_xyz'])-origin)[::-1]/spacing
            seed=(seed-center)*spacing/step+(config.patch_size-1)/2
            counts[record['split']]['predicted_box_targets']+=1
            updated.update(crop_source='frozen predicted box',proposal_id=proposal['proposal_id'])
        updated.update(seed_patch=seed.tolist(),patch_spacing_um=step)
        for k,v in [('images',sample),('masks',truth),('weights',weight),('valid',valid)]:arrays[k][number]=v
        records.append(updated)
    for a in arrays.values():a.flush()
    result=dict(identity=dict(config=asdict(config),cases_sha256=hashlib.sha256(json.dumps(collection['cases'],sort_keys=True).encode()).hexdigest()),
        records=records,summary=counts,policy='Every complete manual training cell retained. Frozen matched predicted boxes supply crop geometry and seeds; unmatched marked cells use manual boxes. Other cells are not members of this target, not global negative examples. Validation checkpoint boundary score uses only the original 441 matched validation identities; all 574 validation targets still evaluated for loss. Test targets never cached.')
    _json(args.output/'index.json',result);print(json.dumps(counts,indent=2),flush=True)
if __name__=='__main__':main()
