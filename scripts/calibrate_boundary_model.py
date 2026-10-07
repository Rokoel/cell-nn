"""Select the mask cutoff from complete validation-cell crops, never test crops."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import sysconfig

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy import ndimage as ndi


def main():
    if sys.platform.startswith('linux') and not os.environ.get('FISH_CUDA_PATH_READY'):
        libraries=sorted((Path(sysconfig.get_paths()['purelib'])/'nvidia').glob('*/lib'))
        if libraries:
            env=os.environ.copy();env['FISH_CUDA_PATH_READY']='1'
            env['LD_LIBRARY_PATH']=os.pathsep.join(map(str,libraries))+os.pathsep+env.get('LD_LIBRARY_PATH','')
            os.execve(sys.executable,[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],env)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--patches',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='gpu',choices=['auto','cpu','gpu'])
    args=parser.parse_args()
    from fish_viewer.neural.engine import runtime
    from fish_viewer.refinement.config import RefinementConfig,read_model
    from fish_viewer.refinement.network import network
    from fish_viewer.refinement.training import batch
    from fish_viewer.refinement.inference import seeded_component
    manifest=read_model(args.model);config=RefinementConfig(**manifest['config'])
    tf,_,_,_=runtime(args.device);model=network(tf,config);model.load_weights(args.model/manifest['checkpoint'])
    records=json.loads((args.patches/'index.json').read_text())['records']
    ids=[r['index'] for r in records if r['split']=='validation']
    if not ids:raise ValueError('No validation targets.')
    arrays={n:np.load(args.patches/(n+'.npy'),mmap_mode='r') for n in ('images','masks','weights','valid')}
    thresholds=[.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7]
    values={str(t):[] for t in thresholds};rng=np.random.default_rng(42)
    for i,index in enumerate(ids):
        if i%25==0:print(f'Validation mask cutoff: {i}/{len(ids)}',flush=True)
        x,y=batch(arrays,records,[index],config,rng)
        logits=model(x,training=False).numpy()[0,...,0];p=1/(1+np.exp(-np.clip(logits,-60,60)))
        truth=y[0,...,0]>0;valid=y[0,...,2]>0;step=records[index]['patch_spacing_um']
        boundary=truth & ~ndi.binary_erosion(truth)
        to_truth=ndi.distance_transform_edt(~boundary,sampling=step)
        seed=np.array(records[index]['seed_patch'])
        for t in thresholds:
            found=seeded_component((p>=t)&valid,seed)
            intersection=np.sum(found & truth);dice=float(2*intersection/max(1,int(found.sum()+truth.sum())))
            bias=float((int(found.sum())-int(truth.sum()))/max(1,int(truth.sum()))*100)
            found_boundary=found & ~ndi.binary_erosion(found)
            if found_boundary.any():
                to_found=ndi.distance_transform_edt(~found_boundary,sampling=step)
                distance=float(np.mean(np.r_[to_truth[found_boundary],to_found[boundary]]))
            else:distance=step*config.patch_size
            values[str(t)].append(dict(dice=dice,volume_bias_percent=bias,absolute_volume_error_percent=abs(bias),
                                        boundary_mean_error_um=distance))
    results={t:{k:float(np.mean([r[k] for r in rows])) for k in rows[0]} for t,rows in values.items()}
    selected=min(results,key=lambda t:(results[t]['boundary_mean_error_um'],results[t]['absolute_volume_error_percent']))
    report=dict(status='complete',validation_cells=len(ids),selected_threshold=float(selected),results=results,
                policy='Minimize mean symmetric boundary error on complete validation target-cell crops; '
                       'volume error breaks ties. Corrected seed/crop setting is assessed separately on real '
                       'detector proposals. Test data not used.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    manifest['config']=dict(manifest['config'],threshold=float(selected));manifest['mask_calibration']=report
    (args.model/'refinement-model.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
