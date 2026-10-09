"""Check frozen models on known synthetic empty fields and incomplete membranes."""
import argparse
import json
import os
from pathlib import Path
import sys
import sysconfig

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    if sys.platform.startswith('linux') and not os.environ.get('FISH_CUDA_PATH_READY'):
        libraries=sorted((Path(sysconfig.get_paths()['purelib'])/'nvidia').glob('*/lib'))
        if libraries:
            env=os.environ.copy();env['FISH_CUDA_PATH_READY']='1'
            env['LD_LIBRARY_PATH']=os.pathsep.join(map(str,libraries))+os.pathsep+env.get('LD_LIBRARY_PATH','')
            os.execve(sys.executable,[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],env)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=['cpu','gpu','auto'],default='cpu')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--boundary-model',type=Path,default=ROOT/'models/refinement');args=p.parse_args()
    import numpy as np
    from scipy import ndimage as ndi
    from fish_viewer.neural.engine import runtime
    from fish_viewer.proposals.config import ProposalConfig,read_model
    from fish_viewer.refinement.config import RefinementConfig,read_model as read_boundary_model
    from fish_viewer.proposals.network import network
    from fish_viewer.refinement.network import network as boundary_network
    from fish_viewer.proposals.inference import predict_proposals,contour_proposals
    from fish_viewer.pipeline.segmentation import Prepared
    manifest=read_model(ROOT/'models/proposals');config=ProposalConfig(**manifest['config'])
    bm=read_boundary_model(args.boundary_model);bc=RefinementConfig(**bm['config'])
    tf,_,_,info=runtime(args.device);detector=network(tf,config);detector.load_weights(ROOT/'models/proposals'/manifest['checkpoint'])
    boundary=boundary_network(tf,bc);boundary.load_weights(args.boundary_model/bm['checkpoint'])
    shape=(64,)*3;grid=np.indices(shape);rng=np.random.default_rng(78)
    radius=np.linalg.norm((grid-31.5)*.3,axis=0)
    controls={'zero':np.zeros(shape),'constant':np.full(shape,.25),
        'linear_gradient':.02+grid[2]/63*.3,
        'white_noise':np.clip(rng.normal(.08,.025,shape),0,1),
        'correlated_noise':np.clip(ndi.gaussian_filter(rng.normal(0,1,shape),.8)*.2+.1,0,1),
        'bright_plane':np.exp(-((grid[2]-31.5)/1.5)**2),
        'incomplete_membrane_arc':np.exp(-((radius-3.5)/.2)**2)*(grid[2]>38)}
    rows=[]
    for name,image in controls.items():
        prepared=Prepared(image.astype(np.float32),(.3,)*3,(.15,)*3,0,[],'synthetic',{}, {})
        proposals=predict_proposals(prepared,detector,config)
        labels,diagnostics=contour_proposals(prepared,proposals,boundary,bc,
            evidence_threshold=manifest.get('crop_surface_evidence_threshold',.65))
        forced=[dict(proposal_id=1,score=1.,center_xyz=[9.45]*3,box_low_xyz=[6.]*3,box_high_xyz=[13.]*3)]
        _,forced_diagnostics=contour_proposals(prepared,forced,boundary,bc,
            evidence_threshold=manifest.get('crop_surface_evidence_threshold',.65))
        row=dict(control=name,proposals=len(proposals),accepted_cells=sum(d['accepted'] for d in diagnostics),
                 forced_proposal_accepted_cells=sum(d['accepted'] for d in forced_diagnostics))
        rows.append(row);print(json.dumps(row),flush=True)
    report=dict(status='complete',passed=all(r['accepted_cells']==0 and r['forced_proposal_accepted_cells']==0 for r in rows),runtime=info,controls=rows,
        policy='Known constructed empty/non-cell controls. Not FISH training negatives or an estimate of real-field precision.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
