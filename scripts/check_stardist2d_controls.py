"""Check experimental 2D-to-3D contours on fixed synthetic no-cell volumes."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    from fish_viewer.neural.engine import runtime,filter_instances
    _,_,_,info=runtime('cpu')
    from stardist.models import StarDist2D
    from fish_viewer.neural.config import NeuralConfig
    from fish_viewer.neural.controls import control_images
    from fish_viewer.neural.preprocessing import membrane_connected
    from fish_viewer.neural.slices import predict_sections,link_sections
    from fish_viewer.pipeline.segmentation import Prepared
    from fish_viewer.pipeline.runner import _json
    manifest=json.loads((args.model/'fish-model.json').read_text());config=NeuralConfig.from_dict(manifest['acceptance_config'])
    model=StarDist2D(None,name=args.model.name,basedir=str(args.model.parent));model.load_weights(manifest['checkpoint'])
    recipe=manifest['experiment']['inference'];rows=[]
    for name,image in control_images().items():
        prepared=Prepared(membrane_connected(image,(.3,)*3),(.3,)*3,(0.,)*3,0,[],name,{}, {},evidence_image=image)
        raw=link_sections(predict_sections(model,prepared.image,recipe['probability'],recipe['nms']),recipe['linking_iou'])
        prediction=filter_instances(raw,prepared,config)
        rows.append(dict(control=name,accepted=len(prediction.cells),rejected=prediction.rejected))
    result=dict(runtime=info,passed=all(r['accepted']==0 for r in rows),controls=rows,
        note='Fixed synthetic blank/constant/noise/one-sided arc controls. Not a field precision estimate.')
    args.output.parent.mkdir(parents=True,exist_ok=True);_json(args.output,result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
