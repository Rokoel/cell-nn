"""Apply the experimental fine-tuned XY model to a new Imaris volume."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fish_viewer.worker_runtime import restart_with_cuda


def main():
    restart_with_cuda()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('scan',type=Path);p.add_argument('--model',type=Path,default=ROOT/'models/cells-stardist2d')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cell-channel',type=int,help='Zero-based channel index; otherwise infer from stain metadata and ring evidence')
    p.add_argument('--device',choices=['auto','gpu','cpu'],default='auto')
    args=p.parse_args()
    from fish_viewer.neural.engine import runtime,filter_instances
    runtime(args.device)
    from stardist.models import StarDist2D
    from fish_viewer.neural.config import NeuralConfig
    from fish_viewer.neural.data import prepare_neural
    from fish_viewer.neural.slices import predict_sections,link_sections
    from fish_viewer.reader import entry_for_path,load_dataset
    from fish_viewer.pipeline.runner import _json,save_prediction
    manifest=json.loads((args.model/'fish-model.json').read_text())
    if manifest['engine']!='stardist2d-slice':raise ValueError('Requires a 2D slice model')
    from fish_viewer.neural.preprocessing import RECIPE
    if manifest.get('preprocessing_recipe')!=RECIPE:raise ValueError('Checkpoint preprocessing recipe mismatch')
    config=replace(NeuralConfig.from_dict(manifest['acceptance_config']),cell_channel=args.cell_channel).validate()
    args.output.mkdir(parents=True,exist_ok=False)
    entry=entry_for_path(args.scan);dataset=load_dataset(entry)
    prepared=prepare_neural(dataset,config)
    model=StarDist2D(None,name=args.model.name,basedir=str(args.model.parent));model.load_weights(manifest['checkpoint'])
    recipe=manifest['experiment']['inference']
    raw=link_sections(predict_sections(model,prepared.image,recipe['probability'],recipe['nms']),recipe['linking_iou'])
    prediction=filter_instances(raw,prepared,config)
    save_prediction(args.output/'contours.npz',prediction,prepared,entry,config,{})
    _json(args.output/'report.json',dict(status='complete',engine='stardist2d-slice',image=str(args.scan.resolve()),
        model=str(args.model.resolve()),channel=prepared.channel,cells=len(prediction.cells),predictions=str((args.output/'contours.npz').resolve())))
    print(f'Saved {len(prediction.cells)} contours using cell channel {prepared.channel}: {args.output}')

if __name__=='__main__':main()
