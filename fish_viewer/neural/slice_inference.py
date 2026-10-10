"""Analysis worker for the fine-tuned single-channel XY StarDist model."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import sysconfig
import traceback

from .config import NeuralConfig
from .preprocessing import RECIPE


def read_model(directory):
    directory=Path(directory).expanduser().resolve()
    manifest=json.loads((directory/'fish-model.json').read_text())
    if manifest.get('version')!=1 or manifest.get('engine')!='stardist2d-slice':
        raise ValueError('Choose a fine-tuned 2D StarDist model.')
    NeuralConfig.from_dict(manifest['acceptance_config'])
    if manifest.get('preprocessing_recipe')!=RECIPE:
        raise ValueError('Saved 2D preprocessing recipe does not match this application.')
    recipe=manifest['experiment']['inference']
    if manifest['experiment']['input']['xy_scale']!=2 or not all(0<float(recipe[k])<=1 for k in ('probability','nms','linking_iou')):
        raise ValueError('Unsupported 2D inference recipe.')
    config=json.loads((directory/'config.json').read_text())
    if config.get('n_dim')!=2 or config.get('n_channel_in')!=1:
        raise ValueError('Expected a single-channel 2D checkpoint.')
    if not (directory/manifest['checkpoint']).is_file():raise ValueError('Missing 2D checkpoint.')
    return manifest


def apply(image,model_directory,output,device='auto',cell_channel=None,cancel=None,emit=lambda e:None):
    from ..run_paths import new_run_directory
    from ..pipeline.runner import _json,save_prediction
    from ..pipeline.segmentation import Cancelled,checkpoint
    directory=new_run_directory(output,'stardist2d')
    report=dict(version=1,engine='stardist2d-slice',status='running',directory=str(directory),files=[],failures=[],model_directory=str(Path(model_directory).resolve()))
    try:
        checkpoint(cancel)
        manifest=read_model(model_directory)
        config=replace(NeuralConfig.from_dict(manifest['acceptance_config']),device=device,cell_channel=cell_channel).validate()
        from .engine import runtime,filter_instances
        runtime(device)
        from stardist.models import StarDist2D
        from .data import prepare_neural
        from .slices import predict_sections,link_sections
        from ..reader import entry_for_path,load_dataset
        emit(dict(kind='progress',phase='Loading cell channel',current=0,total=1))
        entry=entry_for_path(image);dataset=load_dataset(entry)
        checkpoint(cancel)
        prepared=prepare_neural(dataset,config,emit,cancel)
        model_directory=Path(model_directory).resolve()
        model=StarDist2D(None,name=model_directory.name,basedir=str(model_directory.parent))
        model.load_weights(manifest['checkpoint'])
        recipe=manifest['experiment']['inference']
        sections=predict_sections(model,prepared.image,recipe['probability'],recipe['nms'],cancel=cancel,
            progress=lambda current,total:emit(dict(kind='progress',phase='Drawing XY cell contours',current=current,total=total)))
        checkpoint(cancel)
        emit(dict(kind='progress',phase='Joining contours into 3D cells',current=0,total=1))
        labels=link_sections(sections,recipe['linking_iou'])
        checkpoint(cancel)
        prediction=filter_instances(labels,prepared,config)
        path=directory/'contours.npz';save_prediction(path,prediction,prepared,entry,config,{})
        report['files']=[dict(image=str(entry.image),predictions=str(path),cells=len(prediction.cells),channel=prepared.channel)]
        report['status']='complete'
    except Cancelled:report['status']='cancelled'
    except Exception:report.update(status='failed',error=traceback.format_exc())
    _json(directory/'report.json',report)
    return report


def main():
    if sys.platform.startswith('linux') and not os.environ.get('FISH_CUDA_PATH_READY'):
        libraries=sorted((Path(sysconfig.get_paths()['purelib'])/'nvidia').glob('*/lib'))
        if libraries:
            environment=dict(os.environ,FISH_CUDA_PATH_READY='1',LD_LIBRARY_PATH=os.pathsep.join(map(str,libraries))+os.pathsep+os.environ.get('LD_LIBRARY_PATH',''))
            os.execve(sys.executable,[sys.executable,'-m','fish_viewer.neural.slice_inference',*sys.argv[1:]],environment)
    from .__main__ import Cancellation
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['auto','cpu','gpu'],default='auto')
    p.add_argument('--cell-channel',type=int);p.add_argument('--cancel-file',type=Path)
    args=p.parse_args()
    def notify(event):
        message=dict(kind='progress',phase=event.get('phase',event.get('stage','Drawing cell contours')),
            current=event.get('current',event.get('progress',0)),total=event.get('total',100))
        print(json.dumps(message),flush=True)
    report=apply(args.image,args.model,args.output,args.device,args.cell_channel,Cancellation(args.cancel_file),notify)
    print(json.dumps(dict(kind='finished',report=report)),flush=True)
    return 0 if report['status']=='complete' else 1

if __name__=='__main__':raise SystemExit(main())
