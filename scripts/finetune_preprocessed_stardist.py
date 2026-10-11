"""Fine-tune conservative StarDist on the fixed enhanced cache, preserving held-out groups."""
import argparse
from dataclasses import asdict,replace
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from fish_viewer.neural.config import NeuralConfig
from fish_viewer.neural.precision import evidence_preset
from fish_viewer.neural.model_picker import read_model
from fish_viewer.pipeline.runner import _json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--initial-model',type=Path,default=ROOT/'models/cells-conservative')
    parser.add_argument('--initial-training-report', type=Path, help='Training report with the initial checkpoint acquisition split.')
    parser.add_argument('--epochs',type=int,default=40)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    audit=json.loads(args.audit.read_text());manifest=read_model(args.initial_model)
    if audit['config'].get('preprocessing')!='membrane-connected-v1':raise ValueError('Completed enhanced cache required.')
    source=args.initial_training_report
    if source is None:
        if args.initial_model.resolve() != (ROOT/'models/cells-conservative').resolve():
            parser.error('A custom initial model requires --initial-training-report to verify its acquisition split.')
        source=ROOT/'training_metadata/precision-fish/training/neural-train-20260915-222552-439128/report.json'
    old=json.loads(source.read_text())
    for split in ('training','validation','test'):
        current={c['image'] for c in audit['cases'] if c['audit']['split']==split}
        if current!=set(old['split'][split]):raise ValueError('Initial checkpoint uses a different acquisition split; held-out safety cannot be verified.')
    config=replace(evidence_preset(NeuralConfig.from_dict(manifest['config'])),preprocessing='membrane-connected-v1',
        cell_channel=None,learning_rate=.00003,batch_size=4,epochs=args.epochs,steps_per_epoch=50,
        patience=12,validate_every=5,validation_images=8,streaming_validation=True,
        patch_z=80,patch_xy=80,select_balanced_acceptance=True,probability_threshold=.65,membrane_support=.6,
        synthetic_background_fraction=.25,device='gpu').validate()
    audit['config']=asdict(config);audit['parent_preprocessing_audit']=str(args.audit.resolve())
    path=args.output/'audit/audit.json';path.parent.mkdir();_json(path,audit)
    _json(args.output/'initialization-audit.json',dict(model=str(args.initial_model.resolve()),original_training_report=str(source.resolve()),
        exact_split_match=True,policy='Same training/validation/test input sets as the pretrained checkpoint. No held-out scan contributes gradient updates.'))
    request=args.output/'request.json';_json(request,dict(mode='train',config=asdict(config),
        entries=[dict(image=audit['cases'][0]['image'],scene=audit['cases'][0]['scene'])],
        output=str((args.output/'training').resolve()),audit=str(path.resolve()),
        initial_model=str(args.initial_model.resolve()),cancel_file=str((args.output/'cancel').resolve())))
    os.execv(sys.executable,[sys.executable,'-m','fish_viewer.neural','--request',str(request)])


if __name__=='__main__':main()
