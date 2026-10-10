"""Build a separate full-split preprocessing cache and train a fresh StarDist."""
from dataclasses import asdict,replace
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from fish_viewer.neural.config import NeuralConfig
from fish_viewer.neural.audit import derive_precision_audit
from fish_viewer.neural.preprocessing import membrane_connected,RECIPE
from fish_viewer.pipeline.runner import _json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=ROOT/'artifacts/evidence-fish/audit/audit.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=120)
    p.add_argument('--steps',type=int,default=50);p.add_argument('--device',choices=['auto','cpu','gpu'],default='gpu')
    p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    if args.output.exists():raise ValueError('Choose a new run directory; existing audits and models are preserved.')
    args.output.mkdir(parents=True)
    original=json.loads(args.source.read_text())
    if original['config'].get('preprocessing','none')!='none':raise ValueError('A raw source audit is required; preprocessing must not be applied twice.')
    config=replace(NeuralConfig.from_dict(original['config']),preprocessing='membrane-connected-v1',cell_channel=None,
        epochs=args.epochs,steps_per_epoch=args.steps,batch_size=4,rays=128,patience=20,validate_every=10,validation_images=8,
        streaming_validation=True,patch_z=80,patch_xy=80,probability_threshold=.65,nms_threshold=.25,
        curate_training_shapes=False,use_shape_filter=False,select_balanced_acceptance=True,
        calibrate_probability=False,device=args.device).validate()
    audit=args.output/'audit/audit.json'
    derived=derive_precision_audit(args.source,audit,config)
    # Linked labels/references and raw evidence remain immutable; only image.npy changes.
    started=time.monotonic();records=[]
    for i,case in enumerate(derived['cases']):
        source=Path(case['directory']);destination=audit.parent/f'{i+1:04d}-{Path(case["image"]).stem}';destination.mkdir()
        from fish_viewer.channel_hints import membrane_stain_channels
        from fish_viewer.reader import ImarisReader,load_dataset
        from fish_viewer.models import LibraryEntry
        from fish_viewer.pipeline.segmentation import normalize
        from skimage.transform import resize
        hints=membrane_stain_channels(case['image'],ImarisReader(case['image']).metadata.channels)
        previous_channel=case['audit']['channel'];channel=previous_channel
        if hints and channel not in hints:
            scores={r['channel']:r['score'] for r in case['audit']['channel_scores']}
            channel=max(hints,key=lambda c:scores[c])
        for name,target in [('labels.npy','labels.npy'),('references.npz','references.npz')]:
            os.link(source/name,destination/target)
        image=np.load(source/'image.npy',mmap_mode='r')
        if channel==previous_channel:
            os.link(source/'image.npy',destination/'evidence.npy')
        else:
            dataset=load_dataset(LibraryEntry(Path(case['image'])))
            image=resize(normalize(dataset.volumes[channel]),image.shape,order=1,preserve_range=True,anti_aliasing=True).astype(np.float32)
            np.save(destination/'evidence.npy',image)
            del dataset
        case['audit']['channel']=channel
        case['audit']['channel_selection']=dict(previous=previous_channel,selected=channel,stain_candidates=hints,
            policy='Declared Lam/LBR stain wavelengths restrict candidates; ring evidence ranks within them. No manual masks or signal annotations used.')
        enhanced=membrane_connected(image,case['audit']['spacing_zyx'])
        np.save(destination/'image.npy',enhanced)
        digest=hashlib.sha256(enhanced.tobytes()).hexdigest()
        case['directory']=str(destination.resolve());case['audit']['preprocessing']=RECIPE
        case['audit']['network_fingerprint']=digest
        _json(destination/'labels.json',case['audit'])
        records.append(dict(image=case['image'],split=case['audit']['split'],channel=channel,previous_channel=previous_channel,sha256=digest))
        _json(args.output/'preprocessing-progress.json',dict(status='running',current=i+1,total=len(derived['cases']),elapsed_s=time.monotonic()-started))
        if (i+1)%50==0 or i+1==len(derived['cases']):print(f'Preprocessed {i+1}/{len(derived["cases"])} scans',flush=True)
    derived['preprocessing']=RECIPE;_json(audit,derived)
    _json(args.output/'preprocessing-progress.json',dict(status='complete',current=len(records),total=len(records),elapsed_s=time.monotonic()-started,recipe=RECIPE,records=records))
    settings=args.output/'settings.json';_json(settings,dict(engine='stardist3d',config=asdict(config)))
    if args.prepare_only:return 0
    request=args.output/'request.json';_json(request,dict(mode='train',config=asdict(config),entries=[dict(image=derived['cases'][0]['image'],scene=derived['cases'][0]['scene'])],
        output=str((args.output/'training').resolve()),audit=str(audit.resolve()),cancel_file=str((args.output/'cancel').resolve())))
    os.execv(sys.executable,[sys.executable,'-m','fish_viewer.neural','--request',str(request)])

if __name__=='__main__':raise SystemExit(main())
