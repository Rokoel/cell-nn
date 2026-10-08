"""Build fresh signal crops as frozen cell predictions become available."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fish_viewer.pipeline.runner import _json
from fish_viewer.signals.config import SignalConfig
from fish_viewer.signals.data import build_audit


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection-audit',type=Path,required=True)
    p.add_argument('--training-cell-report',type=Path,required=True)
    p.add_argument('--validation-cell-report',type=Path,required=True)
    p.add_argument('--test-cell-report',type=Path,required=True)
    p.add_argument('--candidate-calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    collection=read(args.collection_audit)
    if collection['status']!='complete':raise ValueError('Completed collection audit required.')
    state=dict(version=1,status='running',phase='Waiting for validation candidate selection',cases=0,chunks=[])
    def emit(event):
        if event.get('kind')=='progress':
            state['progress']=event;_json(args.output/'progress.json',state)
        print(json.dumps(event,allow_nan=False),flush=True)
    while True:
        if args.candidate_calibration.is_file():
            calibration=read(args.candidate_calibration)
            if calibration['status']=='complete':break
            if calibration['status']=='failed':raise ValueError('Candidate validation failed.')
        _json(args.output/'progress.json',state);time.sleep(10)
    # Positive assignment is a training-only policy, not a candidate parameter.
    # Freeze it before building any crops; it cannot alter the calibration scores.
    config=replace(SignalConfig(**calibration['configs'][calibration['selected']]),one_to_one_positives=True)
    config.validate();_json(args.output/'frozen-config.json',asdict(config))
    calibration['training_target_policy']='One-to-one positive assignment; additional near-marker candidates ignored. Frozen before crop creation.'
    _json(args.output/'candidate-selection.json',calibration)
    reports={
        'training':args.training_cell_report,
        'validation':args.validation_cell_report,
        'test':args.test_cell_report,
    }
    seen=set();assembled=[];cell_rows=[]
    identities=None
    try:
        while len(seen)<len(collection['cases']):
            made_progress=False
            for split,path in reports.items():
                cells=read(path)
                if cells['status'] not in ('running','complete'):
                    raise ValueError(f'{split} cell prediction failed: {cells.get("error",cells["status"])}')
                current=(cells['checkpoint_sha256'],cells['boundary_sha256'])
                if identities is None:identities=current
                if current!=identities:raise ValueError('Cell model checkpoints differ between folds.')
                available={r['image']:r for r in cells['files']}
                ready=[c for c in collection['cases'] if c['audit']['split']==split and c['image'] in available and c['image'] not in seen]
                if len(ready)<20 and cells['status']=='running':continue
                if not ready:continue
                chunk=args.output/f'chunk-{len(state["chunks"])+1:04d}-{split}'
                chunk.mkdir()
                subset=dict(collection,cases=ready)
                _json(chunk/'collection.json',subset)
                snapshot=dict(cells,status='complete',files=[available[c['image']] for c in ready],
                              snapshot_of=str(path.resolve()),snapshot_policy='Only fully saved scan predictions; fixed model weights. The overall upstream run may still be processing later scans.')
                _json(chunk/'cells.json',snapshot)
                state['phase']=f'Auditing {split} signal crops';_json(args.output/'progress.json',state)
                audit=build_audit(chunk/'cells.json',chunk/'collection.json',chunk/'audit',config,emit)
                if audit['status']!='complete':raise ValueError(f'Signal audit errors: {audit["errors"]}')
                assembled.extend(audit['cases']);cell_rows.extend(snapshot['files'])
                seen.update(c['image'] for c in ready)
                state['chunks'].append(dict(split=split,scans=len(ready),audit=str(chunk/'audit/audit.json')))
                state['cases']=len(seen);_json(args.output/'progress.json',state)
                made_progress=True
            if not made_progress:
                state['phase']='Waiting for remaining training cell contours';_json(args.output/'progress.json',state);time.sleep(10)
        if seen!={c['image'] for c in collection['cases']}:
            raise ValueError('Signal cases do not cover the original collection exactly.')
        combined_cells=dict(read(args.training_cell_report),status='complete',files=cell_rows,
                            component_reports={k:str(v.resolve()) for k,v in reports.items()})
        combined_cells['summary']=dict(scans=len(cell_rows),cells=sum(c['cells'] for c in cell_rows),
                                       proposals=sum(len(c['proposals']) for c in cell_rows))
        _json(args.output/'cell-report.json',combined_cells)
        summary={}
        for split in reports:
            cases=[c for c in assembled if c['split']==split]
            summary[split]=dict(scans=len(cases),markers=sum(len(c['markers']) for c in cases),
                assigned_markers=sum(sum(m['channel'] is not None for m in c['markers']) for c in cases),
                eligible_signals=sum(len(c['truth']) for c in cases),positive_patches=sum(c['positive_patches'] for c in cases),
                negative_patches=sum(c['negative_patches'] for c in cases),marked_cells=sum(c['marked_cells'] for c in cases),
                ambiguous_cells=sum(c['ambiguous_cells'] for c in cases),predicted_cells=sum(c['predicted_cells'] for c in cases))
        groups={split:{c['group'] for c in assembled if c['split']==split} for split in reports}
        if any(groups[a]&groups[b] for a,b in [('training','validation'),('training','test'),('validation','test')]):
            raise ValueError('Acquisition leakage between signal folds.')
        audit=dict(version=1,engine='signal-audit',status='complete',directory=str(args.output.resolve()),
                   config=asdict(config),cases=assembled,summary=summary,errors=[],
                   cell_report=str((args.output/'cell-report.json').resolve()),cell_model=combined_cells['model_directory'],
                   collection_audit=str(args.collection_audit.resolve()),split_policy=collection.get('split_policy',collection.get('policy')),
                   cell_checkpoint_sha256=identities[0],boundary_checkpoint_sha256=identities[1],
                   supervision='Only predicted-cell interiors intersecting unambiguous fully marked manual cells. A is a signal. Unmarked cells ignored. One-to-one positive candidate assignment.',
                   candidate_selection=str((args.output/'candidate-selection.json').resolve()))
        _json(args.output/'audit.json',audit)
        state.update(status='complete',phase='Signal crops ready',summary=summary)
        _json(args.output/'progress.json',state);print(json.dumps(state),flush=True)
    except Exception as exc:
        state.update(status='failed',error=str(exc));_json(args.output/'progress.json',state)
        raise


if __name__=='__main__':main()
