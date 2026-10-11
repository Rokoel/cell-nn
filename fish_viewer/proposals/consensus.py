"""Keep whole cell contours only when two detector pipelines agree."""
import argparse
import json
from pathlib import Path
import traceback

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..pipeline.runner import _json
from ..pipeline.segmentation import checkpoint, Cancelled
from ..refinement.inference import resample_labels
from ..run_paths import new_run_directory


def agree(primary, secondary, minimum_iou=.5):
    """One-to-one full-instance IoU matching; never intersect or shrink masks."""
    if primary.shape != secondary.shape:
        raise ValueError('Consensus masks must share a calibrated grid.')
    if not 0 < minimum_iou <= 1:
        raise ValueError('Consensus IoU must be in (0, 1].')
    a, sizes_a = np.unique(primary, return_counts=True)
    b, sizes_b = np.unique(secondary, return_counts=True)
    sizes_a = sizes_a[a > 0]; a = a[a > 0]
    sizes_b = sizes_b[b > 0]; b = b[b > 0]
    intersections = np.zeros((len(a), len(b)), dtype=np.int64)
    overlap = (primary > 0) & (secondary > 0)
    if overlap.any():
        ia = np.searchsorted(a, primary[overlap]); ib = np.searchsorted(b, secondary[overlap])
        intersections = np.bincount(ia * len(b) + ib, minlength=len(a)*len(b)).reshape(len(a), len(b))
    unions = sizes_a[:, None] + sizes_b[None, :] - intersections
    ious = np.divide(intersections, unions, out=np.zeros_like(unions,dtype=float), where=unions > 0)
    # Maximize qualifying match count first, IoU second. Subthreshold overlaps
    # cannot steal an assignment from a real agreement.
    weights = (ious >= minimum_iou) * (1 + min(len(a), len(b)) + ious)
    rows, cols = linear_sum_assignment(-weights)
    matches = [dict(cell_id=int(a[i]), secondary_cell_id=int(b[j]), iou=float(ious[i,j]))
               for i,j in zip(rows,cols) if ious[i,j] >= minimum_iou]
    kept = [m['cell_id'] for m in matches]
    output = primary.copy(); output[~np.isin(output,kept)] = 0
    return output, matches


def apply(primary_report, secondary_report, output, minimum_iou=.5, emit=lambda e:None, cancel=None):
    directory = new_run_directory(output,'cell-consensus')
    report = dict(version=1,engine='cell-consensus',status='running',directory=str(directory),files=[],failures=[])
    try:
        primary=json.loads(Path(primary_report).read_text()); secondary=json.loads(Path(secondary_report).read_text())
        if any(r['status']!='complete' or r.get('failures') for r in (primary,secondary)):
            raise ValueError('Consensus requires two complete cell reports.')
        other={str(Path(row['image']).resolve()):row for row in secondary['files']}
        for index,row in enumerate(primary['files']):
            checkpoint(cancel)
            emit(dict(kind='progress',phase='Keeping cells detected by both approaches',current=index,total=len(primary['files'])))
            partner=other[str(Path(row['image']).resolve())]
            with np.load(row['predictions'],allow_pickle=False) as f:
                labels=f['labels'];metadata=json.loads(str(f['metadata']))
            with np.load(partner['predictions'],allow_pickle=False) as f:
                other_labels=f['labels'];other_metadata=json.loads(str(f['metadata']))
            for key in ('fingerprint','channel','time'):
                if metadata[key]!=other_metadata[key]:
                    raise ValueError(f'Consensus source mismatch: {key}.')
            if any(m.get('prediction_role')=='candidate_only' for m in (metadata,other_metadata)):
                raise ValueError('Provisional masks cannot establish consensus.')
            other_labels=resample_labels(other_labels,np.array(other_metadata['spacing_zyx']),np.array(other_metadata['origin_xyz']),
                labels.shape,metadata['spacing_zyx'],metadata['origin_xyz'])
            original_count=len(np.unique(labels[labels>0]))
            labels,matches=agree(labels,other_labels,minimum_iou)
            kept={m['cell_id'] for m in matches}
            metadata['cells']=[c for c in metadata['cells'] if c['label'] in kept]
            metadata.pop('metrics',None)
            metadata['consensus']=dict(minimum_iou=minimum_iou,matches=matches,original_cells=original_count,
                primary_report=str(Path(primary_report).resolve()),secondary_report=str(Path(secondary_report).resolve()),
                policy='One-to-one full-object IoU agreement; keep complete primary contours, never intersect masks.')
            folder=directory/f'{index+1:04d}';folder.mkdir()
            path=folder/'contours.npz';np.savez_compressed(path,labels=labels,metadata=np.array(json.dumps(metadata,allow_nan=False)))
            new_row=dict(row,predictions=str(path),cells=len(matches),consensus=metadata['consensus'])
            new_row.pop('metrics',None);report['files'].append(new_row)
        report.update(status='complete',summary=dict(cells=sum(r['cells'] for r in report['files']),
            original_cells=sum(r['consensus']['original_cells'] for r in report['files']),minimum_iou=minimum_iou))
    except Cancelled:report['status']='cancelled'
    except Exception:report.update(status='failed',error=traceback.format_exc())
    _json(directory/'report.json',report);emit(dict(kind='finished',report=report));return report


def main():
    from ..worker_runtime import Cancellation
    import signal
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary-report',type=Path,required=True);p.add_argument('--secondary-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--minimum-iou',type=float,default=.5)
    p.add_argument('--cancel-file',type=Path);args=p.parse_args()
    cancel=Cancellation(args.cancel_file);signal.signal(signal.SIGINT,lambda *_:cancel.event.set())
    r=apply(args.primary_report,args.secondary_report,args.output,args.minimum_iou,
        emit=lambda e:print(json.dumps(e,allow_nan=False),flush=True),cancel=cancel)
    return 0 if r['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
