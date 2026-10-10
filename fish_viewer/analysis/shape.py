"""Apply a physical 3D sphericity gate to finished contours, without reshaping."""
import argparse
import json
from pathlib import Path
import traceback

import numpy as np
from scipy import ndimage as ndi

from ..pipeline.segmentation import mask_properties, checkpoint, Cancelled
from ..pipeline.runner import _json
from ..run_paths import new_run_directory


def filter_labels(labels, spacing, minimum, cancel=None):
    if not 0 <= minimum <= 1: raise ValueError('Minimum roundness must lie between 0 and 1.')
    result=labels.copy(); rows=[]
    for cell, box in enumerate(ndi.find_objects(labels), 1):
        checkpoint(cancel)
        if box is None: continue
        props=mask_properties(labels[box]==cell, np.asarray(spacing))
        accepted=props['sphericity'] >= minimum
        if not accepted: result[box][labels[box]==cell]=0
        rows.append(dict(cell_id=cell, sphericity=props['sphericity'], accepted=bool(accepted)))
    return result, rows


def apply(cells_report, output, minimum, cancel=None):
    directory=new_run_directory(output, 'roundness')
    report=dict(version=1, status='running', directory=str(directory), files=[], failures=[])
    try:
        source=json.loads(Path(cells_report).read_text())
        for index, row in enumerate(source['files']):
            checkpoint(cancel)
            with np.load(row['predictions'], allow_pickle=False) as f:
                labels=f['labels']; metadata=json.loads(str(f['metadata']))
            labels, scores=filter_labels(labels, metadata['spacing_zyx'], minimum, cancel)
            kept={r['cell_id'] for r in scores if r['accepted']}
            metadata['cells']=[c for c in metadata.get('cells',[]) if c['label'] in kept]
            metadata['roundness_filter']=dict(minimum=minimum, scores=scores, rejected=len(scores)-len(kept))
            path=directory/f'contours-{index}.npz'
            np.savez_compressed(path, labels=labels, metadata=np.array(json.dumps(metadata)))
            report['files'].append(dict(row, predictions=str(path), cells=len(kept)))
        report['status']='complete'
    except Cancelled: report['status']='cancelled'
    except Exception: report.update(status='failed', error=traceback.format_exc())
    _json(directory/'report.json',report)
    return report


def main():
    from ..neural.__main__ import Cancellation
    p=argparse.ArgumentParser();p.add_argument('--cells-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--minimum',type=float,required=True)
    p.add_argument('--cancel-file',type=Path);args=p.parse_args()
    report=apply(args.cells_report,args.output,args.minimum,Cancellation(args.cancel_file))
    print(json.dumps(dict(kind='finished',report=report)),flush=True)
    return 0 if report['status']=='complete' else 1

if __name__=='__main__':raise SystemExit(main())
