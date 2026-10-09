"""Show fixed test cases most undersized by the original model; no selection/tuning."""
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from fish_viewer.neural.audit import read_references


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--comparison',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    run=json.loads(args.report.read_text());comparison=json.loads(args.comparison.read_text())
    baseline=json.loads(Path(run['baseline_report']).read_text())
    old={str(Path(r['image']).relative_to(ROOT)):r for r in baseline['files']};new={str(Path(r['image']).relative_to(ROOT)):r for r in run['files']}
    audit=json.loads((ROOT/'artifacts/refinement-fish/audit/audit.json').read_text());cases={str(Path(c['image']).relative_to(ROOT)):c for c in audit['cases']}
    selected=sorted(comparison['cells'],key=lambda r:r['baseline_volume_um3']/r['manual_volume_um3'])[:8]
    fig=Figure(figsize=(13,14),layout='constrained');FigureCanvasAgg(fig);axes=fig.subplots(4,2).ravel()
    for ax,r in zip(axes,selected):
        key=r['image'];case=cases[key];cell=r['predicted_cell']
        image=np.load(Path(case['directory'])/'image.npy',mmap_mode='r');refs,names=read_references(Path(case['directory'])/'references.npz')
        box,mask=refs[r['reference_id']-1]
        z=box[0].start+int(np.argmax(mask.sum(axis=(1,2))))
        truth=np.zeros(image.shape[1:],bool);truth[box[1:]]=mask[z-box[0].start]
        with np.load(old[key]['predictions']) as a:before=a['labels'][z]==cell
        with np.load(new[key]['predictions']) as a:after=a['labels'][z]==cell
        low=np.maximum(0,np.array([box[1].start,box[2].start])-12);high=np.minimum(image.shape[1:],np.array([box[1].stop,box[2].stop])+12)
        region=tuple(slice(int(a),int(b)) for a,b in zip(low,high))
        ax.imshow(image[z][region],cmap='gray',vmin=0,vmax=.8,origin='lower')
        for m,color in [(truth,'gold'),(before,'tomato'),(after,'cyan')]:
            if m[region].any():ax.contour(m[region],levels=[.5],colors=[color],linewidths=1.3)
        old_bias=(r['baseline_volume_um3']/r['manual_volume_um3']-1)*100;new_bias=(r['refined_volume_um3']/r['manual_volume_um3']-1)*100
        ax.set_title(f'{Path(key).stem}\nCell {cell}: volume bias {old_bias:+.1f}% → {new_bias:+.1f}%',fontsize=9)
        ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Gold: manual · Red: current default · Cyan: candidate\nEight most undersized baseline test cells; largest manual-area slice shown',fontsize=12)
    args.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.output,dpi=160)
if __name__=='__main__':main()
