"""Plot full test-set detection and conditional geometry errors from completed runs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,default=ROOT/'reports/all-heldout-comparison')
args=parser.parse_args();report=json.loads((args.run/'report.json').read_text())
if report['status']!='complete' or report['completed_mode_runs']!=report['expected_mode_runs']:raise ValueError('Complete coverage is required before plotting.')
args.output.mkdir(parents=True,exist_ok=True)
colors={'default':'#2878b5','hybrid':'#dc7c22','consensus':'#4b9b64'}
labels={'default':'Default','hybrid':'Hybrid StarDist','consensus':'Consensus'}
fig,axes=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
summary=report['summary']['test']['modes'];positions=np.arange(3)
for i,mode in enumerate(colors):
 row=summary[mode]
 axes[0].bar(i-.16,row['matched_complete_before_filter'],.3,color=colors[mode],alpha=.45)
 axes[0].bar(i+.16,row['matched_complete_final'],.3,color=colors[mode])
axes[0].set_xticks(positions,[labels[m] for m in colors]);axes[0].set_ylabel('Matched complete manual cells')
axes[0].set_title('Detection: before / after signal filtering')
axes[0].axhline(summary['default']['complete_references'],color='#777777',linestyle=':',label='Manual complete cells')
axes[0].legend(handles=[Patch(facecolor='#777777',alpha=.45,label='Before filtering'),Patch(facecolor='#777777',label='With signals'),Line2D([0],[0],color='#777777',linestyle=':',label='Manual complete cells')],fontsize=8)
for ax,field,truth_key,pred_key,percent,title in [
    (axes[1],'volume_pairs','manual_um3','predicted_um3',True,'Volume error'),
    (axes[2],'distance_pairs','manual_um','predicted_um',False,'Distance error')]:
 for mode,color in colors.items():
  pairs=[p for row in report['results'] if row['split']=='test' and row['mode']==mode for p in row[field] if not p['cut']]
  errors=np.array([abs(p[pred_key]-p[truth_key])/(p[truth_key] if percent else 1)*(100 if percent else 1) for p in pairs])
  if not len(errors):continue
  errors=np.sort(errors);cdf=np.arange(1,len(errors)+1)/len(errors)*100
  ax.step(np.r_[0,errors],np.r_[0,cdf],where='post',color=color,label=f'{labels[mode]} (n={len(errors)})')
 ax.set_ylim(0,101);ax.set_ylabel('Matched pairs at or below error (%)')
 ax.set_xlabel('Absolute volume error (%)' if percent else 'Absolute A–B distance error (µm)')
 ax.set_title(title);ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle('Full held-out test set: 71 scans\nGeometry curves use each mode’s own matched complete-cell subset; all outliers retained.',fontsize=12)
fig.savefig(args.output/'test-comparison.png',dpi=180);fig.savefig(args.output/'test-comparison.svg');plt.close(fig)
