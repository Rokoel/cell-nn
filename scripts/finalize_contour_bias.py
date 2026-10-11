"""Freeze validation choice, evaluate test contours and verify the candidate app."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'artifacts/contour-bias-20261009'

def call(name,arguments):
    with (RUN/(name+'.log')).open('w') as stream:
        subprocess.run([sys.executable,*map(str,arguments)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)

def main():
    state=json.loads((RUN/'training-stages.json').read_text())
    if state['status']!='complete':raise ValueError('Validation stages must finish before test evaluation')
    call('post-training-thresholds',['scripts/experiment_contour_bias.py','--audit',ROOT/'artifacts/refinement-fish/audit/audit.json',
        '--baseline',ROOT/'artifacts/proposal-fish/validation-contours-final/center-contours-20261007-055507-430727-vlcnm8p3/report.json',
        '--model',state['best_model'],'--output',RUN/'post-training-thresholds','--mode','threshold'])
    cutoff=json.loads((RUN/'post-training-thresholds/report.json').read_text())['summary']
    eligible=[(float(key.split('-')[-1]),row) for key,row in cutoff.items() if row['cells_still_matched']>=.99*row['cells'] and row['volume_mape']<=row['baseline']['volume_mape']]
    threshold,metrics=min(eligible,key=lambda item:(item[1]['boundary_error_um'],item[1]['volume_mape']))
    state.update(selected_threshold=threshold,final_threshold_validation_metrics=metrics,best_score=metrics['boundary_error_um'])
    (RUN/'training-stages.json').write_text(json.dumps(state,indent=2))
    candidate=ROOT/'models/refinement-bias-candidate'
    if candidate.exists():raise ValueError('Candidate already exists; do not overwrite selected models')
    shutil.copytree(state['best_model'],candidate)
    manifest=json.loads((candidate/'refinement-model.json').read_text())
    manifest['config'].update(threshold=state['selected_threshold'],margin_um=state['selected_margin_um'],patch_size=state['selected_patch_size'])
    manifest.update(checkpoint_sha256=state['selected_checkpoint_sha256'],inference_crop_margin_um=state['selected_margin_um'],
        selection_policy='Frozen before test. Original-grid validation symmetric surface error on fixed 441 matches, non-increasing baseline volume MAPE and >=99% original IoU matches. All test outliers retained.',
        experiment=state,experimental=True)
    manifest['training_config']=dict(manifest['config'])
    if not manifest['config'].get('residual_correction',False):
        # Preserve v0.4.0 inference compatibility for the unchanged architecture.
        for key in ('surface_loss_weight','surface_distance_clip_um','tversky_beta','residual_correction'):
            manifest['config'].pop(key,None)
    (candidate/'refinement-model.json').write_text(json.dumps(manifest,indent=2)+'\n')
    provenance=json.loads((ROOT/'models/origins.json').read_text());provenance[candidate.name]=dict(source=state['best_model'],sha256={
        name:hashlib.sha256((candidate/name).read_bytes()).hexdigest() for name in ('best.weights.h5','refinement-model.json')})
    (ROOT/'models/origins.json').write_text(json.dumps(provenance,indent=2)+'\n')
    freeze=dict(status='frozen_before_test',model=str(candidate),checkpoint_sha256=state['selected_checkpoint_sha256'],
        config=manifest['config'],validation_selected_stage=state['best_stage'])
    (RUN/'test-freeze.json').write_text(json.dumps(freeze,indent=2))
    call('candidate-empty-controls',['scripts/check_cell_crop_controls.py','--device','cpu','--boundary-model',candidate,'--output',RUN/'candidate-empty-controls.json'])
    benchmark=json.loads((ROOT/'reports/center-box-contours/test-measurements/report.json').read_text())
    audit=ROOT/'artifacts/refinement-fish/audit/audit.json';baseline=Path(benchmark['cell_report'])
    call('test-contours',['scripts/experiment_contour_bias.py','--audit',audit,'--baseline',baseline,'--model',candidate,
        '--output',RUN/'test-contours','--mode','apply','--threshold',state['selected_threshold'],
        '--margin',state['selected_margin_um'],'--size',state['selected_patch_size']])
    call('test-comparison',['scripts/compare_refined_boundaries.py','--audit',audit,'--baseline-report',baseline,
        '--refined-report',RUN/'test-contours/report.json','--output',RUN/'test-comparison'])
    call('test-signals',['scripts/evaluate_contour_bias_signals.py','--report',RUN/'test-contours/report.json',
        '--cell-comparison',RUN/'test-comparison/comparison.json','--output',RUN/'test-signals'])
    call('app-smoke',['scripts/smoke_analysis.py','--device','cpu','--boundary-model',candidate])
    call('final-report',['scripts/report_contour_bias.py'])
    call('test-overlays',['scripts/plot_contour_bias_overlays.py','--report',RUN/'test-contours/report.json',
        '--comparison',RUN/'test-comparison/comparison.json','--output',ROOT/'reports/contour-bias/test-overlays.png'])
    (RUN/'completion.json').write_text(json.dumps(dict(status='complete',model=str(candidate),selected_stage=state['best_stage'],
        test_report=str(RUN/'test-comparison/comparison.json'),signal_report=str(RUN/'test-signals/report.json'),
        app_smoke_log=str(RUN/'app-smoke.log'),defaults_changed=False),indent=2))
if __name__=='__main__':main()
