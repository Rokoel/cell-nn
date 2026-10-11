"""Run registered contour fine-tuning stages sequentially, validation only."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/contour-bias-20261009'
BASELINE=ROOT/'artifacts/proposal-fish/validation-contours-final/center-contours-20261007-055507-430727-vlcnm8p3/report.json'

def save(state):
    target=RUN/'training-stages.json';temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2));temporary.replace(target)

def main():
    patches=RUN/'patches/index.json'
    if not patches.exists():raise ValueError('Complete production-box patch cache required')
    best_model=ROOT/'models/refinement'
    initial=json.loads((RUN/'geometry/report.json').read_text())['summary']['margin-2-size-64']
    best_score=initial['boundary_error_um'];baseline_mape=initial['baseline']['volume_mape']
    state=dict(status='running',best_model=str(best_model),best_stage='threshold',best_score=best_score,stages=[])
    save(state)
    for stage,beta,residual in [('surface-loss',.5,False),('asymmetric-loss',.6,False),('residual-correction',.6,True)]:
        output=RUN/stage
        if output.exists():raise ValueError('Do not overwrite previous training stages')
        state['current_stage']=stage;save(state)
        arguments=[sys.executable,str(ROOT/'scripts/train_contour_bias.py'),'--initial-model',str(best_model),
            '--output',str(output),'--patches',str(patches.parent),'--threshold','.45','--beta',str(beta)]
        if residual:arguments.append('--residual')
        with (RUN/(stage+'.log')).open('w') as stream:
            subprocess.run(arguments,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        train_path=next(output.glob('*/report.json'));training=json.loads(train_path.read_text())
        candidate=Path(training['model_directory'])
        if training['status']!='complete':raise ValueError(training.get('error'))
        state['current_stage']=stage+' validation';save(state)
        score_output=RUN/(stage+'-validation')
        with (RUN/(stage+'-validation.log')).open('w') as stream:
            subprocess.run([sys.executable,str(ROOT/'scripts/experiment_contour_bias.py'),'--audit',
                str(ROOT/'artifacts/refinement-fish/audit/audit.json'),'--baseline',str(BASELINE),
                '--model',str(candidate),'--output',str(score_output),'--mode','score','--threshold','.45'],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        result=json.loads((score_output/'report.json').read_text())['summary']['apply']
        eligible=result['cells_still_matched']>=.99*result['cells'] and result['volume_mape']<=baseline_mape
        selected=eligible and result['boundary_error_um']<best_score-1e-5
        state['stages'].append(dict(stage=stage,model=str(candidate),training_report=str(train_path),
            selected=selected,eligible=eligible,best_epoch=training['best_epoch'],epochs=len(training['epochs']),
            initial_validation=training.get('initial_validation'),metrics=result))
        if selected:
            best_model=candidate;best_score=result['boundary_error_um'];state['best_stage']=stage
        state.update(best_model=str(best_model),best_score=best_score);save(state)
    state.update(status='complete',current_stage='Validation selection frozen',selected_threshold=.45,selected_margin_um=2,selected_patch_size=64,
        selected_checkpoint_sha256=hashlib.sha256((best_model/'best.weights.h5').read_bytes()).hexdigest())
    save(state);print(json.dumps(state,indent=2))
if __name__=='__main__':main()
