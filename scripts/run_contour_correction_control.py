"""Separate residual architecture from the unsuccessful asymmetric-loss setting."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'artifacts/contour-bias-20261009'

def main():
    state_path=RUN/'training-stages.json';state=json.loads(state_path.read_text())
    if state['status']!='complete':raise ValueError('Finish the registered three training stages first')
    stage='residual-balanced-loss';output=RUN/stage
    if output.exists():raise ValueError('Do not overwrite prior trials')
    initial=Path(next(s['model'] for s in state['stages'] if s['stage']=='surface-loss'))
    state.update(status='running',current_stage=stage);state_path.write_text(json.dumps(state,indent=2))
    with (RUN/(stage+'.log')).open('w') as stream:
        subprocess.run([sys.executable,str(ROOT/'scripts/train_contour_bias.py'),'--initial-model',str(initial),
            '--output',str(output),'--patches',str(RUN/'patches'),'--threshold','.45','--beta','.5','--residual'],
            cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    train_path=next(output.glob('*/report.json'));training=json.loads(train_path.read_text());candidate=Path(training['model_directory'])
    state['current_stage']=stage+' validation';state_path.write_text(json.dumps(state,indent=2))
    score_output=RUN/(stage+'-validation')
    with (RUN/(stage+'-validation.log')).open('w') as stream:
        subprocess.run([sys.executable,str(ROOT/'scripts/experiment_contour_bias.py'),'--audit',str(ROOT/'artifacts/refinement-fish/audit/audit.json'),
            '--baseline',str(ROOT/'artifacts/proposal-fish/validation-contours-final/center-contours-20261007-055507-430727-vlcnm8p3/report.json'),
            '--model',str(candidate),'--output',str(score_output),'--mode','score','--threshold','.45'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    result=json.loads((score_output/'report.json').read_text())['summary']['apply']
    eligible=result['cells_still_matched']>=.99*result['cells'] and result['volume_mape']<=result['baseline']['volume_mape']
    selected=eligible and result['boundary_error_um']<state['best_score']-1e-5
    state['stages'].append(dict(stage=stage,model=str(candidate),training_report=str(train_path),selected=selected,eligible=eligible,
        best_epoch=training['best_epoch'],epochs=len(training['epochs']),initial_validation=training.get('initial_validation'),metrics=result,
        rationale='Asymmetric loss failed to improve its parent. Control separates the residual architecture from that loss setting; registered before the control and without any test evaluation.'))
    if selected:
        state.update(best_model=str(candidate),best_stage=stage,best_score=result['boundary_error_um'],
            selected_checkpoint_sha256=hashlib.sha256((candidate/'best.weights.h5').read_bytes()).hexdigest())
    state.update(status='complete',current_stage='Validation selection frozen after residual loss control')
    state_path.write_text(json.dumps(state,indent=2));print(json.dumps(state,indent=2))
if __name__=='__main__':main()
