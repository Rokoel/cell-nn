"""Wait for the complete refreshed corpus, then train and compare fixed models."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--baseline-model',type=Path,required=True)
    p.add_argument('--config',type=Path,help='Optional training recipe; audited candidate inputs remain frozen.')
    p.add_argument('--initial-model',type=Path,help='Optional signal checkpoint for fine-tuning.')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    while True:
        path=args.data/'progress.json'
        if path.is_file():
            state=json.loads(path.read_text())
            if state['status']=='failed':raise ValueError(state['error'])
            if state['status']=='complete':break
        time.sleep(10)
    root=Path(__file__).resolve().parents[1]
    def run(name,arguments):
        with (args.output/(name+'.log')).open('w') as stream:
            subprocess.run([sys.executable,*arguments],cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=True)
    training_args=['-m','fish_viewer.signals','--mode','train','--audit',str((args.data/'audit.json').resolve()),
                    '--output',str((args.output/'training').resolve()),'--device','gpu']
    if args.config:training_args+=['--config',str(args.config.resolve())]
    if args.initial_model:training_args+=['--initial-model',str(args.initial_model.resolve())]
    run('training',training_args)
    reports=list((args.output/'training').glob('*/report.json'))
    if len(reports)!=1:raise ValueError('Expected exactly one completed fresh training run.')
    report=json.loads(reports[0].read_text())
    if report['status']!='complete':raise ValueError(report.get('error','Signal training incomplete.'))
    model=report['model_directory']
    run('controls',['-m','fish_viewer.signals','--mode','controls','--model',model,
                    '--output',str((args.output/'controls').resolve()),'--device','gpu'])
    for split in ('validation','test'):
        run('comparison-'+split,['scripts/compare_signal_models.py','--audit',str((args.data/'audit.json').resolve()),
            '--old-model',str(args.baseline_model.resolve()),'--new-model',model,'--split',split,
            '--output',str((args.output/('comparison-'+split)).resolve()),'--device','gpu'])
    result=dict(status='complete',training_report=str(reports[0].resolve()),model=model,
                validation_comparison=str((args.output/'comparison-validation/report.json').resolve()),
                test_comparison=str((args.output/'comparison-test/report.json').resolve()),
                controls=str((args.output/'controls/controls.json').resolve()))
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
