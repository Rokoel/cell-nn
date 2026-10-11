"""Exploratory signal-first evaluation on reproducibly random held-out test scans."""
import argparse
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PySide6 import QtCore
from fish_viewer.analysis.controller import AnalysisController,default_models,default_proposal_model,default_refinement_model
from benchmark_all_heldout import evaluate_run, save, frozen_models, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,default=ROOT/'artifacts/refinement-fish/audit/audit.json')
    p.add_argument('--baseline',type=Path,default=ROOT/'artifacts/all-heldout-20261009-213558/report.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=3)
    p.add_argument('--seed',type=int,default=1010);p.add_argument('--roundness',type=float,default=.72)
    p.add_argument('--device',choices=['cpu','gpu','auto'],default='cpu');args=p.parse_args()
    collection=json.loads(args.audit.read_text());baseline=json.loads(args.baseline.read_text())
    models=frozen_models();old=baseline['identity']['models']
    for key in ('default_detector','contours','signals'):
        if models[key]['checkpoint_sha256']!=old[key]['checkpoint_sha256'] or models[key]['manifest_sha256']!=old[key]['manifest_sha256']:
            raise ValueError('Historical baseline uses different model weights or settings.')
    cases=sorted([c for c in collection['cases'] if c['audit']['split']=='test'],key=lambda c:c['image'])
    if not 1<=args.count<=len(cases):raise ValueError('Invalid scan count.')
    cases=random.Random(args.seed).sample(cases,args.count)
    train_groups={c['audit']['group'] for c in collection['cases'] if c['audit']['split']=='training'}
    assert not train_groups & {c['audit']['group'] for c in cases}
    args.output.mkdir(parents=True,exist_ok=False)
    app=QtCore.QCoreApplication([]);controller=AnalysisController();results=[];failures=[];active={}
    report=dict(status='running',seed=args.seed,minimum_roundness=args.roundness,models=models,
        audit_sha256=sha(args.audit),scans=[c['image'] for c in cases],results=results,failures=failures,
        baseline_source=str(args.baseline),baseline_reused=True,
        limitations='Small exploratory subset, not blind testing. Earlier default results reused for the same scans/checkpoints; new signal-first inference and geometry freshly executed. Incomplete manual annotations cannot establish unmatched-cell precision. Conditional matched-cell errors; all outliers retained.')
    def persist():save(args.output/'report.json',report)
    def fail(error):failures.append(dict(image=active['case']['image'],error=error));report['status']='failed';persist();app.exit(1)
    def complete(state):
        try:
            save(active['folder']/'pipeline-result.json',state)
            metrics=evaluate_run(active['case'],state,collection,active['folder'])
            row=next(r for r in baseline['results'] if r['image']==active['case']['image'] and r['mode']=='default')
            results.append(dict(signal_first=metrics,previous_crop_default=row))
            persist();print(f"Completed {len(results)}/{len(cases)} {metrics['scan']}",flush=True)
            QtCore.QTimer.singleShot(0,next_run)
        except Exception:
            import traceback;fail(traceback.format_exc())
    original_launch=controller._launch
    def launch(index,arguments):
        if controller.stages[index][0]=='export':
            controller.state.update(status='complete',export_omitted=True);controller._stop()
            complete(dict(controller.state,report_path=str(controller.directory/'analysis-result.json')))
        else:original_launch(index,arguments)
    controller._launch=launch;controller.failed.connect(fail)
    controller.progress.connect(lambda e:save(args.output/'progress.json',dict(scan=len(results)+1,scan_total=len(cases),**e)))
    app.aboutToQuit.connect(controller.shutdown)
    def next_run():
        if len(results)==len(cases):
            report['status']='complete';persist();app.exit(0);return
        case=cases[len(results)];folder=args.output/f'{len(results)+1:03d}-{Path(case["image"]).stem}';folder.mkdir()
        active.update(case=case,folder=folder)
        try:
            controller.start(case['image'],folder/'runs',default_proposal_model(),default_models()[1],
                refinement=default_refinement_model(),device=args.device,signal_first=True,roundness=args.roundness)
        except Exception as exc:
            fail(str(exc))
    persist();QtCore.QTimer.singleShot(0,next_run);return app.exec()

if __name__=='__main__':raise SystemExit(main())
