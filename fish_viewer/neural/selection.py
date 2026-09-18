"""Bounded validation selection with mandatory membrane and negative controls."""
from dataclasses import asdict,replace
from collections import Counter
import numpy as np

from ..pipeline.evaluation import evaluate,aggregate
from ..pipeline.segmentation import checkpoint,mask_properties
from .precision import shape_eligible


def acceptance_candidates(config):
    return [replace(config,probability_threshold=p,membrane_support=s).validate()
        for p,s in ((.65,.60),(.55,.55),(.55,.50),(.45,.50))]


def choose_candidate(rows,target=.60):
    safe=[r for r in rows if r['controls']['passed'] and r.get('eligible_reference_count',0)>0]
    if not safe:raise ValueError('No acceptance setting passed negative controls with evaluable validation cells.')
    enough=[r for r in safe if r['eligible_detection_rate']>=target]
    if enough:
        return min(enough,key=lambda r:(r['predicted_count'],-r['config']['probability_threshold'],-r['config']['membrane_support']))
    return min(safe,key=lambda r:(-r['eligible_detection_rate'],r['predicted_count'],-r['config']['probability_threshold']))


def select_acceptance(model,validation,config,emit,cancel=None):
    from .engine import predict,aggregate_subsets
    from .controls import evaluate_controls
    candidates=acceptance_candidates(config);rows=[]
    selections={}
    for case in validation:
        selections[id(case)]=[i for i,((_,mask),cut) in enumerate(zip(case.references,case.reference_cut))
            if not cut and shape_eligible(mask_properties(mask,np.asarray(case.prepared.spacing)),config)]
    for index,candidate in enumerate(candidates):
        checkpoint(cancel)
        emit(dict(kind='progress',phase='Synthetic negative controls',current=index,total=len(candidates)))
        controls=evaluate_controls(model,candidate)
        row=dict(index=index,config=asdict(candidate),controls=controls)
        if controls['passed']:
            scores=[];eligible=[];rejections=Counter()
            for number,case in enumerate(validation):
                checkpoint(cancel)
                emit(dict(kind='progress',phase='Validation acceptance settings',current=index*len(validation)+number,
                    total=len(candidates)*len(validation),file=str(case.entry.image),candidate=index+1))
                prediction,metrics=predict(model,case,candidate,cancel,emit)
                if metrics['proposal_limit_reached']:raise ValueError('Proposal cap reached during acceptance selection; no partial sweep will be selected.')
                ids=selections[id(case)]
                eligible.append(evaluate(prediction.labels,[case.references[i] for i in ids],
                    [case.names[i] for i in ids],case.prepared.spacing,cancel))
                scores.append(metrics);rejections.update(prediction.rejected)
            cells=[c for score in eligible for c in score['per_cell']]
            row.update(validation=aggregate_subsets(scores),eligible=aggregate(eligible),
                eligible_reference_count=len(cells),eligible_detection_rate=sum(c['detected'] for c in cells)/len(cells) if cells else 0,
                predicted_count=sum(s['predicted_count'] for s in scores),rejections=dict(rejections))
        rows.append(row)
        emit(dict(kind='balanced_candidate',candidate=row))
    selected=choose_candidate(rows)
    return candidates[selected['index']],dict(candidates=rows,selected_index=selected['index'],target_eligible_recall=.60,
        target_met=selected['eligible_detection_rate']>=.60,
        policy='Require zero accepted synthetic negatives and membrane evidence for every candidate. Among settings with at least 60% pooled detection of shape-eligible validation references, choose the fewest contours; otherwise choose the highest detection within these bounded settings. Test images are not used. Counts are a conservative selection preference, not measured false positives.')
