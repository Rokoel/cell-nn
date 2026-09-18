"""One-to-one center matching in physical units and per-cell/channel counts."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def suppress(rows,scores,threshold,separation):
    selected=[]
    for index in np.argsort(-np.asarray(scores),kind='stable'):
        if scores[index]<threshold:continue
        row=rows[index]
        if any(old['channel']==row['channel'] and old['cell_id']==row['cell_id'] and
            np.linalg.norm(np.asarray(old['xyz'])-row['xyz'])<separation for old in selected):continue
        selected.append(dict(row,score=float(scores[index])))
    return selected


def evaluate_points(predictions,references,radius):
    matched=[]
    if predictions and references:
        distance=cdist([p['xyz'] for p in predictions],[p['xyz'] for p in references])
        compatible=np.array([[p['channel']==r['channel'] and p['cell_id']==r['cell_id'] for r in references] for p in predictions])
        cost=np.where(compatible&(distance<=radius),distance,1e6)
        left,right=linear_sum_assignment(cost)
        matched=[float(distance[a,b]) for a,b in zip(left,right) if cost[a,b]<1e6]
    tp=len(matched);fp=len(predictions)-tp;fn=len(references)-tp
    return dict(tp=tp,fp=fp,fn=fn,distances_um=matched)


def summarize(rows):
    tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows)
    distances=[d for row in rows for d in row['distances_um']]
    precision=tp/(tp+fp) if tp+fp else None
    recall=tp/(tp+fn) if tp+fn else None
    return dict(tp=tp,fp=fp,fn=fn,predictions=tp+fp,references=tp+fn,precision=precision,recall=recall,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
        f05=1.25*tp/(1.25*tp+fp+.25*fn) if 1.25*tp+fp+.25*fn else 0.,
        localization_rmse_um=float(np.sqrt(np.mean(np.square(distances)))) if distances else None,
        localization_mean_um=float(np.mean(distances)) if distances else None)


def evaluate_cases(cases,probabilities,threshold,config):
    rows=[];counts=[];by_channel={}
    for case,scores in zip(cases,probabilities):
        selected=suppress(case['candidates'],scores,threshold,config.candidate_separation_um)
        metric=evaluate_points(selected,case['truth'],config.match_radius_um);rows.append(metric)
        groups={(r['cell_id'],r['channel']) for r in selected+case['truth']}
        for cell_id,channel in groups:
            predicted=sum(p['cell_id']==cell_id and p['channel']==channel for p in selected)
            expected=sum(p['cell_id']==cell_id and p['channel']==channel for p in case['truth'])
            counts.append(abs(predicted-expected))
        for channel in {r['channel'] for r in selected+case['truth']}:
            by_channel.setdefault(str(channel),[]).append(evaluate_points(
                [p for p in selected if p['channel']==channel],[p for p in case['truth'] if p['channel']==channel],config.match_radius_um))
    result=summarize(rows)
    result.update(count_mae_per_observed_cell_channel=float(np.mean(counts)) if counts else None,
        by_channel={key:summarize(value) for key,value in by_channel.items()})
    return result


def select_threshold(cases,probabilities,config):
    table=[]
    for threshold in np.r_[np.arange(.05,1,.025),.99,.995]:
        metrics=evaluate_cases(cases,probabilities,float(threshold),config)
        table.append(dict(threshold=float(threshold),metrics=metrics))
    qualified=[row for row in table if row['metrics']['predictions']>=5 and (row['metrics']['precision'] or 0)>=config.target_precision]
    if qualified:
        best=max(qualified,key=lambda row:(row['metrics']['recall'] or 0,row['metrics']['precision'],row['threshold']))
    else:
        best=max(table,key=lambda row:(row['metrics']['f05'],row['metrics']['precision'] or 0,row['threshold']))
    return dict(selected=best,target_precision=config.target_precision,target_met=bool(qualified),candidates=table,
        policy='Validation only: maximize recall at target precision with at least 5 detections; otherwise maximize F0.5. Matching within same cell and channel at declared physical tolerance.')
