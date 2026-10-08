"""One-to-one center matching in physical units and per-cell/channel counts."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def detection_separation(config):
    return config.detection_separation_um or config.candidate_separation_um


def suppress(rows,scores,threshold,separation):
    selected=[];bins={}
    for index in np.argsort(-np.asarray(scores),kind='stable'):
        if scores[index]<threshold:continue
        row=rows[index]
        xyz=np.asarray(row['xyz'])
        if separation>0:
            key=tuple(np.floor(xyz/separation).astype(int))
            neighbors=(point for dz in (-1,0,1) for dy in (-1,0,1) for dx in (-1,0,1)
                for point in bins.get((row['channel'],row['cell_id'],key[0]+dx,key[1]+dy,key[2]+dz),()))
            if any(np.linalg.norm(point-xyz)<separation for point in neighbors):continue
            bins.setdefault((row['channel'],row['cell_id'],*key),[]).append(xyz)
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
        selected=suppress(case['candidates'],scores,threshold,detection_separation(config))
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
    # Suppression is greedy in score order: lowering the cutoff only appends
    # detections. Sweep the observed scores and maintain maximum bipartite
    # matching, then run distance-minimizing matching at the selected cutoff.
    # This avoids missing useful cutoffs between values of a coarse score grid.
    events=[];neighbors=[];owners=[]
    references=sum(len(c['truth']) for c in cases)
    for index,(case,scores) in enumerate(zip(cases,probabilities)):
        selected=suppress(case['candidates'],scores,0.,detection_separation(config))
        compatible=[]
        for point in selected:
            compatible.append([i for i,ref in enumerate(case['truth']) if point['channel']==ref['channel']
                and point['cell_id']==ref['cell_id'] and np.linalg.norm(np.asarray(point['xyz'])-ref['xyz'])<=config.match_radius_um])
        neighbors.append(compatible);owners.append({})
        events.extend((p['score'],index,i) for i,p in enumerate(selected))
    events.sort(key=lambda item:-item[0])
    def augment(case,prediction,visited):
        for ref in neighbors[case][prediction]:
            if ref in visited:continue
            visited.add(ref)
            if ref not in owners[case] or augment(case,owners[case][ref],visited):
                owners[case][ref]=prediction;return True
        return False
    table=[];tp=0;predictions=0;position=0
    while position<len(events):
        threshold=events[position][0]
        while position<len(events) and events[position][0]==threshold:
            _,case,prediction=events[position]
            tp+=int(augment(case,prediction,set()));predictions+=1;position+=1
        metrics=summarize([dict(tp=tp,fp=predictions-tp,fn=references-tp,distances_um=[])])
        table.append(dict(threshold=threshold,metrics=metrics))
    if not table:
        table=[dict(threshold=1.,metrics=evaluate_cases(cases,probabilities,1.,config))]
    qualified=[row for row in table if row['metrics']['predictions']>=5 and (row['metrics']['precision'] or 0)>=config.target_precision]
    if qualified:
        best=max(qualified,key=lambda row:(row['metrics']['recall'] or 0,row['metrics']['precision'],row['threshold']))
    else:
        best=max(table,key=lambda row:(row['metrics']['f05'],row['metrics']['precision'] or 0,row['threshold']))
    selected=dict(threshold=best['threshold'],metrics=evaluate_cases(cases,probabilities,best['threshold'],config))
    # Keep the complete computation exact while limiting report/manifest size.
    retained=sorted(set(np.linspace(0,len(table)-1,min(512,len(table))).astype(int).tolist()+[table.index(best)]))
    return dict(selected=selected,target_precision=config.target_precision,target_met=bool(qualified),
        evaluated_score_thresholds=len(table),candidates=[table[i] for i in retained],
        policy='Validation observed scores only: maximize recall at target precision with at least 5 detections; otherwise maximize F0.5. Matching within same cell and channel at declared physical tolerance. Stored curve may be downsampled; selection is exact.')
