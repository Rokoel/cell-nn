import numpy as np
import pytest
from fish_viewer.proposals.consensus import agree


def test_consensus_keeps_full_contours_not_voxel_intersection():
    a=np.zeros((24,)*3,np.uint32);b=np.zeros_like(a)
    a[3:11,3:11,3:11]=7;a[14:20,14:20,14:20]=12
    b[4:12,3:11,3:11]=42
    result,matches=agree(a,b)
    assert [m['cell_id'] for m in matches]==[7]
    assert matches[0]['secondary_cell_id']==42
    assert np.array_equal(result==7,a==7)
    assert result[3,3,3]==7 and b[3,3,3]==0
    assert not (result==12).any()


def test_consensus_one_other_cell_cannot_confirm_two_cells():
    a=np.zeros((20,)*3,np.uint32);b=np.zeros_like(a)
    a[3:7,3:9,3:9]=1;a[7:11,3:9,3:9]=2
    b[3:11,3:9,3:9]=9
    result,matches=agree(a,b)
    assert len(matches)==1
    assert len(np.unique(result[result>0]))==1


def test_touching_or_empty_masks_do_not_count_as_agreement():
    a=np.zeros((20,)*3,np.uint32);b=np.zeros_like(a)
    a[2:8,2:8,2:8]=5;b[8:14,2:8,2:8]=6
    assert not agree(a,b)[0].any()
    assert not agree(a,np.zeros_like(a))[1]
    assert not agree(np.zeros_like(a),b)[1]
    with pytest.raises(ValueError):agree(a,b,0)


def test_evaluation_counts_sparse_cell_ids_after_consensus():
    from fish_viewer.pipeline.evaluation import evaluate
    labels=np.zeros((24,)*3,np.uint32)
    labels[3:9,3:9,3:9]=7;labels[14:20,14:20,14:20]=120
    empty=evaluate(labels,[],[],np.ones(3))
    assert empty['predicted_count']==2 and empty['unscored_predictions']==2
    reference=[((slice(3,9),)*3,np.ones((6,)*3,bool))]
    matched=evaluate(labels,reference,['cell'],np.ones(3))
    assert matched['predicted_count']==2 and matched['matched_count']==1
    assert matched['unscored_predictions']==1


@pytest.mark.parametrize('mismatch',[None,'channel','fingerprint','time'])
def test_consensus_worker_validates_sources_and_publishes_only_agreed_cells(tmp_path,mismatch):
    import json
    from fish_viewer.proposals.consensus import apply
    image=tmp_path/'scan.ims'
    reports=[]
    for index in range(2):
        folder=tmp_path/str(index);folder.mkdir()
        labels=np.zeros((24,)*3,np.uint32);labels[3:11,3:11,3:11]=7
        if not index:labels[14:20,14:20,14:20]=12
        metadata=dict(image=str(image),fingerprint='same',channel=0,time=0,
            spacing_zyx=[.2]*3,origin_xyz=[0]*3,cells=[dict(label=7),dict(label=12)] if not index else [dict(label=7)],metrics={'stale':True})
        if index and mismatch:metadata[mismatch]='different' if mismatch=='fingerprint' else 1
        path=folder/'contours.npz';np.savez_compressed(path,labels=labels,metadata=np.array(json.dumps(metadata)))
        report=folder/'report.json';report.write_text(json.dumps(dict(status='complete',files=[dict(image=str(image),predictions=str(path))])))
        reports.append(report)
    result=apply(*reports,tmp_path/'output')
    if mismatch:
        assert result['status']=='failed' and 'source mismatch' in result['error']
        assert not list((tmp_path/'output').rglob('contours.npz'))
    else:
        assert result['status']=='complete' and result['summary']['cells']==1
        with np.load(result['files'][0]['predictions']) as f:
            assert set(np.unique(f['labels']))=={0,7}
            metadata=json.loads(str(f['metadata']))
            assert metadata['cells']==[dict(label=7)] and 'metrics' not in metadata
            assert metadata['consensus']['matches'][0]['iou']==1
