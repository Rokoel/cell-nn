import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from summarize_all_heldout import aggregate


def dataset(tmp_path):
    image=str(tmp_path/'scan.ims');case=dict(image=image,audit=dict(split='test',reference_cut=[False,True],channel=0))
    (tmp_path/'selection.json').write_text(json.dumps(dict(cases=[case])))
    (tmp_path/'identity.json').write_text('{}')
    for mode,volume,distance in [('default',9.,.8),('hybrid',11.,1.2),('consensus',12.,.9)]:
        folder=tmp_path/'001-scan'/mode;folder.mkdir(parents=True)
        row=dict(image=image,scan='scan.ims',mode=mode,split='test',group='g1',references=2,complete_references=1,
            cells_before_filter=2,final_cells=2,signals=2,matched_before_filter=2,matched_final=2,
            unscored_before_filter=0,unscored_final=0,
            matches_before_filter=[dict(reference_id=0),dict(reference_id=1)],matches_final=[dict(reference_id=0),dict(reference_id=1)],
            volume_pairs=[dict(reference_id=0,manual_um3=10.,predicted_um3=volume,cut=False),dict(reference_id=1,manual_um3=20.,predicted_um3=5.,cut=True)],
            distance_pairs=[dict(reference_id=0,channel=0,manual_xyz=[0,0,0],manual_um=1.,predicted_um=distance),
                            dict(reference_id=1,channel=0,manual_xyz=[1,0,0],manual_um=2.,predicted_um=3.)])
        (folder/'metrics.json').write_text(json.dumps(row));(folder/'pipeline-result.json').write_text('{"status":"complete"}')
    return tmp_path


def test_complete_cell_accuracy_does_not_include_truncated_volume(tmp_path):
    report=aggregate(dataset(tmp_path));summary=report['summary']['test']
    assert report['completed_mode_runs']==3 and report['expected_scans']==1
    default=summary['modes']['default']
    assert default['matched_complete_final']==1 and default['matched_cut_final']==1
    assert default['volume_pairs']['mape_percent']==pytest.approx(42.5)
    assert default['volume_pairs_complete_cells']['mape_percent']==pytest.approx(10.)
    assert default['distance_pairs_complete_cells']['mae']==pytest.approx(.2)
    assert summary['common_matches_complete_cells']['volume_pairs']['default']['n']==1
    assert summary['common_matches_complete_cells']['volume_pairs']['consensus']['mape_percent']==pytest.approx(20.)


def test_full_report_cannot_be_published_when_a_mode_is_missing(tmp_path):
    root=dataset(tmp_path);(root/'001-scan/consensus/metrics.json').unlink()
    with pytest.raises(FileNotFoundError):aggregate(root)
    assert not (root/'report.json').exists()


def test_consensus_channel_choice_uses_stain_and_unique_metadata_hint():
    from benchmark_consensus_channels import channel_for

    case = dict(image='scan.ims', audit=dict(channel=2,
        channel_selection=dict(stain_candidates=[1, 2]),
        channel_scores=[dict(channel=0, metadata_hint=False), dict(channel=1, metadata_hint=True)]))
    assert channel_for(case)[0] == 2
    case['audit']['channel_selection']['stain_candidates'] = []
    assert channel_for(case)[0] == 1
    case['audit']['channel_scores'].append(dict(channel=2, metadata_hint=True))
    with pytest.raises(ValueError, match='Ambiguous'):
        channel_for(case)


def test_corrected_consensus_summary_excludes_cut_cells_and_zero_denominators(tmp_path):
    from benchmark_consensus_channels import summarize

    root = dataset(tmp_path)
    row = json.loads((root/'001-scan/consensus/metrics.json').read_text())
    row['cut_references'] = [False, True]
    row['distance_pairs'][0]['manual_um'] = 0
    result = summarize([row])['test']
    assert result['scans'] == 1
    assert result['matched_complete_final'] == 1
    assert result['volume']['n'] == 1
    assert result['volume']['mape_percent'] == pytest.approx(20)
    assert result['distance']['n'] == 1
    assert result['distance']['mae'] == pytest.approx(.9)
    assert result['distance_relative']['n'] == 0


def test_corrected_consensus_publish_requires_every_scan(tmp_path):
    from benchmark_consensus_channels import publish

    root = dataset(tmp_path)
    with pytest.raises(FileNotFoundError):
        publish(root, tmp_path/'published')
    assert not (tmp_path/'published/summary.json').exists()


def test_corrected_consensus_publish_checks_actual_worker_channel(tmp_path):
    import numpy as np
    from benchmark_consensus_channels import publish

    root = dataset(tmp_path)
    folder = root/'001-scan'
    row = json.loads((folder/'consensus/metrics.json').read_text())
    row.update(cell_channel=1, channel_policy='verified', cut_references=[False, True])
    (folder/'metrics.json').write_text(json.dumps(row))
    predictions = folder/'labels.npz'
    np.savez(predictions, metadata=np.array(json.dumps(dict(channel=1, consensus=dict(matches=1)))))
    measured = folder/'measured.json'
    measured.write_text(json.dumps(dict(membrane_channel=1, unfiltered_cell_predictions=str(predictions))))
    (folder/'pipeline-result.json').write_text(json.dumps(dict(status='complete', report_path=str(folder/'pipeline.json'), signal_report=str(measured))))
    (root/'identity.json').write_text(json.dumps(dict(channels={row['image']: [1, 'verified']}, minimum_roundness=.72)))
    report = publish(root, root/'published')
    assert report['summary']['test']['volume']['mape_percent'] == pytest.approx(20)
    np.savez(predictions, metadata=np.array(json.dumps(dict(channel=0, consensus=dict(matches=1)))))
    with pytest.raises(ValueError, match='requested membrane channel'):
        publish(root, root/'incorrect')
    assert not (root/'incorrect/summary.json').exists()
