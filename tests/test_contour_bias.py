from dataclasses import replace
import numpy as np
import pytest
from scipy import ndimage as ndi
from fish_viewer.refinement.config import RefinementConfig
from fish_viewer.refinement.training import batch,validation_boundaries


def test_surface_targets_follow_augmented_geometry_and_physical_spacing():
    size=32
    image=np.zeros((1,size,size,size),np.float32)
    mask=np.zeros_like(image);mask[:,8:24,8:24,8:24]=1
    arrays=dict(images=image,masks=mask,weights=np.ones_like(image),valid=np.ones_like(image))
    records=[dict(seed_patch=[15.5]*3,patch_spacing_um=.2)]
    config=replace(RefinementConfig(),patch_size=size,surface_loss_weight=.1,crop_scale_jitter=.1,crop_shift_um=.2)
    _,y=batch(arrays,records,[0],config,np.random.default_rng(2),augment=True)
    truth=y[0,...,0]>.5
    expected=(ndi.distance_transform_edt(~truth,sampling=.2)-ndi.distance_transform_edt(truth,sampling=.2))/2
    assert y.shape[-1]==4
    assert np.allclose(y[0,...,3],np.clip(expected,-1,1))


@pytest.mark.neural
def test_surface_loss_and_asymmetric_loss_ignore_unknown_and_penalize_missing_voxels():
    import tensorflow as tf
    from fish_viewer.refinement.network import loss
    truth=np.zeros((1,8,8,8,4),np.float32)
    truth[...,1]=1;truth[:,2:6,2:6,2:6,2]=1
    truth[:,2:6,2:6,2:6,0]=1;truth[...,3]=1;truth[...,3][truth[...,0]>0]=-1
    logits=tf.Variable(tf.zeros((1,8,8,8,1)))
    config=replace(RefinementConfig(),surface_loss_weight=.1,tversky_beta=.6)
    with tf.GradientTape() as tape:value,_=loss(tf,tf.constant(truth),logits,config)
    gradient=tape.gradient(value,logits).numpy()
    assert not gradient[:,:2].any()
    assert gradient[0,3,3,3,0]<0
    # Shift logits inside marked target: covering it should lower the objective.
    grown=np.zeros(logits.shape,np.float32);grown[:,2:6,2:6,2:6]=2
    assert float(loss(tf,truth,grown,config)[0])<float(value)


def test_validation_boundary_scores_empty_prediction_as_failure():
    truth=np.zeros((1,16,16,16,4),np.float32);truth[:,4:12,4:12,4:12,0]=1;truth[...,2]=1
    record=[dict(seed_patch=[8,8,8],patch_spacing_um=.2)]
    config=RefinementConfig()
    good=validation_boundaries(truth[...,0:1],truth,[0],record,config)[0]
    bad=validation_boundaries(np.zeros((1,16,16,16,1)),truth,[0],record,config)[0]
    assert good==(0.,0.,0.)
    assert bad[0]>1 and bad[1]==100


def test_invalid_new_loss_config_is_rejected():
    with pytest.raises(ValueError):replace(RefinementConfig(),tversky_beta=.9).validate()
    with pytest.raises(ValueError):replace(RefinementConfig(),surface_loss_weight=-.1).validate()


@pytest.mark.neural
def test_residual_network_starts_identical_and_freezes_original(tmp_path):
    import tensorflow as tf
    from fish_viewer.refinement.network import network
    config=replace(RefinementConfig(),patch_size=32,filters=4,depth=2)
    original=network(tf,config)
    path=tmp_path/'original.weights.h5';original.save_weights(path)
    correction=network(tf,replace(config,residual_correction=True))
    frozen=correction.get_layer('frozen_contour');frozen.load_weights(path)
    x=np.random.default_rng(1).normal(size=(1,32,32,32,2)).astype(np.float32)
    assert np.allclose(original(x).numpy(),correction(x).numpy(),atol=1e-6)
    assert not frozen.trainable and not frozen.trainable_variables
    saved=tmp_path/'residual.weights.h5';correction.save_weights(saved)
    restored=network(tf,replace(config,residual_correction=True));restored.load_weights(saved)
    assert np.allclose(correction(x).numpy(),restored(x).numpy())


def test_fixed_cell_guard_retains_original_when_no_membrane_is_visible():
    from scripts.experiment_contour_bias import guarded_mask
    old=np.zeros((24,)*3,bool);old[8:16,8:16,8:16]=True
    scores=np.zeros(old.shape,np.float32);scores[5:19,5:19,5:19]=.9
    mask,reason=guarded_mask(scores,.45,np.array([12,12,12]),old,np.zeros_like(old),np.zeros(old.shape),np.array([.2]*3))
    assert reason=='Insufficient membrane evidence'
    assert np.array_equal(mask,old)


def test_fixed_cell_application_preserves_ids_and_volumes_on_rejected_edit(tmp_path,monkeypatch):
    import json,sys
    from dataclasses import asdict
    from scripts import experiment_contour_bias as experiment
    from fish_viewer.neural.audit import save_references
    from fish_viewer.neural import engine
    from fish_viewer.refinement import network as network_module
    model=tmp_path/'model';model.mkdir()
    (model/'best.weights.h5').write_bytes(b'fake')
    (model/'refinement-model.json').write_text(json.dumps(dict(version=1,engine='seeded-unet3d',checkpoint='best.weights.h5',config=asdict(RefinementConfig()))))
    labels=np.zeros((24,)*3,np.uint32);labels[7:17,7:17,7:17]=3
    cache=tmp_path/'cache';cache.mkdir();np.save(cache/'image.npy',np.zeros(labels.shape,np.float32))
    box=(slice(7,17),)*3;save_references(cache/'references.npz',[(box,np.ones((10,)*3,bool))],['manual'])
    image=str(tmp_path/'scan.ims')
    info=dict(version=1,image=image,spacing_zyx=[.3]*3,origin_xyz=[0]*3,channel=0,channel_scores=[],fingerprint='same',cells=[dict(label=3,volume_um3=27.)])
    archive=tmp_path/'original.npz';np.savez(archive,labels=labels,metadata=np.array(json.dumps(info)))
    audit=tmp_path/'audit.json';audit.write_text(json.dumps(dict(cases=[dict(image=image,directory=str(cache),audit=dict(split='validation',group='one',reference_cut=[False]))])))
    baseline=tmp_path/'baseline.json';baseline.write_text(json.dumps(dict(status='complete',files=[dict(image=image,predictions=str(archive),proposals=[dict(cell_id=3,accepted=True,center_xyz=[3.5]*3,box_low_xyz=[2.1]*3,box_high_xyz=[5.1]*3)],metrics=dict(per_cell=[dict(predicted_label=3,iou=1.)]))])))
    class EmptyModel:
        def load_weights(self,path):pass
        def __call__(self,x,training=False):return np.full((1,64,64,64,1),-20,np.float32)
    monkeypatch.setattr(engine,'runtime',lambda *a:(None,None,None,{}))
    monkeypatch.setattr(network_module,'network',lambda *a:EmptyModel())
    monkeypatch.setenv('FISH_CUDA_PATH_READY','1')
    output=tmp_path/'result'
    monkeypatch.setattr(sys,'argv',['experiment','--audit',str(audit),'--baseline',str(baseline),'--model',str(model),'--output',str(output),'--mode','apply'])
    experiment.main()
    report=json.loads((output/'report.json').read_text());assert report['status']=='complete'
    with np.load(report['files'][0]['predictions']) as result:
        assert np.array_equal(result['labels'],labels)
        metadata=json.loads(str(result['metadata']))
    assert metadata['cells'][0]['label']==3
    assert metadata['cells'][0]['volume_um3']==pytest.approx(27.)
    assert not metadata['refinement']['cells'][0]['accepted']


def test_overlap_removal_does_not_accept_small_fragments():
    from scripts.experiment_contour_bias import guarded_mask
    scores=np.zeros((24,)*3,np.float32);scores[7:16,7:16,7:16]=.9
    old=np.zeros(scores.shape,bool);old[2:11,2:11,2:11]=True
    neighbors=np.zeros_like(old);neighbors[13:16,7:16,7:16]=True
    mask,reason=guarded_mask(scores,.45,np.array([8]*3),old,neighbors,np.ones(scores.shape),np.array([.2]*3))
    assert reason=='Too small after neighbor protection'
    assert np.array_equal(mask,old)
