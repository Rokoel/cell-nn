import numpy as np
import pytest
from types import SimpleNamespace
from fish_viewer.boundary.inference import BoundaryPredictor
from fish_viewer.refinement.membrane import snap
from test_membrane_snap import sphere


def test_tiled_prediction_preserves_grid_and_probabilities():
    predictor=BoundaryPredictor.__new__(BoundaryPredictor)
    predictor.manifest={'training_xy_um':.1,'patch_size':128}
    predictor.model=lambda x,training=False:np.zeros_like(x)
    image=np.random.default_rng(3).random((2,87,141),dtype=np.float32)
    probability=predictor.volume(image,(.3,.15,.12))
    assert probability.shape==image.shape
    np.testing.assert_allclose(probability,.5,atol=1e-6)


def test_guidance_does_not_move_contours_on_blank_raw_channel():
    labels,prepared,_=sphere()
    guidance=prepared.image.copy();prepared.image.fill(0)
    result,diagnostics=snap(labels,prepared,guidance=guidance)
    np.testing.assert_array_equal(result,labels)
    assert not diagnostics[0]['accepted']


def test_invalid_guidance_rejected():
    labels,prepared,_=sphere()
    with pytest.raises(ValueError,match='probabilities'):
        snap(labels,prepared,guidance=np.full(labels.shape,np.nan))


@pytest.mark.neural
def test_unknown_pixels_have_no_loss_gradient():
    from fish_viewer.boundary.model import tensorflow,loss
    tf=tensorflow('cpu')
    labels=np.array([[[[3],[1]],[[0],[0]]]],np.uint8)
    logits=tf.Variable(np.zeros(labels.shape,np.float32))
    with tf.GradientTape() as tape:value=loss(tf,tf.constant(labels),logits)
    gradient=tape.gradient(value,logits).numpy()
    assert gradient[0,0,0,0]!=0 and gradient[0,0,1,0]!=0
    assert np.all(gradient[0,1]==0)
    with tf.GradientTape() as tape:value=loss(tf,tf.zeros_like(tf.constant(labels)),logits)
    assert float(value)==0
    assert np.all(tape.gradient(value,logits).numpy()==0)


def test_training_rejects_annotations_from_heldout_sources(tmp_path):
    import json
    from fish_viewer.boundary.training import train
    source=tmp_path/'heldout.ims'
    folder=tmp_path/'annotations/crop-1';folder.mkdir(parents=True)
    (folder/'metadata.json').write_text(json.dumps(dict(schema_version=2,source=str(source),cell_channel_index=0)))
    audit=tmp_path/'audit.json'
    audit.write_text(json.dumps({'cases':[{'image':str(source),'audit':{'split':'test','channel':0,'group':'heldout'}}]}))
    with pytest.raises(ValueError,match='training split'):
        train(folder.parent,audit,tmp_path/'output')
