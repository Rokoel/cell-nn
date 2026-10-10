import json
from pathlib import Path
import numpy as np
import pytest
from fish_viewer.neural.slices import SlicePatches,link_sections
from fish_viewer.neural.audit import save_references


def test_slice_training_ignores_unmarked_neighbor_and_preserves_target(tmp_path):
    image=np.zeros((5,40,40),np.float32);image[:,25:30,25:30]=1
    np.save(tmp_path/'image.npy',image)
    mask=np.ones((3,8,9),bool)
    box=(slice(1,4),slice(8,16),slice(9,18))
    save_references(tmp_path/'references.npz',[(box,mask)],['manual'])
    case=dict(directory=str(tmp_path),image='scan.ims',audit=dict(training_reference_ids=[1],shape_zyx=list(image.shape)))
    patches=SlicePatches([case],size=32,sections=3)
    x,y=patches.pair(1)
    assert np.count_nonzero(y==1)==mask[1].sum()*4
    assert not np.any(y==0)
    assert np.any(x[y<0]>.5)  # Visible unmarked cell stays unknown.
    _,background=patches.pair(patches.real)
    assert np.all(background==0)


def test_linking_keeps_touching_cells_distinct_and_missing_slice_breaks_track():
    a=np.zeros((4,12,12),np.int32)
    a[0,2:6,2:6]=1;a[0,2:6,6:10]=2
    a[1,2:6,2:6]=8;a[1,2:6,6:10]=7
    a[3,2:6,2:6]=1
    out=link_sections(a)
    assert out[0,3,3]==out[1,3,3]
    assert out[0,3,7]==out[1,3,7]
    assert out[1,3,3]!=out[1,3,7]
    assert out[3,3,3]!=out[0,3,3]


def test_linking_does_not_merge_two_tracks_when_single_polygon_spans_both():
    a=np.zeros((2,12,12),np.int32)
    a[0,2:6,2:6]=1;a[0,2:6,6:10]=2
    a[1,2:6,2:10]=1
    out=link_sections(a)
    assert len(np.unique(out[0]))==3
    assert len(np.unique(out[1]))==2
    assert out[0,3,3]!=out[0,3,7]


def test_linking_validates_label_input():
    with pytest.raises(ValueError):link_sections(np.ones((2,3,3),float))
    with pytest.raises(ValueError):link_sections(np.full((2,3,3),-1,int))


@pytest.mark.neural
def test_batched_2d_inference_matches_official_padding_and_sparse_losses(tmp_path):
    import importlib.util
    if not importlib.util.find_spec('stardist'):pytest.skip('Optional StarDist unavailable')
    from fish_viewer.neural.engine import runtime
    tf,_,_,_=runtime('cpu')
    from stardist.models import Config2D,StarDist2D
    from scipy import ndimage as ndi
    from fish_viewer.neural.slices import predict_sections
    model=StarDist2D(Config2D(n_rays=16,unet_n_depth=2,unet_n_filter_base=8,grid=(2,2),
        train_background_reg=0,train_tensorboard=False),name='tiny',basedir=str(tmp_path))
    image=np.random.default_rng(42).uniform(0,1,(2,31,37)).astype('float32')
    actual=predict_sections(model,image,.5,batch_size=2)
    for z,plane in enumerate(image):
        labels,_=model.predict_instances(ndi.zoom(plane,2,order=1,prefilter=False),axes='YX',
            prob_thresh=.5,nms_thresh=.3,sparse=False,verbose=False,show_tile_progress=False)
        expected=labels[::2,::2]
        # Instance numbers are arbitrary; batched kernels can reorder equal scores.
        assert np.array_equal(actual[z]>0,expected>0)
        for number in np.unique(actual[z]):
            if number:
                corresponding=np.unique(expected[actual[z]==number])
                assert len(corresponding)==1 and corresponding[0]>0
                assert np.array_equal(actual[z]==number,expected==corresponding[0])
    model.prepare_for_training()
    probability=tf.Variable([.4,.4,.4],dtype=tf.float32)
    with tf.GradientTape() as tape:
        loss=model.keras_model.loss[0](tf.constant([-1.,0.,1.]),probability)
    gradients=tape.gradient(loss,probability).numpy()
    assert gradients[0]==0 and gradients[1]>0 and gradients[2]<0
    distances=tf.Variable(tf.ones((1,1,1,16)))
    with tf.GradientTape() as tape:
        loss=model.keras_model.loss[1](tf.zeros((1,1,1,17)),distances)
    assert np.all(tape.gradient(loss,distances).numpy()==0)
