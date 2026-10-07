"""Exercise a real U-Net gradient and checkpoint round-trip on CPU or GPU."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import sysconfig
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    if sys.platform.startswith('linux') and not os.environ.get('FISH_CUDA_PATH_READY'):
        libraries=sorted((Path(sysconfig.get_paths()['purelib'])/'nvidia').glob('*/lib'))
        if libraries:
            env=os.environ.copy();env['FISH_CUDA_PATH_READY']='1'
            env['LD_LIBRARY_PATH']=os.pathsep.join(map(str,libraries))+os.pathsep+env.get('LD_LIBRARY_PATH','')
            os.execve(sys.executable,[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],env)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['auto','cpu','gpu'],default='cpu')
    parser.add_argument('--size',type=int,default=32)
    args=parser.parse_args()
    from fish_viewer.neural.engine import runtime
    from fish_viewer.refinement.config import RefinementConfig
    from fish_viewer.refinement.network import network,loss
    import numpy as np
    tf,_,_,info=runtime(args.device)
    config=replace(RefinementConfig(),patch_size=args.size)
    if args.device=='gpu':tf.keras.mixed_precision.set_global_policy('mixed_float16')
    model=network(tf,config);optimizer=tf.keras.optimizers.Adam(.001)
    if args.device=='gpu':optimizer=tf.keras.mixed_precision.LossScaleOptimizer(optimizer,initial_scale=128.)
    grid=np.indices((config.patch_size,)*3)
    radius=np.linalg.norm(grid-(config.patch_size-1)/2,axis=0)
    x=np.stack([np.exp(-((radius-config.patch_size/5)/1.5)**2),
                np.exp(-radius**2/50)],axis=-1).astype(np.float32)[None]
    y=np.stack([radius<config.patch_size/5,np.ones_like(radius),np.ones_like(radius)],axis=-1).astype(np.float32)[None]
    x=np.repeat(x,2,axis=0);y=np.repeat(y,2,axis=0)
    @tf.function(jit_compile=False)
    def step():
        with tf.GradientTape() as tape:
            value,dice=loss(tf,y,model(x,training=True))
            scaled=optimizer.scale_loss(value) if hasattr(optimizer,'scale_loss') else value
        gradients=tape.gradient(scaled,model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        return value,dice
    original_weights=[v.numpy().copy() for v in model.trainable_variables]
    start=time.monotonic();results=[]
    for i in range(10):
        s=time.monotonic();result=[float(v.numpy()) for v in step()]
        assert np.isfinite(result).all()
        results.append(dict(step=i+1,seconds=time.monotonic()-s,loss=result[0],dice=result[1]))
        print(json.dumps(results[-1]),flush=True)
    assert any(not np.array_equal(a,v.numpy()) for a,v in zip(original_weights,model.trainable_variables)), 'No weight updates occurred.'
    assert results[-1]['loss']<results[0]['loss'], 'Synthetic training did not reduce loss.'
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'best.weights.h5';model.save_weights(path)
        expected=model(x,training=False).numpy()
        restored=network(tf,config);restored.load_weights(path)
        np.testing.assert_allclose(expected,restored(x,training=False).numpy(),atol=1e-5)
    print(json.dumps(dict(status='complete',seconds=time.monotonic()-start,runtime=info,steps=results)))


if __name__=='__main__':main()
