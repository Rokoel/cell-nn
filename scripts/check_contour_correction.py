"""Exercise frozen residual correction under actual GPU mixed precision."""
import json,os,sys,sysconfig
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if not os.environ.get('FISH_CUDA_PATH_READY'):
    libs=sorted((Path(sysconfig.get_paths()['purelib'])/'nvidia').glob('*/lib'))
    env=os.environ.copy();env['FISH_CUDA_PATH_READY']='1';env['LD_LIBRARY_PATH']=os.pathsep.join(map(str,libs))+os.pathsep+env.get('LD_LIBRARY_PATH','')
    os.execve(sys.executable,[sys.executable,str(Path(__file__).resolve())],env)
import numpy as np
from fish_viewer.neural.engine import runtime
from fish_viewer.refinement.config import RefinementConfig,read_model
from fish_viewer.refinement.network import network,loss
from fish_viewer.refinement.training import batch
run=ROOT/'artifacts/contour-bias-20261009';source=Path(json.loads((run/'training-stages.json').read_text())['best_model'])
config=replace(RefinementConfig(**read_model(source)['config']),residual_correction=True,surface_loss_weight=.1,tversky_beta=.6)
tf,_,_,info=runtime('gpu');tf.keras.mixed_precision.set_global_policy('mixed_float16')
model=network(tf,config);frozen=model.get_layer('frozen_contour');frozen.load_weights(source/'best.weights.h5')
records=json.loads((run/'patches/index.json').read_text())['records']
arrays={k:np.load(run/'patches'/(k+'.npy'),mmap_mode='r') for k in ('images','masks','weights','valid')}
index=next(r['index'] for r in records if r['split']=='training')
x,y=batch(arrays,records,[index,-1],config,np.random.default_rng(1),augment=True)
initial=model(x,training=False).numpy();baseline=frozen(x,training=False).numpy()
assert np.allclose(initial,baseline,atol=1e-5)
before=[w.numpy().copy() for w in frozen.weights]
optimizer=tf.keras.mixed_precision.LossScaleOptimizer(tf.keras.optimizers.Adam(.00005),initial_scale=128)
@tf.function
def step(x,y):
    with tf.GradientTape() as tape:
        value,dice=loss(tf,y,model(x,training=True),config);scaled=optimizer.scale_loss(value)
    gradients=tape.gradient(scaled,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    return value,dice
value,dice=step(x,y)
assert np.isfinite([float(value),float(dice)]).all()
assert all(np.array_equal(a,w.numpy()) for a,w in zip(before,frozen.weights))
after=model(x,training=False).numpy();assert np.isfinite(after).all()
assert not np.array_equal(after,initial)
report=dict(status='complete',passed=True,precision='mixed_float16',output_dtype=str(after.dtype),loss=float(value),dice=float(dice),frozen_base_unchanged=True,zero_initial_correction=True,runtime=info,
            note='One real training crop plus synthetic empty crop. Preflight only; not a selected training checkpoint.')
(run/'residual-preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
