import json
from pathlib import Path
import numpy as np


def network(tf):
    L = tf.keras.layers
    inputs = L.Input((None, None, 1))
    x = inputs
    skips = []
    def block(x, channels):
        for _ in range(2):
            x = L.Conv2D(channels, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
        return x
    for channels in (8, 16, 32):
        x = block(x, channels); skips.append(x); x = L.MaxPool2D()(x)
    x = block(x, 64)
    for channels, skip in zip((32, 16, 8), reversed(skips)):
        x = L.UpSampling2D()(x); x = L.Concatenate()([x, skip]); x = block(x, channels)
    return tf.keras.Model(inputs, L.Conv2D(1, 1)(x), name='sparse_boundary_unet2d')


def normalize(image):
    low, high = np.percentile(image, (1, 99.5))
    if high <= low:
        low, high = 0, max(float(np.max(image)), 1)
    return np.clip((image.astype(np.float32) - low) / max(high - low, 1e-6), 0, 1)


def loss(tf, labels, logits):
    """Unknown contributes exactly zero; balance classes within each image."""
    truth = tf.cast(labels == 3, tf.float32)
    valid = tf.cast(labels != 0, tf.float32)
    positive = truth * valid; negative = (1-truth)*valid
    axes = (1, 2, 3)
    pn = tf.reduce_sum(positive, axes, keepdims=True)
    nn = tf.reduce_sum(negative, axes, keepdims=True)
    weights = positive/tf.maximum(pn, 1) + negative/tf.maximum(nn, 1)
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=truth, logits=logits)
    ce = tf.reduce_sum(bce*weights, axes)/tf.maximum(tf.reduce_sum(weights, axes), 1e-6)
    prob = tf.math.sigmoid(logits)*valid
    dice = (2*tf.reduce_sum(prob*truth, axes)+1)/(tf.reduce_sum(prob+positive, axes)+1)
    return tf.reduce_mean(ce + .3*(1-dice)*tf.cast(tf.reshape(pn, [-1])>0, tf.float32))


def read_model(path):
    folder = Path(path)
    manifest = json.loads((folder/'boundary-model.json').read_text())
    if manifest['engine'] != 'sparse-boundary-unet2d' or not (folder/manifest['checkpoint']).is_file():
        raise ValueError('Invalid 2D boundary model.')
    return manifest


def tensorflow(device='auto', seed=42):
    import os
    if device == 'cpu': os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '4')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if device == 'gpu' and not gpus: raise RuntimeError('TensorFlow GPU unavailable.')
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.utils.set_random_seed(seed)
    return tf
