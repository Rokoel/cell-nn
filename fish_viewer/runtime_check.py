"""Exercise each neural backend in a separate packaged worker process."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=['tensorflow', 'torch'], required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default='auto')
    args = parser.parse_args()
    if args.engine == 'tensorflow':
        from .neural.engine import runtime
        tf, _, _, info = runtime(args.device)
        device = '/GPU:0' if info['device'] == 'GPU' else '/CPU:0'
        with tf.device(device):
            output = tf.keras.layers.Conv3D(2, 3)(tf.ones((1, 8, 8, 8, 1)))
        assert bool(tf.reduce_all(tf.math.is_finite(output)))
        info['convolution_device'] = output.device
        if args.device == 'gpu' and 'GPU' not in output.device:
            raise RuntimeError('TensorFlow convolution did not run on GPU.')
    else:
        import torch
        from .analysis.controller import default_models
        from .spots.inference import load
        model, _, _, device = load(default_models()[1], args.device)
        with torch.inference_mode():
            output = model(torch.ones((1, 3, 16, 16, 16), device=device))['heatmaps'][0]
        assert bool(torch.isfinite(output).all())
        info = dict(version=torch.__version__, device=str(output.device),
                    model='signal-center-unet3d', checksum=float(output.sum()))
    print(json.dumps(dict(engine=args.engine, status='passed', **info)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
