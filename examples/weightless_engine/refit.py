'''
Script that refits TRT-LLM engine(s) with weights in a TRT-LLM checkpoint.
'''
import argparse
import copy
import json
import os
import re
import shutil
from pathlib import Path

import safetensors
import tensorrt as trt

from tensorrt_llm.commands.build import preprocess_weights
from tensorrt_llm.models import PretrainedConfig

ENGINE_RE = re.compile('rank(\d+).engine')
LOGGER = trt.Logger(trt.ILogger.Severity.INFO)


def refit_engine(engine_path: str, refit_engine_dir: str, checkpoint_dir: str,
                 model_config: PretrainedConfig):
    # This function loops through all weights in the checkpoint and does a textual match between
    # checkpoint weight names and engine weight names.
    rank = int(ENGINE_RE.fullmatch(os.path.basename(engine_path)).group(1))
    ckpt_path = os.path.join(checkpoint_dir, f'rank{rank}.safetensors')
    if not Path(ckpt_path).exists():
        raise RuntimeError(
            f'Could not find checkpoint file corresponding to `{engine_path}`. Does `{ckpt_path} exist?'
        )
    with open(engine_path, "rb") as f, trt.Runtime(LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    refitter = trt.Refitter(engine, LOGGER)
    refittable_weights = set(refitter.get_all_weights())
    weights = {}
    with safetensors.safe_open(ckpt_path, framework='pt', device='cpu') as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    if weights is None:
        raise RuntimeError(f'No weights found. Quitting.')

    rank_config = copy.deepcopy(model_config)
    rank_config.set_rank(rank)
    preprocess_weights(weights, rank_config)

    for key, weight in weights.items():
        # Refit all possible weights. This should be a superset of the weights we need to refit.
        if not key in refittable_weights:
            continue
        assert refitter.set_named_weights(
            key, weight.numpy()), f'Failed to refit weight: `{key}`'

    assert refitter.refit_cuda_engine(), f'Failed to refit engine.'
    refit_engine_path = os.path.join(refit_engine_dir,
                                     os.path.basename(engine_path))
    with open(refit_engine_path, 'wb') as f:
        print(f'\nWriting refitted engine to `{refit_engine_path}`')
        s_config = engine.create_serialization_config()
        s_config.flags &= ~(1 << int(trt.SerializationFlag.EXCLUDE_WEIGHTS))
        f.write(engine.serialize_with_config(s_config))

    del refitter


def refit(engine_dir: str, checkpoint_dir: str, model_config: PretrainedConfig):
    refit_engine_dir = engine_dir + '.refit'
    os.makedirs(refit_engine_dir, exist_ok=True)
    shutil.copyfile(os.path.join(engine_dir, 'config.json'),
                    os.path.join(refit_engine_dir, 'config.json'))
    engine_paths = list(Path(engine_dir).glob('*.engine'))
    for path in engine_paths:
        refit_engine(path, refit_engine_dir, checkpoint_dir, model_config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--engine_dir',
        type=str,
        default=None,
        help=
        'Path to trt-llm engines. These engines must have been built from a pruned checkpoint, or otherwise be refittable.'
    )
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default=None,
                        help='Path to checkpoint containing desired weights')
    args = parser.parse_args()

    if args.engine_dir is None or not Path(args.engine_dir).exists():
        raise RuntimeError(
            f'Please supply a valid --engine_dir (found `{args.engine_dir}`)')
    if args.checkpoint_dir is None or not Path(args.checkpoint_dir).exists():
        raise RuntimeError(
            f'Please supply a valid --checkpoint_dir (found `{args.checkpoint_dir}`)'
        )

    with open(os.path.join(args.engine_dir, 'config.json'), 'r') as f:
        engine_config = json.load(f)

    with open(os.path.join(args.checkpoint_dir, 'config.json'), 'r') as f:
        checkpoint_config = json.load(f)

    engine_arch = engine_config['pretrained_config']['architecture']
    checkpoint_arch = checkpoint_config['architecture']
    model_config = PretrainedConfig.from_dict(
        engine_config['pretrained_config'])
    if engine_arch != checkpoint_arch:
        raise RuntimeError(
            f'Engine Architecture and Checkpoint Architecture do not match. ' +
            f'Engine Architecture: `{engine_arch}`, Checkpoint Architecture: `{checkpoint_arch}`'
        )

    refit(os.path.normpath(args.engine_dir),
          os.path.normpath(args.checkpoint_dir), model_config)


if __name__ == '__main__':
    main()
