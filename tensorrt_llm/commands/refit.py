'''
Script that refits TRT-LLM engine(s) with weights in a TRT-LLM checkpoint.
'''
import argparse
import copy
import json
import os
import re
import shutil
import time
from pathlib import Path

import tensorrt as trt

from tensorrt_llm._common import _is_building
from tensorrt_llm._utils import np_dtype_to_trt
from tensorrt_llm.builder import EngineConfig, optimize_model_with_config
from tensorrt_llm.models import MODEL_MAP, PretrainedConfig

from ..logger import logger

ENGINE_RE = re.compile('rank(\d+).engine')


@_is_building
def refit_engine(engine_path: str, refit_engine_dir: str, checkpoint_dir: str,
                 engine_config: EngineConfig, fixed_weights_names: list):
    # This function loops through all weights in the model and does a textual match between
    # checkpoint weight names and engine weight names.
    rank = int(ENGINE_RE.fullmatch(os.path.basename(engine_path)).group(1))
    tik = time.time()
    with open(engine_path,
              "rb") as f, trt.Runtime(logger.trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Load TRT engine time: {t}')

    refitter = trt.Refitter(engine, logger.trt_logger)
    refittable_weights = set(refitter.get_all_weights())

    # Load model.
    tik = time.time()
    rank_config = PretrainedConfig.from_dict(
        engine_config.pretrained_config.to_dict())
    rank_config.set_rank(rank)

    architecture = rank_config.architecture
    assert architecture in MODEL_MAP, \
        f"Unsupported model architecture: {architecture}"
    model_cls = MODEL_MAP[architecture]
    model = model_cls.from_checkpoint(checkpoint_dir, config=rank_config)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Load checkpoint(s) time: {t}')

    # There are weights preprocess during optimize model.
    tik = time.time()
    build_config = copy.deepcopy(engine_config.build_config)
    optimize_model_with_config(model, build_config)
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Preprocess weights time: {t}')

    # Refit engine.
    tik = time.time()
    refitted_weights = []
    for name, buf in model.named_parameters():
        if name in refittable_weights:
            assert buf.is_inited, f"Failed because weight `{name}` is not initialized in model."
            weight = buf._value
            weights_value = trt.Weights(np_dtype_to_trt(weight.dtype),
                                        weight.ctypes.data, weight.size)
            assert refitter.set_named_weights(
                name, weights_value), f'Failed to refit weight: `{name}`'
            refitted_weights.append(name)
        else:
            if name not in fixed_weights_names:
                logger.warning(
                    f"model weights `{name}` (shape: {buf._value.shape}) is not refittable, this means that we might not be able to update the engine using fine-tuned checkpoint!"
                )

    # Validate all refittable weights are provided.
    if len(refitted_weights) != len(refittable_weights):
        raise RuntimeError(
            f'Missing refittable weights {refittable_weights.difference(refitted_weights)} from {checkpoint_dir}'
        )

    assert refitter.refit_cuda_engine(), f'Failed to refit engine.'
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Execute GPU refit graph time: {t}')

    tik = time.time()
    refit_engine_path = os.path.join(refit_engine_dir,
                                     os.path.basename(engine_path))
    with open(refit_engine_path, 'wb') as f:
        logger.info(f'\nWriting refitted engine to `{refit_engine_path}`')
        s_config = engine.create_serialization_config()
        s_config.flags &= ~(1 << int(trt.SerializationFlag.EXCLUDE_WEIGHTS))
        f.write(engine.serialize_with_config(s_config))
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Write TRT engine to disk time: {t}')

    del refitter


def refit(engine_dir: str, checkpoint_dir: str, engine_config: EngineConfig,
          output_dir: str, fixed_weights_names: list):
    refit_engine_dir = output_dir
    os.makedirs(refit_engine_dir, exist_ok=True)
    shutil.copyfile(os.path.join(engine_dir, 'config.json'),
                    os.path.join(refit_engine_dir, 'config.json'))
    engine_paths = list(Path(engine_dir).glob('*.engine'))
    for path in engine_paths:
        refit_engine(path, refit_engine_dir, checkpoint_dir, engine_config,
                     fixed_weights_names)


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
    parser.add_argument('--output_dir',
                        type=str,
                        default=None,
                        help="Output path of the refit model")
    parser.add_argument('--log_level', type=str, default='info')
    args = parser.parse_args()

    logger.set_level(args.log_level)
    if args.engine_dir is None or not Path(args.engine_dir).exists():
        raise RuntimeError(
            f'Please supply a valid --engine_dir (found `{args.engine_dir}`)')
    if args.checkpoint_dir is None or not Path(args.checkpoint_dir).exists():
        raise RuntimeError(
            f'Please supply a valid --checkpoint_dir (found `{args.checkpoint_dir}`)'
        )

    engine_config = EngineConfig.from_json_file(
        os.path.join(args.engine_dir, 'config.json'))

    with open(os.path.join(args.checkpoint_dir, 'config.json'), 'r') as f:
        checkpoint_config = json.load(f)

    engine_arch = engine_config.pretrained_config.architecture
    checkpoint_arch = checkpoint_config['architecture']
    if engine_arch != checkpoint_arch:
        raise RuntimeError(
            f'Engine Architecture and Checkpoint Architecture do not match. ' +
            f'Engine Architecture: `{engine_arch}`, Checkpoint Architecture: `{checkpoint_arch}`'
        )

    # The fixed weights are not read from checkpoint, they are hardcoded buffer from the model itself. These values remain constant across different fine-tuned checkpoints.
    fixed_wts_in_model = []
    model_cls = MODEL_MAP[engine_arch]
    model = model_cls.from_config(engine_config.pretrained_config)
    for name, param in model.named_parameters():
        if param.is_inited():
            fixed_wts_in_model.append(name)

    refit(engine_dir=os.path.normpath(args.engine_dir),
          checkpoint_dir=os.path.normpath(args.checkpoint_dir),
          engine_config=engine_config,
          output_dir=os.path.normpath(args.output_dir),
          fixed_weights_names=fixed_wts_in_model)


if __name__ == '__main__':
    main()
