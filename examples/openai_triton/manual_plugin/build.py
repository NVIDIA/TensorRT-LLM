# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import math
import time
from pathlib import Path

import tensorrt as trt
from plugin import LAYER_NAME, FmhaLayer, get_engine_name

import tensorrt_llm
from tensorrt_llm.builder import Builder, BuilderConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.network import net_guard


def build_engine(builder: Builder, builder_config: BuilderConfig,
                 engine_name: str, args: argparse.Namespace) -> trt.IHostMemory:
    '''

    @brief: Build a TensorRT engine.
    @param args: The cmd line arguments.
    @return: The built or refitted engine.
    '''

    # Initialize Module
    softmax_scale = 1.0 / math.sqrt(args.head_size)
    layer = FmhaLayer(args.num_heads, args.head_size, softmax_scale, args.dtype)

    # Module -> Network
    network = builder.create_network()
    network.trt_network.name = engine_name
    with net_guard(network):
        # Prepare
        inputs = layer.prepare_inputs(args.max_batch_size, args.max_seq_len)
        # Forward
        logger.debug(f'model inputs: {inputs}')
        out = layer(*inputs)
        out.trt_tensor.name = 'out'

    # Network -> Engine
    engine = builder.build_engine(network, builder_config)
    config_path = Path(args.output_dir) / 'config.json'
    builder.save_config(builder_config, str(config_path))
    return engine


def build(args):
    tensorrt_llm.logger.set_level(args.log_level)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    builder = Builder()
    cache = None
    builder_config = builder.create_builder_config(
        name=LAYER_NAME,
        precision=args.dtype,
        timing_cache=args.timing_cache if cache is None else cache,
        profiling_verbosity=args.profiling_verbosity)

    engine_name = get_engine_name(args.head_size, args.dtype)
    engine = build_engine(builder, builder_config, engine_name, args)
    assert engine is not None

    engine_path = output_dir / engine_name
    logger.info(f'Serializing engine to {str(engine_path)}...')
    tik = time.time()
    with engine_path.open('wb') as f:
        f.write(engine)
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')

    ok = builder.save_timing_cache(builder_config,
                                   Path(args.output_dir) / "model.cache")
    assert ok, "Failed to save timing cache."


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_batch_size', type=int, default=4)
    parser.add_argument('--max_seq_len', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--head_size', type=int, default=64)
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'float32'])
    parser.add_argument(
        '--timing_cache',
        type=str,
        default='model.cache',
        help='The path of to read timing cache from, will be ignored '
        'if the file does not exist')
    parser.add_argument(
        '--profiling_verbosity',
        type=str,
        default='layer_names_only',
        choices=['layer_names_only', 'detailed', 'none'],
        help=
        'The profiling verbosity for the generated TRT engine. Set to detailed can inspect tactic choices and kernel parameters.'
    )
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs',
        help='The path to save the serialized engine files, timing cache '
        'file and model configs')
    args = parser.parse_args()

    logger.set_level(args.log_level)
    logger.info('Parameters'.center(40, '='))
    for k, v in vars(args).items():
        logger.info(f' - {k.ljust(15, ".")}: {v}')
    logger.info(''.center(40, '='))

    tik = time.time()
    logger.info('Build TensorRT engine.')
    build(args)
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Total time of building TRT engine: {t}')
