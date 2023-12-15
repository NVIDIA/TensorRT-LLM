# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import time
from concurrent.futures import ProcessPoolExecutor, wait
from importlib.machinery import SourceFileLoader
from multiprocessing import get_context
from typing import Union

import torch

from ..builder import BuildConfig, build
from ..logger import logger
from ..models import PretrainedConfig


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--model_config', type=str, default=None)
    parser.add_argument('--build_config', type=str, default=None)
    parser.add_argument('--model_cls_file', type=str, default=None)
    parser.add_argument('--model_cls_name', type=str, default=None)
    parser.add_argument(
        '--timing_cache',
        type=str,
        default='model.cache',
        help=
        'The path of to read timing cache from, will be ignored if the file does not exist'
    )
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='engine_outputs',
        help=
        'The path to save the serialized engine files, timing cache file and model configs'
    )
    parser.add_argument('--workers',
                        type=int,
                        default='1',
                        help='The number of workers for building in parallel')
    parser.add_argument('--max_batch_size', type=int, default=1)
    parser.add_argument('--max_input_len', type=int, default=1024)
    parser.add_argument('--max_output_len', type=int, default=1024)
    parser.add_argument('--max_beam_width', type=int, default=1)
    parser.add_argument('--max_num_tokens', type=int, default=None)
    parser.add_argument(
        '--max_prompt_embedding_table_size',
        type=int,
        default=0,
        help='Setting to a value > 0 enables support for prompt tuning.')
    parser.add_argument('--use_gpt_attention_plugin',
                        nargs='?',
                        const='float16',
                        type=str,
                        default=False,
                        choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--use_gemm_plugin',
                        nargs='?',
                        const='float16',
                        type=str,
                        default=False,
                        choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--use_lookup_plugin',
                        nargs='?',
                        const='float16',
                        type=str,
                        default=False,
                        choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--enable_context_fmha',
                        default=False,
                        action='store_true')
    parser.add_argument('--enable_context_fmha_fp32_acc',
                        default=False,
                        action='store_true')
    parser.add_argument(
        '--multi_block_mode',
        default=False,
        action='store_true',
        help=
        'Split long kv sequence into multiple blocks (applied to generation MHA kernels). \
                        It is beneifical when batchxnum_heads cannot fully utilize GPU.'
    )
    parser.add_argument('--remove_input_padding',
                        default=False,
                        action='store_true')
    parser.add_argument(
        '--paged_kv_cache',
        action="store_true",
        default=False,
        help=
        'By default we use contiguous KV cache. By setting this flag you enable paged KV cache'
    )
    parser.add_argument('--tokens_per_block',
                        type=int,
                        default=64,
                        help='Number of tokens per block in paged KV cache')
    parser.add_argument(
        '--use_custom_all_reduce',
        action='store_true',
        help=
        'Activates latency-optimized algorithm for all-reduce instead of NCCL.')
    parser.add_argument('--gather_all_token_logits',
                        action='store_true',
                        default=False)

    args = parser.parse_args()

    return args


def build_and_save_shard(rank, gpu_id, ckpt_dir, build_config, output_dir,
                         log_level, model_config, model_cls):
    torch.cuda.set_device(gpu_id)
    logger.set_level(log_level)
    engine = build(build_config,
                   rank,
                   ckpt_dir,
                   model_config,
                   model_cls=model_cls)
    engine.save(output_dir)


def build_and_save(ckpt_dir_or_model_config: str,
                   build_config: Union[str, BuildConfig],
                   output_dir: str,
                   workers: int = 1,
                   log_level: str = 'info',
                   model_cls=None):
    ckpt_dir = ckpt_dir_or_model_config
    if ckpt_dir_or_model_config.lower().endswith('.json'):
        model_config = PretrainedConfig.from_json_file(ckpt_dir_or_model_config)
        ckpt_dir = None
    else:
        model_config = PretrainedConfig.from_json_file(
            os.path.join(ckpt_dir_or_model_config, 'config.json'))

    if workers == 1:
        for rank in range(model_config.mapping.world_size):
            build_and_save_shard(rank, rank % workers, ckpt_dir, build_config,
                                 output_dir, log_level, model_config, model_cls)
    else:
        with ProcessPoolExecutor(mp_context=get_context('spawn'),
                                 max_workers=workers) as p:
            futures = [
                p.submit(build_and_save_shard, rank, rank % workers, ckpt_dir,
                         build_config, output_dir, log_level, model_config,
                         model_cls)
                for rank in range(model_config.mapping.world_size)
            ]
            wait(futures)


def main():
    args = parse_arguments()
    logger.set_level(args.log_level)
    tik = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model_cls = None
    if args.model_cls_file is not None:
        assert args.model_cls_name is not None
        loader = SourceFileLoader('models', args.model_cls_file)
        mod = loader.load_module()
        model_cls = getattr(mod, args.model_cls_name)

    workers = min(torch.cuda.device_count(), args.workers)

    build_config = args.build_config
    if args.build_config is None:
        build_config = BuildConfig.from_dict({
            'max_input_len':
            args.max_input_len,
            'max_output_len':
            args.max_output_len,
            'max_batch_size':
            args.max_batch_size,
            'max_beam_width':
            args.max_beam_width,
            'max_num_tokens':
            args.max_num_tokens,
            'max_prompt_embedding_table_size':
            args.max_prompt_embedding_table_size,
            'gather_all_token_logits':
            args.gather_all_token_logits,
            'plugin_config': {
                'gpt_attention_plugin': args.use_gpt_attention_plugin,
                'gemm_plugin': args.use_gemm_plugin,
                'enable_context_fmha': args.enable_context_fmha,
                'enable_context_fmha_fp32_acc':
                args.enable_context_fmha_fp32_acc,
                'remove_input_padding': args.remove_input_padding,
                'paged_kv_cache': args.paged_kv_cache,
                'tokens_per_block': args.tokens_per_block,
                'lookup_plugin': args.use_lookup_plugin,
                'use_custom_all_reduce': args.use_custom_all_reduce,
            }
        })

    source = args.checkpoint_dir if args.checkpoint_dir is not None else args.model_config
    build_and_save(source, build_config, args.output_dir, workers,
                   args.log_level, model_cls)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Total time of building all engines: {t}')


if __name__ == '__main__':
    main()
