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
import copy
import os
import time
from concurrent.futures import ProcessPoolExecutor, wait
from importlib.machinery import SourceFileLoader
from multiprocessing import get_context
from typing import Union

import torch

from .._common import check_max_num_tokens
from ..builder import BuildConfig, Builder
from ..graph_rewriting import optimize
from ..logger import logger
from ..models import MODEL_MAP, PretrainedConfig, PretrainedModel
from ..network import net_guard
from ..runtime.engine import Engine, EngineConfig
from ..version import __version__


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
    parser.add_argument('--use_selective_scan_plugin',
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
    parser.add_argument(
        '--gather_all_token_logits',
        action='store_true',
        default=False,
        help='Enable both gather_context_logits and gather_generation_logits')
    parser.add_argument('--gather_context_logits',
                        action='store_true',
                        default=False,
                        help='Gather context logits')
    parser.add_argument('--gather_generation_logits',
                        action='store_true',
                        default=False,
                        help='Gather generation logits')
    parser.add_argument('--strongly_typed', action='store_true', default=False)
    parser.add_argument('--logits_dtype',
                        type=str,
                        default=None,
                        choices=['float16', 'float32'])

    args = parser.parse_args()
    if args.gather_all_token_logits:
        args.gather_context_logits = True
        args.gather_generation_logits = True

    return args


def build_model(model: PretrainedModel, build_config: BuildConfig) -> Engine:
    builder = Builder()
    builder_config = builder.create_builder_config(
        precision=model.config.dtype,
        int8=model.config.quant_mode.has_act_or_weight_quant()
        or model.config.quant_mode.has_int8_kv_cache(),
        strongly_typed=build_config.strongly_typed,
        quant_mode=model.config.quant_mode)

    network = builder.create_network()
    network._plugin_config = build_config.plugin_config

    use_weight_only = model.config.quant_mode.is_weight_only()
    per_group = model.config.quant_mode.has_per_group_scaling()
    use_smooth_quant = model.config.quant_mode.has_act_and_weight_quant()
    if use_weight_only:
        if per_group:
            network.plugin_config.set_weight_only_groupwise_quant_matmul_plugin(
                dtype='float16')
        else:
            network.plugin_config.set_weight_only_quant_matmul_plugin(
                dtype='float16')
    if use_smooth_quant:
        network.plugin_config.set_smooth_quant_gemm_plugin(dtype='float16')
        network.plugin_config.set_rmsnorm_quantization_plugin(dtype='float16')
        network.plugin_config.set_layernorm_quantization_plugin(dtype='float16')
        network.plugin_config.set_quantize_tensor_plugin()
        network.plugin_config.set_quantize_per_token_plugin()
    nccl_plugin = model.config.dtype if model.config.mapping.world_size > 1 else False
    if nccl_plugin:
        network.plugin_config.set_nccl_plugin(
            nccl_plugin, network.plugin_config.use_custom_all_reduce)

    with net_guard(network):
        # Prepare
        network.set_named_parameters(model.named_parameters())

        # Forward
        inputs = model.prepare_inputs(
            max_batch_size=build_config.max_batch_size,
            max_input_len=build_config.max_input_len,
            max_seq_len=build_config.max_input_len +
            build_config.max_output_len,
            use_cache=True,
            max_beam_width=build_config.max_beam_width,
            max_num_tokens=build_config.max_num_tokens,
            prompt_embedding_table_size=build_config.
            max_prompt_embedding_table_size,
            gather_context_logits=build_config.gather_context_logits,
            gather_generation_logits=build_config.gather_generation_logits)
        model(**inputs)

    optimize(network)

    # Network -> Engine
    engine = builder.build_engine(network, builder_config)
    engine_config = EngineConfig(model.config, build_config, __version__)

    return Engine(engine_config, engine)


def build(build_config: BuildConfig,
          rank: int = 0,
          ckpt_dir: str = None,
          model_config: Union[str, PretrainedConfig] = None,
          weights=None,
          model_cls=None,
          **kwargs) -> Engine:
    if ckpt_dir is not None:
        model_config = PretrainedConfig.from_json_file(
            os.path.join(ckpt_dir, 'config.json'))
    else:
        assert model_config is not None
        if isinstance(model_config, PretrainedConfig):
            model_config = model_config
        else:
            model_config = PretrainedConfig.from_json_file(model_config)

    logits_dtype = kwargs.pop('logits_dtype', None)
    if logits_dtype is not None:
        model_config.logits_dtype = logits_dtype
    model_config.use_prompt_tuning = build_config.max_prompt_embedding_table_size > 0

    assert rank < model_config.mapping.world_size
    architecture = model_config.architecture

    if model_cls is None:
        if architecture not in MODEL_MAP:
            raise RuntimeError(
                f'Unsupported model architecture: {architecture}')
        model_cls = MODEL_MAP[architecture]

    rank_config = copy.deepcopy(model_config)
    rank_config.set_rank(rank)

    if ckpt_dir is not None:
        model = model_cls.from_checkpoint(ckpt_dir,
                                          rank=rank,
                                          config=rank_config)
    else:
        model = model_cls.from_config(rank_config)
        if weights is not None:
            model.load(weights)

    return build_model(model, build_config)


def build_and_save(rank, gpu_id, ckpt_dir, build_config, output_dir, log_level,
                   model_config, model_cls, **kwargs):
    torch.cuda.set_device(gpu_id)
    logger.set_level(log_level)
    engine = build(build_config,
                   rank,
                   ckpt_dir,
                   model_config,
                   model_cls=model_cls,
                   **kwargs)
    engine.save(output_dir)


def parallel_build(ckpt_dir_or_model_config: str,
                   build_config: BuildConfig,
                   output_dir: str,
                   workers: int = 1,
                   log_level: str = 'info',
                   model_cls=None,
                   **kwargs):
    ckpt_dir = ckpt_dir_or_model_config
    if ckpt_dir_or_model_config.lower().endswith('.json'):
        model_config = PretrainedConfig.from_json_file(ckpt_dir_or_model_config)
        ckpt_dir = None
    else:
        model_config = PretrainedConfig.from_json_file(
            os.path.join(ckpt_dir_or_model_config, 'config.json'))

    if workers == 1:
        for rank in range(model_config.mapping.world_size):
            build_and_save(rank, rank % workers, ckpt_dir, build_config,
                           output_dir, log_level, model_config, model_cls,
                           **kwargs)
    else:
        with ProcessPoolExecutor(mp_context=get_context('spawn'),
                                 max_workers=workers) as p:
            futures = [
                p.submit(build_and_save, rank, rank % workers, ckpt_dir,
                         build_config, output_dir, log_level, model_config,
                         model_cls, **kwargs)
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

    if args.build_config is None:
        args.max_num_tokens = check_max_num_tokens(
            max_num_tokens=args.max_num_tokens,
            max_batch_size=args.max_batch_size,
            max_input_len=args.max_input_len,
            remove_input_padding=args.remove_input_padding)
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
            'gather_context_logits':
            args.gather_context_logits,
            'gather_generation_logits':
            args.gather_generation_logits,
            'strongly_typed':
            args.strongly_typed,
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
                'selective_scan_plugin': args.use_selective_scan_plugin,
            }
        })
    else:
        build_config = BuildConfig.from_json_file(args.build_config)

    source = args.checkpoint_dir if args.checkpoint_dir is not None else args.model_config
    kwargs = {'logits_dtype': args.logits_dtype}
    parallel_build(source, build_config, args.output_dir, workers,
                   args.log_level, model_cls, **kwargs)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Total time of building all engines: {t}')


if __name__ == '__main__':
    main()
