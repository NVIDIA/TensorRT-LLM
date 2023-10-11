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
from pathlib import Path

import torch
import torch.multiprocessing as mp

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import weight_only_quantize
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode

from weight import load_from_ft, parse_ft_config, check_embedding_share  # isort:skip

MODEL_NAME = "opt"


def get_engine_name(model, dtype, tp_size, rank):
    return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)


def serialize_engine(engine, path):
    logger.info(f'Serializing engine to {path}...')
    tik = time.time()
    with open(path, 'wb') as f:
        f.write(bytearray(engine))
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='world size, only support tensor parallelism now')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'float32'])
    parser.add_argument(
        '--timing_cache',
        type=str,
        default='model.cache',
        help=
        'The path of to read timing cache from, will be ignored if the file does not exist'
    )
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--vocab_size', type=int, default=50272)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_positions', type=int, default=2048)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--hidden_act', type=str, default='relu')
    parser.add_argument('--max_batch_size', type=int, default=256)
    parser.add_argument('--max_input_len', type=int, default=200)
    parser.add_argument('--max_output_len', type=int, default=200)
    parser.add_argument('--max_beam_width', type=int, default=1)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--post_norm', dest='pre_norm', action='store_false')
    parser.add_argument('--do_layer_norm_before',
                        default=False,
                        action='store_true')
    parser.add_argument('--use_gpt_attention_plugin',
                        nargs='?',
                        const='float16',
                        type=str,
                        default=False,
                        choices=['float16', 'float32'])
    parser.add_argument('--use_gemm_plugin',
                        nargs='?',
                        const='float16',
                        type=str,
                        default=False,
                        choices=['float16', 'float32'])
    parser.add_argument('--use_layernorm_plugin',
                        nargs='?',
                        const='float16',
                        type=str,
                        default=False,
                        choices=['float16', 'float32'])
    parser.add_argument('--parallel_build', default=False, action='store_true')
    parser.add_argument('--enable_context_fmha',
                        default=False,
                        action='store_true')
    parser.add_argument('--enable_context_fmha_fp32_acc',
                        default=False,
                        action='store_true')
    parser.add_argument('--gpus_per_node', type=int, default=8)
    parser.add_argument(
        '--output_dir',
        type=str,
        default='gpt_outputs',
        help=
        'The path to save the serialized engine files, timing cache file and model configs'
    )
    parser.add_argument('--remove_input_padding',
                        default=False,
                        action='store_true')

    parser.add_argument(
        '--int8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. int8_kv_cache chooses int8 quantization for KV'
    )
    parser.add_argument(
        '--use_weight_only',
        default=False,
        action="store_true",
        help='Quantize weights for the various GEMMs to INT4/INT8.'
        'See --weight_only_precision to set the precision')

    parser.add_argument(
        '--weight_only_precision',
        const='int8',
        type=str,
        nargs='?',
        default='int8',
        choices=['int8', 'int4'],
        help=
        'Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )
    parser.add_argument(
        '--max_prompt_embedding_table_size',
        type=int,
        default=0,
        help='Setting to a value > 0 enables support for prompt tuning.')

    parser.add_argument(
        '--use_parallel_embedding',
        action="store_true",
        default=False,
        help=
        'By default embedding parallelism is disabled. By setting this flag, embedding parallelism is enabled'
    )
    parser.add_argument(
        '--embedding_sharding_dim',
        type=int,
        default=0,
        choices=[0, 1],
        help=
        'By default the embedding lookup table is sharded along vocab dimension (embedding_sharding_dim=0). '
        'To shard it along hidden dimension, set embedding_sharding_dim=1'
        'Note: embedding sharing is only enabled when embedding_sharding_dim = 0'
    )
    parser.add_argument(
        '--use_embedding_sharing',
        action="store_true",
        default=False,
        help=
        'Try to reduce the engine size by sharing the embedding lookup table between two layers.'
        'Note: the flag might not take effect when the criteria are not met.')
    parser.add_argument(
        '--use_lookup_plugin',
        nargs='?',
        const=None,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates the lookup plugin which enables embedding sharing. It is also required for language modeling embedding weight sharing."
    )
    args = parser.parse_args()
    if args.use_weight_only:
        args.quant_mode = QuantMode.use_weight_only(
            args.weight_only_precision == 'int4')
    else:
        args.quant_mode = QuantMode(0)

    if args.int8_kv_cache:
        args.quant_mode = args.quant_mode.set_int8_kv_cache()
    if args.model_dir is not None:
        n_embd, n_head, n_layer, n_positions, vocab_size, do_layer_norm_before = parse_ft_config(
            Path(args.model_dir) / "config.ini")
        args.n_embd = n_embd
        args.n_head = n_head
        args.n_layer = n_layer
        args.n_positions = n_positions
        args.vocab_size = vocab_size
        args.do_layer_norm_before = do_layer_norm_before
    return args


def build_rank_engine(builder: Builder,
                      builder_config: tensorrt_llm.builder.BuilderConfig,
                      engine_name, rank, args):
    '''
       @brief: Build the engine on the given rank.
       @param rank: The rank to build the engine.
       @param args: The cmd line arguments.
       @return: The built engine.
    '''
    kv_dtype = str_dtype_to_trt(args.dtype)

    # Share_embedding_table can be set True only when:
    # 1) the weight for lm_head() does not exist while other weights exist
    # 2) For multiple-processes, use_parallel_embedding=True and embedding_sharding_dim == 0.
    # Besides, for TensorRT 9.0, we can observe the engine size reduction when the lookup and gemm plugin are enabled.
    share_embedding_table = False
    if args.use_embedding_sharing:
        if args.world_size > 1:
            if args.model_dir is not None and args.embedding_sharding_dim == 0 and args.use_parallel_embedding:
                share_embedding_table = check_embedding_share(args.model_dir)
        else:
            if args.model_dir is not None:
                share_embedding_table = check_embedding_share(args.model_dir)

        if not share_embedding_table:
            logger.warning(f'Cannot share the embedding lookup table.')

    # Initialize Module
    tensorrt_llm_gpt = tensorrt_llm.models.OPTLMHeadModel(
        num_layers=args.n_layer,
        num_heads=args.n_head,
        hidden_size=args.n_embd,
        vocab_size=args.vocab_size,
        hidden_act=args.hidden_act,
        max_position_embeddings=args.n_positions,
        dtype=kv_dtype,
        mapping=Mapping(world_size=args.world_size,
                        rank=rank,
                        tp_size=args.world_size),  # TP only
        pre_norm=args.pre_norm,
        do_layer_norm_before=args.do_layer_norm_before,
        use_prompt_tuning=args.max_prompt_embedding_table_size > 0,
        use_parallel_embedding=args.use_parallel_embedding,
        embedding_sharding_dim=args.embedding_sharding_dim,
        share_embedding_table=share_embedding_table)
    if args.use_weight_only:
        tensorrt_llm_gpt = weight_only_quantize(tensorrt_llm_gpt,
                                                args.quant_mode)
    if args.model_dir is not None:
        load_from_ft(tensorrt_llm_gpt,
                     args.model_dir,
                     rank,
                     args.world_size,
                     fp16=(args.dtype == 'float16'),
                     use_parallel_embedding=args.use_parallel_embedding,
                     sharding_dim=args.embedding_sharding_dim,
                     share_embedding_table=share_embedding_table)

    # Module -> Network
    network = builder.create_network()
    network.trt_network.name = engine_name
    if args.use_gpt_attention_plugin:
        network.plugin_config.set_gpt_attention_plugin(
            args.use_gpt_attention_plugin)
    if args.use_gemm_plugin:
        network.plugin_config.set_gemm_plugin(args.use_gemm_plugin)
    if args.use_layernorm_plugin:
        network.plugin_config.set_layernorm_plugin(args.use_layernorm_plugin)
    if args.use_lookup_plugin:
        # Use the plugin for the embedding parallelism and sharing
        network.plugin_config.set_lookup_plugin(dtype=args.dtype)
    assert not (args.enable_context_fmha and args.enable_context_fmha_fp32_acc)
    if args.enable_context_fmha:
        network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
    if args.enable_context_fmha_fp32_acc:
        network.plugin_config.set_context_fmha(
            ContextFMHAType.enabled_with_fp32_acc)
    if args.use_weight_only:
        assert (args.dtype == 'float16')
        network.plugin_config.set_weight_only_quant_matmul_plugin(
            dtype=args.dtype)
    if args.world_size > 1:
        network.plugin_config.set_nccl_plugin(args.dtype)
    if args.remove_input_padding:
        network.plugin_config.enable_remove_input_padding()
    with net_guard(network):
        # Prepare
        network.set_named_parameters(tensorrt_llm_gpt.named_parameters())

        # Forward
        inputs = tensorrt_llm_gpt.prepare_inputs(
            args.max_batch_size,
            args.max_input_len,
            args.max_output_len,
            True,
            args.max_beam_width,
            prompt_embedding_table_size=args.max_prompt_embedding_table_size)
        tensorrt_llm_gpt(*inputs)

    tensorrt_llm.graph_rewriting.optimize(network)

    engine = None

    # Network -> Engine
    engine = builder.build_engine(network, builder_config)
    if rank == 0:
        config_path = os.path.join(args.output_dir, 'config.json')
        builder.save_config(builder_config, config_path)
    return engine


def build(rank, args):
    torch.cuda.set_device(rank % args.gpus_per_node)
    tensorrt_llm.logger.set_level(args.log_level)
    os.makedirs(args.output_dir, exist_ok=True)

    builder = Builder()
    cache = None
    for cur_rank in range(args.world_size):
        # skip other ranks if parallel_build is enabled
        if args.parallel_build and cur_rank != rank:
            continue

        builder_config = builder.create_builder_config(
            name=MODEL_NAME,
            precision=args.dtype,
            timing_cache=args.timing_cache if cache is None else cache,
            tensor_parallel=args.world_size,  # TP only
            parallel_build=args.parallel_build,
            num_layers=args.n_layer,
            num_heads=args.n_head,
            hidden_size=args.n_embd,
            vocab_size=args.vocab_size,
            hidden_act=args.hidden_act,
            max_position_embeddings=args.n_positions,
            max_batch_size=args.max_batch_size,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            use_prompt_tuning=args.max_prompt_embedding_table_size > 0,
            int8=(args.quant_mode.has_act_and_weight_quant()
                  or args.quant_mode.has_int8_kv_cache()))

        engine_name = get_engine_name(MODEL_NAME, args.dtype, args.world_size,
                                      cur_rank)
        engine = build_rank_engine(builder, builder_config, engine_name,
                                   cur_rank, args)
        assert engine is not None, f'Failed to build engine for rank {cur_rank}'

        if cur_rank == 0:
            # Use in-memory timing cache for multiple builder passes.
            if not args.parallel_build:
                cache = builder_config.trt_builder_config.get_timing_cache()

        serialize_engine(engine, os.path.join(args.output_dir, engine_name))

    if rank == 0:
        ok = builder.save_timing_cache(
            builder_config, os.path.join(args.output_dir, "model.cache"))
        assert ok, "Failed to save timing cache."


if __name__ == '__main__':
    args = parse_arguments()
    logger.set_level(args.log_level)
    tik = time.time()
    if args.parallel_build and args.world_size > 1 and \
            torch.cuda.device_count() >= args.world_size:
        logger.warning(
            f'Parallelly build TensorRT engines. Please make sure that all of the {args.world_size} GPUs are totally free.'
        )
        mp.spawn(build, nprocs=args.world_size, args=(args, ))
    else:
        args.parallel_build = False
        logger.info('Serially build TensorRT engines.')
        build(0, args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Total time of building all {args.world_size} engines: {t}')
