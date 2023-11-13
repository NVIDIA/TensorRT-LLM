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

import torch
import torch.multiprocessing as mp
from transformers import AutoConfig, AutoModelForCausalLM

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.layers.attention import PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import YiForCausalLM
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType

from weight import load_from_hf_yi

def get_engine_name(model, dtype, tp_size, pp_size, rank):
    return f'{model}_{dtype}_tp{tp_size}_pp{pp_size}_rank{rank}.engine'

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
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--pp_size', type=int, default=1)
    parser.add_argument('--model_dir', type=str, default='')
    parser.add_argument('--dtype',
                        type=str,
                        default='bfloat16',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument(
        '--timing_cache',
        type=str,
        default='model.cache',
        help=
        'The path of to read timing cache from, will be ignored if the file does not exist'
    )
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--vocab_size', type=int, default=64000)
    parser.add_argument('--n_layer', type=int, default=32)
    parser.add_argument('--n_positions', type=int, default=4096)
    parser.add_argument('--n_embd', type=int, default=4096)
    parser.add_argument('--n_head', type=int, default=32)
    parser.add_argument('--n_kv_head', type=int, default=4)
    parser.add_argument('--inter_size', type=int, default=11008)
    parser.add_argument('--hidden_act', type=str, default='silu')
    parser.add_argument('--rope_theta', type=float, default=5000000.0)
    parser.add_argument('--rope_scaling', type=float, default=None)
    parser.add_argument('--rms_norm_eps', type=float, default=1e-5)
    parser.add_argument('--max_batch_size', type=int, default=1)
    parser.add_argument('--max_input_len', type=int, default=2048)
    parser.add_argument('--max_output_len', type=int, default=512)
    parser.add_argument('--max_beam_width', type=int, default=1)
    parser.add_argument('--use_gpt_attention_plugin',
                        nargs='?',
                        const='bfloat16',
                        type=str,
                        default=False,
                        choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--use_gemm_plugin',
                        nargs='?',
                        const='bfloat16',
                        type=str,
                        default=False,
                        choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--use_rmsnorm_plugin',
                        nargs='?',
                        const='bfloat16',
                        type=str,
                        default=False,
                        choices=['float16', 'float32', 'bfloat16'])
    parser.add_argument('--enable_context_fmha',
                        default=False,
                        action='store_true')
    parser.add_argument('--enable_context_fmha_fp32_acc',
                        default=False,
                        action='store_true')
    parser.add_argument('--parallel_build', default=False, action='store_true')
    parser.add_argument('--gpus_per_node', type=int, default=8)
    parser.add_argument('--builder_opt', type=int, default=None)
    parser.add_argument(
        '--output_dir',
        type=str,
        default='yi_outputs',
        help=
        'The path to save the serialized engine files, timing cache file and model configs'
    )
    parser.add_argument('--remove_input_padding',
                        default=False,
                        action='store_true')
    parser.add_argument(
        '--use_parallel_embedding',
        action='store_true',
        default=False,
        help=
        'By default embedding parallelism is disabled. By setting this flag, embedding parallelism is enabled'
    )
    parser.add_argument(
        '--embedding_sharding_dim',
        type=int,
        default=1,  # Meta does TP on hidden dim
        choices=[0, 1],
        help=
        'By default the embedding lookup table is sharded along vocab dimension (embedding_sharding_dim=0). '
        'To shard it along hidden dimension, set embedding_sharding_dim=1'
        'Note: embedding sharing is only enabled when embedding_sharding_dim = 0'
    )
    parser.add_argument(
        '--use_inflight_batching',
        action='store_true',
        default=False,
        help='Activates inflight batching mode of gptAttentionPlugin.')
    parser.add_argument(
        '--paged_kv_cache',
        action='store_true',
        default=False,
        help=
        'By default we use contiguous KV cache. By setting this flag you enable paged KV cache'
    )
    parser.add_argument('--tokens_per_block',
                        type=int,
                        default=64,
                        help='Number of tokens per block in paged KV cache')
    parser.add_argument(
        '--max_num_tokens',
        type=int,
        default=None,
        help='Define the max number of tokens supported by the engine')

    args = parser.parse_args()

    if args.use_inflight_batching:
        if not args.use_gpt_attention_plugin:
            args.use_gpt_attention_plugin = 'bfloat16'
            logger.info(
                f"Using GPT attention plugin for inflight batching mode. Setting to default '{args.use_gpt_attention_plugin}'"
            )
        if not args.remove_input_padding:
            args.remove_input_padding = True
            logger.info(
                'Using remove input padding for inflight batching mode.')
        if not args.paged_kv_cache:
            args.paged_kv_cache = True
            logger.info('Using paged KV cache for inflight batching mode.')

    if args.max_num_tokens is not None:
        assert args.enable_context_fmha

    if args.model_dir is not None:
        hf_config = AutoConfig.from_pretrained(args.model_dir,
                                               trust_remote_code=True)
        args.inter_size = hf_config.intermediate_size
        args.n_embd = hf_config.hidden_size
        args.n_head = hf_config.num_attention_heads
        args.n_kv_head = hf_config.num_key_value_heads
        args.n_layer = hf_config.num_hidden_layers
        args.n_positions = hf_config.max_position_embeddings
        args.vocab_size = hf_config.vocab_size
        args.hidden_act = hf_config.hidden_act
        args.rope_theta = hf_config.rope_theta
        if hasattr(hf_config, 'rope_scaling'):
            args.rope_scaling = hf_config.rope_scaling
        if hasattr(hf_config, 'rms_norm_eps'):
            args.rms_norm_eps = hf_config.rms_norm_eps

    if args.dtype == 'bfloat16':
        assert args.use_gemm_plugin, 'Please use gemm plugin when dtype is bfloat16'
    assert (args.n_kv_head % args.tp_size) == 0 or (args.tp_size % args.n_kv_head) == 0
    assert args.pp_size * args.tp_size == args.world_size

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
    mapping = Mapping(world_size=args.world_size,
                      rank=rank,
                      tp_size=args.tp_size,
                      pp_size=args.pp_size)
    assert args.n_layer % args.pp_size == 0, \
        f'num_layers {args.n_layer} must be a multiple of pipeline parallelism size {args.pp_size}'
    # Initialize Module
    tensorrt_llm_yi = YiForCausalLM(
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        num_key_value_heads=args.n_kv_head,
        hidden_size=args.n_embd,
        vocab_size=args.vocab_size,
        hidden_act=args.hidden_act,
        max_position_embeddings=args.n_positions,
        dtype=str_dtype_to_trt(args.dtype),
        intermediate_size=args.inter_size,
        position_embedding_type=PositionEmbeddingType.rope_gpt_neox, 
        rope_theta=args.rope_theta,
        rope_scaling=args.rope_scaling,
        mapping=mapping,
        use_parallel_embedding=args.use_parallel_embedding,
        embedding_sharding_dim=args.embedding_sharding_dim,
        rms_norm_eps=args.rms_norm_eps)

    if args.model_dir is not None:
        logger.info(f'Loading HF Yi ... from {args.model_dir}')
        tik = time.time()
        hf_yi = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            device_map='cpu',
            torch_dtype='auto',
            trust_remote_code=True)
        tok = time.time()
        t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
        load_from_hf_yi(tensorrt_llm_yi,
                              hf_yi,
                              dtype=args.dtype,
                              mapping=mapping)
        logger.info(f'HF Yi loaded. Total time: {t}')
        del hf_yi

    # Module -> Network
    network = builder.create_network()
    network.trt_network.name = engine_name

    if args.use_gpt_attention_plugin:
        network.plugin_config.set_gpt_attention_plugin(
            dtype=args.use_gpt_attention_plugin)

    if args.use_gemm_plugin:
        network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)

    if args.use_rmsnorm_plugin:
        network.plugin_config.set_rmsnorm_plugin(dtype=args.use_rmsnorm_plugin)

    assert not (args.enable_context_fmha and args.enable_context_fmha_fp32_acc)

    if args.enable_context_fmha:
        network.plugin_config.set_context_fmha(ContextFMHAType.enabled)

    if args.enable_context_fmha_fp32_acc:
        network.plugin_config.set_context_fmha(
            ContextFMHAType.enabled_with_fp32_acc)

    if args.world_size > 1:
        network.plugin_config.set_nccl_plugin(args.dtype)

    if args.remove_input_padding:
        network.plugin_config.enable_remove_input_padding()
    if args.paged_kv_cache:
        network.plugin_config.enable_paged_kv_cache(args.tokens_per_block)

    with net_guard(network):
        network.set_named_parameters(tensorrt_llm_yi.named_parameters())
        inputs = tensorrt_llm_yi.prepare_inputs(args.max_batch_size,
                                                      args.max_input_len,
                                                      args.max_output_len,
                                                      True, #use_cache
                                                      args.max_beam_width,
                                                      args.max_num_tokens)
        tensorrt_llm_yi(*inputs)
        # TODO
        # add enable debugging
        # support to onnx exporting

    tensorrt_llm.graph_rewriting.optimize(network)

    engine = None

    engine = builder.build_engine(network, builder_config)
    if rank == 0:
        config_path = os.path.join(args.output_dir, 'config.json')
        builder.save_config(builder_config, config_path)
    return engine

def build(rank, args):
    torch.cuda.set_device(rank % args.gpus_per_node)
    tensorrt_llm.logger.set_level(args.log_level)

    # when doing serializing build, all ranks share one engine
    builder = Builder()

    cache = None
    model_name = 'yi'
    for cur_rank in range(args.world_size):
        # skip other ranks if parallel_build is enabled
        if args.parallel_build and cur_rank != rank:
            continue
        builder_config = builder.create_builder_config(
            name=model_name,
            precision=args.dtype,
            timing_cache=args.timing_cache if cache is None else cache,
            tensor_parallel=args.tp_size,
            pipeline_parallel=args.pp_size,
            parallel_build=args.parallel_build,
            num_layers=args.n_layer,
            num_heads=args.n_head,
            num_kv_heads=args.n_kv_head,
            hidden_size=args.n_embd,
            vocab_size=args.vocab_size,
            hidden_act=args.hidden_act,
            max_position_embeddings=args.n_positions,
            max_batch_size=args.max_batch_size,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            max_num_tokens=args.max_num_tokens,
            opt_level=args.builder_opt,
            )
        engine_name = get_engine_name(model_name, args.dtype, args.tp_size, args.pp_size, cur_rank)
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
            builder_config, os.path.join(args.output_dir, 'model.cache'))
        assert ok, 'Failed to save timing cache.'


if __name__ == '__main__':
    args = parse_arguments()
    # create folder on main process to avoid conflict
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
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
