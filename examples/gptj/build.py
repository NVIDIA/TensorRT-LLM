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
import os
import time
from pathlib import Path

# isort: off
import torch
import torch.multiprocessing as mp
import tensorrt as trt
# isort: on
from transformers import AutoModelForCausalLM
from weight import (get_scaling_factors, load_from_awq_gpt_j,
                    load_from_bin_gpt_j, load_from_hf_gpt_j, parse_config)

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import quantize_model
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.profiler import check_gpt_mem_usage
from tensorrt_llm.quantization import QuantMode

MODEL_NAME = "gptj"
hf_gpt = None
awq_gptj_config = None


def get_engine_name(model, dtype, tp_size, rank):
    return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)


def serialize_engine(engine, path):
    logger.info(f'Serializing engine to {path}...')
    tik = time.time()
    with open(path, 'wb') as f:
        f.write(engine)
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')


def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='world size, only support tensor parallelism now')
    parser.add_argument(
        '--model_dir',
        type=str,
        default=None,
        help='The path to HF GPT-J model / checkpoints to read weights from')
    parser.add_argument('--quant_ckpt_path', type=str, default=None)
    parser.add_argument(
        '--ft_model_dir',
        type=str,
        default=None,
        help=
        'The path to FT-format (binary) GPT-J model / checkpoints to read weights from'
    )
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'float32'])
    parser.add_argument('--logits_dtype',
                        type=str,
                        default='float32',
                        choices=['float16', 'float32'])
    parser.add_argument(
        '--timing_cache',
        type=str,
        default='model.cache',
        help=
        'The path of to read timing cache from, will be ignored if the file does not exist'
    )
    parser.add_argument(
        '--profiling_verbosity',
        type=str,
        default='layer_names_only',
        choices=['layer_names_only', 'detailed', 'none'],
        help=
        'The profiling verbosity for the generated TRT engine. Set to detailed can inspect tactic choices and kernel parameters.'
    )
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--vocab_size', type=int, default=50401)
    parser.add_argument('--n_layer', type=int, default=28)
    parser.add_argument('--n_positions', type=int, default=2048)
    parser.add_argument('--n_embd', type=int, default=4096)
    parser.add_argument('--n_head', type=int, default=16)
    parser.add_argument('--hidden_act', type=str, default='gelu')
    parser.add_argument('--rotary_dim', type=int, default=64)
    parser.add_argument('--max_batch_size', type=int, default=256)
    parser.add_argument('--max_input_len', type=int, default=200)
    parser.add_argument('--max_output_len', type=int, default=200)
    parser.add_argument('--max_beam_width', type=int, default=1)
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
    parser.add_argument(
        '--multi_block_mode',
        default=False,
        action='store_true',
        help=
        'Split long kv sequence into multiple blocks (applied to generation MHA kernels). \
                        It is beneifical when batchxnum_heads cannot fully utilize GPU.'
    )
    parser.add_argument('--gpus_per_node', type=int, default=8)
    parser.add_argument(
        '--output_dir',
        type=str,
        default='engine_outputs',
        help=
        'The path to save the serialized engine files, timing cache file and model configs'
    )
    parser.add_argument('--remove_input_padding',
                        default=False,
                        action='store_true')
    parser.add_argument('--enable_fp8', default=False, action='store_true')
    parser.add_argument(
        '--quantized_fp8_model_path',
        type=str,
        default=None,
        help='Path of a quantized model checkpoint that in .npz format')
    parser.add_argument(
        '--fp8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. fp8_kv_cache chooses fp8 quantization for KV'
    )
    parser.add_argument(
        '--int8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. int8_kv_cache chooses int8 quantization for KV'
    )
    parser.add_argument(
        '--use_inflight_batching',
        action="store_true",
        default=False,
        help="Activates inflight batching mode of gptAttentionPlugin.")
    parser.add_argument(
        '--paged_kv_cache',
        action="store_true",
        default=False,
        help=
        'By default we use contiguous KV cache. By setting this flag you enable paged KV cache'
    )
    parser.add_argument('--tokens_per_block',
                        type=int,
                        default=128,
                        help='Number of tokens per block in paged KV cache')
    parser.add_argument(
        '--max_num_tokens',
        type=int,
        default=None,
        help='Define the max number of tokens supported by the engine')
    parser.add_argument(
        '--per_group',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor to scale weights in the int4 range. '
        'per_group chooses at run time, and for each group, a custom scaling factor. '
        'The flag is built for GPTQ/AWQ quantization.')
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
        choices=['int8', 'int4', 'int4_awq'],
        help=
        'Define the precision for the weights when using weight-only quantization.'
        'You must also use --use_weight_only for that argument to have an impact.'
    )
    parser.add_argument(
        '--quantize_lm_head',
        default=False,
        action="store_true",
        help='Quantize lm_head weights as well when using int4_awq.')
    parser.add_argument(
        '--strongly_typed',
        default=False,
        action="store_true",
        help=
        'This option is introduced with trt 9.1.0.1+ and will reduce the building time significantly for fp8.'
    )
    args = parser.parse_args(args)

    logger.set_level(args.log_level)

    if not args.remove_input_padding:
        if args.use_gpt_attention_plugin:
            logger.warning(
                f"It is recommended to specify --remove_input_padding when using GPT attention plugin"
            )

    if args.model_dir is not None:
        global hf_gpt
        logger.info(f'Loading HF GPTJ model from {args.model_dir}...')
        hf_gpt = AutoModelForCausalLM.from_pretrained(args.model_dir)
        args.n_embd = hf_gpt.config.n_embd
        args.n_head = hf_gpt.config.n_head
        args.n_layer = hf_gpt.config.n_layer
        args.n_positions = hf_gpt.config.n_positions
        args.vocab_size = hf_gpt.config.vocab_size
    elif args.ft_model_dir is not None:
        logger.info(f"Setting model configuration from {args.ft_model_dir}.")
        n_embd, n_head, n_layer, n_positions, vocab_size, _, hidden_act, rotary_pct, bias, inter_size, multi_query_mode, dtype, prompt_num_tasks, prompt_max_vocab_size = parse_config(
            Path(args.ft_model_dir) / "config.ini")
        args.n_embd = n_embd
        args.n_head = n_head
        args.n_layer = n_layer
        args.n_positions = n_positions
        args.vocab_size = vocab_size
        args.hidden_act = hidden_act
        args.rotary_pct = rotary_pct
        args.bias = bias
        args.dtype = dtype
        args.inter_size = inter_size
        args.multi_query_mode = multi_query_mode

    if args.quantize_lm_head and args.weight_only_precision == 'int4_awq':
        if args.vocab_size % 64 != 0:
            args.vocab_size = int((args.vocab_size + 63) / 64) * 64
            logger.info("To use awq we pad it to {}.".format(args.vocab_size))

    if args.use_weight_only:
        if args.per_group:
            args.quant_mode = QuantMode.from_description(
                quantize_weights=True,
                quantize_activations=False,
                per_token=False,
                per_channel=False,
                per_group=True,
                use_int4_weights=True)
        else:
            args.quant_mode = QuantMode.use_weight_only(
                args.weight_only_precision == 'int4')
    else:
        args.quant_mode = QuantMode(0)

    if args.weight_only_precision == 'int4_awq' and args.quantize_lm_head:
        if args.vocab_size % 64 != 0:
            args.vocab_size = int((args.vocab_size + 63) / 64) * 64
            logger.info("To use awq we pad vocab_size to {}.".format(
                args.vocab_size))

    if args.int8_kv_cache:
        args.quant_mode = args.quant_mode.set_int8_kv_cache()
    elif args.fp8_kv_cache:
        assert (
            args.use_gpt_attention_plugin
        ), "You have to use GPT attention plugin when fp8 KV cache is set"
        args.quant_mode = args.quant_mode.set_fp8_kv_cache()

    if args.enable_fp8:
        args.quant_mode = args.quant_mode.set_fp8_qdq()

    if args.use_inflight_batching:
        if not args.use_gpt_attention_plugin:
            args.use_gpt_attention_plugin = 'float16'
            logger.info(
                f"Using GPT attention plugin for inflight batching mode. Setting to default '{args.use_gpt_attention_plugin}'"
            )
        if not args.remove_input_padding:
            args.remove_input_padding = True
            logger.info(
                "Using remove input padding for inflight batching mode.")
        if not args.paged_kv_cache:
            args.paged_kv_cache = True
            logger.info("Using paged KV cache for inflight batching mode.")

    if args.max_num_tokens is not None:
        assert args.enable_context_fmha

    assert (math.log2(args.tokens_per_block).is_integer()
            ), "tokens_per_block must be power of 2"
    if args.enable_context_fmha or args.enable_context_fmha_fp32_acc:
        assert (args.tokens_per_block >=
                128), "Context fMHA requires >= 128 tokens per block"

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
    kv_dtype = trt.float16 if args.dtype == 'float16' else trt.float32
    mapping = Mapping(world_size=args.world_size,
                      rank=rank,
                      tp_size=args.world_size)  # TP only

    # Initialize Module
    tensorrt_llm_gpt = tensorrt_llm.models.GPTJForCausalLM(
        num_layers=args.n_layer,
        num_heads=args.n_head,
        hidden_size=args.n_embd,
        vocab_size=args.vocab_size,
        hidden_act=args.hidden_act,
        max_position_embeddings=args.n_positions,
        rotary_dim=args.rotary_dim,
        dtype=kv_dtype,
        logits_dtype=args.logits_dtype,
        mapping=mapping,
        quant_mode=args.quant_mode)

    quantize_kwargs = {}
    if args.use_weight_only and args.per_group:
        assert args.weight_only_precision == 'int4_awq'
        quantize_kwargs = {
            "group_size": 128,
            "zero": False,
            "pre_quant_scale": True,
            "exclude_modules": ['lm_head'] if not args.quantize_lm_head else [],
        }
    tensorrt_llm_gpt = quantize_model(tensorrt_llm_gpt, args.quant_mode,
                                      **quantize_kwargs)

    if args.model_dir is not None:
        assert hf_gpt is not None, f'Could not load weights from hf_gpt model as it is not loaded yet.'
        if args.enable_fp8:
            gptj_scaling_factors = get_scaling_factors(
                args.quantized_fp8_model_path, args.n_layer, args.quant_mode)
        else:
            gptj_scaling_factors = None
        if args.use_weight_only and args.weight_only_precision == 'int4_awq' and args.per_group:
            load_from_awq_gpt_j(tensorrt_llm_gpt,
                                quant_ckpt_path=args.quant_ckpt_path,
                                quantize_lm_head=args.quantize_lm_head,
                                ft_model_dir=args.ft_model_dir,
                                mapping=mapping,
                                fp16=(args.dtype == 'float16'))
        else:
            load_from_hf_gpt_j(tensorrt_llm_gpt,
                               hf_gpt,
                               fp16=(args.dtype == 'float16'),
                               scaling_factors=gptj_scaling_factors)
    elif args.ft_model_dir is not None:
        load_from_bin_gpt_j(tensorrt_llm_gpt, args.ft_model_dir, rank,
                            args.world_size, args.dtype)

    # Module -> Network
    network = builder.create_network()
    network.trt_network.name = engine_name
    if args.use_gpt_attention_plugin:
        network.plugin_config.set_gpt_attention_plugin(
            dtype=args.use_gpt_attention_plugin)
    if args.use_gemm_plugin:
        if not args.enable_fp8:
            network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
        else:
            logger.info(
                "Gemm plugin does not support FP8. Disabled Gemm plugin.")
    if args.use_layernorm_plugin:
        network.plugin_config.set_layernorm_plugin(
            dtype=args.use_layernorm_plugin)
    assert not (args.enable_context_fmha and args.enable_context_fmha_fp32_acc)
    if args.enable_context_fmha:
        network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
    if args.enable_context_fmha_fp32_acc:
        network.plugin_config.set_context_fmha(
            ContextFMHAType.enabled_with_fp32_acc)
    if args.multi_block_mode:
        network.plugin_config.enable_mmha_multi_block_mode()
    if args.use_weight_only:
        if args.per_group:
            network.plugin_config.set_weight_only_groupwise_quant_matmul_plugin(
                dtype='float16')
        else:
            network.plugin_config.set_weight_only_quant_matmul_plugin(
                dtype='float16')
    if args.world_size > 1:
        network.plugin_config.set_nccl_plugin(args.dtype)
    if args.remove_input_padding:
        network.plugin_config.enable_remove_input_padding()
    if args.paged_kv_cache:
        network.plugin_config.enable_paged_kv_cache(args.tokens_per_block)

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
            max_num_tokens=args.max_num_tokens)

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
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # when doing serializing build, all ranks share one engine
    builder = Builder()

    cache = None
    for cur_rank in range(args.world_size):
        # skip other ranks if parallel_build is enabled
        if args.parallel_build and cur_rank != rank:
            continue
        # NOTE: int8 flag is required to be true when INT8 tensors are exposed to TRT
        # TRT-LLM has INT8 I/O when act/weights are quantized without group-scaling (AWQ, GPTQ)
        # OR INT8 KV cache is set to contiguous (without paged KV cache enabled).
        int8_trt_flag = (args.quant_mode.has_act_or_weight_quant()
                         and not args.quant_mode.has_per_group_scaling()) or (
                             not args.paged_kv_cache
                             and args.quant_mode.has_int8_kv_cache())

        builder_config = builder.create_builder_config(
            name=MODEL_NAME,
            precision=args.dtype,
            timing_cache=args.timing_cache if cache is None else cache,
            profiling_verbosity=args.profiling_verbosity,
            tensor_parallel=args.world_size,  # TP only
            parallel_build=args.parallel_build,
            num_layers=args.n_layer,
            num_heads=args.n_head,
            hidden_size=args.n_embd,
            vocab_size=args.vocab_size,
            hidden_act=args.hidden_act,
            max_position_embeddings=args.n_positions,
            max_batch_size=args.max_batch_size,
            max_beam_width=args.max_beam_width,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            max_num_tokens=args.max_num_tokens,
            int8=int8_trt_flag,
            quant_mode=args.quant_mode,
            strongly_typed=args.strongly_typed)

        engine_name = get_engine_name(MODEL_NAME, args.dtype, args.world_size,
                                      cur_rank)
        engine = build_rank_engine(builder, builder_config, engine_name,
                                   cur_rank, args)
        assert engine is not None, f'Failed to build engine for rank {cur_rank}'

        local_num_kv_heads = (args.n_head + args.world_size -
                              1) // args.world_size
        kv_dtype = str_dtype_to_trt(args.dtype)
        if args.quant_mode.has_int8_kv_cache():
            kv_dtype = str_dtype_to_trt('int8')
        elif args.quant_mode.has_fp8_kv_cache():
            kv_dtype = str_dtype_to_trt('fp8')
        check_gpt_mem_usage(
            engine=engine,
            kv_dtype=kv_dtype,
            use_gpt_attention_plugin=args.use_gpt_attention_plugin,
            paged_kv_cache=args.paged_kv_cache,
            max_batch_size=args.max_batch_size,
            max_beam_width=args.max_beam_width,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            local_num_kv_heads=local_num_kv_heads,
            head_size=args.n_embd / args.n_head,
            num_layers=args.n_layer)

        if cur_rank == 0:
            # Use in-memory timing cache for multiple builder passes.
            if not args.parallel_build:
                cache = builder_config.trt_builder_config.get_timing_cache()

        serialize_engine(engine, os.path.join(args.output_dir, engine_name))

    if rank == 0:
        ok = builder.save_timing_cache(
            builder_config, os.path.join(args.output_dir, "model.cache"))
        assert ok, "Failed to save timing cache."


def run_build(args=None):
    args = parse_arguments(args)
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


if __name__ == '__main__':
    run_build()
