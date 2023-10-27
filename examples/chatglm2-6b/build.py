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
import json
import time
from pathlib import Path
from typing import List

import torch
import torch.multiprocessing as mp
import transformers
from weight import load_from_hf

import tensorrt_llm
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import (ChatGLM2_6BHeadModel, smooth_quantize,
                                 weight_only_quantize)
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode

MODEL_NAME = "chatglm2-6b"


def get_engine_name(model, dtype, tp_size, rank):
    return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)


def find_engines(dir: Path,
                 model_name: str = "*",
                 dtype: str = "*",
                 tp_size: str = "*",
                 rank: str = "*") -> List[Path]:
    template = f"{model_name}_{dtype}_tp{tp_size}_rank{rank}.engine"
    return list(dir.glob(template))


def serialize_engine(engine, path):
    logger.info(f'Serializing engine to {path}...')
    tik = time.time()
    with open(path, 'wb') as f:
        f.write(bytearray(engine))
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')


def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='world size, only support tensor parallelism now')
    parser.add_argument('--model_dir', type=str, default="pyTorchModel")
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument(
        '--timing_cache',
        type=str,
        default='model.cache',
        help=
        'The path of to read timing cache from, will be ignored if the file does not exist'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='verbose',
        choices=['verbose', 'info', 'warning', 'error', 'internal_error'])
    parser.add_argument('--max_batch_size', type=int, default=8)
    parser.add_argument('--max_input_len', type=int, default=1024)
    parser.add_argument('--max_output_len', type=int, default=1024)
    parser.add_argument('--max_beam_width', type=int, default=1)
    parser.add_argument(
        '--use_gpt_attention_plugin',
        nargs='?',
        const='float16',
        default='float16',
        choices=['float32', 'float16', 'bfloat16', False],
        help=
        "Activates attention plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_gemm_plugin',
        nargs='?',
        const='float16',
        type=str,
        default='float16',
        choices=['float32', 'float16', 'bfloat16', False],
        help=
        "Activates GEMM plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_rmsnorm_plugin',
        nargs='?',
        const='float16',
        type=str,
        default='float16',
        choices=['float32', 'float16', 'bfloat16', False],
        help=
        "Activates rmsnorm plugin. You can specify the plugin dtype or leave blank to use the model dtype.",
    )
    parser.add_argument('--gather_all_token_logits',
                        action='store_true',
                        default=False)
    parser.add_argument('--parallel_build', default=False, action='store_true')
    parser.add_argument('--enable_context_fmha',
                        default=False,
                        action='store_true')
    parser.add_argument('--enable_context_fmha_fp32_acc',
                        default=False,
                        action='store_true')
    parser.add_argument('--gpus_per_node', type=int, default=8)
    parser.add_argument('--builder_opt', type=int, default=None)
    parser.add_argument(
        '--output_dir',
        type=Path,
        default='trtModel',
        help=
        'The path to save the serialized engine files, timing cache file and model configs'
    )
    parser.add_argument('--remove_input_padding',
                        default=False,
                        action='store_true')
    parser.add_argument(
        '--use_inflight_batching',
        action="store_true",
        default=False,
        help="Activates inflight batching mode of gptAttentionPlugin.")

    # Arguments related to the quantization of the model.
    parser.add_argument(
        '--use_smooth_quant',
        default=False,
        action="store_true",
        help=
        'Use the SmoothQuant method to quantize activations and weights for the various GEMMs.'
        'See --per_channel and --per_token for finer-grained quantization options.'
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
        '--per_channel',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor for the GEMM\'s result. '
        'per_channel instead uses a different static scaling factor for each channel. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument(
        '--per_token',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor to scale activations in the int8 range. '
        'per_token chooses at run time, and for each token, a custom scaling factor. '
        'The latter is usually more accurate, but a little slower.')
    parser.add_argument(
        '--int8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. int8_kv_cache chooses int8 quantization for KV'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=None,
        help=
        'Seed to use when initializing the random number generator for torch.')
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
        '--enable_fp8',
        default=False,
        action='store_true',
    )
    parser.add_argument(
        '--fp8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. fp8_kv_cache chooses fp8 quantization for KV'
    )
    parser.add_argument(
        '--max_num_tokens',
        type=int,
        default=None,
        help='Define the max number of tokens supported by the engine')
    parser.add_argument(
        '--strongly_typed',
        default=False,
        action="store_true",
        help=
        'This option is introduced with trt 9.1.0.1+ and will reduce the building time significantly for fp8.'
    )
    parser.add_argument(
        '--use_custom_all_reduce',
        action='store_true',
        help=
        'Activates latency-optimized algorithm for all-reduce instead of NCCL.')
    args = parser.parse_args(args)
    logger.set_level(args.log_level)

    args.apply_query_key_layer_scaling = False  # always False in TRT-LLM
    args.hidden_act = 'swiglu'
    args.multi_block_mode = False

    plugins_args = [
        'use_gpt_attention_plugin',
        'use_gemm_plugin',
        'use_rmsnorm_plugin',
    ]
    for plugin_arg in plugins_args:
        if getattr(args, plugin_arg) is None:
            logger.info(
                f"{plugin_arg} set, without specifying a value. Using {args.dtype} automatically."
            )
            setattr(args, plugin_arg, args.dtype)

    assert args.model_dir is not None
    with open(Path(args.model_dir) / "config.json", "r") as f:
        js = json.loads(f.read())
    assert js["_name_or_path"] == "THUDM/" + MODEL_NAME

    args.apply_residual_connection_post_layernorm = js[
        "apply_residual_connection_post_layernorm"]
    args.eos_token_id = js["eos_token_id"]
    args.ffn_hidden_size = js["ffn_hidden_size"]
    args.hidden_size = js["hidden_size"]
    args.kv_channels = js["kv_channels"]
    args.layernorm_epsilon = js["layernorm_epsilon"]
    args.linear_bias = js["add_bias_linear"]
    args.multi_query_mode = js["multi_query_attention"]
    args.max_seq_length = min(args.max_input_len + args.max_output_len,
                              js["seq_length"])
    args.num_kv_heads = js["multi_query_group_num"]
    args.num_heads = js["num_attention_heads"]
    args.num_layers = js["num_layers"]
    args.pad_token_id = js["pad_token_id"]
    args.qkv_bias = js["add_qkv_bias"]
    args.rmsnorm = js["rmsnorm"]
    args.use_cache = js["use_cache"]
    args.vocab_size = js["padded_vocab_size"]

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

    assert not (
        args.use_smooth_quant and args.use_weight_only
    ), "You cannot enable both SmoothQuant and INT8 weight-only together."

    if args.use_smooth_quant:
        args.quant_mode = QuantMode.use_smooth_quant(args.per_token,
                                                     args.per_channel)
    elif args.use_weight_only:
        args.quant_mode = QuantMode.use_weight_only(
            args.weight_only_precision == 'int4')
    else:
        args.quant_mode = QuantMode(0)

    if args.int8_kv_cache:
        args.quant_mode = args.quant_mode.set_int8_kv_cache()

    if args.fp8_kv_cache:
        assert (
            args.use_gpt_attention_plugin or args.use_inflight_batching
        ), "You have to use GPT attention plugin when fp8 KV cache is set"
        args.quant_mode = args.quant_mode.set_fp8_kv_cache()

    if args.enable_fp8:
        args.quant_mode = args.quant_mode.set_fp8_qdq()

    if args.max_num_tokens is not None:
        assert args.enable_context_fmha

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

    # Initialize Module
    args.mapping = Mapping(
        world_size=args.world_size,
        rank=rank,
        tp_size=args.world_size,
    )
    trtllm_model = ChatGLM2_6BHeadModel(args=args)

    if args.use_smooth_quant:
        trtllm_model = smooth_quantize(trtllm_model, args.quant_mode)
    elif args.use_weight_only:
        trtllm_model = weight_only_quantize(trtllm_model, args.quant_mode)

    if args.model_dir is not None:
        hf_model = transformers.AutoModel.from_pretrained(
            args.model_dir, trust_remote_code=True).cpu()
        trtllm_model = load_from_hf(
            trtllm_model,
            hf_model,
            mapping=args.mapping,
            dtype=args.dtype,
        )
        del hf_model

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
    if args.remove_input_padding:
        network.plugin_config.enable_remove_input_padding()
    if args.paged_kv_cache:
        network.plugin_config.enable_paged_kv_cache(args.tokens_per_block)

    # Quantization plugins.
    if args.use_smooth_quant:
        network.plugin_config.set_smooth_quant_gemm_plugin(dtype=args.dtype)
        network.plugin_config.set_layernorm_quantization_plugin(
            dtype=args.dtype)

        network.plugin_config.set_quantize_tensor_plugin()
        network.plugin_config.set_quantize_per_token_plugin()
    elif args.use_weight_only:
        network.plugin_config.set_weight_only_quant_matmul_plugin(
            dtype=args.dtype)

    if args.world_size > 1:
        network.plugin_config.set_nccl_plugin(args.dtype,
                                              args.use_custom_all_reduce)

    with net_guard(network):
        # Prepare
        network.set_named_parameters(trtllm_model.named_parameters())

        # Forward
        inputs = trtllm_model.prepare_inputs(
            max_batch_size=args.max_batch_size,
            max_input_len=args.max_input_len,
            max_new_tokens=args.max_output_len,
            use_cache=True,
            max_beam_width=args.max_beam_width,
        )
        trtllm_model(*inputs)

    tensorrt_llm.graph_rewriting.optimize(network)

    engine = None

    # Network -> Engine
    engine = builder.build_engine(network, builder_config)
    if rank == 0:
        config_path = args.output_dir / 'config.json'
        builder.save_config(builder_config, config_path)
    return engine


def build(rank, args):
    torch.cuda.set_device(rank % args.gpus_per_node)
    tensorrt_llm.logger.set_level(args.log_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timing_cache_file = args.output_dir / "model.cache"
    timing_cache = timing_cache_file

    builder = Builder()

    for cur_rank in range(args.world_size):
        # skip other ranks if parallel_build is enabled
        if args.parallel_build and cur_rank != rank:
            continue
        builder_config = builder.create_builder_config(
            precision=args.dtype,
            timing_cache=timing_cache,
            tensor_parallel=args.world_size,
            int8=(args.quant_mode.has_act_or_weight_quant()
                  or args.quant_mode.has_int8_kv_cache()),
            fp8=args.enable_fp8,
            strongly_typed=args.strongly_typed,
            opt_level=args.builder_opt,
            apply_query_key_layer_scaling=args.apply_query_key_layer_scaling,
            eos_token_id=args.eos_token_id,
            gather_all_token_logits=args.gather_all_token_logits,
            hidden_act=args.hidden_act,
            hidden_size=args.hidden_size,
            max_batch_size=args.max_batch_size,
            max_input_len=args.max_input_len,
            max_num_tokens=args.max_output_len + args.max_input_len,
            max_output_len=args.max_output_len,
            max_position_embeddings=args.max_seq_length,
            multi_query_mode=args.multi_query_mode,
            name=MODEL_NAME,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            num_layers=args.num_layers,
            pad_token_id=args.pad_token_id,
            paged_kv_cache=args.paged_kv_cache,
            parallel_build=args.parallel_build,
            quant_mode=int(args.quant_mode),
            remove_input_padding=args.remove_input_padding,
            vocab_size=args.vocab_size,
        )

        engine_name = get_engine_name(MODEL_NAME, args.dtype, args.world_size,
                                      cur_rank)
        engine = build_rank_engine(builder, builder_config, engine_name,
                                   cur_rank, args)
        assert engine is not None, f'Failed to build engine for rank {cur_rank}'

        if cur_rank == 0:
            # Use in-memory timing cache for multiple builder passes.
            if not args.parallel_build:
                timing_cache = builder_config.trt_builder_config.get_timing_cache(
                )

        serialize_engine(engine, args.output_dir / engine_name)

    if rank == 0:
        ok = builder.save_timing_cache(builder_config, timing_cache_file)
        assert ok, "Failed to save timing cache."


def run_build(args=None):
    args = parse_arguments(args)

    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)

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


if __name__ == '__main__':
    run_build()
