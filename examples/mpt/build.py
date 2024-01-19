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
from typing import List

import torch
import torch.multiprocessing as mp

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.layers import PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import quantize_model
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode

from weight import get_scaling_factors, load_from_ft, parse_ft_config, check_embedding_share, load_from_awq_mpt  # isort:skip

MODEL_NAME = "mpt"


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
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--quant_ckpt_path', type=str, default=None)
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'float32', 'bfloat16'])
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
    parser.add_argument('--vocab_size', type=int, default=51200)
    parser.add_argument('--n_layer', type=int, default=24)
    parser.add_argument('--n_positions', type=int, default=1024)
    parser.add_argument('--n_embd', type=int, default=1024)
    parser.add_argument('--n_head', type=int, default=16)
    parser.add_argument('--n_kv_head', type=int, default=None)
    parser.add_argument('--hidden_act', type=str, default='gelu')
    parser.add_argument(
        '--rotary_pct',
        type=float,
        default=0.0,
        help="Setting this to a value > 0.0 (and <= 1.0) activates RoPE.")
    parser.add_argument('--inter_size', type=int, default=None)
    parser.add_argument('--no_bias', action="store_false")
    parser.add_argument('--max_batch_size', type=int, default=256)
    parser.add_argument('--max_input_len', type=int, default=200)
    parser.add_argument('--max_output_len', type=int, default=200)
    parser.add_argument('--max_beam_width', type=int, default=1)
    parser.add_argument(
        '--use_gpt_attention_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates attention plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_gemm_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates GEMM plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_layernorm_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates layernorm plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
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
    parser.add_argument('--builder_opt', type=int, default=None)
    parser.add_argument(
        '--output_dir',
        type=Path,
        default='engine_outputs',
        help=
        'The path to save the serialized engine files, timing cache file and model configs'
    )
    parser.add_argument('--remove_input_padding',
                        default=False,
                        action='store_true')

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
        choices=['int8', 'int4', 'int4_awq'],
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
        '--per_group',
        default=False,
        action="store_true",
        help=
        'By default, we use a single static scaling factor to scale weights in the int4 range. '
        'per_group chooses at run time, and for each group, a custom scaling factor. '
        'The flag is built for GPTQ/AWQ quantization.')
    parser.add_argument('--group_size',
                        type=int,
                        default=128,
                        help='Group size used in GPTQ/AWQ quantization.')
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
                        default=128,
                        help='Number of tokens per block in paged KV cache')
    parser.add_argument(
        '--max_num_tokens',
        type=int,
        default=None,
        help='Define the max number of tokens supported by the engine')
    parser.add_argument(
        '--max_prompt_embedding_table_size',
        type=int,
        default=0,
        help='Setting to a value > 0 enables support for prompt tuning.')
    parser.add_argument(
        '--use_inflight_batching',
        action="store_true",
        default=False,
        help="Activates inflight batching mode of gptAttentionPlugin.")
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
        help="Activates the lookup plugin which enables embedding sharing.")

    parser.add_argument('--enable_fp8', default=False, action='store_true')
    parser.add_argument(
        '--fp8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. fp8_kv_cache chooses fp8 quantization for KV'
    )

    parser.add_argument(
        '--strongly_typed',
        default=False,
        action="store_true",
        help=
        'This option is introduced with trt 9.1.0.1+ and will reduce the building time significantly for fp8.'
    )
    parser.add_argument(
        '--position_embedding_type',
        default='alibi',
        choices=PositionEmbeddingType.choices(),
        help='Set the position embedding type.',
    )
    parser.add_argument(
        '--quantized_fp8_model_path',
        type=str,
        default=None,
        help='Path of a quantized model checkpoint in .npz format')
    args = parser.parse_args(args)
    logger.set_level(args.log_level)

    args.bias = not args.no_bias
    if args.inter_size is None:
        args.inter_size = 4 * args.n_embd

    if args.model_dir is not None:
        logger.info(f"Setting model configuration from {args.model_dir}.")

        n_embd, n_head, n_layer, n_positions, vocab_size, _, hidden_act, rotary_pct, bias, inter_size, n_kv_head, dtype, prompt_num_tasks, prompt_max_vocab_size, position_embedding_type = parse_ft_config(
            Path(args.model_dir) / "config.ini")
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
        args.n_kv_head = n_kv_head
        args.position_embedding_type = position_embedding_type
    plugins_args = [
        'use_gpt_attention_plugin', 'use_gemm_plugin', 'use_layernorm_plugin',
        'use_lookup_plugin'
    ]
    for plugin_arg in plugins_args:
        if getattr(args, plugin_arg) is None:
            logger.info(
                f"{plugin_arg} set, without specifying a value. Using {args.dtype} automatically."
            )
            setattr(args, plugin_arg, args.dtype)

    assert not (
        args.use_smooth_quant and args.use_weight_only
    ), "You cannot enable both SmoothQuant and INT8 weight-only together."

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

    if args.use_smooth_quant:
        args.quant_mode = QuantMode.use_smooth_quant(args.per_token,
                                                     args.per_channel)
    elif args.use_weight_only:
        args.quant_mode = QuantMode.from_description(
            quantize_weights=True,
            quantize_activations=False,
            per_token=False,
            per_channel=False,
            per_group=args.per_group,
            use_int4_weights="int4" in args.weight_only_precision)
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

    args.position_embedding_type = PositionEmbeddingType[
        args.position_embedding_type]

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

    if share_embedding_table:
        logger.info(
            'Engine will share embedding and language modeling weights.')
    # TP only
    mapping = Mapping(world_size=args.world_size,
                      rank=rank,
                      tp_size=args.world_size)
    # Initialize Module
    tensorrt_llm_gpt = tensorrt_llm.models.GPTLMHeadModel(
        num_layers=args.n_layer,
        num_heads=args.n_head,
        hidden_size=args.n_embd,
        inter_size=args.inter_size,
        vocab_size=args.vocab_size,
        hidden_act=args.hidden_act,
        max_position_embeddings=args.n_positions,
        position_embedding_type=args.position_embedding_type,
        rotary_embedding_percentage=args.rotary_pct,
        dtype=kv_dtype,
        logits_dtype=args.logits_dtype,
        mapping=mapping,
        apply_query_key_layer_scaling=builder_config.
        apply_query_key_layer_scaling,
        quant_mode=args.quant_mode,
        bias=args.bias,
        num_kv_heads=args.n_kv_head,
        use_prompt_tuning=args.max_prompt_embedding_table_size > 0,
        use_parallel_embedding=args.use_parallel_embedding,
        embedding_sharding_dim=args.embedding_sharding_dim,
        share_embedding_table=share_embedding_table)

    quantize_kwargs = {}
    if args.use_smooth_quant or args.use_weight_only:
        if args.weight_only_precision == 'int4_awq':
            quantize_kwargs = {
                "group_size": args.group_size,
                "zero": False,
                "pre_quant_scale": True,
                "exclude_modules": ['lm_head'],
            }
        elif args.weight_only_precision == 'int4_gptq':
            quantize_kwargs = {
                "group_size": args.group_size,
                "zero": True,
                "pre_quant_scale": False,
            }
    elif args.enable_fp8 or args.fp8_kv_cache:
        logger.info(f'Loading scaling factors from '
                    f'{args.quantized_fp8_model_path}')
        quant_scales = get_scaling_factors(args.quantized_fp8_model_path,
                                           num_layers=args.n_layer,
                                           quant_mode=args.quant_mode)
        quantize_kwargs = {"quant_scales": quant_scales}
    tensorrt_llm_gpt = quantize_model(tensorrt_llm_gpt, args.quant_mode,
                                      **quantize_kwargs)
    if args.per_group:
        assert args.weight_only_precision == 'int4_awq', "We only support awq for now."
        load_from_awq_mpt(tensorrt_llm_mpt=tensorrt_llm_gpt,
                          quant_ckpt_path=args.quant_ckpt_path,
                          mapping=mapping,
                          dtype=args.dtype,
                          ft_model_dir=args.model_dir)
    else:
        load_from_ft(tensorrt_llm_gpt, args.model_dir, rank, args.world_size,
                     args.dtype, args.use_parallel_embedding,
                     args.embedding_sharding_dim, share_embedding_table)
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
        if args.per_group:
            network.plugin_config.set_weight_only_groupwise_quant_matmul_plugin(
                dtype=args.dtype)
        else:
            network.plugin_config.set_weight_only_quant_matmul_plugin(
                dtype=args.dtype)

    if args.world_size > 1:
        network.plugin_config.set_nccl_plugin(args.dtype)

    if args.use_lookup_plugin:
        # Use the plugin for the embedding parallelism and sharing
        network.plugin_config.set_lookup_plugin(dtype=args.dtype)

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
            max_num_tokens=args.max_num_tokens,
            prompt_embedding_table_size=args.max_prompt_embedding_table_size)
        tensorrt_llm_gpt(*inputs)

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
    timing_cache_file = args.timing_cache if args.timing_cache else args.output_dir / "model.cache"
    timing_cache = timing_cache_file

    builder = Builder()
    apply_query_key_layer_scaling = False
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
            timing_cache=timing_cache,
            profiling_verbosity=args.profiling_verbosity,
            tensor_parallel=args.world_size,  # TP only
            parallel_build=args.parallel_build,
            num_layers=args.n_layer,
            num_heads=args.n_head,
            num_kv_heads=args.n_kv_head if args.n_kv_head else args.n_head,
            hidden_size=args.n_embd,
            vocab_size=args.vocab_size,
            hidden_act=args.hidden_act,
            max_position_embeddings=args.n_positions,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            max_batch_size=args.max_batch_size,
            max_beam_width=args.max_beam_width,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            max_num_tokens=args.max_num_tokens,
            int8=int8_trt_flag,
            opt_level=args.builder_opt,
            strongly_typed=args.strongly_typed,
            use_prompt_tuning=args.max_prompt_embedding_table_size > 0,
            quant_mode=args.quant_mode,
            use_parallel_embedding=args.use_parallel_embedding)

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
        del engine

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
