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
from tensorrt_llm._common import check_max_num_tokens
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.layers import MoeConfig, PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import quantize_model
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.profiler import check_gpt_mem_usage
from tensorrt_llm.quantization import QuantMode

from weight import load_from_ft, parse_ft_config, check_embedding_share  # isort:skip

MODEL_NAME = "gpt"


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


def override_args_from_model_dir(args: argparse.Namespace) -> None:
    if args.model_dir is not None:
        logger.info(f"Setting model configuration from {args.model_dir}.")
        parsed_params = parse_ft_config(Path(args.model_dir) / "config.ini")
        args.n_embd = parsed_params["n_embd"]
        args.n_head = parsed_params["n_head"]
        args.n_layer = parsed_params["n_layer"]
        args.n_positions = parsed_params["n_positions"]
        args.vocab_size = parsed_params["vocab_size"]
        args.hidden_act = parsed_params["hidden_act"]
        if parsed_params["rotary_pct"] is not None:
            args.rotary_pct = parsed_params["rotary_pct"]
        if parsed_params["rotary_base"] is not None:
            args.rotary_base = parsed_params["rotary_base"]
        if parsed_params["rotary_scaling"] is not None:
            args.rotary_scaling = parsed_params["rotary_scaling"]
        args.bias = parsed_params["bias"]
        args.dtype = parsed_params["dtype"]
        args.inter_size = parsed_params["inter_size"]
        args.multi_query_mode = parsed_params["multi_query_mode"]


def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='world size, only support tensor parallelism now')
    parser.add_argument('--model_dir', type=str, default=None)
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
    parser.add_argument('--hidden_act', type=str, default='gelu')
    parser.add_argument('--rotary_base', type=float, default=10000.0)
    parser.add_argument('--rotary_scaling', nargs=2, type=str, default=None)
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
    parser.add_argument(
        "--multi_query_mode",
        "-mq",
        default=False,
        action='store_true',
        help=
        "Whether this model uses multi-query attention mechanism (default: False)"
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
                        default=128,
                        help='Number of tokens per block in paged KV cache')
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

    parser.add_argument('--enable_fp8', default=False, action='store_true')
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
        help=
        'Define the max number of tokens supported by the engine, note that it takes no effect if --remove_input_padding is not set'
    )
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
    parser.add_argument(
        '--use_lora_plugin',
        nargs='?',
        const=None,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help="Activates the lora plugin which enables embedding sharing.")
    parser.add_argument(
        '--max_draft_len',
        type=int,
        default=0,
        help=
        'Maximum lengths of draft tokens for speculative decoding target model.'
    )
    parser.add_argument(
        '--use_paged_context_fmha',
        action='store_true',
        help=
        'Activates paged context FMHA. This mode of the context FMHA is required for chunked context, speculative decoding and reuse of KV cache blocks. Context FMHA performance is worse when this mode is on.'
    )
    parser.add_argument(
        '--use_context_fmha_for_generation',
        action='store_true',
        help=
        'Activates context FMHA for generation phase instead of MMHA. Use only for testing and debug.'
    )
    parser.add_argument(
        '--lora_target_modules',
        nargs='+',
        default=None,
        choices=[
            "attn_qkv",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_dense",
            "mlp_h_to_4h",
            "mlp_gate",
            "mlp_4h_to_h",
        ],
        help=
        "Add lora in which modules. Only be activated when use_lora_plugin is enabled."
    )
    parser.add_argument(
        '--max_lora_rank',
        type=int,
        default=64,
        help='maximum lora rank for different lora modules. '
        'It is used to compute the workspace size of lora plugin.')
    parser.add_argument(
        '--moe_num_experts',
        default=0,
        type=int,
        help='Specify the number of experts to use for MOE layers')
    parser.add_argument(
        '--moe_top_k',
        default=0,
        type=int,
        help=
        'Specify the top_k value to use for MOE layers. Default to 1 if --moe_num_experts is set'
    )
    parser.add_argument(
        '--moe_tp_mode',
        default=MoeConfig.ParallelismMode.TENSOR_PARALLEL,
        type=int,
        help=
        'Controls how to distribute experts in TP. Check layers/moe.py for accepted values',
    )
    parser.add_argument(
        '--moe_renorm_mode',
        default=MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
        type=int,
        help=
        'Controls renormalization after gate logits. Check layers/moe.py for accepted values',
    )
    args = parser.parse_args(args)
    logger.set_level(args.log_level)

    if not args.remove_input_padding:
        if args.use_gpt_attention_plugin:
            logger.warning(
                f"It is recommended to specify --remove_input_padding when using GPT attention plugin"
            )

    args.bias = not args.no_bias
    if args.inter_size is None:
        args.inter_size = 4 * args.n_embd

    override_args_from_model_dir(args)
    plugins_args = [
        'use_gpt_attention_plugin', 'use_gemm_plugin', 'use_layernorm_plugin',
        'use_lookup_plugin', 'use_lora_plugin'
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

    assert (math.log2(args.tokens_per_block).is_integer()
            ), "tokens_per_block must be power of 2"
    if args.enable_context_fmha or args.enable_context_fmha_fp32_acc:
        assert (args.tokens_per_block >=
                128), "Context fMHA requires >= 128 tokens per block"

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

    if args.rotary_scaling is not None:
        assert args.use_gpt_attention_plugin, "RoPE scaling is only supported through GPT attention plugin."
        rotary_scaling = {
            "type": args.rotary_scaling[0],
            "factor": float(args.rotary_scaling[1])
        }
        assert rotary_scaling["type"] in ["linear", "dynamic"]
        assert rotary_scaling["factor"] > 1.0
        args.rotary_scaling = rotary_scaling

    args.max_num_tokens = check_max_num_tokens(
        max_num_tokens=args.max_num_tokens,
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        remove_input_padding=args.remove_input_padding)

    if args.moe_num_experts and args.moe_top_k == 0:
        args.moe_top_k = 1
    args.moe_config = MoeConfig(args.moe_num_experts, args.moe_top_k,
                                args.moe_tp_mode,
                                args.moe_renorm_mode).validate()

    if args.gather_all_token_logits:
        args.gather_context_logits = True
        args.gather_generation_logits = True

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
            'Engine will try to share embedding and language modeling weights. Note: Flag --use_lookup_plugin and --use_gemm_plugin are also needed for now.'
        )

    # Initialize Module
    tensorrt_llm_gpt = tensorrt_llm.models.GPTLMHeadModel(
        num_layers=args.n_layer,
        num_heads=args.n_head,
        hidden_size=args.n_embd,
        inter_size=args.inter_size,
        vocab_size=args.vocab_size,
        hidden_act=args.hidden_act,
        max_position_embeddings=args.n_positions,
        position_embedding_type=PositionEmbeddingType.learned_absolute
        if args.rotary_pct == 0.0 else PositionEmbeddingType.rope_gpt_neox,
        rotary_embedding_percentage=args.rotary_pct,
        rotary_base=args.rotary_base,
        rotary_scaling=args.rotary_scaling,
        dtype=kv_dtype,
        logits_dtype=args.logits_dtype,
        mapping=Mapping(world_size=args.world_size,
                        rank=rank,
                        tp_size=args.world_size),  # TP only
        apply_query_key_layer_scaling=builder_config.
        apply_query_key_layer_scaling,
        quant_mode=args.quant_mode,
        bias=args.bias,
        num_kv_heads=1 if args.multi_query_mode else args.n_head,
        use_prompt_tuning=args.max_prompt_embedding_table_size > 0,
        use_parallel_embedding=args.use_parallel_embedding,
        embedding_sharding_dim=args.embedding_sharding_dim,
        share_embedding_table=share_embedding_table,
        moe_config=args.moe_config,
        max_lora_rank=args.max_lora_rank,
    )

    if args.use_smooth_quant or args.use_weight_only:
        tensorrt_llm_gpt = quantize_model(tensorrt_llm_gpt, args.quant_mode)

    if args.model_dir is not None:
        gpt_dummy_fp8_scaling_factors = {
            'fc_act': [0.5 for _ in range(args.n_layer)],
            'fc_weights': [0.5 for _ in range(args.n_layer)],
            'proj_act': [0.5 for _ in range(args.n_layer)],
            'proj_weights': [0.5 for _ in range(args.n_layer)],
            'qkv_act': [0.5 for _ in range(args.n_layer)],
            'qkv_weights': [0.5 for _ in range(args.n_layer)],
            'qkv_output': [0.5 for _ in range(args.n_layer)],
            'dense_act': [0.5 for _ in range(args.n_layer)],
            'dense_weights': [0.5 for _ in range(args.n_layer)],
        }

        load_from_ft(tensorrt_llm_gpt,
                     args.model_dir,
                     rank,
                     args.world_size,
                     args.dtype,
                     args.use_parallel_embedding,
                     args.embedding_sharding_dim,
                     share_embedding_table,
                     scaling_factors=gpt_dummy_fp8_scaling_factors
                     if args.enable_fp8 else None)

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
    if args.use_lora_plugin:
        network.plugin_config.set_lora_plugin(dtype=args.use_lora_plugin)

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

    if args.use_lookup_plugin:
        # Use the plugin for the embedding parallelism and sharing
        network.plugin_config.set_lookup_plugin(dtype=args.dtype)

    if args.use_paged_context_fmha or args.max_draft_len > 0:
        assert args.enable_context_fmha or args.enable_context_fmha_fp32_acc, "context fmha must be enabled"
        network.plugin_config.set_paged_context_fmha()

    if args.use_context_fmha_for_generation:
        logger.warning(
            f'use_context_fmha_for_generation is set. This flag must be used only for testing'
        )
        assert args.use_gpt_attention_plugin and args.paged_kv_cache and args.use_paged_context_fmha, "use_context_fmha_for_generation must be used with paged KV cache and attention."
        network.plugin_config.set_context_fmha_for_generation()

    with net_guard(network):
        # Prepare
        network.set_named_parameters(tensorrt_llm_gpt.named_parameters())

        # Forward
        inputs = tensorrt_llm_gpt.prepare_inputs(
            max_batch_size=args.max_batch_size,
            max_input_len=args.max_input_len,
            max_seq_len=args.max_input_len + args.max_output_len,
            use_cache=True,
            max_beam_width=args.max_beam_width,
            max_num_tokens=args.max_num_tokens,
            prompt_embedding_table_size=args.max_prompt_embedding_table_size,
            gather_context_logits=args.gather_context_logits,
            gather_generation_logits=args.gather_generation_logits,
            max_draft_len=args.max_draft_len,
            lora_target_modules=args.lora_target_modules)
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
        # NOTE: when only int8 kv cache is used together with paged kv cache no int8 tensors are exposed to TRT
        int8_trt_flag = args.quant_mode.has_act_or_weight_quant() or (
            args.paged_kv_cache == False
            and args.quant_mode.has_int8_kv_cache())
        num_kv_heads = 1 if args.multi_query_mode else args.n_head
        builder_config = builder.create_builder_config(
            name=MODEL_NAME,
            precision=args.dtype,
            timing_cache=timing_cache,
            profiling_verbosity=args.profiling_verbosity,
            tensor_parallel=args.world_size,  # TP only
            parallel_build=args.parallel_build,
            num_layers=args.n_layer,
            num_heads=args.n_head,
            num_kv_heads=num_kv_heads,
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
            max_draft_len=args.max_draft_len,
            int8=int8_trt_flag,
            opt_level=args.builder_opt,
            strongly_typed=args.strongly_typed,
            max_prompt_embedding_table_size=args.
            max_prompt_embedding_table_size,
            gather_context_logits=args.gather_context_logits,
            gather_generation_logits=args.gather_generation_logits,
            quant_mode=args.quant_mode,
            use_parallel_embedding=args.use_parallel_embedding,
            lora_target_modules=args.lora_target_modules,
        )

        engine_name = get_engine_name(MODEL_NAME, args.dtype, args.world_size,
                                      cur_rank)
        engine = build_rank_engine(builder, builder_config, engine_name,
                                   cur_rank, args)
        assert engine is not None, f'Failed to build engine for rank {cur_rank}'

        local_num_kv_heads = (num_kv_heads + args.world_size -
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
            max_seq_len=args.max_input_len + args.max_output_len,
            local_num_kv_heads=local_num_kv_heads,
            head_size=args.n_embd / args.n_head,
            num_layers=args.n_layer)

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
