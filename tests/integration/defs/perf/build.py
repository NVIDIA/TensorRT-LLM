# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import multiprocessing as mp
import os
import time
from collections import OrderedDict

# isort: off
import torch
import tensorrt as trt
# isort: on

from allowed_configs import (get_allowed_models, get_build_config,
                             get_model_config, get_model_family)

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import BuildConfig, Builder, build
from tensorrt_llm.functional import LayerNormPositionType, LayerNormType
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import PretrainedConfig
from tensorrt_llm.models.modeling_utils import QuantConfig, optimize_model
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantAlgo, QuantMode
from tensorrt_llm.quantization.quantize import quantize

WEIGHT_STREAMING_DISABLED_VAL = "1.0"


def parse_arguments():
    parser = argparse.ArgumentParser(description='Build TensorRT LLM models.')
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        required=True,
                        choices=get_allowed_models(),
                        help='Specify model you want to build.')
    parser.add_argument(
        '--mode',
        type=str,
        default="plugin",
        choices=['ootb', 'plugin', 'plugin-ifb', 'ootb-except-mha'],
        help=
        ('Choose mode between ootb/plugin/ootb-except-mha. '
         '\"ootb\" means the engines will be built without any plugins, '
         '\"plugin\" means the engines will be built with tuned recipe of using plugins.'
         '\"plugin-ifb\" will include additional options required for inflight batching.'
         '\"ootb-except-mha\" means the engines will be built with only attention plugins.'
         ))

    parser.add_argument(
        '--dtype',
        type=str,
        default='float16',
        choices=['float16', 'bfloat16', 'float32'],
        help='Choose data type between float16/bfloat16/float32.')
    parser.add_argument(
        '--quantization',
        type=str,
        default=None,
        choices=[
            'fp8', 'fp8_gemm', 'fp8_kv_cache', 'int8_sq_per_tensor',
            'int8_sq_per_token_channel', 'int8_weight_only', 'int4_weight_only',
            'int4_weight_only_awq', 'int4_weight_only_gptq'
        ],
        help="Optimize the model with specified quantization recipe")

    parser.add_argument(
        '--input_timing_cache',
        type=str,
        default=None,
        help=
        'The path to read timing cache, will be ignored if the file does not exist'
    )
    parser.add_argument('--output_timing_cache',
                        type=str,
                        default='model.cache',
                        help='The path to write timing cache')

    parser.add_argument(
        '--profiling_verbosity',
        type=str,
        default='layer_names_only',
        choices=['layer_names_only', 'detailed', 'none'],
        help=
        'The profiling verbosity for the generated TRT engine. Set to detailed can inspect tactic choices and kernel parameters.'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default="error",
        choices=['verbose', 'info', 'warning', 'error', 'internal_error'],
        help=
        'Choose log level between verbose/info/warning/error/internal_error.')

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='TensorRT engines will be saved to the specified path.')

    parser.add_argument(
        '--max_beam_width',
        type=int,
        default=None,
        help=
        ('If this option is specified, it will override the max beam width of '
         'TRT engines to the specified value instead of using pre-defined one'))
    parser.add_argument(
        '--max_input_len',
        type=int,
        default=None,
        help=
        ('If this option is specified, it will override the max input len of '
         'TRT engines to the specified value instead of using pre-defined one'))
    parser.add_argument(
        '--max_seq_len',
        '--max_decoder_seq_len',
        dest='max_seq_len',
        type=int,
        default=None,
        help=
        ('If this option is specified, it will override the max sequence len of '
         'TRT engines to the specified value instead of using pre-defined one'))
    parser.add_argument(
        '--max_batch_size',
        type=int,
        default=None,
        help=
        ('If this option is specified, it will override the max batch size of '
         'TRT engines to the specified value instead of using pre-defined one'))
    parser.add_argument('--force_num_layer_1',
                        default=False,
                        action='store_true',
                        help='Quick sanity check with num_layer=1.')
    parser.add_argument('--serial_build',
                        default=False,
                        action='store_true',
                        help="Build engines serially")
    parser.add_argument(
        '--multiple_profiles',
        default=False,
        action='store_true',
        help=
        'This option will benefit performance, but will increase the engine build time.'
    )

    parser.add_argument(
        '--weight_streaming',
        default=False,
        action='store_true',
        help=
        'Specify whether offloading weights to CPU and streaming loading at runtime.',
    )

    parser.add_argument(
        '--monitor_memory',
        default=False,
        action='store_true',
        help='Specify whether turning on the memory monitor flag.',
    )

    parser.add_argument(
        '--rank',
        type=int,
        default=None,
        help=
        "The rank of the model to be built, only used when --serial_build is specified"
    )
    parser.add_argument(
        '--world_size',
        type=int,
        default=None,
        help=
        "The number of gpus to be used for inference, only used when --serial_build is specified"
    )
    parser.add_argument(
        '--opt_batch_size',
        type=int,
        default=None,
        help=
        "If opt_batch_size option is specified, it will override the opt batch size."
        "This flag only takes effect when `--mode=ootb` is added. For other modes, please use --opt_num_tokens to replace it."
    )

    parser.add_argument(
        '--opt_num_tokens',
        type=int,
        default=None,
        help="It equals to max_batch_size*max_beam_width by default, set this "
        "value as close as possible to the actual number of tokens on your workload. "
        "Note that this argument might be removed in the future."
        "This flag only takes effect when `--mode` is not `ootb`. For ootb mode, please use --opt_batch_size to replace it."
    )

    return parser.parse_args()


def serialize_engine(engine, path):
    logger.info(f'Serializing engine to {path}...')
    tik = time.time()
    with open(path, 'wb') as f:
        # engine object is already complies with python buffer protocol, no need to
        # convert it to bytearray before write, converting to bytearray consumes lots of memory
        f.write(engine)
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')


def get_quant_config(quantization: str):
    if quantization == "fp8":
        return QuantConfig(quant_algo=QuantAlgo.FP8,
                           kv_cache_quant_algo=QuantAlgo.FP8)
    elif quantization == "fp8_gemm":
        return QuantConfig(quant_algo=QuantAlgo.FP8)
    elif quantization == "fp8_kv_cache":
        return QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)
    elif quantization == "int8_sq_per_tensor":
        return QuantConfig(quant_algo=QuantAlgo.W8A8_SQ_PER_TENSOR_PLUGIN)
    elif quantization == "int8_sq_per_token_channel":
        return QuantConfig(
            quant_algo=QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN)
    elif quantization == "int8_sq_per_channel_ootb":
        return QuantConfig(quant_algo=QuantAlgo.W8A8_SQ_PER_CHANNEL)
    elif quantization == "int8_weight_only":
        return QuantConfig(quant_algo=QuantAlgo.W8A16)
    elif quantization == "int4_weight_only":
        return QuantConfig(quant_algo=QuantAlgo.W4A16)
    elif quantization == "int4_weight_only_awq":
        return QuantConfig(quant_algo=QuantAlgo.W4A16_AWQ)
    elif quantization == "int4_weight_only_gptq":
        return QuantConfig(quant_algo=QuantAlgo.W4A16_GPTQ)
    elif quantization is None:
        return QuantConfig()
    else:
        raise Exception(f"Unexpected quantization: {quantization}")


def build_gpt(args):
    build_config = get_build_config(args.model)
    build_config = BuildConfig.from_dict(build_config)
    model_config = get_model_config(args.model)
    if args.force_num_layer_1:
        model_config['num_layers'] = 1

    # More parameters
    if args.serial_build and args.rank is not None and args.world_size is not None:
        runtime_rank = args.rank
        world_size = args.world_size
    else:
        runtime_rank = tensorrt_llm.mpi_rank()
        world_size = tensorrt_llm.mpi_world_size()
    if not args.serial_build:
        torch.cuda.set_device(runtime_rank)

    if args.profiling_verbosity != "layer_names_only":
        build_config.profiling_verbosity = args.profiling_verbosity

    if args.max_batch_size is not None:
        build_config.max_batch_size = args.max_batch_size
    if args.max_input_len is not None:
        build_config.max_input_len = args.max_input_len
    if args.max_seq_len is not None:
        build_config.max_seq_len = args.max_seq_len
    if args.max_beam_width is not None:
        build_config.max_beam_width = args.max_beam_width
    if args.opt_batch_size is not None:
        build_config.opt_batch_size = args.opt_batch_size
    if args.opt_num_tokens is not None:
        build_config.opt_num_tokens = args.opt_num_tokens
    build_config.weight_streaming = getattr(args, "weight_streaming", False)
    build_config.max_num_tokens = build_config.max_batch_size * max(
        build_config.max_input_len, build_config.max_beam_width)

    if args.mode != "ootb" and args.opt_batch_size is not None:
        raise Exception(
            f'--opt_batch_size only used when mode is ootb. Please using --opt_num_tokens instead it.'
        )
    if args.mode == "ootb" and args.opt_num_tokens is not None:
        raise Exception(
            f'--opt_num_tokens does not support ootb mode. Please using --opt_batch_size instead it.'
        )

    quant_config = get_quant_config(args.quantization)
    quant_algo = quant_config.quant_algo
    kv_cache_quant_algo = quant_config.kv_cache_quant_algo
    quant_mode = quant_config.quant_mode

    # Initialize Module
    family = get_model_family(args.model)
    if family == "gpt":
        if model_config['num_kv_heads'] is None:
            model_config['num_kv_heads'] = model_config['num_heads']
        if model_config['inter_size'] is None:
            model_config['inter_size'] = model_config['hidden_size'] * 4
        if model_config['position_embedding_type'] is None:
            model_config['position_embedding_type'] = 'learned_absolute'

        config = {
            'architecture': 'GPTForCausalLM',
            'dtype': args.dtype,
            'num_hidden_layers': model_config['num_layers'],
            'num_attention_heads': model_config['num_heads'],
            'num_key_value_heads': model_config['num_kv_heads'],
            'hidden_size': model_config['hidden_size'],
            'intermediate_size': model_config['inter_size'],
            'norm_epsilon': 1e-05,
            'vocab_size': model_config['vocab_size'],
            'position_embedding_type': model_config['position_embedding_type'],
            'max_position_embeddings': model_config['n_positions'],
            'hidden_act': model_config['hidden_act'],
            'quantization': {
                'quant_algo': quant_algo,
                'kv_cache_quant_algo': kv_cache_quant_algo,
                'group_size': 128,
            },
            'mapping': {
                'world_size': world_size,
                'tp_size': world_size,
                'rank': runtime_rank
            },
            'bias': model_config['bias'],
            'apply_query_key_layer_scaling': False,
            'rotary_pct': model_config['rotary_pct'],
            'moe': {
                'num_experts': model_config["moe_num_experts"],
                'top_k': model_config["moe_top_k"],
            },
        }
        config = PretrainedConfig.from_dict(config)
        tensorrt_llm_model = tensorrt_llm.models.GPTForCausalLM(config)

    elif family == "opt":
        config = {
            'architecture': 'OPTForCausalLM',
            'dtype': args.dtype,
            'vocab_size': model_config['vocab_size'],
            'hidden_size': model_config['hidden_size'],
            'num_hidden_layers': model_config['num_layers'],
            'num_attention_heads': model_config['num_heads'],
            'hidden_act': model_config['hidden_act'],
            'max_position_embeddings': model_config['n_positions'],
            'mapping': {
                'world_size': world_size,
                'tp_size': world_size,
                'rank': runtime_rank
            },
            'use_parallel_embedding': False,
            'embedding_sharding_dim': 0,
            'do_layer_norm_before': model_config['do_layer_norm_before'],
            'quantization': {
                'quant_algo': quant_algo,
                'kv_cache_quant_algo': kv_cache_quant_algo,
                'group_size': 128
            }
        }
        config = PretrainedConfig.from_dict(config)
        tensorrt_llm_model = tensorrt_llm.models.OPTForCausalLM(config)

    elif family == "llama":
        config = {
            'architecture':
            'LlamaForCausalLM',
            'dtype':
            args.dtype,
            'logits_dtype':
            'float32',
            'num_hidden_layers':
            model_config['num_layers'],
            'num_attention_heads':
            model_config['num_heads'],
            'num_key_value_heads':
            model_config['num_heads'] if model_config['num_kv_heads'] is None
            else model_config['num_kv_heads'],
            'hidden_size':
            model_config['hidden_size'],
            'intermediate_size':
            model_config['inter_size'],
            'vocab_size':
            model_config['vocab_size'],
            'position_embedding_type':
            'rope_gpt_neox',
            'max_position_embeddings':
            model_config['n_positions'],
            'hidden_act':
            model_config['hidden_act'],
            'quantization': {
                'quant_algo': quant_algo,
                'kv_cache_quant_algo': kv_cache_quant_algo,
                'group_size': 128
            },
            'mapping': {
                'world_size': world_size,
                'tp_size': world_size,
                'moe_tp_size': world_size,
                'moe_ep_size': 1,
                'rank': runtime_rank
            },
            'moe': {
                'num_experts': model_config["moe_num_experts"],
                'top_k': model_config["moe_top_k"],
            }
        }
        config = PretrainedConfig.from_dict(config)
        tensorrt_llm_model = tensorrt_llm.models.LLaMAForCausalLM(config)
        tensorrt_llm_model = optimize_model(tensorrt_llm_model,
                                            use_fused_mlp=True)
    elif family == "gptj":
        config = {
            'architecture': 'GPTJForCausalLM',
            'dtype': args.dtype,
            'vocab_size': model_config['vocab_size'],
            'hidden_size': model_config['hidden_size'],
            'num_hidden_layers': model_config['num_layers'],
            'num_attention_heads': model_config['num_heads'],
            'hidden_act': model_config['hidden_act'],
            'max_position_embeddings': model_config['n_positions'],
            'position_embedding_type': 'rope_gptj',
            'rotary_dim': model_config['rotary_dim'],
            'mapping': {
                'world_size': world_size,
                'tp_size': world_size,
                'rank': runtime_rank
            },
            'use_parallel_embedding': False,
            'embedding_sharding_dim': 0,
            'do_layer_norm_before': model_config['do_layer_norm_before'],
            'quantization': {
                'quant_algo': quant_algo,
                'kv_cache_quant_algo': kv_cache_quant_algo,
                'group_size': 128
            }
        }
        config = PretrainedConfig.from_dict(config)
        tensorrt_llm_model = tensorrt_llm.models.GPTJForCausalLM(config)

    elif family == "gptneox":
        config = {
            'architecture':
            'GPTNeoXForCausalLM',
            'dtype':
            args.dtype,
            'num_hidden_layers':
            model_config['num_layers'],
            'num_attention_heads':
            model_config['num_heads'],
            'hidden_size':
            model_config['hidden_size'],
            'vocab_size':
            model_config['vocab_size'],
            'position_embedding_type':
            'rope_gpt_neox',
            'max_position_embeddings':
            model_config['n_positions'],
            'rotary_emb_base':
            10000,
            'rotary_pct':
            1.0 * model_config['rotary_dim'] * model_config['num_heads'] /
            model_config['hidden_size'],
            'hidden_act':
            model_config['hidden_act'],
            'mapping': {
                'world_size': world_size,
                'tp_size': world_size,
                'rank': runtime_rank
            },
            'use_parallel_embedding':
            False,
            'embedding_sharding_dim':
            0,
            'quantization': {
                'quant_algo': quant_algo,
                'kv_cache_quant_algo': kv_cache_quant_algo,
                'group_size': 128,
            }
        }
        config = PretrainedConfig.from_dict(config)
        tensorrt_llm_model = tensorrt_llm.models.GPTNeoXForCausalLM(config)

    elif family == "chatglm":
        config = {
            'architecture': 'ChatGLMModel',
            'dtype': args.dtype,
            'num_hidden_layers': model_config['num_layers'],
            'num_attention_heads': model_config['num_heads'],
            'num_key_value_heads': model_config['num_kv_heads'],
            'hidden_size': model_config['hidden_size'],
            'intermediate_size': model_config['inter_size'],
            'norm_epsilon': 1e-5,
            'vocab_size': model_config['vocab_size'],
            'position_embedding_type': 'chatglm',
            'max_position_embeddings': model_config['n_positions'],
            'hidden_act': model_config['hidden_act'],
            'quantization': {
                'quant_algo': quant_algo,
                'kv_cache_quant_algo': kv_cache_quant_algo
            },
            'mapping': {
                'world_size': world_size,
                'tp_size': world_size,
                'rank': runtime_rank
            },
            'chatglm_version': 'chatglm',
            'add_bias_linear': True,
            'add_qkv_bias': True,
            'apply_query_key_layer_scaling': False,
            'apply_residual_connection_post_layernorm': False,
            'rmsnorm': False,
        }
        config = PretrainedConfig.from_dict(config)
        tensorrt_llm_model = tensorrt_llm.models.ChatGLMForCausalLM(config)

    elif family in ["chatglm2", "chatglm3"]:
        config = {
            'architecture': 'ChatGLMModel',
            'dtype': args.dtype,
            'num_hidden_layers': model_config['num_layers'],
            'num_attention_heads': model_config['num_heads'],
            'num_key_value_heads': model_config['num_kv_heads'],
            'hidden_size': model_config['hidden_size'],
            'intermediate_size': model_config['inter_size'],
            'norm_epsilon': 1e-5,
            'vocab_size': model_config['vocab_size'],
            'position_embedding_type': 'rope_gptj',
            'max_position_embeddings': model_config['n_positions'],
            'hidden_act': model_config['hidden_act'],
            'quantization': {
                'quant_algo': quant_algo,
                'kv_cache_quant_algo': kv_cache_quant_algo
            },
            'mapping': {
                'world_size': world_size,
                'tp_size': world_size,
                'rank': runtime_rank
            },
            'chatglm_version': family,
            'add_bias_linear': False,
            'add_qkv_bias': True,
            'apply_query_key_layer_scaling': False,
            'apply_residual_connection_post_layernorm': False,
            'rmsnorm': True,
        }
        config = PretrainedConfig.from_dict(config)
        tensorrt_llm_model = tensorrt_llm.models.ChatGLMForCausalLM(config)

    elif family == "glm":
        config = {
            'architecture': 'GLMModel',
            'dtype': args.dtype,
            'num_hidden_layers': model_config['num_layers'],
            'num_attention_heads': model_config['num_heads'],
            'num_key_value_heads': model_config['num_kv_heads'],
            'hidden_size': model_config['hidden_size'],
            'intermediate_size': model_config['inter_size'],
            'norm_epsilon': 1e-5,
            'vocab_size': model_config['vocab_size'],
            'position_embedding_type': 'learned_absolute',
            'max_position_embeddings': model_config['n_positions'],
            'hidden_act': model_config['hidden_act'],
            'quantization': {
                'quant_algo': quant_algo,
                'kv_cache_quant_algo': kv_cache_quant_algo
            },
            'mapping': {
                'world_size': world_size,
                'tp_size': world_size,
                'rank': runtime_rank
            },
            'chatglm_version': 'glm',
            'add_bias_linear': True,
            'add_qkv_bias': True,
            'apply_query_key_layer_scaling': False,
            'apply_residual_connection_post_layernorm': False,
            'rmsnorm': False,
        }
        config = PretrainedConfig.from_dict(config)
        tensorrt_llm_model = tensorrt_llm.models.ChatGLMForCausalLM(config)

    elif family == "bloom":
        config = {
            'architecture': 'BloomForCausalLM',
            'dtype': args.dtype,
            'vocab_size': model_config['vocab_size'],
            'hidden_size': model_config['hidden_size'],
            'num_hidden_layers': model_config['num_layers'],
            'num_attention_heads': model_config['num_heads'],
            'hidden_act': model_config['hidden_act'],
            'max_position_embeddings': model_config['n_positions'],
            'mapping': {
                'world_size': world_size,
                'tp_size': world_size,
                'rank': runtime_rank
            },
            'use_parallel_embedding': (args.model == 'bloom_176b'),
            'embedding_sharding_dim': 0,
            'quantization': {
                'quant_algo': quant_algo,
                'kv_cache_quant_algo': kv_cache_quant_algo,
                'group_size': 128
            }
        }
        config = PretrainedConfig.from_dict(config)
        tensorrt_llm_model = tensorrt_llm.models.BloomForCausalLM(config)
    elif family == "falcon":
        config = {
            'architecture':
            'FalconForCausalLM',
            'dtype':
            args.dtype,
            'num_hidden_layers':
            model_config['num_layers'],
            'num_attention_heads':
            model_config['num_heads'],
            'num_key_value_heads':
            model_config['num_heads'] if model_config['num_kv_heads'] is None
            else model_config['num_kv_heads'],
            'hidden_size':
            model_config['hidden_size'],
            'vocab_size':
            model_config['vocab_size'],
            'position_embedding_type':
            'alibi_with_scale'
            if model_config['use_alibi'] else 'rope_gpt_neox',
            'max_position_embeddings':
            model_config['n_positions'],
            'hidden_act':
            model_config['hidden_act'],
            'quantization': {
                'quant_algo': quant_algo,
                'kv_cache_quant_algo': kv_cache_quant_algo,
                'group_size': 128
            },
            'mapping': {
                'world_size': world_size,
                'tp_size': world_size,
                'rank': runtime_rank
            },
            'bias':
            model_config['bias'],
            'parallel_attention':
            model_config['parallel_attention'],
            'new_decoder_architecture':
            model_config['new_decoder_architecture'],
        }
        if quant_mode.is_weight_only() and quant_mode.has_per_group_scaling():
            config['quantization'].update({
                'has_zero_point': False,
                'pre_quant_scale': True,
            })
        config = PretrainedConfig.from_dict(config)
        tensorrt_llm_model = tensorrt_llm.models.FalconForCausalLM(config)

    elif family == "baichuan":
        config = {
            'architecture':
            'BaichuanForCausalLM',
            'dtype':
            args.dtype,
            'logits_dtype':
            'float32',
            'vocab_size':
            model_config['vocab_size'],
            'max_position_embeddings':
            model_config['n_positions'],
            'hidden_size':
            model_config['hidden_size'],
            'num_hidden_layers':
            model_config['num_layers'],
            'num_attention_heads':
            model_config['num_heads'],
            'num_key_value_heads':
            model_config['num_heads'],
            'hidden_act':
            model_config['hidden_act'],
            'intermediate_size':
            model_config['inter_size'],
            'position_embedding_type':
            'alibi_with_scale' if '7b' in args.model else 'rope_gpt_neox',
            'quantization': {
                'quant_algo': quant_algo,
                'kv_cache_quant_algo': kv_cache_quant_algo,
                'group_size': 128
            },
            'mapping': {
                'world_size': world_size,
                'tp_size': world_size,
                'rank': runtime_rank
            },
        }
        config = PretrainedConfig.from_dict(config)
        tensorrt_llm_model = tensorrt_llm.models.BaichuanForCausalLM(config)

    elif family == "internlm":
        config = {
            'architecture':
            'LlamaForCausalLM',
            'dtype':
            args.dtype,
            'num_hidden_layers':
            model_config['num_layers'],
            'num_attention_heads':
            model_config['num_heads'],
            'num_key_value_heads':
            model_config['num_heads'] if model_config['num_kv_heads'] is None
            else model_config['num_kv_heads'],
            'hidden_size':
            model_config['hidden_size'],
            'intermediate_size':
            model_config['inter_size'],
            'vocab_size':
            model_config['vocab_size'],
            'position_embedding_type':
            'rope_gpt_neox',
            'max_position_embeddings':
            model_config['n_positions'],
            'hidden_act':
            model_config['hidden_act'],
            'quantization': {
                'quant_algo': quant_algo,
                'kv_cache_quant_algo': kv_cache_quant_algo
            },
            'mapping': {
                'world_size': world_size,
                'tp_size': world_size,
                'rank': runtime_rank
            },
            'attn_bias':
            model_config['bias'],
        }
        if quant_mode.is_weight_only():
            if 'awq' in args.quantization:
                config['quantization'].update({
                    "group_size": 128,
                    "has_zero_point": False,
                    "pre_quant_scale": True,
                })
            elif 'gptq' in args.quantization:
                config['quantization'].update({
                    "group_size": 128,
                    "has_zero_point": True,
                    "pre_quant_scale": False,
                })
        config = PretrainedConfig.from_dict(config)
        tensorrt_llm_model = tensorrt_llm.models.LLaMAForCausalLM(config)

    elif family == "qwen":
        config = {
            'architecture':
            'QWenForCausalLM',
            'dtype':
            args.dtype,
            'num_hidden_layers':
            model_config['num_layers'],
            'num_attention_heads':
            model_config['num_heads'],
            'num_key_value_heads':
            model_config['num_heads'] if model_config['num_kv_heads'] is None
            else model_config['num_kv_heads'],
            'seq_length':
            model_config['n_positions'],
            'hidden_size':
            model_config['hidden_size'],
            'intermediate_size':
            model_config['inter_size'],
            'vocab_size':
            model_config['vocab_size'],
            'position_embedding_type':
            'rope_gpt_neox',
            'max_position_embeddings':
            model_config['n_positions'],
            'hidden_act':
            model_config['hidden_act'],
            'quantization': {
                'group_size': 128,
                'quant_algo': quant_algo,
                'kv_cache_quant_algo': kv_cache_quant_algo
            },
            'mapping': {
                'world_size': world_size,
                'tp_size': world_size,
                'rank': runtime_rank
            },
            'moe': {
                'num_experts': model_config["moe_num_experts"],
                'top_k': model_config["moe_top_k"],
            },
            'qwen_type':
            'qwen',
        }
        config = PretrainedConfig.from_dict(config)
        tensorrt_llm_model = tensorrt_llm.models.QWenForCausalLM(config)
    elif family == "qwen2":
        config = {
            'architecture':
            'QWenForCausalLM',
            'dtype':
            args.dtype,
            'num_hidden_layers':
            model_config['num_layers'],
            'num_attention_heads':
            model_config['num_heads'],
            'num_key_value_heads':
            model_config['num_heads'] if model_config['num_kv_heads'] is None
            else model_config['num_kv_heads'],
            'seq_length':
            model_config['n_positions'],
            'hidden_size':
            model_config['hidden_size'],
            'intermediate_size':
            model_config['inter_size'],
            'vocab_size':
            model_config['vocab_size'],
            'position_embedding_type':
            'rope_gpt_neox',
            'max_position_embeddings':
            model_config['n_positions'],
            'hidden_act':
            model_config['hidden_act'],
            'quantization': {
                'group_size': 128,
                'quant_algo': quant_algo,
                'kv_cache_quant_algo': kv_cache_quant_algo
            },
            'mapping': {
                'world_size': world_size,
                'tp_size': world_size,
                'rank': runtime_rank
            },
            'moe': {
                'num_experts': model_config["moe_num_experts"],
                'top_k': model_config["moe_top_k"],
            },
            'qwen_type':
            'qwen2',
        }
        config = PretrainedConfig.from_dict(config)
        tensorrt_llm_model = tensorrt_llm.models.QWenForCausalLM(config)
    elif family == "mamba":
        config = {
            'architecture': 'MambaForCausalLM',
            'dtype': args.dtype,
            'vocab_size': model_config['vocab_size'],
            'hidden_size': model_config['hidden_size'],
            'num_hidden_layers': model_config['num_layers'],
            'num_attention_heads': model_config['num_heads'],
            'hidden_act': model_config['hidden_act'],
            'state_size': model_config['state_size'],
            'conv_kernel': model_config['conv_kernel'],
            'layer_types': model_config['layer_types'],
            'rnn_hidden_size': model_config['rnn_hidden_size'],
            'rnn_head_size': model_config['rnn_head_size'],
            'rnn_conv_dim_size': model_config['rnn_conv_dim_size'],
            'rms_norm': True,
            'residual_in_fp32': True,
            'pad_vocab_size_multiple': 8,
            'use_bias': model_config['use_bias'],
            'mamba_version': model_config['mamba_version'],
            'ssm_rmsnorm': model_config['ssm_rmsnorm'],
            'ngroups': model_config['ngroups'],
            'chunk_size': model_config['chunk_size'],
        }
        config = PretrainedConfig.from_dict(config)
        tensorrt_llm_model = tensorrt_llm.models.MambaForCausalLM(config)
    elif family == "recurrentgemma":
        config = {
            'architecture': 'RecurrentGemmaForCausalLM',
            'dtype': args.dtype,
            'vocab_size': model_config['vocab_size'],
            'hidden_size': model_config['hidden_size'],
            'num_hidden_layers': model_config['num_layers'],
            'num_attention_heads': model_config['num_heads'],
            'num_key_value_heads': model_config['num_kv_heads'],
            'hidden_act': model_config['hidden_act'],
            'intermediate_size': model_config['inter_size'],
            'rms_norm': True,
            'norm_epsilon': 1e-6,
            'quantization': {
                'group_size': 128,
                'quant_algo': quant_algo,
                'kv_cache_quant_algo': kv_cache_quant_algo
            },
            'mapping': {
                'world_size': world_size,
                'tp_size': world_size,
                'rank': runtime_rank
            },
            'position_embedding_type': model_config['position_embedding_type'],
            'rotary_percentage': model_config['rotary_pct'],
            'max_position_embeddings': model_config['n_positions'],
            'conv_kernel': model_config['conv_kernel'],
            'state_size': model_config['state_size'],
            'layer_types': model_config['layer_types'],
            'rnn_hidden_size': model_config['rnn_hidden_size'],
            'rnn_head_size': model_config['rnn_head_size'],
            'rnn_conv_dim_size': model_config['rnn_conv_dim_size'],
            'logits_soft_cap': model_config['logits_soft_cap'],
            'rotary_pct': model_config['rotary_pct'],
        }
        config = PretrainedConfig.from_dict(config)
        tensorrt_llm_model = tensorrt_llm.models.RecurrentGemmaForCausalLM(
            config)
        tensorrt_llm_model = optimize_model(tensorrt_llm_model,
                                            use_fused_mlp=True,
                                            use_fused_rg_lru=True)
    elif family == "phi3":
        config = {
            'architecture':
            'PhiForCausalLM',
            'dtype':
            args.dtype,
            'rotary_base':
            10000.0,
            'num_hidden_layers':
            model_config['num_layers'],
            'num_attention_heads':
            model_config['num_heads'],
            'num_key_value_heads':
            model_config['num_heads'] if model_config['num_kv_heads'] is None
            else model_config['num_kv_heads'],
            'hidden_size':
            model_config['hidden_size'],
            'intermediate_size':
            model_config['inter_size'],
            'vocab_size':
            model_config['vocab_size'],
            'position_embedding_type':
            'rope_gpt_neox',
            'max_position_embeddings':
            model_config['n_positions'],
            'hidden_act':
            model_config['hidden_act'],
            'quantization': {
                'quant_algo': quant_algo,
                'kv_cache_quant_algo': kv_cache_quant_algo,
                'group_size': 128
            },
            'mapping': {
                'world_size': world_size,
                'tp_size': world_size,
                'rank': runtime_rank
            },
        }
        config = PretrainedConfig.from_dict(config)
        tensorrt_llm_model = tensorrt_llm.models.Phi3ForCausalLM(config)

    else:
        raise Exception(f'Unexpected model: {args.model}')

    # Plugins
    build_config.plugin_config.to_legacy_setting()
    if args.mode in ['plugin', 'plugin-ifb']:
        build_config.plugin_config.gpt_attention_plugin = args.dtype
        build_config.plugin_config.set_context_fmha(ContextFMHAType.enabled)
        build_config.plugin_config.remove_input_padding = True
        build_config.plugin_config.moe_plugin = args.dtype
        build_config.plugin_config.mamba_conv1d_plugin = args.dtype

        if args.quantization is None or "fp8" not in args.quantization:
            build_config.plugin_config.gemm_plugin = args.dtype

        # Quantization plugins.
        use_weight_only = quant_mode.is_weight_only()
        if use_weight_only:
            build_config.plugin_config.weight_only_quant_matmul_plugin = args.dtype

        use_smooth_quant = quant_mode.has_act_and_weight_quant()
        if use_smooth_quant:
            build_config.plugin_config.set_smooth_quant_plugins(
                dtype=args.dtype)

        use_qserve = quant_mode.has_act_and_weight_quant() and quant_mode._any(
            QuantMode.INT4_WEIGHTS)
        if use_qserve:
            build_config.plugin_config.set_qserve_plugins(dtype=args.dtype)

        # Inflight batching
        if args.mode == 'plugin-ifb':
            build_config.plugin_config.enable_paged_kv_cache()
            build_config.plugin_config.paged_state = True
    elif args.mode == 'ootb-except-mha':
        build_config.plugin_config.gpt_attention_plugin = args.dtype
        build_config.plugin_config.set_context_fmha(ContextFMHAType.enabled)
        build_config.plugin_config.remove_input_padding = True

    if args.mode not in ('plugin', 'plugin-ifb'):
        build_config.plugin_config.smooth_quant_plugins = False

    if world_size > 1:
        build_config.plugin_config.set_nccl_plugin(dtype=args.dtype)

    if args.multiple_profiles:
        build_config.plugin_config.multiple_profiles = True

    # Enable trt monitor memory for perf tests
    build_config.monitor_memory = args.monitor_memory

    start = time.time()
    engine = build(tensorrt_llm_model, build_config)
    assert engine.engine is not None, f'Failed to build engine for rank {runtime_rank}'
    build_time = round(time.time() - start, 2)

    engine.save(args.output_dir)

    return engine, build_time


def build_bert(args):
    family = get_model_family(args.model)
    build_config = get_build_config(args.model, return_dict=False)
    model_config = get_model_config(args.model)
    if args.force_num_layer_1:
        model_config['num_layers'] = 1

    # More parameters
    if args.serial_build and args.rank is not None and args.world_size is not None:
        runtime_rank = args.rank
        world_size = args.world_size
    else:
        runtime_rank = tensorrt_llm.mpi_rank()
        world_size = tensorrt_llm.mpi_world_size()
    if not args.serial_build:
        torch.cuda.set_device(runtime_rank)

    num_kv_heads = model_config['num_heads'] \
        if model_config['num_kv_heads'] is None else model_config['num_kv_heads']
    max_batch_size = build_config.max_batch_size \
        if args.max_batch_size is None else args.max_batch_size
    max_input_len = build_config.max_input_len \
        if args.max_input_len is None else args.max_input_len
    bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
    inlen_range = [1, (max_input_len + 1) // 2, max_input_len]

    is_weight_streaming = getattr(args, "weight_streaming", False)

    builder = Builder()
    builder_config = builder.create_builder_config(
        name=args.model,
        precision=args.dtype,
        timing_cache=args.input_timing_cache,
        profiling_verbosity=args.profiling_verbosity,
        tensor_parallel=world_size,  # TP only
        parallel_build=True,
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        num_kv_heads=num_kv_heads,
        hidden_size=model_config['hidden_size'],
        vocab_size=model_config['vocab_size'],
        hidden_act=model_config['hidden_act'],
        max_position_embeddings=model_config['n_positions'],
        max_batch_size=max_batch_size,
        max_input_len=max_input_len,
        strongly_typed=True,
        weight_streaming=is_weight_streaming,
        monitor_memory=args.monitor_memory,
    )
    engine_name = '{}_{}_tp{}_rank{}.engine'.format(args.model, args.dtype,
                                                    world_size, runtime_rank)

    # Initialize model
    config = {
        'architecture': 'BertModel',
        'dtype': args.dtype,
        'num_hidden_layers': model_config['num_layers'],
        'num_attention_heads': model_config['num_heads'],
        'hidden_size': model_config['hidden_size'],
        'vocab_size': model_config['vocab_size'],
        'position_embedding_type': 'learned_absolute',
        'max_position_embeddings': model_config['n_positions'],
        'hidden_act': model_config['hidden_act'],
        'type_vocab_size': model_config['type_vocab_size'],
        'pad_token_id':
        None if family == 'bert' else 1,  # hard code for RoBERTa here
        'is_roberta': (family == 'roberta'),
        'mapping': {
            'world_size': world_size,
            'tp_size': world_size,
            'rank': runtime_rank
        },
    }
    config = PretrainedConfig.from_dict(config)
    tensorrt_llm_bert = tensorrt_llm.models.BertModel(config)

    # Module -> Network
    network = builder.create_network()
    network.trt_network.name = engine_name
    network.plugin_config.to_legacy_setting()

    # Plugins
    if args.mode == 'plugin':
        network.plugin_config.bert_attention_plugin = args.dtype
        network.plugin_config.gemm_plugin = args.dtype
        network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
    elif args.mode == 'ootb-except-mha':
        network.plugin_config.bert_attention_plugin = args.dtype
        network.plugin_config.set_context_fmha(ContextFMHAType.enabled)

    if world_size > 1:
        network.plugin_config.set_nccl_plugin(dtype=args.dtype)

    with net_guard(network):
        # Prepare
        network.set_named_parameters(tensorrt_llm_bert.named_parameters())

        # Forward
        input_ids = tensorrt_llm.Tensor(
            name='input_ids',
            dtype=trt.int32,
            shape=[-1, -1],
            dim_range=OrderedDict([('batch_size', [bs_range]),
                                   ('input_len', [inlen_range])]),
        )
        input_lengths = tensorrt_llm.Tensor(name='input_lengths',
                                            dtype=trt.int32,
                                            shape=[-1],
                                            dim_range=OrderedDict([
                                                ('batch_size', [bs_range])
                                            ]))
        hidden_states = tensorrt_llm_bert(input_ids=input_ids,
                                          input_lengths=input_lengths)

        # Mark outputs
        hidden_states_dtype = str_dtype_to_trt(args.dtype)
        hidden_states.mark_output('hidden_states', hidden_states_dtype)

    # Network -> Engine
    start = time.time()
    engine = builder.build_engine(network, builder_config)
    assert engine is not None, f'Failed to build engine for rank {runtime_rank}'
    build_time = round(time.time() - start, 2)

    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        serialize_path = os.path.join(args.output_dir, engine_name)
        serialize_engine(engine, serialize_path)
        if runtime_rank == 0:
            config_path = os.path.join(args.output_dir, 'config.json')
            builder_config.plugin_config = network.plugin_config
            builder.save_config(builder_config, config_path)
            if args.output_timing_cache:
                # Save timing cache to output_dir if not absolute path
                timing_cache_path = args.output_timing_cache if os.path.isabs(
                    args.output_timing_cache) else os.path.join(
                        args.output_dir, args.output_timing_cache)
                ok = builder.save_timing_cache(builder_config,
                                               timing_cache_path)
                if not ok:
                    logger.warning("Failed to save timing cache.")

    return engine, build_time


def enc_dec_build_helper(component, build_config, model_config, args):
    # More parameters
    if args.serial_build and args.rank is not None and args.world_size is not None:
        runtime_rank = args.rank
        world_size = args.world_size
    else:
        runtime_rank = tensorrt_llm.mpi_rank()
        world_size = tensorrt_llm.mpi_world_size()
    if not args.serial_build:
        torch.cuda.set_device(runtime_rank)

    family = get_model_family(args.model)
    logits_dtype = 'float32'
    if family == 'bart':
        q_scaling = 1.0
        has_attention_qkvo_bias = True
        has_mlp_bias = True
        has_model_final_layernorm = False
        has_position_embedding = True
        has_embedding_layernorm = True
        layernorm_type = LayerNormType.LayerNorm
        relative_attention = False
        layernorm_position = LayerNormPositionType.pre_layernorm if model_config.get(
            'normalize_before', True) else LayerNormPositionType.post_layernorm
        rescale_before_lm_head = False
    elif family == 'whisper':
        q_scaling = 1.0
        has_position_embedding = True
        relative_attention = False
        has_embedding_layernorm = False
        has_attention_qkvo_bias = True
        has_mlp_bias = True
        has_model_final_layernorm = True
        layernorm_position = LayerNormPositionType.pre_layernorm
        layernorm_type = LayerNormType.LayerNorm
        rescale_before_lm_head = False
        logits_dtype = args.dtype
        model_config['n_mels']
    else:
        q_scaling = 1 / model_config['head_size']**.5
        has_attention_qkvo_bias = False
        has_mlp_bias = False
        has_model_final_layernorm = True
        has_position_embedding = False
        has_embedding_layernorm = False
        layernorm_type = LayerNormType.RmsNorm
        relative_attention = True
        layernorm_position = LayerNormPositionType.pre_layernorm
        if family == 't5':
            rescale_before_lm_head = True
        else:
            rescale_before_lm_head = False

    quant_config = get_quant_config(args.quantization)
    quant_mode = quant_config.quant_mode
    use_weight_only = quant_mode.is_weight_only()

    # Plugins
    build_config.plugin_config.to_legacy_setting()
    if args.mode in ['plugin', 'plugin-ifb']:
        build_config.plugin_config.bert_attention_plugin = args.dtype
        build_config.plugin_config.gpt_attention_plugin = args.dtype
        build_config.plugin_config.set_context_fmha(ContextFMHAType.enabled)
        build_config.plugin_config.gemm_plugin = args.dtype
        build_config.plugin_config.remove_input_padding = True
        build_config.plugin_config.enable_paged_kv_cache()
        build_config.plugin_config.paged_state = True
        if use_weight_only:
            build_config.plugin_config.weight_only_quant_matmul_plugin = args.dtype
    elif args.mode == 'ootb-except-mha':
        build_config.plugin_config.bert_attention_plugin = args.dtype
        build_config.plugin_config.gpt_attention_plugin = args.dtype
        build_config.plugin_config.set_context_fmha(ContextFMHAType.enabled)

    if world_size > 1:
        build_config.plugin_config.set_nccl_plugin(dtype=args.dtype)

    # build engine
    mapping = Mapping(world_size=world_size,
                      rank=runtime_rank,
                      tp_size=world_size,
                      pp_size=1)  # TP only

    if component == 'encoder':
        if family == 'whisper':
            pretrained_config = PretrainedConfig.from_dict({
                'architecture':
                "WhisperEncoder",
                'dtype':
                args.dtype,
                'num_hidden_layers':
                model_config['num_layers'],
                'num_attention_heads':
                model_config['num_heads'],
                'hidden_size':
                model_config['hidden_size'],
                'has_position_embedding':
                has_position_embedding,
                'n_mels':
                model_config['n_mels'],
                'max_position_embeddings':
                1500,
                'vocab_size':
                model_config['vocab_size'],
                'hidden_act':
                "gelu",
                'num_languages':
                100,
                'mapping': {
                    'world_size': mapping.world_size,
                    'tp_size': mapping.tp_size,
                    'pp_size': mapping.pp_size,
                    'rank': mapping.rank,
                },
            })
            tllm_model = tensorrt_llm.models.WhisperEncoder(pretrained_config)
            if use_weight_only:
                tllm_model = quantize(tllm_model, quant_config)
        else:
            pretrained_config = PretrainedConfig.from_dict({
                'architecture':
                "EncoderModel",
                'dtype':
                args.dtype,
                'logits_dtype':
                logits_dtype,
                'num_hidden_layers':
                model_config['num_layers'],
                'num_attention_heads':
                model_config['num_heads'],
                'hidden_size':
                model_config['hidden_size'],
                'norm_epsilon':
                1e-6,
                'vocab_size':
                model_config['vocab_size'],
                'hidden_act':
                model_config['hidden_act'],
                'mapping': {
                    'world_size': mapping.world_size,
                    'tp_size': mapping.tp_size,
                    'pp_size': mapping.pp_size,
                    'rank': mapping.rank,
                },
                'use_parallel_embedding':
                False,
                'embedding_sharding_dim':
                0,
                'max_position_embeddings':
                model_config.get('n_positions', 0),
                'use_prompt_tuning':
                False,
                'head_size':
                model_config['head_size'],
                'has_position_embedding':
                has_position_embedding,
                'layernorm_type':
                layernorm_type,
                'has_attention_qkvo_bias':
                has_attention_qkvo_bias,
                'has_mlp_bias':
                has_mlp_bias,
                'has_model_final_layernorm':
                has_model_final_layernorm,
                'has_embedding_layernorm':
                has_embedding_layernorm,
                'has_embedding_scale':
                model_config.get('has_embedding_scale', False),
                'intermediate_size':
                model_config['ffn_hidden_size'],
                'q_scaling':
                q_scaling,
                'layernorm_position':
                layernorm_position,
                'relative_attention':
                relative_attention,
                'max_distance':
                model_config.get('max_distance', 0),
                'num_buckets':
                model_config.get('num_buckets', 0),
                'model_type':
                family,
            })
            tllm_model = tensorrt_llm.models.EncoderModel(pretrained_config)
    elif component == 'decoder':
        pretrained_config = PretrainedConfig.from_dict({
            'architecture':
            "DecoderModel",
            'dtype':
            args.dtype,
            'logits_dtype':
            logits_dtype,
            'num_hidden_layers':
            model_config['num_layers'],
            'num_attention_heads':
            model_config['num_heads'],
            'hidden_size':
            model_config['hidden_size'],
            'norm_epsilon':
            1e-6,
            'vocab_size':
            model_config['vocab_size'],
            'hidden_act':
            model_config['hidden_act'],
            'mapping': {
                'world_size': mapping.world_size,
                'tp_size': mapping.tp_size,
                'pp_size': mapping.pp_size,
                'rank': mapping.rank,
            },
            'use_parallel_embedding':
            False,
            'embedding_sharding_dim':
            0,
            'max_position_embeddings':
            model_config.get('n_positions', 0),
            'use_prompt_tuning':
            False,
            'head_size':
            model_config['head_size'],
            'has_position_embedding':
            has_position_embedding,
            'layernorm_type':
            layernorm_type,
            'has_attention_qkvo_bias':
            has_attention_qkvo_bias,
            'has_mlp_bias':
            has_mlp_bias,
            'has_model_final_layernorm':
            has_model_final_layernorm,
            'has_embedding_layernorm':
            has_embedding_layernorm,
            'has_embedding_scale':
            model_config.get('has_embedding_scale', False),
            'intermediate_size':
            model_config['ffn_hidden_size'],
            'q_scaling':
            q_scaling,
            'layernorm_position':
            layernorm_position,
            'relative_attention':
            relative_attention,
            'max_distance':
            model_config.get('max_distance', 0),
            'num_buckets':
            model_config.get('num_buckets', 0),
            'model_type':
            family,
            'rescale_before_lm_head':
            rescale_before_lm_head,
            'encoder_hidden_size':
            model_config['hidden_size'],
            'encoder_num_heads':
            model_config['num_heads'],
            'encoder_head_size':
            model_config['head_size'],
            'skip_cross_kv':
            model_config['skip_cross_kv'],
            'use_implicit_relative_attention':
            model_config['use_implicit_relative_attention'],
            'decoder_start_token_id':
            model_config['decoder_start_token_id'],
        })
        tllm_model = tensorrt_llm.models.DecoderModel(pretrained_config)
        if use_weight_only and family == 'whisper':
            tllm_model = quantize(tllm_model, quant_config)

    tllm_model.precompute_relative_attention_bias(build_config)

    start = time.time()
    engine = build(tllm_model, build_config)
    assert engine.engine is not None, f'Failed to build engine for rank {runtime_rank}'
    build_time = round(time.time() - start, 2)

    engine.save(os.path.join(args.output_dir, component))

    return engine, model_config, build_time


def build_enc_dec(args):
    build_config = get_build_config(args.model)
    build_config = BuildConfig.from_dict(build_config)
    model_config = get_model_config(args.model)
    if args.force_num_layer_1:
        model_config['num_layers'] = 1

    if args.profiling_verbosity != "layer_names_only":
        build_config.profiling_verbosity = args.profiling_verbosity

    if args.max_batch_size is not None:
        build_config.max_batch_size = args.max_batch_size
    if args.max_input_len is not None:
        build_config.max_encoder_input_len = args.max_input_len
        build_config.max_input_len = args.max_input_len
    if args.max_seq_len is not None:
        build_config.max_seq_len = args.max_seq_len
    if args.max_beam_width is not None:
        build_config.max_beam_width = args.max_beam_width
    if args.opt_batch_size is not None:
        build_config.opt_batch_size = args.opt_batch_size
    if args.opt_num_tokens is not None:
        build_config.opt_num_tokens = args.opt_num_tokens
    build_config.max_num_tokens = build_config.max_batch_size * max(
        build_config.max_encoder_input_len, build_config.max_beam_width)

    encoder_max_seq_len = build_config.max_encoder_input_len
    decoder_max_seq_len = build_config.max_seq_len

    # Enable trt monitor memory for perf tests
    build_config.monitor_memory = args.monitor_memory

    # for encoder, input len and output len both equal to max_encoder_input_len
    build_config.max_input_len = encoder_max_seq_len
    build_config.max_seq_len = encoder_max_seq_len
    encoder_engine, encoder_model_config, encoder_build_time = enc_dec_build_helper(
        component='encoder',
        build_config=build_config,
        model_config=model_config,
        args=args)

    # for decoder, input len equals to 1 and output len equals to max_seq_len
    build_config.max_input_len = 1
    build_config.max_seq_len = decoder_max_seq_len
    decoder_engine, decoder_model_config, decoder_build_time = enc_dec_build_helper(
        component='decoder',
        build_config=build_config,
        model_config=model_config,
        args=args)

    return encoder_engine, decoder_engine, encoder_model_config, decoder_model_config, encoder_build_time, decoder_build_time


def main(args):
    logger.set_level(args.log_level)
    if args.model in get_allowed_models(benchmark_type="gpt"):
        engine = build_gpt(args)[0]
        engine_size = engine.engine.nbytes
    elif args.model in get_allowed_models(benchmark_type="bert"):
        engine = build_bert(args)[0]
        engine_size = engine.nbytes
    elif args.model in get_allowed_models(benchmark_type="enc_dec"):
        encoder_engine, decoder_engine = build_enc_dec(args)[:2]
        engine_size = encoder_engine.engine.nbytes + decoder_engine.engine.nbytes
    else:
        raise Exception(f'Unexpected model: {args.model}')

    # Print engine size for CI/CD to track.
    logger.info(
        f"Total engine size per GPU is {engine_size / 1048576:.2f} MiB.")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    args = parse_arguments()
    main(args)
