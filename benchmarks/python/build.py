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
import multiprocessing as mp
import os
import time
from collections import OrderedDict

import tensorrt as trt
import torch
from allowed_configs import (get_allowed_models, get_build_config,
                             get_model_family)
from base_benchmark import get_engine_name, serialize_engine

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.layers import MoeConfig, PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.models import PretrainedConfig, quantize_model
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode


def parse_arguments():
    parser = argparse.ArgumentParser(description='Build TensorRT-LLM models.')
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
        choices=['ootb', 'plugin', 'ootb-except-mha'],
        help=
        ('Choose mode between ootb/plugin/ootb-except-mha. '
         '\"ootb\" means the engines will be built without any plugins, '
         '\"plugin\" means the engines will be built with tuned recipe of using plugins.'
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
        '--max_output_len',
        type=int,
        default=None,
        help=
        ('If this option is specified, it will override the max output len of '
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

    return parser.parse_args()


def get_quant_mode(quantization):
    quant_mode = QuantMode(0)
    strongly_typed = False
    use_smooth_quant = False
    per_token = False
    per_channel = False
    weight_only_precision = 'int8'

    if quantization == "fp8":
        strongly_typed = True
        quant_mode = quant_mode.set_fp8_qdq()
        quant_mode = quant_mode.set_fp8_kv_cache()

    elif quantization == "fp8_gemm":
        strongly_typed = True
        quant_mode = quant_mode.set_fp8_qdq()

    elif quantization == "fp8_kv_cache":
        strongly_typed = True
        quant_mode = quant_mode.set_fp8_kv_cache()

    elif quantization == "int8_sq_per_tensor":
        use_smooth_quant = True
        quant_mode = QuantMode.use_smooth_quant(per_token, per_channel)

    elif quantization == "int8_sq_per_token_channel":
        use_smooth_quant = True
        per_token = True
        per_channel = True
        quant_mode = QuantMode.use_smooth_quant(per_token, per_channel)

    elif quantization == "int8_weight_only":
        use_smooth_quant = False
        weight_only_precision = 'int8'
        quant_mode = QuantMode.use_weight_only(False)

    elif quantization == "int4_weight_only":
        weight_only_precision = 'int4'
        quant_mode = QuantMode.use_weight_only(True)

    elif quantization == "int4_weight_only_awq":
        weight_only_precision = 'int4_awq'
        quant_mode = QuantMode.from_description(quantize_weights=True,
                                                quantize_activations=False,
                                                per_token=False,
                                                per_channel=False,
                                                per_group=True,
                                                use_int4_weights=True)

    elif quantization == "int4_weight_only_gptq":
        weight_only_precision = 'int4_gptq'
        quant_mode = QuantMode.from_description(quantize_weights=True,
                                                quantize_activations=False,
                                                per_token=False,
                                                per_channel=False,
                                                per_group=True,
                                                use_int4_weights=True)

    elif quantization == None:
        pass

    else:
        raise Exception(f'Unexpected quantization: {quantization}')

    return quant_mode, strongly_typed, use_smooth_quant, weight_only_precision


def build_gpt(args):
    build_config = get_build_config(args.model)
    if args.force_num_layer_1:
        build_config['num_layers'] = 1

    # More parameters
    if args.serial_build and args.rank is not None and args.world_size is not None:
        runtime_rank = args.rank
        world_size = args.world_size
    else:
        runtime_rank = tensorrt_llm.mpi_rank()
        world_size = tensorrt_llm.mpi_world_size()
    if not args.serial_build:
        torch.cuda.set_device(runtime_rank)

    num_kv_heads = build_config['num_heads'] \
        if build_config['num_kv_heads'] is None else build_config['num_kv_heads']
    apply_query_key_layer_scaling = False
    max_batch_size = build_config['max_batch_size'] \
        if args.max_batch_size is None else args.max_batch_size
    max_input_len = build_config['max_input_len'] \
        if args.max_input_len is None else args.max_input_len
    max_output_len = build_config['max_output_len'] \
        if args.max_output_len is None else args.max_output_len
    max_beam_width = build_config['max_beam_width'] \
        if args.max_beam_width is None else args.max_beam_width
    quant_mode, strongly_typed, use_smooth_quant, weight_only_precision = get_quant_mode(
        args.quantization)
    use_weight_only = quant_mode.is_weight_only()

    builder = Builder()
    builder_config = builder.create_builder_config(
        name=args.model,
        precision=args.dtype,
        timing_cache=None,
        tensor_parallel=world_size,  # TP only
        parallel_build=True,
        num_layers=build_config['num_layers'],
        num_heads=build_config['num_heads'],
        num_kv_heads=num_kv_heads,
        hidden_size=build_config['hidden_size'],
        vocab_size=build_config['vocab_size'],
        hidden_act=build_config['hidden_act'],
        max_position_embeddings=build_config['n_positions'],
        apply_query_key_layer_scaling=apply_query_key_layer_scaling,
        max_batch_size=max_batch_size,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        int8=(quant_mode.has_act_and_weight_quant()
              or quant_mode.is_int8_weight_only()),
        quant_mode=quant_mode,
        use_refit=False,
        opt_level=build_config['builder_opt'],
        strongly_typed=strongly_typed)
    engine_name = get_engine_name(args.model, args.dtype, world_size,
                                  runtime_rank)

    kv_dtype = str_dtype_to_trt(args.dtype)

    # Initialize Module
    family = get_model_family(args.model)
    if family == "gpt":
        tensorrt_llm_model = tensorrt_llm.models.GPTLMHeadModel(
            num_layers=build_config['num_layers'],
            num_heads=build_config['num_heads'],
            hidden_size=build_config['hidden_size'],
            vocab_size=build_config['vocab_size'],
            hidden_act=build_config['hidden_act'],
            max_position_embeddings=build_config['n_positions'],
            dtype=kv_dtype,
            mapping=tensorrt_llm.Mapping(world_size=world_size,
                                         tp_size=world_size),  # TP only
            apply_query_key_layer_scaling=builder_config.
            apply_query_key_layer_scaling,
            position_embedding_type=PositionEmbeddingType.learned_absolute
            if build_config['position_embedding_type'] is None else
            PositionEmbeddingType[build_config['position_embedding_type']],
            rotary_embedding_percentage=build_config['rotary_pct'],
            quant_mode=quant_mode,
            bias=build_config['bias'],
            moe_config=MoeConfig(build_config["moe_num_experts"],
                                 build_config["moe_top_k"]))
    elif family == "opt":
        config = {
            'architecture': 'OPTForCausalLM',
            'dtype': args.dtype,
            'vocab_size': build_config['vocab_size'],
            'hidden_size': build_config['hidden_size'],
            'num_hidden_layers': build_config['num_layers'],
            'num_attention_heads': build_config['num_heads'],
            'hidden_act': build_config['hidden_act'],
            'max_position_embeddings': build_config['n_positions'],
            'mapping': {
                'world_size': world_size,
                'tp_size': world_size
            },
            'use_parallel_embedding': False,
            'share_embedding_table': False,
            'embedding_sharding_dim': 0,
            'do_layer_norm_before': build_config['do_layer_norm_before'],
            'quantization': {
                'use_smooth_quant':
                quant_mode.has_act_and_weight_quant(),
                'per_channel':
                quant_mode.has_per_channel_scaling(),
                'per_token':
                quant_mode.has_per_token_dynamic_scaling(),
                'per_group':
                quant_mode.has_per_group_scaling(),
                'group_size':
                128,
                'int8_kv_cache':
                quant_mode.has_int8_kv_cache(),
                'enable_fp8':
                quant_mode.has_fp8_qdq(),
                'fp8_kv_cache':
                quant_mode.has_fp8_kv_cache(),
                'use_weight_only':
                quant_mode.is_weight_only(),
                'weight_only_precision':
                'int8' if quant_mode.is_int8_weight_only() else 'int4',
            }
        }
        config = PretrainedConfig.from_dict(config)
        tensorrt_llm_model = tensorrt_llm.models.OPTForCausalLM(config)
    elif family == "llama":
        tensorrt_llm_model = tensorrt_llm.models.LLaMAForCausalLM(
            num_layers=build_config['num_layers'],
            num_heads=build_config['num_heads'],
            num_kv_heads=num_kv_heads,
            hidden_size=build_config['hidden_size'],
            vocab_size=build_config['vocab_size'],
            hidden_act=build_config['hidden_act'],
            max_position_embeddings=build_config['n_positions'],
            dtype=kv_dtype,
            mlp_hidden_size=build_config['inter_size'],
            mapping=tensorrt_llm.Mapping(world_size=world_size,
                                         tp_size=world_size),  # TP only
            quant_mode=quant_mode,
            use_fused_mlp=True,
            moe_config=MoeConfig(build_config["moe_num_experts"],
                                 build_config["moe_top_k"]))
    elif family == "gptj":
        tensorrt_llm_model = tensorrt_llm.models.GPTJForCausalLM(
            num_layers=build_config['num_layers'],
            num_heads=build_config['num_heads'],
            hidden_size=build_config['hidden_size'],
            vocab_size=build_config['vocab_size'],
            hidden_act=build_config['hidden_act'],
            max_position_embeddings=build_config['n_positions'],
            rotary_dim=build_config['rotary_dim'],
            dtype=kv_dtype,
            mapping=tensorrt_llm.Mapping(world_size=world_size,
                                         tp_size=world_size),  # TP only
            quant_mode=quant_mode)
    elif family == "gptneox":
        tensorrt_llm_model = tensorrt_llm.models.GPTNeoXForCausalLM(
            num_layers=build_config['num_layers'],
            num_heads=build_config['num_heads'],
            hidden_size=build_config['hidden_size'],
            vocab_size=build_config['vocab_size'],
            hidden_act=build_config['hidden_act'],
            max_position_embeddings=build_config['n_positions'],
            rotary_dim=build_config['rotary_dim'],
            dtype=kv_dtype,
            mapping=tensorrt_llm.Mapping(world_size=world_size,
                                         tp_size=world_size),  # TP only
            apply_query_key_layer_scaling=builder_config.
            apply_query_key_layer_scaling)
    elif family == "chatglm":
        tensorrt_llm_model = tensorrt_llm.models.ChatGLMHeadModel(
            num_layers=build_config['num_layers'],
            num_heads=build_config['num_heads'],
            hidden_size=build_config['hidden_size'],
            vocab_size=build_config['vocab_size'],
            hidden_act=build_config['hidden_act'],
            max_position_embeddings=build_config['n_positions'],
            dtype=kv_dtype,
            mapping=tensorrt_llm.Mapping(world_size=world_size,
                                         tp_size=world_size),  # TP only
            apply_query_key_layer_scaling=builder_config.
            apply_query_key_layer_scaling,
            quant_mode=quant_mode,
            model_name="chatglm_6b")

    elif family == "chatglm2":
        tensorrt_llm_model = tensorrt_llm.models.ChatGLMHeadModel(
            num_layers=build_config['num_layers'],
            num_heads=build_config['num_heads'],
            hidden_size=build_config['hidden_size'],
            vocab_size=build_config['vocab_size'],
            hidden_act=build_config['hidden_act'],
            max_position_embeddings=build_config['n_positions'],
            dtype=kv_dtype,
            mapping=tensorrt_llm.Mapping(world_size=world_size,
                                         tp_size=world_size),  # TP only
            apply_query_key_layer_scaling=builder_config.
            apply_query_key_layer_scaling,
            quant_mode=quant_mode,
            model_name="chatglm2_6b")

    elif family == "chatglm3":
        tensorrt_llm_model = tensorrt_llm.models.ChatGLMHeadModel(
            num_layers=build_config['num_layers'],
            num_heads=build_config['num_heads'],
            hidden_size=build_config['hidden_size'],
            vocab_size=build_config['vocab_size'],
            hidden_act=build_config['hidden_act'],
            max_position_embeddings=build_config['n_positions'],
            dtype=kv_dtype,
            mapping=tensorrt_llm.Mapping(world_size=world_size,
                                         tp_size=world_size),  # TP only
            apply_query_key_layer_scaling=builder_config.
            apply_query_key_layer_scaling,
            quant_mode=quant_mode,
            model_name="chatglm3_6b")

    elif family == "bloom":
        config = {
            'architecture': 'BloomForCausalLM',
            'dtype': args.dtype,
            'vocab_size': build_config['vocab_size'],
            'hidden_size': build_config['hidden_size'],
            'num_hidden_layers': build_config['num_layers'],
            'num_attention_heads': build_config['num_heads'],
            'hidden_act': build_config['hidden_act'],
            'max_position_embeddings': build_config['n_positions'],
            'mapping': {
                'world_size': world_size,
                'tp_size': world_size
            },
            'use_parallel_embedding': (args.model == 'bloom_176b'),
            'share_embedding_table': False,
            'embedding_sharding_dim': 0,
            'quantization': {
                'use_smooth_quant':
                quant_mode.has_act_and_weight_quant(),
                'per_channel':
                quant_mode.has_per_channel_scaling(),
                'per_token':
                quant_mode.has_per_token_dynamic_scaling(),
                'per_group':
                quant_mode.has_per_group_scaling(),
                'group_size':
                128,
                'int8_kv_cache':
                quant_mode.has_int8_kv_cache(),
                'enable_fp8':
                quant_mode.has_fp8_qdq(),
                'fp8_kv_cache':
                quant_mode.has_fp8_kv_cache(),
                'use_weight_only':
                quant_mode.is_weight_only(),
                'weight_only_precision':
                'int8' if quant_mode.is_int8_weight_only() else 'int4',
            }
        }
        config = PretrainedConfig.from_dict(config)
        tensorrt_llm_model = tensorrt_llm.models.BloomForCausalLM(config)
    elif family == "falcon":
        tensorrt_llm_model = tensorrt_llm.models.FalconForCausalLM(
            num_layers=build_config['num_layers'],
            num_heads=build_config['num_heads'],
            num_kv_heads=num_kv_heads,
            hidden_size=build_config['hidden_size'],
            vocab_size=build_config['vocab_size'],
            max_position_embeddings=build_config['n_positions'],
            dtype=kv_dtype,
            bias=build_config['bias'],
            quant_mode=quant_mode,
            use_alibi=build_config['use_alibi'],
            new_decoder_architecture=build_config['new_decoder_architecture'],
            parallel_attention=build_config['parallel_attention'],
            mapping=tensorrt_llm.Mapping(world_size=world_size,
                                         tp_size=world_size))
    else:
        raise Exception(f'Unexpected model: {args.model}')

    quant_kwargs = {}
    if family == "llama" and use_weight_only:
        if weight_only_precision == 'int4_awq':
            quant_kwargs = {
                "group_size": 128,
                "zero": False,
                "pre_quant_scale": True,
                "exclude_modules": [],
            }
        elif weight_only_precision == 'int4_gptq':
            quant_kwargs = {
                "group_size": 128,
                "zero": True,
                "pre_quant_scale": False,
            }
    if family not in ['opt', 'bloom']:
        tensorrt_llm_model = quantize_model(tensorrt_llm_model, quant_mode,
                                            **quant_kwargs)

    # Module -> Network
    network = builder.create_network()
    network.trt_network.name = engine_name

    # Plugins
    if args.mode == 'plugin':
        network.plugin_config.set_gpt_attention_plugin(dtype=args.dtype)
        network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
        network.plugin_config.enable_remove_input_padding()
        network.plugin_config.set_lookup_plugin(dtype=args.dtype)
        if args.quantization is None or "fp8" not in args.quantization:
            network.plugin_config.set_gemm_plugin(dtype=args.dtype)

        # Quantization plugins.
        if use_smooth_quant:
            network.plugin_config.set_smooth_quant_gemm_plugin(dtype=args.dtype)
            network.plugin_config.set_layernorm_quantization_plugin(
                dtype=args.dtype)
            network.plugin_config.set_quantize_tensor_plugin()
            network.plugin_config.set_quantize_per_token_plugin()
        elif use_weight_only:
            network.plugin_config.set_weight_only_quant_matmul_plugin(
                dtype=args.dtype)
        elif family == "llama" and quant_mode.has_act_and_weight_quant():
            # RMS norm plugin for SmoothQuant
            network.plugin_config.set_rmsnorm_quantization_plugin(
                dtype=args.dtype)
    elif args.mode == 'ootb-except-mha':
        network.plugin_config.set_gpt_attention_plugin(dtype=args.dtype)
        network.plugin_config.set_context_fmha(ContextFMHAType.enabled)

    if world_size > 1:
        network.plugin_config.set_nccl_plugin(
            dtype=args.dtype,
            use_custom_all_reduce=build_config["use_custom_all_reduce"])

    with net_guard(network):
        # Prepare
        network.set_named_parameters(tensorrt_llm_model.named_parameters())

        # Forward
        inputs = tensorrt_llm_model.prepare_inputs(max_batch_size,
                                                   max_input_len,
                                                   max_output_len, True,
                                                   max_beam_width)
        if family in ['opt', 'bloom']:
            tensorrt_llm_model(**inputs)
        else:
            tensorrt_llm_model(*inputs)

    if args.mode == 'plugin':
        tensorrt_llm.graph_rewriting.optimize(network)

    # Network -> Engine
    start = time.time()
    engine = builder.build_engine(network, builder_config)
    assert engine is not None, f'Failed to build engine for rank {runtime_rank}'
    build_time = round(time.time() - start, 2)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        serialize_path = os.path.join(args.output_dir, engine_name)
        serialize_engine(engine, serialize_path)
        if runtime_rank == 0:
            config_path = os.path.join(args.output_dir, 'config.json')
            builder_config.plugin_config = network.plugin_config
            builder.save_config(builder_config, config_path)
    return engine, build_time


def build_bert(args):
    build_config = get_build_config(args.model)
    if args.force_num_layer_1:
        build_config['num_layers'] = 1

    # More parameters
    if args.serial_build and args.rank is not None and args.world_size is not None:
        runtime_rank = args.rank
        world_size = args.world_size
    else:
        runtime_rank = tensorrt_llm.mpi_rank()
        world_size = tensorrt_llm.mpi_world_size()
    if not args.serial_build:
        torch.cuda.set_device(runtime_rank)

    num_kv_heads = build_config['num_heads'] \
        if build_config['num_kv_heads'] is None else build_config['num_kv_heads']
    max_batch_size = build_config['max_batch_size'] \
        if args.max_batch_size is None else args.max_batch_size
    max_input_len = build_config['max_input_len'] \
        if args.max_input_len is None else args.max_input_len
    bs_range = [1, (max_batch_size + 1) // 2, max_batch_size]
    inlen_range = [1, (max_input_len + 1) // 2, max_input_len]

    builder = Builder()
    builder_config = builder.create_builder_config(
        name=args.model,
        precision=args.dtype,
        timing_cache=None,
        tensor_parallel=world_size,  # TP only
        parallel_build=True,
        num_layers=build_config['num_layers'],
        num_heads=build_config['num_heads'],
        num_kv_heads=num_kv_heads,
        hidden_size=build_config['hidden_size'],
        vocab_size=build_config['vocab_size'],
        hidden_act=build_config['hidden_act'],
        max_position_embeddings=build_config['n_positions'],
        max_batch_size=max_batch_size,
        max_input_len=max_input_len,
        opt_level=build_config['builder_opt'])
    engine_name = get_engine_name(args.model, args.dtype, world_size,
                                  runtime_rank)

    # Initialize model
    tensorrt_llm_bert = tensorrt_llm.models.BertModel(
        num_layers=build_config['num_layers'],
        num_heads=build_config['num_heads'],
        hidden_size=build_config['hidden_size'],
        vocab_size=build_config['vocab_size'],
        hidden_act=build_config['hidden_act'],
        max_position_embeddings=build_config['n_positions'],
        type_vocab_size=build_config['type_vocab_size'],
        mapping=tensorrt_llm.Mapping(world_size=world_size, tp_size=world_size))

    # Module -> Network
    network = builder.create_network()
    network.trt_network.name = engine_name

    # Plugins
    if args.mode == 'plugin':
        network.plugin_config.set_bert_attention_plugin(dtype=args.dtype)
        network.plugin_config.set_gemm_plugin(dtype=args.dtype)
        network.plugin_config.enable_qk_half_accum()
        network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
    elif args.mode == 'ootb-except-mha':
        network.plugin_config.set_bert_attention_plugin(dtype=args.dtype)
        network.plugin_config.set_context_fmha(ContextFMHAType.enabled)

    if world_size > 1:
        network.plugin_config.set_nccl_plugin(
            dtype=args.dtype,
            use_custom_all_reduce=build_config["use_custom_all_reduce"])

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
    return engine, build_time


def main(args):
    logger.set_level(args.log_level)
    if args.model in get_allowed_models(benchmark_type="gpt"):
        build_gpt(args)
    elif args.model in get_allowed_models(benchmark_type="bert"):
        build_bert(args)
    else:
        raise Exception(f'Unexpected model: {args.model}')


if __name__ == '__main__':
    mp.set_start_method('spawn')
    args = parse_arguments()
    main(args)
