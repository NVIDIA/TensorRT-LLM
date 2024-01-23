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

# isort: off
import torch
import torch.multiprocessing as mp
import tensorrt as trt
# isort: on
from transformers import AutoConfig, AutoModelForCausalLM
from weight import (load_from_awq_qwen, load_from_ft, load_from_gptq_qwen,
                    load_from_hf_qwen)

import tensorrt_llm
from tensorrt_llm._common import check_max_num_tokens
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import quantize_model
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode

MODEL_NAME = "qwen"

import onnx
from onnx import TensorProto, helper

now_dir = os.path.dirname(os.path.abspath(__file__))


def trt_dtype_to_onnx(dtype):
    if dtype == trt.float16:
        return TensorProto.DataType.FLOAT16
    elif dtype == trt.float32:
        return TensorProto.DataType.FLOAT
    elif dtype == trt.int32:
        return TensorProto.DataType.INT32
    else:
        raise TypeError("%s is not supported" % dtype)


def to_onnx(network, path):
    inputs = []
    for i in range(network.num_inputs):
        network_input = network.get_input(i)
        inputs.append(
            helper.make_tensor_value_info(
                network_input.name, trt_dtype_to_onnx(network_input.dtype),
                list(network_input.shape)))

    outputs = []
    for i in range(network.num_outputs):
        network_output = network.get_output(i)
        outputs.append(
            helper.make_tensor_value_info(
                network_output.name, trt_dtype_to_onnx(network_output.dtype),
                list(network_output.shape)))

    nodes = []
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        layer_inputs = []
        for j in range(layer.num_inputs):
            ipt = layer.get_input(j)
            if ipt is not None:
                layer_inputs.append(layer.get_input(j).name)
        layer_outputs = [
            layer.get_output(j).name for j in range(layer.num_outputs)
        ]
        nodes.append(
            helper.make_node(str(layer.type),
                             name=layer.name,
                             inputs=layer_inputs,
                             outputs=layer_outputs,
                             domain="com.nvidia"))

    onnx_model = helper.make_model(helper.make_graph(nodes,
                                                     'attention',
                                                     inputs,
                                                     outputs,
                                                     initializer=None),
                                   producer_name='NVIDIA')
    onnx.save(onnx_model, path)


def get_engine_name(model, dtype, tp_size, pp_size, rank):
    if pp_size == 1:
        return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)
    return '{}_{}_tp{}_pp{}_rank{}.engine'.format(model, dtype, tp_size,
                                                  pp_size, rank)


def serialize_engine(engine, path):
    logger.info(f'Serializing engine to {path}...')
    tik = time.time()
    with open(path, 'wb') as f:
        f.write(engine)
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='world size, only support tensor parallelism now')
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--pp_size', type=int, default=1)
    parser.add_argument('--hf_model_dir', type=str, default=None)
    parser.add_argument('--ft_dir_path', type=str, default=None)
    parser.add_argument("--quant_ckpt_path", type=str, default=None)
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'bfloat16', 'float16'])
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
    parser.add_argument('--log_level',
                        type=str,
                        default='info',
                        choices=[
                            'internal_error',
                            'error',
                            'warning',
                            'info',
                            'verbose',
                        ])
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--n_layer', type=int, default=32)
    parser.add_argument('--n_positions', type=int, default=2048)
    parser.add_argument('--n_embd', type=int, default=4096)
    parser.add_argument('--n_head', type=int, default=32)
    parser.add_argument('--n_kv_head', type=int, default=32)
    parser.add_argument('--inter_size', type=int, default=11008)
    parser.add_argument('--hidden_act', type=str, default='silu')
    parser.add_argument('--max_batch_size', type=int, default=2)
    parser.add_argument('--max_input_len', type=int, default=2048)
    parser.add_argument('--max_output_len', type=int, default=2048)
    parser.add_argument('--max_beam_width', type=int, default=1)
    parser.add_argument('--rotary_base', type=float, default=10000.0)
    parser.add_argument('--rotary_scaling', nargs=2, type=str, default=None)
    parser.add_argument('--use_gpt_attention_plugin',
                        nargs='?',
                        type=str,
                        default=None,
                        choices=['float16', 'bfloat16', 'float32', None])
    parser.add_argument('--use_gemm_plugin',
                        nargs='?',
                        type=str,
                        default=None,
                        choices=['float16', 'bfloat16', 'float32', None])
    parser.add_argument('--parallel_build', default=False, action='store_true')
    parser.add_argument('--enable_context_fmha',
                        default=False,
                        action='store_true')
    parser.add_argument('--enable_context_fmha_fp32_acc',
                        default=False,
                        action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--enable_debug_output',
                        default=False,
                        action='store_true')
    parser.add_argument('--gpus_per_node', type=int, default=8)
    parser.add_argument('--builder_opt', type=int, default=None)
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
    parser.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="group size used in gptq/awq quantization.",
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
        choices=['int8', 'int4', 'int4_awq', 'int4_gptq'],
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
        help=
        'Define the max number of tokens supported by the engine, note that it takes no effect if --remove_input_padding is not set'
    )

    parser.add_argument(
        '--int8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. int8_kv_cache chooses int8 quantization for KV'
    )
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
        default=1,  # Meta does TP on hidden dim
        choices=[0, 1],
        help=
        'By default the embedding lookup table is sharded along vocab dimension (embedding_sharding_dim=0). '
        'To shard it along hidden dimension, set embedding_sharding_dim=1'
        'Note: embedding sharing is only enabled when embedding_sharding_dim = 0'
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
    parser.add_argument('--use_rmsnorm_plugin',
                        nargs='?',
                        const='float16',
                        type=str,
                        default=False,
                        choices=['float16', 'float32', 'bfloat16'])
    parser.add_argument(
        '--max_prompt_embedding_table_size',
        type=int,
        default=0,
        help='Setting to a value > 0 enables support for prompt tuning.')
    parser.add_argument(
        '--use_lookup_plugin',
        nargs='?',
        const=None,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help="Activates the lookup plugin which enables embedding sharing.")

    args = parser.parse_args()
    assert not (
        args.use_smooth_quant and args.use_weight_only
    ), "You cannot enable both SmoothQuant and INT8 weight-only together."

    if not args.remove_input_padding:
        if args.use_gpt_attention_plugin:
            logger.warning(
                f"It is recommended to specify --remove_input_padding when using GPT attention plugin"
            )

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

    if args.int8_kv_cache:
        args.quant_mode = args.quant_mode.set_int8_kv_cache()

    if args.hf_model_dir is not None:
        hf_config = AutoConfig.from_pretrained(
            args.hf_model_dir,
            trust_remote_code=True,
        )
        args.inter_size = hf_config.intermediate_size  # override the inter_size for QWen
        args.n_embd = hf_config.hidden_size
        args.n_head = hf_config.num_attention_heads
        if hasattr(hf_config, "num_key_value_heads"):
            args.n_kv_head = hf_config.num_key_value_heads
        args.n_layer = hf_config.num_hidden_layers
        args.n_positions = hf_config.max_position_embeddings
        args.vocab_size = hf_config.vocab_size
        args.hidden_act = "silu"
        args.kv_channels = hf_config.kv_channels
        args.rotary_emb_base = hf_config.rotary_emb_base

    if args.n_kv_head is not None and args.n_kv_head != args.n_head:
        assert (args.n_head % args.n_kv_head) == 0, \
            "MQA/GQA requires the number of heads to be divisible by the number of K/V heads."
        assert args.n_kv_head == args.tp_size, \
        "The current implementation of GQA requires the number of K/V heads to match the number of GPUs." \
        "This limitation will be removed in a future version."

    assert args.pp_size * args.tp_size == args.world_size

    if args.weight_only_precision == 'int4_awq' and args.quantize_lm_head:
        if args.vocab_size % 64 != 0:
            args.vocab_size = int((args.vocab_size + 63) / 64) * 64
            logger.info("To use awq we pad vocab_size to {}.".format(
                args.vocab_size))

    args.max_num_tokens = check_max_num_tokens(
        max_num_tokens=args.max_num_tokens,
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        remove_input_padding=args.remove_input_padding)

    assert (math.log2(args.tokens_per_block).is_integer()
            ), "tokens_per_block must be power of 2"
    if args.enable_context_fmha or args.enable_context_fmha_fp32_acc:
        assert (args.tokens_per_block >=
                128), "Context fMHA requires >= 128 tokens per block"

    return args


def build_rank_engine(builder: Builder,
                      builder_config: tensorrt_llm.builder.BuilderConfig,
                      engine_name, rank, multi_query_mode, args):
    '''
       @brief: Build the engine on the given rank.
       @param rank: The rank to build the engine.
       @param args: The cmd line arguments.
       @return: The built engine.
    '''
    kv_dtype = str_dtype_to_trt(args.dtype)
    mapping = Mapping(world_size=args.world_size,
                      rank=rank,
                      tp_size=args.tp_size,
                      pp_size=args.pp_size)

    # Initialize Module
    tensorrt_llm_qwen = tensorrt_llm.models.QWenForCausalLM(
        num_layers=args.n_layer,
        num_heads=args.n_head,
        num_kv_heads=args.n_kv_head,
        hidden_size=args.n_embd,
        seq_length=args.max_input_len,
        vocab_size=args.vocab_size,
        hidden_act=args.hidden_act,
        max_position_embeddings=args.n_positions,
        dtype=kv_dtype,
        mlp_hidden_size=args.inter_size,
        neox_rotary_style=True,
        mapping=mapping,
        rotary_base=args.rotary_base,
        rotary_scaling=args.rotary_scaling,
        use_parallel_embedding=args.use_parallel_embedding,
        embedding_sharding_dim=args.embedding_sharding_dim,
        quant_mode=args.quant_mode,
        use_prompt_tuning=args.max_prompt_embedding_table_size > 0,
    )
    quantize_kwargs = {}
    use_gemm_woq_plugin = args.use_gemm_plugin and args.use_weight_only
    if args.use_smooth_quant or args.use_weight_only:
        if args.weight_only_precision == 'int4_awq':
            exclude_modules = ['lm_head'] if not args.quantize_lm_head else []
            quantize_kwargs = {
                "group_size": args.group_size,
                "zero": False,
                "pre_quant_scale": True,
                "exclude_modules": exclude_modules,
            }
        elif args.weight_only_precision == 'int4_gptq':
            quantize_kwargs = {
                "group_size": args.group_size,
                "zero": True,
                "pre_quant_scale": False,
            }
    tensorrt_llm_qwen = quantize_model(tensorrt_llm_qwen, args.quant_mode,
                                       **quantize_kwargs)
    ft_dir_path = args.ft_dir_path
    if args.per_group:
        if args.weight_only_precision == 'int4_awq':
            load_from_awq_qwen(tensorrt_llm_qwen=tensorrt_llm_qwen,
                               quant_ckpt_path=args.quant_ckpt_path,
                               quantize_lm_head=args.quantize_lm_head,
                               mapping=mapping,
                               dtype=args.dtype)
        else:
            load_from_gptq_qwen(tensorrt_llm_qwen=tensorrt_llm_qwen,
                                quant_ckpt_path=args.quant_ckpt_path,
                                mapping=mapping,
                                dtype=args.dtype)
    elif args.hf_model_dir is not None and \
        (ft_dir_path is None or not os.path.exists(ft_dir_path)):
        logger.info(f'Loading HF QWen ... from {args.hf_model_dir}')
        tik = time.time()
        hf_qwen = AutoModelForCausalLM.from_pretrained(
            args.hf_model_dir,
            device_map={
                "transformer": "cpu",
                "lm_head": "cpu"
            },  # Load to CPU memory
            torch_dtype="auto",
            trust_remote_code=True,
        )
        tok = time.time()
        t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
        logger.info(f'HF QWen loaded. Total time: {t}')
        load_from_hf_qwen(tensorrt_llm_qwen,
                          hf_qwen,
                          mapping,
                          max_position_embeddings=args.n_positions,
                          kv_channels=args.kv_channels,
                          rotary_emb_base=args.rotary_emb_base,
                          dtype=args.dtype,
                          multi_query_mode=multi_query_mode,
                          use_gemm_woq_plugin=use_gemm_woq_plugin)
        del hf_qwen
    elif ft_dir_path is not None:
        dir_path = ft_dir_path
        logger.info(f'Loading FT QWen ... from {ft_dir_path}')
        load_from_ft(tensorrt_llm_qwen,
                     dir_path,
                     mapping,
                     dtype=args.dtype,
                     multi_query_mode=multi_query_mode,
                     use_gemm_woq_plugin=use_gemm_woq_plugin)
    else:
        raise ValueError(
            "You must specify either --hf_model_dir or --ft_dir_path")

    # Module -> Network
    network = builder.create_network()
    network.trt_network.name = engine_name
    if args.use_gpt_attention_plugin:
        network.plugin_config.set_gpt_attention_plugin(
            dtype=args.use_gpt_attention_plugin)
    if args.use_gemm_plugin:
        network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
    # Quantization plugins.
    if args.use_smooth_quant:
        network.plugin_config.set_smooth_quant_gemm_plugin(dtype=args.dtype)
        network.plugin_config.set_rmsnorm_quantization_plugin(dtype=args.dtype)
        network.plugin_config.set_quantize_tensor_plugin()
        network.plugin_config.set_quantize_per_token_plugin()
    assert not (args.enable_context_fmha and args.enable_context_fmha_fp32_acc)
    if args.enable_context_fmha:
        network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
    if args.enable_context_fmha_fp32_acc:
        network.plugin_config.set_context_fmha(
            ContextFMHAType.enabled_with_fp32_acc)
    if args.use_weight_only and args.use_gemm_plugin:
        if args.per_group:
            network.plugin_config.set_weight_only_groupwise_quant_matmul_plugin(
                dtype='float16')
        else:
            network.plugin_config.set_weight_only_quant_matmul_plugin(
                dtype='float16')
    if args.world_size > 1:
        network.plugin_config.set_nccl_plugin(args.dtype,
                                              args.use_custom_all_reduce)
    if args.remove_input_padding:
        network.plugin_config.enable_remove_input_padding()

    if args.paged_kv_cache:
        network.plugin_config.enable_paged_kv_cache(args.tokens_per_block)
    if args.use_rmsnorm_plugin:
        network.plugin_config.set_rmsnorm_plugin(dtype=args.use_rmsnorm_plugin)
    if args.use_lookup_plugin:
        network.plugin_config.set_lookup_plugin(dtype=args.dtype)

    with net_guard(network):
        # Prepare
        network.set_named_parameters(tensorrt_llm_qwen.named_parameters())

        # Forward
        inputs = tensorrt_llm_qwen.prepare_inputs(
            max_batch_size=args.max_batch_size,
            max_input_len=args.max_input_len,
            max_seq_len=args.max_input_len + args.max_output_len,
            use_cache=True,
            max_beam_width=args.max_beam_width,
            max_num_tokens=args.max_num_tokens,
            prompt_embedding_table_size=args.max_prompt_embedding_table_size,
        )
        tensorrt_llm_qwen(*inputs)
        if args.enable_debug_output:
            # mark intermediate nodes' outputs
            for k, v in tensorrt_llm_qwen.named_network_outputs():
                v = v.trt_tensor
                v.name = k
                network.trt_network.mark_output(v)
                v.dtype = kv_dtype
        if args.visualize:
            model_path = os.path.join(args.output_dir, 'test.onnx')
            to_onnx(network.trt_network, model_path)

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
    multi_query_mode = (args.n_kv_head
                        is not None) and (args.n_kv_head != args.n_head)

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
            max_beam_width=args.max_beam_width,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            max_num_tokens=args.max_num_tokens,
            int8=int8_trt_flag,
            fp8=args.quant_mode.has_fp8_qdq(),
            quant_mode=args.quant_mode,
            strongly_typed=args.strongly_typed,
            opt_level=args.builder_opt,
            max_prompt_embedding_table_size=args.
            max_prompt_embedding_table_size,
        )
        engine_name = get_engine_name(MODEL_NAME, args.dtype, args.tp_size,
                                      args.pp_size, cur_rank)
        engine = build_rank_engine(builder, builder_config, engine_name,
                                   cur_rank, multi_query_mode, args)
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
