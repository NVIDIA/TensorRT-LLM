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
import json
import math
import os
import time
from pathlib import Path

# isort: off
import torch
import torch.multiprocessing as mp
import tensorrt as trt
# isort: on
from transformers import AutoConfig, AutoModelForCausalLM
from weight import (get_scaling_factors, load_from_awq_internlm,
                    load_from_binary, load_from_gptq_internlm,
                    load_from_hf_internlm, load_from_meta_internlm)

import tensorrt_llm
from tensorrt_llm._common import check_max_num_tokens
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.layers.attention import PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import quantize_model
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode

from weight import parse_ft_config  # isort:skip

MODEL_NAME = "internlm"

# 2 routines: get_engine_name, serialize_engine
# are direct copy from gpt example, TODO: put in utils?

import onnx
from onnx import TensorProto, helper


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
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--pp_size', type=int, default=1)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--ft_model_dir', type=str, default=None)
    parser.add_argument('--meta_ckpt_dir', type=str, default=None)
    parser.add_argument('--quant_ckpt_path', type=str, default=None)
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
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--vocab_size', type=int, default=103168)
    parser.add_argument('--n_layer', type=int, default=32)
    parser.add_argument('--n_positions', type=int, default=2048)
    parser.add_argument('--n_embd', type=int, default=4096)
    parser.add_argument('--n_head', type=int, default=32)
    parser.add_argument('--n_kv_head', type=int, default=None)
    parser.add_argument('--multiple_of', type=int, default=256)
    parser.add_argument('--ffn_dim_multiplier', type=float, default=1.0)
    parser.add_argument('--inter_size', type=int, default=None)
    parser.add_argument('--hidden_act', type=str, default='silu')
    parser.add_argument('--rms_norm_eps', type=float, default=1e-06)
    parser.add_argument('--max_batch_size', type=int, default=8)
    parser.add_argument('--max_input_len', type=int, default=2048)
    parser.add_argument('--max_output_len', type=int, default=512)
    parser.add_argument('--max_beam_width', type=int, default=1)
    parser.add_argument('--rotary_base', type=float, default=10000.0)
    parser.add_argument('--rotary_scaling', nargs=2, type=str, default=None)
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
    parser.add_argument('--use_rmsnorm_plugin',
                        nargs='?',
                        const='float16',
                        type=str,
                        default=False,
                        choices=['float16', 'float32', 'bfloat16'])
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
        '--enable_fp8',
        default=False,
        action='store_true',
        help='Use FP8 Linear layer for Attention QKV/Dense and MLP.')
    parser.add_argument(
        '--fp8_kv_cache',
        default=False,
        action="store_true",
        help=
        'By default, we use dtype for KV cache. fp8_kv_cache chooses int8 quantization for KV'
    )
    parser.add_argument(
        '--quantized_fp8_model_path',
        type=str,
        default=None,
        help='Path of a quantized model checkpoint in .npz format')
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

    args = parser.parse_args()
    tensorrt_llm.logger.set_level(args.log_level)

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
    elif args.fp8_kv_cache:
        args.quant_mode = args.quant_mode.set_fp8_kv_cache()
    if args.enable_fp8:
        args.quant_mode = args.quant_mode.set_fp8_qdq()

    if args.rotary_scaling is not None:
        rotary_scaling = {
            "type": args.rotary_scaling[0],
            "factor": float(args.rotary_scaling[1])
        }
        assert rotary_scaling["type"] in ["linear", "dynamic"]
        assert rotary_scaling["factor"] > 1.0
        args.rotary_scaling = rotary_scaling
        if rotary_scaling["type"] == "dynamic":
            assert not args.remove_input_padding, "TODO: Not supported yet"

    # Since gpt_attenttion_plugin is the only way to apply RoPE now,
    # force use the plugin for now with the correct data type.
    if args.model_dir is not None:
        hf_config = AutoConfig.from_pretrained(args.model_dir,
                                               trust_remote_code=True)
        args.inter_size = hf_config.intermediate_size  # override the inter_size for InternLM
        args.n_embd = hf_config.hidden_size
        args.n_head = hf_config.num_attention_heads
        if hasattr(hf_config, "num_key_value_heads"):
            args.n_kv_head = hf_config.num_key_value_heads
        args.n_layer = hf_config.num_hidden_layers
        args.n_positions = hf_config.max_position_embeddings
        args.vocab_size = hf_config.vocab_size
        args.hidden_act = hf_config.hidden_act
        args.rms_norm_eps = hf_config.rms_norm_eps
        args.attn_bias = hf_config.bias
    elif args.meta_ckpt_dir is not None:
        # Not tested
        with open(Path(args.meta_ckpt_dir, "params.json")) as fp:
            meta_config: dict = json.load(fp)
        args.n_embd = meta_config["dim"]
        args.n_head = meta_config["n_heads"]
        args.n_layer = meta_config["n_layers"]
        args.n_kv_head = meta_config.get("n_kv_heads", args.n_head)
        args.multiple_of = meta_config["multiple_of"]
        args.ffn_dim_multiplier = meta_config.get("ffn_dim_multiplier", 1)
        n_embd = int(4 * args.n_embd * 2 / 3)
        args.inter_size = args.multiple_of * (
            (int(n_embd * args.ffn_dim_multiplier) + args.multiple_of - 1) //
            args.multiple_of)
        args.rms_norm_eps = meta_config["norm_eps"]
    elif args.ft_model_dir is not None:
        n_embd, n_head, n_layer, n_positions, vocab_size, hidden_act, inter_size, n_kv_head, attn_bias = parse_ft_config(
            Path(args.ft_model_dir) / "config.ini")
        args.inter_size = inter_size  # override the inter_size for InternLM
        args.n_kv_head = n_kv_head
        args.n_embd = n_embd
        args.n_head = n_head
        args.n_layer = n_layer
        args.n_positions = n_positions
        args.vocab_size = vocab_size
        args.hidden_act = hidden_act
        args.rms_norm_eps = 1e-06
        logger.warning("Set rms_norm_eps to 1e-06 directly.")
        args.attn_bias = attn_bias

    if args.n_kv_head is None:
        args.n_kv_head = args.n_head
    elif args.n_kv_head != args.n_head:
        assert (args.n_head % args.n_kv_head) == 0, \
            "MQA/GQA requires the number of heads to be divisible by the number of K/V heads."
        assert (args.n_kv_head % args.tp_size) == 0 or (args.tp_size % args.n_kv_head) == 0, \
            "MQA/GQA requires either the number of K/V heads to be divisible by the tensor parallelism size OR " \
            "the tensor parallelism size to be divisible by the number of K/V heads."

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

    if args.inter_size is None:
        # this should not be need when loading a real model
        # but it is helpful when creating a dummy model without loading any real weights
        n_embd = int(4 * args.n_embd * 2 / 3)
        args.inter_size = args.multiple_of * (
            (int(n_embd * args.ffn_dim_multiplier) + args.multiple_of - 1) //
            args.multiple_of)
        logger.info(f"Setting inter_size to {args.inter_size}.")

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
    dtype = str_dtype_to_trt(args.dtype)
    mapping = Mapping(world_size=args.world_size,
                      rank=rank,
                      tp_size=args.tp_size,
                      pp_size=args.pp_size)

    assert args.n_layer % args.pp_size == 0, \
        f"num_layers {args.n_layer} must be a multiple of pipeline parallelism size {args.pp_size}"
    # Initialize Module
    tensorrt_llm_internlm = tensorrt_llm.models.LLaMAForCausalLM(
        num_layers=args.n_layer,
        num_heads=args.n_head,
        num_kv_heads=args.n_kv_head,
        hidden_size=args.n_embd,
        vocab_size=args.vocab_size,
        hidden_act=args.hidden_act,
        max_position_embeddings=args.n_positions,
        dtype=dtype,
        mlp_hidden_size=args.inter_size,
        position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
        mapping=mapping,
        rotary_base=args.rotary_base,
        rotary_scaling=args.rotary_scaling,
        use_parallel_embedding=args.use_parallel_embedding,
        embedding_sharding_dim=args.embedding_sharding_dim,
        use_fused_mlp=False,
        attn_bias=args.attn_bias,
        quant_mode=args.quant_mode,
        rms_norm_eps=args.rms_norm_eps)
    if args.use_smooth_quant:
        tensorrt_llm_internlm = quantize_model(tensorrt_llm_internlm,
                                               args.quant_mode)
    elif args.use_weight_only:
        if args.weight_only_precision == 'int8':
            tensorrt_llm_internlm = quantize_model(tensorrt_llm_internlm,
                                                   args.quant_mode)
        elif args.weight_only_precision == 'int4':
            tensorrt_llm_internlm = quantize_model(tensorrt_llm_internlm,
                                                   args.quant_mode)
        elif args.weight_only_precision == 'int4_awq':
            exclude_modules = ['lm_head'] if not args.quantize_lm_head else []
            tensorrt_llm_internlm = quantize_model(
                model=tensorrt_llm_internlm,
                quant_mode=args.quant_mode,
                group_size=args.group_size,
                zero=False,
                pre_quant_scale=True,
                exclude_modules=exclude_modules)
        elif args.weight_only_precision == 'int4_gptq':
            tensorrt_llm_internlm = quantize_model(model=tensorrt_llm_internlm,
                                                   quant_mode=args.quant_mode,
                                                   group_size=args.group_size,
                                                   zero=True,
                                                   pre_quant_scale=False)
    elif args.enable_fp8 or args.fp8_kv_cache:
        logger.info(f'Loading scaling factors from '
                    f'{args.quantized_fp8_model_path}')
        quant_scales = get_scaling_factors(args.quantized_fp8_model_path,
                                           num_layers=args.n_layer,
                                           quant_mode=args.quant_mode)
        tensorrt_llm_internlm = quantize_model(tensorrt_llm_internlm,
                                               quant_mode=args.quant_mode,
                                               quant_scales=quant_scales)

    use_gemm_woq_plugin = args.use_gemm_plugin and args.use_weight_only
    if args.per_group:
        if args.weight_only_precision == 'int4_awq':
            load_from_awq_internlm(tensorrt_llm_internlm=tensorrt_llm_internlm,
                                   quant_ckpt_path=args.quant_ckpt_path,
                                   quantize_lm_head=args.quantize_lm_head,
                                   mapping=mapping,
                                   dtype=args.dtype)
        else:
            load_from_gptq_internlm(tensorrt_llm_internlm=tensorrt_llm_internlm,
                                    quant_ckpt_path=args.quant_ckpt_path,
                                    mapping=mapping,
                                    dtype=args.dtype)
    elif args.meta_ckpt_dir is not None:
        load_from_meta_internlm(tensorrt_llm_internlm, args.meta_ckpt_dir,
                                mapping, args.dtype)
    elif args.model_dir is not None:
        logger.info(f'Loading HF InternLM ... from {args.model_dir}')
        tik = time.time()
        hf_internlm = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            device_map={
                "model": "cpu",
                "lm_head": "cpu"
            },  # Load to CPU memory
            torch_dtype="auto",
            trust_remote_code=True)
        tok = time.time()
        t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
        logger.info(f'HF InternLM loaded. Total time: {t}')
        load_from_hf_internlm(tensorrt_llm_internlm,
                              hf_internlm,
                              mapping=mapping,
                              dtype=args.dtype,
                              use_gemm_woq_plugin=use_gemm_woq_plugin)
        del hf_internlm
    elif args.ft_model_dir is not None:
        load_from_binary(tensorrt_llm_internlm,
                         args.ft_model_dir,
                         mapping,
                         fp16=(args.dtype == 'float16'),
                         multi_query_mode=(args.n_kv_head != args.n_head),
                         dtype=args.dtype,
                         use_gemm_woq_plugin=use_gemm_woq_plugin)

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
    if args.multi_block_mode:
        network.plugin_config.enable_mmha_multi_block_mode()
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

    with net_guard(network):
        # Prepare
        network.set_named_parameters(tensorrt_llm_internlm.named_parameters())

        # Forward
        inputs = tensorrt_llm_internlm.prepare_inputs(
            max_batch_size=args.max_batch_size,
            max_input_len=args.max_input_len,
            max_seq_len=args.max_input_len + args.max_output_len,
            use_cache=True,
            max_beam_width=args.max_beam_width,
            max_num_tokens=args.max_num_tokens)
        tensorrt_llm_internlm(*inputs)
        if args.enable_debug_output:
            # mark intermediate nodes' outputs
            for k, v in tensorrt_llm_internlm.named_network_outputs():
                v = v.trt_tensor
                v.name = k
                network.trt_network.mark_output(v)
                v.dtype = dtype
        if args.visualize:
            model_path = os.path.join(args.output_dir, 'test.onnx')
            to_onnx(network.trt_network, model_path)

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
    logger.set_level(args.log_level)
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
            opt_level=args.builder_opt)
        engine_name = get_engine_name(MODEL_NAME, args.dtype, args.tp_size,
                                      args.pp_size, cur_rank)
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
