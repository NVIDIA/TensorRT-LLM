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
from typing import Union

import onnx
import tensorrt as trt
import torch
import torch.multiprocessing as mp
from onnx import TensorProto, helper
from transformers import AutoModelForCausalLM, FalconConfig

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import fp8_quantize
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode

from weight import get_scaling_factors  # isort:skip
from weight import load_from_hf_falcon  # isort:skip
from weight import load_from_hf_checkpoint  # isort:skip

MODEL_NAME = 'falcon'


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
        f.write(bytearray(engine))
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')


def load_falcon_config(model_dir: Union[str, Path]) -> FalconConfig:
    """ Helper utility to laod FalconConfig.

    A pretrained checkpoint from modeling_RW.py has a different structure
    and is not compatible with `transformers.FalconConfig` and
    `transformers.FalconModel`. We need to manually set the config values.
    """

    config = FalconConfig.from_pretrained(model_dir)
    if config.model_type not in ['RefinedWebModel', 'RefinedWeb']:
        return config

    if config.model_type == 'RefinedWeb':
        # Case 1. Falcon-40B / Falcon-40B-instruct
        # https://huggingface.co/tiiuae/falcon-40b/blob/main/config.json
        config.num_hidden_layers = config.n_layer
        config.num_attention_heads = config.n_head
        config.num_kv_heads = config.n_head_kv
        config.new_decoder_architecture = True
    elif config.model_type == 'RefinedWebModel':
        # Case 2. Falcon-7B / Falcon-7B-instruct
        # https://huggingface.co/tiiuae/falcon-7b/blob/main/config.json
        config.num_hidden_layers = config.n_layer
        config.num_attention_heads = config.n_head
        config.num_kv_heads = 1 if config.multi_query else config.n_head
        config.new_decoder_architecture = False
    else:
        raise ValueError("Shouldn't reach here.")
    config.model_type = 'falcon'

    return config


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--pp_size', type=int, default=1)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument(
        '--timing_cache',
        type=str,
        default='model.cache',
        help='The path of to read timing cache from, will be ignored if the '
        'file does not exist')
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--vocab_size', type=int, default=65024)
    parser.add_argument('--n_layer', type=int, default=36)
    parser.add_argument('--n_positions', type=int, default=2048)
    parser.add_argument('--n_embd', type=int, default=4096)
    parser.add_argument('--n_head', type=int, default=64)
    parser.add_argument('--n_kv_head', type=int, default=None)
    parser.add_argument('--mlp_hidden_size', type=int, default=None)
    parser.add_argument('--max_batch_size', type=int, default=8)
    parser.add_argument('--max_input_len', type=int, default=1024)
    parser.add_argument('--max_output_len', type=int, default=1024)
    parser.add_argument('--max_beam_width', type=int, default=1)
    parser.add_argument('--use_gpt_attention_plugin',
                        nargs='?',
                        const='float16',
                        type=str,
                        default=False,
                        choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--parallel_attention',
                        action='store_true',
                        help='Use Falcon parallel attention.')
    parser.add_argument('--new_decoder_architecture',
                        action='store_true',
                        help='Use the new Falcon decoder architecture. '
                        'If enabled, --parallel_attention will be ignored.')
    parser.add_argument(
        '--alibi',
        action='store_true',
        help='Use ALiBi positional encoding. If disabled, RoPE will be used.')
    parser.add_argument('--logits_dtype',
                        type=str,
                        default='float32',
                        choices=['float16', 'float32'])
    parser.add_argument('--use_gemm_plugin',
                        nargs='?',
                        const='float16',
                        type=str,
                        default=False,
                        choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument(
        '--use_layernorm_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help="Activates layernorm plugin. You can specify the plugin dtype or "
        "leave blank to use the model dtype.")
    parser.add_argument('--parallel_build', default=False, action='store_true')
    parser.add_argument('--enable_context_fmha',
                        default=False,
                        action='store_true')
    parser.add_argument('--enable_context_fmha_fp32_acc',
                        default=False,
                        action='store_true')
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument('--load_by_shard',
                        action='store_true',
                        help='Load a pretrained model shard-by-shard.')
    parser.add_argument('--enable_debug_output',
                        default=False,
                        action='store_true')
    parser.add_argument('--gpus_per_node', type=int, default=8)
    parser.add_argument('--builder_opt', type=int, default=None)
    parser.add_argument(
        '--output_dir',
        type=str,
        default='falcon_outputs',
        help='The path to save the serialized engine files, timing cache '
        'file and model configs')
    parser.add_argument('--remove_input_padding',
                        default=False,
                        action='store_true')
    parser.add_argument(
        '--strongly_typed',
        default=False,
        action="store_true",
        help=
        'This option is introduced with trt 9.1.0.1+ and will reduce the building time significantly for fp8.'
    )
    # Arguments related to the quantization of the model.
    parser.add_argument(
        '--enable_fp8',
        default=False,
        action='store_true',
        help='Use FP8 Linear layer for Attention QKV/Dense and MLP.')
    parser.add_argument(
        '--quantized_fp8_model_path',
        type=str,
        default=None,
        help='Path of a quantized model checkpoint in .npz format')
    parser.add_argument(
        '--fp8_kv_cache',
        default=False,
        action="store_true",
        help='By default, we use dtype for KV cache. fp8_kv_cache chooses int8 '
        'quantization for KV')
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
                        default=64,
                        help='Number of tokens per block in paged KV cache')
    parser.add_argument(
        '--max_num_tokens',
        type=int,
        default=None,
        help='Define the max number of tokens supported by the engine')

    parser.add_argument(
        '--use_custom_all_reduce',
        action='store_true',
        help=
        'Activates latency-optimized algorithm for all-reduce instead of NCCL.')

    args = parser.parse_args()

    logger.set_level(args.log_level)

    if not args.remove_input_padding:
        if args.use_gpt_attention_plugin:
            logger.warning(
                f"It is recommended to specify --remove_input_padding when using GPT attention plugin"
            )

    if args.use_inflight_batching:
        if not args.use_gpt_attention_plugin:
            args.use_gpt_attention_plugin = 'float16'
            logger.info(
                f"Using GPT attention plugin for inflight batching mode. "
                f"Setting to default '{args.use_gpt_attention_plugin}'")
        if not args.remove_input_padding:
            args.remove_input_padding = True
            logger.info(
                'Using remove input padding for inflight batching mode.')
        if not args.paged_kv_cache:
            args.paged_kv_cache = True
            logger.info('Using paged KV cache for inflight batching mode.')

    if args.max_num_tokens is not None:
        assert args.enable_context_fmha

    args.quant_mode = QuantMode(0)
    if args.fp8_kv_cache:
        args.quant_mode = args.quant_mode.set_fp8_kv_cache()
    if args.enable_fp8:
        args.quant_mode = args.quant_mode.set_fp8_qdq()

    if args.model_dir is not None:
        hf_config = load_falcon_config(args.model_dir)
        args.n_embd = hf_config.hidden_size
        args.n_head = hf_config.num_attention_heads
        args.n_kv_head = hf_config.num_kv_heads
        args.n_layer = hf_config.num_hidden_layers
        args.vocab_size = hf_config.vocab_size
        args.alibi = hf_config.alibi
        args.bias = hf_config.bias

        # Falcon variants.
        args.parallel_attention = hf_config.parallel_attn
        args.new_decoder_architecture = hf_config.new_decoder_architecture

        # FalconConfig sets num_kv_heads by num_heads if not provided, even
        # though multi-query attention case. We here manually correct the
        # value of number of K/V heads.
        if not hf_config.new_decoder_architecture and hf_config.multi_query:
            args.n_kv_head = 1
    else:
        args.n_kv_head = args.n_kv_head or args.n_head
    assert (args.n_head % args.n_kv_head) == 0, \
        "MQA/GQA requires the number of heads to be divisible by the number "\
        "of K/V heads."
    assert args.n_kv_head % args.tp_size == 0 \
        or args.tp_size % args.n_kv_head == 0, \
        "MQA/GQA requires either the number of K/V heads to be divisible by "\
        "the tensor parallelism size OR the tensor parallelism size to be "\
        "divisible by the number of K/V heads."
    assert args.pp_size * args.tp_size == args.world_size

    if not args.use_gpt_attention_plugin and not args.alibi:
        args.use_gpt_attention_plugin = args.dtype
        logger.warning(
            f"RoPE does not support without GPT attention plugin. Set by "
            f"use_gpt_attention_plugin={args.dtype}.")

    logger.info(' Build Arguments '.center(100, '='))
    for k, v in vars(args).items():
        logger.info(f' - {k.ljust(30, ".")}: {v}')
    logger.info('=' * 100)

    return args


def build_rank_engine(builder: Builder,
                      builder_config: tensorrt_llm.builder.BuilderConfig,
                      engine_name: str, rank: int,
                      args: argparse.Namespace) -> trt.IHostMemory:
    '''
       @brief: Build the engine on the given rank.
       @param rank: The rank to build the engine.
       @param args: The cmd line arguments.
       @return: The built engine.
    '''
    dtype = str_dtype_to_trt(args.dtype)

    # Initialize Module
    mapping = Mapping(
        world_size=args.world_size,
        rank=rank,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
    )
    assert args.n_layer % args.pp_size == 0, \
        f"num_layers {args.n_layer} must be a multiple of pipeline "\
        f"parallelism size {args.pp_size}"

    tensorrt_llm_falcon = tensorrt_llm.models.FalconForCausalLM(
        num_layers=args.n_layer,
        num_heads=args.n_head,
        num_kv_heads=args.n_kv_head,
        hidden_size=args.n_embd,
        vocab_size=args.vocab_size,
        max_position_embeddings=args.n_positions,
        dtype=dtype,
        quant_mode=args.quant_mode,
        bias=args.bias,
        use_alibi=args.alibi,
        logits_dtype=args.logits_dtype,
        mapping=mapping,
        parallel_attention=args.parallel_attention,
        new_decoder_architecture=args.new_decoder_architecture)

    if args.enable_fp8 or args.fp8_kv_cache:
        logger.info(f'Loading scaling factors from '
                    f'{args.quantized_fp8_model_path}')
        quant_scales = get_scaling_factors(args.quantized_fp8_model_path,
                                           num_layers=args.n_layer,
                                           quant_mode=args.quant_mode)
        tensorrt_llm_falcon = fp8_quantize(tensorrt_llm_falcon,
                                           quant_mode=args.quant_mode,
                                           quant_scales=quant_scales)
    if args.model_dir is not None:
        logger.info(f'Loading HF Falcon ... from {args.model_dir}')
        tik = time.time()
        if not args.load_by_shard:
            hf_falcon = AutoModelForCausalLM.from_pretrained(
                args.model_dir, trust_remote_code=True)
            load_from_hf_falcon(tensorrt_llm_falcon,
                                hf_falcon,
                                mapping,
                                dtype=args.dtype)
            del hf_falcon
        else:
            load_from_hf_checkpoint(tensorrt_llm_falcon,
                                    args.model_dir,
                                    mapping,
                                    dtype=args.dtype)
        tok = time.time()
        t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
        logger.info(f'HF Falcon loaded. Total time: {t}')

    # Module -> Network
    network = builder.create_network()
    network.trt_network.name = engine_name
    if args.use_gpt_attention_plugin:
        network.plugin_config.set_gpt_attention_plugin(
            dtype=args.use_gpt_attention_plugin)
    if args.use_gemm_plugin:
        network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
    if args.use_layernorm_plugin:
        network.plugin_config.set_layernorm_plugin(
            dtype=args.use_layernorm_plugin)

    # Quantization plugins.
    assert not (args.enable_context_fmha and args.enable_context_fmha_fp32_acc)
    if args.enable_context_fmha:
        network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
    if args.enable_context_fmha_fp32_acc:
        network.plugin_config.set_context_fmha(
            ContextFMHAType.enabled_with_fp32_acc)

    if args.world_size > 1:
        network.plugin_config.set_nccl_plugin(args.dtype,
                                              args.use_custom_all_reduce)
    if args.remove_input_padding:
        network.plugin_config.enable_remove_input_padding()
    if args.paged_kv_cache:
        network.plugin_config.enable_paged_kv_cache(args.tokens_per_block)
    with net_guard(network):
        # Prepare
        network.set_named_parameters(tensorrt_llm_falcon.named_parameters())
        inputs = tensorrt_llm_falcon.prepare_inputs(
            max_batch_size=args.max_batch_size,
            max_input_len=args.max_input_len,
            max_new_tokens=args.max_output_len,
            use_cache=True,
            max_beam_width=args.max_beam_width,
            max_num_tokens=args.max_num_tokens)
        tensorrt_llm_falcon(*inputs)
        if args.enable_debug_output:
            # mark intermediate nodes' outputs
            for k, v in tensorrt_llm_falcon.named_network_outputs():
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
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # when doing serializing build, all ranks share one engine
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
            tensor_parallel=args.tp_size,
            pipeline_parallel=args.pp_size,
            num_layers=args.n_layer,
            num_heads=args.n_head,
            num_kv_heads=args.n_kv_head,
            hidden_size=args.n_embd,
            vocab_size=args.vocab_size,
            alibi=args.alibi,
            parallel_build=args.parallel_build,
            new_decoder_architecture=args.new_decoder_architecture,
            max_position_embeddings=args.n_positions,
            max_batch_size=args.max_batch_size,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            max_num_tokens=args.max_num_tokens,
            fp8=args.quant_mode.has_fp8_qdq(),
            quant_mode=args.quant_mode,
            strongly_typed=args.strongly_typed,
            opt_level=args.builder_opt)
        engine_name = get_engine_name(MODEL_NAME, args.dtype, args.tp_size,
                                      args.pp_size, cur_rank)
        engine = build_rank_engine(builder, builder_config, engine_name,
                                   cur_rank, args)
        assert engine is not None, \
            f'Failed to build engine for rank {cur_rank}'

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
            f'Parallelly build TensorRT engines. Please make sure that all '
            f'of the {args.world_size} GPUs are totally free.')
        mp.spawn(build, nprocs=args.world_size, args=(args, ))
    else:
        args.parallel_build = False
        logger.info('Serially build TensorRT engines.')
        build(0, args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Total time of building all {args.world_size} engines: {t}')
