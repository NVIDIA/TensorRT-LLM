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
import os
from collections import OrderedDict

import onnx
import tensorrt as trt
from onnx import TensorProto, helper

import tensorrt_llm
from tensorrt_llm.builder import Builder
from tensorrt_llm.functional import assertion, shape
from tensorrt_llm.network import net_guard


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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size',
                        type=int,
                        default=1,
                        help='world size, only support tensor parallelism now')
    parser.add_argument('--dtype', type=str, default='float32')
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--vocab_size', type=int, default=51200)
    parser.add_argument('--n_layer', type=int, default=24)
    parser.add_argument('--n_positions', type=int, default=1024)
    parser.add_argument('--n_embd', type=int, default=1024)
    parser.add_argument('--n_head', type=int, default=16)
    parser.add_argument('--hidden_act', type=str, default='gelu')
    parser.add_argument('--max_batch_size', type=int, default=256)
    parser.add_argument('--max_input_len', type=int, default=200)
    parser.add_argument('--max_output_len', type=int, default=200)
    parser.add_argument('--use_gpt_attention_plugin',
                        default=False,
                        action='store_true')
    parser.add_argument('--use_gemm_plugin', default=False, action='store_true')
    parser.add_argument('--use_layernorm_plugin',
                        default=False,
                        action='store_true')
    parser.add_argument('--output_dir', type=str, default='gpt_outputs')
    return parser.parse_args()


def prepare_inputs(args):
    # Prepare inputs
    head_size = args.n_embd // args.n_head
    max_len = args.max_input_len + args.max_output_len
    bs_range = [1, (args.max_batch_size + 1) // 2, args.max_batch_size]
    inlen_range = [1, (args.max_input_len + 1) // 2, args.max_input_len]
    max_len_range = [1, (max_len + 1) // 2, max_len]
    step_range = [1, 1, args.max_input_len + 1]

    input_ids = tensorrt_llm.Tensor(name='input_ids',
                                    dtype=trt.int32,
                                    shape=[-1, -1],
                                    dim_range=OrderedDict([
                                        ('batch_size', [bs_range, bs_range]),
                                        ('input_len', [inlen_range, 1]),
                                    ]))
    kv_dtype = trt.float16 if args.dtype == 'float16' else trt.float32
    past_key_value = []
    sequence_length = None
    shape_tensor = None
    if not args.use_gpt_attention_plugin:
        for i in range(args.n_layer):
            kv_dim_range = OrderedDict([
                ('batch_size', [bs_range, bs_range]),
                ('num_heads', [args.n_head, args.n_head]),
                ('past_key_len', [0, max_len_range]),
                ('head_size', [head_size, head_size]),
            ])
            k = tensorrt_llm.Tensor(name=f'past_key_{i}',
                                    dtype=kv_dtype,
                                    shape=[-1, args.n_head, -1, head_size],
                                    dim_range=kv_dim_range)
            v = tensorrt_llm.Tensor(name=f'past_value_{i}',
                                    dtype=kv_dtype,
                                    shape=[-1, args.n_head, -1, head_size],
                                    dim_range=kv_dim_range)
            past_key_value.append((k, v))
            # TODO(kaiyu): Remove this when TRT fix the named dimension
            assertion(shape(input_ids, 0) == shape(k, 0), 'batch size')
            assertion(shape(k, 2) == shape(v, 2), 'kv cache len')
    else:
        for i in range(args.n_layer):
            past_key_value.append(
                tensorrt_llm.Tensor(
                    name=f'past_{i}',
                    dtype=kv_dtype,
                    shape=[2, -1, args.n_head, -1, head_size],
                    dim_range=OrderedDict([
                        ('2', [2, 2]), ('batch_size', [bs_range, bs_range]),
                        ('num_heads', [args.n_head, args.n_head]),
                        ('past_key_len', [max_len_range, max_len_range]),
                        ('head_size', [head_size, head_size])
                    ]),
                ))
        sequence_length = tensorrt_llm.Tensor(
            name='sequence_length',
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([('batch_size', [bs_range, bs_range])]),
        )

        shape_tensor = tensorrt_llm.Tensor(
            name='shape_tensor',
            dtype=trt.int32,
            shape=[-1, -1],
            dim_range=OrderedDict([('step', [step_range, max_len_range]),
                                   ('cur_seq_len', [0, max_len_range])]))
    return (input_ids, None, past_key_value, sequence_length, shape_tensor,
            True)


if __name__ == '__main__':
    args = parse_arguments()
    tensorrt_llm.logger.set_level(args.log_level)
    tensorrt_llm.set_default_dtype(args.dtype)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    kv_dtype = trt.float16 if args.dtype == 'float16' else trt.float32

    builder = Builder()

    # Initialize Module
    apply_query_key_layer_scaling = False
    tensorrt_llm_gpt = tensorrt_llm.models.GPTLMHeadModel(
        num_layers=args.n_layer,
        num_heads=args.n_head,
        hidden_size=args.n_embd,
        vocab_size=args.vocab_size,
        hidden_act=args.hidden_act,
        max_position_embeddings=args.n_positions,
        dtype=kv_dtype,
        tensor_parallel=args.world_size,  # TP only
        tensor_parallel_group=list(range(args.world_size)),  # TP only
        apply_query_key_layer_scaling=apply_query_key_layer_scaling)

    # Module -> Network
    network = builder.create_network()
    if args.use_gpt_attention_plugin:
        network.plugin_config.set_gpt_attention_plugin()
    if args.use_gemm_plugin:
        network.plugin_config.set_gemm_plugin()
    if args.use_layernorm_plugin:
        network.plugin_config.set_layernorm_plugin()
    with net_guard(network):
        # Prepare
        network.set_named_parameters(tensorrt_llm_gpt.named_parameters())

        # Forward
        inputs = prepare_inputs(args)
        lm_logits, presents = tensorrt_llm_gpt(*inputs)

        # Mark outputs
        lm_logits.mark_output('logits', kv_dtype)
        if not args.use_gpt_attention_plugin:
            for i, present in enumerate(presents):
                k, v = present
                k.mark_output(f'present_key_{i}', kv_dtype)
                v.mark_output(f'present_value_{i}', kv_dtype)
        else:
            for i, present in enumerate(presents):
                present.mark_output(f'present_{i}', kv_dtype)

    model_path = os.path.join(args.output_dir, 'test.onnx')
    to_onnx(network.trt_network, model_path)
