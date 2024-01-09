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
import time
from operator import attrgetter

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm._utils import pad_vocab_size
from tensorrt_llm.models import GPTNeoXForCausalLM

UINT4_TO_INT4_FLAG = 1
GPTQ_FLAG = 1
GROUP_SIZE = 128


def numpy_split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    else:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx])


def torch_split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    else:
        return (torch.split(v, v.shape[dim] // tp_size,
                            dim=dim)[idx]).contiguous()


def unpack_int32_into_int8(w_packed):
    # Unpack inputs packed in int32/float32 into uint4 and store them in int8 format
    w_packed_int4x2 = w_packed.contiguous().view(torch.uint8)
    w_unpacked = torch.zeros(w_packed_int4x2.shape[0],
                             w_packed_int4x2.shape[1] * 2,
                             dtype=torch.int8)
    w_unpacked[:, ::2] = w_packed_int4x2 % 16
    w_unpacked[:, 1::2] = w_packed_int4x2 // 16
    return w_unpacked.contiguous()


def preprocess_groupwise_weight_params(qweight_unpacked_int8, scales_fp16,
                                       qzeros_unpacked_int8):
    packer = torch.ops.fastertransformer.pack_int8_tensor_to_packed_int4
    preprocessor = torch.ops.fastertransformer.preprocess_weights_for_mixed_gemm

    qweight_interleaved = preprocessor(packer(qweight_unpacked_int8),
                                       torch.quint4x2).view(torch.float16)

    # zeros = zeros * scales
    zeros_x_scales_fp16 = (-qzeros_unpacked_int8 + 8 * UINT4_TO_INT4_FLAG -
                           GPTQ_FLAG) * scales_fp16
    zeros_x_scales_fp16 = zeros_x_scales_fp16.half()

    # return processed interleaved weight, original scales and zeros * scales
    return qweight_interleaved.contiguous().numpy(), scales_fp16.contiguous(
    ).numpy(), zeros_x_scales_fp16.contiguous().numpy()


def load_from_hf_gpt_neox(tensorrt_llm_gpt_neox: GPTNeoXForCausalLM,
                          hf_gpt_neox,
                          fp16=False,
                          rank=0,
                          tp_size=1,
                          use_weight_only_groupwise_quant_matmul_plugin=False):

    hf_model_gptneox_block_names = [
        "input_layernorm.weight",
        "input_layernorm.bias",
        "post_attention_layernorm.weight",
        "post_attention_layernorm.bias",
    ]

    tensorrt_llm_model_gptneox_block_names = [
        "input_layernorm.weight",
        "input_layernorm.bias",
        "post_attention_layernorm.weight",
        "post_attention_layernorm.bias",
    ]

    if not use_weight_only_groupwise_quant_matmul_plugin:
        hf_model_gptneox_block_names += [
            "attention.dense.weight",
            "attention.dense.bias",
            "mlp.dense_h_to_4h.weight",
            "mlp.dense_h_to_4h.bias",
            "mlp.dense_4h_to_h.weight",
            "mlp.dense_4h_to_h.bias",
        ]
        tensorrt_llm_model_gptneox_block_names += [
            "attention.dense.weight",
            "attention.dense.bias",
            "mlp.fc.weight",
            "mlp.fc.bias",
            "mlp.proj.weight",
            "mlp.proj.bias",
        ]

    if not use_weight_only_groupwise_quant_matmul_plugin:
        tensorrt_llm.logger.info('Loading weights from HF GPT-NeoX...')
    else:
        tensorrt_llm.logger.info(
            'Loading weights from GPTQ quantized HF GPT-NeoX...')

    tik = time.time()

    torch_dtype = torch.float16 if fp16 else torch.float32
    hf_gpt_neox_state_dict = hf_gpt_neox.state_dict()

    # [vocab_size, hidden_size]
    v = hf_gpt_neox_state_dict.get('gpt_neox.embed_in.weight').to(
        torch_dtype).cpu().numpy()
    if tensorrt_llm_gpt_neox._use_parallel_embedding:
        v = numpy_split(v, tp_size, rank,
                        tensorrt_llm_gpt_neox._embedding_sharding_dim)
    tensorrt_llm_gpt_neox.vocab_embedding.weight.value = v

    n_layer = hf_gpt_neox.config.num_hidden_layers

    for layer_idx in range(n_layer):
        prefix = "gpt_neox.layers." + str(layer_idx) + "."
        for idx, hf_attr in enumerate(hf_model_gptneox_block_names):
            v = hf_gpt_neox_state_dict.get(prefix + hf_attr).to(
                torch_dtype).cpu().numpy()

            layer = attrgetter(tensorrt_llm_model_gptneox_block_names[idx])(
                tensorrt_llm_gpt_neox.layers[layer_idx])

            if tp_size > 1:
                if 'dense.weight' in hf_attr:
                    # [n=hidden_size, k=hidden_size] ->
                    # [n=hidden_size, k=hidden_size // tp_size]
                    split_v = numpy_split(v, tp_size, rank, dim=1)
                elif 'dense_h_to_4h.weight' in hf_attr:
                    # [hidden_size * 4, hidden_size] ->
                    # [hidden_size * 4 // tp_size, hidden_size]
                    split_v = numpy_split(v, tp_size, rank, dim=0)
                elif 'dense_h_to_4h.bias' in hf_attr:
                    # [hidden_size * 4] -> [hidden_size * 4 // tp_size]
                    split_v = numpy_split(v, tp_size, rank, dim=0)
                elif 'dense_4h_to_h.weight' in hf_attr:
                    # [hidden_size, hidden_size * 4] ->
                    # [hidden_size, hidden_size * 4 // tp_size]
                    split_v = numpy_split(v, tp_size, rank, dim=1)
                else:
                    split_v = v
                setattr(layer, 'value', split_v)
            else:
                setattr(layer, 'value', v)

        num_heads = hf_gpt_neox.config.num_attention_heads
        hidden_size = hf_gpt_neox.config.hidden_size
        head_size = hidden_size // num_heads

        if not use_weight_only_groupwise_quant_matmul_plugin:
            # Attention QKV Linear
            # qkv_weights [num_heads x (q|k|v), hidden_size] ->
            # [(num_heads x q)|(num_heads x k)|(num_heads x v), hidden_size]
            qkv_weights = hf_gpt_neox_state_dict.get(
                prefix + "attention.query_key_value.weight")
            qkv_bias = hf_gpt_neox_state_dict.get(
                prefix + "attention.query_key_value.bias")

            new_qkv_weight_shape = torch.Size(
                [num_heads, 3, head_size * qkv_weights.size()[-1]])
            new_qkv_bias_shape = torch.Size([num_heads, 3, head_size])

            qkv_weights = qkv_weights.view(new_qkv_weight_shape).permute(
                1, 0, 2).reshape([hidden_size * 3, hidden_size])
            qkv_bias = qkv_bias.view(new_qkv_bias_shape).permute(
                1, 0, 2).reshape([hidden_size * 3])

            if tp_size > 1:
                qkv_weights = qkv_weights.reshape(
                    3, hidden_size, hidden_size).to(torch_dtype).cpu().numpy()
                split_qkv_weights = numpy_split(
                    qkv_weights, tp_size, rank,
                    dim=1).reshape(3 * (hidden_size // tp_size), hidden_size)
                tensorrt_llm_gpt_neox.layers[layer_idx].attention.qkv.weight.value = \
                    np.ascontiguousarray(split_qkv_weights)

                qkv_bias = qkv_bias.reshape(
                    3, hidden_size).to(torch_dtype).cpu().numpy()
                split_qkv_bias = numpy_split(qkv_bias, tp_size, rank,
                                             dim=1).reshape(
                                                 3 * (hidden_size // tp_size))
                tensorrt_llm_gpt_neox.layers[layer_idx].attention.qkv.bias.value = \
                    np.ascontiguousarray(split_qkv_bias)
            else:
                tensorrt_llm_gpt_neox.layers[layer_idx].attention.qkv.weight.value = \
                    qkv_weights.to(torch_dtype).cpu().numpy()
                tensorrt_llm_gpt_neox.layers[layer_idx].attention.qkv.bias.value = \
                    qkv_bias.to(torch_dtype).cpu().numpy()
        else:
            # use_weight_only_groupwise_quant_matmul_plugin

            qweight_int32 = hf_gpt_neox_state_dict.get(
                prefix + "attention.query_key_value.qweight")
            scales_fp16 = hf_gpt_neox_state_dict.get(
                prefix + "attention.query_key_value.scales")
            qzeros_int32 = hf_gpt_neox_state_dict.get(
                prefix + "attention.query_key_value.qzeros")
            biases_fp16 = hf_gpt_neox_state_dict.get(
                prefix + "attention.query_key_value.bias")

            # [hidden_size // 8, hidden_size * 3] -> [hidden_size * 3, hidden_size]
            qweight_unpacked_int8 = unpack_int32_into_int8(
                qweight_int32.T).contiguous() - 8
            # [hidden_size // GROUP_SIZE, hidden_size * 3 // 8] ->
            # [hidden_size // GROUP_SIZE, hidden_size * 3]
            qzeros_unpacked_int8 = unpack_int32_into_int8(qzeros_int32)

            # qkv_weights [num_heads x (q|k|v), hidden_size] ->
            # [(num_heads x q)|(num_heads x k)|(num_heads x v), hidden_size]
            new_qkv_weight_shape = torch.Size(
                [num_heads, 3, head_size * qweight_unpacked_int8.size()[-1]])
            # [hidden_size * 3, hidden_size]
            qweight_unpacked_int8 = qweight_unpacked_int8.view(
                new_qkv_weight_shape).permute(1, 0, 2).reshape(
                    [hidden_size * 3, hidden_size]).contiguous()

            new_qkv_scale_shape = torch.Size(
                [num_heads, 3, head_size * (hidden_size // GROUP_SIZE)])
            # [hidden_size * 3, hidden_size // GROUP_SIZE]
            scales_fp16 = scales_fp16.T.contiguous().view(
                new_qkv_scale_shape).permute(1, 0, 2).reshape(
                    [hidden_size * 3, hidden_size // GROUP_SIZE]).contiguous()

            new_qkv_zero_shape = torch.Size(
                [num_heads, 3, head_size * (hidden_size // GROUP_SIZE)])
            # [hidden_size * 3, hidden_size // GROUP_SIZE]
            qzeros_unpacked_int8 = qzeros_unpacked_int8.T.contiguous().view(
                new_qkv_zero_shape).permute(1, 0, 2).reshape(
                    [hidden_size * 3, hidden_size // GROUP_SIZE]).contiguous()

            new_qkv_bias_shape = torch.Size([num_heads, 3, head_size])
            biases_fp16 = biases_fp16.view(new_qkv_bias_shape).permute(
                1, 0, 2).reshape([hidden_size * 3]).numpy()

            if tp_size > 1:
                qweight_unpacked_int8 = qweight_unpacked_int8.reshape(
                    [3, hidden_size, hidden_size])
                qweight_unpacked_int8 = torch_split(qweight_unpacked_int8,
                                                    tp_size,
                                                    rank,
                                                    dim=1)
                qweight_unpacked_int8 = qweight_unpacked_int8.reshape(
                    [3 * hidden_size // tp_size, hidden_size])

                scales_fp16 = scales_fp16.reshape(
                    [3, hidden_size, hidden_size // GROUP_SIZE])
                scales_fp16 = torch_split(scales_fp16, tp_size, rank, dim=1)
                scales_fp16 = scales_fp16.reshape(
                    [3 * hidden_size // tp_size, hidden_size // GROUP_SIZE])

                qzeros_unpacked_int8 = qzeros_unpacked_int8.reshape(
                    [3, hidden_size, hidden_size // GROUP_SIZE])
                qzeros_unpacked_int8 = torch_split(qzeros_unpacked_int8,
                                                   tp_size,
                                                   rank,
                                                   dim=1)
                qzeros_unpacked_int8 = qzeros_unpacked_int8.reshape(
                    [3 * hidden_size // tp_size, hidden_size // GROUP_SIZE])

                biases_fp16 = biases_fp16.reshape([3, hidden_size])
                biases_fp16 = numpy_split(biases_fp16, tp_size, rank, dim=1)
                biases_fp16 = biases_fp16.reshape([3 * hidden_size // tp_size])

            qweight_fp32, scales_fp16, zeros_fp16 = preprocess_groupwise_weight_params(
                qweight_unpacked_int8.T.contiguous(),
                scales_fp16.T.contiguous(), qzeros_unpacked_int8.T.contiguous())

            tensorrt_llm_gpt_neox.layers[layer_idx].attention.qkv.qweight.value = \
                qweight_fp32
            tensorrt_llm_gpt_neox.layers[layer_idx].attention.qkv.scale.value = \
                scales_fp16
            tensorrt_llm_gpt_neox.layers[layer_idx].attention.qkv.zero.value = \
                zeros_fp16
            tensorrt_llm_gpt_neox.layers[layer_idx].attention.qkv.bias.value = \
                biases_fp16

            qweight_int32 = hf_gpt_neox_state_dict.get(
                prefix + "attention.dense.qweight")
            scales_fp16 = hf_gpt_neox_state_dict.get(prefix +
                                                     "attention.dense.scales")
            qzeros_int32 = hf_gpt_neox_state_dict.get(prefix +
                                                      "attention.dense.qzeros")
            biases_fp16 = hf_gpt_neox_state_dict.get(
                prefix + "attention.dense.bias").numpy()

            # [k=hidden_size // 8, n=hidden_size] -> [n=hidden_size, k=hidden_size]
            qweight_unpacked_int8 = unpack_int32_into_int8(
                qweight_int32.T).contiguous() - 8
            # [n=hidden_size, k=hidden_size] -> [k=hidden_size, n=hidden_size]
            qweight_unpacked_int8 = qweight_unpacked_int8.T.contiguous()
            # [k=hidden_size // GROUP_SIZE, n=hidden_size // 8] ->
            # [k=hidden_size // GROUP_SIZE, n=hidden_size]
            qzeros_unpacked_int8 = unpack_int32_into_int8(qzeros_int32)

            if tp_size > 1:
                qweight_unpacked_int8 = torch_split(qweight_unpacked_int8,
                                                    tp_size,
                                                    rank,
                                                    dim=0)
                scales_fp16 = torch_split(scales_fp16, tp_size, rank, dim=0)
                qzeros_unpacked_int8 = torch_split(qzeros_unpacked_int8,
                                                   tp_size,
                                                   rank,
                                                   dim=0)
                if rank > 0:
                    biases_fp16 = np.zeros_like(biases_fp16)

            qweight_fp32, scales_fp16, zeros_fp16 = preprocess_groupwise_weight_params(
                qweight_unpacked_int8, scales_fp16, qzeros_unpacked_int8)

            tensorrt_llm_gpt_neox.layers[layer_idx].attention.dense.qweight.value = \
                qweight_fp32
            tensorrt_llm_gpt_neox.layers[layer_idx].attention.dense.scale.value = \
                scales_fp16
            tensorrt_llm_gpt_neox.layers[layer_idx].attention.dense.zero.value = \
                zeros_fp16
            tensorrt_llm_gpt_neox.layers[layer_idx].attention.dense.bias.value = \
                biases_fp16

            qweight_int32 = hf_gpt_neox_state_dict.get(
                prefix + "mlp.dense_h_to_4h.qweight")
            scales_fp16 = hf_gpt_neox_state_dict.get(prefix +
                                                     "mlp.dense_h_to_4h.scales")
            qzeros_int32 = hf_gpt_neox_state_dict.get(
                prefix + "mlp.dense_h_to_4h.qzeros")
            biases_fp16 = hf_gpt_neox_state_dict.get(
                prefix + "mlp.dense_h_to_4h.bias").numpy()

            # [hidden_size // 8, hidden_size * 4] -> [hidden_size, hidden_size * 4]
            qweight_unpacked_int8 = unpack_int32_into_int8(
                qweight_int32.T).contiguous() - 8
            qweight_unpacked_int8 = qweight_unpacked_int8.T.contiguous()

            # [hidden_size // GROUP_SIZE, hidden_size * 4 // 8] ->
            # [hidden_size // GROUP_SIZE, hidden_size * 4]
            qzeros_unpacked_int8 = unpack_int32_into_int8(qzeros_int32)

            if tp_size > 1:
                # [hidden_size, hidden_size * 4] ->
                # [hidden_size, hidden_size * 4 // tp_size]
                qweight_unpacked_int8 = torch_split(qweight_unpacked_int8,
                                                    tp_size,
                                                    rank,
                                                    dim=1)
                # [hidden_size // GROUP_SIZE, hidden_size * 4] ->
                # [hidden_size // GROUP_SIZE, hidden_size * 4 // tp_size]
                scales_fp16 = torch_split(scales_fp16, tp_size, rank, dim=1)
                # [hidden_size // GROUP_SIZE, hidden_size * 4] ->
                # [hidden_size // GROUP_SIZE, hidden_size * 4 // tp_size]
                qzeros_unpacked_int8 = torch_split(qzeros_unpacked_int8,
                                                   tp_size,
                                                   rank,
                                                   dim=1)
                # [hidden_size * 4] -> [hidden_size * 4 // tp_size]
                biases_fp16 = numpy_split(biases_fp16, tp_size, rank, dim=0)

            qweight_fp32, scales_fp16, zeros_fp16 = preprocess_groupwise_weight_params(
                qweight_unpacked_int8, scales_fp16, qzeros_unpacked_int8)

            tensorrt_llm_gpt_neox.layers[layer_idx].mlp.fc.qweight.value = \
                qweight_fp32
            tensorrt_llm_gpt_neox.layers[layer_idx].mlp.fc.scale.value = \
                scales_fp16
            tensorrt_llm_gpt_neox.layers[layer_idx].mlp.fc.zero.value = \
                zeros_fp16
            tensorrt_llm_gpt_neox.layers[layer_idx].mlp.fc.bias.value = \
                biases_fp16

            qweight_int32 = hf_gpt_neox_state_dict.get(
                prefix + "mlp.dense_4h_to_h.qweight")
            scales_fp16 = hf_gpt_neox_state_dict.get(prefix +
                                                     "mlp.dense_4h_to_h.scales")
            qzeros_int32 = hf_gpt_neox_state_dict.get(
                prefix + "mlp.dense_4h_to_h.qzeros")
            biases_fp16 = hf_gpt_neox_state_dict.get(
                prefix + "mlp.dense_4h_to_h.bias").numpy()

            # [hidden_size * 4 // 8, hidden_size] -> [hidden_size * 4, hidden_size]
            qweight_unpacked_int8 = unpack_int32_into_int8(
                qweight_int32.T).contiguous() - 8
            qweight_unpacked_int8 = qweight_unpacked_int8.T.contiguous()

            # [hidden_size * 4 // GROUP_SIZE, hidden_size // 8] ->
            # [hidden_size * 4 // GROUP_SIZE, hidden_size]
            qzeros_unpacked_int8 = unpack_int32_into_int8(qzeros_int32)

            if tp_size > 1:
                # [hidden_size * 4, hidden_size] ->
                # [hidden_size * 4 // tp_size, hidden_size]
                qweight_unpacked_int8 = torch_split(qweight_unpacked_int8,
                                                    tp_size,
                                                    rank,
                                                    dim=0)
                # [hidden_size * 4 // GROUP_SIZE, hidden_size] ->
                # [hidden_size * 4 // GROUP_SIZE // tp_size, hidden_size] ->
                scales_fp16 = torch_split(scales_fp16, tp_size, rank, dim=0)
                # [hidden_size * 4 // GROUP_SIZE, hidden_size] ->
                # [hidden_size * 4 // GROUP_SIZE // tp_size, hidden_size]
                qzeros_unpacked_int8 = torch_split(qzeros_unpacked_int8,
                                                   tp_size,
                                                   rank,
                                                   dim=0)
                if rank > 0:
                    biases_fp16 = np.zeros_like(biases_fp16)

            qweight_fp32, scales_fp16, zeros_fp16 = preprocess_groupwise_weight_params(
                qweight_unpacked_int8, scales_fp16, qzeros_unpacked_int8)

            tensorrt_llm_gpt_neox.layers[layer_idx].mlp.proj.qweight.value = \
                qweight_fp32
            tensorrt_llm_gpt_neox.layers[layer_idx].mlp.proj.scale.value = \
                scales_fp16
            tensorrt_llm_gpt_neox.layers[layer_idx].mlp.proj.zero.value = \
                zeros_fp16
            tensorrt_llm_gpt_neox.layers[layer_idx].mlp.proj.bias.value = \
                biases_fp16

    v = hf_gpt_neox_state_dict.get('gpt_neox.final_layer_norm.weight')
    tensorrt_llm_gpt_neox.ln_f.weight.value = v.to(torch_dtype).cpu().numpy()

    v = hf_gpt_neox_state_dict.get('gpt_neox.final_layer_norm.bias')
    tensorrt_llm_gpt_neox.ln_f.bias.value = v.to(torch_dtype).cpu().numpy()

    v = hf_gpt_neox_state_dict.get('embed_out.weight').to(
        torch_dtype).cpu().numpy()
    if tp_size > 1:
        # [vocab_size, hidden_size] ->
        # [vocab_size // tp_size, hidden_size]
        if v.shape[0] % tp_size != 0:
            # padding
            vocab_size_padded = pad_vocab_size(v.shape[0], tp_size)
            pad_width = vocab_size_padded - v.shape[0]
            v = np.pad(v, ((0, pad_width), (0, 0)),
                       'constant',
                       constant_values=0)

        split_v = numpy_split(v, tp_size, rank, dim=0)
        tensorrt_llm_gpt_neox.lm_head.weight.value = split_v
    else:
        tensorrt_llm_gpt_neox.lm_head.weight.value = v

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    tensorrt_llm.logger.info(f'Weights loaded. Total time: {t}')
