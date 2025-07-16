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

import torch

from tensorrt_llm.models.convert_utils import (get_weight, get_weight_and_bias,
                                               split, split_matrix_tp,
                                               split_qkv_bias_tp, split_qkv_tp)

from ..._utils import pad_vocab_size


def shuffle_qkv_weights(weights, config):
    # Input weights are organized as
    # (q00, q01, ... q0m, k0, v0), (q10, q11, ... q1m, k1, v1), ... (qn0, qn1, ... qnm, kn, vn)
    # where n = num_kv_heads, m = num_attention_heads // num_kv_heads (i.e. #q_heads per kv_head)
    #
    # Output weights will be organized as
    # (q00, q01, ..., qnm), (k0, k1, .., kn), (v0, v1, .., vn)

    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    num_q_per_kv = num_heads // num_kv_heads

    hidden_size = config.hidden_size
    head_dim = hidden_size // num_heads

    input_shape = weights.shape
    if weights.dim() < 2:
        weights = weights.unsqueeze(1)

    weights = weights.reshape(num_kv_heads, (num_q_per_kv + 2), head_dim,
                              weights.shape[-1])
    q = weights[:, :-2, :, :]
    k = weights[:, -2, :, :]
    v = weights[:, -1, :, :]

    # num_heads x head_dim x hidden_size
    q = q.reshape(-1, q.shape[2], q.shape[3])

    # num_heads + (2 * num_kv_heads) x head_dim x hidden_size
    weights = torch.cat([q, k, v], dim=0)
    weights = weights.reshape(-1, weights.shape[2])

    weights = weights.squeeze()
    assert input_shape == weights.shape

    return weights


def split_embedding(
    param: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    use_parallel_embedding: bool = False,
    sharding_dim: int = 0,
) -> torch.Tensor:
    if param is None:
        return None
    if not use_parallel_embedding:
        return param

    vocab_size, hidden_size = param.size()
    if sharding_dim == 0:
        if vocab_size % tp_size != 0:
            vocab_size_padded = pad_vocab_size(vocab_size, tp_size)
            pad_width = vocab_size_padded - vocab_size
            param = torch.nn.functional.pad(param, (0, 0, 0, pad_width),
                                            value=0)
        else:
            assert hidden_size % tp_size == 0
    return split(param, tp_size, tp_rank, dim=sharding_dim)


def get_tllm_linear_weight(weight,
                           prefix,
                           bias=None,
                           use_weight_only=False,
                           plugin_weight_only_quant_type=torch.int8):
    results = {}
    if use_weight_only:
        v = weight.t().contiguous()
        processed_torch_weights, torch_weight_scales = \
            torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                v, plugin_weight_only_quant_type)
        results[prefix + '.weight'] = processed_torch_weights
        results[prefix + '.per_channel_scale'] = torch_weight_scales
    else:
        results[prefix + '.weight'] = weight.contiguous()

    if bias is not None:
        results[prefix + '.bias'] = bias

    return results


def split_weights_tp(config, weights, dtype):
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    hidden_size = config.hidden_size
    moe_variant = config.architecture == "PhiMoEForCausalLM"

    mha_mode = num_heads == num_kv_heads
    tp_size = config.mapping.tp_size
    rank = config.mapping.tp_rank
    moe_tp_size = config.mapping.moe_tp_size
    moe_tp_rank = config.mapping.moe_tp_rank
    use_weight_only = config.quant_mode.is_weight_only()
    plugin_weight_only_quant_type = None
    if use_weight_only and config.quant_mode.is_int8_weight_only() == 'int8':
        plugin_weight_only_quant_type = torch.int8
    elif use_weight_only and config.quant_mode.is_int4_weight_only() == 'int4':
        plugin_weight_only_quant_type = torch.quint4x2

    def get_quant_weight(weight, prefix, bias):
        return get_tllm_linear_weight(weight, prefix, bias, use_weight_only,
                                      plugin_weight_only_quant_type)

    for layer_id in range(config.num_hidden_layers):
        layer_prefix = f"transformer.layers.{layer_id}."

        prefix = layer_prefix + 'attention.qkv'
        qkv_weight, qkv_bias = get_weight_and_bias(weights, prefix, dtype)

        if not mha_mode:
            num_q_per_kv = num_heads // num_kv_heads

            qkv_weight = qkv_weight.reshape(num_q_per_kv + 2, -1, hidden_size)
            q = qkv_weight[:num_q_per_kv, :, :].reshape(-1, hidden_size)
            k = qkv_weight[num_q_per_kv:num_q_per_kv + 1, :, :].reshape(
                -1, hidden_size)
            v = qkv_weight[num_q_per_kv + 1:num_q_per_kv + 2, :, :].reshape(
                -1, hidden_size)
            split_weight = torch.cat(
                [split(x, tp_size, rank) for x in [q, k, v]], dim=0)
            if qkv_bias is not None:
                qkv_bias = qkv_bias.reshape(num_q_per_kv + 2, -1)
                q = qkv_bias[:num_q_per_kv, :].reshape(-1)
                k = qkv_bias[num_q_per_kv:num_q_per_kv + 1, :].reshape(-1)
                v = qkv_bias[num_q_per_kv + 1:num_q_per_kv + 2, :].reshape(-1)
                split_bias = torch.cat(
                    [split(x, tp_size, rank) for x in [q, k, v]], dim=0)
            else:
                split_bias = None
        else:
            split_weight = split_qkv_tp(qkv_weight, num_heads, hidden_size,
                                        tp_size, rank)
            if qkv_bias is not None:
                split_bias = split_qkv_bias_tp(qkv_bias, num_heads, hidden_size,
                                               tp_size, rank)
            else:
                split_bias = None
        weights.update(get_quant_weight(split_weight, prefix, split_bias))

        prefix = layer_prefix + 'attention.dense'
        attn_dense_weight, attn_dense_bias = get_weight_and_bias(
            weights, prefix, dtype)
        split_v = split_matrix_tp(attn_dense_weight, tp_size, rank, dim=1)
        weights.update(get_quant_weight(split_v, prefix, attn_dense_bias))

        prefix = layer_prefix + 'mlp.fc'
        if not moe_variant:
            mlp_fc_weight, mlp_fc_bias = get_weight_and_bias(
                weights, prefix, dtype)
            split_v = split_matrix_tp(mlp_fc_weight, tp_size, rank, dim=0)
            if mlp_fc_bias is not None:
                bias = split_matrix_tp(mlp_fc_bias, tp_size, rank, dim=0)
            else:
                bias = None
            weights.update(get_quant_weight(split_v, prefix, bias))
        else:
            mlp_fc_weight = get_weight(weights, prefix, dtype)
            w3 = split_matrix_tp(mlp_fc_weight, 2, 0, dim=1)
            split_w3 = split_matrix_tp(w3, moe_tp_size, moe_tp_rank, dim=1)
            w1 = split_matrix_tp(mlp_fc_weight, 2, 1, dim=1)
            split_w1 = split_matrix_tp(w1, moe_tp_size, moe_tp_rank, dim=1)
            split_v = torch.concat([split_w3, split_w1], dim=-2)
            weights.update(get_quant_weight(split_v, prefix, None))

        prefix = layer_prefix + 'mlp.proj'
        if not moe_variant:
            mlp_proj_weight, mlp_proj_bias = get_weight_and_bias(
                weights, prefix, dtype)
            split_v = split_matrix_tp(mlp_proj_weight, tp_size, rank, dim=1)
            weights.update(get_quant_weight(split_v, prefix, mlp_proj_bias))
        else:
            mlp_proj_weight = get_weight(weights, prefix, dtype)
            split_v = split_matrix_tp(mlp_proj_weight,
                                      moe_tp_size,
                                      moe_tp_rank,
                                      dim=2)
            weights.update(get_quant_weight(split_v, prefix, None))

    weights['transformer.vocab_embedding.weight'] = split_embedding(
        weights['transformer.vocab_embedding.weight'], tp_size, rank)
    weights['lm_head.weight'] = split_matrix_tp(weights['lm_head.weight'],
                                                tp_size,
                                                rank,
                                                dim=0)
    if moe_variant:
        weights['lm_head.bias'] = split_matrix_tp(weights['lm_head.bias'],
                                                  tp_size,
                                                  rank,
                                                  dim=0)

    return weights
