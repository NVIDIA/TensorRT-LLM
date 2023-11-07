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
import time

import torch
import torch.nn.functional as F

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy
from tensorrt_llm.quantization import QuantMode


def tile_kv_weight_bias(v, kv_num_head, tp_size):
    head_size = v.shape[0] // kv_num_head
    reps = tp_size // kv_num_head
    if v.ndim == 1:
        v = v.reshape(kv_num_head, head_size)[:, None, :]
        v = v.expand(kv_num_head, reps, head_size).reshape(-1).clone()
    else:
        hidden_size = v.shape[1]
        v = v.reshape(kv_num_head, head_size, hidden_size)[:, None, :, :]
        v = v.expand(kv_num_head, reps, head_size,
                     hidden_size).reshape(-1, hidden_size).clone()
    return v


def split_qkv(v, tp_size, rank, hidden_size, num_heads, num_kv_heads):
    head_size = hidden_size // num_heads
    if tp_size == 1:
        return v

    assert v.shape[0] == hidden_size + head_size * num_kv_heads * 2
    query = v[:hidden_size]
    key = v[hidden_size:hidden_size + head_size * num_kv_heads]
    value = v[hidden_size + head_size * num_kv_heads:hidden_size +
              head_size * num_kv_heads * 2]

    if num_kv_heads < tp_size:
        key = tile_kv_weight_bias(key, num_kv_heads, tp_size)
        value = tile_kv_weight_bias(value, num_kv_heads, tp_size)
    assert (key.shape[0] % (tp_size * head_size)) == 0
    assert (value.shape[0] % (tp_size * head_size)) == 0

    q_tmp = torch.chunk(query, tp_size, dim=0)[rank]
    k_tmp = torch.chunk(key, tp_size, dim=0)[rank]
    v_tmp = torch.chunk(value, tp_size, dim=0)[rank]
    return torch.concatenate([q_tmp, k_tmp, v_tmp], dim=0).contiguous()


def load_quant_weight(src, value_dst, scale_dst, plugin_weight_only_quant_type):
    v = torch.transpose(src, dim0=0, dim1=1).contiguous()
    processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
        v, plugin_weight_only_quant_type)
    value_dst.value = torch_to_numpy(processed_torch_weights)
    scale_dst.value = torch_to_numpy(torch_weight_scales)


def load_from_hf(
    trt_model,
    hf_model,
    mapping=None,
    dtype="float32",
    model_version="3",
    multi_query_mode=False,
):
    # [TODO] Merge model_version=="1" and model_version>="2"
    tensorrt_llm.logger.info("Loading weights from HF")
    tik = time.time()

    torch_type = str_dtype_to_torch(dtype)
    quant_mode = getattr(trt_model, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()

    hidden_size = hf_model.config.hidden_size
    num_heads = hf_model.config.num_attention_heads

    layers_per_pipeline_stage = trt_model.num_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage))
    feed_weight_count = 0

    if model_version == "1":
        num_kv_heads = hf_model.config.num_attention_heads

        if mapping.is_first_pp_rank():
            # Embedding
            weight = hf_model.transformer.word_embeddings.weight.to(
                torch_type).detach().cpu()
            trt_model.embedding.weight.value = torch_to_numpy(weight)
            feed_weight_count += 1
        if mapping.is_last_pp_rank():
            # Final normalization
            weight = hf_model.transformer.final_layernorm.weight.to(
                torch_type).detach().cpu()
            trt_model.final_norm.weight.value = torch_to_numpy(weight)
            bias = hf_model.transformer.final_layernorm.bias.to(
                torch_type).detach().cpu()
            trt_model.final_norm.bias.value = torch_to_numpy(bias)
            feed_weight_count += 2

            # Final LM
            weight = hf_model.lm_head.weight.to(torch_type).detach().cpu()
            if weight.shape[0] % mapping.tp_size != 0:
                pad_width = trt_model.lm_head.out_features * mapping.tp_size - weight.shape[
                    0]
                weight = F.pad(weight, (0, 0, 0, pad_width))
            split_weight = torch.chunk(weight, mapping.tp_size,
                                       dim=0)[mapping.rank]
            trt_model.lm_head.weight.value = torch_to_numpy(split_weight)
            feed_weight_count += 1

        for layer_idx in range(28):
            if layer_idx not in layers_range:
                continue
            i = int(layer_idx) - mapping.pp_rank * layers_per_pipeline_stage
            if i >= trt_model.num_layers:
                continue

            # Pre normalization
            weight = hf_model.transformer.layers[i].input_layernorm.weight.to(
                torch_type).detach().cpu()
            trt_model.layers[i].pre_norm.weight.value = torch_to_numpy(weight)
            bias = hf_model.transformer.layers[i].input_layernorm.bias.to(
                torch_type).detach().cpu()
            trt_model.layers[i].pre_norm.bias.value = torch_to_numpy(bias)
            feed_weight_count += 2

            # QKV multiplication weight
            weight = hf_model.transformer.layers[
                i].attention.query_key_value.weight.to(
                    torch_type).detach().cpu()
            split_weight = split_qkv(weight, mapping.tp_size, mapping.tp_rank,
                                     hidden_size, num_heads, num_kv_heads)
            dst = trt_model.layers[i].attention.qkv
            if use_weight_only:
                load_quant_weight(
                    src=split_weight,
                    value_dst=dst.weight,
                    scale_dst=dst.per_channel_scale,
                    plugin_weight_only_quant_type=plugin_weight_only_quant_type)
            else:
                dst.weight.value = torch_to_numpy(split_weight)
            feed_weight_count += 1

            # QKV multiplication bias
            bias = hf_model.transformer.layers[
                i].attention.query_key_value.bias.to(torch_type).detach().cpu()
            split_bias = split_qkv(bias, mapping.tp_size, mapping.tp_rank,
                                   hidden_size, num_heads, num_kv_heads)
            trt_model.layers[i].attention.qkv.bias.value = torch_to_numpy(
                split_bias)
            feed_weight_count += 1

            # Dense multiplication weight (no bias)
            weight = hf_model.transformer.layers[i].attention.dense.weight.to(
                torch_type).detach().cpu()
            split_weight = torch.chunk(weight, mapping.tp_size,
                                       dim=1)[mapping.rank]
            dst = trt_model.layers[i].attention.dense
            if use_weight_only:
                load_quant_weight(
                    src=split_weight,
                    value_dst=dst.weight,
                    scale_dst=dst.per_channel_scale,
                    plugin_weight_only_quant_type=plugin_weight_only_quant_type)
            else:
                dst.weight.value = torch_to_numpy(split_weight)
            feed_weight_count += 1

            # Post normalization
            weight = hf_model.transformer.layers[
                i].post_attention_layernorm.weight.to(
                    torch_type).detach().cpu()
            trt_model.layers[i].post_norm.weight.value = torch_to_numpy(weight)
            bias = hf_model.transformer.layers[
                i].post_attention_layernorm.bias.to(torch_type).detach().cpu()
            trt_model.layers[i].post_norm.bias.value = torch_to_numpy(bias)
            feed_weight_count += 2

            # Multilayer perceptron h -> 4h (no bias)
            weight = hf_model.transformer.layers[i].mlp.dense_h_to_4h.weight.to(
                torch_type).detach().cpu()
            split_weight = torch.chunk(weight, mapping.tp_size,
                                       dim=0)[mapping.rank]
            dst = trt_model.layers[i].mlp.fc
            if use_weight_only:
                load_quant_weight(
                    src=split_weight,
                    value_dst=dst.weight,
                    scale_dst=dst.per_channel_scale,
                    plugin_weight_only_quant_type=plugin_weight_only_quant_type)
            else:
                dst.weight.value = torch_to_numpy(split_weight)
            feed_weight_count += 1

            # Multilayer perceptron 4h -> h (no bias)
            weight = hf_model.transformer.layers[i].mlp.dense_4h_to_h.weight.to(
                torch_type).detach().cpu()
            split_weight = torch.chunk(weight, mapping.tp_size,
                                       dim=1)[mapping.rank]
            dst = trt_model.layers[i].mlp.proj
            if use_weight_only:
                load_quant_weight(
                    src=split_weight,
                    value_dst=dst.weight,
                    scale_dst=dst.per_channel_scale,
                    plugin_weight_only_quant_type=plugin_weight_only_quant_type)
            else:
                dst.weight.value = torch_to_numpy(split_weight)
            feed_weight_count += 1

        assert feed_weight_count == 4 + trt_model.num_layers * 9, "Some weights not loaded from HF"

    else:
        num_kv_heads = hf_model.config.multi_query_group_num

        if mapping.is_first_pp_rank():
            # Embedding
            weight = hf_model.transformer.embedding.word_embeddings.weight.to(
                torch_type).detach().cpu()
            trt_model.embedding.weight.value = torch_to_numpy(weight)
            feed_weight_count += 1
        if mapping.is_last_pp_rank():
            # Final normalization
            weight = hf_model.transformer.encoder.final_layernorm.weight.to(
                torch_type).detach().cpu()
            trt_model.final_norm.weight.value = torch_to_numpy(weight)
            feed_weight_count += 1

            # Final LM
            weight = hf_model.transformer.output_layer.weight.to(
                torch_type).detach().cpu()
            if weight.shape[0] % mapping.tp_size != 0:
                pad_width = trt_model.lm_head.out_features * mapping.tp_size - weight.shape[
                    0]
                weight = F.pad(weight, (0, 0, 0, pad_width))
            split_weight = torch.chunk(weight, mapping.tp_size,
                                       dim=0)[mapping.rank]
            trt_model.lm_head.weight.value = torch_to_numpy(split_weight)
            feed_weight_count += 1

        for layer_idx in range(28):
            if layer_idx not in layers_range:
                continue
            i = int(layer_idx) - mapping.pp_rank * layers_per_pipeline_stage
            if i >= trt_model.num_layers:
                continue

            # Pre normalization
            weight = hf_model.transformer.encoder.layers[
                i].input_layernorm.weight.to(torch_type).detach().cpu()
            trt_model.layers[i].pre_norm.weight.value = torch_to_numpy(weight)
            feed_weight_count += 1

            # QKV multiplication weight
            weight = hf_model.transformer.encoder.layers[
                i].self_attention.query_key_value.weight.to(
                    torch_type).detach().cpu()
            split_weight = split_qkv(weight, mapping.tp_size, mapping.tp_rank,
                                     hidden_size, num_heads, num_kv_heads)
            dst = trt_model.layers[i].attention.qkv
            if use_weight_only:
                load_quant_weight(
                    src=split_weight,
                    value_dst=dst.weight,
                    scale_dst=dst.per_channel_scale,
                    plugin_weight_only_quant_type=plugin_weight_only_quant_type)
            else:
                dst.weight.value = torch_to_numpy(split_weight)
            feed_weight_count += 1

            # QKV multiplication bias
            bias = hf_model.transformer.encoder.layers[
                i].self_attention.query_key_value.bias.to(
                    torch_type).detach().cpu()
            split_bias = split_qkv(bias, mapping.tp_size, mapping.tp_rank,
                                   hidden_size, num_heads, num_kv_heads)
            trt_model.layers[i].attention.qkv.bias.value = torch_to_numpy(
                split_bias)
            feed_weight_count += 1

            # Dense multiplication weight (no bias)
            weight = hf_model.transformer.encoder.layers[
                i].self_attention.dense.weight.to(torch_type).detach().cpu()
            split_weight = torch.chunk(weight, mapping.tp_size,
                                       dim=1)[mapping.rank]
            dst = trt_model.layers[i].attention.dense
            if use_weight_only:
                load_quant_weight(
                    src=split_weight,
                    value_dst=dst.weight,
                    scale_dst=dst.per_channel_scale,
                    plugin_weight_only_quant_type=plugin_weight_only_quant_type)
            else:
                dst.weight.value = torch_to_numpy(split_weight)
            feed_weight_count += 1

            # Post normalization
            weight = hf_model.transformer.encoder.layers[
                i].post_attention_layernorm.weight.to(
                    torch_type).detach().cpu()
            trt_model.layers[i].post_norm.weight.value = torch_to_numpy(weight)
            feed_weight_count += 1

            # Multilayer perceptron h -> 4h (no bias)
            weight = hf_model.transformer.encoder.layers[
                i].mlp.dense_h_to_4h.weight.to(torch_type).detach().cpu()
            split_weight = torch.chunk(weight, 2 * mapping.tp_size, dim=0)
            # swap first and second half weight in columns to adapt trt_llm Swiglu
            split_weight = torch.cat(
                [
                    split_weight[mapping.rank + mapping.tp_size],
                    split_weight[mapping.rank],
                ],
                dim=0,
            )
            dst = trt_model.layers[i].mlp.fc
            if use_weight_only:
                load_quant_weight(
                    src=split_weight,
                    value_dst=dst.weight,
                    scale_dst=dst.per_channel_scale,
                    plugin_weight_only_quant_type=plugin_weight_only_quant_type)
            else:
                dst.weight.value = torch_to_numpy(split_weight)
            feed_weight_count += 1

            # Multilayer perceptron 4h -> h (no bias)
            weight = hf_model.transformer.encoder.layers[
                i].mlp.dense_4h_to_h.weight.to(torch_type).detach().cpu()
            split_weight = torch.chunk(weight, mapping.tp_size,
                                       dim=1)[mapping.rank]
            dst = trt_model.layers[i].mlp.proj
            if use_weight_only:
                load_quant_weight(
                    src=split_weight,
                    value_dst=dst.weight,
                    scale_dst=dst.per_channel_scale,
                    plugin_weight_only_quant_type=plugin_weight_only_quant_type)
            else:
                dst.weight.value = torch_to_numpy(split_weight)
            feed_weight_count += 1

        assert feed_weight_count == 3 + trt_model.num_layers * 7, "Some weights not loaded from HF"

    tok = time.time()

    tensorrt_llm.logger.info("Loading weights finish in %.2fs" % (tok - tik))
    return trt_model
