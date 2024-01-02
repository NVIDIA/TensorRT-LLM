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
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import transformers

import tensorrt_llm.logger as logger
from tensorrt_llm._utils import (str_dtype_to_np, str_dtype_to_torch,
                                 torch_to_numpy)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.quantized.quant import get_dummy_quant_scales
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
    processed_torch_weights, torch_weight_scales = \
        torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
            v, plugin_weight_only_quant_type)
    value_dst.value = torch_to_numpy(processed_torch_weights)
    scale_dst.value = torch_to_numpy(torch_weight_scales)


def load_from_hf(
    trtllm_model,
    hf_model_dir: Path = None,
    mapping: Mapping = Mapping(),
    dtype: str = "float32",
    model_name: str = None,
):

    assert model_name is not None, "Model name must be set"

    if not hf_model_dir.exists():
        logger.info(f"No weight file found from {hf_model_dir}")
        logger.info(f"Use random weights")
        return
    else:
        logger.info("Loading weights from HF")

    tik = time.time()

    hf_model = transformers.AutoModel.from_pretrained(hf_model_dir,
                                                      trust_remote_code=True)
    hidden_size = hf_model.config.hidden_size
    num_heads = hf_model.config.num_attention_heads
    num_layers = hf_model.config.num_layers

    torch_type = str_dtype_to_torch(dtype)
    quant_mode = getattr(trtllm_model, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()

    layers_per_pipeline_stage = num_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage))
    feed_weight_count = 0

    if model_name in ["chatglm_6b", "glm_10b"]:
        num_kv_heads = hf_model.config.num_attention_heads
    elif model_name in [
            "chatglm2_6b",
            "chatglm2_6b_32k",
            "chatglm3_6b",
            "chatglm3_6b_base",
            "chatglm3_6b_32k",
    ]:
        num_kv_heads = hf_model.config.multi_query_group_num

    if mapping.is_first_pp_rank():
        # Embedding
        if model_name in ["chatglm_6b"]:
            weight = hf_model.transformer.word_embeddings.weight.to(
                torch_type).detach()
            trtllm_model.embedding.weight.value = torch_to_numpy(weight)
            feed_weight_count += 1
        elif model_name in [
                "chatglm2_6b",
                "chatglm2_6b_32k",
                "chatglm3_6b",
                "chatglm3_6b_base",
                "chatglm3_6b_32k",
        ]:
            weight = hf_model.transformer.embedding.word_embeddings.weight.to(
                torch_type).detach()
            trtllm_model.embedding.weight.value = torch_to_numpy(weight)
            feed_weight_count += 1
        elif model_name in ["glm_10b"]:
            weight = hf_model.word_embeddings.weight.to(torch_type).detach()
            trtllm_model.embedding.weight.value = torch_to_numpy(weight)
            weight = hf_model.transformer.position_embeddings.weight.to(
                torch_type).detach()
            trtllm_model.position_embeddings.weight.value = torch_to_numpy(
                weight)
            weight = hf_model.transformer.block_position_embeddings.weight.to(
                torch_type).detach()
            trtllm_model.block_embeddings.weight.value = torch_to_numpy(weight)
            feed_weight_count += 3

    if mapping.is_last_pp_rank():
        # Final normalization
        if model_name in ["chatglm_6b"]:
            weight = hf_model.transformer.final_layernorm.weight.to(
                torch_type).detach()
            trtllm_model.final_norm.weight.value = torch_to_numpy(weight)
            bias = hf_model.transformer.final_layernorm.bias.to(
                torch_type).detach()
            trtllm_model.final_norm.bias.value = torch_to_numpy(bias)
            feed_weight_count += 2
        elif model_name in [
                "chatglm2_6b",
                "chatglm2_6b_32k",
                "chatglm3_6b",
                "chatglm3_6b_base",
                "chatglm3_6b_32k",
        ]:
            weight = hf_model.transformer.encoder.final_layernorm.weight.to(
                torch_type).detach()
            trtllm_model.final_norm.weight.value = torch_to_numpy(weight)
            feed_weight_count += 1
        elif model_name in ["glm_10b"]:
            weight = hf_model.transformer.final_layernorm.weight.to(
                torch_type).detach()
            trtllm_model.final_norm.weight.value = torch_to_numpy(weight)
            bias = hf_model.transformer.final_layernorm.bias.to(
                torch_type).detach()
            trtllm_model.final_norm.bias.value = torch_to_numpy(bias)
            feed_weight_count += 2

        # Final LM
        if model_name in ["chatglm_6b"]:
            weight = hf_model.lm_head.weight.to(torch_type).detach()
            if weight.shape[0] % mapping.tp_size != 0:
                pad_width = trtllm_model.lm_head.out_features * mapping.tp_size - weight.shape[
                    0]
                weight = F.pad(weight, (0, 0, 0, pad_width))
            split_weight = torch.chunk(weight, mapping.tp_size,
                                       dim=0)[mapping.rank]
            trtllm_model.lm_head.weight.value = torch_to_numpy(split_weight)
            feed_weight_count += 1
        elif model_name in [
                "chatglm2_6b",
                "chatglm2_6b_32k",
                "chatglm3_6b",
                "chatglm3_6b_base",
                "chatglm3_6b_32k",
        ]:
            weight = hf_model.transformer.output_layer.weight.to(
                torch_type).detach()
            if weight.shape[0] % mapping.tp_size != 0:
                pad_width = trtllm_model.lm_head.out_features * mapping.tp_size - weight.shape[
                    0]
                weight = F.pad(weight, (0, 0, 0, pad_width))
            split_weight = torch.chunk(weight, mapping.tp_size,
                                       dim=0)[mapping.rank]
            trtllm_model.lm_head.weight.value = torch_to_numpy(split_weight)
            feed_weight_count += 1
        elif model_name in ["glm_10b"]:
            weight = hf_model.word_embeddings.weight.to(torch_type).detach()
            if weight.shape[0] % mapping.tp_size != 0:
                pad_width = trtllm_model.lm_head.out_features * mapping.tp_size - weight.shape[
                    0]
                weight = F.pad(weight, (0, 0, 0, pad_width))
            split_weight = torch.chunk(weight, mapping.tp_size,
                                       dim=0)[mapping.rank]
            trtllm_model.lm_head.weight.value = torch_to_numpy(split_weight)
            feed_weight_count += 1

    # Weight per layer
    for layer_idx in layers_range:
        i = layer_idx - mapping.pp_rank * layers_per_pipeline_stage
        layer = trtllm_model.layers[i]

        # Pre normalization
        if model_name in ["chatglm_6b"]:
            weight = hf_model.transformer.layers[i].input_layernorm.weight.to(
                torch_type).detach()
            layer.pre_norm.weight.value = torch_to_numpy(weight)
            bias = hf_model.transformer.layers[i].input_layernorm.bias.to(
                torch_type).detach()
            layer.pre_norm.bias.value = torch_to_numpy(bias)
            feed_weight_count += 2
        elif model_name in [
                "chatglm2_6b",
                "chatglm2_6b_32k",
                "chatglm3_6b",
                "chatglm3_6b_base",
                "chatglm3_6b_32k",
        ]:
            weight = hf_model.transformer.encoder.layers[
                i].input_layernorm.weight.to(torch_type).detach()
            layer.pre_norm.weight.value = torch_to_numpy(weight)
            feed_weight_count += 1
        elif model_name in ["glm_10b"]:
            weight = hf_model.transformer.layers[i].input_layernorm.weight.to(
                torch_type).detach()
            layer.pre_norm.weight.value = torch_to_numpy(weight)
            bias = hf_model.transformer.layers[i].input_layernorm.bias.to(
                torch_type).detach()
            layer.pre_norm.bias.value = torch_to_numpy(bias)
            feed_weight_count += 2

        # QKV multiplication weight
        if model_name in ["chatglm_6b"]:
            weight = hf_model.transformer.layers[
                i].attention.query_key_value.weight.to(torch_type).detach()
        elif model_name in [
                "chatglm2_6b",
                "chatglm2_6b_32k",
                "chatglm3_6b",
                "chatglm3_6b_base",
                "chatglm3_6b_32k",
        ]:
            weight = hf_model.transformer.encoder.layers[
                i].self_attention.query_key_value.weight.to(
                    torch_type).detach()
        elif model_name in ["glm_10b"]:
            weight = hf_model.transformer.layers[
                i].attention.query_key_value.weight.to(torch_type).detach()

        split_weight = split_qkv(weight, mapping.tp_size, mapping.tp_rank,
                                 hidden_size, num_heads, num_kv_heads)
        dst = layer.attention.qkv
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
        if model_name in ["chatglm_6b"]:
            bias = hf_model.transformer.layers[
                i].attention.query_key_value.bias.to(torch_type).detach()
        elif model_name in [
                "chatglm2_6b",
                "chatglm2_6b_32k",
                "chatglm3_6b",
                "chatglm3_6b_base",
                "chatglm3_6b_32k",
        ]:
            bias = hf_model.transformer.encoder.layers[
                i].self_attention.query_key_value.bias.to(torch_type).detach()
        elif model_name in ["glm_10b"]:
            bias = hf_model.transformer.layers[
                i].attention.query_key_value.bias.to(torch_type).detach()

        split_bias = split_qkv(bias, mapping.tp_size, mapping.tp_rank,
                               hidden_size, num_heads, num_kv_heads)
        layer.attention.qkv.bias.value = torch_to_numpy(split_bias)
        feed_weight_count += 1

        # Dense multiplication weight
        if model_name in ["chatglm_6b"]:
            weight = hf_model.transformer.layers[i].attention.dense.weight.to(
                torch_type).detach()
        elif model_name in [
                "chatglm2_6b",
                "chatglm2_6b_32k",
                "chatglm3_6b",
                "chatglm3_6b_base",
                "chatglm3_6b_32k",
        ]:
            weight = hf_model.transformer.encoder.layers[
                i].self_attention.dense.weight.to(torch_type).detach()
        elif model_name in ["glm_10b"]:
            weight = hf_model.transformer.layers[i].attention.dense.weight.to(
                torch_type).detach()

        split_weight = torch.chunk(weight, mapping.tp_size, dim=1)[mapping.rank]
        dst = layer.attention.dense
        if use_weight_only:
            load_quant_weight(
                src=split_weight,
                value_dst=dst.weight,
                scale_dst=dst.per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
        else:
            dst.weight.value = np.ascontiguousarray(
                torch_to_numpy(split_weight))
        feed_weight_count += 1

        # Dense multiplication bias, only GLM-10B
        if model_name in ["glm_10b"]:
            bias = hf_model.transformer.layers[i].attention.dense.bias.to(
                torch_type).detach()
            split_bias = split_qkv(bias, mapping.tp_size, mapping.tp_rank,
                                   hidden_size, num_heads, num_kv_heads)
            layer.attention.dense.bias.value = torch_to_numpy(split_bias)
            feed_weight_count += 1

        # Post normalization
        if model_name in ["chatglm_6b"]:
            weight = hf_model.transformer.layers[
                i].post_attention_layernorm.weight.to(torch_type).detach()
            layer.post_norm.weight.value = torch_to_numpy(weight)
            bias = hf_model.transformer.layers[
                i].post_attention_layernorm.bias.to(torch_type).detach()
            layer.post_norm.bias.value = torch_to_numpy(bias)
            feed_weight_count += 2
        elif model_name in [
                "chatglm2_6b",
                "chatglm2_6b_32k",
                "chatglm3_6b",
                "chatglm3_6b_base",
                "chatglm3_6b_32k",
        ]:
            weight = hf_model.transformer.encoder.layers[
                i].post_attention_layernorm.weight.to(torch_type).detach()
            layer.post_norm.weight.value = torch_to_numpy(weight)
            feed_weight_count += 1
        elif model_name in ["glm_10b"]:
            weight = hf_model.transformer.layers[
                i].post_attention_layernorm.weight.to(torch_type).detach()
            layer.post_norm.weight.value = torch_to_numpy(weight)
            bias = hf_model.transformer.layers[
                i].post_attention_layernorm.bias.to(torch_type).detach()
            layer.post_norm.bias.value = torch_to_numpy(bias)
            feed_weight_count += 2

        # Multilayer perceptron h -> 4h weight
        if model_name in ["chatglm_6b"]:
            weight = hf_model.transformer.layers[i].mlp.dense_h_to_4h.weight.to(
                torch_type).detach()
            split_weight = torch.chunk(weight, mapping.tp_size,
                                       dim=0)[mapping.rank]
        elif model_name in [
                "chatglm2_6b",
                "chatglm2_6b_32k",
                "chatglm3_6b",
                "chatglm3_6b_base",
                "chatglm3_6b_32k",
        ]:
            weight = hf_model.transformer.encoder.layers[
                i].mlp.dense_h_to_4h.weight.to(torch_type).detach()
            split_weight = torch.chunk(weight, 2 * mapping.tp_size, dim=0)
            # swap first and second half weight in columns to adapt trt_llm Swiglu
            split_weight = torch.cat(
                [
                    split_weight[mapping.rank + mapping.tp_size],
                    split_weight[mapping.rank],
                ],
                dim=0,
            )
        elif model_name in ["glm_10b"]:
            weight = hf_model.transformer.layers[i].mlp.dense_h_to_4h.weight.to(
                torch_type).detach()
            split_weight = torch.chunk(weight, mapping.tp_size,
                                       dim=0)[mapping.rank]

        dst = layer.mlp.fc
        if use_weight_only:
            load_quant_weight(
                src=split_weight,
                value_dst=dst.weight,
                scale_dst=dst.per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
        else:
            dst.weight.value = torch_to_numpy(split_weight)
        feed_weight_count += 1

        # Multilayer perceptron h -> 4h bias, only GLM-10B
        if model_name in ["glm_10b"]:
            bias = hf_model.transformer.layers[i].mlp.dense_h_to_4h.bias.to(
                torch_type).detach()
            split_bias = split_qkv(bias, mapping.tp_size, mapping.tp_rank,
                                   hidden_size, num_heads, num_kv_heads)
            layer.mlp.fc.bias.value = torch_to_numpy(split_bias)
            feed_weight_count += 1

        # Multilayer perceptron 4h -> h weight
        if model_name in ["chatglm_6b"]:
            weight = hf_model.transformer.layers[i].mlp.dense_4h_to_h.weight.to(
                torch_type).detach()
        elif model_name in [
                "chatglm2_6b",
                "chatglm2_6b_32k",
                "chatglm3_6b",
                "chatglm3_6b_base",
                "chatglm3_6b_32k",
        ]:
            weight = hf_model.transformer.encoder.layers[
                i].mlp.dense_4h_to_h.weight.to(torch_type).detach()
        elif model_name in ["glm_10b"]:
            weight = hf_model.transformer.layers[i].mlp.dense_4h_to_h.weight.to(
                torch_type).detach()

        split_weight = torch.chunk(weight, mapping.tp_size, dim=1)[mapping.rank]
        dst = layer.mlp.proj
        if use_weight_only:
            load_quant_weight(
                src=split_weight,
                value_dst=dst.weight,
                scale_dst=dst.per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
        else:
            dst.weight.value = np.ascontiguousarray(
                torch_to_numpy(split_weight))
        feed_weight_count += 1

        # Multilayer perceptron 4h -> h bias, only GLM-10B
        if model_name in ["glm_10b"]:
            bias = hf_model.transformer.layers[i].mlp.dense_4h_to_h.bias.to(
                torch_type).detach()
            split_bias = split_qkv(bias, mapping.tp_size, mapping.tp_rank,
                                   hidden_size, num_heads, num_kv_heads)
            layer.mlp.proj.bias.value = torch_to_numpy(split_bias)
            feed_weight_count += 1

    del hf_model
    tok = time.time()

    # Final check
    if model_name in ["chatglm_6b"]:
        weight_count = 4 + num_layers * 9
    elif model_name in [
            "chatglm2_6b",
            "chatglm2_6b_32k",
            "chatglm3_6b",
            "chatglm3_6b_base",
            "chatglm3_6b_32k",
    ]:
        weight_count = 3 + num_layers * 7
    elif model_name in ["glm_10b"]:
        weight_count = 6 + num_layers * 12
    if feed_weight_count < weight_count:
        logger.error("%d weights not loaded from HF" %
                     (weight_count - feed_weight_count))
        return None
    logger.info("Loading weights finish in %.2fs" % (tok - tik))
    return


def load_from_awq(
        trtllm_model,
        quant_ckpt_path,
        mapping=Mapping(),
        dtype="float16",
        model_name=None,
):

    assert model_name is not None, "Model name must be set"
    assert model_name in [
            "chatglm2_6b",
            "chatglm2_6b_32k",
            "chatglm3_6b",
            "chatglm3_6b_base",
            "chatglm3_6b_32k",
    ], \
        f"INT4-AWQ is not supported in {model_name} yet."

    if not (Path(quant_ckpt_path).exists() and quant_ckpt_path.endswith(".pt")):
        logger.info(f"No .pt weight file found from {quant_ckpt_path}")
        return
    else:
        logger.info("Loading weights from groupwise AWQ checkpoint")

    tik = time.time()

    awq_weight = torch.load(quant_ckpt_path, map_location=torch.device('cpu'))

    num_layers = trtllm_model.num_layers
    name = "transformer.encoder.layers.0.self_attention.query_key_value"
    group_size = awq_weight[name + ".weight"].numel() // \
        awq_weight[name + ".weight_quantizer._amax"].numel()

    torch_dtype = str_dtype_to_torch(dtype)
    int8_to_int4 = torch.ops.fastertransformer.pack_int8_tensor_to_packed_int4
    int4_to_int4x2 = torch.ops.fastertransformer.preprocess_weights_for_mixed_gemm

    layers_per_pipeline_stage = num_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))
    feed_weight_count = 0

    def AWQ_quantize_pack_preprocess(weight, scale):
        weight = weight / scale.repeat_interleave(group_size, dim=0)
        weight = torch.clamp(torch.round(weight.cuda()).char(), -8, 7)
        weight = int8_to_int4(weight.cpu())
        weight = int4_to_int4x2(weight, torch.quint4x2)
        return weight.view(torch.float16)

    def process_and_assign_weight(op, prefix, tp_dim=0):
        name = prefix + ".weight"
        weight = awq_weight[name].T.contiguous()
        k, n = weight.shape
        tp_size = mapping.tp_size
        tp_rank = mapping.tp_rank
        weight = weight.split(weight.shape[tp_dim] // tp_size,
                              dim=tp_dim)[tp_rank]

        name = prefix + ".weight_quantizer._amax"
        amax = awq_weight[name]
        amax = amax.reshape(n, k // group_size).T.contiguous()
        amax = amax.split(amax.shape[tp_dim] // tp_size, dim=tp_dim)[tp_rank]

        name = prefix + ".input_quantizer._pre_quant_scale"
        pre_quant_scale = awq_weight[name].reshape(1, k)
        if tp_dim == 0:
            pre_quant_scale = pre_quant_scale.split(k // mapping.tp_size,
                                                    dim=1)[mapping.tp_rank]

        scale = amax / 8.0
        op.weight.value = AWQ_quantize_pack_preprocess(weight, scale)
        op.weights_scaling_factor.value = scale.to(torch_dtype).cpu().numpy()
        op.prequant_scaling_factor.value = pre_quant_scale.to(
            torch_dtype).cpu().numpy()

    if mapping.is_first_pp_rank():
        # Embedding
        v = awq_weight['transformer.embedding.word_embeddings.weight']
        trtllm_model.embedding.weight.value = v.to(torch_dtype).cpu().numpy()
        feed_weight_count += 1

    if mapping.is_last_pp_rank():
        # Final normalization
        v = awq_weight['transformer.encoder.final_layernorm.weight']
        trtllm_model.final_norm.weight.value = v.to(torch_dtype).cpu().numpy()
        feed_weight_count += 1

        # Final LM
        op = trtllm_model.lm_head
        prefix = "transformer.output_layer"
        process_and_assign_weight(op, prefix, 1)
        feed_weight_count += 1

    # Weight per layer
    for layer_idx in layers_range:
        i = layer_idx - mapping.pp_rank * layers_per_pipeline_stage
        prefix = "transformer.encoder.layers." + str(layer_idx) + "."
        layer = trtllm_model.layers[i]

        # Pre normalization
        name = prefix + "input_layernorm.weight"
        layer.pre_norm.weight.value = awq_weight[name].detach().cpu().numpy()
        feed_weight_count += 1

        # QKV multiplication weight
        op = layer.attention.qkv
        name = prefix + "self_attention.query_key_value"
        process_and_assign_weight(op, name, 1)
        feed_weight_count += 1

        # QKV multiplication bias
        name = prefix + "self_attention.query_key_value.bias"
        layer.attention.qkv.bias.value = awq_weight[name].detach().cpu().numpy()
        feed_weight_count += 1

        # Dense multiplication
        op = layer.attention.dense
        name = prefix + "self_attention.dense"
        process_and_assign_weight(op, name, 0)
        feed_weight_count += 1

        # Post normalization
        name = prefix + "post_attention_layernorm.weight"
        layer.post_norm.weight.value = awq_weight[name].detach().cpu().numpy()
        feed_weight_count += 1

        # Multilayer perceptron h -> 4h
        op = layer.mlp.fc
        name = prefix + "mlp.dense_h_to_4h"
        # swap first and second half weight in columns to adapt trt_llm Swiglu
        v = awq_weight[name + ".weight"]
        v = torch.split(v, v.shape[0] // 2, 0)
        v = torch.concat(v[::-1], 0)
        awq_weight[name + ".weight"] = v
        process_and_assign_weight(op, name, 1)
        feed_weight_count += 1

        # Multilayer perceptron 4h -> h
        op = layer.mlp.proj
        name = prefix + "mlp.dense_4h_to_h"
        process_and_assign_weight(op, name, 0)
        feed_weight_count += 1

    tok = time.time()

    # Final check
    if model_name in ["chatglm_6b"]:
        weight_count = 4 + num_layers * 9
    elif model_name in [
            "chatglm2_6b",
            "chatglm2_6b_32k",
            "chatglm3_6b",
            "chatglm3_6b_base",
            "chatglm3_6b_32k",
    ]:
        weight_count = 3 + num_layers * 7
    elif model_name in ["glm_10b"]:
        weight_count = 6 + num_layers * 12
    if feed_weight_count < weight_count:
        logger.error("%d weights not loaded from HF" %
                     (weight_count - feed_weight_count))
        return None
    logger.info("Loading weights finish in %.2fs" % (tok - tik))
    return


def load_from_sq(
    trtllm_model,
    dir_path,
    mapping: Mapping = Mapping(),
    dtype: str = "float32",
    model_name: str = None,
):

    assert model_name is not None, "Model name must be set"
    assert model_name in [
            "chatglm2_6b_32k",
            "chatglm3_6b",
            "chatglm3_6b_base",
            "chatglm3_6b_32k",
    ], \
        f"INT4-AWQ is not supported in {model_name} yet."

    if not dir_path.exists():
        logger.info(f"No weight file found from {dir_path}")
        return
    else:
        logger.info("Loading weights from FT SmoothQuant checkpoint")

    tik = time.time()

    n_embd = 4096
    n_layer = 28
    vocab_size = 65024
    inter_size = 13696
    np_dtype = str_dtype_to_np(dtype)
    quant_mode = getattr(trtllm_model, 'quant_mode', QuantMode(0))
    tensor_parallel = mapping.tp_size
    rank = mapping.tp_rank

    def fromfile(dir_path, name, shape=None, dtype=None):
        dtype = np_dtype if dtype is None else dtype
        p = dir_path + '/' + name
        if Path(p).exists():
            t = np.fromfile(p, dtype=dtype)
            if shape is not None:
                t = t.reshape(shape)
            return t
        return None

    def squeezeQKV(t):
        q = t[..., :4096]
        k = t[..., 4096:4096 + 256]
        v = t[..., 4096 * 2:4096 * 2 + 256]
        if isinstance(t, torch.Tensor):
            return torch.concatenate([q, k, v], axis=-1).contiguous()
        elif isinstance(t, np.ndarray):
            return np.ascontiguousarray(np.concatenate([q, k, v], axis=-1))
        else:
            print(f"Type {type(t)} is not supported")
            return None

    def set_smoothquant_scale_factors(
        module,
        pre_scale_weight,
        dir_path,
        basename,
        shape,
        per_tok_dyn,
        per_channel,
        is_qkv=False,
        rank=None,
    ):

        shape if (per_channel or is_qkv) else [1, 1]
        if per_tok_dyn:
            if pre_scale_weight is not None:
                pre_scale_weight.value = np.array([1.0], dtype=np.float32)
            t = fromfile(
                dir_path,
                f"{basename}scale_w_quant_orig.{suffix}",
                None,
                np.float32,
            )
            if "attention.query_key_value" in basename:
                t = squeezeQKV(t.reshape(1, 3 * 4096))
            elif "attention.dense" in basename:
                t = t.reshape(1, 4096)
            elif "h_to_4h" in basename:
                t = t.reshape(1, 27392)
                #v_left = t[:, :13696]
                #v_right = t[:, 13696:]
                #t = np.concatenate([v_left, v_right], axis=-1)
            elif "4h_to_h" in basename:
                t = t.reshape(1, 4096)
            module.per_channel_scale.value = t
        else:
            t = fromfile(
                dir_path,
                f"{basename}scale_x_orig_quant.bin",
                [1],
                np.float32,
            )
            pre_scale_weight.value = t
            t = fromfile(
                dir_path,
                f"{basename}scale_y_accum_quant.{suffix}",
                [1, 4096 * 3],
                np.float32,
            )
            module.per_channel_scale.value = squeezeQKV(t)
            t = fromfile(
                dir_path,
                f"{basename}scale_y_quant_orig.bin",
                [1, 1],
                np.float32,
            )
            module.act_scale.value = t

    # Determine the quantization mode.
    quant_mode = getattr(trtllm_model, "quant_mode", QuantMode(0))
    # Do we use SmoothQuant?
    use_smooth_quant = quant_mode.has_act_and_weight_quant()
    # Do we use quantization per token?
    quant_per_token_dyn = quant_mode.has_per_token_dynamic_scaling()
    # Do we use quantization per channel?
    quant_per_channel = quant_mode.has_per_channel_scaling()
    # Int8 KV cache
    use_int8_kv_cache = quant_mode.has_int8_kv_cache()
    #Enable FP8 Gemm
    quant_mode.has_fp8_qdq()

    # Debug
    suffix = f"{rank}.bin"
    if use_smooth_quant:
        sq_prefix = "int8."
        if quant_per_channel:
            sq_prefix += "col."
        suffix = sq_prefix + suffix

    # The type of weights.
    w_type = np_dtype if not use_smooth_quant else np.int8

    # Embedding
    v = fromfile(dir_path, 'model.wte.bin', [vocab_size, n_embd])
    trtllm_model.embedding.weight.value = np.ascontiguousarray(v)

    # Final normalization
    v = fromfile(dir_path, 'model.final_layernorm.weight.bin')
    trtllm_model.final_norm.weight.value = np.ascontiguousarray(v)

    # Final LM
    v = fromfile(dir_path, 'model.lm_head.weight.bin', [vocab_size, n_embd])
    dst = trtllm_model.lm_head.weight
    if vocab_size % tensor_parallel != 0:
        pad_width = trtllm_model.lm_head.out_features * tensor_parallel - vocab_size
        v = np.pad(v, ((0, pad_width), (0, 0)), 'constant', constant_values=0)
    v = torch.chunk(v, mapping.tp_size, dim=0)[mapping.rank]
    dst.value = np.ascontiguousarray(v)

    # Weight per layer
    for i in range(n_layer):
        c_attn_out_dim = 4608
        layer = trtllm_model.layers[i]

        # Pre normalization
        v = fromfile(dir_path, f'model.layers.{i}.input_layernorm.weight.bin')
        layer.pre_norm.weight.value = v

        # QKV multiplication weight
        t = fromfile(
            dir_path,
            f'model.layers.{i}.attention.query_key_value.weight.' + suffix,
            [n_embd * 3 * n_embd],
            w_type,
        )
        # [4096,4096*3]->[4096,4608]->[4608,4096]
        t = squeezeQKV(t.transpose(1, 0))
        layer.attention.qkv.weight.value = np.ascontiguousarray(t)
        set_smoothquant_scale_factors(
            layer.attention.qkv,
            layer.pre_norm.scale_to_int,
            dir_path,
            f'model.layers.{i}.attention.query_key_value.',
            [1, c_attn_out_dim],
            quant_per_token_dyn,
            quant_per_channel,
            rank=rank,
            is_qkv=True,
        )

        # QKV multiplication bias
        if layer.bias:
            t = fromfile(
                dir_path,
                f'model.layers.{i}.attention.query_key_value.bias.{rank}.bin',
            )
            layer.attention.qkv.bias.value = np.ascontiguousarray(t)

        # Dense multiplication weight
        t = fromfile(
            dir_path,
            f'model.layers.{i}.attention.dense.weight.' + suffix,
            [n_embd // tensor_parallel, n_embd],
            w_type,
        )
        t = t.transpose(1, 0)
        layer.attention.dense.weight.value = np.ascontiguousarray(t)
        set_smoothquant_scale_factors(
            layer.attention.dense,
            getattr(layer.attention, "quantization_scaling_factor", None),
            dir_path,
            'model.layers.' + str(i) + '.attention.dense.',
            [1, n_embd],
            quant_per_token_dyn,
            quant_per_channel,
        )
        # change it to the real smoother if dense layer is applied smooth quant
        layer.attention.dense.smoother.value = np.ones(
            [1, n_embd // tensor_parallel],
            dtype=np.float32,
        )

        # Dense multiplication bias, only GLM-10B
        if layer.dense_bias:
            t = fromfile(
                dir_path,
                f'model.layers.{i}.attention.dense.bias.bin',
            )
            layer.attention.dense.bias = np.ascontiguousarray(t)

        # Post normalization
        t = fromfile(
            dir_path,
            f'model.layers.{i}.post_attention_layernorm.weight.bin',
        )
        layer.post_norm.weight.value = np.ascontiguousarray(t)

        # Multilayer perceptron h -> 4h weight
        t = fromfile(
            dir_path,
            f'model.layers.{i}.mlp.dense_h_to_4h.weight.' + suffix,
            [n_embd, inter_size * 2 // tensor_parallel],
            w_type,
        )
        t = t.transpose(1, 0)
        #v_left = t[:, :inter_size * 2 // tensor_parallel]
        #v_right = t[:, inter_size * 2 // tensor_parallel:]
        #t = np.concatenate([v_right, v_left], axis=-1)
        layer.mlp.fc.weight.value = np.ascontiguousarray(t)
        set_smoothquant_scale_factors(
            layer.mlp.fc,
            layer.post_norm.scale_to_int,
            dir_path,
            f'model.layers.{i}.mlp.dense_h_to_4h.',
            [1, inter_size // tensor_parallel],
            quant_per_token_dyn,
            quant_per_channel,
            rank=rank,
        )

        # Multilayer perceptron 4h -> h weight
        t = fromfile(
            dir_path,
            f'model.layers.{i}.mlp.dense_4h_to_h.weight.' + suffix,
            [inter_size // tensor_parallel, n_embd],
            w_type,
        )
        t = t.transpose(1, 0)
        layer.mlp.proj.weight.value = np.ascontiguousarray(t)
        set_smoothquant_scale_factors(
            layer.mlp.proj,
            getattr(layer.mlp, "quantization_scaling_factor", None),
            dir_path,
            f'model.layers.{i}.mlp.dense_4h_to_h.',
            [1, n_embd],
            quant_per_token_dyn,
            quant_per_channel,
        )
        # change it to the real smoother if proj layer is applied smooth quant
        layer.mlp.proj.smoother.value = np.ones(
            [1, inter_size // tensor_parallel],
            dtype=np.float32,
        )

        if use_int8_kv_cache:
            t = fromfile(
                dir_path,
                f'model.layers.{i}.attention.query_key_value.scale_y_quant_orig.bin',
                [1],
                np.float32,
            )
            layer.attention.kv_orig_quant_scale.value = 1.0 / t
            layer.attention.kv_quant_orig_scale.value = t

    tok = time.time()
    logger.info("Loading weights finish in %.2fs" % (tok - tik))
    return


#===============================================================================
def load_from_gptq(
        trtllm_model,
        quant_ckpt_path,
        mapping=Mapping(),
        dtype="float16",
):

    print("Not supported yet.")

    return None


def get_scaling_factors(
    model_path: Union[str, Path],
    num_layers: int,
    quant_mode: Optional[QuantMode] = None,
) -> Optional[Dict[str, List[int]]]:
    """ Get the scaling factors for model

    Returns a dictionary of scaling factors for the selected layers of the model.

    Args:
        model_path (str): Path to the quantized model
        layers (list): List of layers to get the scaling factors for. If None,
            all layers are selected.

    Returns:
        dict: Dictionary of scaling factors for the selected layers of the model.

        example:

        {
            'qkv_act': qkv_act_scale,
            'qkv_weights': qkv_weights_scale,
            'qkv_output' : qkv_outputs_scale,
            'dense_act': dense_act_scale,
            'dense_weights': dense_weights_scale,
            'fc_act': fc_act_scale,
            'fc_weights': fc_weights_scale,
            'gate_act': gate_act_scale,
            'gate_weights': gate_weights_scale,
            'proj_act': proj_act_scale,
            'proj_weights': proj_weights_scale,
        }
    """

    if model_path is None:
        logger.warning(f"--quantized_fp8_model_path not specified. "
                       f"Initialize quantization scales automatically.")
        return get_dummy_quant_scales(num_layers)
    weight_dict = np.load(model_path)

    # yapf: disable
    scaling_factor = {
        'qkv_act': [],
        'qkv_weights': [],
        'qkv_output': [],
        'dense_act': [],
        'dense_weights': [],
        'fc_act': [],
        'fc_weights': [],
        'gate_act': [],
        'gate_weights': [],
        'proj_act': [],
        'proj_weights': [],
    }

    for layer in range(num_layers):
        scaling_factor['qkv_act'].append(max(
            weight_dict[f'_np:layers:{layer}:attention:qkv:q:activation_scaling_factor'].item(),
            weight_dict[f'_np:layers:{layer}:attention:qkv:k:activation_scaling_factor'].item(),
            weight_dict[f'_np:layers:{layer}:attention:qkv:v:activation_scaling_factor'].item()
            ))
        scaling_factor['qkv_weights'].append(max(
            weight_dict[f'_np:layers:{layer}:attention:qkv:q:weights_scaling_factor'].item(),
            weight_dict[f'_np:layers:{layer}:attention:qkv:k:weights_scaling_factor'].item(),
            weight_dict[f'_np:layers:{layer}:attention:qkv:v:weights_scaling_factor'].item()
            ))
        if quant_mode is not None and quant_mode.has_fp8_kv_cache():
            # Not calibrarting KV cache.
            scaling_factor['qkv_output'].append(1.0)
        scaling_factor['dense_act'].append(weight_dict[f'_np:layers:{layer}:attention:dense:activation_scaling_factor'].item())
        scaling_factor['dense_weights'].append(weight_dict[f'_np:layers:{layer}:attention:dense:weights_scaling_factor'].item())
        scaling_factor['fc_act'].append(weight_dict[f'_np:layers:{layer}:mlp:fc:activation_scaling_factor'].item())
        scaling_factor['fc_weights'].append(weight_dict[f'_np:layers:{layer}:mlp:fc:weights_scaling_factor'].item())
        scaling_factor['gate_act'].append(weight_dict[f'_np:layers:{layer}:mlp:gate:activation_scaling_factor'].item())
        scaling_factor['gate_weights'].append(weight_dict[f'_np:layers:{layer}:mlp:gate:weights_scaling_factor'].item())
        scaling_factor['proj_act'].append(weight_dict[f'_np:layers:{layer}:mlp:proj:activation_scaling_factor'].item())
        scaling_factor['proj_weights'].append(weight_dict[f'_np:layers:{layer}:mlp:proj:weights_scaling_factor'].item())
    # yapf: enable
    for k, v in scaling_factor.items():
        assert len(v) == num_layers, \
        f'Expect scaling factor {k} of length {num_layers}, got {len(v)}'

    return scaling_factor
