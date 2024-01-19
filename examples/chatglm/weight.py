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
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import transformers

import tensorrt_llm.logger as logger
from tensorrt_llm._utils import (str_dtype_to_np, str_dtype_to_torch,
                                 torch_to_numpy)
from tensorrt_llm.mapping import Mapping
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

    hf_model = transformers.AutoModel.from_pretrained(
        hf_model_dir,
        trust_remote_code=True,
    )
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

    if model_name in ["chatglm_6b", "glm_2b", "glm_10b", "glm_10b_chinese"]:
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
        dst = trtllm_model.vocab_embedding.weight
        if model_name in ["chatglm_6b"]:
            v = hf_model.transformer.word_embeddings.weight
            dst.value = torch_to_numpy(v.to(torch_type).detach())
            feed_weight_count += 1
        elif model_name in [
                "chatglm2_6b",
                "chatglm2_6b_32k",
                "chatglm3_6b",
                "chatglm3_6b_base",
                "chatglm3_6b_32k",
        ]:
            v = hf_model.transformer.embedding.word_embeddings.weight
            dst.value = torch_to_numpy(v.to(torch_type).detach())
            feed_weight_count += 1
        elif model_name in ["glm_2b", "glm_10b", "glm_10b_chinese"]:
            v = hf_model.word_embeddings.weight
            dst.value = torch_to_numpy(v.to(torch_type).detach())
            v = hf_model.transformer.position_embeddings.weight
            v = v.to(torch_type).detach()
            trtllm_model.position_embedding.weight.value = torch_to_numpy(v)
            v = hf_model.transformer.block_position_embeddings.weight
            v = v.to(torch_type).detach()
            trtllm_model.block_embedding.weight.value = torch_to_numpy(v)
            feed_weight_count += 3

    if mapping.is_last_pp_rank():
        # Final normalization
        dst = trtllm_model.final_norm.weight
        if model_name in ["chatglm_6b", "glm_2b", "glm_10b", "glm_10b_chinese"]:
            v = hf_model.transformer.final_layernorm.weight
            dst.value = torch_to_numpy(v.to(torch_type).detach())
            v = hf_model.transformer.final_layernorm.bias
            v = v.to(torch_type).detach()
            trtllm_model.final_norm.bias.value = torch_to_numpy(v)
            feed_weight_count += 2
        elif model_name in [
                "chatglm2_6b",
                "chatglm2_6b_32k",
                "chatglm3_6b",
                "chatglm3_6b_base",
                "chatglm3_6b_32k",
        ]:
            v = hf_model.transformer.encoder.final_layernorm.weight
            dst.value = torch_to_numpy(v.to(torch_type).detach())
            feed_weight_count += 1

        # Final LM
        output_features = trtllm_model.lm_head.out_features
        if model_name in ["chatglm_6b"]:
            v = hf_model.lm_head.weight
        elif model_name in [
                "chatglm2_6b",
                "chatglm2_6b_32k",
                "chatglm3_6b",
                "chatglm3_6b_base",
                "chatglm3_6b_32k",
        ]:
            v = hf_model.transformer.output_layer.weight
        elif model_name in ["glm_2b", "glm_10b", "glm_10b_chinese"]:
            v = hf_model.word_embeddings.weight

        v = v.to(torch_type).detach()
        if v.shape[0] % mapping.tp_size != 0:
            pad_width = output_features * mapping.tp_size - v.shape[0]
            v = F.pad(v, (0, 0, 0, pad_width))
        v = torch.chunk(v, mapping.tp_size, dim=0)[mapping.tp_rank]
        trtllm_model.lm_head.weight.value = torch_to_numpy(v)
        feed_weight_count += 1

    # Weight per layer
    for layer_idx in layers_range:
        i = layer_idx - mapping.pp_rank * layers_per_pipeline_stage
        layer = trtllm_model.layers[i]
        if model_name in ["chatglm_6b", "glm_2b", "glm_10b", "glm_10b_chinese"]:
            hf_layer = hf_model.transformer.layers[i]
        elif model_name in [
                "chatglm2_6b",
                "chatglm2_6b_32k",
                "chatglm3_6b",
                "chatglm3_6b_base",
                "chatglm3_6b_32k",
        ]:
            hf_layer = hf_model.transformer.encoder.layers[i]

        # Pre normalization
        if model_name in ["chatglm_6b", "glm_2b", "glm_10b", "glm_10b_chinese"]:
            v = hf_layer.input_layernorm.weight.to(torch_type).detach()
            layer.pre_norm.weight.value = torch_to_numpy(v)
            v = hf_layer.input_layernorm.bias.to(torch_type).detach()
            layer.pre_norm.bias.value = torch_to_numpy(v)
            feed_weight_count += 2
        elif model_name in [
                "chatglm2_6b",
                "chatglm2_6b_32k",
                "chatglm3_6b",
                "chatglm3_6b_base",
                "chatglm3_6b_32k",
        ]:
            v = hf_layer.input_layernorm.weight.to(torch_type).detach()
            layer.pre_norm.weight.value = torch_to_numpy(v)
            feed_weight_count += 1

        # QKV multiplication weight
        if model_name in ["chatglm_6b", "glm_2b", "glm_10b", "glm_10b_chinese"]:
            v = hf_layer.attention.query_key_value.weight
        elif model_name in [
                "chatglm2_6b",
                "chatglm2_6b_32k",
                "chatglm3_6b",
                "chatglm3_6b_base",
                "chatglm3_6b_32k",
        ]:
            v = hf_layer.self_attention.query_key_value.weight

        v = v.to(torch_type).detach()
        v = split_qkv(v, mapping.tp_size, mapping.tp_rank, hidden_size,
                      num_heads, num_kv_heads)
        dst = layer.attention
        if use_weight_only:
            load_quant_weight(
                src=v,
                value_dst=dst.qkv.weight,
                scale_dst=dst.qkv.per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
        else:
            dst.qkv.weight.value = torch_to_numpy(v)
        feed_weight_count += 1

        # QKV multiplication bias
        if model_name in ["chatglm_6b", "glm_2b", "glm_10b", "glm_10b_chinese"]:
            v = hf_layer.attention.query_key_value.bias
        elif model_name in [
                "chatglm2_6b",
                "chatglm2_6b_32k",
                "chatglm3_6b",
                "chatglm3_6b_base",
                "chatglm3_6b_32k",
        ]:
            v = hf_layer.self_attention.query_key_value.bias

        v = v.to(torch_type).detach()
        v = split_qkv(v, mapping.tp_size, mapping.tp_rank, hidden_size,
                      num_heads, num_kv_heads)
        layer.attention.qkv.bias.value = torch_to_numpy(v)
        feed_weight_count += 1

        # Dense multiplication weight
        if model_name in ["chatglm_6b", "glm_2b", "glm_10b", "glm_10b_chinese"]:
            v = hf_layer.attention.dense.weight
        elif model_name in [
                "chatglm2_6b",
                "chatglm2_6b_32k",
                "chatglm3_6b",
                "chatglm3_6b_base",
                "chatglm3_6b_32k",
        ]:
            v = hf_layer.self_attention.dense.weight

        v = v.to(torch_type).detach()
        v = torch.chunk(v, mapping.tp_size, dim=1)[mapping.tp_rank]
        dst = layer.attention.dense
        if use_weight_only:
            load_quant_weight(
                src=v,
                value_dst=dst.weight,
                scale_dst=dst.per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
        else:
            # need np.ascontiguousarray since weight is split by column
            dst.weight.value = np.ascontiguousarray(torch_to_numpy(v))
        feed_weight_count += 1

        # Dense multiplication bias, only GLM-10B
        if model_name in ["glm_2b", "glm_10b", "glm_10b_chinese"]:
            v = hf_layer.attention.dense.bias.to(torch_type).detach()
            v = split_qkv(v, mapping.tp_size, mapping.tp_rank, hidden_size,
                          num_heads, num_kv_heads)
            layer.attention.dense.bias.value = torch_to_numpy(v)
            feed_weight_count += 1

        # Post normalization
        if model_name in ["chatglm_6b", "glm_2b", "glm_10b", "glm_10b_chinese"]:
            v = hf_layer.post_attention_layernorm.weight
            v = v.to(torch_type).detach()
            layer.post_norm.weight.value = torch_to_numpy(v)
            v = hf_layer.post_attention_layernorm.bias
            v = v.to(torch_type).detach()
            layer.post_norm.bias.value = torch_to_numpy(v)
            feed_weight_count += 2
        elif model_name in [
                "chatglm2_6b",
                "chatglm2_6b_32k",
                "chatglm3_6b",
                "chatglm3_6b_base",
                "chatglm3_6b_32k",
        ]:
            v = hf_layer.post_attention_layernorm.weight
            v = v.to(torch_type).detach()
            layer.post_norm.weight.value = torch_to_numpy(v)
            feed_weight_count += 1

        # Multilayer perceptron h -> 4h weight
        if model_name in ["chatglm_6b", "glm_2b", "glm_10b", "glm_10b_chinese"]:
            v = hf_layer.mlp.dense_h_to_4h.weight.to(torch_type).detach()
            v = torch.chunk(v, mapping.tp_size, dim=0)[mapping.tp_rank]
        elif model_name in [
                "chatglm2_6b",
                "chatglm2_6b_32k",
                "chatglm3_6b",
                "chatglm3_6b_base",
                "chatglm3_6b_32k",
        ]:
            v = hf_layer.mlp.dense_h_to_4h.weight.to(torch_type).detach()
            v = torch.chunk(v, 2 * mapping.tp_size, dim=0)
            # swap halves of weight in column to adapt TRT-LLM's Swiglu
            v = torch.cat(
                [
                    v[mapping.tp_rank + mapping.tp_size],
                    v[mapping.tp_rank],
                ],
                dim=0,
            )

        dst = layer.mlp.fc
        if use_weight_only:
            load_quant_weight(
                src=v,
                value_dst=dst.weight,
                scale_dst=dst.per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
        else:
            dst.weight.value = torch_to_numpy(v)
        feed_weight_count += 1

        # Multilayer perceptron h -> 4h bias, only GLM-10B
        if model_name in ["glm_2b", "glm_10b", "glm_10b_chinese"]:
            bias = hf_layer.mlp.dense_h_to_4h.bias.to(torch_type).detach()
            split_bias = split_qkv(bias, mapping.tp_size, mapping.tp_rank,
                                   hidden_size, num_heads, num_kv_heads)
            layer.mlp.fc.bias.value = torch_to_numpy(split_bias)
            feed_weight_count += 1

        # Multilayer perceptron 4h -> h weight
        if model_name in ["chatglm_6b", "glm_2b", "glm_10b", "glm_10b_chinese"]:
            v = hf_layer.mlp.dense_4h_to_h.weight.to(torch_type).detach()
        elif model_name in [
                "chatglm2_6b",
                "chatglm2_6b_32k",
                "chatglm3_6b",
                "chatglm3_6b_base",
                "chatglm3_6b_32k",
        ]:
            v = hf_layer.mlp.dense_4h_to_h.weight.to(torch_type).detach()

        v = torch.chunk(v, mapping.tp_size, dim=1)[mapping.tp_rank]
        dst = layer.mlp.proj
        if use_weight_only:
            load_quant_weight(
                src=v,
                value_dst=dst.weight,
                scale_dst=dst.per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
        else:
            # need np.ascontiguousarray since weight is split by column
            dst.weight.value = np.ascontiguousarray(torch_to_numpy(v))
        feed_weight_count += 1

        # Multilayer perceptron 4h -> h bias, only GLM-10B
        if model_name in ["glm_2b", "glm_10b", "glm_10b_chinese"]:
            bias = hf_layer.mlp.dense_4h_to_h.bias.to(torch_type).detach()
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
    elif model_name in ["glm_2b", "glm_10b", "glm_10b_chinese"]:
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

    if not (str(quant_ckpt_path).endswith(".pt") and quant_ckpt_path.exists()):
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
        trtllm_model.vocab_embedding.weight.value = v.to(
            torch_dtype).cpu().numpy()
        feed_weight_count += 1

    if mapping.is_last_pp_rank():
        # Final normalization
        v = awq_weight['transformer.encoder.final_layernorm.weight']
        trtllm_model.final_norm.weight.value = v.to(torch_dtype).cpu().numpy()
        feed_weight_count += 1

        # Final LM
        prefix = "transformer.output_layer"
        process_and_assign_weight(trtllm_model.lm_head, prefix, 1)
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
        name = prefix + "self_attention.query_key_value"
        process_and_assign_weight(layer.attention.qkv, name, 1)
        feed_weight_count += 1

        # QKV multiplication bias
        name = prefix + "self_attention.query_key_value.bias"
        layer.attention.qkv.bias.value = awq_weight[name].detach().cpu().numpy()
        feed_weight_count += 1

        # Dense multiplication
        name = prefix + "self_attention.dense"
        process_and_assign_weight(layer.attention.dense, name, 0)
        feed_weight_count += 1

        # Post normalization
        name = prefix + "post_attention_layernorm.weight"
        layer.post_norm.weight.value = awq_weight[name].detach().cpu().numpy()
        feed_weight_count += 1

        # Multilayer perceptron h -> 4h
        name = prefix + "mlp.dense_h_to_4h"
        # swap halves of weight in column to adapt TRT-LLM's Swiglu
        v = awq_weight[name + ".weight"]
        v = torch.split(v, v.shape[0] // 2, 0)
        v = torch.concat(v[::-1], 0)
        awq_weight[name + ".weight"] = v
        process_and_assign_weight(layer.mlp.fc, name, 1)
        feed_weight_count += 1

        # Multilayer perceptron 4h -> h
        name = prefix + "mlp.dense_4h_to_h"
        process_and_assign_weight(layer.mlp.proj, name, 0)
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
    elif model_name in ["glm_2b", "glm_10b", "glm_10b_chinese"]:
        weight_count = 6 + num_layers * 12
    if feed_weight_count < weight_count:
        logger.error("%d weights not loaded from HF" %
                     (weight_count - feed_weight_count))
        return None
    logger.info("Loading weights finish in %.2fs" % (tok - tik))
    return


def load_from_sq(
    trtllm_model,
    model_dir: Path = None,
    mapping: Mapping = Mapping(),
    dtype: str = "float16",
    model_name: str = None,
):

    assert model_name is not None, "Model name must be set"
    assert model_name in [
            "chatglm2_6b",
            "chatglm2_6b_32k",
            "chatglm3_6b",
            "chatglm3_6b_base",
            "chatglm3_6b_32k",
    ], \
        f"Smooth Quantization is not supported in {model_name} yet."

    if not model_dir.exists():
        logger.info(f"No weight file found from {model_dir}")
        return
    else:
        logger.info(f"Loading weights from {model_dir}")

    tik = time.time()

    quant_mode = getattr(trtllm_model, "quant_mode", QuantMode(0))
    use_smooth_quant = quant_mode.has_act_and_weight_quant()
    quant_per_token_dyn = quant_mode.has_per_token_dynamic_scaling()
    quant_per_channel = quant_mode.has_per_channel_scaling()
    use_int8_kv_cache = quant_mode.has_int8_kv_cache()
    suffix = f"{mapping.tp_rank}.bin"
    if use_smooth_quant:
        sq_prefix = "int8."
        if quant_per_channel:
            sq_prefix += "col."
        suffix = sq_prefix + suffix

    with open(model_dir / "config.json", "r") as f:
        js = json.loads(f.read())
    ffn_hidden_size = js["ffn_hidden_size"]
    hidden_size = js["hidden_size"]
    n_groups = js["multi_query_group_num"]
    num_attention_heads = js["num_attention_heads"]
    num_layers = js["num_layers"]
    vocab_size = js["padded_vocab_size"]
    qkv_size = (hidden_size + hidden_size // num_attention_heads * n_groups * 2)
    qkv_size = qkv_size // mapping.tp_size
    np_dtype = str_dtype_to_np(dtype)
    w_type = np.int8 if use_smooth_quant else np_dtype  # The type of weights

    layers_per_pipeline_stage = num_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage))

    def fromfile(
        name,
        shape=None,
        dtype=None,
    ):
        dtype = np_dtype if dtype is None else dtype
        p = model_dir / name
        if Path(p).exists():
            v = np.fromfile(p, dtype=dtype)
            if shape is not None:
                v = v.reshape(shape)
            return v
        return None

    def set_smoothquant_scale_factors(
        module,
        pre_scale_weight,
        basename,
        shape,
        per_tok_dyn,
        per_channel,
        is_qkv=False,
        rank=None,
    ):
        suffix = "bin"
        if per_channel:
            if rank is not None:
                suffix = f"{rank}." + suffix
            suffix = "col." + suffix

        col_shape = shape if (per_channel or is_qkv) else [1, 1]

        if per_tok_dyn:
            if pre_scale_weight is not None:
                pre_scale_weight.value = np.array([1.0], dtype=np.float32)
            if is_qkv and not per_channel:
                name = f"{basename}scale_w_quant_orig.{rank}.{suffix}"
            else:
                name = f"{basename}scale_w_quant_orig.{suffix}"
            v = fromfile(name, col_shape, np.float32)
            module.per_channel_scale.value = np.ascontiguousarray(v)
        else:
            name = f"{basename}scale_x_orig_quant.bin",
            v = fromfile(name, [1], np.float32)
            pre_scale_weight.value = np.ascontiguousarray(v)
            if is_qkv:
                name = f"{basename}scale_y_accum_quant.{rank}.{suffix}"
            else:
                name = f"{basename}scale_y_accum_quant.{suffix}"
            v = fromfile(name, col_shape, np.float32)
            module.per_channel_scale.value = np.ascontiguousarray(v)
            name = f"{basename}scale_y_quant_orig.bin"
            v = fromfile(name, [1, 1], np.float32)
            module.act_scale.value = np.ascontiguousarray(v)

    def set_smoother(
        module,
        basename,
        shape,
        rank,
    ):
        v = fromfile(f"{basename}.smoother.{rank}.bin", shape, np.float32)
        module.smoother.value = np.ascontiguousarray(v)

    if mapping.is_first_pp_rank():
        # Embedding
        v = fromfile('embedding.weight.bin', [vocab_size, hidden_size])
        trtllm_model.vocab_embedding.weight.value = np.ascontiguousarray(v)

    if mapping.is_last_pp_rank():
        # Final normalization
        v = fromfile('final_norm.weight.bin', [hidden_size])
        trtllm_model.final_norm.weight.value = np.ascontiguousarray(v)

        # Final LM
        v = fromfile('lm_head.weight.bin', [vocab_size, hidden_size])
        if vocab_size % mapping.tp_size != 0:
            pad_width = trtllm_model.lm_head.out_features * mapping.tp_size - vocab_size
            v = np.pad(v, [[0, pad_width], [0, 0]])
        v = np.split(v, mapping.tp_size, axis=0)[mapping.tp_rank]
        trtllm_model.lm_head.weight.value = np.ascontiguousarray(v)

    # Weight per layer
    for layer_idx in layers_range:
        i = layer_idx - mapping.pp_rank * layers_per_pipeline_stage
        layer = trtllm_model.layers[i]

        # Pre normalization
        v = fromfile(f'layers.{i}.pre_norm.weight.bin', [hidden_size])
        layer.pre_norm.weight.value = np.ascontiguousarray(v)

        # QKV multiplication weight
        basename = f'layers.{i}.attention.query_key_value.'
        name = basename + 'weight.' + suffix
        v = fromfile(name, [hidden_size, qkv_size], w_type)
        dst = layer.attention.qkv.weight
        dst.value = np.ascontiguousarray(v.transpose(1, 0))
        set_smoothquant_scale_factors(
            module=layer.attention.qkv,
            pre_scale_weight=layer.pre_norm.scale_to_int,
            basename=basename,
            shape=[1, qkv_size],
            per_tok_dyn=quant_per_token_dyn,
            per_channel=quant_per_channel,
            is_qkv=True,
            rank=mapping.tp_rank,
        )

        # QKV multiplication bias
        if layer.bias:
            name = f'layers.{i}.attention.query_key_value.bias.{mapping.tp_rank}.bin'
            v = fromfile(name, [qkv_size])
            layer.attention.qkv.bias.value = np.ascontiguousarray(v)

        # Dense multiplication weight
        basename = f'layers.{i}.attention.dense.'
        name = basename + 'weight.' + suffix
        shape = [hidden_size // mapping.tp_size, hidden_size]
        v = fromfile(name, shape, w_type)
        dst = layer.attention.dense.weight
        dst.value = np.ascontiguousarray(v.transpose(1, 0))
        dense_scale = layer.attention.quantization_scaling_factor
        set_smoothquant_scale_factors(
            module=layer.attention.dense,
            pre_scale_weight=dense_scale,
            basename=basename,
            shape=[1, hidden_size],
            per_tok_dyn=quant_per_token_dyn,
            per_channel=quant_per_channel,
            is_qkv=False,
            rank=None,
        )
        set_smoother(
            module=layer.attention.dense,
            basename=basename[:-1],
            shape=[1, hidden_size // mapping.tp_size],
            rank=mapping.tp_rank,
        )

        # Dense multiplication bias, only GLM-10B
        if layer.dense_bias:
            v = fromfile(f'layers.{i}.attention.dense.bias.bin', [hidden_size])
            layer.attention.dense.bias = np.ascontiguousarray(v)

        # Post normalization
        v = fromfile(f'layers.{i}.post_norm.weight.bin', [hidden_size])
        layer.post_norm.weight.value = np.ascontiguousarray(v)

        # Multilayer perceptron h -> 4h weight
        basename = f'layers.{i}.mlp.fc.'
        name = basename + 'weight.' + suffix
        shape = [hidden_size, ffn_hidden_size * 2 // mapping.tp_size]
        v = fromfile(name, shape, w_type)
        # swap halves of weight in column to adapt TRT-LLM's Swiglu
        v_left = v[:, :ffn_hidden_size // mapping.tp_size]
        v_right = v[:, ffn_hidden_size // mapping.tp_size:]
        v = np.concatenate([v_right, v_left], axis=-1)
        layer.mlp.fc.weight.value = np.ascontiguousarray(v.transpose(1, 0))
        set_smoothquant_scale_factors(
            module=layer.mlp.fc,
            pre_scale_weight=layer.post_norm.scale_to_int,
            basename=basename,
            shape=[1, ffn_hidden_size * 2 // mapping.tp_size],
            per_tok_dyn=quant_per_token_dyn,
            per_channel=quant_per_channel,
            is_qkv=False,
            rank=mapping.tp_rank,
        )

        # Multilayer perceptron 4h -> h weight
        basename = f'layers.{i}.mlp.proj.'
        name = basename + 'weight.' + suffix
        shape = [ffn_hidden_size // mapping.tp_size, hidden_size]
        v = fromfile(name, shape, w_type)
        layer.mlp.proj.weight.value = np.ascontiguousarray(v.transpose(1, 0))
        proj_scale = layer.mlp.quantization_scaling_factor
        set_smoothquant_scale_factors(
            module=layer.mlp.proj,
            pre_scale_weight=proj_scale,
            basename=basename,
            shape=[1, hidden_size],
            per_tok_dyn=quant_per_token_dyn,
            per_channel=quant_per_channel,
            is_qkv=False,
            rank=None,
        )
        set_smoother(
            module=layer.mlp.proj,
            basename=basename[:-1],
            shape=[1, ffn_hidden_size // mapping.tp_size],
            rank=mapping.tp_rank,
        )

        if use_int8_kv_cache:
            name = f'layers.{i}.attention.query_key_value.scale_y_quant_orig.bin'
            v = fromfile(name, [1], np.float32)
            dst = layer.attention
            dst.kv_cache_scaling_factor.value = np.ascontiguousarray(v)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Weights loaded. Total time: {t}')


def load_from_gptq(
        trtllm_model,
        quant_ckpt_path,
        mapping=Mapping(),
        dtype="float16",
):

    print("Not supported yet.")

    return None


def get_scaling_factors(
    model_path: Path = None,
    num_layers: int = 0,
    quant_mode: QuantMode = None,
):
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
