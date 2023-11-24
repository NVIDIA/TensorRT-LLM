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
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import transformers

import tensorrt_llm
import tensorrt_llm.logger as logger
from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.quantized.quant import get_dummy_quant_scales
from tensorrt_llm.quantization import QuantMode


def split(weight: np.ndarray, tp_size: int, rank: int = 0, dim: int = 0):
    if tp_size == 1:
        return weight
    elif weight.ndim == 1:
        return np.ascontiguousarray(np.split(weight, tp_size)[rank].copy())
    return np.ascontiguousarray(
        np.split(weight, tp_size, axis=dim)[rank].copy())


def split_matrix(weight: np.ndarray, tp_size: int, rank: int, dim: int):
    return np.ascontiguousarray(split(weight, tp_size, rank, dim=dim))


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
    hf_model_dir,
    mapping=Mapping(),
    dtype="float32",
    model_name=None,
    multi_query_mode=False,
):

    assert model_name is not None, "Model name must be set"

    tensorrt_llm.logger.info("Loading weights from HF")

    if not Path(hf_model_dir).exists():
        tensorrt_llm.logger.info(
            "No weight file found from %s, use random weights" % hf_model_dir)
        return trt_model

    tik = time.time()

    hf_model = transformers.AutoModel.from_pretrained(hf_model_dir,
                                                      trust_remote_code=True)
    hidden_size = hf_model.config.hidden_size
    num_heads = hf_model.config.num_attention_heads
    num_layers = hf_model.config.num_layers

    torch_type = str_dtype_to_torch(dtype)
    quant_mode = getattr(trt_model, 'quant_mode', QuantMode(0))
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
            trt_model.embedding.weight.value = torch_to_numpy(weight)
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
            trt_model.embedding.weight.value = torch_to_numpy(weight)
            feed_weight_count += 1
        elif model_name in ["glm_10b"]:
            weight = hf_model.word_embeddings.weight.to(torch_type).detach()
            trt_model.embedding.weight.value = torch_to_numpy(weight)
            weight = hf_model.transformer.position_embeddings.weight.to(
                torch_type).detach()
            trt_model.position_embeddings.weight.value = torch_to_numpy(weight)
            weight = hf_model.transformer.block_position_embeddings.weight.to(
                torch_type).detach()
            trt_model.block_embeddings.weight.value = torch_to_numpy(weight)
            feed_weight_count += 3

    if mapping.is_last_pp_rank():
        # Final normalization
        if model_name in ["chatglm_6b"]:
            weight = hf_model.transformer.final_layernorm.weight.to(
                torch_type).detach()
            trt_model.final_norm.weight.value = torch_to_numpy(weight)
            bias = hf_model.transformer.final_layernorm.bias.to(
                torch_type).detach()
            trt_model.final_norm.bias.value = torch_to_numpy(bias)
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
            trt_model.final_norm.weight.value = torch_to_numpy(weight)
            feed_weight_count += 1
        elif model_name in ["glm_10b"]:
            weight = hf_model.transformer.final_layernorm.weight.to(
                torch_type).detach()
            trt_model.final_norm.weight.value = torch_to_numpy(weight)
            bias = hf_model.transformer.final_layernorm.bias.to(
                torch_type).detach()
            trt_model.final_norm.bias.value = torch_to_numpy(bias)
            feed_weight_count += 2

        # Final LM
        if model_name in ["chatglm_6b"]:
            weight = hf_model.lm_head.weight.to(torch_type).detach()
            if weight.shape[0] % mapping.tp_size != 0:
                pad_width = trt_model.lm_head.out_features * mapping.tp_size - weight.shape[
                    0]
                weight = F.pad(weight, (0, 0, 0, pad_width))
            split_weight = torch.chunk(weight, mapping.tp_size,
                                       dim=0)[mapping.rank]
            trt_model.lm_head.weight.value = torch_to_numpy(split_weight)
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
                pad_width = trt_model.lm_head.out_features * mapping.tp_size - weight.shape[
                    0]
                weight = F.pad(weight, (0, 0, 0, pad_width))
            split_weight = torch.chunk(weight, mapping.tp_size,
                                       dim=0)[mapping.rank]
            trt_model.lm_head.weight.value = torch_to_numpy(split_weight)
            feed_weight_count += 1
        elif model_name in ["glm_10b"]:
            weight = hf_model.word_embeddings.weight.to(torch_type).detach()
            if weight.shape[0] % mapping.tp_size != 0:
                pad_width = trt_model.lm_head.out_features * mapping.tp_size - weight.shape[
                    0]
                weight = F.pad(weight, (0, 0, 0, pad_width))
            split_weight = torch.chunk(weight, mapping.tp_size,
                                       dim=0)[mapping.rank]
            trt_model.lm_head.weight.value = torch_to_numpy(split_weight)
            feed_weight_count += 1

    # Weight per layer
    for layer_idx in range(num_layers):
        if layer_idx not in layers_range:
            continue
        i = int(layer_idx) - mapping.pp_rank * layers_per_pipeline_stage
        if i >= num_layers:
            continue

        # Pre normalization
        if model_name in ["chatglm_6b"]:
            weight = hf_model.transformer.layers[i].input_layernorm.weight.to(
                torch_type).detach()
            trt_model.layers[i].pre_norm.weight.value = torch_to_numpy(weight)
            bias = hf_model.transformer.layers[i].input_layernorm.bias.to(
                torch_type).detach()
            trt_model.layers[i].pre_norm.bias.value = torch_to_numpy(bias)
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
            trt_model.layers[i].pre_norm.weight.value = torch_to_numpy(weight)
            feed_weight_count += 1
        elif model_name in ["glm_10b"]:
            weight = hf_model.transformer.layers[i].input_layernorm.weight.to(
                torch_type).detach()
            trt_model.layers[i].pre_norm.weight.value = torch_to_numpy(weight)
            bias = hf_model.transformer.layers[i].input_layernorm.bias.to(
                torch_type).detach()
            trt_model.layers[i].pre_norm.bias.value = torch_to_numpy(bias)
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
        trt_model.layers[i].attention.qkv.bias.value = torch_to_numpy(
            split_bias)
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
        dst = trt_model.layers[i].attention.dense
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
            trt_model.layers[i].attention.dense.bias.value = torch_to_numpy(
                split_bias)
            feed_weight_count += 1

        # Post normalization
        if model_name in ["chatglm_6b"]:
            weight = hf_model.transformer.layers[
                i].post_attention_layernorm.weight.to(torch_type).detach()
            trt_model.layers[i].post_norm.weight.value = torch_to_numpy(weight)
            bias = hf_model.transformer.layers[
                i].post_attention_layernorm.bias.to(torch_type).detach()
            trt_model.layers[i].post_norm.bias.value = torch_to_numpy(bias)
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
            trt_model.layers[i].post_norm.weight.value = torch_to_numpy(weight)
            feed_weight_count += 1
        elif model_name in ["glm_10b"]:
            weight = hf_model.transformer.layers[
                i].post_attention_layernorm.weight.to(torch_type).detach()
            trt_model.layers[i].post_norm.weight.value = torch_to_numpy(weight)
            bias = hf_model.transformer.layers[
                i].post_attention_layernorm.bias.to(torch_type).detach()
            trt_model.layers[i].post_norm.bias.value = torch_to_numpy(bias)
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

        # Multilayer perceptron h -> 4h bias, only GLM-10B
        if model_name in ["glm_10b"]:
            bias = hf_model.transformer.layers[i].mlp.dense_h_to_4h.bias.to(
                torch_type).detach()
            split_bias = split_qkv(bias, mapping.tp_size, mapping.tp_rank,
                                   hidden_size, num_heads, num_kv_heads)
            trt_model.layers[i].mlp.fc.bias.value = torch_to_numpy(split_bias)
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
        dst = trt_model.layers[i].mlp.proj
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
            trt_model.layers[i].mlp.proj.bias.value = torch_to_numpy(split_bias)
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
        tensorrt_llm.logger.error("%d weights not loaded from HF" %
                                  (weight_count - feed_weight_count))
        return None
    tensorrt_llm.logger.info("Loading weights finish in %.2fs" % (tok - tik))
    return trt_model


def get_scaling_factors(
    model_path: Union[str, Path],
    num_layers: int,
    quant_mode: Optional[QuantMode] = None,
) -> Optional[Dict[str, List[int]]]:
    """ Get the scaling factors for Falcon model

    Returns a dictionary of scaling factors for the selected layers of the
    Falcon model.

    Args:
        model_path (str): Path to the quantized Falcon model
        layers (list): List of layers to get the scaling factors for. If None,
            all layers are selected.

    Returns:
        dict: Dictionary of scaling factors for the selected layers of the
        Falcon model.

        example:

        {
            'qkv_act': qkv_act_scale,
            'qkv_weights': qkv_weights_scale,
            'qkv_out' : qkv_outputs_scale,
            'dense_act': dense_act_scale,
            'dense_weights': dense_weights_scale,
            'fc_act': fc_act_scale,
            'fc_weights': fc_weights_scale,
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
        scaling_factor['proj_act'].append(weight_dict[f'_np:layers:{layer}:mlp:proj:activation_scaling_factor'].item())
        scaling_factor['proj_weights'].append(weight_dict[f'_np:layers:{layer}:mlp:proj:weights_scaling_factor'].item())
    # yapf: enable
    for k, v in scaling_factor.items():
        assert len(v) == num_layers, \
        f'Expect scaling factor {k} of length {num_layers}, got {len(v)}'

    return scaling_factor
