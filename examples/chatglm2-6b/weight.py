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

import numpy as np
import torch

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy
from tensorrt_llm.quantization import QuantMode


def load_from_hf(
    tensorrt_llm_model,
    hf_model,
    mapping=None,
    dtype="float32",
    multi_query_mode=False,
):
    tensorrt_llm.logger.info("Loading weights from HF ChatGLM2-6B")
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_model, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()

    torch_type = str_dtype_to_torch(dtype)
    tensorrt_llm_model.embedding.weight.value = torch_to_numpy(
        hf_model.transformer.embedding.word_embeddings.weight.to(
            torch_type).detach().cpu())
    tensorrt_llm_model.encoder.final_layernorm.weight.value = torch_to_numpy(
        hf_model.transformer.encoder.final_layernorm.weight.to(
            torch_type).detach().cpu())
    tensorrt_llm_model.lm_head.weight.value = torch_to_numpy(
        hf_model.transformer.output_layer.weight.to(torch_type).detach().cpu())

    def load_quant_weight(src, value_dst, scale_dst,
                          plugin_weight_only_quant_type):
        v = np.ascontiguousarray(src.transpose())
        processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
            torch.tensor(v), plugin_weight_only_quant_type)
        value_dst.value = torch_to_numpy(processed_torch_weights)
        scale_dst.value = torch_to_numpy(torch_weight_scales)

    for i in range(28):
        tensorrt_llm_model.encoder.layers[
            i].input_layernorm.weight.value = torch_to_numpy(
                hf_model.transformer.encoder.layers[i].input_layernorm.weight.
                to(torch_type).detach().cpu())
        tensorrt_llm_model.encoder.layers[
            i].post_layernorm.weight.value = torch_to_numpy(
                hf_model.transformer.encoder.layers[i].post_attention_layernorm.
                weight.to(torch_type).detach().cpu())
        tensorrt_llm_model.encoder.layers[
            i].self_attention.qkv.bias.value = torch_to_numpy(
                hf_model.transformer.encoder.layers[i].self_attention.
                query_key_value.bias.to(torch_type).detach().cpu())
        # swap first and second half weight columns to adapt trt_llm Swiglu
        h_to_4h_weight = hf_model.transformer.encoder.layers[
            i].mlp.dense_h_to_4h.weight.to(torch_type).detach().cpu()
        h_to_4h_weight = torch.split(h_to_4h_weight,
                                     h_to_4h_weight.shape[0] // 2, 0)
        h_to_4h_weight = torch_to_numpy(torch.concat(h_to_4h_weight[::-1], 0))
        if use_weight_only:
            load_quant_weight(
                src=h_to_4h_weight,
                value_dst=tensorrt_llm_model.encoder.layers[i].mlp.fc.weight,
                scale_dst=tensorrt_llm_model.encoder.layers[i].mlp.fc.
                per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
            load_quant_weight(
                src=torch_to_numpy(
                    hf_model.transformer.encoder.layers[i].mlp.dense_4h_to_h.
                    weight.to(torch_type).detach().cpu()),
                value_dst=tensorrt_llm_model.encoder.layers[i].mlp.proj.weight,
                scale_dst=tensorrt_llm_model.encoder.layers[i].mlp.proj.
                per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
            load_quant_weight(
                src=torch_to_numpy(
                    hf_model.transformer.encoder.layers[i].self_attention.
                    query_key_value.weight.to(torch_type).detach().cpu()),
                value_dst=tensorrt_llm_model.encoder.layers[i].self_attention.
                qkv.weight,
                scale_dst=tensorrt_llm_model.encoder.layers[i].self_attention.
                qkv.per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
            load_quant_weight(
                src=torch_to_numpy(
                    hf_model.transformer.encoder.layers[i].self_attention.dense.
                    weight.to(torch_type).detach().cpu()),
                value_dst=tensorrt_llm_model.encoder.layers[i].self_attention.
                dense.weight,
                scale_dst=tensorrt_llm_model.encoder.layers[i].self_attention.
                dense.per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)

        else:
            tensorrt_llm_model.encoder.layers[
                i].self_attention.qkv.weight.value = torch_to_numpy(
                    hf_model.transformer.encoder.layers[i].self_attention.
                    query_key_value.weight.to(torch_type).detach().cpu())
            tensorrt_llm_model.encoder.layers[
                i].self_attention.dense.weight.value = torch_to_numpy(
                    hf_model.transformer.encoder.layers[i].self_attention.dense.
                    weight.to(torch_type).detach().cpu())
            tensorrt_llm_model.encoder.layers[
                i].mlp.fc.weight.value = h_to_4h_weight
            tensorrt_llm_model.encoder.layers[
                i].mlp.proj.weight.value = torch_to_numpy(
                    hf_model.transformer.encoder.layers[i].mlp.dense_4h_to_h.
                    weight.to(torch_type).detach().cpu())

    tok = time.time()
    tensorrt_llm.logger.info("Loading weights finish in %.2fs" % (tok - tik))
    return tensorrt_llm_model
