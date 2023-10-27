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
    max_seq_length=2048,
    multi_query_mode=False,
):
    tensorrt_llm.logger.info("Loading weights from HF ChatGLM-6B")
    tik = time.time()

    quant_mode = getattr(tensorrt_llm_model, 'quant_mode', QuantMode(0))
    if quant_mode.is_int8_weight_only():
        plugin_weight_only_quant_type = torch.int8
    elif quant_mode.is_int4_weight_only():
        plugin_weight_only_quant_type = torch.quint4x2
    use_weight_only = quant_mode.is_weight_only()

    torch_type = str_dtype_to_torch(dtype)
    tensorrt_llm_model.embedding.weight.value = torch_to_numpy(
        hf_model.transformer.word_embeddings.weight.to(
            torch_type).detach().cpu())
    tensorrt_llm_model.final_layernorm.weight.value = torch_to_numpy(
        hf_model.transformer.final_layernorm.weight.to(
            torch_type).detach().cpu())
    tensorrt_llm_model.final_layernorm.bias.value = torch_to_numpy(
        hf_model.transformer.final_layernorm.bias.to(torch_type).detach().cpu())
    tensorrt_llm_model.lm_head.weight.value = torch_to_numpy(
        hf_model.lm_head.weight.to(torch_type).detach().cpu())

    def load_quant_weight(src, value_dst, scale_dst,
                          plugin_weight_only_quant_type):
        v = np.ascontiguousarray(src.transpose())
        processed_torch_weights, torch_weight_scales = torch.ops.fastertransformer.symmetric_quantize_last_axis_of_batched_matrix(
            torch.tensor(v), plugin_weight_only_quant_type)
        value_dst.value = torch_to_numpy(processed_torch_weights)
        scale_dst.value = torch_to_numpy(torch_weight_scales)

    for i in range(28):
        tensorrt_llm_model.layers[
            i].input_layernorm.weight.value = torch_to_numpy(
                hf_model.transformer.layers[i].input_layernorm.weight.to(
                    torch_type).detach().cpu())
        tensorrt_llm_model.layers[
            i].input_layernorm.bias.value = torch_to_numpy(
                hf_model.transformer.layers[i].input_layernorm.bias.to(
                    torch_type).detach().cpu())
        tensorrt_llm_model.layers[
            i].post_layernorm.weight.value = torch_to_numpy(
                hf_model.transformer.layers[i].post_attention_layernorm.weight.
                to(torch_type).detach().cpu())
        tensorrt_llm_model.layers[i].post_layernorm.bias.value = torch_to_numpy(
            hf_model.transformer.layers[i].post_attention_layernorm.bias.to(
                torch_type).detach().cpu())
        tensorrt_llm_model.layers[i].attention.qkv.bias.value = torch_to_numpy(
            hf_model.transformer.layers[i].attention.query_key_value.bias.to(
                torch_type).detach().cpu())
        if use_weight_only:
            load_quant_weight(
                src=torch_to_numpy(
                    hf_model.transformer.layers[i].mlp.dense_h_to_4h.weight.to(
                        torch_type).detach().cpu()),
                value_dst=tensorrt_llm_model.layers[i].mlp.fc.weight,
                scale_dst=tensorrt_llm_model.layers[i].mlp.fc.per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
            load_quant_weight(
                src=torch_to_numpy(
                    hf_model.transformer.layers[i].mlp.dense_4h_to_h.weight.to(
                        torch_type).detach().cpu()),
                value_dst=tensorrt_llm_model.layers[i].mlp.proj.weight,
                scale_dst=tensorrt_llm_model.layers[i].mlp.proj.
                per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
            load_quant_weight(
                src=torch_to_numpy(
                    hf_model.transformer.layers[i].attention.query_key_value.
                    weight.to(torch_type).detach().cpu()),
                value_dst=tensorrt_llm_model.layers[i].attention.qkv.weight,
                scale_dst=tensorrt_llm_model.layers[i].attention.qkv.
                per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)
            load_quant_weight(
                src=torch_to_numpy(
                    hf_model.transformer.layers[i].attention.dense.weight.to(
                        torch_type).detach().cpu()),
                value_dst=tensorrt_llm_model.layers[i].attention.dense.weight,
                scale_dst=tensorrt_llm_model.layers[i].attention.dense.
                per_channel_scale,
                plugin_weight_only_quant_type=plugin_weight_only_quant_type)

        else:
            tensorrt_llm_model.layers[
                i].attention.qkv.weight.value = torch_to_numpy(
                    hf_model.transformer.layers[i].attention.query_key_value.
                    weight.to(torch_type).detach().cpu())
            tensorrt_llm_model.layers[
                i].attention.dense.weight.value = torch_to_numpy(
                    hf_model.transformer.layers[i].attention.dense.weight.to(
                        torch_type).detach().cpu())
            tensorrt_llm_model.layers[i].mlp.fc.weight.value = torch_to_numpy(
                hf_model.transformer.layers[i].mlp.dense_h_to_4h.weight.to(
                    torch_type).detach().cpu())
            tensorrt_llm_model.layers[i].mlp.proj.weight.value = torch_to_numpy(
                hf_model.transformer.layers[i].mlp.dense_4h_to_h.weight.to(
                    torch_type).detach().cpu())

    tok = time.time()
    tensorrt_llm.logger.info("Loading weights finish in %.2fs" % (tok - tik))
    return tensorrt_llm_model
