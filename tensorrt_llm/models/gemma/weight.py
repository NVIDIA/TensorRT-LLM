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
import os
from typing import Dict, Optional

import numpy as np
import torch

from tensorrt_llm.models.modeling_utils import PretrainedConfig

from ..._utils import numpy_to_torch
from ...logger import logger
from ...mapping import Mapping

Weights = Dict[str, torch.Tensor]


def quantize_fp8_weights(weights: Weights, num_layers: int,
                         mapping: Mapping) -> Weights:

    def get_scaling_factor(weight):
        amax = weight.max()
        scale = 448.0 / amax
        return scale

    layers_range = mapping.pp_layers(num_layers)
    scaling_factors = {}
    scaled_weights = {}
    trt_llm_prefix = "transformer.layers"
    for l in layers_range:
        # attention.qkv.weight
        for name in [
                "attention.qkv", "attention.dense", "mlp.fc", "mlp.gate",
                "mlp.proj"
        ]:
            trt_llm_name = ".".join((trt_llm_prefix, str(l), name, "weight"))
            scale_name = ".".join(
                (trt_llm_prefix, str(l), name, "weights_scaling_factor"))
            weight = weights[trt_llm_name].float()
            dtype = weights[trt_llm_name].dtype
            scale = get_scaling_factor(weight)
            scaled_weights[trt_llm_name] = (weight *
                                            scale).to(dtype).contiguous()
            scaling_factors[scale_name] = numpy_to_torch(
                np.asarray([1 / scale]).astype(np.float32))
    return scaling_factors


def load_from_fp8_gemma(quant_ckpt_path: Optional[str], num_layers: int,
                        mapping: Mapping, fp8_kv_cache: bool,
                        weight_scales: Weights):
    """
    Get the fp8 scaling factors.
    """
    fake_fp8_sf_dt = torch.float32

    if quant_ckpt_path is not None and os.path.isfile(quant_ckpt_path):
        fp8_gemma = np.load(quant_ckpt_path)
    else:
        fp8_gemma = None
        logger.info(
            f"There is not quantized checkpoint, use dummy fp8 scaling factors instead."
        )
    weights = {}

    def get_fp8_gemma(name: str) -> np.ndarray:
        if fp8_gemma is not None:
            return fp8_gemma[name]
        else:
            return torch.tensor([1.0], dtype=fake_fp8_sf_dt).numpy()

    layers_range = mapping.pp_layers(num_layers)
    for l in layers_range:
        prefix = f'_np:layers:{l}'
        tllm_prex = f'transformer.layers.{l-layers_range[0]}'

        weights[f'{tllm_prex}.attention.qkv.activation_scaling_factor'] = max(
            get_fp8_gemma(
                f'{prefix}:attention:qkv:q:activation_scaling_factor'),
            get_fp8_gemma(
                f'{prefix}:attention:qkv:k:activation_scaling_factor'),
            get_fp8_gemma(
                f'{prefix}:attention:qkv:v:activation_scaling_factor'))
        weights[f'{tllm_prex}.attention.qkv.weights_scaling_factor'] = max(
            get_fp8_gemma(f'{prefix}:attention:qkv:q:weights_scaling_factor'),
            get_fp8_gemma(f'{prefix}:attention:qkv:k:weights_scaling_factor'),
            get_fp8_gemma(f'{prefix}:attention:qkv:v:weights_scaling_factor'))
        weights[
            f'{tllm_prex}.attention.dense.activation_scaling_factor'] = get_fp8_gemma(
                f'{prefix}:attention:dense:activation_scaling_factor')
        weights[
            f'{tllm_prex}.attention.dense.weights_scaling_factor'] = get_fp8_gemma(
                f'{prefix}:attention:dense:weights_scaling_factor')

        weights[
            f'{tllm_prex}.mlp.fc.activation_scaling_factor'] = get_fp8_gemma(
                f'{prefix}:mlp:fc:activation_scaling_factor')
        weights[f'{tllm_prex}.mlp.fc.weights_scaling_factor'] = get_fp8_gemma(
            f'{prefix}:mlp:fc:weights_scaling_factor')

        weights[
            f'{tllm_prex}.mlp.gate.activation_scaling_factor'] = get_fp8_gemma(
                f'{prefix}:mlp:gate:activation_scaling_factor')
        weights[f'{tllm_prex}.mlp.gate.weights_scaling_factor'] = get_fp8_gemma(
            f'{prefix}:mlp:gate:weights_scaling_factor')

        weights[
            f'{tllm_prex}.mlp.proj.activation_scaling_factor'] = get_fp8_gemma(
                f'{prefix}:mlp:proj:activation_scaling_factor')
        weights[f'{tllm_prex}.mlp.proj.weights_scaling_factor'] = get_fp8_gemma(
            f'{prefix}:mlp:proj:weights_scaling_factor')

        if fp8_kv_cache:
            # Not calibrating KV cache.
            scaling_factor = 1.0
            weights[
                f'{tllm_prex}.attention.kv_cache_scaling_factor'] = torch.tensor(
                    [scaling_factor], dtype=fake_fp8_sf_dt).numpy()
            if fp8_gemma is None:
                weights.update(weight_scales)

    for key in weights:
        if isinstance(weights[key], np.ndarray):
            weights[key] = numpy_to_torch(weights[key])
    return weights


def dummy_weights_awq(weights: Weights, precision: str,
                      trt_llm_config: PretrainedConfig, group_size: int):
    packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
    use_fp8_kv_cache = trt_llm_config.quant_mode.has_fp8_kv_cache()
    use_int8_kv_cache = trt_llm_config.quant_mode.has_int8_kv_cache()
    num_layers = trt_llm_config.num_hidden_layers
    for name in list(weights):
        if any([
                _name in name for _name in [
                    'mlp.proj.weight', 'mlp.gate.weight', 'mlp.fc.weight',
                    'attention.qkv.weight', 'attention.dense.weight'
                ]
        ]):
            print("Processing:", name)
            weight = np.ascontiguousarray(weights[name].T)
            in_dim, out_dim = weight.shape
            scale = np.amax(weight) / 7
            weights_scaling_factor = np.ones([out_dim, in_dim // group_size
                                              ]) * scale.astype(np.float32)
            weight_smoothed = (weight.astype(np.float32) / scale).astype(
                np.int8)
            weight_smoothed[weight_smoothed < -8] = -8
            weight_smoothed[weight_smoothed > 7] = 7
            prequant_scaling_factor = np.ones([in_dim], dtype=weight.dtype)
            weights[name] = packer(
                torch.from_numpy(weight_smoothed)).T.contiguous().numpy()
            weights[name.replace(
                'weight', 'prequant_scaling_factor')] = prequant_scaling_factor
            weights[name.replace(
                'weight',
                'weights_scaling_factor')] = weights_scaling_factor.astype(
                    weight.dtype)
            if precision == "w4a8_awq":
                alpha = np.array([1], dtype=np.float32)
                weights[name.replace('weight', 'alpha')] = alpha
    if use_fp8_kv_cache or use_int8_kv_cache:
        for l in range(num_layers):
            t = np.array([1], dtype=np.float32)
            weights[
                f"transformer.layers.{l}.attention.kv_cache_scaling_factor"] = t

    return weights
