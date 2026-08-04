# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, Mapping

from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


def update_quant_config_from_compressed_tensors(
    quant_config: QuantConfig, hf_quant_config: Mapping[str, Any]
) -> None:
    """Mutate QuantConfig from an llm-compressor compressed-tensors config."""
    config_groups = hf_quant_config.get("config_groups")
    if config_groups is None:
        raise ValueError(f"config_groups is not set in {hf_quant_config}.")

    weights_quant_config = config_groups["group_0"]["weights"]
    inputs_quant_config = config_groups["group_0"]["input_activations"]
    weights_quant_strategy = weights_quant_config["strategy"]
    inputs_quant_strategy = inputs_quant_config["strategy"]

    if weights_quant_config["num_bits"] == 8:
        if weights_quant_strategy == "channel":
            if inputs_quant_strategy != "token":
                raise ValueError(f"Unsupported inputs_quant_strategy: {inputs_quant_strategy}.")
            quant_config.quant_algo = QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN
        elif weights_quant_strategy == "block":
            if inputs_quant_strategy != "group":
                raise ValueError(f"Unsupported inputs_quant_strategy: {inputs_quant_strategy}.")
            quant_config.quant_algo = QuantAlgo.FP8_BLOCK_SCALES
            group_size = inputs_quant_config["group_size"]

            # TRT-LLM only supports group_size=128 for FP8_BLOCK_SCALES.
            if group_size != 128:
                raise ValueError(f"Unsupported group_size: {group_size}. Supported: 128.")
            quant_config.group_size = group_size

        else:
            raise ValueError(
                f"Unsupported weights_quant_strategy: {weights_quant_strategy}. "
                "Supported strategies: 'channel', 'block'."
            )
    elif (
        weights_quant_config["num_bits"] == 4
        and weights_quant_config.get("type") == "float"
        and weights_quant_strategy == "tensor_group"
    ):
        # llm-compressor NVFP4: weights FP4 with FP8 per-group scales
        # (group_size=16), scaled by an FP32 global scale.
        if inputs_quant_strategy != "tensor_group":
            raise ValueError(
                f"Unsupported inputs_quant_strategy for NVFP4: {inputs_quant_strategy}."
            )
        group_size = weights_quant_config["group_size"]
        if group_size != 16:
            raise ValueError(f"Unsupported group_size: {group_size}. Supported: 16 for NVFP4.")
        quant_config.quant_algo = QuantAlgo.NVFP4
        quant_config.group_size = group_size
    else:
        raise ValueError(
            f"Unsupported quant_bits: {weights_quant_config['num_bits']}. "
            "Supported: 8 (FP8) or 4 (NVFP4)."
        )

    # kv_cache_scheme (llm-compressor): FP8 per-tensor KV cache.
    kv_cache_scheme = hf_quant_config.get("kv_cache_scheme")
    if kv_cache_scheme is not None:
        if kv_cache_scheme.get("num_bits") == 8 and kv_cache_scheme.get("type") == "float":
            if quant_config.kv_cache_quant_algo in (None, QuantAlgo.FP8):
                quant_config.kv_cache_quant_algo = QuantAlgo.FP8
            else:
                raise ValueError(
                    f"Specified kv_cache_quant_algo={quant_config.kv_cache_quant_algo}, "
                    "conflicting with FP8 KV cache from HF quant config."
                )
        else:
            raise ValueError(f"Unsupported kv_cache_scheme: {kv_cache_scheme}.")

    hf_exclude_modules = hf_quant_config.get("modules_to_not_convert", None)
    if hf_exclude_modules is not None:
        quant_config.exclude_modules = list(
            set(hf_exclude_modules + hf_quant_config.get("ignore", []))
        )
    else:
        quant_config.exclude_modules = hf_quant_config.get("ignore", [])
