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

import pytest

from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.models.quant_config_utils import update_quant_config_from_compressed_tensors
from tensorrt_llm.quantization.mode import QuantAlgo


def _compressed_tensors_config(weights=None, input_activations=None, **overrides):
    config = {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "weights": weights
                or {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "tensor_group",
                    "group_size": 16,
                },
                "input_activations": input_activations
                or {
                    "strategy": "tensor_group",
                },
            },
        },
    }
    config.update(overrides)
    return config


def test_update_quant_config_from_compressed_tensors_parses_nvfp4():
    gate_exclude = "re:model\\.layers\\.\\d+\\.mlp\\.gate"
    quant_config = QuantConfig()
    update_quant_config_from_compressed_tensors(
        quant_config,
        _compressed_tensors_config(
            kv_cache_scheme={
                "num_bits": 8,
                "type": "float",
            },
            modules_to_not_convert=[gate_exclude],
            ignore=["lm_head"],
        ),
    )

    assert quant_config.quant_algo == QuantAlgo.NVFP4
    assert quant_config.group_size == 16
    assert quant_config.kv_cache_quant_algo == QuantAlgo.FP8
    assert set(quant_config.exclude_modules) == {gate_exclude, "lm_head"}


def test_update_quant_config_from_compressed_tensors_parses_fp8_block_scales():
    quant_config = QuantConfig()
    update_quant_config_from_compressed_tensors(
        quant_config,
        _compressed_tensors_config(
            weights={
                "num_bits": 8,
                "strategy": "block",
            },
            input_activations={
                "num_bits": 8,
                "strategy": "group",
                "group_size": 128,
            },
            ignore=["lm_head"],
        ),
    )

    assert quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
    assert quant_config.group_size == 128
    assert quant_config.exclude_modules == ["lm_head"]


def test_update_quant_config_from_compressed_tensors_parses_fp8_channel():
    quant_config = QuantConfig()
    update_quant_config_from_compressed_tensors(
        quant_config,
        _compressed_tensors_config(
            weights={
                "num_bits": 8,
                "strategy": "channel",
            },
            input_activations={
                "num_bits": 8,
                "strategy": "token",
            },
            ignore=["lm_head"],
        ),
    )

    assert quant_config.quant_algo == QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN
    assert quant_config.exclude_modules == ["lm_head"]


def test_update_quant_config_from_compressed_tensors_parses_scheme_named_group():
    # Some compressed-tensors checkpoints (e.g. NIM fp8 packaging) key
    # config_groups by the scheme name ("FP8_DYNAMIC") instead of "group_0".
    # The single group must still parse rather than raising KeyError: 'group_0'.
    config = _compressed_tensors_config(
        weights={
            "num_bits": 8,
            "strategy": "channel",
        },
        input_activations={
            "num_bits": 8,
            "strategy": "token",
        },
        ignore=["lm_head"],
    )
    groups = config["config_groups"]
    groups["FP8_DYNAMIC"] = groups.pop("group_0")  # re-key: scheme name, not group_0

    quant_config = QuantConfig()
    update_quant_config_from_compressed_tensors(quant_config, config)

    assert quant_config.quant_algo == QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN
    assert quant_config.exclude_modules == ["lm_head"]


def test_update_quant_config_from_compressed_tensors_rejects_ambiguous_groups():
    # Multiple groups, none named "group_0" -> ambiguous which to apply globally.
    config = _compressed_tensors_config(
        weights={
            "num_bits": 8,
            "strategy": "channel",
        },
        input_activations={
            "num_bits": 8,
            "strategy": "token",
        },
    )
    group = config["config_groups"].pop("group_0")
    config["config_groups"]["FP8_DYNAMIC"] = group
    config["config_groups"]["FP8_STATIC"] = dict(group)

    with pytest.raises(ValueError, match="exactly one config group"):
        update_quant_config_from_compressed_tensors(QuantConfig(), config)


@pytest.mark.parametrize(
    "weights,input_activations,error_match",
    [
        (
            {
                "num_bits": 8,
                "strategy": "block",
            },
            {
                "num_bits": 8,
                "strategy": "group",
                "group_size": 64,
            },
            "Supported: 128",
        ),
        (
            {
                "num_bits": 4,
                "type": "float",
                "strategy": "tensor_group",
                "group_size": 32,
            },
            {
                "strategy": "tensor_group",
            },
            "Supported: 16 for NVFP4",
        ),
    ],
)
def test_update_quant_config_from_compressed_tensors_rejects_group_sizes(
    weights, input_activations, error_match
):
    with pytest.raises(ValueError, match=error_match):
        update_quant_config_from_compressed_tensors(
            QuantConfig(),
            _compressed_tensors_config(weights=weights, input_activations=input_activations),
        )


@pytest.mark.parametrize(
    "weights,input_activations,error_match",
    [
        (
            {
                "num_bits": 8,
                "strategy": "tensor",
            },
            {
                "num_bits": 8,
                "strategy": "token",
            },
            "Unsupported weights_quant_strategy",
        ),
        (
            {
                "num_bits": 4,
                "type": "float",
                "strategy": "tensor_group",
                "group_size": 16,
            },
            {
                "strategy": "token",
            },
            "Unsupported inputs_quant_strategy for NVFP4",
        ),
    ],
)
def test_update_quant_config_from_compressed_tensors_rejects_strategies(
    weights, input_activations, error_match
):
    with pytest.raises(ValueError, match=error_match):
        update_quant_config_from_compressed_tensors(
            QuantConfig(),
            _compressed_tensors_config(weights=weights, input_activations=input_activations),
        )


def test_update_quant_config_from_compressed_tensors_requires_config_groups():
    with pytest.raises(ValueError, match="config_groups is not set"):
        update_quant_config_from_compressed_tensors(
            QuantConfig(),
            {
                "quant_method": "compressed-tensors",
            },
        )


def test_update_quant_config_from_compressed_tensors_rejects_kv_cache_scheme():
    with pytest.raises(ValueError, match="Unsupported kv_cache_scheme"):
        update_quant_config_from_compressed_tensors(
            QuantConfig(),
            _compressed_tensors_config(
                kv_cache_scheme={
                    "num_bits": 4,
                    "type": "float",
                }
            ),
        )


def test_update_quant_config_from_compressed_tensors_rejects_weight_num_bits():
    with pytest.raises(ValueError, match="Unsupported quant_bits"):
        update_quant_config_from_compressed_tensors(
            QuantConfig(),
            _compressed_tensors_config(
                weights={
                    "num_bits": 3,
                    "strategy": "block",
                },
                input_activations={
                    "num_bits": 8,
                    "strategy": "group",
                    "group_size": 128,
                },
            ),
        )


def test_update_quant_config_from_compressed_tensors_rejects_kv_cache_conflict():
    with pytest.raises(ValueError, match="conflicting with FP8 KV cache"):
        update_quant_config_from_compressed_tensors(
            QuantConfig(kv_cache_quant_algo=QuantAlgo.NVFP4),
            _compressed_tensors_config(
                kv_cache_scheme={
                    "num_bits": 8,
                    "type": "float",
                }
            ),
        )
