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

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.models.modeling_utils import QuantAlgo

pytestmark = pytest.mark.cpu_only


def _compressed_tensors_nvfp4_config(**overrides):
    config = {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "tensor_group",
                    "group_size": 16,
                },
                "input_activations": {
                    "strategy": "tensor_group",
                },
            },
        },
    }
    config.update(overrides)
    return config


def test_load_hf_quant_config_parses_nvfp4_with_kv_cache_scheme():
    gate_exclude = "re:model\\.layers\\.\\d+\\.mlp\\.gate"
    hf_quant_config = _compressed_tensors_nvfp4_config(
        kv_cache_scheme={
            "num_bits": 8,
            "type": "float",
        },
        modules_to_not_convert=[gate_exclude],
        ignore=["lm_head"],
    )

    quant_config, layer_quant_config = ModelConfig.load_hf_quant_config(
        hf_quant_config, moe_backend="CUTLASS"
    )

    assert layer_quant_config is None
    assert quant_config.quant_algo == QuantAlgo.NVFP4
    assert quant_config.group_size == 16
    assert quant_config.kv_cache_quant_algo == QuantAlgo.FP8
    assert set(quant_config.exclude_modules) == {gate_exclude, "lm_head"}


def _modelopt_hf_quant_config(quant_algo, group_size=None, **quantization_overrides):
    """Mimic a ModelOpt ``hf_quant_config.json`` (producer + quantization)."""
    quantization = {
        "quant_algo": quant_algo,
        "kv_cache_quant_algo": None,
        "group_size": group_size,
        "exclude_modules": ["lm_head"],
    }
    quantization.update(quantization_overrides)
    return {
        "producer": {"name": "modelopt", "version": "0.47.0rc0"},
        "quantization": quantization,
    }


@pytest.mark.parametrize(
    ("quant_algo", "group_size"),
    [
        ("W4A16_NVFP4", 16),
        # ModelOpt's nvfp4_*_weight_only recipes may export 32-element blocks,
        # which the weight-only dequantization path supports.
        ("W4A16_NVFP4", 32),
        ("W4A16_NVFP4", None),
        ("NVFP4", 16),
        ("NVFP4", None),
    ],
)
def test_modelopt_quant_config_accepts_supported_nvfp4_group_sizes(quant_algo, group_size):
    quant_config, layer_quant_config = ModelConfig.load_hf_quant_config(
        _modelopt_hf_quant_config(quant_algo, group_size), moe_backend="CUTLASS"
    )

    assert layer_quant_config is None
    assert quant_config.quant_algo == QuantAlgo(quant_algo)
    assert quant_config.group_size == group_size


@pytest.mark.parametrize(
    ("quant_algo", "group_size"),
    [
        # W4A4 NVFP4 GEMMs are hardware-bound to 16-element scale blocks.
        ("NVFP4", 32),
        ("NVFP4_ARC", 32),
        # The weight-only path dequantizes in software but only knows 16 and 32.
        ("W4A16_NVFP4", 64),
        ("W4A16_NVFP4", 128),
    ],
)
def test_modelopt_quant_config_rejects_unsupported_nvfp4_group_size(quant_algo, group_size):
    with pytest.raises(ValueError, match=f"group_size={group_size}.*{quant_algo}"):
        ModelConfig.load_hf_quant_config(
            _modelopt_hf_quant_config(quant_algo, group_size), moe_backend="CUTLASS"
        )


def test_modelopt_mixed_precision_validates_per_layer_nvfp4_group_size():
    quantized_layers = {
        "model.layers.0.mlp.down_proj": {"quant_algo": "W4A16_NVFP4", "group_size": 32},
        "model.layers.0.self_attn.o_proj": {"quant_algo": "FP8"},
    }

    quant_config, layer_quant_config = ModelConfig.load_hf_quant_config(
        _modelopt_hf_quant_config("MIXED_PRECISION", quantized_layers=quantized_layers),
        moe_backend="CUTLASS",
    )

    assert quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
    assert layer_quant_config["model.layers.0.mlp.down_proj"].group_size == 32
    assert layer_quant_config["model.layers.0.self_attn.o_proj"].quant_algo == QuantAlgo.FP8

    quantized_layers["model.layers.0.mlp.gate_proj"] = {"quant_algo": "NVFP4", "group_size": 32}
    with pytest.raises(ValueError, match=r"for layer 'model\.layers\.0\.mlp\.gate_proj'"):
        ModelConfig.load_hf_quant_config(
            _modelopt_hf_quant_config("MIXED_PRECISION", quantized_layers=quantized_layers),
            moe_backend="CUTLASS",
        )
