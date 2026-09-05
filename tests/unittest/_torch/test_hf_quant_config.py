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


def _mixed_precision_quant_cfg(tmp_path, quantized_layers, exclude_modules=None):
    import json

    inner = {"quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": None}
    if exclude_modules is not None:
        inner["exclude_modules"] = exclude_modules
    (tmp_path / "quant_cfg.json").write_text(
        json.dumps(
            {
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_quant_algo": None,
                "quantized_layers": quantized_layers,
            }
        )
    )
    return inner


def test_mixed_precision_excludes_fp8_block_scaled_kv_b_proj(tmp_path):
    inner = _mixed_precision_quant_cfg(
        tmp_path,
        {
            "model.layers.0.self_attn.fused_a": {"quant_algo": "FP8_BLOCK_SCALES"},
            "model.layers.0.self_attn.kv_b_proj": {"quant_algo": "FP8_BLOCK_SCALES"},
            "model.layers.3.mlp.experts": {"quant_algo": "W4A8_AWQ"},
        },
    )
    quant_config, layer_quant_config = ModelConfig._build_modelopt_quant_config(
        inner, str(tmp_path), "CUTLASS"
    )

    assert quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
    assert layer_quant_config["model.layers.3.mlp.experts"].quant_algo == QuantAlgo.W4A8_AWQ
    for pattern in ("*kv_b_proj*", "*k_b_proj*", "*eh_proj"):
        assert pattern in quant_config.exclude_modules
    assert quant_config.is_module_excluded_from_quantization("model.layers.0.self_attn.kv_b_proj")
    assert quant_config.is_module_excluded_from_quantization("kv_b_proj")


def test_mixed_precision_keeps_user_exclusions_and_skips_non_mla(tmp_path):
    inner = _mixed_precision_quant_cfg(
        tmp_path,
        {
            "model.layers.0.self_attn.kv_b_proj": {"quant_algo": "FP8_BLOCK_SCALES"},
        },
        exclude_modules=["lm_head", "*kv_b_proj*"],
    )
    quant_config, _ = ModelConfig._build_modelopt_quant_config(inner, str(tmp_path), "CUTLASS")
    assert quant_config.exclude_modules.count("*kv_b_proj*") == 1
    assert "lm_head" in quant_config.exclude_modules

    non_mla = _mixed_precision_quant_cfg(
        tmp_path,
        {
            "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8_BLOCK_SCALES"},
            "model.layers.0.mlp.experts": {"quant_algo": "W4A8_AWQ"},
        },
    )
    quant_config, _ = ModelConfig._build_modelopt_quant_config(non_mla, str(tmp_path), "CUTLASS")
    assert not quant_config.exclude_modules

    nvfp4_kv_b_proj = _mixed_precision_quant_cfg(
        tmp_path,
        {
            "model.layers.0.self_attn.kv_b_proj": {"quant_algo": "NVFP4"},
        },
    )
    quant_config, _ = ModelConfig._build_modelopt_quant_config(
        nvfp4_kv_b_proj, str(tmp_path), "CUTLASS"
    )
    assert not quant_config.exclude_modules
