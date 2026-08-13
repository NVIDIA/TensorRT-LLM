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

from typing import Any

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


def _fp8_block_scales_config(**overrides: Any) -> dict[str, Any]:
    config = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    config.update(overrides)
    return config


def test_fp8_block_scales_honours_ignored_layers() -> None:
    """vLLM/AutoFP8-style "fp8" configs record BF16 layers as "ignored_layers".

    These checkpoints carry no "modules_to_not_convert" at all, so before this
    was read every exclusion was dropped and the layers were built as FP8.
    """
    ignored = ["lm_head", "model.layers.0.self_attn.g_proj"]
    hf_quant_config = _fp8_block_scales_config(ignored_layers=ignored)

    quant_config, _ = ModelConfig.load_hf_quant_config(hf_quant_config, moe_backend="CUTLASS")

    assert quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
    for name in ignored:
        assert name in quant_config.exclude_modules
        assert quant_config.is_module_excluded_from_quantization(name)
    # the built-in defaults are still applied
    assert "*kv_b_proj*" in quant_config.exclude_modules


def test_fp8_block_scales_merges_both_exclusion_keys() -> None:
    """Both keys are honoured, and an entry named twice appears once.

    Producers commonly write the same module under both keys, and one of the
    entries here also collides with a built-in default, so this fails if the
    merge stops de-duplicating.
    """
    # "lm_head" appears only under ignored_layers, so this still fails if that
    # key is not read; "*kv_b_proj*" duplicates a built-in default.
    hf_quant_config = _fp8_block_scales_config(
        ignored_layers=["lm_head", "*kv_b_proj*"],
        modules_to_not_convert=["model.layers.1.mlp.down_proj"],
    )

    quant_config, _ = ModelConfig.load_hf_quant_config(hf_quant_config, moe_backend="CUTLASS")

    excluded = quant_config.exclude_modules
    assert "lm_head" in excluded
    assert "model.layers.1.mlp.down_proj" in excluded
    assert excluded.count("lm_head") == 1
    assert excluded.count("*kv_b_proj*") == 1
    assert len(excluded) == len(set(excluded))


def test_fp8_block_scales_without_exclusions_keeps_defaults() -> None:
    quant_config, _ = ModelConfig.load_hf_quant_config(
        _fp8_block_scales_config(), moe_backend="CUTLASS"
    )

    assert quant_config.exclude_modules == ["*kv_b_proj*", "*k_b_proj*", "*eh_proj"]


def test_compressed_tensors_honours_ignore_key() -> None:
    """Characterization: compressed-tensors keeps its "ignore" list.

    update_quant_config_from_compressed_tensors already merged "ignore", so
    this passes with or without the general merge below. Kept to pin that the
    hoisted merge does not REGRESS the path that already worked, including the
    per-expert regex form a hybrid checkpoint uses.
    """
    ignore = [
        "lm_head",
        r"re:^model\.layers\.4[0-7]\.mlp\.experts\.[0-9]+\.(gate_proj|up_proj|down_proj)$",
    ]
    hf_quant_config = _compressed_tensors_nvfp4_config(ignore=ignore)

    quant_config, _ = ModelConfig.load_hf_quant_config(hf_quant_config, moe_backend="CUTLASS")

    for entry in ignore:
        assert entry in quant_config.exclude_modules, f"{entry!r} was dropped"


def test_ignore_and_ignored_layers_are_both_merged() -> None:
    """A config carrying both spellings keeps both, de-duplicated."""
    hf_quant_config = _fp8_block_scales_config(
        ignored_layers=["model.layers.3.self_attn.g_proj", "lm_head"],
        ignore=["model.layers.9.mlp.experts", "lm_head"],
    )

    quant_config, _ = ModelConfig.load_hf_quant_config(hf_quant_config, moe_backend="CUTLASS")

    ex = quant_config.exclude_modules
    for entry in ("model.layers.3.self_attn.g_proj", "model.layers.9.mlp.experts", "lm_head"):
        assert entry in ex, f"{entry!r} missing"
    assert ex.count("lm_head") == 1, "duplicate not collapsed"
