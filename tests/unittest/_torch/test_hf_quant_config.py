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

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

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


def _modelopt_hf_quant_config(
    quant_algo: str, group_size: int | None = None, **quantization_overrides: object
) -> dict:
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
        ("NVFP4_ARC", 16),
    ],
)
def test_modelopt_quant_config_accepts_supported_nvfp4_group_sizes(
    quant_algo: str, group_size: int | None
) -> None:
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
def test_modelopt_quant_config_rejects_unsupported_nvfp4_group_size(
    quant_algo: str, group_size: int
) -> None:
    with pytest.raises(ValueError, match=f"group_size={group_size}.*{quant_algo}"):
        ModelConfig.load_hf_quant_config(
            _modelopt_hf_quant_config(quant_algo, group_size), moe_backend="CUTLASS"
        )


def test_modelopt_mixed_precision_validates_per_layer_nvfp4_group_size() -> None:
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


@pytest.mark.parametrize(
    "layer_name",
    [
        "model.language_model.layers.0.mlp.experts",
        "model.layers.0.mlp.experts.0.gate_proj",
    ],
)
def test_modelopt_rejects_32_element_nvfp4_experts_before_backend_selection(
    layer_name: str,
) -> None:
    with pytest.raises(ValueError, match="only supported by the dense W4A16_NVFP4 Linear path"):
        ModelConfig.load_hf_quant_config(
            _modelopt_hf_quant_config(
                "MIXED_PRECISION",
                quantized_layers={layer_name: {"quant_algo": "W4A16_NVFP4", "group_size": 32}},
            ),
            moe_backend="CUTLASS",
        )


@pytest.mark.parametrize("group_size", [None, 16, 32, 128])
def test_modelopt_preserves_w4a8_nvfp4_fp8_group_size(group_size: int | None) -> None:
    quant_config, _ = ModelConfig.load_hf_quant_config(
        _modelopt_hf_quant_config("W4A8_NVFP4_FP8", group_size), moe_backend="TRTLLM"
    )
    assert quant_config.quant_algo == QuantAlgo.W4A8_NVFP4_FP8
    assert quant_config.group_size == group_size


def _mixed_precision_quant_cfg(
    tmp_path: Path,
    quantized_layers: dict[str, dict[str, str]],
    exclude_modules: list[str] | None = None,
) -> dict[str, object]:
    """Write the checkpoint's per-layer ModelOpt recipes."""
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


def test_mixed_precision_excludes_only_fp8_mla_projections(tmp_path: Path) -> None:
    """Exclusions preserve mixed algorithms across layers and projection types."""
    recipes = {
        "model.layers.0.self_attn.kv_b_proj": "FP8_BLOCK_SCALES",
        "model.layers.0.self_attn.k_b_proj": "NVFP4",
        "model.layers.0.eh_proj": "W4A8_AWQ",
        "model.layers.1.self_attn.kv_b_proj": "NVFP4",
        "model.layers.1.self_attn.k_b_proj": "FP8_BLOCK_SCALES",
        "model.layers.1.eh_proj": "FP8_BLOCK_SCALES",
        "model.layers.2.self_attn.kv_b_proj": "W4A8_AWQ",
        "model.layers.3.self_attn.q_proj": "FP8_BLOCK_SCALES",
        "model.layers.3.mlp.experts": "W4A8_AWQ",
    }
    expected_exclusions = {
        "model.layers.0.self_attn.kv_b_proj",
        "model.layers.1.self_attn.k_b_proj",
        "model.layers.1.eh_proj",
    }
    inner = _mixed_precision_quant_cfg(
        tmp_path, {name: {"quant_algo": algo} for name, algo in recipes.items()}
    )
    quant_config, layer_quant_config = ModelConfig._build_modelopt_quant_config(
        inner, str(tmp_path), "CUTLASS"
    )
    assert quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
    assert set(quant_config.exclude_modules) == expected_exclusions
    assert not quant_config.is_module_excluded_from_quantization("kv_b_proj")
    assert not quant_config.is_module_excluded_from_quantization(
        "model.layers.4.self_attn.kv_b_proj"
    )
    # Exercise the actual assignment/exclusion order, not just the config list.
    modules = {}
    for name in recipes:
        module = Linear(
            128,
            128,
            bias=False,
            dtype=torch.bfloat16,
            quant_config=QuantConfig(),
            skip_create_weights_in_init=True,
        )
        modules[name] = module
    model = SimpleNamespace(
        model_config=SimpleNamespace(
            quant_config=quant_config, quant_config_dict=layer_quant_config
        ),
        named_modules=lambda: modules.items(),
    )
    DecoderModelForCausalLM.apply_layerwise_quant_config(model)
    DecoderModelForCausalLM.apply_quant_config_exclude_modules(model)
    for name, algo in recipes.items():
        assert layer_quant_config[name].quant_algo == QuantAlgo(algo)
        expected = None if name in expected_exclusions else QuantAlgo(algo)
        assert modules[name].quant_config.quant_algo == expected


@pytest.mark.parametrize("exclusion", ["model.layers.0.self_attn.kv_b_proj", "*kv_b_proj*"])
def test_mixed_precision_keeps_user_exclusions(tmp_path: Path, exclusion: str) -> None:
    """Preserve exact or wildcard user exclusions without redundant entries."""
    inner = _mixed_precision_quant_cfg(
        tmp_path,
        {"model.layers.0.self_attn.kv_b_proj": {"quant_algo": "FP8_BLOCK_SCALES"}},
        exclude_modules=["lm_head", exclusion],
    )
    quant_config, _ = ModelConfig._build_modelopt_quant_config(inner, str(tmp_path), "CUTLASS")
    assert quant_config.exclude_modules == ["lm_head", exclusion]


@pytest.mark.parametrize(
    ("name", "algo"),
    [
        ("model.layers.0.self_attn.q_proj", "FP8_BLOCK_SCALES"),
        ("model.layers.0.self_attn.kv_b_proj", "NVFP4"),
        ("model.layers.0.self_attn.kv_b_proj", "W4A8_AWQ"),
    ],
)
def test_mixed_precision_skips_non_fp8_mla(tmp_path: Path, name: str, algo: str) -> None:
    """Unrelated projection types and non-FP8 recipes gain no exclusions."""
    inner = _mixed_precision_quant_cfg(tmp_path, {name: {"quant_algo": algo}})
    quant_config, _ = ModelConfig._build_modelopt_quant_config(inner, str(tmp_path), "CUTLASS")
    assert not quant_config.exclude_modules
