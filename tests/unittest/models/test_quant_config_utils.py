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
from unittest.mock import patch

import pytest

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.models.quant_config_utils import update_quant_config_from_compressed_tensors
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only


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


def test_update_quant_config_from_compressed_tensors_single_group_has_no_layer_configs():
    # Regression: a single-group checkpoint stays a global quant_algo, and
    # supplying module names must not turn it into a per-layer config.
    quant_config = QuantConfig()
    layer_quant_configs = update_quant_config_from_compressed_tensors(
        quant_config,
        _compressed_tensors_config(ignore=["lm_head"]),
        module_names=["model.layers.0.mlp.down_proj", "lm_head"],
    )

    assert layer_quant_configs is None
    assert quant_config.quant_algo == QuantAlgo.NVFP4


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


def test_update_quant_config_from_compressed_tensors_mxfp4_with_fp8_kv_cache():
    quant_config = QuantConfig()
    update_quant_config_from_compressed_tensors(
        quant_config,
        _compressed_tensors_config(
            weights={
                "num_bits": 4,
                "type": "float",
                "strategy": "group",
                "group_size": 32,
            },
            format="mxfp4-pack-quantized",
            kv_cache_scheme={
                "num_bits": 8,
                "type": "float",
            },
            ignore=["lm_head"],
        ),
    )

    assert quant_config.quant_algo == QuantAlgo.W4A16_MXFP4
    assert quant_config.group_size == 32
    # The MXFP4 branch returns early; kv_cache_scheme must still be honored.
    assert quant_config.kv_cache_quant_algo == QuantAlgo.FP8
    assert set(quant_config.exclude_modules) == {"lm_head"}


def test_update_quant_config_from_compressed_tensors_group_format_overrides_top_level():
    # Multi-group checkpoints declare "format" per group; a group-level format
    # must win over the checkpoint's top-level one.
    config = _compressed_tensors_config(
        weights={
            "num_bits": 4,
            "type": "float",
            "strategy": "group",
            "group_size": 32,
        },
        format="mixed-precision",
    )
    config["config_groups"]["group_0"]["format"] = "mxfp4-pack-quantized"

    quant_config = QuantConfig()
    update_quant_config_from_compressed_tensors(quant_config, config)

    assert quant_config.quant_algo == QuantAlgo.W4A16_MXFP4
    assert quant_config.group_size == 32


# Module names and config groups shaped like unsloth/Qwen3.8-27B-NVFP4, whose
# text decoder interleaves GDN ("linear_attn") and full-attention blocks. The
# real checkpoint has 64 blocks with the FP8 MLP tail at 56-63; here it has 4
# with the tail at 2-3.
_QWEN38_FP8_MLP_LAYERS = (2, 3)

_QWEN38_MODULE_NAMES = [
    # GDN block: in_proj_a/in_proj_b stay bf16, the rest is FP8.
    "model.language_model.layers.0.linear_attn",
    "model.language_model.layers.0.linear_attn.in_proj_a",
    "model.language_model.layers.0.linear_attn.in_proj_b",
    "model.language_model.layers.0.linear_attn.in_proj_qkv",
    "model.language_model.layers.0.linear_attn.in_proj_z",
    "model.language_model.layers.0.linear_attn.out_proj",
    "model.language_model.layers.0.input_layernorm",
    "model.language_model.layers.0.mlp.gate_proj",
    "model.language_model.layers.0.mlp.up_proj",
    "model.language_model.layers.0.mlp.down_proj",
    # Full-attention block.
    "model.language_model.layers.1.self_attn.q_proj",
    "model.language_model.layers.1.self_attn.k_proj",
    "model.language_model.layers.1.self_attn.v_proj",
    "model.language_model.layers.1.self_attn.o_proj",
    "model.language_model.layers.1.mlp.gate_proj",
    "model.language_model.layers.1.mlp.down_proj",
    # FP8 MLP tail.
    "model.language_model.layers.2.mlp.gate_proj",
    "model.language_model.layers.2.mlp.down_proj",
    "model.language_model.layers.3.mlp.gate_proj",
    "model.language_model.layers.3.mlp.down_proj",
    # Never quantized by this recipe.
    "model.visual.blocks.0.attn.qkv",
    "mtp.layers.0.mlp.down_proj",
    "lm_head",
]

_QWEN38_IGNORE = [
    # Non-recursive in compressed-tensors: the GDN module itself is unquantized
    # but its in_proj_qkv/in_proj_z/out_proj children are FP8.
    "model.language_model.layers.0.linear_attn",
    "model.language_model.layers.0.linear_attn.in_proj_a",
    "model.language_model.layers.0.linear_attn.in_proj_b",
    "model.visual.blocks.0.attn.qkv",
    "re:^mtp.*",
]


def _qwen38_dense_config(**overrides):
    """A compressed-tensors config shaped like unsloth/Qwen3.8-27B-NVFP4.

    ``group_0`` is FP8 per-channel/per-token (attention, GDN projections,
    ``lm_head`` and the MLP tail); ``group_1`` is NVFP4 (every other MLP).
    Both groups target the tail MLPs, so the checkpoint only loads correctly
    if target precedence resolves those to ``group_0``.
    """
    fp8_mlp_layers = "|".join(str(layer) for layer in _QWEN38_FP8_MLP_LAYERS)
    config = {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "kv_cache_scheme": {
            "num_bits": 8,
            "type": "float",
            "strategy": "tensor",
        },
        "config_groups": {
            "group_0": {
                "format": "float-quantized",
                "targets": [
                    r"re:.*self_attn\.(q|k|v|o)_proj$",
                    r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
                    r"re:.*lm_head",
                    rf"re:.*layers\.({fp8_mlp_layers})\.mlp\.(gate|up|down)_proj$",
                ],
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "channel",
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "token",
                },
            },
            "group_1": {
                "format": "nvfp4-pack-quantized",
                "targets": [r"re:.*mlp\.(gate|up|down)_proj$"],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "tensor_group",
                    "group_size": 16,
                },
                "input_activations": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "tensor_group",
                    "group_size": 16,
                },
            },
        },
        "ignore": list(_QWEN38_IGNORE),
    }
    config.update(overrides)
    return config


def test_update_quant_config_from_compressed_tensors_qwen38_dense_mixed_precision():
    quant_config = QuantConfig()
    layer_quant_configs = update_quant_config_from_compressed_tensors(
        quant_config, _qwen38_dense_config(), _QWEN38_MODULE_NAMES
    )

    assert quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
    assert quant_config.kv_cache_quant_algo == QuantAlgo.FP8

    algos = {name: cfg.quant_algo for name, cfg in layer_quant_configs.items()}
    prefix = "model.language_model.layers"
    expected = {
        # Early MLP blocks: W4A4 NVFP4 (only group_1 matches).
        f"{prefix}.0.mlp.gate_proj": QuantAlgo.NVFP4,
        f"{prefix}.0.mlp.up_proj": QuantAlgo.NVFP4,
        f"{prefix}.0.mlp.down_proj": QuantAlgo.NVFP4,
        f"{prefix}.1.mlp.gate_proj": QuantAlgo.NVFP4,
        f"{prefix}.1.mlp.down_proj": QuantAlgo.NVFP4,
        # Tail MLP blocks: matched by both groups, group_0 wins on precedence.
        f"{prefix}.2.mlp.gate_proj": QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN,
        f"{prefix}.2.mlp.down_proj": QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN,
        f"{prefix}.3.mlp.gate_proj": QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN,
        f"{prefix}.3.mlp.down_proj": QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN,
        # Attention, GDN projections and lm_head: FP8.
        f"{prefix}.1.self_attn.q_proj": QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN,
        f"{prefix}.1.self_attn.k_proj": QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN,
        f"{prefix}.1.self_attn.v_proj": QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN,
        f"{prefix}.1.self_attn.o_proj": QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN,
        f"{prefix}.0.linear_attn.in_proj_qkv": QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN,
        f"{prefix}.0.linear_attn.in_proj_z": QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN,
        f"{prefix}.0.linear_attn.out_proj": QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN,
        "lm_head": QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN,
    }
    assert algos == expected

    # NVFP4 entries keep their group size; every entry carries the global KV
    # cache algo, matching the modelopt MIXED_PRECISION path.
    assert layer_quant_configs[f"{prefix}.0.mlp.down_proj"].group_size == 16
    assert all(cfg.kv_cache_quant_algo == QuantAlgo.FP8 for cfg in layer_quant_configs.values())


def test_update_quant_config_from_compressed_tensors_mixed_precision_skips_ignored_modules():
    quant_config = QuantConfig()
    layer_quant_configs = update_quant_config_from_compressed_tensors(
        quant_config, _qwen38_dense_config(), _QWEN38_MODULE_NAMES
    )

    prefix = "model.language_model.layers"
    # Producer-ignored modules, and modules no config group targets, stay out
    # of the mapping: they inherit the global MIXED_PRECISION config.
    for name in (
        f"{prefix}.0.linear_attn",
        f"{prefix}.0.linear_attn.in_proj_a",
        f"{prefix}.0.linear_attn.in_proj_b",
        f"{prefix}.0.input_layernorm",
        "model.visual.blocks.0.attn.qkv",
        "mtp.layers.0.mlp.down_proj",
    ):
        assert name not in layer_quant_configs


def test_update_quant_config_from_compressed_tensors_mixed_ignore_stays_in_layer_map():
    quant_config = QuantConfig()
    layer_quant_configs = update_quant_config_from_compressed_tensors(
        quant_config, _qwen38_dense_config(), _QWEN38_MODULE_NAMES
    )

    prefix = "model.language_model.layers"
    # compressed-tensors ignore entries are non-recursive and are applied
    # while constructing the authoritative layer map. Copying them into
    # TRT-LLM's recursive exclude list would shadow quantized children.
    assert quant_config.exclude_modules == []
    assert f"{prefix}.0.linear_attn" not in layer_quant_configs
    assert f"{prefix}.0.linear_attn.in_proj_a" not in layer_quant_configs
    assert f"{prefix}.0.linear_attn.out_proj" in layer_quant_configs

    # No module with a per-layer config may be excluded by the final config.
    assert not [
        name
        for name in layer_quant_configs
        if quant_config.is_module_excluded_from_quantization(name)
    ]


def test_update_quant_config_from_compressed_tensors_keeps_modules_to_not_convert():
    # modules_to_not_convert is written in TRT-LLM's pattern language, so it
    # keeps TRT-LLM's recursive semantics and must survive the ignore-entry
    # filtering even when it shadows a module a config group targets.
    prefix = "model.language_model.layers"
    quant_config = QuantConfig()
    update_quant_config_from_compressed_tensors(
        quant_config,
        _qwen38_dense_config(modules_to_not_convert=[f"{prefix}.1.self_attn"]),
        _QWEN38_MODULE_NAMES,
    )

    assert f"{prefix}.1.self_attn" in quant_config.exclude_modules
    assert quant_config.is_module_excluded_from_quantization(f"{prefix}.1.self_attn.q_proj")


def test_update_quant_config_from_compressed_tensors_mixed_precision_without_module_names():
    # The global algo is resolvable without the checkpoint's module list; the
    # per-layer configs are not.
    quant_config = QuantConfig()
    layer_quant_configs = update_quant_config_from_compressed_tensors(
        quant_config, _qwen38_dense_config()
    )

    assert layer_quant_configs is None
    assert quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
    assert quant_config.kv_cache_quant_algo == QuantAlgo.FP8
    assert quant_config.exclude_modules == []


def test_update_quant_config_from_compressed_tensors_exact_target_beats_regex_target():
    # compressed-tensors orders exact-name targets before "re:" targets, so an
    # exact target wins even when a regex target also matches.
    config = _qwen38_dense_config()
    config["config_groups"]["group_0"]["targets"] = ["model.layers.0.mlp.down_proj"]
    config["config_groups"]["group_1"]["targets"] = [r"re:.*mlp\.(gate|up|down)_proj$"]
    config["ignore"] = []

    layer_quant_configs = update_quant_config_from_compressed_tensors(
        QuantConfig(),
        config,
        ["model.layers.0.mlp.down_proj", "model.layers.1.mlp.down_proj"],
    )

    assert layer_quant_configs["model.layers.0.mlp.down_proj"].quant_algo == (
        QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN
    )
    assert layer_quant_configs["model.layers.1.mlp.down_proj"].quant_algo == QuantAlgo.NVFP4


def test_update_quant_config_from_compressed_tensors_mixed_precision_requires_targets():
    config = _qwen38_dense_config()
    del config["config_groups"]["group_1"]["targets"]

    with pytest.raises(ValueError, match="has no 'targets'"):
        update_quant_config_from_compressed_tensors(QuantConfig(), config, _QWEN38_MODULE_NAMES)


def test_update_quant_config_from_compressed_tensors_rejects_any_unmatched_target():
    # A class-name target cannot be resolved from tensor names. It must fail
    # even when another target produces a non-empty layer map.
    config = _qwen38_dense_config()
    config["config_groups"]["group_0"]["targets"].append("Linear")
    with pytest.raises(ValueError, match=r"matched no checkpoint module: \['Linear'\]"):
        update_quant_config_from_compressed_tensors(QuantConfig(), config, _QWEN38_MODULE_NAMES)


def test_update_quant_config_from_compressed_tensors_copies_per_module_configs():
    layer_quant_configs = update_quant_config_from_compressed_tensors(
        QuantConfig(), _qwen38_dense_config(), _QWEN38_MODULE_NAMES
    )

    assert len({id(config) for config in layer_quant_configs.values()}) == len(layer_quant_configs)


def test_update_quant_config_from_compressed_tensors_rejects_empty_config_groups():
    with pytest.raises(ValueError, match="config_groups is empty"):
        update_quant_config_from_compressed_tensors(
            QuantConfig(),
            {
                "quant_method": "compressed-tensors",
                "config_groups": {},
            },
        )


def test_read_checkpoint_module_names_from_local_index(tmp_path):
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.mlp.gate_proj.weight_packed": "model-1.safetensors",
                    "model.layers.0.mlp.gate_proj.weight_scale": "model-1.safetensors",
                    "lm_head.weight": "model-2.safetensors",
                }
            }
        )
    )

    assert ModelConfig._read_checkpoint_module_names(str(tmp_path)) == [
        "model.layers.0.mlp.gate_proj",
        "lm_head",
    ]


def test_read_checkpoint_module_names_resolves_hub_index(tmp_path):
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"weight_map": {"model.layers.0.self_attn.q_proj.weight": "model.safetensors"}})
    )

    with patch(
        "tensorrt_llm._torch.model_config.transformers.utils.hub.cached_file",
        return_value=str(index_path),
    ) as cached_file:
        module_names = ModelConfig._read_checkpoint_module_names("org/model")

    cached_file.assert_called_once_with("org/model", "model.safetensors.index.json")
    assert module_names == ["model.layers.0.self_attn.q_proj"]


def test_model_config_builds_layer_map_for_compressed_tensors(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    f"{module_name}.weight": "model.safetensors"
                    for module_name in _QWEN38_MODULE_NAMES
                }
            }
        )
    )

    quant_config, layer_quant_configs = ModelConfig.load_hf_quant_config(
        _qwen38_dense_config(), "AUTO", checkpoint_dir=str(tmp_path)
    )

    assert quant_config.quant_algo == QuantAlgo.MIXED_PRECISION
    assert layer_quant_configs["lm_head"].quant_algo == QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN
