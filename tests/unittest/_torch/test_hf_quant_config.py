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


# Module names chosen so that their order differs from any string-hash order
# with high probability; a merge through a set would reorder them.
_HF_EXCLUDE_MODULES = [
    "model.layers.3.mlp.gate",
    "model.layers.0.self_attn.q_proj",
    "lm_head",
    "model.layers.11.mlp.shared_expert_gate",
    "model.layers.2.linear_attn.in_proj_ba",
]


def _compressed_tensors_fp8_block_config(**overrides):
    config = {
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "weights": {"num_bits": 8, "type": "float", "strategy": "block"},
                "input_activations": {"strategy": "group", "group_size": 128},
            },
        },
    }
    config.update(overrides)
    return config


def _compressed_tensors_mxfp4_config(**overrides):
    config = {
        "quant_method": "compressed-tensors",
        "format": "mxfp4-pack-quantized",
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "group",
                    "group_size": 32,
                },
                "input_activations": None,
            },
        },
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    "hf_quant_config, appended_modules",
    [
        pytest.param(
            {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "weight_block_size": [128, 128],
                "activation_scheme": "dynamic",
                "modules_to_not_convert": _HF_EXCLUDE_MODULES + ["lm_head"],
            },
            ["*kv_b_proj*", "*k_b_proj*", "*eh_proj"],
            id="fp8-block-scales",
        ),
        pytest.param(
            {
                "quant_method": "mxfp4",
                "modules_to_not_convert": _HF_EXCLUDE_MODULES + ["lm_head"],
            },
            [
                "block.*.attn.out",
                "block.*.mlp.gate",
                "block.*.attn.qkv",
                "embedding",
                "unembedding",
            ],
            id="mxfp4",
        ),
        pytest.param(
            _compressed_tensors_fp8_block_config(
                modules_to_not_convert=_HF_EXCLUDE_MODULES,
                ignore=["lm_head", "model.embed_tokens"],
            ),
            ["model.embed_tokens"],
            id="compressed-tensors-fp8-block",
        ),
        pytest.param(
            _compressed_tensors_mxfp4_config(
                modules_to_not_convert=_HF_EXCLUDE_MODULES,
                ignore=["lm_head", "model.embed_tokens"],
            ),
            ["model.embed_tokens"],
            id="compressed-tensors-mxfp4",
        ),
    ],
)
def test_load_hf_quant_config_keeps_exclude_modules_in_checkpoint_order(
    hf_quant_config: dict, appended_modules: list[str]
) -> None:
    """The merged list feeds identity fingerprints, so its order must not vary."""
    quant_config, _ = ModelConfig.load_hf_quant_config(hf_quant_config, moe_backend="CUTLASS")

    assert quant_config.exclude_modules == _HF_EXCLUDE_MODULES + appended_modules


def test_dtypes_json_quant_config_keeps_exclude_modules_in_checkpoint_order(tmp_path) -> None:
    dtypes = {f"{name}.weight": "BF16" for name in _HF_EXCLUDE_MODULES}
    dtypes["model.layers.0.mlp.experts.weight.blocks"] = "MXFP4"
    dtypes["model.layers.0.mlp.experts.weight.scales"] = "UE8"
    dtypes_json_file = tmp_path / "dtypes.json"
    dtypes_json_file.write_text(json.dumps(dtypes), encoding="utf-8")

    quant_config, _ = ModelConfig.load_quant_config_from_dtypes_json(
        str(dtypes_json_file), moe_backend="CUTLASS"
    )

    assert quant_config.exclude_modules == _HF_EXCLUDE_MODULES


def _write_inkling2_checkpoint(path, hf_quant_config) -> None:
    # Inkling 2's layout: a ModelOpt block inline in config.json, and the per-tensor
    # precisions in an hf_quant_config.json of its own format.
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Inkling2ForConditionalGeneration"],
                "model_type": "inkling2",
                "text_config": {"num_hidden_layers": 2, "local_layer_ids": [0]},
                "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
            }
        )
    )
    (path / "hf_quant_config.json").write_text(json.dumps(hf_quant_config))


def test_foreign_hf_quant_config_leaves_quantization_to_the_model(tmp_path) -> None:
    _write_inkling2_checkpoint(
        tmp_path, {"quant_method": "inkling2", "model": {"parameter_groups": []}}
    )

    config = ModelConfig.from_pretrained(str(tmp_path))

    # Neither the file nor the inline ModelOpt block configures module-wide quantization.
    assert config.quant_config.quant_algo is None
    assert config.quant_config_dict is None


def test_llm_api_loader_leaves_a_foreign_hf_quant_config_to_the_model(tmp_path) -> None:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
    from tensorrt_llm.llmapi.llm_utils import ModelLoader

    _write_inkling2_checkpoint(
        tmp_path, {"quant_method": "inkling2", "model": {"parameter_groups": []}}
    )
    llm_args = TorchLlmArgs(model=str(tmp_path))

    # Not pre-quantized in the ModelOpt sense: no module-wide algorithm, from the file or
    # from the inline config.json block.
    assert ModelLoader(llm_args)._update_from_hf_quant_config() is False
    assert llm_args.quant_config.quant_algo is None


@pytest.mark.parametrize("hf_quant_config", [{"model": {}}, ["inkling2"]])
def test_unrecognized_hf_quant_config_still_fails(tmp_path, hf_quant_config) -> None:
    # Without a quant_method, or not a mapping at all, the file is not recognized as a
    # foreign format and the ModelOpt parser reports it as before.
    _write_inkling2_checkpoint(tmp_path, hf_quant_config)

    with pytest.raises(ValueError):
        ModelConfig.from_pretrained(str(tmp_path))
