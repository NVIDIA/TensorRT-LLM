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
        hf_quant_config, moe_backend="CUTLASS")

    assert layer_quant_config is None
    assert quant_config.quant_algo == QuantAlgo.NVFP4
    assert quant_config.group_size == 16
    assert quant_config.kv_cache_quant_algo == QuantAlgo.FP8
    assert set(quant_config.exclude_modules) == {gate_exclude, "lm_head"}


@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("checkpoint_format", ["gptq", "gptq_v2"])
def test_load_hf_gptq_config(group_size, checkpoint_format):
    config, per_layer = ModelConfig.load_hf_quant_config(
        {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": group_size,
            "checkpoint_format": checkpoint_format,
            "desc_act": False,
            "modules_to_not_convert": ["model.layers.0.self_attn.o_proj"],
        },
        moe_backend="CUTLASS",
    )
    assert config.quant_algo == QuantAlgo.W4A16_GPTQ
    assert config.group_size == group_size
    assert config.has_zero_point
    assert config.exclude_modules == [
        "model.layers.0.self_attn.o_proj", "lm_head"
    ]
    assert per_layer is None


@pytest.mark.parametrize(
    "override,error",
    [
        ({
            "bits": 8
        }, "bits=4"),
        ({
            "group_size": 32
        }, "group_size"),
        ({
            "desc_act": True
        }, "desc_act"),
        ({
            "checkpoint_format": "marlin"
        }, "checkpoint_format"),
        ({
            "dynamic": {
                ".*": {
                    "bits": 8
                }
            }
        }, "dynamic"),
        ({
            "modules_in_block_to_quantize": [["q_proj"]]
        }, "modules_in_block"),
        ({
            "lm_head": True
        }, "lm_head"),
    ],
)
def test_load_hf_gptq_config_rejects_unsupported(override, error):
    config = {"quant_method": "gptq", "bits": 4, "group_size": 128}
    config.update(override)
    with pytest.raises(ValueError, match=error):
        ModelConfig.load_hf_quant_config(config, moe_backend="CUTLASS")


@pytest.mark.parametrize("from_model_kwargs", [False, True])
@pytest.mark.parametrize("kv_dtype", ["auto", "fp8"])
def test_llm_api_loads_gptq_config(tmp_path, from_model_kwargs, kv_dtype):
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, TorchLlmArgs
    from tensorrt_llm.llmapi.llm_utils import ModelLoader

    hf_config = {
        "quant_method": "gptq",
        "bits": 4,
        "group_size": 128,
        "desc_act": False
    }
    kwargs = {"quantization_config": hf_config} if from_model_kwargs else None
    if not from_model_kwargs:
        (tmp_path / "config.json").write_text(
            json.dumps({"quantization_config": hf_config}))
    args = TorchLlmArgs(
        model=str(tmp_path),
        gpus_per_node=1,
        model_kwargs=kwargs,
        kv_cache_config=KvCacheConfig(dtype=kv_dtype),
    )
    loader = ModelLoader(args)
    assert loader._update_from_hf_quant_config() is True
    assert args.quant_config.quant_algo == QuantAlgo.W4A16_GPTQ
    assert args.quant_config.group_size == 128
    assert args.quant_config.has_zero_point
    assert args.quant_config.exclude_modules == ["lm_head"]
    assert args.quant_config.kv_cache_quant_algo == (QuantAlgo.FP8 if kv_dtype
                                                     == "fp8" else None)


@pytest.mark.parametrize("from_model_kwargs", [False, True])
def test_llm_api_rejects_gptq_activation_order(tmp_path, from_model_kwargs):
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
    from tensorrt_llm.llmapi.llm_utils import ModelLoader

    hf_config = {
        "quant_method": "gptq",
        "bits": 4,
        "group_size": 128,
        "desc_act": True
    }
    kwargs = {"quantization_config": hf_config} if from_model_kwargs else None
    if not from_model_kwargs:
        (tmp_path / "config.json").write_text(
            json.dumps({"quantization_config": hf_config}))
    args = TorchLlmArgs(model=str(tmp_path),
                        gpus_per_node=1,
                        model_kwargs=kwargs)
    with pytest.raises(ValueError, match="desc_act"):
        ModelLoader(args)._update_from_hf_quant_config()
