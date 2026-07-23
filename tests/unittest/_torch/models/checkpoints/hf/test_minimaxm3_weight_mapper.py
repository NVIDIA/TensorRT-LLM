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

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.auto_mapper import AutoCheckpointMapper
from tensorrt_llm._torch.models.checkpoints.hf.minimaxm3_weight_mapper import (
    MiniMaxM3HfWeightMapper,
)
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_minimaxm3 import (
    MiniMaxM3ForCausalLM,
    MiniMaxM3VLForConditionalGeneration,
)
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

_NUM_KV_HEADS = 4
_ROWS_PER_HEAD = 2


def _make_mapper(tp_size: int = 8, num_kv_heads: int = _NUM_KV_HEADS) -> MiniMaxM3HfWeightMapper:
    config = SimpleNamespace(num_key_value_heads=num_kv_heads, num_attention_heads=8)
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=Mapping(world_size=tp_size, rank=0, tp_size=tp_size),
    )
    model = SimpleNamespace(model_config=model_config, config=config)
    mapper = MiniMaxM3HfWeightMapper()
    mapper.init_model_and_config(model, model_config)
    return mapper


def _duplicate_heads(tensor: torch.Tensor, repetitions: int) -> torch.Tensor:
    return (
        tensor.reshape(_NUM_KV_HEADS, _ROWS_PER_HEAD, -1)
        .repeat_interleave(repetitions, dim=0)
        .reshape(_NUM_KV_HEADS * repetitions * _ROWS_PER_HEAD, -1)
    )


@pytest.mark.parametrize(
    "architecture",
    [
        "MiniMaxM3SparseForCausalLM",
        "MiniMaxM3SparseForConditionalGeneration",
    ],
)
def test_mapper_registration_and_mx_fallback(architecture: str) -> None:
    assert isinstance(AutoCheckpointMapper.get("HF", architecture), MiniMaxM3HfWeightMapper)
    assert isinstance(AutoCheckpointMapper.get("MX", architecture), MiniMaxM3HfWeightMapper)


@pytest.mark.parametrize(
    "model_class",
    [MiniMaxM3ForCausalLM, MiniMaxM3VLForConditionalGeneration],
)
def test_load_weights_exposes_weight_mapper(model_class: type) -> None:
    assert "weight_mapper" in inspect.signature(model_class.load_weights).parameters


@pytest.mark.parametrize("scale_name", ["weight_scale_inv", "weight_scale"])
def test_tp8_mxfp8_duplicates_kv_weight_and_scale_via_callbacks(scale_name: str) -> None:
    mapper = _make_mapper()
    module = SimpleNamespace(quant_config=QuantConfig(quant_algo=QuantAlgo.MXFP8))
    prefix = "model.layers.0.self_attn"

    weights = {}
    sources = {}
    for offset, projection in enumerate(("k_proj", "v_proj")):
        weight = torch.arange(24, dtype=torch.float32).reshape(8, 3) + offset * 100
        scale = torch.arange(16, dtype=torch.uint8).reshape(8, 2) + offset * 32
        weights[f"{prefix}.{projection}.weight"] = weight
        weights[f"{prefix}.{projection}.{scale_name}"] = scale
        sources[projection] = {"weight": weight, scale_name: scale}

    mapped = mapper.apply_callbacks(module, "qkv_proj", prefix.split("."), weights)

    assert mapped[0] == {}
    for projection, projected in zip(("k_proj", "v_proj"), mapped[1:]):
        for name, source in sources[projection].items():
            torch.testing.assert_close(projected[name], _duplicate_heads(source, repetitions=2))


def test_nvfp4_scale_behavior_is_preserved() -> None:
    mapper = _make_mapper()
    module = SimpleNamespace(quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4))
    weight = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    weight_scale = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    scale_inv = torch.arange(16, dtype=torch.uint8).reshape(8, 2)

    mapped = mapper._duplicate_kv_weights(
        module,
        "k_proj",
        {
            "weight": weight,
            "weight_scale": weight_scale,
            "weight_scale_inv": scale_inv,
        },
    )

    torch.testing.assert_close(mapped["weight"], _duplicate_heads(weight, repetitions=2))
    torch.testing.assert_close(
        mapped["weight_scale"], _duplicate_heads(weight_scale, repetitions=2)
    )
    assert mapped["weight_scale_inv"] is scale_inv


def test_quant_config_none_is_guarded() -> None:
    mapper = _make_mapper()
    module = SimpleNamespace(quant_config=None)
    weight = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    scale_inv = torch.arange(16, dtype=torch.uint8).reshape(8, 2)

    mapped = mapper._duplicate_kv_weights(
        module,
        "k_proj",
        {
            "weight": weight,
            "weight_scale_inv": scale_inv,
        },
    )

    torch.testing.assert_close(mapped["weight"], _duplicate_heads(weight, repetitions=2))
    assert mapped["weight_scale_inv"] is scale_inv


def test_kv_scale_is_not_expanded_when_kv_heads_cover_tp() -> None:
    mapper = _make_mapper(tp_size=2)
    module = SimpleNamespace(quant_config=QuantConfig(quant_algo=QuantAlgo.MXFP8))
    scale_inv = torch.arange(16, dtype=torch.uint8).reshape(8, 2)

    mapped = mapper._duplicate_kv_weights(
        module,
        "v_proj",
        {"weight_scale_inv": scale_inv},
    )

    torch.testing.assert_close(mapped["weight_scale_inv"], scale_inv)


def test_gate_bias_params_map() -> None:
    mapper = MiniMaxM3HfWeightMapper()
    source_name = "model.layers.3.block_sparse_moe.e_score_correction_bias"
    target_name = "model.layers.3.block_sparse_moe.gate.e_score_correction_bias"
    bias = torch.arange(4, dtype=torch.float32)
    gate_weight = torch.ones(4, 4)

    renamed = mapper.rename_by_params_map(
        mapper.params_map,
        {
            source_name: bias,
            "model.layers.3.block_sparse_moe.gate.weight": gate_weight,
        },
    )

    assert source_name not in renamed
    assert renamed[target_name] is bias
    assert renamed["model.layers.3.block_sparse_moe.gate.weight"] is gate_weight


def test_load_weights_accepts_base_mapper_without_params_map() -> None:
    config = SimpleNamespace(num_key_value_heads=_NUM_KV_HEADS, num_attention_heads=8)
    model_config = ModelConfig(pretrained_config=config, mapping=Mapping())
    model = object.__new__(MiniMaxM3ForCausalLM)
    torch.nn.Module.__init__(model)
    model.model_config = model_config
    mapper = HfWeightMapper()
    source_name = "model.layers.3.block_sparse_moe.e_score_correction_bias"
    target_name = "model.layers.3.block_sparse_moe.gate.e_score_correction_bias"
    bias = torch.arange(4, dtype=torch.float32)

    with patch.object(DecoderModelForCausalLM, "load_weights") as base_load_weights:
        model.load_weights({source_name: bias}, weight_mapper=mapper)

    call_kwargs = base_load_weights.call_args.kwargs
    assert call_kwargs["weight_mapper"] is mapper
    renamed = mapper.rename_by_params_map(call_kwargs["params_map"], {source_name: bias})
    assert renamed[target_name] is bias
