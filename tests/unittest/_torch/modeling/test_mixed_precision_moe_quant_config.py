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
"""Unit tests for per-layer routed-experts quant-config resolution used to serve MIXED_PRECISION
MoE checkpoints (GLM-4.x, Qwen3-MoE). The fused MoE uses one format for a layer's whole experts
block; ModelOpt exports per-expert keys, so the helper must derive the block format from a
representative routed expert and otherwise fall back to the global config."""
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.models.modeling_glm import Glm4MoE
from tensorrt_llm._torch.models.modeling_qwen3_moe import \
    _get_experts_quant_config as qwen3_get_experts_quant_config
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

GLOBAL = QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION)
NVFP4 = QuantConfig(quant_algo=QuantAlgo.NVFP4)
FP8 = QuantConfig(quant_algo=QuantAlgo.FP8)

# both implementations share identical logic
GETTERS = [Glm4MoE._get_experts_quant_config, qwen3_get_experts_quant_config]


def _model_config(quant_config_dict):
    return SimpleNamespace(quant_config=GLOBAL, quant_config_dict=quant_config_dict)


@pytest.mark.parametrize("get_cfg", GETTERS)
def test_no_per_layer_dict_returns_global(get_cfg):
    # Uniform checkpoints have no per-layer table -> fall back to the global config.
    assert get_cfg(_model_config(None), 3) is GLOBAL


@pytest.mark.parametrize("get_cfg", GETTERS)
def test_bare_experts_key(get_cfg):
    cfg = {"model.layers.3.mlp.experts": NVFP4}
    assert get_cfg(_model_config(cfg), 3) is NVFP4


@pytest.mark.parametrize("get_cfg", GETTERS)
def test_per_expert_keys_fallback(get_cfg):
    # ModelOpt exports per-expert keys, not a bare "...mlp.experts" key; derive from a member.
    cfg = {
        "model.layers.3.mlp.experts.0.gate_proj": FP8,
        "model.layers.3.mlp.experts.0.up_proj": FP8,
        "model.layers.3.mlp.experts.0.down_proj": FP8,
        "model.layers.3.self_attn.q_proj": NVFP4,  # must be ignored (not an expert)
    }
    assert get_cfg(_model_config(cfg), 3) is FP8


@pytest.mark.parametrize("get_cfg", GETTERS)
def test_other_layer_keys_return_global(get_cfg):
    # Only a *different* layer's experts are present -> no match for layer 3 -> global.
    cfg = {"model.layers.5.mlp.experts.0.gate_proj": NVFP4}
    assert get_cfg(_model_config(cfg), 3) is GLOBAL
