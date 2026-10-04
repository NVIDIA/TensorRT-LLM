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
import torch

from tensorrt_llm._torch.configs.kolibri import Kolibri1Config


@pytest.fixture
def tiny_kolibri_config():
    """Create a minimal Kolibri1Config for testing."""
    return Kolibri1Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=5,  # 4 SWA + 1 Full Attention
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=64,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=64,
    )


def test_kolibri_config_schedule(tiny_kolibri_config):
    """Test layer_types alternating schedule."""
    assert len(tiny_kolibri_config.layer_types) == 5
    assert tiny_kolibri_config.layer_types[:4] == ["sliding_attention"] * 4
    assert tiny_kolibri_config.layer_types[4] == "full_attention"


def test_kolibri_routing_method():
    """Test Kolibri 1 routing method: biased selection, unbiased sigmoid weights."""
    from tensorrt_llm._torch.models.modeling_kolibri import Kolibri1RoutingMethod

    num_experts = 8
    top_k = 2

    # Biases that flip selection: expert 0 has low logit but huge bias
    bias = torch.zeros(num_experts, dtype=torch.float32)
    bias[0] = 100.0

    routing = Kolibri1RoutingMethod(
        top_k=top_k,
        num_experts=num_experts,
        e_score_correction_bias=bias,
        norm_topk_prob=False,
    )

    logits = torch.tensor([[1.0, 5.0, 4.0, 2.0, 0.0, -1.0, -2.0, -3.0]])

    topk_ids, topk_weights = routing.apply(logits)

    # Expert 0 should be selected due to bias, even though its raw logit was 1.0 vs 5.0 and 4.0
    selected = set(topk_ids[0].tolist())
    assert 0 in selected
    assert 1 in selected  # 5.0 + 0 is second highest

    # Weights must be sigmoid(raw logits), NOT sigmoid(logits + bias)
    for idx, weight in zip(topk_ids[0].tolist(), topk_weights[0].tolist()):
        expected_weight = torch.sigmoid(logits[0, idx]).item()
        assert abs(weight - expected_weight) < 1e-5


def test_kolibri_weight_mapper():
    """Test Kolibri 1 weight mapper key remapping."""
    from tensorrt_llm._torch.models.checkpoints.hf.kolibri_weight_mapper import (
        Kolibri1HfWeightMapper,
    )

    mapper = Kolibri1HfWeightMapper()
    raw_weights = {
        "model.layers.0.moe.router.expert_bias": torch.zeros(384),
        "model.layers.1.mlp.expert_bias": torch.zeros(384),
        "model.layers.2.mlp.gate.expert_bias": torch.zeros(384),
        "model.layers.0.shared_experts.gate_proj.weight": torch.zeros(512, 2560),
    }

    remapped = mapper.preprocess_weights(raw_weights)

    assert "model.layers.0.mlp.gate.e_score_correction_bias" in remapped
    assert "model.layers.1.mlp.gate.e_score_correction_bias" in remapped
    assert "model.layers.2.mlp.gate.e_score_correction_bias" in remapped
    assert "model.layers.0.mlp.shared_experts.gate_proj.weight" in remapped
