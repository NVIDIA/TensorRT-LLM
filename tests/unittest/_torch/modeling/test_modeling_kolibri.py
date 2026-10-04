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


@pytest.mark.cpu_only
def test_kolibri_config_schedule(tiny_kolibri_config):
    """Test layer_types alternating schedule."""
    assert len(tiny_kolibri_config.layer_types) == 5
    assert tiny_kolibri_config.layer_types[:4] == ["sliding_attention"] * 4
    assert tiny_kolibri_config.layer_types[4] == "full_attention"


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
def test_kolibri_gate_load_bias():
    """Test Kolibri1Gate loads e_score_correction_bias and integrates with routing."""
    from tensorrt_llm._torch.models.modeling_kolibri import (
        Kolibri1Gate,
        Kolibri1RoutingMethod,
    )

    gate = Kolibri1Gate(in_features=64, out_features=8)
    # Bias that flips selection: expert 0 has low logit but huge bias
    bias = torch.zeros(8, dtype=torch.float32)
    bias[0] = 100.0
    weight = torch.randn(8, 64)

    gate.load_weights([{"weight": weight, "e_score_correction_bias": bias}])
    assert torch.allclose(gate.e_score_correction_bias, bias)

    routing = Kolibri1RoutingMethod(
        top_k=2,
        num_experts=8,
        callable_e_score_correction_bias=lambda: gate.e_score_correction_bias,
    )
    assert torch.allclose(routing.e_score_correction_bias, bias)

    # Verify that the loaded bias actively affects expert selection in apply()
    logits = torch.tensor([[1.0, 5.0, 4.0, 2.0, 0.0, -1.0, -2.0, -3.0]])
    topk_ids, topk_weights = routing.apply(logits)

    selected = set(topk_ids[0].tolist())
    assert 0 in selected
    assert 1 in selected

    # Verify weights are based on uncorrected sigmoid of raw logits
    for idx, weight_val in zip(topk_ids[0].tolist(), topk_weights[0].tolist()):
        expected_weight = torch.sigmoid(logits[0, idx]).item()
        assert abs(weight_val - expected_weight) < 1e-5


@pytest.mark.cpu_only
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_kolibri_e2e_dummy_forward(tiny_kolibri_config):
    """Test full Kolibri 1 model forward pass on GPU with dummy weights (QA / local smoke test)."""
    import json
    import tempfile

    from tensorrt_llm.llmapi import LLM, KvCacheConfig, SamplingParams

    with tempfile.TemporaryDirectory() as tmp_dir:
        config_dict = tiny_kolibri_config.to_dict()
        config_dict["architectures"] = ["Kolibri1ForCausalLM"]
        config_dict["torch_dtype"] = "bfloat16"
        with open(f"{tmp_dir}/config.json", "w") as f:
            json.dump(config_dict, f)

        llm = LLM(
            model=tmp_dir,
            backend="pytorch",
            load_format="dummy",
            skip_tokenizer_init=True,
            kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.1),
        )

        outputs = llm.generate(
            [[1, 2, 3, 4, 5]],
            sampling_params=SamplingParams(
                max_tokens=4, ignore_eos=True, detokenize=False),
        )
        assert len(outputs) == 1
        assert len(outputs[0].outputs[0].token_ids) == 4
