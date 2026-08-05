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

from tensorrt_llm._torch.models.checkpoints.mistral.weight_mapper import MistralWeightMapper


@pytest.fixture
def expected_renames():
    return {
        # Top-level embeddings and output projections
        "tok_embeddings.weight": "model.embed_tokens.weight",
        "output.weight": "lm_head.weight",
        "norm.weight": "model.norm.weight",
        # Per-layer attention projection weights (pixtral_mapping + mistral_llm_mapping)
        "layers.0.attention.wq.weight": "model.layers.0.self_attn.q_proj.weight",
        "layers.0.attention.wk.weight": "model.layers.0.self_attn.k_proj.weight",
        "layers.0.attention.wv.weight": "model.layers.0.self_attn.v_proj.weight",
        "layers.0.attention.wo.weight": "model.layers.0.self_attn.o_proj.weight",
        # Per-layer MLP weights
        "layers.0.feed_forward.w1.weight": "model.layers.0.mlp.gate_proj.weight",
        "layers.0.feed_forward.w2.weight": "model.layers.0.mlp.down_proj.weight",
        "layers.0.feed_forward.w3.weight": "model.layers.0.mlp.up_proj.weight",
        # Layernorms
        "layers.0.attention_norm.weight": "model.layers.0.input_layernorm.weight",
        "layers.0.ffn_norm.weight": "model.layers.0.post_attention_layernorm.weight",
        # Quantization scales: compound key must win over individual token
        "layers.0.attention.kv_fake_quantizer.qscale_act": "model.layers.0.self_attn.kv_scale",
        "layers.0.attention.qscale_act": "model.layers.0.self_attn.input_scale",
        # Unknown keys must pass through unchanged
        "some.unknown.tensor": "some.unknown.tensor",
    }


def test_rename_by_params_map(expected_renames):
    mapper = MistralWeightMapper()
    dummy = torch.tensor(0.0)
    input_weights = {k: dummy for k in expected_renames}

    result = mapper.rename_by_params_map(mapper.mistral_llm_mapping, input_weights)

    mismatches = {k: v for k, v in expected_renames.items() if v not in result}
    assert not mismatches, (
        "Keys not renamed as expected (input -> expected):\n"
        + "\n".join(f"  {k!r} -> {v!r}" for k, v in mismatches.items())
        + f"\nActual keys: {sorted(result.keys())}"
    )
    assert type(result) is dict
