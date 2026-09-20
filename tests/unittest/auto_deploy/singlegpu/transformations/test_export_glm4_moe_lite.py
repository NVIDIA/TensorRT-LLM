# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for GLM4 MoE Lite export with reduced experts.

Split from test_export.py because these tests require TRT-LLM custom ops
(noaux_tc_op) and cannot run in standalone mode.
"""

import pytest
import torch
import torch.nn as nn
from torch.fx import GraphModule

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_glm4_moe_lite import (
    Glm4MoeLiteConfig,
    Glm4MoeLiteForCausalLM,
)


def _make_tiny_glm4_config(n_routed_experts: int = 8) -> Glm4MoeLiteConfig:
    """Create a minimal ``Glm4MoeLiteConfig`` suitable for unit tests."""
    return Glm4MoeLiteConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=16,
        n_routed_experts=n_routed_experts,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        first_k_dense_replace=1,
        max_position_embeddings=128,
        rope_scaling=None,
        pad_token_id=0,
    )


def _count_moe_experts_in_graph(gm: GraphModule) -> int:
    """Return the number of experts in the first ``torch_moe`` call in *gm*."""
    for node in gm.graph.nodes:
        if node.op == "call_function" and "torch_moe" in str(node.target):
            return len(node.args[3])  # w1_weight list length
    return 0


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GLM4 MoE Lite requires CUDA (uses noaux_tc_op)"
)
@pytest.mark.parametrize("n_routed_experts", [8, 16])
@pytest.mark.parametrize("num_moe_experts_for_export", [2])
def test_glm4_moe_lite_export_with_reduced_experts(n_routed_experts, num_moe_experts_for_export):
    """Export a tiny Glm4MoeLiteForCausalLM with reduced experts and verify correctness."""
    device = "cuda"
    config = _make_tiny_glm4_config(n_routed_experts=n_routed_experts)
    model = Glm4MoeLiteForCausalLM(config).to(device)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (1, 8), device=device)
    position_ids = torch.arange(8, device=device).unsqueeze(0)
    sample_kwargs = {"input_ids": input_ids, "position_ids": position_ids}

    gm_full = torch_export_to_gm(model, kwargs=sample_kwargs)

    gm_reduced = torch_export_to_gm(
        model,
        kwargs=sample_kwargs,
        num_moe_experts_for_export=num_moe_experts_for_export,
    )

    assert _count_moe_experts_in_graph(gm_full) == n_routed_experts
    assert _count_moe_experts_in_graph(gm_reduced) == n_routed_experts

    full_keys = set(gm_full.state_dict().keys())
    reduced_keys = set(gm_reduced.state_dict().keys())
    assert full_keys == reduced_keys, (
        f"State-dict key mismatch.\n"
        f"  Only in full: {full_keys - reduced_keys}\n"
        f"  Only in reduced: {reduced_keys - full_keys}"
    )

    gm_reduced.load_state_dict(model.state_dict(), strict=False)

    for name, mod in model.named_modules():
        if hasattr(mod, "experts") and isinstance(mod.experts, nn.ModuleList):
            assert len(mod.experts) == n_routed_experts, (
                f"Expert list in '{name}' was not restored to {n_routed_experts}"
            )
