# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Graph-level unit tests for the inject_lora transform.

Builds a small FX graph with a few torch_linear_simple ops (mimicking a
single transformer layer), runs inject_lora, and asserts the exact
resulting graph structure.
"""

import pytest
import torch
import torch.nn as nn

from tensorrt_llm._torch.auto_deploy.transform.library.lora import InjectLora, InjectLoraConfig


def _build_small_graph():
    """Build a tiny FX graph mimicking one Llama layer.

    Graph structure (6 ops):
        x → q_proj(x) → q
        x → k_proj(x) → k
        x → v_proj(x) → v
        x → gate_proj(x) → gate
        x → down_proj(gate) → out
        output(q, k, v, out)

    Weight names follow HF convention:
        model.layers.0.self_attn.q_proj.weight
        model.layers.0.self_attn.k_proj.weight
        model.layers.0.self_attn.v_proj.weight
        model.layers.0.mlp.gate_proj.weight
        model.layers.0.mlp.down_proj.weight
    """
    hidden = 16

    class TinyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            # Use nested submodules so weight names match HF convention
            self.model = nn.Module()
            self.model.layers = nn.ModuleList()
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Linear(hidden, hidden, bias=False)
            layer.self_attn.k_proj = nn.Linear(hidden, hidden, bias=False)
            layer.self_attn.v_proj = nn.Linear(hidden, hidden, bias=False)
            layer.mlp = nn.Module()
            layer.mlp.gate_proj = nn.Linear(hidden, hidden, bias=False)
            layer.mlp.down_proj = nn.Linear(hidden, hidden, bias=False)
            self.model.layers.append(layer)

        def forward(self, x):
            layer = self.model.layers[0]
            q = torch.ops.auto_deploy.torch_linear_simple(x, layer.self_attn.q_proj.weight, None)
            k = torch.ops.auto_deploy.torch_linear_simple(x, layer.self_attn.k_proj.weight, None)
            v = torch.ops.auto_deploy.torch_linear_simple(x, layer.self_attn.v_proj.weight, None)
            gate = torch.ops.auto_deploy.torch_linear_simple(x, layer.mlp.gate_proj.weight, None)
            out = torch.ops.auto_deploy.torch_linear_simple(gate, layer.mlp.down_proj.weight, None)
            return q, k, v, out

    model = TinyLayer()
    # Export to FX graph
    x = torch.randn(2, hidden)
    gm = torch.export.export(model, (x,), strict=False).module()
    return gm


def _count_ops(gm, op):
    """Count occurrences of a specific op in the graph."""
    return sum(1 for node in gm.graph.nodes if node.op == "call_function" and node.target == op)


def _get_nodes_by_op(gm, op):
    """Get all nodes matching a specific op."""
    return [node for node in gm.graph.nodes if node.op == "call_function" and node.target == op]


def test_inject_lora_inserts_delta_and_add_for_targets():
    """inject_lora adds lora_delta + add for each target linear, leaves non-targets alone."""
    gm = _build_small_graph()

    # Before: 5 torch_linear_simple, 0 lora_delta, 0 add
    assert _count_ops(gm, torch.ops.auto_deploy.torch_linear_simple.default) == 5
    assert _count_ops(gm, torch.ops.auto_deploy.lora_delta.default) == 0
    assert _count_ops(gm, torch.ops.aten.add.Tensor) == 0

    # Run inject_lora targeting q_proj, k_proj, v_proj only (not gate_proj, down_proj)
    config = InjectLoraConfig(
        stage="pattern_matcher",
        lora_targets=[
            {"hf_module_name": "q_proj", "module_id": 1},
            {"hf_module_name": "k_proj", "module_id": 2},
            {"hf_module_name": "v_proj", "module_id": 3},
        ],
    )
    transform = InjectLora(config)
    gm, info = transform._apply(gm, None, None, None)

    assert not info.skipped
    assert info.num_matches == 3  # q, k, v

    # After: 5 linears still there + 3 lora_delta + 3 add
    assert _count_ops(gm, torch.ops.auto_deploy.torch_linear_simple.default) == 5
    assert _count_ops(gm, torch.ops.auto_deploy.lora_delta.default) == 3
    assert _count_ops(gm, torch.ops.aten.add.Tensor) == 3


def test_inject_lora_delta_nodes_have_correct_args():
    """Each lora_delta node has the correct (layer_id, module_id, output_size)."""
    gm = _build_small_graph()

    config = InjectLoraConfig(
        stage="pattern_matcher",
        lora_targets=[
            {"hf_module_name": "q_proj", "module_id": 1},
            {"hf_module_name": "k_proj", "module_id": 2},
            {"hf_module_name": "v_proj", "module_id": 3},
        ],
    )
    transform = InjectLora(config)
    gm, _ = transform._apply(gm, None, None, None)

    lora_nodes = _get_nodes_by_op(gm, torch.ops.auto_deploy.lora_delta.default)
    assert len(lora_nodes) == 3

    # Check each lora_delta node's compile-time constants
    expected = {
        1: "attn_q",  # module_id=1 → ATTENTION_Q
        2: "attn_k",  # module_id=2 → ATTENTION_K
        3: "attn_v",  # module_id=3 → ATTENTION_V
    }
    for node in lora_nodes:
        _x, _linear_out, layer_id, module_id = node.args
        assert layer_id == 0, f"Expected layer_id=0, got {layer_id}"
        assert module_id in expected, f"Unexpected module_id={module_id}"


def test_inject_lora_add_nodes_tagged():
    """Each add node inserted by inject_lora has is_lora_add=True metadata."""
    gm = _build_small_graph()

    config = InjectLoraConfig(
        stage="pattern_matcher",
        lora_targets=[{"hf_module_name": "q_proj", "module_id": 1}],
    )
    transform = InjectLora(config)
    gm, _ = transform._apply(gm, None, None, None)

    add_nodes = _get_nodes_by_op(gm, torch.ops.aten.add.Tensor)
    assert len(add_nodes) == 1
    assert add_nodes[0].meta.get("is_lora_add") is True


def test_inject_lora_non_target_linears_untouched():
    """Linears not in lora_targets have no lora_delta or add nodes."""
    gm = _build_small_graph()

    # Only target q_proj — gate_proj and down_proj should be untouched
    config = InjectLoraConfig(
        stage="pattern_matcher",
        lora_targets=[{"hf_module_name": "q_proj", "module_id": 1}],
    )
    transform = InjectLora(config)
    gm, info = transform._apply(gm, None, None, None)

    assert info.num_matches == 1

    # gate_proj and down_proj linears should still feed directly to their users
    # (no add node in between)
    lin_nodes = _get_nodes_by_op(gm, torch.ops.auto_deploy.torch_linear_simple.default)
    for node in lin_nodes:
        weight_name = ""
        if hasattr(node.args[1], "target"):
            weight_name = node.args[1].target
        if "gate_proj" in weight_name or "down_proj" in weight_name:
            # This linear's users should NOT include an add node
            for user in node.users:
                assert user.target != torch.ops.aten.add.Tensor, (
                    f"Non-target linear {weight_name} should not have a LoRA add node"
                )


def test_inject_lora_missing_hf_linear_raises():
    """Each LoRA target must map to an HF linear present in the exported graph."""
    gm = _build_small_graph()

    config = InjectLoraConfig(
        stage="pattern_matcher",
        lora_targets=[{"hf_module_name": "qkv_proj", "module_id": 0}],
    )
    transform = InjectLora(config)

    with pytest.raises(ValueError, match="pre-fusion exported graph"):
        transform._apply(gm, None, None, None)


def test_inject_lora_empty_targets_is_noop():
    """Empty lora_targets is a no-op."""
    gm = _build_small_graph()

    config = InjectLoraConfig(
        stage="pattern_matcher",
        lora_targets=[],
    )
    transform = InjectLora(config)
    gm, info = transform._apply(gm, None, None, None)

    assert info.skipped
    assert info.num_matches == 0
    assert _count_ops(gm, torch.ops.auto_deploy.lora_delta.default) == 0
