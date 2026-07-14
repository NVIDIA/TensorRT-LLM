# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for multi-stream Q/KV projection parallelism in MLA attention.

The test builds a minimal mock model that mirrors the MLA fork pattern:
a shared input feeds two parallel linear chains (one heavier "Q-like",
one lighter "KV-like") whose outputs are combined with an add.

The transform should:
  1. Detect the fork point (shared input with 2+ linear users).
  2. Identify the lighter KV-like linear (no downstream linear within
     a few hops) vs. the heavier Q-like chain (has a downstream linear).
  3. Move the KV linear onto the auxiliary CUDA stream.
  4. Preserve numerical correctness.
  5. Be compatible with CUDA graph capture & replay.
"""

import pytest
import torch
import torch.nn as nn
from torch.fx import Graph, GraphModule, Node

from tensorrt_llm._torch.auto_deploy.transform.library.multi_stream_attn import (
    _AUX_WORKSPACE_ID,
    _build_aux_stream_all_gather_args,
    _execute_kv_path_in_aux_stream,
    _execute_kv_proj_in_aux_stream,
    _find_kv_proj_linears,
)
from tensorrt_llm._torch.auto_deploy.utils.multi_stream_utils import cuda_stream_manager
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

_TRTLLM_AG_AVAILABLE = getattr(torch.ops.auto_deploy, "trtllm_dist_all_gather", None) is not None
_skip_no_trtllm_ag = pytest.mark.skipif(
    not _TRTLLM_AG_AVAILABLE,
    reason="trtllm_dist_all_gather not registered (standalone mode)",
)

# ---------------------------------------------------------------------------
# Helpers -- mock MLA-like module
# ---------------------------------------------------------------------------


class MockMLABlock(nn.Module):
    """Simplified MLA-like attention block with Q and KV projection chains.

    Q chain (heavier):  q_a_proj -> relu (stand-in for rms_norm) -> q_b_proj
    KV chain (lighter):  kv_a_proj
    Merge: add(q_b_proj_output, kv_a_proj_output)

    The layernorm at the output simulates the inter-layer distance in a real
    transformer (output projection, residual add, layernorm) so that the
    next layer's fork point is beyond the BFS max_depth from this layer's
    KV linear.
    """

    def __init__(self, hidden_dim: int, q_inner_dim: int, kv_out_dim: int):
        super().__init__()
        # Q chain: two linears with a non-linearity in between
        self.q_a_proj = nn.Linear(hidden_dim, q_inner_dim, bias=False)
        self.q_b_proj = nn.Linear(q_inner_dim, kv_out_dim, bias=False)
        # KV chain: single linear
        self.kv_a_proj = nn.Linear(hidden_dim, kv_out_dim, bias=False)
        # Inter-layer distance (layernorm + relu simulate residual + norm)
        self.layernorm = nn.LayerNorm(kv_out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Q chain: q_a_proj -> relu -> q_b_proj
        q = self.q_a_proj(x)
        q = torch.nn.functional.relu(q)
        q = self.q_b_proj(q)
        # KV chain: kv_a_proj
        kv = self.kv_a_proj(x)
        out = q + kv
        # Inter-layer distance to push next layer's linears beyond BFS depth
        return self.layernorm(torch.nn.functional.relu(out))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _build_gm(model, example_input):
    """Export *model* to an FX GraphModule."""
    egm = torch.export.export(model, (example_input,))
    return egm.module()


class _UnfusedMLAWeights(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, hidden: int):
        super().__init__()
        self.q_w = nn.Parameter(torch.randn(q_dim, hidden))
        self.kv_w = nn.Parameter(torch.randn(kv_dim, hidden))


def _set_linear_meta(node: Node, out_features: int) -> None:
    node.meta["val"] = torch.empty(2, out_features)


def _make_unfused_mla_graph(backend: str) -> tuple[GraphModule, Node, Node]:
    """Build a minimal unfused MLA graph: fork -> Q/KV linears + KV all_gather."""
    weights = _UnfusedMLAWeights(q_dim=64, kv_dim=32, hidden=128)
    graph = Graph()
    fork = graph.placeholder("fork")
    fork.meta["val"] = torch.empty(2, 128)

    q_linear = graph.call_function(
        torch.ops.aten.linear.default,
        (fork, weights.q_w),
    )
    _set_linear_meta(q_linear, 64)

    kv_linear = graph.call_function(
        torch.ops.aten.linear.default,
        (fork, weights.kv_w),
    )
    _set_linear_meta(kv_linear, 32)

    if backend == "trtllm":
        kv_ag = graph.call_function(
            torch.ops.auto_deploy.trtllm_dist_all_gather.default,
            (kv_linear, "SYMM_MEM", -1, None),
        )
        ag_op = torch.ops.auto_deploy.trtllm_dist_all_gather
    else:
        kv_ag = graph.call_function(
            torch.ops.auto_deploy.torch_dist_all_gather.default,
            (kv_linear, -1, None),
        )
        ag_op = torch.ops.auto_deploy.torch_dist_all_gather
    kv_ag.meta["val"] = torch.empty(2, 32)

    out = graph.call_function(torch.ops.aten.add.Tensor, (q_linear, kv_ag))
    graph.output((out,))

    return GraphModule(weights, graph), kv_ag, ag_op


@_skip_no_trtllm_ag
def test_build_aux_stream_all_gather_args_trtllm():
    gm, kv_ag, _ = _make_unfused_mla_graph("trtllm")
    new_input = gm.graph.call_function(
        torch.ops.aten.add.Tensor, args=(gm.graph.placeholder("x"),) * 2
    )

    args = _build_aux_stream_all_gather_args(kv_ag, new_input)
    assert args == (new_input, "SYMM_MEM", -1, None, _AUX_WORKSPACE_ID)


def test_build_aux_stream_all_gather_args_torch():
    gm, kv_ag, _ = _make_unfused_mla_graph("torch")
    new_input = gm.graph.call_function(
        torch.ops.aten.add.Tensor, args=(gm.graph.placeholder("y"),) * 2
    )

    args = _build_aux_stream_all_gather_args(kv_ag, new_input)
    assert args == (new_input, -1, None)


@_skip_no_trtllm_ag
def test_pattern0_kv_path_rewrite_trtllm_all_gather_args():
    gm, kv_ag, ag_op = _make_unfused_mla_graph("trtllm")
    kv_ag.args = (kv_ag.args[0], "AUTO", -1, None)

    gm, num_matches = _execute_kv_path_in_aux_stream(gm, world_size=2)
    assert num_matches == 1

    ag_nodes = [n for n in gm.graph.nodes if is_op(n, ag_op)]
    assert len(ag_nodes) == 1
    ag = ag_nodes[0]
    assert ag.args[1] == "AUTO"
    assert ag.args[2] == -1
    assert ag.args[3] is None
    assert ag.args[4] == _AUX_WORKSPACE_ID


def test_pattern0_kv_path_rewrite_torch_all_gather_args():
    gm, kv_ag, ag_op = _make_unfused_mla_graph("torch")
    kv_ag.args = (kv_ag.args[0], -1)

    gm, num_matches = _execute_kv_path_in_aux_stream(gm, world_size=2)
    assert num_matches == 1

    ag_nodes = [n for n in gm.graph.nodes if is_op(n, ag_op)]
    assert len(ag_nodes) == 1
    ag = ag_nodes[0]
    assert len(ag.args) == 2
    assert ag.args[1] == -1


def test_pattern_matching_single_block():
    """The pattern matcher should find exactly one pair for a single MLA block."""
    model = MockMLABlock(128, 64, 128).eval().to("cuda")
    example_input = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example_input)

    pairs = _find_kv_proj_linears(gm)
    assert len(pairs) == 1, f"Expected 1 fork-point pair, got {len(pairs)}"


def test_pattern_matching_multi_block():
    """Multiple layers with sufficient inter-layer distance should all be matched."""
    hidden_dim, q_inner_dim, kv_out_dim = 128, 64, 128
    model = (
        nn.Sequential(
            MockMLABlock(hidden_dim, q_inner_dim, kv_out_dim),
            MockMLABlock(kv_out_dim, q_inner_dim, kv_out_dim),
        )
        .eval()
        .to("cuda")
    )
    example_input = torch.randn(4, hidden_dim, device="cuda")
    gm = _build_gm(model, example_input)

    pairs = _find_kv_proj_linears(gm)
    assert len(pairs) == 2, f"Expected 2 fork-point pairs, got {len(pairs)}"


def test_numerical_correctness():
    """After the transform the GraphModule must produce the same output as the original model."""
    hidden_dim, q_inner_dim, kv_out_dim = 128, 64, 128
    cuda_stream_manager.add_device(torch.cuda.current_device())

    model = MockMLABlock(hidden_dim, q_inner_dim, kv_out_dim).eval().to("cuda")
    example_input = torch.randn(4, hidden_dim, device="cuda")
    gm = _build_gm(model, example_input)

    test_x = torch.randn(4, hidden_dim, device="cuda")
    ref_output = model(test_x)

    gm, num_replaced = _execute_kv_proj_in_aux_stream(gm)

    assert num_replaced == 1, f"Expected 1 replacement, got {num_replaced}"

    y = gm(test_x)
    assert torch.allclose(y, ref_output, atol=1e-5), (
        f"Output mismatch: max diff = {(y - ref_output).abs().max().item()}"
    )


def test_numerical_correctness_multi_block():
    """Multi-block correctness test."""
    hidden_dim, q_inner_dim, kv_out_dim = 128, 64, 128
    cuda_stream_manager.add_device(torch.cuda.current_device())

    model = (
        nn.Sequential(
            MockMLABlock(hidden_dim, q_inner_dim, kv_out_dim),
            MockMLABlock(kv_out_dim, q_inner_dim, kv_out_dim),
        )
        .eval()
        .to("cuda")
    )
    example_input = torch.randn(4, hidden_dim, device="cuda")
    gm = _build_gm(model, example_input)

    test_x = torch.randn(4, hidden_dim, device="cuda")
    ref_output = model(test_x)

    gm, num_replaced = _execute_kv_proj_in_aux_stream(gm)

    assert num_replaced == 2, f"Expected 2 replacements, got {num_replaced}"

    y = gm(test_x)
    assert torch.allclose(y, ref_output, atol=1e-5), (
        f"Output mismatch: max diff = {(y - ref_output).abs().max().item()}"
    )


def test_cuda_graph_compatibility():
    """The transformed GraphModule must work under CUDA graph capture and replay."""
    hidden_dim, q_inner_dim, kv_out_dim = 128, 64, 128
    cuda_stream_manager.add_device(torch.cuda.current_device())

    model = MockMLABlock(hidden_dim, q_inner_dim, kv_out_dim).eval().to("cuda")
    example_input = torch.randn(4, hidden_dim, device="cuda")
    gm = _build_gm(model, example_input)

    test_x = torch.randn(4, hidden_dim, device="cuda")
    ref_output = model(test_x)

    gm, num_replaced = _execute_kv_proj_in_aux_stream(gm)
    assert num_replaced == 1

    # Allocate static buffers for CUDA graph capture.
    static_x = torch.randn(4, hidden_dim, device="cuda")
    static_output = torch.randn(4, kv_out_dim, device="cuda")

    # Warm up (required before capture).
    for _ in range(3):
        static_output.copy_(gm(static_x))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output.copy_(gm(static_x))

    static_x.copy_(test_x)
    graph.replay()

    assert torch.allclose(static_output, ref_output, atol=1e-5), (
        f"CUDA graph output mismatch: max diff = {(static_output - ref_output).abs().max().item()}"
    )


def test_no_match_on_single_linear():
    """A node with only one linear user should not be matched."""

    class SingleLinear(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            return self.fc(x)

    model = SingleLinear(64).eval().to("cuda")
    example_input = torch.randn(4, 64, device="cuda")
    gm = _build_gm(model, example_input)

    pairs = _find_kv_proj_linears(gm)
    assert len(pairs) == 0, f"Expected 0 matches, got {len(pairs)}"


def test_no_match_when_both_have_downstream_linear():
    """When *both* branches have downstream linears the pattern should not match."""

    class BothHeavy(nn.Module):
        def __init__(self, dim, inner):
            super().__init__()
            self.fc_a1 = nn.Linear(dim, inner, bias=False)
            self.fc_a2 = nn.Linear(inner, dim, bias=False)
            self.fc_b1 = nn.Linear(dim, inner, bias=False)
            self.fc_b2 = nn.Linear(inner, dim, bias=False)

        def forward(self, x):
            a = self.fc_a2(torch.relu(self.fc_a1(x)))
            b = self.fc_b2(torch.relu(self.fc_b1(x)))
            return a + b

    model = BothHeavy(64, 32).eval().to("cuda")
    example_input = torch.randn(4, 64, device="cuda")
    gm = _build_gm(model, example_input)

    pairs = _find_kv_proj_linears(gm)
    assert len(pairs) == 0, f"Expected 0 matches, got {len(pairs)}"
