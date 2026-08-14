# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the generalized multi-stream GEMM parallelization transform.

The transform targets fork points where 2+ fp8 linear ops share the same input
and moves the **largest** (by weight shape) to the auxiliary CUDA stream.

Architecture patterns tested:

  **Two-branch fork** -- Two linears of different sizes sharing one input.
    The larger one should be moved to the aux stream.

  **Three-branch fork (MHA-like)** -- Three linears (q_proj, k_proj, v_proj)
    with q_proj being the largest.

  **Four-branch fork (linear attention-like)** -- Four linears sharing one
    input, with one being significantly larger than the others.

  **Skip already-handled** -- Fork points that already have multi-stream ops
    are skipped to avoid conflicts.

  **Single linear** -- No fork point, should produce zero matches.

  **Equal-weight linears** -- All linears at a fork point have the same weight
    size; one should still be selected deterministically.

Each pattern is tested for:
  1. Pattern matching -- correct number of fork points found.
  2. Largest GEMM identification -- correct linear moved to aux stream.
  3. Graph structure -- ``record_event_passthrough`` and aux nodes present.
  4. Numerical correctness -- output matches eager reference.
  5. CUDA graph compatibility -- capture + replay produces correct output.
  6. Multi-layer stacking -- multiple fork points handled independently.
"""

from typing import Optional

import torch
import torch.nn as nn

from tensorrt_llm._torch.auto_deploy.transform.library.multi_stream_gemm import (
    _estimate_weight_size,
    _find_gemm_fork_points,
    _parallelize_largest_gemm,
)
from tensorrt_llm._torch.auto_deploy.utils.multi_stream_utils import (
    cuda_stream_manager,
    record_event_passthrough,
)

# ---------------------------------------------------------------------------
# Mock fp8 linear custom op -- mimics trtllm_finegrained_fp8_linear signature
# ---------------------------------------------------------------------------


@torch.library.custom_op("auto_deploy::mock_fp8_linear_gemm_test", mutates_args=())
def mock_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Mock fp8 linear: simple matmul standing in for the real kernel."""
    out = input @ weight.to(input.dtype).t()
    if bias is not None:
        out = out + bias
    return out


@mock_fp8_linear.register_fake
def _mock_fp8_linear_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    out_features = weight.shape[0]
    return torch.empty((*input.shape[:-1], out_features), dtype=input.dtype, device=input.device)


_MOCK_OPS = [torch.ops.auto_deploy.mock_fp8_linear_gemm_test]


# ---------------------------------------------------------------------------
# Helper: wrap nn.Linear as our mock fp8 linear in forward
# ---------------------------------------------------------------------------


class MockFP8Linear(nn.Module):
    """Wrapper that calls the mock fp8 linear custom op with real weight tensors."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_scale = nn.Parameter(
            torch.ones(max(1, out_features // 128), max(1, in_features // 128)),
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.auto_deploy.mock_fp8_linear_gemm_test(
            x, self.weight, None, self.weight_scale
        )


# ---------------------------------------------------------------------------
# Mock model architectures
# ---------------------------------------------------------------------------


class TwoBranchFork(nn.Module):
    """Two fp8 linears sharing one input, different weight sizes.

    branch_a: [large_out, hidden] -- larger GEMM
    branch_b: [small_out, hidden] -- smaller GEMM
    Merge: concat -> project back
    """

    def __init__(self, hidden_dim: int, large_out: int, small_out: int):
        super().__init__()
        self.branch_a = MockFP8Linear(hidden_dim, large_out)  # Larger
        self.branch_b = MockFP8Linear(hidden_dim, small_out)  # Smaller
        self.proj = nn.Linear(large_out + small_out, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.branch_a(x)
        b = self.branch_b(x)
        combined = torch.cat([a, b], dim=-1)
        return self.norm(self.proj(combined))


class ThreeBranchFork(nn.Module):
    """Three fp8 linears sharing one input (MHA-like: q, k, v projections).

    q_proj: [large_dim, hidden] -- largest
    k_proj: [small_dim, hidden] -- smaller
    v_proj: [small_dim, hidden] -- smaller
    Merge: sum all outputs (simplified from real attention)
    """

    def __init__(self, hidden_dim: int, large_dim: int, small_dim: int):
        super().__init__()
        self.q_proj = MockFP8Linear(hidden_dim, large_dim)  # Largest
        self.k_proj = MockFP8Linear(hidden_dim, small_dim)
        self.v_proj = MockFP8Linear(hidden_dim, small_dim)
        self.o_proj = nn.Linear(large_dim, hidden_dim, bias=False)
        self.k_down = nn.Linear(small_dim, large_dim, bias=False)
        self.v_down = nn.Linear(small_dim, large_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_down(self.k_proj(x))
        v = self.v_down(self.v_proj(x))
        combined = q + k + v
        return self.norm(self.o_proj(combined))


class FourBranchFork(nn.Module):
    """Four fp8 linears sharing one input (linear attention-like).

    in_proj_qkv: [large_dim, hidden] -- largest
    in_proj_z:   [mid_dim, hidden]
    in_proj_b:   [small_dim, hidden]
    in_proj_a:   [small_dim, hidden]
    Merge: sum all (simplified)
    """

    def __init__(self, hidden_dim: int, large_dim: int, mid_dim: int, small_dim: int):
        super().__init__()
        self.in_proj_qkv = MockFP8Linear(hidden_dim, large_dim)  # Largest
        self.in_proj_z = MockFP8Linear(hidden_dim, mid_dim)
        self.in_proj_b = MockFP8Linear(hidden_dim, small_dim)
        self.in_proj_a = MockFP8Linear(hidden_dim, small_dim)
        # Project all down to hidden_dim for summation
        self.down_qkv = nn.Linear(large_dim, hidden_dim, bias=False)
        self.down_z = nn.Linear(mid_dim, hidden_dim, bias=False)
        self.down_b = nn.Linear(small_dim, hidden_dim, bias=False)
        self.down_a = nn.Linear(small_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.down_qkv(self.in_proj_qkv(x))
        z = self.down_z(self.in_proj_z(x))
        b = self.down_b(self.in_proj_b(x))
        a = self.down_a(self.in_proj_a(x))
        return self.norm(qkv + z + b + a)


class EqualWeightFork(nn.Module):
    """Two fp8 linears with identical weight shapes sharing one input.

    Tests that the transform handles ties deterministically.
    """

    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.branch_a = MockFP8Linear(hidden_dim, out_dim)
        self.branch_b = MockFP8Linear(hidden_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.branch_a(x) + self.branch_b(x))


class SingleLinearModel(nn.Module):
    """Only one fp8 linear -- no fork point, no match expected."""

    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc = MockFP8Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _build_gm(model, example_input):
    """Export *model* to an FX ``GraphModule``."""
    return torch.export.export(model, (example_input,)).module()


def _get_targets(gm):
    """Return the set of ``call_function`` targets in *gm*."""
    return {n.target for n in gm.graph.nodes if n.op == "call_function"}


def _count_aux_ops(gm):
    """Count how many aux-stream variant ops are in the graph."""
    count = 0
    for n in gm.graph.nodes:
        if n.op == "call_function" and hasattr(n.target, "__name__"):
            if n.target.__name__.endswith("_aux"):
                count += 1
    return count


def _assert_numerical_correctness(gm, model, test_x, *, atol=1e-4):
    """Assert that *gm* and *model* produce the same output on *test_x*."""
    ref = model(test_x)
    out = gm(test_x)
    assert torch.allclose(out, ref, atol=atol), (
        f"Output mismatch: max diff = {(out - ref).abs().max().item()}"
    )


def _assert_cuda_graph_correctness(gm, model, test_x, *, atol=1e-4):
    """Assert correctness under CUDA graph capture + replay."""
    ref = model(test_x)
    static_x = torch.randn_like(test_x)
    static_out = torch.empty_like(ref)

    # Warm up
    for _ in range(3):
        static_out.copy_(gm(static_x))

    cuda_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph):
        static_out.copy_(gm(static_x))

    static_x.copy_(test_x)
    cuda_graph.replay()

    assert torch.allclose(static_out, ref, atol=atol), (
        f"CUDA graph output mismatch: max diff = {(static_out - ref).abs().max().item()}"
    )


# ===================================================================
# Tests -- Two-branch fork
# ===================================================================


def test_two_branch_pattern_matching():
    """A fork with two fp8 linears should produce exactly one fork point."""
    model = TwoBranchFork(128, 256, 64).eval().to("cuda")
    example = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example)

    fork_points = _find_gemm_fork_points(gm, _MOCK_OPS)
    assert len(fork_points) == 1, f"Expected 1 fork point, got {len(fork_points)}"
    # Should have 2 linear users at the fork point
    assert len(fork_points[0][1]) == 2


def test_two_branch_largest_moved_to_aux():
    """The larger linear (256 out-features) should be the one on the aux stream."""
    cuda_stream_manager.add_device(torch.cuda.current_device())
    model = TwoBranchFork(128, 256, 64).eval().to("cuda")
    example = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example)

    gm, num = _parallelize_largest_gemm(gm, _MOCK_OPS)

    assert num == 1, f"Expected 1 replacement, got {num}"
    targets = _get_targets(gm)
    assert record_event_passthrough in targets, "record_event_passthrough not in graph"


def test_two_branch_numerical_correctness():
    """Numerical output must match the original model after the transform."""
    cuda_stream_manager.add_device(torch.cuda.current_device())
    model = TwoBranchFork(128, 256, 64).eval().to("cuda")
    example = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example)

    gm, num = _parallelize_largest_gemm(gm, _MOCK_OPS)
    assert num == 1

    _assert_numerical_correctness(gm, model, torch.randn(4, 128, device="cuda"))


def test_two_branch_cuda_graph():
    """CUDA graph capture + replay for two-branch fork."""
    cuda_stream_manager.add_device(torch.cuda.current_device())
    model = TwoBranchFork(128, 256, 64).eval().to("cuda")
    example = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example)

    gm, num = _parallelize_largest_gemm(gm, _MOCK_OPS)
    assert num == 1

    _assert_cuda_graph_correctness(gm, model, torch.randn(4, 128, device="cuda"))


def test_two_branch_multi_layer():
    """Two stacked layers -- both fork points should be transformed."""
    cuda_stream_manager.add_device(torch.cuda.current_device())
    model = (
        nn.Sequential(
            TwoBranchFork(128, 256, 64),
            TwoBranchFork(128, 256, 64),
        )
        .eval()
        .to("cuda")
    )
    example = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example)

    gm, num = _parallelize_largest_gemm(gm, _MOCK_OPS)
    assert num == 2, f"Expected 2 replacements, got {num}"

    _assert_numerical_correctness(gm, model, torch.randn(4, 128, device="cuda"))


# ===================================================================
# Tests -- Three-branch fork (MHA-like)
# ===================================================================


def test_three_branch_pattern_matching():
    """A fork with three fp8 linears should produce exactly one fork point."""
    model = ThreeBranchFork(128, 256, 64).eval().to("cuda")
    example = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example)

    fork_points = _find_gemm_fork_points(gm, _MOCK_OPS)
    assert len(fork_points) == 1, f"Expected 1 fork point, got {len(fork_points)}"
    assert len(fork_points[0][1]) == 3


def test_three_branch_numerical_correctness():
    """Numerical correctness for three-branch fork."""
    cuda_stream_manager.add_device(torch.cuda.current_device())
    model = ThreeBranchFork(128, 256, 64).eval().to("cuda")
    example = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example)

    gm, num = _parallelize_largest_gemm(gm, _MOCK_OPS)
    assert num == 1

    _assert_numerical_correctness(gm, model, torch.randn(4, 128, device="cuda"))


def test_three_branch_cuda_graph():
    """CUDA graph capture + replay for three-branch fork."""
    cuda_stream_manager.add_device(torch.cuda.current_device())
    model = ThreeBranchFork(128, 256, 64).eval().to("cuda")
    example = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example)

    gm, num = _parallelize_largest_gemm(gm, _MOCK_OPS)
    assert num == 1

    _assert_cuda_graph_correctness(gm, model, torch.randn(4, 128, device="cuda"))


# ===================================================================
# Tests -- Four-branch fork (linear attention-like)
# ===================================================================


def test_four_branch_pattern_matching():
    """A fork with four fp8 linears should produce exactly one fork point."""
    model = FourBranchFork(128, 512, 256, 64).eval().to("cuda")
    example = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example)

    fork_points = _find_gemm_fork_points(gm, _MOCK_OPS)
    assert len(fork_points) == 1, f"Expected 1 fork point, got {len(fork_points)}"
    assert len(fork_points[0][1]) == 4


def test_four_branch_numerical_correctness():
    """Numerical correctness for four-branch fork."""
    cuda_stream_manager.add_device(torch.cuda.current_device())
    model = FourBranchFork(128, 512, 256, 64).eval().to("cuda")
    example = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example)

    gm, num = _parallelize_largest_gemm(gm, _MOCK_OPS)
    assert num == 1

    _assert_numerical_correctness(gm, model, torch.randn(4, 128, device="cuda"))


def test_four_branch_cuda_graph():
    """CUDA graph capture + replay for four-branch fork."""
    cuda_stream_manager.add_device(torch.cuda.current_device())
    model = FourBranchFork(128, 512, 256, 64).eval().to("cuda")
    example = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example)

    gm, num = _parallelize_largest_gemm(gm, _MOCK_OPS)
    assert num == 1

    _assert_cuda_graph_correctness(gm, model, torch.randn(4, 128, device="cuda"))


def test_four_branch_multi_layer():
    """Two stacked four-branch layers -- both should be transformed."""
    cuda_stream_manager.add_device(torch.cuda.current_device())
    model = (
        nn.Sequential(
            FourBranchFork(128, 512, 256, 64),
            FourBranchFork(128, 512, 256, 64),
        )
        .eval()
        .to("cuda")
    )
    example = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example)

    gm, num = _parallelize_largest_gemm(gm, _MOCK_OPS)
    assert num == 2, f"Expected 2 replacements, got {num}"

    _assert_numerical_correctness(gm, model, torch.randn(4, 128, device="cuda"))


# ===================================================================
# Tests -- Edge cases
# ===================================================================


def test_single_linear_no_match():
    """A single fp8 linear (no fork) should produce zero matches."""
    model = SingleLinearModel(128, 256).eval().to("cuda")
    example = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example)

    fork_points = _find_gemm_fork_points(gm, _MOCK_OPS)
    assert len(fork_points) == 0, f"Expected 0 fork points, got {len(fork_points)}"

    cuda_stream_manager.add_device(torch.cuda.current_device())
    gm, num = _parallelize_largest_gemm(gm, _MOCK_OPS)
    assert num == 0

    # Graph should NOT contain any stream-management nodes.
    targets = _get_targets(gm)
    assert record_event_passthrough not in targets


def test_equal_weight_deterministic():
    """Even with equal weights, the transform should succeed (one moved to aux)."""
    cuda_stream_manager.add_device(torch.cuda.current_device())
    model = EqualWeightFork(128, 128).eval().to("cuda")
    example = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example)

    gm, num = _parallelize_largest_gemm(gm, _MOCK_OPS)
    assert num == 1, f"Expected 1 replacement with equal weights, got {num}"

    _assert_numerical_correctness(gm, model, torch.randn(4, 128, device="cuda"))


def test_weight_size_estimation():
    """_estimate_weight_size should return the product of weight dimensions."""
    model = TwoBranchFork(128, 256, 64).eval().to("cuda")
    example = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example)

    fork_points = _find_gemm_fork_points(gm, _MOCK_OPS)
    assert len(fork_points) == 1

    _, linears = fork_points[0]
    sizes = [_estimate_weight_size(gm, ln) for ln in linears]

    # One should be 256*128=32768, the other 64*128=8192
    assert max(sizes) > min(sizes), f"Expected different sizes but got {sizes}"
    assert 256 * 128 in sizes, f"Expected 256*128=32768 in sizes, got {sizes}"
    assert 64 * 128 in sizes, f"Expected 64*128=8192 in sizes, got {sizes}"


def test_skip_already_handled_fork_point():
    """Fork points with existing multi-stream ops should be skipped.

    We manually insert a ``record_event_passthrough`` into the graph before
    running the transform and verify that the fork point is skipped.
    """
    cuda_stream_manager.add_device(torch.cuda.current_device())
    model = TwoBranchFork(128, 256, 64).eval().to("cuda")
    example = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example)

    # Manually inject a record_event_passthrough at the fork point.
    fork_points = _find_gemm_fork_points(gm, _MOCK_OPS)
    assert len(fork_points) == 1
    fork_node = fork_points[0][0]

    graph = gm.graph
    # Insert a dummy record_event_passthrough as a user of the fork point.
    with graph.inserting_after(fork_node):
        graph.call_function(record_event_passthrough, args=(fork_node,))

    # Now the transform should skip this fork point.
    gm2, num = _parallelize_largest_gemm(gm, _MOCK_OPS)
    assert num == 0, f"Expected 0 replacements (fork point already handled), got {num}"


def test_idempotent_double_application():
    """Applying the transform twice should be idempotent (second pass is no-op)."""
    cuda_stream_manager.add_device(torch.cuda.current_device())
    model = TwoBranchFork(128, 256, 64).eval().to("cuda")
    example = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example)

    gm, num1 = _parallelize_largest_gemm(gm, _MOCK_OPS)
    assert num1 == 1

    # Second application -- the fork point now has a record_event_passthrough
    # user, so it should be skipped.
    gm, num2 = _parallelize_largest_gemm(gm, _MOCK_OPS)
    assert num2 == 0, f"Expected 0 on second pass, got {num2}"

    # Should still be numerically correct after double application.
    _assert_numerical_correctness(gm, model, torch.randn(4, 128, device="cuda"))


# ===================================================================
# Tests -- Topological ordering (regression test for downstream users
#          of the largest linear appearing before remaining linears)
# ===================================================================


class InterleavedDownstreamFork(nn.Module):
    """Model where the largest linear's downstream ops appear BEFORE the remaining linears in graph order.

    This mirrors real MHA patterns where:
      q_proj(x) -> reshape_q -> ...   (largest, downstream appears early)
      k_proj(x) -> reshape_k -> ...   (remaining, appears later)
      v_proj(x) -> reshape_v -> ...   (remaining, appears later)

    Without proper topological fix, the aux node (replacing q_proj) would be
    inserted after v_proj, but reshape_q would still reference it from its
    original early position, causing a 'used before defined' error.
    """

    def __init__(self, hidden_dim: int, large_dim: int, small_dim: int):
        super().__init__()
        self.q_proj = MockFP8Linear(hidden_dim, large_dim)  # Largest
        self.k_proj = MockFP8Linear(hidden_dim, small_dim)
        self.v_proj = MockFP8Linear(hidden_dim, small_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        # Project q through extra ops to simulate reshape/view chain
        self.q_down = nn.Linear(large_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # q_proj and its downstream appear first in graph order
        q = self.q_down(torch.relu(self.q_proj(x)))
        # k_proj, v_proj appear later
        k = self.k_proj(x)
        v = self.v_proj(x)
        # Join point: all three branches merge
        # Project k, v to hidden_dim for addition
        return self.norm(q + k[..., : x.shape[-1]] + v[..., : x.shape[-1]])


class DeepDownstreamFork(nn.Module):
    """Model where the largest linear has a deep chain of downstream ops before the remaining linears.

    Tests transitive movement.

    largest -> relu -> linear -> relu -> linear -> ...
    remaining_1, remaining_2 appear after the chain
    """

    def __init__(self, hidden_dim: int, large_dim: int, small_dim: int):
        super().__init__()
        self.large_proj = MockFP8Linear(hidden_dim, large_dim)  # Largest
        self.small_proj_1 = MockFP8Linear(hidden_dim, small_dim)
        self.small_proj_2 = MockFP8Linear(hidden_dim, small_dim)
        # Deep chain on the largest branch
        self.chain_1 = nn.Linear(large_dim, large_dim, bias=False)
        self.chain_2 = nn.Linear(large_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Deep chain: large_proj -> relu -> chain_1 -> relu -> chain_2
        big = self.chain_2(torch.relu(self.chain_1(torch.relu(self.large_proj(x)))))
        # Remaining linears appear after the chain in graph order
        s1 = self.small_proj_1(x)
        s2 = self.small_proj_2(x)
        return self.norm(big + s1[..., : x.shape[-1]] + s2[..., : x.shape[-1]])


def test_interleaved_downstream_topological_order():
    """Regression: aux node must not violate topological order.

    This checks when the largest linear's downstream ops precede the remaining linears.
    """
    cuda_stream_manager.add_device(torch.cuda.current_device())
    model = InterleavedDownstreamFork(128, 256, 128).eval().to("cuda")
    example = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example)

    # This should NOT raise 'used before defined' RuntimeError.
    gm, num = _parallelize_largest_gemm(gm, _MOCK_OPS)
    assert num == 1, f"Expected 1 replacement, got {num}"

    # Verify graph is topologically valid by running it.
    test_x = torch.randn(4, 128, device="cuda")
    _assert_numerical_correctness(gm, model, test_x)


def test_interleaved_downstream_cuda_graph():
    """CUDA graph capture + replay for interleaved downstream pattern."""
    cuda_stream_manager.add_device(torch.cuda.current_device())
    model = InterleavedDownstreamFork(128, 256, 128).eval().to("cuda")
    example = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example)

    gm, num = _parallelize_largest_gemm(gm, _MOCK_OPS)
    assert num == 1

    _assert_cuda_graph_correctness(gm, model, torch.randn(4, 128, device="cuda"))


def test_deep_downstream_topological_order():
    """Regression: deep chain of ops downstream of the largest linear must all be moved after the aux node."""
    cuda_stream_manager.add_device(torch.cuda.current_device())
    model = DeepDownstreamFork(128, 256, 128).eval().to("cuda")
    example = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example)

    gm, num = _parallelize_largest_gemm(gm, _MOCK_OPS)
    assert num == 1, f"Expected 1 replacement, got {num}"

    test_x = torch.randn(4, 128, device="cuda")
    _assert_numerical_correctness(gm, model, test_x)


def test_deep_downstream_cuda_graph():
    """CUDA graph for deep downstream chain pattern."""
    cuda_stream_manager.add_device(torch.cuda.current_device())
    model = DeepDownstreamFork(128, 256, 128).eval().to("cuda")
    example = torch.randn(4, 128, device="cuda")
    gm = _build_gm(model, example)

    gm, num = _parallelize_largest_gemm(gm, _MOCK_OPS)
    assert num == 1

    _assert_cuda_graph_correctness(gm, model, torch.randn(4, 128, device="cuda"))
