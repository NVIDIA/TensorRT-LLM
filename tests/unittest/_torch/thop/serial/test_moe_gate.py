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

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import tensorrt_llm  # noqa: F401  # Import to load C++ extensions

# Constants from config (can be adjusted for testing)
N_EXPERTS = 256
N_EXPERTS_PRO = 384
TOPK = 6


def pytorch_gate_forward(
    scores: torch.Tensor,
    bias: torch.Tensor = None,
    input_ids: torch.Tensor = None,
    tid2eid: torch.Tensor = None,
    route_scale: float = 1.5,
    is_hash: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference PyTorch implementation of gate forward.

    Args:
        scores: [batch_size, n_experts] - pre-computed scores
        bias: [n_experts] - expert bias (used in non-hash mode)
        input_ids: [batch_size] - token IDs (used in hash mode)
        tid2eid: [vocab_size, topk] - token to expert mapping (used in hash mode)
        route_scale: scalar to multiply final weights
        is_hash: whether to use hash routing or topk routing

    Returns:
        weights: [batch_size, topk] - normalized routing weights
        indices: [batch_size, topk] - selected expert indices
    """

    # Apply score function: softplus + sqrt
    scores = F.softplus(scores).sqrt()
    original_scores = scores

    # Add bias for topk selection (non-hash mode)
    if not is_hash and bias is not None:
        scores = scores + bias.float()

    # Select experts
    if is_hash:
        # Hash mode: directly lookup from tid2eid
        indices = tid2eid[input_ids]  # [batch_size, topk]
    else:
        # Topk mode: select top-k experts
        indices = scores.topk(TOPK, dim=-1)[1]  # [batch_size, topk]

    # Gather original weights (without bias)
    weights = original_scores.gather(1, indices.long())

    # Normalize weights
    weights = weights / weights.sum(dim=-1, keepdim=True)

    # Apply route scale
    weights = weights * route_scale

    return weights, indices.int()


class TestMoeGate:
    """Test suite for MOE gate forward operation."""

    @pytest.mark.parametrize("batch_size", [1, 4, 32, 128])
    @pytest.mark.parametrize("route_scale", [1.0, 1.5, 2.0])
    def test_gate_forward_non_hash(self, batch_size, route_scale):
        """Test non-hash (topk) mode gate forward."""
        # Generate random scores
        scores = torch.randn(batch_size, N_EXPERTS, dtype=torch.float32, device="cuda")
        bias = torch.randn(N_EXPERTS, dtype=torch.float32, device="cuda")

        # Pre-allocate outputs
        out_weights = torch.empty(batch_size, TOPK, dtype=torch.float32, device="cuda")
        out_indices = torch.empty(batch_size, TOPK, dtype=torch.int32, device="cuda")

        # Empty tensors for hash mode parameters
        input_ids_tensor = torch.empty(0, dtype=torch.int32, device="cuda")
        tid2eid_tensor = torch.empty(0, dtype=torch.int32, device="cuda")

        # Call CUDA kernel
        torch.ops.trtllm.gate_forward(
            scores,
            bias,
            input_ids_tensor,
            tid2eid_tensor,
            out_weights,
            out_indices,
            TOPK,
            route_scale,
            False,
        )

        # Compute reference
        ref_weights, ref_indices = pytorch_gate_forward(
            scores, bias=bias, route_scale=route_scale, is_hash=False
        )

        # Verify shapes
        assert out_weights.shape == (batch_size, TOPK), (
            f"weights shape mismatch: {out_weights.shape}"
        )
        assert out_indices.shape == (batch_size, TOPK), (
            f"indices shape mismatch: {out_indices.shape}"
        )

        # Verify values
        assert torch.equal(out_indices, ref_indices), (
            f"Indices mismatch:\nCUDA: {out_indices}\nRef:  {ref_indices}"
        )
        assert torch.allclose(out_weights, ref_weights, rtol=1e-4, atol=1e-5), (
            f"Weights mismatch (max diff: {(out_weights - ref_weights).abs().max():.6f})"
        )

    @pytest.mark.parametrize("batch_size", [1, 4, 32, 128])
    @pytest.mark.parametrize("route_scale", [1.0, 1.5, 2.0])
    @pytest.mark.parametrize("vocab_size", [1000, 10000])
    def test_gate_forward_hash(self, batch_size, route_scale, vocab_size):
        """Test hash mode gate forward."""
        # Generate random scores
        scores = torch.randn(batch_size, N_EXPERTS, dtype=torch.float32, device="cuda")
        input_ids = torch.randint(0, vocab_size, (batch_size,), dtype=torch.int32, device="cuda")
        tid2eid = torch.randint(0, N_EXPERTS, (vocab_size, TOPK), dtype=torch.int32, device="cuda")

        # Pre-allocate outputs
        out_weights = torch.empty(batch_size, TOPK, dtype=torch.float32, device="cuda")
        out_indices = torch.empty(batch_size, TOPK, dtype=torch.int32, device="cuda")

        # Empty tensors for non-hash mode parameters
        bias = torch.empty(0, dtype=torch.float32, device="cuda")

        # Call CUDA kernel
        torch.ops.trtllm.gate_forward(
            scores, bias, input_ids, tid2eid, out_weights, out_indices, TOPK, route_scale, True
        )

        # Compute reference
        ref_weights, ref_indices = pytorch_gate_forward(
            scores, input_ids=input_ids, tid2eid=tid2eid, route_scale=route_scale, is_hash=True
        )

        # Verify shapes
        assert out_weights.shape == (batch_size, TOPK), (
            f"weights shape mismatch: {out_weights.shape}"
        )
        assert out_indices.shape == (batch_size, TOPK), (
            f"indices shape mismatch: {out_indices.shape}"
        )

        # Verify values
        assert torch.equal(out_indices, ref_indices), (
            f"Indices mismatch:\nCUDA: {out_indices}\nRef:  {ref_indices}"
        )
        assert torch.allclose(out_weights, ref_weights, rtol=1e-4, atol=1e-5), (
            f"Weights mismatch (max diff: {(out_weights - ref_weights).abs().max():.6f})"
        )

    @pytest.mark.parametrize("is_hash", [False, True])
    def test_gate_forward_384_experts(self, is_hash):
        """Test DeepSeek-V4-Pro expert count."""
        batch_size = 8
        route_scale = 1.5
        scores = torch.randn(batch_size, N_EXPERTS_PRO, dtype=torch.float32, device="cuda")
        out_weights = torch.empty(batch_size, TOPK, dtype=torch.float32, device="cuda")
        out_indices = torch.empty(batch_size, TOPK, dtype=torch.int32, device="cuda")

        if is_hash:
            vocab_size = 1024
            input_ids = torch.randint(0, vocab_size, (batch_size,), dtype=torch.int32, device="cuda")
            tid2eid = torch.randint(0, N_EXPERTS_PRO, (vocab_size, TOPK), dtype=torch.int32, device="cuda")
            bias = torch.empty(0, dtype=torch.float32, device="cuda")
            ref_weights, ref_indices = pytorch_gate_forward(
                scores, input_ids=input_ids, tid2eid=tid2eid, route_scale=route_scale, is_hash=True
            )
        else:
            bias = torch.randn(N_EXPERTS_PRO, dtype=torch.float32, device="cuda")
            input_ids = torch.empty(0, dtype=torch.int32, device="cuda")
            tid2eid = torch.empty(0, dtype=torch.int32, device="cuda")
            ref_weights, ref_indices = pytorch_gate_forward(
                scores, bias=bias, route_scale=route_scale, is_hash=False
            )

        torch.ops.trtllm.gate_forward(
            scores, bias, input_ids, tid2eid, out_weights, out_indices, TOPK, route_scale, is_hash
        )

        assert torch.equal(out_indices, ref_indices)
        assert torch.allclose(out_weights, ref_weights, rtol=1e-4, atol=1e-5)

    def test_gate_forward_deterministic(self):
        """Test that gate forward is deterministic across multiple runs."""
        batch_size = 16
        scores = torch.randn(batch_size, N_EXPERTS, dtype=torch.float32, device="cuda")
        bias = torch.randn(N_EXPERTS, dtype=torch.float32, device="cuda")

        # Run multiple times
        results_weights = []
        results_indices = []
        for _ in range(5):
            out_weights = torch.empty(batch_size, TOPK, dtype=torch.float32, device="cuda")
            out_indices = torch.empty(batch_size, TOPK, dtype=torch.int32, device="cuda")

            torch.ops.trtllm.gate_forward(
                scores,
                bias,
                torch.empty(0, dtype=torch.int32, device="cuda"),
                torch.empty(0, dtype=torch.int32, device="cuda"),
                out_weights,
                out_indices,
                TOPK,
                1.5,
                False,
            )

            results_weights.append(out_weights.clone())
            results_indices.append(out_indices.clone())

        # Verify all results are identical
        for i in range(1, len(results_weights)):
            assert torch.equal(results_indices[0], results_indices[i]), (
                "Gate forward is not deterministic (indices differ)"
            )
            assert torch.equal(results_weights[0], results_weights[i]), (
                "Gate forward is not deterministic (weights differ)"
            )

    def test_gate_forward_edge_cases(self):
        """Test edge cases."""
        # Test with single token
        scores = torch.randn(1, N_EXPERTS, dtype=torch.float32, device="cuda")
        bias = torch.randn(N_EXPERTS, dtype=torch.float32, device="cuda")
        out_weights = torch.empty(1, TOPK, dtype=torch.float32, device="cuda")
        out_indices = torch.empty(1, TOPK, dtype=torch.int32, device="cuda")

        torch.ops.trtllm.gate_forward(
            scores,
            bias,
            torch.empty(0, dtype=torch.int32, device="cuda"),
            torch.empty(0, dtype=torch.int32, device="cuda"),
            out_weights,
            out_indices,
            TOPK,
            1.5,
            False,
        )

        # Verify weights sum to approximately route_scale
        assert torch.allclose(
            out_weights.sum(dim=-1), torch.tensor([1.5], device="cuda"), rtol=1e-4
        )

        # Test with all equal scores (tie-breaking should be deterministic)
        scores = torch.ones(4, N_EXPERTS, dtype=torch.float32, device="cuda")
        bias = torch.zeros(N_EXPERTS, dtype=torch.float32, device="cuda")
        out_weights = torch.empty(4, TOPK, dtype=torch.float32, device="cuda")
        out_indices = torch.empty(4, TOPK, dtype=torch.int32, device="cuda")

        torch.ops.trtllm.gate_forward(
            scores,
            bias,
            torch.empty(0, dtype=torch.int32, device="cuda"),
            torch.empty(0, dtype=torch.int32, device="cuda"),
            out_weights,
            out_indices,
            TOPK,
            1.0,
            False,
        )

        # With equal scores, weights should be equal
        assert torch.allclose(out_weights, out_weights[0:1].expand(4, TOPK), rtol=1e-4)

    @pytest.mark.parametrize("batch_size", [32, 128, 256, 1024])
    def test_gate_forward_performance(self, batch_size):
        """Benchmark gate forward for different batch sizes."""
        scores = torch.randn(batch_size, N_EXPERTS, dtype=torch.float32, device="cuda")
        bias = torch.randn(N_EXPERTS, dtype=torch.float32, device="cuda")
        out_weights = torch.empty(batch_size, TOPK, dtype=torch.float32, device="cuda")
        out_indices = torch.empty(batch_size, TOPK, dtype=torch.int32, device="cuda")

        # Warmup
        for _ in range(10):
            torch.ops.trtllm.gate_forward(
                scores,
                bias,
                torch.empty(0, dtype=torch.int32, device="cuda"),
                torch.empty(0, dtype=torch.int32, device="cuda"),
                out_weights,
                out_indices,
                TOPK,
                1.5,
                False,
            )

        torch.cuda.synchronize()

        # Benchmark
        import time

        n_runs = 100
        start = time.perf_counter()
        for _ in range(n_runs):
            torch.ops.trtllm.gate_forward(
                scores,
                bias,
                torch.empty(0, dtype=torch.int32, device="cuda"),
                torch.empty(0, dtype=torch.int32, device="cuda"),
                out_weights,
                out_indices,
                TOPK,
                1.5,
                False,
            )
        torch.cuda.synchronize()
        end = time.perf_counter()

        avg_time_ms = (end - start) * 1000.0 / n_runs
        print(f"\nBatch size {batch_size}: {avg_time_ms:.4f} ms per iteration")

        # Basic sanity check that it completed successfully
        assert out_weights.shape == (batch_size, TOPK)
        assert out_indices.shape == (batch_size, TOPK)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
