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

"""Correctness tests for the Triton MoE router kernel."""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.moe_router import (  # noqa: F401
    torch_moe_router,
    triton_moe_router,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.triton_kernels.moe_router import moe_router


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "M, E, top_k",
    [
        (1, 8, 2),
        (4, 8, 2),
        (8, 16, 4),
        (32, 64, 8),
        (128, 8, 2),
    ],
)
def test_triton_moe_router_matches_torch(M, E, top_k, dtype):
    """Triton kernel output matches torch reference for weights and indices."""
    torch.manual_seed(42)
    logits = torch.randn(M, E, device="cuda", dtype=dtype)

    # Torch reference (custom op)
    expected_weights, expected_indices = torch_moe_router(logits, top_k, True)

    # Triton backend (custom op)
    actual_weights, actual_indices = triton_moe_router(logits, top_k, True)

    # Indices must match exactly
    torch.testing.assert_close(actual_indices, expected_indices)
    # Weights should match within tolerance (both are float32)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "M, E, top_k",
    [
        (1, 8, 2),
        (8, 16, 4),
        (32, 64, 8),
    ],
)
def test_triton_moe_router_no_normalize(M, E, top_k, dtype):
    """Triton kernel works correctly without normalization."""
    torch.manual_seed(42)
    logits = torch.randn(M, E, device="cuda", dtype=dtype)

    expected_weights, expected_indices = torch_moe_router(logits, top_k, False)
    actual_weights, actual_indices = triton_moe_router(logits, top_k, False)

    torch.testing.assert_close(actual_indices, expected_indices)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("E", [7, 13, 33, 65])
def test_triton_moe_router_non_power_of_2_experts(E):
    """Triton kernel handles non-power-of-2 number of experts correctly."""
    torch.manual_seed(42)
    top_k = min(2, E)
    logits = torch.randn(8, E, device="cuda", dtype=torch.bfloat16)

    expected_weights, expected_indices = torch_moe_router(logits, top_k, True)
    actual_weights, actual_indices = triton_moe_router(logits, top_k, True)

    torch.testing.assert_close(actual_indices, expected_indices)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-4, atol=1e-4)


def test_triton_moe_router_large():
    """Triton kernel works on large realistic shapes (e.g., Mixtral-style)."""
    torch.manual_seed(42)
    # Mixtral: 8 experts, top_k=2, typical batch of 2048 tokens
    M, E, top_k = 2048, 8, 2
    logits = torch.randn(M, E, device="cuda", dtype=torch.bfloat16)

    expected_weights, expected_indices = torch_moe_router(logits, top_k, True)
    actual_weights, actual_indices = triton_moe_router(logits, top_k, True)

    torch.testing.assert_close(actual_indices, expected_indices)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-3, atol=1e-3)


def test_triton_moe_router_large_expert_count():
    """Triton kernel works with many experts (e.g., DeepSeek-style 256 experts)."""
    torch.manual_seed(42)
    M, E, top_k = 512, 256, 8
    logits = torch.randn(M, E, device="cuda", dtype=torch.bfloat16)

    expected_weights, expected_indices = torch_moe_router(logits, top_k, True)
    actual_weights, actual_indices = triton_moe_router(logits, top_k, True)

    torch.testing.assert_close(actual_indices, expected_indices)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-3, atol=1e-3)


def test_triton_moe_router_kernel_direct():
    """Test the Triton kernel launcher directly (bypassing custom op wrapper)."""
    torch.manual_seed(42)
    M, E, top_k = 16, 8, 2
    logits = torch.randn(M, E, device="cuda", dtype=torch.float16)

    # Direct kernel call
    actual_weights, actual_indices = moe_router(logits, top_k, normalize=True)

    # Torch reference
    import torch.nn.functional as F

    probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    expected_weights, expected_indices = torch.topk(probs, top_k, dim=-1)
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)

    torch.testing.assert_close(actual_indices, expected_indices)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-4, atol=1e-4)


def test_triton_moe_router_top_k_equals_1():
    """Triton kernel works with top_k=1 (single expert selection)."""
    torch.manual_seed(42)
    M, E = 16, 8
    logits = torch.randn(M, E, device="cuda", dtype=torch.bfloat16)

    expected_weights, expected_indices = torch_moe_router(logits, 1, True)
    actual_weights, actual_indices = triton_moe_router(logits, 1, True)

    torch.testing.assert_close(actual_indices, expected_indices)
    # With top_k=1 and normalize=True, weights should all be 1.0
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-5, atol=1e-5)


def test_triton_moe_router_weights_sum_to_one():
    """When normalize=True, top-k weights should sum to 1 for each token."""
    torch.manual_seed(42)
    M, E, top_k = 32, 16, 4
    logits = torch.randn(M, E, device="cuda", dtype=torch.bfloat16)

    weights, _ = triton_moe_router(logits, top_k, True)
    sums = weights.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-5)
