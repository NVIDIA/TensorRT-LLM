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

# Import to trigger op registration
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "T, E, H, top_k",
    [
        (1, 8, 64, 2),
        (4, 8, 128, 2),
        (8, 16, 64, 4),
        (32, 64, 128, 8),
        (128, 8, 256, 2),
    ],
)
def test_triton_moe_router_matches_torch(T, E, H, top_k, dtype):
    """Triton router output matches torch reference."""
    torch.manual_seed(42)
    hidden_states = torch.randn(T, H, device="cuda", dtype=dtype)
    weight = torch.randn(E, H, device="cuda", dtype=dtype)
    bias = torch.randn(E, device="cuda", dtype=dtype)

    expected = torch.ops.auto_deploy.torch_moe_router(hidden_states, weight, bias, top_k)
    actual = torch.ops.auto_deploy.triton_moe_router(hidden_states, weight, bias, top_k)

    # Both return [T, E] with softmax scores scattered at top-k positions
    # Check non-zero positions match
    torch.testing.assert_close(
        (actual > 0).long(), (expected > 0).long(),
    )
    # Check values at non-zero positions
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("E", [7, 13, 33])
def test_triton_moe_router_non_power_of_2_experts(E):
    """Triton kernel handles non-power-of-2 number of experts."""
    torch.manual_seed(42)
    T, H, top_k = 8, 64, min(2, E)
    hidden_states = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(E, H, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(E, device="cuda", dtype=torch.bfloat16)

    expected = torch.ops.auto_deploy.torch_moe_router(hidden_states, weight, bias, top_k)
    actual = torch.ops.auto_deploy.triton_moe_router(hidden_states, weight, bias, top_k)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_triton_moe_router_3d_input():
    """Triton router handles 3D input [B, S, H]."""
    torch.manual_seed(42)
    B, S, H, E, top_k = 2, 16, 64, 8, 2
    hidden_states = torch.randn(B, S, H, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(E, H, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(E, device="cuda", dtype=torch.bfloat16)

    expected = torch.ops.auto_deploy.torch_moe_router(hidden_states, weight, bias, top_k)
    actual = torch.ops.auto_deploy.triton_moe_router(hidden_states, weight, bias, top_k)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_triton_moe_router_large():
    """Triton kernel works on large realistic shapes."""
    torch.manual_seed(42)
    T, E, H, top_k = 2048, 8, 4096, 2
    hidden_states = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(E, H, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(E, device="cuda", dtype=torch.bfloat16)

    expected = torch.ops.auto_deploy.torch_moe_router(hidden_states, weight, bias, top_k)
    actual = torch.ops.auto_deploy.triton_moe_router(hidden_states, weight, bias, top_k)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_triton_moe_router_scores_sum():
    """Router scores for each token should sum to ~1.0 (softmax over top-k)."""
    torch.manual_seed(42)
    T, E, H, top_k = 32, 16, 128, 4
    hidden_states = torch.randn(T, H, device="cuda", dtype=torch.float32)
    weight = torch.randn(E, H, device="cuda", dtype=torch.float32)
    bias = torch.randn(E, device="cuda", dtype=torch.float32)

    scores = torch.ops.auto_deploy.triton_moe_router(hidden_states, weight, bias, top_k)
    sums = scores.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-4, atol=1e-4)
