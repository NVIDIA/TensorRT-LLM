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

"""Correctness tests for the Triton backend of torch_moe_dense_mlp.

Compares triton_moe_dense_mlp against the torch reference (torch_moe_dense_mlp)
across multiple dtypes, shapes, and parameter configurations.
"""

import pytest
import torch

# Trigger registration of custom ops
from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.torch_moe import (  # noqa: F401
    torch_moe_dense_mlp,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.triton_moe_dense_mlp import (  # noqa: F401
    triton_moe_dense_mlp,
)


def _make_dense_moe_inputs(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    dtype: torch.dtype,
    device: str = "cuda",
    alpha: float = 1.0,
    limit: float = 10.0,
):
    """Create synthetic inputs for torch_moe_dense_mlp / triton_moe_dense_mlp."""
    torch.manual_seed(42)
    num_tokens = batch_size * seq_len
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    # Routing weights: softmax over experts for each token
    routing_logits = torch.randn(num_tokens, num_experts, device=device, dtype=dtype)
    routing_weights = torch.softmax(routing_logits, dim=-1)
    # gate_up_w: [E, H, 2I] — interleaved gate and up projections
    gate_up_w = torch.randn(
        num_experts, hidden_size, 2 * intermediate_size, device=device, dtype=dtype
    ) * 0.02
    gate_up_b = torch.randn(num_experts, 2 * intermediate_size, device=device, dtype=dtype) * 0.02
    # down_w: [E, I, H]
    down_w = torch.randn(
        num_experts, intermediate_size, hidden_size, device=device, dtype=dtype
    ) * 0.02
    down_b = torch.randn(num_experts, hidden_size, device=device, dtype=dtype) * 0.02
    return hidden_states, routing_weights, gate_up_w, gate_up_b, down_w, down_b, alpha, limit


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "batch_size,seq_len,hidden_size,intermediate_size,num_experts",
    [
        (1, 1, 32, 16, 2),
        (2, 4, 64, 32, 3),
        (4, 8, 128, 64, 4),
    ],
)
def test_triton_moe_dense_mlp_matches_torch(
    batch_size, seq_len, hidden_size, intermediate_size, num_experts, dtype
):
    """Triton kernel output matches torch reference within tolerance."""
    inputs = _make_dense_moe_inputs(
        batch_size, seq_len, hidden_size, intermediate_size, num_experts, dtype
    )
    expected = torch_moe_dense_mlp(*inputs)
    actual = triton_moe_dense_mlp(*inputs)

    rtol = 1e-2 if dtype != torch.float32 else 1e-4
    atol = 1e-2 if dtype != torch.float32 else 1e-5
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "batch_size,seq_len,hidden_size,intermediate_size,num_experts",
    [
        (2, 3, 48, 24, 2),  # non-power-of-2 intermediate_size
        (1, 5, 33, 17, 3),  # non-power-of-2 hidden and intermediate
        (3, 7, 64, 31, 2),  # odd intermediate_size
    ],
)
def test_triton_moe_dense_mlp_non_power_of_2(
    batch_size, seq_len, hidden_size, intermediate_size, num_experts
):
    """Triton kernel handles non-power-of-2 dimensions correctly."""
    dtype = torch.bfloat16
    inputs = _make_dense_moe_inputs(
        batch_size, seq_len, hidden_size, intermediate_size, num_experts, dtype
    )
    expected = torch_moe_dense_mlp(*inputs)
    actual = triton_moe_dense_mlp(*inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 1.07, 2.0])
@pytest.mark.parametrize("limit", [1.0, 5.0, 10.0, 20.0])
def test_triton_moe_dense_mlp_alpha_limit_variants(alpha, limit):
    """Triton kernel works with different alpha and limit values."""
    dtype = torch.float16
    inputs = _make_dense_moe_inputs(
        batch_size=2, seq_len=4, hidden_size=64, intermediate_size=32,
        num_experts=3, dtype=dtype, alpha=alpha, limit=limit
    )
    expected = torch_moe_dense_mlp(*inputs)
    actual = triton_moe_dense_mlp(*inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_triton_moe_dense_mlp_large():
    """Triton kernel works on larger realistic shapes."""
    dtype = torch.bfloat16
    inputs = _make_dense_moe_inputs(
        batch_size=2, seq_len=32, hidden_size=256, intermediate_size=128,
        num_experts=4, dtype=dtype
    )
    expected = torch_moe_dense_mlp(*inputs)
    actual = triton_moe_dense_mlp(*inputs)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_triton_moe_dense_mlp_single_token():
    """Edge case: single token input."""
    dtype = torch.float16
    inputs = _make_dense_moe_inputs(
        batch_size=1, seq_len=1, hidden_size=64, intermediate_size=32,
        num_experts=2, dtype=dtype
    )
    expected = torch_moe_dense_mlp(*inputs)
    actual = triton_moe_dense_mlp(*inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_triton_moe_dense_mlp_single_expert():
    """Edge case: single expert (degenerates to standard MLP)."""
    dtype = torch.float16
    inputs = _make_dense_moe_inputs(
        batch_size=2, seq_len=4, hidden_size=64, intermediate_size=32,
        num_experts=1, dtype=dtype
    )
    expected = torch_moe_dense_mlp(*inputs)
    actual = triton_moe_dense_mlp(*inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
