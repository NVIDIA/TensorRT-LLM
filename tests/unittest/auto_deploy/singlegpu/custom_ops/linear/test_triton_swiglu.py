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

"""Correctness tests for the Triton SwiGLU MLP backend.

Compares triton_swiglu_mlp against torch_swiglu_mlp across multiple dtypes,
shapes, and configurations (with/without bias).
"""

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.linear.swiglu import (  # noqa: F401
    torch_swiglu_mlp,
    triton_swiglu_mlp,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.linear.triton_swiglu import triton_swiglu_activation

# ---------------------------------------------------------------------------
# Triton SwiGLU activation kernel tests (low-level)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    [(1, 128), (4, 512), (8, 1024), (32, 4096)],
)
def test_triton_swiglu_activation_matches_torch(shape, dtype):
    """Triton SwiGLU activation matches torch reference within tolerance."""
    torch.manual_seed(42)
    gate = torch.randn(*shape, device="cuda", dtype=dtype)
    up = torch.randn(*shape, device="cuda", dtype=dtype)

    # Torch reference: silu(gate) * up, computed in float32 to match Triton kernel precision
    gate_f32 = gate.float()
    up_f32 = up.float()
    expected = (torch.nn.functional.silu(gate_f32) * up_f32).to(dtype)
    actual = triton_swiglu_activation(gate, up)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "shape",
    [(1, 127), (4, 513), (8, 1023), (2, 3, 257)],
)
def test_triton_swiglu_activation_non_power_of_2(shape):
    """Triton activation handles non-power-of-2 dimensions correctly."""
    torch.manual_seed(42)
    gate = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    # Compute reference in float32 to match Triton kernel precision
    gate_f32 = gate.float()
    up_f32 = up.float()
    expected = (torch.nn.functional.silu(gate_f32) * up_f32).to(torch.bfloat16)
    actual = triton_swiglu_activation(gate, up)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Full triton_swiglu_mlp op tests (end-to-end including GEMMs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "batch_size,hidden_size,intermediate_size",
    [
        (1, 128, 256),
        (4, 256, 512),
        (8, 512, 1024),
    ],
)
def test_triton_swiglu_mlp_no_bias(batch_size, hidden_size, intermediate_size, dtype):
    """Triton SwiGLU MLP without bias matches torch reference."""
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    gate_weight = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=dtype) * 0.02
    up_weight = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=dtype) * 0.02
    down_weight = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=dtype) * 0.02

    expected = torch_swiglu_mlp(input_tensor, gate_weight, up_weight, down_weight, None, None, None)
    actual = triton_swiglu_mlp(input_tensor, gate_weight, up_weight, down_weight, None, None, None)

    # MLP tolerance must account for the Triton kernel computing silu(gate)*up in float32
    # while torch uses native dtype — the down-projection GEMM amplifies these activation
    # differences proportionally to sqrt(intermediate_size).
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "batch_size,hidden_size,intermediate_size",
    [
        (1, 128, 256),
        (4, 256, 512),
    ],
)
def test_triton_swiglu_mlp_with_bias(batch_size, hidden_size, intermediate_size, dtype):
    """Triton SwiGLU MLP with bias matches torch reference."""
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    gate_weight = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=dtype) * 0.02
    up_weight = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=dtype) * 0.02
    down_weight = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=dtype) * 0.02
    gate_bias = torch.randn(intermediate_size, device="cuda", dtype=dtype) * 0.02
    up_bias = torch.randn(intermediate_size, device="cuda", dtype=dtype) * 0.02
    down_bias = torch.randn(hidden_size, device="cuda", dtype=dtype) * 0.02

    expected = torch_swiglu_mlp(
        input_tensor, gate_weight, up_weight, down_weight, gate_bias, up_bias, down_bias
    )
    actual = triton_swiglu_mlp(
        input_tensor, gate_weight, up_weight, down_weight, gate_bias, up_bias, down_bias
    )

    # MLP tolerance must account for the Triton kernel computing silu(gate)*up in float32
    # while torch uses native dtype — the down-projection GEMM amplifies these activation
    # differences proportionally to sqrt(intermediate_size).
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


def test_triton_swiglu_mlp_3d_input():
    """Triton SwiGLU MLP handles 3D input (batch, seq_len, hidden) correctly."""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    batch_size, seq_len, hidden_size, intermediate_size = 2, 16, 256, 512
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
    gate_weight = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=dtype) * 0.02
    up_weight = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=dtype) * 0.02
    down_weight = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=dtype) * 0.02

    expected = torch_swiglu_mlp(input_tensor, gate_weight, up_weight, down_weight, None, None, None)
    actual = triton_swiglu_mlp(input_tensor, gate_weight, up_weight, down_weight, None, None, None)

    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


def test_triton_swiglu_mlp_large():
    """Triton SwiGLU MLP works on large realistic shapes."""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    batch_size, hidden_size, intermediate_size = 32, 4096, 11008

    input_tensor = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    gate_weight = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=dtype) * 0.02
    up_weight = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=dtype) * 0.02
    down_weight = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=dtype) * 0.02

    expected = torch_swiglu_mlp(input_tensor, gate_weight, up_weight, down_weight, None, None, None)
    actual = triton_swiglu_mlp(input_tensor, gate_weight, up_weight, down_weight, None, None, None)

    # Wider tolerance for large shapes — accumulated floating-point differences from
    # float32 vs native-dtype activation scale with sqrt(intermediate_size).
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


def test_triton_swiglu_mlp_single_element():
    """Triton SwiGLU MLP handles single-element batch."""
    torch.manual_seed(42)
    dtype = torch.float16
    hidden_size, intermediate_size = 64, 128

    input_tensor = torch.randn(1, hidden_size, device="cuda", dtype=dtype)
    gate_weight = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=dtype) * 0.02
    up_weight = torch.randn(intermediate_size, hidden_size, device="cuda", dtype=dtype) * 0.02
    down_weight = torch.randn(hidden_size, intermediate_size, device="cuda", dtype=dtype) * 0.02

    expected = torch_swiglu_mlp(input_tensor, gate_weight, up_weight, down_weight, None, None, None)
    actual = triton_swiglu_mlp(input_tensor, gate_weight, up_weight, down_weight, None, None, None)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
