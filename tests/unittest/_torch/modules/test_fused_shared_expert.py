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

"""Unit tests for fused_sigmoid_gate_mul_add Triton kernel.

Covers the shape and dtype combinations actually used by Qwen3-Next /
Qwen3.5 MoE blocks, and a few edge cases (num_tokens=1, hidden not a
multiple of BLOCK_SIZE).
"""

import pytest
import torch

from tensorrt_llm._torch.modules.fused_shared_expert import fused_sigmoid_gate_mul_add

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _reference(
    final_hidden_states: torch.Tensor, gate_logits: torch.Tensor, shared_expert_output: torch.Tensor
) -> torch.Tensor:
    # Reference in fp32 to match the kernel's internal compute precision,
    # then cast back to the I/O dtype.
    gate_bcast = gate_logits.reshape(-1, 1).to(torch.float32)
    s = torch.sigmoid(gate_bcast)
    out = final_hidden_states.to(torch.float32) + s * shared_expert_output.to(torch.float32)
    return out.to(final_hidden_states.dtype)


@pytest.mark.parametrize("num_tokens", [1, 2, 16, 64, 256])
@pytest.mark.parametrize("hidden", [2048, 4096, 8192])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("gate_shape_is_2d", [True, False])
def test_fused_sigmoid_gate_mul_add(num_tokens, hidden, dtype, gate_shape_is_2d):
    torch.manual_seed(42)
    device = "cuda"

    final = torch.randn(num_tokens, hidden, dtype=dtype, device=device)
    shared = torch.randn(num_tokens, hidden, dtype=dtype, device=device)
    if gate_shape_is_2d:
        gate = torch.randn(num_tokens, 1, dtype=dtype, device=device)
    else:
        gate = torch.randn(num_tokens, dtype=dtype, device=device)

    expected = _reference(final, gate, shared)

    out = fused_sigmoid_gate_mul_add(final.clone(), gate, shared)

    assert out.shape == final.shape
    assert out.dtype == final.dtype

    if dtype == torch.float32:
        atol, rtol = 1e-5, 1e-5
    else:  # bfloat16 / float16
        atol, rtol = 2e-2, 2e-2
    torch.testing.assert_close(out, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("hidden", [1, 7, 63, 1023, 1025, 3000])
def test_fused_sigmoid_gate_mul_add_hidden_edge_cases(hidden):
    """Hidden dims that are not multiples of the inner block must still work."""
    torch.manual_seed(0)
    num_tokens = 8
    dtype = torch.bfloat16
    device = "cuda"

    final = torch.randn(num_tokens, hidden, dtype=dtype, device=device)
    gate = torch.randn(num_tokens, 1, dtype=dtype, device=device)
    shared = torch.randn(num_tokens, hidden, dtype=dtype, device=device)

    expected = _reference(final, gate, shared)
    out = fused_sigmoid_gate_mul_add(final.clone(), gate, shared)
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)


def test_fused_sigmoid_gate_mul_add_in_place_semantics():
    """The kernel writes in-place to final_hidden_states; verify aliasing."""
    torch.manual_seed(0)
    num_tokens, hidden = 32, 4096
    dtype = torch.bfloat16
    device = "cuda"

    final = torch.randn(num_tokens, hidden, dtype=dtype, device=device)
    gate = torch.randn(num_tokens, 1, dtype=dtype, device=device)
    shared = torch.randn(num_tokens, hidden, dtype=dtype, device=device)

    expected = _reference(final, gate, shared)

    # Returned tensor must be the same storage as the input.
    final_orig = final
    out = fused_sigmoid_gate_mul_add(final, gate, shared)
    assert out.data_ptr() == final_orig.data_ptr(), (
        "fused_sigmoid_gate_mul_add should update final_hidden_states in-place"
    )
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)


def test_fused_sigmoid_gate_mul_add_output_buffer():
    torch.manual_seed(0)
    num_tokens, hidden = 32, 4096
    dtype = torch.bfloat16
    device = "cuda"

    final = torch.randn(num_tokens, hidden, dtype=dtype, device=device)
    gate = torch.randn(num_tokens, 1, dtype=dtype, device=device)
    shared = torch.randn(num_tokens, hidden, dtype=dtype, device=device)
    output = torch.empty_like(final)

    expected = _reference(final, gate, shared)
    out = fused_sigmoid_gate_mul_add(final, gate, shared, output=output)

    assert out.data_ptr() == output.data_ptr()
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)


def test_fused_sigmoid_gate_mul_add_qwen35_shape():
    """Exact shape used on Qwen3.5-397B ADP4 1k/1k (hidden=4096, B=64)."""
    torch.manual_seed(0)
    num_tokens, hidden = 64, 4096
    dtype = torch.bfloat16
    device = "cuda"

    final = torch.randn(num_tokens, hidden, dtype=dtype, device=device)
    gate = torch.randn(num_tokens, 1, dtype=dtype, device=device)
    shared = torch.randn(num_tokens, hidden, dtype=dtype, device=device)

    expected = _reference(final, gate, shared)
    out = fused_sigmoid_gate_mul_add(final.clone(), gate, shared)
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)


def test_fused_sigmoid_gate_mul_add_non_contig_output():
    # Simulates allocate_output's symmetric-heap buffer: stride(-1)==1 but
    # row stride padded beyond hidden. The kernel must honor stride(0) and
    # the wrapper must not require full contiguity on the output tensor.
    torch.manual_seed(0)
    num_tokens, hidden = 32, 4096
    dtype = torch.bfloat16
    device = "cuda"

    padded = torch.empty(num_tokens, hidden + 64, dtype=dtype, device=device)
    output = padded[:, :hidden]
    assert output.stride(-1) == 1 and output.stride(0) > hidden
    assert not output.is_contiguous()

    final = torch.randn(num_tokens, hidden, dtype=dtype, device=device)
    gate = torch.randn(num_tokens, 1, dtype=dtype, device=device)
    shared = torch.randn(num_tokens, hidden, dtype=dtype, device=device)

    expected = _reference(final, gate, shared)
    out = fused_sigmoid_gate_mul_add(final, gate, shared, output=output)
    assert out.data_ptr() == output.data_ptr()
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)


def test_fused_sigmoid_gate_mul_add_zero_tokens():
    # Early-return path: num_tokens == 0 must not launch the kernel.
    dtype = torch.bfloat16
    device = "cuda"
    final = torch.empty(0, 4096, dtype=dtype, device=device)
    gate = torch.empty(0, 1, dtype=dtype, device=device)
    shared = torch.empty(0, 4096, dtype=dtype, device=device)

    out = fused_sigmoid_gate_mul_add(final, gate, shared)
    assert out.shape == final.shape
    assert out.dtype == final.dtype


def test_fused_sigmoid_gate_mul_add_empty_strided_output():
    # Same intent as the slice-based test, but uses empty_strided directly so
    # the contract — "output may have stride(0) > hidden with stride(-1) == 1" —
    # is expressed without relying on slicing semantics.
    torch.manual_seed(0)
    num_tokens, hidden = 32, 4096
    dtype = torch.bfloat16
    device = "cuda"

    row_stride = hidden + 128
    output = torch.empty_strided((num_tokens, hidden), (row_stride, 1), dtype=dtype, device=device)
    assert output.stride() == (row_stride, 1)
    assert not output.is_contiguous()

    final = torch.randn(num_tokens, hidden, dtype=dtype, device=device)
    gate = torch.randn(num_tokens, 1, dtype=dtype, device=device)
    shared = torch.randn(num_tokens, hidden, dtype=dtype, device=device)

    expected = _reference(final, gate, shared)
    out = fused_sigmoid_gate_mul_add(final, gate, shared, output=output)
    assert out.data_ptr() == output.data_ptr()
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)


def test_fused_sigmoid_gate_mul_add_large_hidden_multi_block():
    # hidden > MAX_BLOCK (8192) — exercises num_col_blocks > 1 path.
    torch.manual_seed(0)
    num_tokens, hidden = 4, 12288
    dtype = torch.bfloat16
    device = "cuda"

    final = torch.randn(num_tokens, hidden, dtype=dtype, device=device)
    gate = torch.randn(num_tokens, 1, dtype=dtype, device=device)
    shared = torch.randn(num_tokens, hidden, dtype=dtype, device=device)

    expected = _reference(final, gate, shared)
    out = fused_sigmoid_gate_mul_add(final.clone(), gate, shared)
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)
