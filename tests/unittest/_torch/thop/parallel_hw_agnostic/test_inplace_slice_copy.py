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
"""Unit tests for trtllm::inplace_slice_copy.

Verifies that the cudaMemcpy2DAsync-backed op produces the same result as a
reference Python slice + Tensor.copy_, for the row-prefix / column-slice
write pattern used in EAGLE3 hidden-state capture.
"""

import pytest
import torch

import tensorrt_llm  # noqa: F401


def _reference(dest_shape, src, dim1_start, dim1_end, dtype):
    dest = torch.zeros(dest_shape, dtype=dtype, device="cuda")
    num_tokens = src.shape[0]
    dest[:num_tokens, dim1_start:dim1_end].copy_(src)
    return dest


def _run(dest_shape, src, dim1_start, dim1_end, dtype):
    dest = torch.zeros(dest_shape, dtype=dtype, device="cuda")
    torch.ops.trtllm.inplace_slice_copy(dest, src, dim1_start, dim1_end)
    return dest


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_full_dest_full_width(dtype):
    """num_tokens == dest.size(0) and slice == full dest width."""
    dest_shape = (16, 64)
    src = torch.randn(16, 64, dtype=dtype, device="cuda")
    out = _run(dest_shape, src, 0, 64, dtype)
    ref = _reference(dest_shape, src, 0, 64, dtype)
    torch.testing.assert_close(out, ref)


def test_partial_rows():
    """num_tokens < dest.size(0): trailing rows must stay zero."""
    dtype = torch.bfloat16
    dest_shape = (32, 64)
    src = torch.randn(8, 64, dtype=dtype, device="cuda")
    out = _run(dest_shape, src, 0, 64, dtype)
    ref = _reference(dest_shape, src, 0, 64, dtype)
    torch.testing.assert_close(out, ref)
    assert torch.all(out[8:] == 0)


def test_column_slice_middle():
    """Write to a middle column band; flanking columns must stay zero."""
    dtype = torch.bfloat16
    dest_shape = (16, 96)
    src = torch.randn(16, 32, dtype=dtype, device="cuda")
    out = _run(dest_shape, src, 32, 64, dtype)
    ref = _reference(dest_shape, src, 32, 64, dtype)
    torch.testing.assert_close(out, ref)
    assert torch.all(out[:, :32] == 0)
    assert torch.all(out[:, 64:] == 0)


def test_layered_capture_pattern():
    """Mimic EAGLE3 hidden-state capture: write each layer into its band."""
    dtype = torch.bfloat16
    num_tokens, hidden_size, num_layers = 12, 48, 3
    dest_shape = (24, hidden_size * num_layers)
    srcs = [
        torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda") for _ in range(num_layers)
    ]

    out = torch.zeros(dest_shape, dtype=dtype, device="cuda")
    for i, s in enumerate(srcs):
        torch.ops.trtllm.inplace_slice_copy(out, s, i * hidden_size, (i + 1) * hidden_size)

    ref = torch.zeros(dest_shape, dtype=dtype, device="cuda")
    for i, s in enumerate(srcs):
        ref[:num_tokens, i * hidden_size : (i + 1) * hidden_size].copy_(s)

    torch.testing.assert_close(out, ref)


def test_empty_src_is_noop():
    """num_tokens == 0 must not modify dest and must not raise."""
    dtype = torch.bfloat16
    dest_shape = (16, 64)
    dest = torch.full(dest_shape, 7, dtype=dtype, device="cuda")
    src = torch.empty(0, 32, dtype=dtype, device="cuda")
    torch.ops.trtllm.inplace_slice_copy(dest, src, 16, 48)
    assert torch.all(dest == 7)


def test_dtype_mismatch_raises():
    dest = torch.zeros(8, 32, dtype=torch.bfloat16, device="cuda")
    src = torch.randn(8, 32, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.inplace_slice_copy(dest, src, 0, 32)


def test_out_of_bounds_raises():
    dtype = torch.bfloat16
    dest = torch.zeros(8, 32, dtype=dtype, device="cuda")
    src = torch.randn(8, 8, dtype=dtype, device="cuda")
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.inplace_slice_copy(dest, src, 28, 36)


def test_negative_dim1_start_raises():
    """A negative dim1_start would underflow the dest pointer."""
    dtype = torch.bfloat16
    dest = torch.zeros(8, 32, dtype=dtype, device="cuda")
    src = torch.randn(8, 8, dtype=dtype, device="cuda")
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.inplace_slice_copy(dest, src, -8, 0)


def test_device_mismatch_raises():
    """dest and src on different CUDA devices must be rejected."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires >= 2 CUDA devices")
    dtype = torch.bfloat16
    dest = torch.zeros(8, 32, dtype=dtype, device="cuda:0")
    src = torch.randn(8, 32, dtype=dtype, device="cuda:1")
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.inplace_slice_copy(dest, src, 0, 32)
