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
"""Unit tests for fused elementwise operations in Mamba2 prefill."""

import pytest
import torch

from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import (
    extract_transpose_xbc_prefill,
    fused_split_rearrange_after_conv1d,
    ssd_output_transpose,
)

skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for triton kernels",
)


def extract_transpose_xbc_prefill_ref(
    zxbcdt: torch.Tensor,
    num_prefill_tokens: int,
    d_inner: int,
    conv_dim: int,
) -> torch.Tensor:
    """Reference implementation for extract_transpose_xbc_prefill."""
    # Extract the xbc slice and transpose
    xbc = zxbcdt[:num_prefill_tokens, d_inner : d_inner + conv_dim]
    return xbc.transpose(0, 1).contiguous()


def fused_split_rearrange_after_conv1d_ref(
    xbc: torch.Tensor,
    d_inner: int,
    n_groups: int,
    d_state: int,
    nheads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation for fused_split_rearrange_after_conv1d."""
    conv_dim, num_prefill_tokens = xbc.shape
    bc_size = n_groups * d_state

    # Transpose and split
    xbc_t = xbc.transpose(0, 1).contiguous()  # [num_prefill_tokens, conv_dim]
    x, B, C = torch.split(xbc_t, [d_inner, bc_size, bc_size], dim=-1)
    x = x.contiguous().view(1, num_prefill_tokens, nheads, head_dim)
    B = B.contiguous().view(1, num_prefill_tokens, n_groups, d_state)
    C = C.contiguous().view(1, num_prefill_tokens, n_groups, d_state)
    return x, B, C


@skip_no_cuda
@pytest.mark.parametrize("num_prefill_tokens", [1, 32, 128, 1024])
@pytest.mark.parametrize(
    "d_inner,conv_dim,d_in_proj", [(256, 512, 800), (512, 1024, 1600), (1024, 2048, 3200)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_extract_transpose_xbc_prefill(num_prefill_tokens, d_inner, conv_dim, d_in_proj, dtype):
    """Test extract_transpose_xbc_prefill matches reference implementation."""
    torch.manual_seed(42)
    device = torch.device("cuda")

    num_total_tokens = num_prefill_tokens + 16
    zxbcdt = torch.randn(num_total_tokens, d_in_proj, dtype=dtype, device=device)
    out_ref = extract_transpose_xbc_prefill_ref(zxbcdt, num_prefill_tokens, d_inner, conv_dim)
    out_fused = extract_transpose_xbc_prefill(zxbcdt, num_prefill_tokens, d_inner, conv_dim)

    assert out_fused.shape == out_ref.shape, f"Shape mismatch: {out_fused.shape} vs {out_ref.shape}"
    torch.testing.assert_close(out_fused, out_ref, rtol=1e-3, atol=1e-3)


@skip_no_cuda
@pytest.mark.parametrize("num_prefill_tokens", [1, 32, 128, 1024])
@pytest.mark.parametrize(
    "nheads,head_dim,n_groups,d_state", [(8, 64, 1, 128), (16, 64, 2, 64), (32, 64, 4, 64)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_split_rearrange_after_conv1d(
    num_prefill_tokens, nheads, head_dim, n_groups, d_state, dtype
):
    """Test fused_split_rearrange_after_conv1d matches reference implementation."""
    torch.manual_seed(42)
    device = torch.device("cuda")

    d_inner = nheads * head_dim
    bc_size = n_groups * d_state
    conv_dim = d_inner + 2 * bc_size
    xbc = torch.randn(conv_dim, num_prefill_tokens, dtype=dtype, device=device)
    x_ref, B_ref, C_ref = fused_split_rearrange_after_conv1d_ref(
        xbc, d_inner, n_groups, d_state, nheads, head_dim
    )
    x_fused, B_fused, C_fused = fused_split_rearrange_after_conv1d(
        xbc, d_inner, n_groups, d_state, nheads, head_dim
    )

    assert x_fused.shape == x_ref.shape, f"x shape mismatch: {x_fused.shape} vs {x_ref.shape}"
    assert B_fused.shape == B_ref.shape, f"B shape mismatch: {B_fused.shape} vs {B_ref.shape}"
    assert C_fused.shape == C_ref.shape, f"C shape mismatch: {C_fused.shape} vs {C_ref.shape}"
    torch.testing.assert_close(x_fused, x_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(B_fused, B_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(C_fused, C_ref, rtol=1e-3, atol=1e-3)


@skip_no_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_extract_transpose_large_input_no_overflow(dtype):
    """
    Test extract_transpose_xbc_prefill with large inputs that would overflow int32.

    This test verifies the fix for integer overflow in Triton kernel offset calculations.
    For NemotronH Nano 12B with max_num_tokens=131072:
    - d_in_proj = 22656
    - max_src_offset = (num_prefill_tokens - 1) * d_in_proj > INT32_MAX

    We use smaller but still overflow-inducing parameters to keep the test memory-efficient.
    """
    torch.manual_seed(42)
    device = torch.device("cuda")

    # Parameters that would cause int32 overflow: 100000 * 22000 = 2.2B > INT32_MAX
    num_prefill_tokens = 100000
    d_in_proj = 22000
    d_inner = 10000
    conv_dim = 12000

    # Check available GPU memory - skip if not enough
    free_memory = torch.cuda.get_device_properties(
        device
    ).total_memory - torch.cuda.memory_allocated(device)
    required_memory = num_prefill_tokens * d_in_proj * 2 * 3  # input + output + overhead
    if free_memory < required_memory:
        pytest.skip(
            f"Insufficient GPU memory: {free_memory / 1e9:.1f}GB available, {required_memory / 1e9:.1f}GB required"
        )

    zxbcdt = torch.randn(num_prefill_tokens, d_in_proj, dtype=dtype, device=device)
    out_ref = extract_transpose_xbc_prefill_ref(zxbcdt, num_prefill_tokens, d_inner, conv_dim)
    out_fused = extract_transpose_xbc_prefill(zxbcdt, num_prefill_tokens, d_inner, conv_dim)

    assert out_fused.shape == out_ref.shape, f"Shape mismatch: {out_fused.shape} vs {out_ref.shape}"
    torch.testing.assert_close(out_fused, out_ref, rtol=1e-3, atol=1e-3)


def ssd_output_transpose_ref(
    out_contig: torch.Tensor,
    num_prefill_tokens: int,
) -> torch.Tensor:
    """Reference implementation: permute+reshape+slice."""
    B, H, D, NC, CS = out_contig.shape
    seqlen = NC * CS
    out_view = out_contig.permute(0, 3, 4, 1, 2).reshape(B, seqlen, H, D)
    dst = out_view[:, :num_prefill_tokens].contiguous().view(num_prefill_tokens, H * D)
    return dst


@skip_no_cuda
@pytest.mark.parametrize(
    "H,D,CS,num_prefill_tokens",
    [
        (32, 64, 256, 1),
        (32, 64, 256, 128),
        (32, 64, 256, 1024),
        (32, 64, 256, 4096),
        (32, 64, 256, 50176),  # exact multiple of CS
        (32, 64, 256, 50224),  # trailing padding
        (16, 64, 128, 8192),
        (64, 64, 256, 16384),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_ssd_output_transpose(H, D, CS, num_prefill_tokens, dtype):
    torch.manual_seed(42)
    device = torch.device("cuda")

    NC = (num_prefill_tokens + CS - 1) // CS
    out_contig = torch.randn(1, H, D, NC, CS, dtype=dtype, device=device)

    dst_ref = ssd_output_transpose_ref(out_contig, num_prefill_tokens)
    dst = torch.empty(num_prefill_tokens, H * D, dtype=dtype, device=device)
    ssd_output_transpose(out_contig, dst, num_prefill_tokens)

    assert dst.shape == dst_ref.shape
    # Pure data movement — must be bit-exact.
    torch.testing.assert_close(dst, dst_ref, rtol=0, atol=0)
