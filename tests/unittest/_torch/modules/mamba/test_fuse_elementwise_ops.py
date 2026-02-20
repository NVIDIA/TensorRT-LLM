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
