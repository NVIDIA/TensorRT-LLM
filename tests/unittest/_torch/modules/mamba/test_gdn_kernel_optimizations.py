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
"""Unit tests for GDN kernel optimizations: fused gating+sigmoid, split_qkv, transpose_and_split."""

import pytest
import torch
import torch.nn.functional as F

skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for triton kernels",
)


# ---- Reference implementations ----


def _ref_gdn_gating_with_sigmoid(A_log, a, dt_bias, b, beta=1.0, threshold=20.0):
    """Reference: g = -exp(A_log) * softplus(a + dt_bias), beta_out = sigmoid(b)."""
    g = -torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias.float(), beta, threshold)
    beta_out = torch.sigmoid(b.float()).to(b.dtype)
    return g, beta_out


def _ref_split_qkv_contiguous(mixed_qkv, q_dim, k_dim, v_dim):
    """Reference: torch.split + contiguous."""
    q, k, v = torch.split(mixed_qkv, [q_dim, k_dim, v_dim], dim=-1)
    return q.contiguous(), k.contiguous(), v.contiguous()


def _ref_transpose_and_split_qkv(prefill_t, decode, q_dim, k_dim, v_dim):
    """Reference: transpose prefill [D,T] -> [T,D], cat with decode, then split."""
    prefill = prefill_t.transpose(0, 1).contiguous()
    mixed = torch.cat([prefill, decode], dim=0)
    q, k, v = torch.split(mixed, [q_dim, k_dim, v_dim], dim=-1)
    return q.contiguous(), k.contiguous(), v.contiguous()


# ---- Tests for fused_gdn_gating_with_sigmoid ----


@skip_no_cuda
@pytest.mark.parametrize("batch_size", [1, 16, 128, 512])
@pytest.mark.parametrize("num_heads", [16, 30])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_gdn_gating_with_sigmoid(batch_size, num_heads, dtype):
    """Test fused gating+sigmoid matches separate sigmoid + gating."""
    from tensorrt_llm._torch.modules.mamba.gdn_mixer import fused_gdn_gating_with_sigmoid

    torch.manual_seed(42)
    device = torch.device("cuda")

    A_log = torch.randn(num_heads, dtype=dtype, device=device)
    a = torch.randn(batch_size, num_heads, dtype=dtype, device=device)
    dt_bias = torch.randn(num_heads, dtype=dtype, device=device)
    b = torch.randn(batch_size, num_heads, dtype=dtype, device=device)

    g_ref, beta_ref = _ref_gdn_gating_with_sigmoid(A_log, a, dt_bias, b)
    g_fused, beta_fused = fused_gdn_gating_with_sigmoid(A_log, a, dt_bias, b)

    assert g_fused.shape == g_ref.shape
    assert beta_fused.shape == beta_ref.shape
    torch.testing.assert_close(g_fused, g_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(beta_fused.float(), beta_ref.float(), rtol=1e-2, atol=1e-2)


# ---- Tests for split_qkv_contiguous ----


@skip_no_cuda
@pytest.mark.parametrize("seq_len", [1, 32, 128, 1024])
@pytest.mark.parametrize(
    "q_dim,k_dim,v_dim,num_q_heads,head_k_dim,num_v_heads,head_v_dim",
    [
        (256, 256, 16, 16, 16, 16, 1),  # Qwen3.5-35B like
        (512, 512, 64, 16, 32, 16, 4),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_split_qkv_contiguous(
    seq_len, q_dim, k_dim, v_dim, num_q_heads, head_k_dim, num_v_heads, head_v_dim, dtype
):
    """Test split_qkv_contiguous matches torch.split + contiguous."""
    from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import split_qkv_contiguous

    torch.manual_seed(42)
    device = torch.device("cuda")

    total_dim = q_dim + k_dim + v_dim
    mixed_qkv = torch.randn(seq_len, total_dim, dtype=dtype, device=device)

    q_ref, k_ref, v_ref = _ref_split_qkv_contiguous(mixed_qkv, q_dim, k_dim, v_dim)
    q_out, k_out, v_out = split_qkv_contiguous(
        mixed_qkv,
        q_dim,
        k_dim,
        v_dim,
        num_q_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
    )

    # Output is 4D [1, T, heads, head_dim], ref is 2D [T, dim]
    assert q_out.shape == (1, seq_len, num_q_heads, head_k_dim)
    assert k_out.shape == (1, seq_len, num_q_heads, head_k_dim)
    assert v_out.shape == (1, seq_len, num_v_heads, head_v_dim)

    torch.testing.assert_close(q_out.view(seq_len, -1), q_ref, rtol=0, atol=0)
    torch.testing.assert_close(k_out.view(seq_len, -1), k_ref, rtol=0, atol=0)
    torch.testing.assert_close(v_out.view(seq_len, -1), v_ref, rtol=0, atol=0)


# ---- Tests for transpose_and_split_qkv ----


@skip_no_cuda
@pytest.mark.parametrize("num_prefill,num_decode", [(32, 8), (128, 16), (1, 1), (256, 0)])
@pytest.mark.parametrize(
    "q_dim,k_dim,v_dim,num_q_heads,head_k_dim,num_v_heads,head_v_dim",
    [
        (256, 256, 16, 16, 16, 16, 1),
        (512, 512, 64, 16, 32, 16, 4),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_transpose_and_split_qkv(
    num_prefill,
    num_decode,
    q_dim,
    k_dim,
    v_dim,
    num_q_heads,
    head_k_dim,
    num_v_heads,
    head_v_dim,
    dtype,
):
    """Test transpose_and_split_qkv matches transpose + cat + split."""
    from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import transpose_and_split_qkv

    if num_decode == 0:
        pytest.skip("transpose_and_split_qkv requires decode tokens")

    torch.manual_seed(42)
    device = torch.device("cuda")

    total_dim = q_dim + k_dim + v_dim
    prefill_t = torch.randn(total_dim, num_prefill, dtype=dtype, device=device)  # [D, T_p]
    decode = torch.randn(num_decode, total_dim, dtype=dtype, device=device)  # [T_d, D]

    q_ref, k_ref, v_ref = _ref_transpose_and_split_qkv(
        prefill_t,
        decode,
        q_dim,
        k_dim,
        v_dim,
    )
    q_out, k_out, v_out = transpose_and_split_qkv(
        prefill_t,
        decode,
        q_dim,
        k_dim,
        v_dim,
        num_q_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
    )

    total_seq = num_prefill + num_decode
    assert q_out.shape == (1, total_seq, num_q_heads, head_k_dim)
    assert k_out.shape == (1, total_seq, num_q_heads, head_k_dim)
    assert v_out.shape == (1, total_seq, num_v_heads, head_v_dim)

    torch.testing.assert_close(q_out.view(total_seq, -1), q_ref, rtol=0, atol=0)
    torch.testing.assert_close(k_out.view(total_seq, -1), k_ref, rtol=0, atol=0)
    torch.testing.assert_close(v_out.view(total_seq, -1), v_ref, rtol=0, atol=0)
