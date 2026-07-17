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
"""Unit tests for GDN kernel optimizations: fused gating+sigmoid, split_qkv,
transpose_and_split, the dense in_proj row permutation, and the multi-row
gated RMSNorm."""

import pytest
import torch
import torch.nn.functional as F

skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for triton kernels",
)


@pytest.mark.parametrize(
    "value,expected",
    [(None, False), ("", False), ("0", False), ("false", False), ("typo", False), ("1", True)],
)
def test_gdn_replay_env_requires_one(monkeypatch, value, expected):
    from tensorrt_llm._torch.utils import is_gdn_replay_enabled

    if value is None:
        monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_USE_GDN_REPLAY", value)

    assert is_gdn_replay_enabled() is expected


# ---- Reference implementations ----


def _ref_gdn_gating_with_sigmoid(A_log, a, dt_bias, b, beta=1.0, threshold=20.0):
    """Reference: g = -exp(A_log) * softplus(a + dt_bias), beta_out = sigmoid(b)."""
    g = -torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias.float(), beta, threshold)
    beta_out = torch.sigmoid(b.float()).to(b.dtype)
    return g, beta_out


def _ref_split_grouped_qkvz(y, num_groups, head_k_dim, head_v_dim, heads_ratio):
    """Split a grouped-interleaved in_proj_qkvz output ([q|k|v|z] per group)."""
    bsz = y.shape[0]
    g = y.view(bsz, num_groups, 2 * head_k_dim + 2 * heads_ratio * head_v_dim)
    q = g[..., :head_k_dim].reshape(bsz, num_groups * head_k_dim)
    k = g[..., head_k_dim : 2 * head_k_dim].reshape(bsz, num_groups * head_k_dim)
    v = g[..., 2 * head_k_dim : 2 * head_k_dim + heads_ratio * head_v_dim].reshape(
        bsz, num_groups * heads_ratio * head_v_dim
    )
    z = g[..., 2 * head_k_dim + heads_ratio * head_v_dim :].reshape(
        bsz, num_groups * heads_ratio, head_v_dim
    )
    return q, k, v, z


def _ref_split_grouped_ba(y, num_groups, heads_ratio):
    bsz = y.shape[0]
    g = y.view(bsz, num_groups, 2 * heads_ratio)
    b = g[..., :heads_ratio].reshape(bsz, num_groups * heads_ratio)
    a = g[..., heads_ratio:].reshape(bsz, num_groups * heads_ratio)
    return b, a


# ---- Tests for the dense in_proj row permutation ----


@pytest.mark.parametrize(
    "num_k_heads,num_v_heads,head_k_dim,head_v_dim",
    [(16, 32, 128, 128), (16, 16, 128, 128), (8, 32, 64, 64)],
)
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_grouped_to_dense_in_proj_perm(num_k_heads, num_v_heads, head_k_dim, head_v_dim, tp_size):
    """Permuting in_proj rows to the dense layout must make plain column
    slices of each TP rank's projection output reproduce the grouped split."""
    from tensorrt_llm._torch.models.checkpoints.hf.qwen3_next_weight_mapper import (
        _permute_rows,
        grouped_to_dense_in_proj_ba_perm,
        grouped_to_dense_in_proj_qkvz_perm,
    )

    torch.manual_seed(42)
    ratio = num_v_heads // num_k_heads
    rows = 2 * num_k_heads * head_k_dim + 2 * num_v_heads * head_v_dim
    w = torch.randn(rows, 32, dtype=torch.bfloat16)
    w_ba = torch.randn(2 * num_v_heads, 32, dtype=torch.bfloat16)
    x = torch.randn(4, 32, dtype=torch.float32)

    perm = grouped_to_dense_in_proj_qkvz_perm(
        num_k_heads, head_k_dim, num_v_heads, head_v_dim, tp_size
    )
    perm_ba = grouped_to_dense_in_proj_ba_perm(num_k_heads, num_v_heads, tp_size)
    assert torch.equal(perm.sort().values, torch.arange(rows))
    w_dense = _permute_rows(w, perm)
    w_ba_dense = _permute_rows(w_ba, perm_ba)

    for rank in range(tp_size):
        ng = num_k_heads // tp_size
        shard = rows // tp_size
        y_grouped = x @ w[rank * shard : (rank + 1) * shard].T.float()
        y_dense = x @ w_dense[rank * shard : (rank + 1) * shard].T.float()
        q_ref, k_ref, v_ref, z_ref = _ref_split_grouped_qkvz(
            y_grouped, ng, head_k_dim, head_v_dim, ratio
        )
        k_end = 2 * ng * head_k_dim
        qkv_dim = k_end + ng * ratio * head_v_dim
        assert torch.equal(y_dense[:, : ng * head_k_dim], q_ref)
        assert torch.equal(y_dense[:, ng * head_k_dim : k_end], k_ref)
        assert torch.equal(y_dense[:, k_end:qkv_dim], v_ref)
        assert torch.equal(y_dense[:, qkv_dim:].view(4, ng * ratio, head_v_dim), z_ref)

        shard_ba = 2 * num_v_heads // tp_size
        yb_grouped = x @ w_ba[rank * shard_ba : (rank + 1) * shard_ba].T.float()
        yb_dense = x @ w_ba_dense[rank * shard_ba : (rank + 1) * shard_ba].T.float()
        b_ref, a_ref = _ref_split_grouped_ba(yb_grouped, ng, ratio)
        assert torch.equal(yb_dense[:, : ng * ratio], b_ref)
        assert torch.equal(yb_dense[:, ng * ratio :], a_ref)


def test_in_proj_perm_quantized_dtypes():
    """Row permutation must be dtype-agnostic (fp8, packed-uint8, 1-D) and
    consistent with FP8 2D-block scale permutation."""
    from tensorrt_llm._torch.models.checkpoints.hf.qwen3_next_weight_mapper import (
        _permute_rows,
        _rows_to_scale_block_perm,
        grouped_to_dense_in_proj_qkvz_perm,
    )

    torch.manual_seed(42)
    perm = grouped_to_dense_in_proj_qkvz_perm(16, 128, 32, 128, 1)
    w_fp8 = torch.randn(12288, 64).to(torch.float8_e4m3fn)
    assert torch.equal(_permute_rows(w_fp8, perm).float(), w_fp8.float()[perm])
    w_packed = torch.randint(0, 255, (12288, 1024), dtype=torch.uint8)
    assert torch.equal(_permute_rows(w_packed, perm), w_packed[perm])
    w_1d = torch.randn(12288, dtype=torch.bfloat16)
    assert torch.equal(_permute_rows(w_1d, perm), w_1d[perm])

    # Expanding block scales to rows then permuting rows must equal permuting
    # scale blocks then expanding.
    scale = torch.randn(12288 // 128, 7)
    block_perm = _rows_to_scale_block_perm(perm, 128)
    lhs = scale.repeat_interleave(128, dim=0)[perm]
    rhs = _permute_rows(scale, block_perm).repeat_interleave(128, dim=0)
    assert torch.equal(lhs, rhs)


# ---- Tests for the multi-row gated RMSNorm ----


def _ref_gated_rmsnorm(x, w, z, eps):
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = xf * rstd * w.float()
    zf = z.float()
    return (y * zf * torch.sigmoid(zf)).to(x.dtype)


@skip_no_cuda
@pytest.mark.parametrize(
    "num_tokens,heads,N",
    [(8192, 32, 128), (7, 32, 128), (1, 1, 128), (333, 4, 64), (1024, 16, 256)],
)
def test_rms_norm_gated_token_major(num_tokens, heads, N):
    """Token-major z (a column-slice view of a wider projection) must match
    the reference on both the multi-row fast path and the generic fallback."""
    from tensorrt_llm._torch.modules.mamba.layernorm_gated import (
        _layer_norm_fwd,
        rms_norm_gated_token_major,
    )

    torch.manual_seed(42)
    device = torch.device("cuda")
    M = num_tokens * heads
    x = torch.randn(M, N, dtype=torch.bfloat16, device=device)
    w = torch.rand(N, dtype=torch.bfloat16, device=device) + 0.5
    wide = torch.randn(num_tokens, heads * N + 512, dtype=torch.bfloat16, device=device)
    z = wide[:, 512:].view(num_tokens, heads, N)

    y = rms_norm_gated_token_major(x, z, w, 1e-6)
    ref = _ref_gated_rmsnorm(x, w, z.reshape(M, N), 1e-6)
    torch.testing.assert_close(y, ref, rtol=1e-2, atol=1e-2)

    # The dense-z dispatch of the generic entry point must agree bitwise.
    y_dense, _, _ = _layer_norm_fwd(
        x,
        w,
        None,
        1e-6,
        z=z.reshape(M, N).contiguous(),
        norm_before_gate=True,
        is_rms_norm=True,
    )
    assert torch.equal(y, y_dense)


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


# ---- Strided (column-slice view) inputs, as produced by the dense in_proj ----


@skip_no_cuda
def test_gdn_gating_strided_views():
    """a/b sliced out of the packed ba projection must match packed inputs."""
    from tensorrt_llm._torch.modules.mamba.gdn_mixer import (
        fused_gdn_gating,
        fused_gdn_gating_with_sigmoid,
    )

    torch.manual_seed(42)
    device = torch.device("cuda")
    ba = torch.randn(300, 64, dtype=torch.bfloat16, device=device)
    b_view, a_view = ba[:, :32], ba[:, 32:]

    A_log = torch.randn(32, dtype=torch.float32, device=device)
    dt_bias = torch.randn(32, dtype=torch.float32, device=device)

    g_v, beta_v = fused_gdn_gating_with_sigmoid(A_log, a_view, dt_bias, b_view)
    g_c, beta_c = fused_gdn_gating_with_sigmoid(
        A_log, a_view.contiguous(), dt_bias, b_view.contiguous()
    )
    assert torch.equal(g_v, g_c) and torch.equal(beta_v, beta_c)
    g_ref, beta_ref = _ref_gdn_gating_with_sigmoid(A_log, a_view, dt_bias, b_view)
    torch.testing.assert_close(g_v, g_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(beta_v, beta_ref.float(), rtol=1e-2, atol=1e-2)

    assert torch.equal(
        fused_gdn_gating(A_log, a_view, dt_bias),
        fused_gdn_gating(A_log, a_view.contiguous(), dt_bias),
    )


@skip_no_cuda
def test_transpose_helpers_strided_views():
    """The prefill transpose/split helpers must read column-slice views in place."""
    from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import (
        extract_transpose_prefill_slice,
        transpose_and_split_qkv,
    )

    torch.manual_seed(42)
    device = torch.device("cuda")
    wide = torch.randn(100, 1536, dtype=torch.bfloat16, device=device)
    view = wide[:, :1024]  # rows contiguous, row stride 1536

    out = extract_transpose_prefill_slice(view, 100, 0, 1024)
    assert torch.equal(out, view.T.contiguous())

    prefill_t = torch.randn(1024, 40, dtype=torch.bfloat16, device=device)
    decode_view = wide[40:80, :1024]
    outs_view = transpose_and_split_qkv(prefill_t, decode_view, 256, 256, 512, 16, 16, 16, 32)
    outs_packed = transpose_and_split_qkv(
        prefill_t, decode_view.contiguous(), 256, 256, 512, 16, 16, 16, 32
    )
    for got, ref in zip(outs_view, outs_packed):
        assert torch.equal(got, ref)
