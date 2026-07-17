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
"""Unit tests for GDN kernel optimizations: fused post-conv split/norm/gating,
decode QKV packing, the dense in_proj row permutation, and the multi-row
gated RMSNorm."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for triton kernels",
)


# ---- Reference implementations ----


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


def _ref_gdn_post_conv(
    prefill,
    decode,
    a,
    b,
    A_log,
    dt_bias,
    num_k_heads,
    head_k_dim,
    num_v_heads,
    head_v_dim,
    beta_dtype,
):
    """Reference GDN post-conv preparation."""
    mixed_qkv = prefill.transpose(0, 1)
    if decode is not None:
        mixed_qkv = torch.cat((mixed_qkv, decode), dim=0)

    q_dim = num_k_heads * head_k_dim
    v_dim = num_v_heads * head_v_dim
    q, k, v = torch.split(mixed_qkv, (q_dim, q_dim, v_dim), dim=-1)
    q = q.view(-1, num_k_heads, head_k_dim).contiguous()
    k = k.view(-1, num_k_heads, head_k_dim).contiguous()
    v = v.view(-1, num_v_heads, head_v_dim).contiguous()

    q_normalized = q.float() / torch.sqrt(
        torch.sum(q.float() * q.float(), dim=-1, keepdim=True) + 1e-6
    )
    k_normalized = k.float() / torch.sqrt(
        torch.sum(k.float() * k.float(), dim=-1, keepdim=True) + 1e-6
    )
    q = q_normalized.to(q.dtype)
    k = k_normalized.to(k.dtype)

    gate_input = a.float() + dt_bias.float()
    g = -torch.exp(A_log.float()) * F.softplus(gate_input, beta=1.0, threshold=20.0)
    beta = torch.sigmoid(b.float()).to(beta_dtype)
    return q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), g.unsqueeze(0), beta.unsqueeze(0)


# ---- Tests for fused_gdn_post_conv ----


@skip_no_cuda
@pytest.mark.parametrize(
    "num_k_heads,num_v_heads,head_k_dim,head_v_dim",
    [
        (16, 32, 128, 128),  # Qwen3.6-35B
        (16, 64, 128, 128),  # Qwen3.5-397B
        (4, 8, 64, 64),
    ],
)
@pytest.mark.parametrize(
    "num_prefill_tokens,num_decode_tokens,beta_dtype",
    [
        (1, 0, torch.float32),
        (127, 0, torch.bfloat16),
        (128, 16, torch.float32),
    ],
)
def test_fused_gdn_post_conv(
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
    num_prefill_tokens,
    num_decode_tokens,
    beta_dtype,
):
    """Test fused post-conv preparation for pure and mixed prefill."""
    from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import fused_gdn_post_conv

    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    qkv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
    num_tokens = num_prefill_tokens + num_decode_tokens

    prefill = torch.randn(qkv_dim, num_prefill_tokens, dtype=dtype, device=device)
    decode = (
        torch.randn(num_decode_tokens, qkv_dim, dtype=dtype, device=device)
        if num_decode_tokens
        else None
    )
    a = torch.randn(num_tokens, num_v_heads, dtype=dtype, device=device)
    b = torch.randn(num_tokens, num_v_heads, dtype=dtype, device=device)
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device) - 2.0
    dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1

    expected = _ref_gdn_post_conv(
        prefill,
        decode,
        a,
        b,
        A_log,
        dt_bias,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
        beta_dtype,
    )
    actual = fused_gdn_post_conv(
        prefill,
        decode,
        a,
        b,
        A_log,
        dt_bias,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
        beta_dtype=beta_dtype,
    )

    for output in actual:
        assert output.is_contiguous()
    for actual_output, expected_output in zip(actual[:3], expected[:3]):
        torch.testing.assert_close(actual_output, expected_output, rtol=1e-2, atol=1e-2)
    for actual_output, expected_output in zip(actual[3:], expected[3:]):
        torch.testing.assert_close(actual_output, expected_output, rtol=1e-4, atol=1e-4)
    assert actual[4].dtype == beta_dtype
    torch.testing.assert_close(actual[4].to(b.dtype), b.sigmoid().unsqueeze(0), rtol=0, atol=0)


@skip_no_cuda
def test_fused_gdn_post_conv_empty():
    """Test the empty-input fast path."""
    from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import fused_gdn_post_conv

    device = torch.device("cuda")
    num_k_heads, num_v_heads = 4, 8
    head_k_dim = head_v_dim = 64
    qkv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
    prefill = torch.empty(qkv_dim, 0, dtype=torch.bfloat16, device=device)
    a = torch.empty(0, num_v_heads, dtype=torch.bfloat16, device=device)
    b = torch.empty_like(a)
    A_log = torch.empty(num_v_heads, dtype=torch.float32, device=device)
    dt_bias = torch.empty_like(A_log)

    q, k, v, g, beta = fused_gdn_post_conv(
        prefill,
        None,
        a,
        b,
        A_log,
        dt_bias,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
    )

    assert q.shape == (1, 0, num_k_heads, head_k_dim)
    assert k.shape == q.shape
    assert v.shape == (1, 0, num_v_heads, head_v_dim)
    assert g.shape == (1, 0, num_v_heads)
    assert beta.shape == g.shape


# ---- Strided (column-slice view) inputs, as produced by the dense in_proj ----


@skip_no_cuda
def test_gdn_gating_strided_views():
    """a sliced out of the packed ba projection must match packed inputs."""
    from tensorrt_llm._torch.modules.mamba.gdn_mixer import fused_gdn_gating

    torch.manual_seed(42)
    device = torch.device("cuda")
    ba = torch.randn(300, 64, dtype=torch.bfloat16, device=device)
    a_view = ba[:, 32:]

    A_log = torch.randn(32, dtype=torch.float32, device=device)
    dt_bias = torch.randn(32, dtype=torch.float32, device=device)

    assert torch.equal(
        fused_gdn_gating(A_log, a_view, dt_bias),
        fused_gdn_gating(A_log, a_view.contiguous(), dt_bias),
    )


@skip_no_cuda
def test_transpose_helpers_strided_views():
    """The prefill transpose helper must read column-slice views in place."""
    from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import (
        extract_transpose_prefill_slice,
    )

    torch.manual_seed(42)
    device = torch.device("cuda")
    wide = torch.randn(100, 1536, dtype=torch.bfloat16, device=device)
    view = wide[:, :1024]  # rows contiguous, row stride 1536

    out = extract_transpose_prefill_slice(view, 100, 0, 1024)
    assert torch.equal(out, view.T.contiguous())


@skip_no_cuda
def test_fused_gdn_post_conv_strided_views():
    """a/b and decode sliced out of wider projections must match packed inputs."""
    from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import fused_gdn_post_conv

    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_k_heads, num_v_heads, head_k_dim, head_v_dim = 4, 8, 64, 64
    qkv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
    num_prefill_tokens, num_decode_tokens = 33, 7
    num_tokens = num_prefill_tokens + num_decode_tokens

    prefill = torch.randn(qkv_dim, num_prefill_tokens, dtype=dtype, device=device)
    wide_qkvz = torch.randn(num_decode_tokens, qkv_dim + 256, dtype=dtype, device=device)
    decode_view = wide_qkvz[:, :qkv_dim]  # rows contiguous, wider row stride
    ba = torch.randn(num_tokens, 2 * num_v_heads, dtype=dtype, device=device)
    b_view, a_view = ba[:, :num_v_heads], ba[:, num_v_heads:]
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device) - 2.0
    dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device) * 0.1

    args = (A_log, dt_bias, num_k_heads, head_k_dim, num_v_heads, head_v_dim)
    outs_view = fused_gdn_post_conv(prefill, decode_view, a_view, b_view, *args)
    outs_packed = fused_gdn_post_conv(
        prefill, decode_view.contiguous(), a_view.contiguous(), b_view.contiguous(), *args
    )
    for got, ref in zip(outs_view, outs_packed):
        assert torch.equal(got, ref)


@skip_no_cuda
@pytest.mark.parametrize(
    "num_k_heads,num_v_heads,head_k_dim,head_v_dim",
    [
        (16, 32, 128, 128),  # Qwen3.6-35B
        (16, 64, 128, 128),  # Qwen3.5-397B
        (4, 8, 64, 64),
    ],
)
@pytest.mark.parametrize("num_tokens", [1, 16, 127])
@pytest.mark.parametrize("strided_input", [False, True])
def test_pack_gdn_decode_qkv(
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
    num_tokens,
    strided_input,
):
    """Test target-verification decode packing without normalization."""
    from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import pack_gdn_decode_qkv

    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    q_dim = num_k_heads * head_k_dim
    v_dim = num_v_heads * head_v_dim
    qkv_dim = 2 * q_dim + v_dim
    if strided_input:
        mixed_qkv = torch.randn(num_tokens, qkv_dim * 2, dtype=dtype, device=device)[:, ::2]
    else:
        mixed_qkv = torch.randn(num_tokens, qkv_dim, dtype=dtype, device=device)

    expected_q, expected_k, expected_v = torch.split(mixed_qkv, (q_dim, q_dim, v_dim), dim=-1)
    actual_q, actual_k, actual_v = pack_gdn_decode_qkv(
        mixed_qkv,
        num_k_heads,
        head_k_dim,
        num_v_heads,
        head_v_dim,
    )

    for output in (actual_q, actual_k, actual_v):
        assert output.is_contiguous()
    torch.testing.assert_close(actual_q.view(num_tokens, -1), expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_k.view(num_tokens, -1), expected_k, rtol=0, atol=0)
    torch.testing.assert_close(actual_v.view(num_tokens, -1), expected_v, rtol=0, atol=0)


# ---- Tests for the GDN compile boundary and derived state ----


def test_gdn_custom_op_forwards_split_inputs(monkeypatch):
    """The compile boundary consumes split inputs and mutates its output."""
    import tensorrt_llm._torch.modules.mamba.gdn_mixer as gdn_mixer

    class FakeMetadata:
        num_tokens = 2
        mamba_metadata = object()

    class FakeLayer:
        def forward_core(
            self,
            mixed_qkv,
            a,
            b,
            attn_metadata,
            mamba_metadata,
            spec_metadata,
            output,
        ):
            assert mixed_qkv.shape[0] == 2
            assert a.shape[0] == 2
            assert b.shape[0] == 2
            assert attn_metadata is metadata
            assert mamba_metadata is metadata.mamba_metadata
            assert spec_metadata is None
            output.fill_(5)

    metadata = FakeMetadata()
    monkeypatch.setattr(
        gdn_mixer,
        "_extract_gdn_extra_attrs",
        lambda layer_idx: (metadata, FakeLayer(), None),
    )
    mixed_qkv = torch.zeros(3, 8)
    a = torch.zeros(3, 2)
    b = torch.zeros(3, 2)
    output = torch.zeros(1, 3, 2, 2)

    gdn_mixer.gdn_custom_op_inplace(mixed_qkv, a, b, "0", output)

    torch.testing.assert_close(output[:, :2], torch.full_like(output[:, :2], 5))
    torch.testing.assert_close(output[:, 2:], torch.zeros_like(output[:, 2:]))


def test_gdn_attaches_only_static_fp8_scale():
    from tensorrt_llm._torch.modules.linear import FP8QDQLinearMethod
    from tensorrt_llm._torch.modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet
    from tensorrt_llm._torch.modules.mamba.layernorm_gated import RMSNorm

    layer = Qwen3NextGatedDeltaNet.__new__(Qwen3NextGatedDeltaNet)
    torch.nn.Module.__init__(layer)
    layer.norm = RMSNorm(8)
    scale = torch.nn.Parameter(torch.tensor(0.13), requires_grad=False)
    layer.out_proj = SimpleNamespace(
        quant_method=FP8QDQLinearMethod(),
        input_scale=scale,
        force_dynamic_quantization=False,
    )

    layer.cache_derived_state()
    assert not isinstance(layer.norm.fp8_scale, torch.nn.Parameter)
    assert "fp8_scale" not in layer.norm.state_dict()
    assert "fp8_scale" not in dict(layer.norm.named_parameters())
    assert torch.equal(layer.norm.fp8_scale, scale)
    assert layer.norm.fp8_scale.data_ptr() == scale.data_ptr()

    layer.out_proj.force_dynamic_quantization = True
    layer.cache_derived_state()
    assert layer.norm.fp8_scale is None
