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
decode QKV packing, the dense in_proj row permutation, the multi-row
gated RMSNorm, the conv1d verify options and the fused MTP target verify."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers import Qwen3NextConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet
from tensorrt_llm._torch.utils import torch_compiling

skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for triton kernels",
)


@pytest.mark.parametrize(
    "value,expected",
    [(None, True), ("", False), ("0", False), ("false", False), ("typo", False), ("1", True)],
)
def test_gdn_replay_env_defaults_on_and_requires_one_when_set(monkeypatch, value, expected):
    from tensorrt_llm._torch.utils import is_gdn_replay_enabled

    if value is None:
        monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_USE_GDN_REPLAY", value)

    assert is_gdn_replay_enabled() is expected


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


def _ref_gated_rmsnorm(x, w, z, eps, gate_activation):
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = xf * rstd * w.float()
    zf = z.float()
    gate = torch.sigmoid(zf)
    if gate_activation == "silu":
        gate *= zf
    return (y * gate).to(x.dtype)


@skip_no_cuda
@pytest.mark.parametrize("gate_activation", ["silu", "sigmoid"])
@pytest.mark.parametrize(
    "num_tokens,heads,N",
    [
        (8192, 32, 128),
        (7, 32, 128),
        (1, 1, 128),
        (333, 4, 64),
        (1024, 16, 256),
        (5, 3, 512),
    ],
)
def test_rms_norm_gated_token_major(gate_activation, num_tokens, heads, N):
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

    y = rms_norm_gated_token_major(x, z, w, 1e-6, gate_activation=gate_activation)
    ref = _ref_gated_rmsnorm(x, w, z.reshape(M, N), 1e-6, gate_activation)
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
        gate_activation=gate_activation,
    )
    assert torch.equal(y, y_dense)


def test_rms_norm_gated_token_major_custom_op_is_functional():
    """The compile boundary returns fresh storage and mutates no arguments."""
    from tensorrt_llm._torch.modules.mamba import layernorm_gated  # noqa: F401

    op = torch.ops.trtllm.rms_norm_gated_token_major.default
    assert all(
        argument.alias_info is None or not argument.alias_info.is_write
        for argument in op._schema.arguments
    )
    assert op._schema.returns[0].alias_info is None


@skip_no_cuda
@pytest.mark.parametrize("output_fp8", [False, True])
@pytest.mark.parametrize("gate_activation", ["silu", "sigmoid"])
def test_rms_norm_gated_token_major_compiles_fullgraph(output_fp8, gate_activation):
    """The functional custom op must remain opaque to torch.compile."""
    from tensorrt_llm._torch.modules.mamba.layernorm_gated import rms_norm_gated_token_major

    torch.manual_seed(42)
    num_tokens, heads, N = 16, 2, 128
    M = num_tokens * heads
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    weight = torch.rand(N, dtype=torch.bfloat16, device="cuda") + 0.5
    wide = torch.randn(num_tokens, heads * N + 64, dtype=torch.bfloat16, device="cuda")
    z = wide[:, 64:].view(num_tokens, heads, N)
    fp8_scale = torch.tensor(0.13, device="cuda") if output_fp8 else None

    expected = rms_norm_gated_token_major(x, z, weight, 1e-6, fp8_scale, gate_activation)
    compiled = torch.compile(rms_norm_gated_token_major, fullgraph=True)
    actual = compiled(x, z, weight, 1e-6, fp8_scale, gate_activation)

    assert torch.equal(actual, expected)


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


@skip_no_cuda
def test_recurrent_state_reset_supports_gdn_pool_views():
    from tensorrt_llm._torch.modules.mamba.recurrent_state_cache import reset_recurrent_state_rows

    num_slots = 4
    ssm_state_size = 6
    conv_state_size = 8
    state_pool = (
        torch.arange(
            num_slots * (ssm_state_size + conv_state_size),
            device="cuda",
            dtype=torch.float32,
        )
        .to(torch.bfloat16)
        .reshape(num_slots, -1)
    )
    state_pool_before = state_pool.clone()
    ssm_states = state_pool[:, :ssm_state_size].view(num_slots, 2, 3)
    conv_states = state_pool[:, ssm_state_size:].view(num_slots, 2, 4)
    state_indices = torch.tensor([1, 3, -1], device="cuda", dtype=torch.int32)
    has_initial_states = torch.tensor([False, True, False], device="cuda", dtype=torch.bool)

    reset_recurrent_state_rows(
        ssm_states,
        state_indices,
        has_initial_states,
        conv_states,
    )

    torch.testing.assert_close(state_pool[1], torch.zeros_like(state_pool[1]))
    for state_idx in (0, 2, 3):
        torch.testing.assert_close(state_pool[state_idx], state_pool_before[state_idx])


@skip_no_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens,num_heads,head_dim", [(1, 2, 4), (5, 3, 8)])
def test_gdn_eager_forward_supplies_output_buffer(
    dtype: torch.dtype, num_tokens: int, num_heads: int, head_dim: int
) -> None:
    config = Qwen3NextConfig(
        architectures=["Qwen3NextForCausalLM"],
        hidden_size=16,
        linear_num_key_heads=1,
        linear_num_value_heads=num_heads,
        linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
        linear_conv_kernel_dim=4,
        torch_dtype=dtype,
    )
    layer = Qwen3NextGatedDeltaNet(
        ModelConfig(pretrained_config=config), aux_stream=None, layer_idx=0
    )
    mixed_qkv = torch.empty(num_tokens, layer.conv_dim_per_tp, dtype=dtype, device="cuda")
    layer._compute_tokenwise_inputs = Mock(return_value=(mixed_qkv, None, None, None, None))
    # This test checks the wrapper's output-buffer allocation and forwarding.
    # Existing mixed-decode tests cover the core's numerical and state behavior, so the core can be
    # mocked without repeating those kernel checks here.
    layer.forward_core = Mock()
    layer._postprocess_gdn_output = Mock()

    with torch_compiling(False):
        layer.forward(
            hidden_states=torch.empty(num_tokens, config.hidden_size, dtype=dtype, device="cuda"),
            attn_metadata=None,
            mamba_metadata=None,
        )

    layer.forward_core.assert_called_once()
    # Check that we pass one buffer to the core call.
    output = layer.forward_core.call_args.kwargs.get("output")
    assert output is not None
    assert output.shape == (1, num_tokens, num_heads, head_dim)
    assert output.dtype == mixed_qkv.dtype
    assert output.device == mixed_qkv.device


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
    from tensorrt_llm._torch.modules.mamba.layernorm_gated import RMSNorm

    layer = Qwen3NextGatedDeltaNet.__new__(Qwen3NextGatedDeltaNet)
    torch.nn.Module.__init__(layer)
    layer.fused_verify = False
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


@pytest.mark.parametrize("ring_mode", [False, True], ids=["triton", "flashinfer_ring"])
def test_gdn_replay_verify_dispatches_by_history_layout(monkeypatch, ring_mode):
    from tensorrt_llm._torch.modules.fla.cache_manager import ReplayStateUpdateMetadata
    from tensorrt_llm._torch.modules.mamba import gdn_mixer

    layer = gdn_mixer.Qwen3NextGatedDeltaNet.__new__(gdn_mixer.Qwen3NextGatedDeltaNet)
    torch.nn.Module.__init__(layer)
    layer.A_log = torch.zeros(4)
    layer.dt_bias = torch.zeros(4)
    calls = {}
    monkeypatch.setattr(
        gdn_mixer,
        "flashinfer_cached_replay_update",
        lambda **kwargs: calls.setdefault("flashinfer", kwargs),
    )
    monkeypatch.setattr(
        gdn_mixer,
        "fused_recurrent_gated_delta_rule_cached_replay_update",
        lambda **kwargs: calls.setdefault("triton", kwargs),
    )

    prev_num_accepted_tokens = torch.tensor([0, 3, 9, 13, 5, 7], dtype=torch.int32)
    cache_buf_idx = torch.zeros(6, dtype=torch.int32)
    layer_cache = SimpleNamespace(
        prev_num_accepted_tokens=prev_num_accepted_tokens,
        cache_buf_idx=cache_buf_idx,
        cache_base=torch.tensor([0, 1, 2, 30, 4, 5], dtype=torch.int32) if ring_mode else None,
        k_ring="k_ring",
        u_ring="u_ring",
        g_ring="g_ring",
        old_x="old_x",
        old_B="old_B",
        old_dt="old_dt",
        old_dA_cumsum="old_dA_cumsum",
    )

    layer._replay_verify_recurrent(
        "q",
        "k",
        "v",
        "a",
        "b",
        "pool",
        state_indices_d=torch.tensor([3, 1, 5, 0], dtype=torch.int32),
        num_decodes=3,
        draft_token_num=4,
        replay_metadata=ReplayStateUpdateMetadata(prev_num_accepted_tokens, cache_buf_idx, 4, 16),
        layer_cache=layer_cache,
        replay_work_items=None,
        replay_n_writes=None,
    )

    if ring_mode:
        assert set(calls) == {"flashinfer"}
        kwargs = calls["flashinfer"]
        assert kwargs["state_indices"].tolist() == [3, 1, 5]
        assert kwargs["hist_len"].tolist() == [13, 3, 7]  # gathered per row
        assert kwargs["cache_base"].tolist() == [30, 1, 5]
        assert kwargs["k_ring"] == "k_ring" and kwargs["history_size"] == 16
    else:
        assert set(calls) == {"triton"}
        kwargs = calls["triton"]
        assert kwargs["state_indices"].tolist() == [3, 1, 5]
        assert kwargs["old_u"] == "old_x"


def test_verify_intermediate_state_indices_reuses_buffers():
    """The verify path's row-index vector is built once, not per GDN layer.

    With a cache-manager-owned ``intermediate_state_indices`` the helper must
    return a view of it (no copy, no allocation); without one it must fall back
    to a process-cached arange sized to the pool, again returned as a view.
    """
    from tensorrt_llm._torch.modules.mamba.gdn_mixer import (
        _cached_arange,
        _verify_intermediate_state_indices,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool_size = 8
    expected = torch.arange(3, dtype=torch.int32, device=device)

    owned = torch.arange(pool_size, dtype=torch.int32, device=device)
    manager = SimpleNamespace(
        intermediate_state_indices=owned,
        get_max_resource_count=lambda: pool_size,
    )
    indices = _verify_intermediate_state_indices(manager, 3, device)
    assert torch.equal(indices, expected)
    assert indices.data_ptr() == owned.data_ptr()  # a view, not a copy

    # No manager-owned buffer (non-speculative manager types): cached fallback.
    manager = SimpleNamespace(
        intermediate_state_indices=None, get_max_resource_count=lambda: pool_size
    )
    first = _verify_intermediate_state_indices(manager, 3, device)
    second = _verify_intermediate_state_indices(manager, 5, device)
    assert torch.equal(first, expected)
    assert torch.equal(second, torch.arange(5, dtype=torch.int32, device=device))
    assert first.data_ptr() == second.data_ptr()  # same cached storage
    assert first.data_ptr() == _cached_arange(pool_size, device).data_ptr()

    # A batch larger than the pool still yields a full index vector: the cached
    # fallback is sized to max(pool, num_decodes), not to the pool alone.
    large = _verify_intermediate_state_indices(manager, pool_size + 4, device)
    assert large.shape[0] == pool_size + 4
    assert torch.equal(large, torch.arange(pool_size + 4, dtype=torch.int32, device=device))

    # A manager-owned buffer shorter than the batch is abandoned for the cached
    # fallback rather than silently returning a short index vector.
    short_owned = torch.arange(2, dtype=torch.int32, device=device)
    manager = SimpleNamespace(
        intermediate_state_indices=short_owned,
        get_max_resource_count=lambda: pool_size,
    )
    widened = _verify_intermediate_state_indices(manager, 5, device)
    assert widened.shape[0] == 5
    assert torch.equal(widened, torch.arange(5, dtype=torch.int32, device=device))
    assert widened.data_ptr() != short_owned.data_ptr()


@triton.jit
def _spin_then_copy(dst, src, n, spin_ns, BLOCK: tl.constexpr):
    tl.extra.cuda.gdc_launch_dependents()
    t0 = tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
    )
    t = t0
    while t - t0 < spin_ns:
        t = tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
        )
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


def _late_copy(dst, src):
    """A producer that releases its PDL dependents at entry and writes ``dst`` ~200 us later: a dependent
    reading ``dst`` before griddepcontrol.wait sees zeros."""
    dst.zero_()
    torch.cuda.synchronize()
    _spin_then_copy[(triton.cdiv(dst.numel(), 1024),)](dst, src, dst.numel(), 200_000, BLOCK=1024)


@triton.jit
def _copy_no_wait(src, dst, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


def _assert_late_copy_overlaps(src):
    """Positive control: a PDL dependent launched after ``_late_copy`` without griddepcontrol.wait reads the
    zeros, so the producer still runs when a dependent starts (else the bitwise checks cannot see a missing wait)."""
    dst, snap = torch.empty_like(src), torch.empty_like(src)
    stale = []
    for _ in range(3):
        _late_copy(dst, src)
        _copy_no_wait[(triton.cdiv(src.numel(), 1024),)](
            dst, snap, src.numel(), BLOCK=1024, launch_pdl=True
        )
        torch.cuda.synchronize()
        stale.append(not torch.equal(snap, src))
    assert any(stale), "the delayed producer did not overlap a PDL dependent: the test is not valid"


@skip_no_cuda
@pytest.mark.parametrize(
    "conv_dim,num_decodes,draft_token_num",
    [(256 + 1024, 2, 7), (512 + 512, 3, 4), (1024 + 2048, 5, 2)],
)
def test_conv1d_verify_update_hoist_and_pdl_wait_bitwise(conv_dim, num_decodes, draft_token_num):
    from tensorrt_llm._torch.modules.mamba.causal_conv1d_triton import causal_conv1d_update
    from tensorrt_llm._utils import get_sm_version

    if get_sm_version() < 90:
        pytest.skip("PDL needs SM90+")
    g = torch.Generator(device="cuda").manual_seed(conv_dim + num_decodes)
    x = torch.randn(num_decodes, draft_token_num, conv_dim, generator=g, device="cuda").to(
        torch.bfloat16
    )
    weight = (torch.randn(conv_dim, 4, generator=g, device="cuda") * 0.4).to(torch.bfloat16)
    state0 = torch.randn(16, conv_dim, 3, generator=g, device="cuda").to(torch.bfloat16)
    slots = torch.tensor([3, 11, 6, 9, 1][:num_decodes], dtype=torch.int32, device="cuda")
    rows = torch.arange(num_decodes, dtype=torch.int32, device="cuda")

    def run(state, x=x, **options):
        windows = torch.zeros(
            num_decodes, draft_token_num, conv_dim, 3, dtype=torch.bfloat16, device="cuda"
        )
        out = causal_conv1d_update(
            x.transpose(1, 2),
            state,
            weight,
            None,
            "silu",
            conv_state_indices=slots,
            intermediate_conv_window=windows,
            intermediate_state_indices=rows,
            **options,
        )
        return out, state, windows

    ref = run(state0.clone())
    # The hoisted x rows and the PDL wait on a producer that writes x late (the conv state, slots and accepted
    # counts are data of earlier steps, read before the wait).
    _assert_late_copy_overlaps(x)
    late = torch.empty_like(x)
    state = state0.clone()
    _late_copy(late, x)
    got = run(state, x=late, launch_dependent_kernels=True, launch_with_pdl_wait=True, hoist_x=True)
    torch.cuda.synchronize()
    for a, b in zip(got, ref):
        assert torch.equal(a, b)


skip_no_sm100f = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs SM100/SM103",
)
_VERIFY_DIM, _VERIFY_HIST, _VERIFY_SLOTS = 128, 16, 256


def _verify_case(heads, n, t, pnat, seed=1000):
    """Per-rank shapes of a 16 k-head / 64 v-head GDN layer: heads = (k heads, v heads)."""
    K = V = _VERIFY_DIM
    H, HV = heads
    g = torch.Generator(device="cuda").manual_seed(seed)
    cd = 2 * H * K + HV * V
    proj = (torch.randn(n * t, cd + HV * V, generator=g, device="cuda") * 0.7).to(torch.bfloat16)
    ba = torch.randn(n * t, 2 * HV, generator=g, device="cuda").to(torch.bfloat16)
    st = dict(
        a_log=torch.log(torch.rand(HV, generator=g, device="cuda") * 7 + 1).float(),
        dt_bias=(torch.rand(HV, generator=g, device="cuda") * 2 - 1).float(),
        norm_w=(torch.rand(V, generator=g, device="cuda") + 0.5).to(torch.bfloat16),
        fp8_scale=torch.tensor(0.02, dtype=torch.float32, device="cuda"),
        state_indices=torch.randperm(_VERIFY_SLOTS, generator=g, device="cuda")[:n].to(torch.int32),
        pnat=torch.full((_VERIFY_SLOTS,), pnat, dtype=torch.int32, device="cuda"),
        buf=torch.zeros(_VERIFY_SLOTS, dtype=torch.int32, device="cuda"),
    )
    st["buf"][2::3] = 1
    kh = torch.randn(_VERIFY_SLOTS, 2, _VERIFY_HIST, H, K, generator=g, device="cuda")
    dec = torch.rand(_VERIFY_SLOTS, 2, HV, _VERIFY_HIST, generator=g, device="cuda") * 0.49 + 0.01
    mut = dict(
        ssm=(torch.randn(_VERIFY_SLOTS, HV, V, K, generator=g, device="cuda") * 0.05).to(
            torch.bfloat16
        ),
        old_k=(kh / kh.norm(dim=-1, keepdim=True)).to(torch.bfloat16),
        old_u=(
            torch.randn(_VERIFY_SLOTS, 2, _VERIFY_HIST, HV, V, generator=g, device="cuda") * 0.5
        ).to(torch.bfloat16),
        old_g=(-torch.cumsum(dec, dim=-1)).float().contiguous(),
    )
    return H, HV, cd, proj, ba, st, mut


def _verify_op(H, HV, cd, n, t, proj, ba, st, mut, fp8, before_launch=None):
    m = {k: v.clone() for k, v in mut.items()}
    if before_launch is not None:
        before_launch(m)
    y = torch.ops.trtllm.gdn_verify(
        proj,
        proj[:, cd:].view(n * t, HV, _VERIFY_DIM),
        ba[:, HV:],
        ba[:, :HV],
        st["a_log"],
        st["dt_bias"],
        st["norm_w"],
        st["fp8_scale"] if fp8 else None,
        1e-6,
        m["ssm"],
        st["state_indices"],
        m["old_u"],
        m["old_k"],
        m["old_g"],
        st["buf"],
        st["pnat"],
    )
    return y, m


def _verify_separate(H, HV, cd, n, t, proj, ba, st, mut, fp8):
    from tensorrt_llm._torch.modules.fla.cached_replay import (
        fused_recurrent_gated_delta_rule_cached_replay_update,
    )
    from tensorrt_llm._torch.modules.mamba.layernorm_gated import rms_norm_gated_token_major

    K = V = _VERIFY_DIM
    m = {k: v.clone() for k, v in mut.items()}
    o = fused_recurrent_gated_delta_rule_cached_replay_update(
        q=proj[:, : H * K].reshape(n, t, H, K),
        k=proj[:, H * K : 2 * H * K].reshape(n, t, H, K),
        v=proj[:, 2 * H * K : cd].reshape(n, t, HV, V),
        g=ba[:, HV:].view(n, t, HV),
        beta=ba[:, :HV].view(n, t, HV),
        ssm_states=m["ssm"],
        state_indices=st["state_indices"],
        old_u=m["old_u"],
        old_k=m["old_k"],
        old_G=m["old_g"],
        old_beta=None,
        cache_buf_idx=st["buf"].clone(),
        prev_num_accepted_tokens=st["pnat"].clone(),
        history_size=_VERIFY_HIST,
        use_qk_l2norm_in_kernel=True,
        A_log=st["a_log"],
        dt_bias=st["dt_bias"],
    )
    y = rms_norm_gated_token_major(
        o.reshape(-1, V),
        proj[:, cd:].view(n * t, HV, V),
        st["norm_w"],
        1e-6,
        fp8_scale=st["fp8_scale"] if fp8 else None,
    )
    return y, m


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


@skip_no_sm100f
@pytest.mark.parametrize(
    "heads,n,t,pnat",
    [
        ((2, 8), 1, 7, 0),
        ((1, 4), 1, 7, 12),
        ((2, 8), 4, 7, 12),
        ((1, 4), 4, 7, 0),
        ((2, 8), 16, 7, 0),
        ((1, 4), 16, 7, 12),
        ((2, 8), 128, 7, 12),
        ((16, 64), 8, 7, 12),
        ((1, 4), 2, 8, 9),
        ((2, 8), 2, 2, 0),
        ((2, 8), 2, 3, 14),
        ((2, 8), 2, 4, 14),
    ],
)
@pytest.mark.parametrize("fp8", [False, True])
def test_gdn_verify_matches_replay_and_norm(heads, n, t, pnat, fp8):
    H, HV, cd, proj, ba, st, mut = _verify_case(heads, n, t, pnat)
    y_ref, m_ref = _verify_separate(H, HV, cd, n, t, proj, ba, st, mut, fp8)
    y, m = _verify_op(H, HV, cd, n, t, proj, ba, st, mut, fp8)
    torch.cuda.synchronize()
    # trtllm::gdn_verify writes the out_proj input [n * t, HV * V]; the reference norm is [n * t * HV, V]
    y_ref = y_ref.view(y.shape)
    if fp8:
        # identical up to FP8 rounding flips of near-tie values
        assert _rel(y, y_ref) < 5e-2
        assert (y.view(torch.uint8) != y_ref.view(torch.uint8)).float().mean().item() < 0.2
    else:
        assert _rel(y, y_ref) < 1e-2
    assert _rel(m["old_u"], m_ref["old_u"]) < 1e-3
    assert _rel(m["old_k"], m_ref["old_k"]) < 1e-2
    assert torch.allclose(m["old_g"], m_ref["old_g"], rtol=1e-5, atol=1e-6)
    assert _rel(m["ssm"], m_ref["ssm"]) < 1e-3
    if pnat + t > _VERIFY_HIST:
        assert not torch.equal(m["ssm"], mut["ssm"]), "commit expected (history full)"


@skip_no_sm100f
def test_gdn_verify_reads_inproj_rows_after_pdl_wait():
    """The verify kernel reads this step's in_proj rows only after griddepcontrol.wait: a producer that
    releases its dependents before writing them still gives the reference output and state."""
    n, t = 4, 7
    H, HV, cd, proj, ba, st, mut = _verify_case((2, 8), n, t, 12)
    y_ref, m_ref = _verify_op(H, HV, cd, n, t, proj, ba, st, mut, fp8=True)
    _assert_late_copy_overlaps(proj)
    late = torch.empty_like(proj)
    y, m = _verify_op(
        H,
        HV,
        cd,
        n,
        t,
        late,
        ba,
        st,
        mut,
        fp8=True,
        before_launch=lambda m: _late_copy(late, proj),
    )
    torch.cuda.synchronize()
    assert torch.equal(y.view(torch.uint8), y_ref.view(torch.uint8))
    assert torch.equal(m["ssm"], m_ref["ssm"])
