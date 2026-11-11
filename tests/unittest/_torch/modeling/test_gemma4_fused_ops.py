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
"""Parity tests: Gemma4 fused ops vs the unfused op chains they replace."""

import pytest
import torch

from tensorrt_llm._torch.modules.gemma4.fused_qkv import gemma4_fused_qkv_norm_rope_quant


def _make_cos_sin(max_pos: int, head_dim: int, rotary_frac: float, theta: float) -> torch.Tensor:
    """[max_pos, 2, head_dim // 2] fp32 table like RotaryEmbedding.

    rotary_frac < 1 emulates Gemma4 full-attention layers: only the first
    rotary_frac * half frequency pairs are non-trivial, the rest are
    zero-frequency (cos=1, sin=0).
    """
    half = head_dim // 2
    n_rot = int(half * rotary_frac)
    denom = 2 * n_rot if rotary_frac == 1.0 else head_dim
    inv_freq = 1.0 / (theta ** (torch.arange(0, 2 * n_rot, 2, dtype=torch.float32) / denom))
    inv_freq = torch.cat([inv_freq, torch.zeros(half - n_rot, dtype=torch.float32)])
    sinusoid = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv_freq)
    return torch.stack([sinusoid.cos(), sinusoid.sin()], dim=1).cuda().contiguous()


def _ref_qkv(qkv, position_ids, cos_sin, q_w, k_w, eps, nq, nk, hd):
    """Per-head RMSNorm (fp32, bf16 round) -> neox RoPE from the fp32 table (bf16 round); v norm-only."""
    n = qkv.shape[0]
    heads = qkv.reshape(n, nq + 2 * nk, hd)
    half = hd // 2
    cs = cos_sin[position_ids.view(-1).long()]
    cos, sin = cs[:, 0].unsqueeze(1), cs[:, 1].unsqueeze(1)

    def norm(x, w):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return (y * w.float() if w is not None else y).to(torch.bfloat16)

    def rope(x):
        x1, x2 = x.float()[..., :half], x.float()[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(torch.bfloat16)

    q = rope(norm(heads[:, :nq], q_w)).reshape(n, -1)
    k = rope(norm(heads[:, nq : nq + nk], k_w)).reshape(n, -1)
    v = norm(heads[:, nq + nk :], None).reshape(n, -1)
    return q, k, v


def _assert_close_bf16(f, r, name):
    assert torch.allclose(f.float(), r.float(), atol=0.02, rtol=0.02), (
        f"{name}: max diff {(f.float() - r.float()).abs().max()}"
    )


def _assert_close_fp8(fused, ref, name):
    # Reduction-order ULP differences may flip an fp8 code by at most one step.
    mismatch = (fused.view(torch.uint8) != ref.view(torch.uint8)).float().mean()
    assert mismatch < 1e-3, f"{name}: fp8 mismatch fraction {mismatch}"
    f, r = fused.float(), ref.float()
    assert torch.allclose(f, r, atol=0.0625, rtol=0.13), f"{name}: max diff {(f - r).abs().max()}"


_E4B_ROPE = [
    (256, 1.0, 10000.0),  # Gemma4-E4B sliding layers
    (512, 0.25, 1000000.0),  # Gemma4-E4B full layers (padded freqs)
]


@pytest.mark.parametrize("hd,rotary_frac,theta", _E4B_ROPE)
@pytest.mark.parametrize("n_tokens", [1, 333, 4096])
@pytest.mark.parametrize("kv_out_fp8", [False, True])
def test_fused_qkv_prep_flashinfer(hd, rotary_frac, theta, n_tokens, kv_out_fp8):
    """Separate BF16 Q with BF16 or FP8 K/V (FlashInfer over a BF16 or FP8 cache)."""
    torch.manual_seed(3)
    nq, nk, max_pos = 8, 2, 16384
    qkv = torch.randn((n_tokens, (nq + 2 * nk) * hd), dtype=torch.bfloat16, device="cuda")
    position_ids = torch.randint(0, max_pos, (n_tokens,), dtype=torch.int32, device="cuda")
    cos_sin = _make_cos_sin(max_pos, hd, rotary_frac, theta)
    q_w = torch.rand((hd,), dtype=torch.bfloat16, device="cuda") + 0.5
    k_w = torch.rand((hd,), dtype=torch.bfloat16, device="cuda") + 0.5

    q, k, v = gemma4_fused_qkv_norm_rope_quant(
        qkv, position_ids, cos_sin, q_w, k_w, 1e-6, nq, nk, hd, out_fp8=False, kv_out_fp8=kv_out_fp8
    )
    rq, rk, rv = _ref_qkv(qkv, position_ids, cos_sin, q_w, k_w, 1e-6, nq, nk, hd)
    assert q.dtype == torch.bfloat16
    _assert_close_bf16(q, rq, "q")
    if kv_out_fp8:
        assert k.dtype == v.dtype == torch.float8_e4m3fn
        _assert_close_fp8(k, rk.to(torch.float8_e4m3fn), "k")
        _assert_close_fp8(v, rv.to(torch.float8_e4m3fn), "v")
    else:
        _assert_close_bf16(k, rk, "k")
        _assert_close_bf16(v, rv, "v")


@pytest.mark.parametrize("hd,rotary_frac,theta", _E4B_ROPE)
@pytest.mark.parametrize("n_tokens", [1, 333, 4096])
def test_fused_qkv_prep_q_only(hd, rotary_frac, theta, n_tokens):
    """num_kv_heads=0 (KV-shared layers): q_norm + RoPE over a row-strided Q-only input."""
    torch.manual_seed(5)
    nq, max_pos = 8, 16384
    q_size = nq * hd
    q = torch.randn((n_tokens, q_size + 128), dtype=torch.bfloat16, device="cuda")[:, :q_size]
    position_ids = torch.randint(0, max_pos, (n_tokens,), dtype=torch.int32, device="cuda")
    cos_sin = _make_cos_sin(max_pos, hd, rotary_frac, theta)
    q_w = torch.rand((hd,), dtype=torch.bfloat16, device="cuda") + 0.5

    fq, fk, fv = gemma4_fused_qkv_norm_rope_quant(
        q, position_ids, cos_sin, q_w, q_w, 1e-6, nq, 0, hd, out_fp8=False
    )
    assert fk is None and fv is None
    assert fq.shape == (n_tokens, q_size) and fq.is_contiguous()
    rq, _, _ = _ref_qkv(q, position_ids, cos_sin, q_w, q_w, 1e-6, nq, 0, hd)
    _assert_close_bf16(fq, rq, "q")
