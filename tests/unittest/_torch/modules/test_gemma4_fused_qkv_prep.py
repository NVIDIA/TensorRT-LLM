# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Parity test: fused Gemma4 QKV prep vs the unfused reference chain.

The reference reproduces the exact serving-path math it replaces:
split_qkv strided views -> RMSNorm (fp32 accum, bf16 round) -> neox RoPE from
the fp32 cos/sin table (bf16 round) -> .to(float8_e4m3fn).
"""

import pytest
import torch

from tensorrt_llm._torch.modules.gemma4.fused_qkv import gemma4_fused_qkv_norm_rope_quant


def _make_cos_sin(max_pos: int, head_dim: int, rotary_frac: float, theta: float) -> torch.Tensor:
    """Build a [max_pos, 2, head_dim//2] fp32 table like RotaryEmbedding.

    rotary_frac < 1 emulates Gemma4 full-attention layers: only the first
    rotary_frac*half frequency pairs are non-trivial, the rest are
    zero-frequency (cos=1, sin=0).
    """
    half = head_dim // 2
    n_rot = int(half * rotary_frac)
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, 2 * n_rot, 2, dtype=torch.float32)
            / (2 * n_rot if rotary_frac == 1.0 else head_dim)
        )
    )
    inv_freq = torch.cat([inv_freq, torch.zeros(half - n_rot, dtype=torch.float32)])
    pos = torch.arange(max_pos, dtype=torch.float32)
    sinusoid = torch.einsum("i,j->ij", pos, inv_freq)
    return torch.stack([sinusoid.cos(), sinusoid.sin()], dim=1).cuda().contiguous()


def _ref_chain(qkv, position_ids, cos_sin, q_w, k_w, eps, nq, nk, hd, out_fp8):
    q_size, kv_size = nq * hd, nk * hd
    q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

    def norm(x, w):
        xf = x.reshape(-1, hd).float()
        r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        y = xf * r
        if w is not None:
            y = y * w.float()
        return y.to(torch.bfloat16)

    qn = norm(q, q_w).view(-1, nq, hd)
    kn = norm(k, k_w).view(-1, nk, hd)
    vn = norm(v, None).view(-1, kv_size)

    half = hd // 2
    cs = cos_sin[position_ids.view(-1).long()]  # [N, 2, half] fp32
    cos = cs[:, 0].unsqueeze(1)
    sin = cs[:, 1].unsqueeze(1)

    def rope(x):
        xf = x.float()
        x1, x2 = xf[..., :half], xf[..., half:]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat([o1, o2], dim=-1).to(torch.bfloat16).reshape(x.shape[0], -1)

    qr, kr = rope(qn), rope(kn)
    if out_fp8:
        f8 = torch.float8_e4m3fn
        return qr.to(f8), kr.to(f8), vn.to(f8)
    return qr, kr, vn


def _assert_close_fp8(fused: torch.Tensor, ref: torch.Tensor, name: str):
    f, r = fused.float(), ref.float()
    # Bitwise-equal for the overwhelming majority; reduction-order ULP
    # differences may flip an fp8 code by at most one step.
    mismatch = (fused.view(torch.uint8) != ref.view(torch.uint8)).float().mean()
    assert mismatch < 1e-3, f"{name}: fp8 mismatch fraction {mismatch}"
    assert torch.allclose(f, r, atol=0.0625, rtol=0.13), f"{name}: max diff {(f - r).abs().max()}"


@pytest.mark.parametrize(
    "nq,nk,hd,rotary_frac,theta",
    [
        (32, 16, 256, 1.0, 10000.0),  # Gemma4-31B sliding layers
        (32, 4, 512, 0.25, 1000000.0),  # Gemma4-31B full layers (padded freqs)
        (8, 2, 128, 1.0, 10000.0),  # generic small
    ],
)
# 256 is an exact multiple of every BLOCK_N tile; 333/6455 exercise the
# masked tail rows; 6455 is the profiled serving prefill size.
@pytest.mark.parametrize("n_tokens", [1, 7, 256, 333, 6455])
def test_fused_qkv_prep_parity_fp8(nq, nk, hd, rotary_frac, theta, n_tokens):
    torch.manual_seed(1234)
    max_pos = 4096
    qkv = torch.randn((n_tokens, (nq + 2 * nk) * hd), dtype=torch.bfloat16, device="cuda")
    position_ids = torch.randint(0, max_pos, (n_tokens,), dtype=torch.int32, device="cuda")
    cos_sin = _make_cos_sin(max_pos, hd, rotary_frac, theta)
    q_w = torch.rand((hd,), dtype=torch.bfloat16, device="cuda") + 0.5
    k_w = torch.rand((hd,), dtype=torch.bfloat16, device="cuda") + 0.5
    eps = 1e-6

    fq, fk, fv = gemma4_fused_qkv_norm_rope_quant(
        qkv, position_ids, cos_sin, q_w, k_w, eps, nq, nk, hd, out_fp8=True
    )
    rq, rk, rv = _ref_chain(qkv, position_ids, cos_sin, q_w, k_w, eps, nq, nk, hd, out_fp8=True)

    _assert_close_fp8(fq, rq, "q")
    _assert_close_fp8(fk, rk, "k")
    _assert_close_fp8(fv, rv, "v")


def test_fused_qkv_prep_parity_bf16_and_strided():
    """BF16 output mode + a row-strided qkv input (view of a wider buffer)."""
    torch.manual_seed(7)
    nq, nk, hd = 32, 16, 256
    n_tokens, max_pos = 65, 2048
    width = (nq + 2 * nk) * hd
    buf = torch.randn((n_tokens, width + 512), dtype=torch.bfloat16, device="cuda")
    qkv = buf[:, :width]  # non-trivial row stride
    position_ids = torch.randint(0, max_pos, (n_tokens,), dtype=torch.int32, device="cuda")
    cos_sin = _make_cos_sin(max_pos, hd, 1.0, 10000.0)
    q_w = torch.rand((hd,), dtype=torch.bfloat16, device="cuda") + 0.5
    k_w = torch.rand((hd,), dtype=torch.bfloat16, device="cuda") + 0.5
    eps = 1e-6

    fq, fk, fv = gemma4_fused_qkv_norm_rope_quant(
        qkv, position_ids, cos_sin, q_w, k_w, eps, nq, nk, hd, out_fp8=False
    )
    rq, rk, rv = _ref_chain(
        qkv.contiguous(), position_ids, cos_sin, q_w, k_w, eps, nq, nk, hd, out_fp8=False
    )

    for f, r, name in ((fq, rq, "q"), (fk, rk, "k"), (fv, rv, "v")):
        assert torch.allclose(f.float(), r.float(), atol=0.02, rtol=0.02), (
            f"{name}: max diff {(f.float() - r.float()).abs().max()}"
        )


if __name__ == "__main__":
    test_fused_qkv_prep_parity_fp8(32, 16, 256, 1.0, 10000.0, 333)
    test_fused_qkv_prep_parity_fp8(32, 16, 256, 1.0, 10000.0, 6455)
    test_fused_qkv_prep_parity_fp8(32, 4, 512, 0.25, 1000000.0, 6455)
    test_fused_qkv_prep_parity_bf16_and_strided()
    print("ALL PARITY CHECKS PASSED")
