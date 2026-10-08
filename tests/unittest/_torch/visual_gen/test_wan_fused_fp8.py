# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerics of the Wan fused FP8 ops against plain PyTorch references."""

import math

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.modules.wan_fused_fp8 import ops  # noqa: F401  (registers ops)
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="Wan fused FP8 kernels target Rubin (SM107).",
)

DIM = 5120
HEADS = 40
HEAD_DIM = 128
FP8 = torch.float8_e4m3fn


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


@pytest.mark.parametrize("epilogue", [0, 1, 2], ids=["none", "bias", "bias_gelu"])
def test_gemm_fp8_matches_reference(epilogue):
    torch.manual_seed(0)
    a8 = (torch.randn(512, DIM, device="cuda") * 4).to(FP8)
    w8 = (torch.randn(1024, DIM, device="cuda") * 4).to(FP8)
    scale_a = torch.tensor([0.05], device="cuda")
    scale_w = torch.tensor([0.002], device="cuda")
    bias = (torch.randn(1024, device="cuda") * 0.1).to(torch.bfloat16)
    ref = (a8.float() * scale_a) @ (w8.float() * scale_w).t()
    if epilogue:
        ref = ref + bias.float()
    if epilogue == 2:
        ref = F.gelu(ref, approximate="tanh")
    out = torch.ops.wanfused.gemm_fp8(a8, w8, scale_a, scale_w, bias, epilogue, None)
    assert _cosine(out, ref) > 0.9999


def test_gemm_fp8_fp8_output_matches_reference():
    torch.manual_seed(0)
    a8 = (torch.randn(512, DIM, device="cuda") * 4).to(FP8)
    w8 = (torch.randn(1024, DIM, device="cuda") * 4).to(FP8)
    scale_a, scale_w = torch.tensor([0.05], device="cuda"), torch.tensor([0.002], device="cuda")
    bias = (torch.randn(1024, device="cuda") * 0.1).to(torch.bfloat16)
    inv_scale_out = torch.tensor([50.0], device="cuda")
    ref = F.gelu(
        (a8.float() * scale_a) @ (w8.float() * scale_w).t() + bias.float(), approximate="tanh"
    )
    out = torch.ops.wanfused.gemm_fp8(a8, w8, scale_a, scale_w, bias, 2, inv_scale_out)
    assert out.dtype == FP8
    assert _cosine(out.float() / inv_scale_out, ref) > 0.999


def test_resid_ln_quant_adaln_matches_reference():
    torch.manual_seed(0)
    batch, seq, eps = 2, 64, 1e-6
    x = torch.randn(batch * seq, DIM, device="cuda").to(torch.bfloat16)
    y = torch.randn(batch * seq, DIM, device="cuda").to(torch.bfloat16)
    ybias = (torch.randn(DIM, device="cuda") * 0.1).to(torch.bfloat16)
    gate = torch.rand(batch, DIM, device="cuda")
    scale = torch.randn(batch, DIM, device="cuda") * 0.1
    shift = torch.randn(batch, DIM, device="cuda") * 0.1
    inv_scale = torch.tensor([20.0], device="cuda")
    x_new, q8 = torch.ops.wanfused.resid_ln_quant(
        x, y, ybias, gate, 2, scale, shift, seq, eps, inv_scale
    )

    y_biased = (y + ybias).float().view(batch, seq, DIM)
    x_ref = (x.float().view(batch, seq, DIM) + y_biased * gate[:, None]).to(torch.bfloat16)
    normed = F.layer_norm(x_ref.float(), (DIM,), eps=eps)
    normed = (normed * (1 + scale[:, None]) + shift[:, None]).to(torch.bfloat16)
    q_ref = (normed.float() * inv_scale).clamp(-448, 448)
    assert _cosine(x_new, x_ref) > 0.99999
    assert _cosine(q8.float(), q_ref) > 0.999


def _reference_attention(qkv, norm_q_w, norm_k_w, cos, sin, eps):
    batch, seq, _ = qkv.shape
    q, k, v = qkv.float().split(DIM, dim=-1)
    q = F.rms_norm(q, (DIM,), norm_q_w.float(), eps)
    k = F.rms_norm(k, (DIM,), norm_k_w.float(), eps)

    def rope(t):
        t = t.view(batch, seq, HEADS, HEAD_DIM // 2, 2)
        c = cos.view(1, seq, 1, HEAD_DIM // 2, 2)[..., 0]
        s = sin.view(1, seq, 1, HEAD_DIM // 2, 2)[..., 0]
        x0, x1 = t[..., 0], t[..., 1]
        return torch.stack([x0 * c - x1 * s, x1 * c + x0 * s], dim=-1).view(
            batch, seq, HEADS, HEAD_DIM
        )

    q, k = rope(q), rope(k)
    v = v.view(batch, seq, HEADS, HEAD_DIM)
    out = F.scaled_dot_product_attention(*(t.transpose(1, 2) for t in (q, k, v)))
    return out.transpose(1, 2).reshape(batch, seq, DIM)


def test_fp8_self_attention_matches_reference():
    torch.manual_seed(0)
    batch, seq, eps = 2, 1024, 1e-6
    qkv = torch.randn(batch, seq, 3 * DIM, device="cuda").to(torch.bfloat16)
    norm_q_w = (torch.rand(DIM, device="cuda") + 0.5).to(torch.bfloat16)
    norm_k_w = (torch.rand(DIM, device="cuda") + 0.5).to(torch.bfloat16)
    angle = torch.rand(seq, HEAD_DIM // 2, device="cuda") * 2 * math.pi
    cos = torch.repeat_interleave(angle.cos(), 2, dim=-1)
    sin = torch.repeat_interleave(angle.sin(), 2, dim=-1)
    out = torch.ops.wanfused.fp8_self_attention(qkv, norm_q_w, norm_k_w, cos, sin, HEADS, eps, True)
    ref = _reference_attention(qkv, norm_q_w, norm_k_w, cos, sin, eps)
    assert out.shape == (batch, seq, DIM)
    assert _cosine(out, ref) > 0.99


def test_fp8_self_attention_fp8_out_matches_bf16_out_then_quant():
    torch.manual_seed(0)
    batch, seq, eps = 2, 1024, 1e-6
    qkv = torch.randn(batch, seq, 3 * DIM, device="cuda").to(torch.bfloat16)
    norm_q_w = (torch.rand(DIM, device="cuda") + 0.5).to(torch.bfloat16)
    norm_k_w = (torch.rand(DIM, device="cuda") + 0.5).to(torch.bfloat16)
    angle = torch.rand(seq, HEAD_DIM // 2, device="cuda") * 2 * math.pi
    cos = torch.repeat_interleave(angle.cos(), 2, dim=-1)
    sin = torch.repeat_interleave(angle.sin(), 2, dim=-1)
    args = (qkv, norm_q_w, norm_k_w, cos, sin, HEADS, eps, True)
    out16 = torch.ops.wanfused.fp8_self_attention(*args)
    out_scale = (out16.float().abs().amax() / 448.0).reshape(1)
    ref8 = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
        out16.reshape(batch * seq, DIM), out_scale
    )[0].view(batch, seq, DIM)
    out8 = torch.ops.wanfused.fp8_self_attention_fp8_out(*args, out_scale)
    assert out8.dtype == torch.float8_e4m3fn and out8.shape == ref8.shape
    # One rounding instead of two: codes differ by at most one step.
    steps = (out8.view(torch.uint8).int() - ref8.view(torch.uint8).int()).abs()
    assert steps.max().item() <= 1
    assert (steps == 0).float().mean().item() > 0.95
