# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Constants and per-shape buffers shared by the Wan fused FP8 ops."""

import math

import torch

from . import _ext

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
HEAD_DIM = 128
LOG2E = 1.0 / math.log(2.0)
# sq = sk = 1 / sqrt(c), c = 1 / (sm_scale * log2 e): unit softmax scale.
QK_MUL = math.sqrt(HEAD_DIM**-0.5 * LOG2E)

_buffers = {}


def shape_buffers(batch: int, seq: int, device: torch.device) -> dict:
    """Per-(batch, seq, device) buffers shared by every layer and call.

    Constants: bmm1 = {scale, scale*log2e} with scale = ln 2, so the FMHA's
    scaleSoftmaxLog2 == 1 (sm_scale is folded into Q/K by QK_MUL); qk_mul and
    mul[:2] = QK_MUL; cu_seqlens/seqlens. Rewritten by v_scale each call:
    mul[2] (1/scale_v), amax_v, scale_v.
    """
    key = (batch, seq, str(device))
    if key not in _buffers:
        f32 = dict(device=device, dtype=torch.float32)
        _buffers[key] = dict(
            bmm1=torch.tensor([1.0 / LOG2E, 1.0], **f32),
            cu_seqlens=torch.arange(0, (batch + 1) * seq, seq, device=device, dtype=torch.int32),
            seqlens=torch.full((batch,), seq, device=device, dtype=torch.int32),
            qk_mul=torch.tensor([QK_MUL, QK_MUL], **f32),
            mul=torch.tensor([QK_MUL, QK_MUL, 1.0], **f32),
            scale_v=torch.ones(1, **f32),
            amax_v=torch.zeros(1, **f32),
        )
    return _buffers[key]


def rope_tables(cos: torch.Tensor, sin: torch.Tensor, tokens: int):
    cos2d = cos.reshape(-1, HEAD_DIM).float().contiguous()
    sin2d = sin.reshape(-1, HEAD_DIM).float().contiguous()
    # Fewer rows than tokens: one table shared by every batch entry.
    seq_per_batch = 0 if cos2d.shape[0] == tokens else cos2d.shape[0]
    return cos2d, sin2d, seq_per_batch


def qkv_mul(bufs: dict, mul_v: torch.Tensor) -> torch.Tensor:
    """Q/K/V quant multipliers [3]: the shared QK_MUL pair and the given V multiplier."""
    return torch.cat([bufs["qk_mul"], mul_v.float().reshape(1)])


def v_scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    """V dequant scale from a (reduced) |V| amax; the epsilon matches vScaleReduceKernel in fused_qkv_fp8.cu."""
    return amax.clamp_min(1e-12) / FP8_MAX


def norm_rope_quant_into(
    qkv2d, num_heads, norm_q_w, norm_k_w, cos, sin, eps, interleave, mul, outs
):
    """QK RMSNorm + RoPE + FP8 quant of packed qkv2d [T, 3*H*D] into the three FP8 outs."""
    cos2d, sin2d, seq_per_batch = rope_tables(cos, sin, qkv2d.shape[0])
    _ext.prep().norm_rope_quant(
        qkv2d,
        num_heads,
        eps,
        norm_q_w,
        norm_k_w,
        cos2d,
        sin2d,
        interleave,
        seq_per_batch,
        mul,
        *outs,
    )
