# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _text_qk_norm_rope(
    QKV,
    Q_WEIGHT,
    K_WEIGHT,
    POSITIONS,
    NUM_Q: tl.constexpr,
    NUM_KV: tl.constexpr,
    DIM: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    EPS: tl.constexpr,
    BASE: tl.constexpr,
    FACTOR: tl.constexpr,
    LOW: tl.constexpr,
    HIGH: tl.constexpr,
    ATTENTION_FACTOR: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offsets = tl.arange(0, DIM)
    lanes = tl.arange(0, 32)
    per_lane: tl.constexpr = DIM // 32
    row = QKV + (token.to(tl.int64) * (NUM_Q + 2 * NUM_KV) + head) * DIM
    x = tl.load(row + offsets).to(tl.float32)
    weight = tl.load(tl.where(head < NUM_Q, Q_WEIGHT, K_WEIGHT) + offsets).to(tl.float32)

    # Each lane accumulates a contiguous vector before the descending XOR reduction.
    ss = tl.full((32,), 0, tl.float32)
    for i in tl.static_range(per_lane):
        element = tl.gather(x, lanes * per_lane + i, 0)
        ss = tl.fma(element, element, ss)
    for i in tl.static_range(5):
        ss += tl.gather(ss, lanes ^ (16 >> i), 0)
    inv_rms = tl.rsqrt(ss * (1.0 / DIM) + EPS)
    normed = x * (tl.gather(inv_rms, offsets // per_lane, 0) * weight)

    half: tl.constexpr = ROTARY_DIM // 2
    # NeoX pairs lanes by XOR. This determines the pairing for partial spans
    # whose half width contains a non-power-of-two number of lane vectors.
    partner = tl.gather(normed, offsets ^ half, 0)
    partner = tl.where(offsets < half, -partner, partner)
    pair = (offsets % half).to(tl.float32)
    exponent = (-2.0 * libdevice.fast_log2f(tl.full((), BASE, tl.float32))) / ROTARY_DIM
    freq = tl.exp2(pair * exponent)
    if FACTOR != 1.0:
        high = tl.full((), HIGH, tl.float32)
        low = tl.full((), LOW, tl.float32)
        high = tl.where(tl.abs(high - low) <= 1.0e-6, high + 0.001, high)
        ramp = tl.minimum(tl.maximum((pair - low) / (high - low), 0.0), 1.0)
        extrapolation = 1.0 - ramp
        freq = (freq / FACTOR) * (1.0 - extrapolation) + freq * extrapolation
    angle = tl.load(POSITIONS + token).to(tl.float32) * freq
    rotated = tl.fma(normed, libdevice.fast_cosf(angle), partner * libdevice.fast_sinf(angle))
    output = tl.where(offsets < ROTARY_DIM, rotated * ATTENTION_FACTOR, normed)
    tl.store(row + offsets, output.to(tl.bfloat16))


def apply_text_qk_norm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    *,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    base: float,
    factor: float = 1.0,
    low: float = 0.0,
    high: float = 0.0,
    attention_factor: float = 1.0,
) -> torch.Tensor:
    """Normalize packed Q/K heads in place and rotate the leading channels at each text position."""
    if head_dim not in (64, 128, 256):
        raise ValueError(f"Unsupported H3 text encoder head dimension {head_dim}")
    if not (0 < rotary_dim <= head_dim) or rotary_dim % 2 or rotary_dim * 16 % head_dim:
        raise ValueError(f"Invalid rotary dimension {rotary_dim} for head dimension {head_dim}")
    if factor == 1 and attention_factor != 1:
        raise ValueError("Unscaled H3 text RoPE requires attention_factor=1")
    if (
        qkv.ndim != 2
        or num_heads_q < 0
        or num_heads_kv < 0
        or qkv.shape[1] != (num_heads_q + 2 * num_heads_kv) * head_dim
        or qkv.dtype != torch.bfloat16
        or not qkv.is_cuda
        or not qkv.is_contiguous()
    ):
        raise ValueError(
            "Expected a contiguous packed CUDA BF16 QKV tensor matching the head counts"
        )
    for weight in (q_weight, k_weight):
        if (
            weight.shape != (head_dim,)
            or weight.device != qkv.device
            or weight.dtype != torch.bfloat16
            or not weight.is_contiguous()
        ):
            raise ValueError(
                "Text Q/K norm weights must be contiguous BF16 head vectors on the QKV device"
            )
    if (
        positions.shape != (qkv.shape[0],)
        or positions.dtype != torch.int32
        or positions.device != qkv.device
        or not positions.is_contiguous()
    ):
        raise ValueError(
            "Expected contiguous int32 text positions, one per token on the QKV device"
        )
    if qkv.shape[0] and num_heads_q + num_heads_kv:
        with torch.cuda.device(qkv.device):
            _text_qk_norm_rope[(qkv.shape[0], num_heads_q + num_heads_kv)](
                qkv,
                q_weight,
                k_weight,
                positions,
                num_heads_q,
                num_heads_kv,
                head_dim,
                rotary_dim,
                eps,
                base,
                factor,
                low,
                high,
                attention_factor,
                num_warps=1,
            )
    return qkv
