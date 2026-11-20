# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language._layouts import BlockedLayout, SliceLayout
from triton.experimental.gluon.language.extra import libdevice


@gluon.jit
def _text_qk_norm_rope(
    QKV,
    Q_WEIGHT,
    K_WEIGHT,
    POSITIONS,
    NUM_Q: gl.constexpr,
    NUM_KV: gl.constexpr,
    DIM: gl.constexpr,
    ROTARY_DIM: gl.constexpr,
    EPS: gl.constexpr,
    BASE: gl.constexpr,
    FACTOR: gl.constexpr,
    LOW: gl.constexpr,
    HIGH: gl.constexpr,
    ATTENTION_FACTOR: gl.constexpr,
    HEADS_PER_TILE: gl.constexpr,
):
    token = gl.program_id(0)
    per_lane: gl.constexpr = DIM // 32
    layout: gl.constexpr = BlockedLayout([1, per_lane], [1, 32], [4, 1], [1, 0])
    head_layout: gl.constexpr = SliceLayout(1, layout)
    heads = gl.program_id(1) * HEADS_PER_TILE + gl.arange(0, HEADS_PER_TILE, layout=head_layout)
    offsets = gl.arange(0, DIM, layout=gl.constexpr(SliceLayout(0, layout)))
    active = heads < NUM_Q + NUM_KV
    rows = (token.to(gl.int64) * (NUM_Q + 2 * NUM_KV) + heads) * DIM
    ptrs = QKV + rows[:, None] + offsets[None, :]
    x = gl.load(ptrs, active[:, None], other=0).to(gl.float32)
    q_weight = gl.load(Q_WEIGHT + offsets).to(gl.float32)
    k_weight = gl.load(K_WEIGHT + offsets).to(gl.float32)
    weight = gl.where(heads[:, None] < NUM_Q, q_weight[None, :], k_weight[None, :])

    # Each lane accumulates adjacent channels in FP32 before the warp reduction.
    columns, _ = gl.broadcast(offsets[None, :], x)
    warp = gl.reshape(x, (HEADS_PER_TILE, 32, per_lane))
    column = gl.reshape(columns, warp.shape) % per_lane
    sum_squares = 0.0
    for i in gl.static_range(per_lane):
        element = gl.sum(gl.where(column == i, warp, 0.0), axis=2)
        sum_squares = gl.fma(element, element, sum_squares)
    inv_rms = gl.rsqrt(gl.sum(sum_squares, axis=1) * (1.0 / DIM) + EPS)
    inv_rms = gl.convert_layout(inv_rms, head_layout)
    normed = x * (inv_rms[:, None] * weight)

    half: gl.constexpr = ROTARY_DIM // 2
    # For a non-power-of-two partial span the text kernel pairs lanes by XOR.
    partner = gl.inline_asm_elementwise(
        "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;",
        constraints="=f,f,r",
        args=[normed, half // per_lane],
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )
    partner = gl.where(offsets[None, :] < half, -partner, partner)
    pair = (offsets % half).to(gl.float32)
    exponent = gl.div_rn(-2.0 * libdevice.fast_log2f(gl.full((), BASE, gl.float32)), ROTARY_DIM)
    freq = gl.exp2(pair * exponent)
    if FACTOR != 1.0:
        high = gl.full((), HIGH, gl.float32)
        low = gl.full((), LOW, gl.float32)
        high = gl.where(gl.abs(high - low) <= 1.0e-6, high + 0.001, high)
        ramp = gl.minimum(gl.maximum(gl.div_rn(pair - low, high - low), 0.0), 1.0)
        extrapolation = 1.0 - ramp
        freq = gl.div_rn(freq, FACTOR) * (1.0 - extrapolation) + freq * extrapolation
    angle = gl.load(POSITIONS + token).to(gl.float32) * freq
    cos = libdevice.fast_cosf(angle)[None, :]
    sin = libdevice.fast_sinf(angle)[None, :]
    rotated = gl.fma(normed, cos, partner * sin)
    output = gl.where(offsets[None, :] < ROTARY_DIM, rotated * ATTENTION_FACTOR, normed)
    gl.store(ptrs, output.to(gl.bfloat16), active[:, None])


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
            heads_per_tile = 16
            grid = (qkv.shape[0], triton.cdiv(num_heads_q + num_heads_kv, heads_per_tile))
            _text_qk_norm_rope[grid](
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
                HEADS_PER_TILE=heads_per_tile,
            )
    return qkv
