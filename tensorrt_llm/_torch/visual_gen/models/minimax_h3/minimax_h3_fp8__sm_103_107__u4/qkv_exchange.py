# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare dynamic per-tensor FP8 QKV in packed Ulysses send order."""

import torch
import triton
import triton.language as tl


@triton.jit
def _magnitude_maximum(tensor, index, valid_rows, strides: tl.constexpr):
    row = index // (56 * 128)
    head = index // 128 % 56
    value = tl.load(
        tensor + row * strides[0] + head * strides[1] + index % 128, row < valid_rows, 0
    )
    # Nonnegative BF16 values have the same ordering as their bit patterns.
    magnitude = value.to(tl.uint16, bitcast=True).to(tl.int32) & 0x7FFF
    return tl.max(magnitude, axis=0)


@triton.jit
def _partial_maxima(
    q,
    k,
    v,
    partials,
    valid_rows,
    tiles,
    qs: tl.constexpr,
    ks: tl.constexpr,
    vs: tl.constexpr,
    tile: tl.constexpr,
):
    block = tl.program_id(0)
    plane = tl.program_id(1)
    index = block * tile + tl.arange(0, tile)
    if plane == 0:
        value = _magnitude_maximum(q, index, valid_rows, qs)
    elif plane == 1:
        value = _magnitude_maximum(k, index, valid_rows, ks)
    else:
        value = _magnitude_maximum(v, index, valid_rows, vs)
    tl.store(partials + plane * tiles + block, value)


@triton.jit
def _final_maxima(partials, output, tiles, tile: tl.constexpr):
    plane = tl.program_id(0)
    index = tl.arange(0, tile)
    values = tl.load(partials + plane * tiles + index, index < tiles, 0)
    value = tl.max(values, axis=0)
    tl.store(output + plane, value.to(tl.uint16).to(tl.bfloat16, bitcast=True))


@triton.jit
def _store_plane(
    source, descales, send, row, element, peer, plane, rows: tl.constexpr, strides: tl.constexpr
):
    head = peer * 14 + element // 128
    valid = (row < rows) & (element < 14 * 128)
    value = tl.load(source + row * strides[0] + head * strides[1] + element % 128, valid, 0).to(
        tl.float32
    )
    factor = tl.div_rn(1.0, tl.load(descales + plane))
    scaled = tl.clamp(value * factor, -448.0, 448.0)
    code = scaled.to(tl.float8e4nv, fp_downcast_rounding="rtne").to(tl.uint8, bitcast=True)
    offset = ((peer * rows + row) * 3 + plane) * (14 * 128) + element
    tl.store(send + offset, code, valid)


@triton.jit
def _pack(
    q,
    k,
    v,
    descales,
    send,
    rows: tl.constexpr,
    qs: tl.constexpr,
    ks: tl.constexpr,
    vs: tl.constexpr,
):
    row = (tl.program_id(0) * 4 + tl.arange(0, 4))[:, None]
    element = tl.arange(0, 2048)[None, :]
    peer = tl.program_id(1)
    plane = tl.program_id(2)
    if plane == 0:
        _store_plane(q, descales, send, row, element, peer, plane, rows, qs)
    elif plane == 1:
        _store_plane(k, descales, send, row, element, peer, plane, rows, ks)
    else:
        _store_plane(v, descales, send, row, element, peer, plane, rows, vs)


_ROPE_CHUNK = 16


@triton.jit
def _v_partial_maxima(v, partials, valid_rows, vs: tl.constexpr, tile: tl.constexpr):
    block = tl.program_id(0)
    index = block * tile + tl.arange(0, tile)
    value = _magnitude_maximum(v, index, valid_rows, vs)
    tl.store(partials + block, value)


@triton.jit
def _qk_norm_rope_and_max_kernel(
    qkv_ptr,
    q_out_ptr,
    k_out_ptr,
    partials_ptr,
    weight_q_ptr,
    weight_k_ptr,
    cos_ptr,
    sin_ptr,
    NUM_TOKENS,
    VALID_ROWS,
    SEQ_LEN,
    NUM_T_BLOCKS,
    NUM_H_BLOCKS,
    qkv_row_stride,
    k_col_offset,
    out_row_stride,
    cos_row_stride,
    cos_col_stride,
    sin_row_stride,
    sin_col_stride,
    eps,
    HEAD_DIM: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    CHUNK: tl.constexpr,
    NUM_ROT_HALF_CHUNKS: tl.constexpr,
    TOKENS_PER_PROGRAM: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
):
    """Per-head QK RMSNorm + partial split-half RoPE with atomic-free per-CTA valid-row maxima."""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    which = tl.program_id(2)
    rows = tl.arange(0, TOKENS_PER_PROGRAM * HEADS_PER_PROGRAM)
    tok = pid_t * TOKENS_PER_PROGRAM + rows // HEADS_PER_PROGRAM
    head = pid_h * HEADS_PER_PROGRAM + rows % HEADS_PER_PROGRAM
    j = tl.arange(0, NUM_CHUNKS)
    kk = tl.arange(0, CHUNK)
    tok_ok = (tok < NUM_TOKENS)[:, None, None]
    tok_valid = (tok < VALID_ROWS)[:, None, None]
    base = (tok.to(tl.int64) * qkv_row_stride + head * HEAD_DIM + which * k_col_offset)[
        :, None, None
    ]
    off = (j * CHUNK)[None, :, None] + kk[None, None, :]

    partner_chunk = tl.where(
        j < NUM_ROT_HALF_CHUNKS,
        j + NUM_ROT_HALF_CHUNKS,
        tl.where(j < 2 * NUM_ROT_HALF_CHUNKS, j - NUM_ROT_HALF_CHUNKS, j),
    )
    partner_off = (partner_chunk * CHUNK)[None, :, None] + kk[None, None, :]
    x = tl.load(qkv_ptr + base + off, mask=tok_ok, other=0.0).to(tl.float32)
    xp = tl.load(qkv_ptr + base + partner_off, mask=tok_ok, other=0.0).to(tl.float32)
    weight_ptr = tl.where(which == 0, weight_q_ptr, weight_k_ptr)
    w = tl.load(weight_ptr + off).to(tl.float32)
    wp = tl.load(weight_ptr + partner_off).to(tl.float32)

    sum_sq = tl.sum(tl.sum(x * x, axis=2), axis=1)
    inv_rms = tl.rsqrt(sum_sq / HEAD_DIM + eps)[:, None, None]
    y = w * (x * inv_rms)
    yp = wp * (xp * inv_rms)

    rot_chunk = (j < 2 * NUM_ROT_HALF_CHUNKS)[None, :, None]
    rot_mask = tok_ok & rot_chunk
    seq = (tok % SEQ_LEN).to(tl.int64)
    c = tl.load(
        cos_ptr + (seq * cos_row_stride)[:, None, None] + off * cos_col_stride,
        mask=rot_mask,
        other=1.0,
    ).to(tl.float32)
    sn = tl.load(
        sin_ptr + (seq * sin_row_stride)[:, None, None] + off * sin_col_stride,
        mask=rot_mask,
        other=0.0,
    ).to(tl.float32)
    rotated = tl.where((j < NUM_ROT_HALF_CHUNKS)[None, :, None], -yp, yp)
    out = tl.where(rot_chunk, y * c + rotated * sn, y)
    out_bf16 = out.to(tl.bfloat16)
    out_base = (tok.to(tl.int64) * out_row_stride + head * HEAD_DIM)[:, None, None]
    out_ptr = tl.where(which == 0, q_out_ptr, k_out_ptr)
    tl.store(out_ptr + out_base + off, out_bf16, mask=tok_ok)

    mag = tl.where(tok_valid, out_bf16.to(tl.uint16, bitcast=True).to(tl.int32) & 0x7FFF, 0)
    local_max = tl.max(tl.max(tl.max(mag, axis=2), axis=1), axis=0)
    tl.store(partials_ptr + (which * NUM_T_BLOCKS + pid_t) * NUM_H_BLOCKS + pid_h, local_max)


def qk_norm_rope_pack_qkv(
    qkv: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float,
    valid_rows: int,
    group,
    *,
    tokens_per_program: int = 1,
    num_warps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused BF16 QK-norm + RoPE with atomic-free local maxima, global MAX, and FP8 pack."""
    rows = qkv.shape[1]
    hd = 56 * 128
    qkv2 = qkv.view(rows, 3 * hd)
    q_buf = torch.empty((1, rows, 56, 128), dtype=torch.bfloat16, device=qkv.device)
    k_buf = torch.empty((1, rows, 56, 128), dtype=torch.bfloat16, device=qkv.device)
    v_view = qkv[:, :, 2 * hd :].view(1, rows, 56, 128)
    t_blocks = triton.cdiv(rows, tokens_per_program)
    h_blocks = 56 // 8
    qk_tiles = t_blocks * h_blocks
    v_tiles = max(1, triton.cdiv(valid_rows * 56 * 128, 8192))
    partials_qk = qkv.new_empty((2, qk_tiles), dtype=torch.int32)
    partials_v = qkv.new_empty((v_tiles,), dtype=torch.int32)
    _qk_norm_rope_and_max_kernel[(t_blocks, h_blocks, 2)](
        qkv2,
        q_buf,
        k_buf,
        partials_qk,
        weight_q,
        weight_k,
        cos,
        sin,
        rows,
        valid_rows,
        rows,
        t_blocks,
        h_blocks,
        qkv2.stride(0),
        hd,
        hd,
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        float(eps),
        HEAD_DIM=128,
        NUM_CHUNKS=128 // _ROPE_CHUNK,
        CHUNK=_ROPE_CHUNK,
        NUM_ROT_HALF_CHUNKS=cos.shape[1] // 2 // _ROPE_CHUNK,
        TOKENS_PER_PROGRAM=tokens_per_program,
        HEADS_PER_PROGRAM=8,
        num_warps=num_warps,
    )
    _v_partial_maxima[(v_tiles,)](v_view, partials_v, valid_rows, v_view.stride()[1:3], 8192, num_warps=8)
    maxima = qkv.new_empty((3,))
    _final_maxima[(2,)](partials_qk, maxima, qk_tiles, triton.next_power_of_2(qk_tiles), num_warps=8)
    _final_maxima[(1,)](partials_v, maxima[2:3], v_tiles, triton.next_power_of_2(v_tiles), num_warps=8)
    torch.distributed.all_reduce(maxima, op=torch.distributed.ReduceOp.MAX, group=group)
    descales = (maxima.float() / 448.0).clamp_min(1.0 / (448.0 * 512.0)).bfloat16().float()
    strides = tuple(t.stride()[1:3] for t in (q_buf, k_buf, v_view))
    send = qkv.new_empty((4, rows, 3, 14, 128), dtype=torch.uint8)
    _pack[(triton.cdiv(rows, 4), 4, 3)](
        q_buf, k_buf, v_view, descales, send, rows, *strides, num_warps=4
    )
    return send.view(torch.float8_e4m3fn), descales


def pack_qkv(q, k, v, valid_rows: int, group) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack BF16 ``[1, S_local, 56, 128]`` inputs with unit innermost stride.

    Returns FP8 ``[4, S_local, 3, 14, 128]`` and three FP32 descales. Only valid
    rows participate in the global maxima. Each peer receives a slice of the heads.
    """
    rows = q.shape[1]
    strides = tuple(t.stride()[1:3] for t in (q, k, v))
    tiles = max(1, triton.cdiv(valid_rows * 56 * 128, 8192))
    partials = q.new_empty((3, tiles), dtype=torch.int32)
    maxima = q.new_empty((3,))
    _partial_maxima[(tiles, 3)](q, k, v, partials, valid_rows, tiles, *strides, 8192, num_warps=8)
    _final_maxima[(3,)](partials, maxima, tiles, triton.next_power_of_2(tiles), num_warps=8)
    torch.distributed.all_reduce(maxima, op=torch.distributed.ReduceOp.MAX, group=group)
    # The native BF16 quantizer rounds the descale before computing the cast factor.
    descales = (maxima.float() / 448.0).clamp_min(1.0 / (448.0 * 512.0)).bfloat16().float()
    send = q.new_empty((4, rows, 3, 14, 128), dtype=torch.uint8)
    _pack[(triton.cdiv(rows, 4), 4, 3)](q, k, v, descales, send, rows, *strides, num_warps=4)
    return send.view(torch.float8_e4m3fn), descales
