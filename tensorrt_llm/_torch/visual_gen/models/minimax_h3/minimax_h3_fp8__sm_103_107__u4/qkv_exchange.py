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
