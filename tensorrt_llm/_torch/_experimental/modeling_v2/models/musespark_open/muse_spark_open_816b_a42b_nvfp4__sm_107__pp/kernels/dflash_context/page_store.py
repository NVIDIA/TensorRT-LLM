# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One launch that writes a context step's drafter K and V rows into the paged draft pool.

``store`` takes the step's store table (``store.host_table``: per request the address of its
post-RoPE K ``[rows, layers, kv_heads, head_dim]``, the address of its fused K/V GEMM output
``[rows, layers * 2 * kv_heads * head_dim]`` whose layer ``l`` row is ``[K_l | V_l]``, its
rows, its block-table row and its first position) and the drafter layers' page addresses,
and for every request, row and layer writes K to plane 0 and V to plane 1 of the page
``table[row * width + position // page]`` at offset ``position % page``: the slots
flashinfer's ``append_paged_kv_cache`` writes for the same rows (HND pages
``[pages, 2, kv_heads, page, head_dim]``, every layer with the same strides).

BF16 in. BF16 pages get copies. FP8 E4M3 pages (``TRTLLM_DFLASH_FP8_DRAFT_KV``) get each value
clamped to +-448 in fp32 (NaN kept) and rounded to nearest even, as ``store.to_cache_dtype``
converts it. The program grid is (row blocks of the longest request, requests); a program
past its request's rows returns.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from . import store as _store

#: Elements a program moves per tensor and layer.
_TILE = 4096


@triton.jit
def _to_e4m3(x):
    """``x`` as FP8 E4M3: clamped to +-448 in fp32 with NaN kept, rounded to nearest even."""
    return tl.clamp(x.to(tl.float32), -448.0, 448.0, propagate_nan=tl.PropagateNan.ALL).to(
        tl.float8e4nv
    )


@triton.jit
def _draft_kv_page_store_kernel(
    meta_ptr,
    layer_ptr,
    table_ptr,
    table_width,
    page_size,
    stride_page,
    stride_plane,
    stride_head,
    stride_token,
    LAYERS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WIDTH: tl.constexpr,
    META: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_W: tl.constexpr,
    OUT_FP8: tl.constexpr = False,
):
    run = tl.program_id(1)
    entry = meta_ptr + run * META
    rows = tl.load(entry + 2)
    t0 = tl.program_id(0).to(tl.int64) * BLOCK_T
    if t0 >= rows:
        return
    k_src = tl.load(entry).to(tl.pointer_type(tl.bfloat16))
    kv_src = tl.load(entry + 1).to(tl.pointer_type(tl.bfloat16))
    table_row = tl.load(entry + 3)
    first = tl.load(entry + 4)

    t = t0 + tl.arange(0, BLOCK_T).to(tl.int64)
    t_mask = t < rows
    position = first + t
    page = tl.load(
        table_ptr + table_row * table_width + position // page_size, mask=t_mask, other=0
    ).to(tl.int64)
    token = position % page_size

    w = tl.arange(0, BLOCK_W)
    w_mask = w < WIDTH
    head = (w // HEAD_DIM).to(tl.int64)
    dim = (w % HEAD_DIM).to(tl.int64)
    mask = t_mask[:, None] & w_mask[None, :]
    dst = (
        page[:, None] * stride_page
        + token[:, None] * stride_token
        + head[None, :] * stride_head
        + dim[None, :]
    )
    k_row = t[:, None] * (LAYERS * WIDTH) + w[None, :]
    kv_row = t[:, None] * (2 * LAYERS * WIDTH) + w[None, :]
    for layer in tl.static_range(LAYERS):
        if OUT_FP8:
            cache = tl.load(layer_ptr + layer).to(tl.pointer_type(tl.float8e4nv))
        else:
            cache = tl.load(layer_ptr + layer).to(tl.pointer_type(tl.bfloat16))
        k = tl.load(k_src + k_row + layer * WIDTH, mask=mask)
        v = tl.load(kv_src + kv_row + (2 * layer + 1) * WIDTH, mask=mask)
        if OUT_FP8:
            k = _to_e4m3(k)
            v = _to_e4m3(v)
        tl.store(cache + dst, k, mask=mask)
        tl.store(cache + stride_plane + dst, v, mask=mask)


def store(
    meta: torch.Tensor,
    layer_addresses: torch.Tensor,
    table: torch.Tensor,
    cache: torch.Tensor,
    *,
    num_runs: int,
    max_rows: int,
    layers: int,
    heads: int,
    head_dim: int,
) -> None:
    """Write every request's K and V rows of the step into the drafter layers' pages.

    ``meta`` int64 ``[num_runs * META_FIELDS]`` and ``layer_addresses`` int64 ``[layers]`` on
    the device; ``table`` the contiguous int32 block table; ``cache`` one drafter layer's
    pages (their dtype, strides and page size; every layer has the same).
    """
    if num_runs == 0 or max_rows == 0:
        return
    width = heads * head_dim
    block_w = triton.next_power_of_2(width)
    block_t = max(1, min(64, _TILE // block_w))
    grid = (triton.cdiv(max_rows, block_t), num_runs)
    _draft_kv_page_store_kernel[grid](
        meta,
        layer_addresses,
        table,
        table.shape[1],
        cache.shape[3],
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        cache.stride(3),
        LAYERS=layers,
        HEAD_DIM=head_dim,
        WIDTH=width,
        META=_store.META_FIELDS,
        BLOCK_T=block_t,
        BLOCK_W=block_w,
        OUT_FP8=cache.dtype == torch.float8_e4m3fn,
        num_warps=4,
    )
