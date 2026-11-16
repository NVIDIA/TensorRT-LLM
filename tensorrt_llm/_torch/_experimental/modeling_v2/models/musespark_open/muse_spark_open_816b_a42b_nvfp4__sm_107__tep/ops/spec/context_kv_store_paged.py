# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash context-K/V side chain over the KV cache manager's pages: its integer prologue and its store, one launch each.

``ops/spec/context_kv_store`` (``drafter_kv_side_chain``) serves the dense full-length arena. Over a PAGED drafter
context (the KV cache manager's pages, ``_ctx_block_tables``) the side chain runs the built-in statements, about 45
launches around its GEMMs:

* ``DFlashWorker._refresh_ctx_block_tables``: the manager's encoded block offsets of the step's requests copied to
  int64 (``.to(int64).clone()``, a ~3 MB copy at max_blocks_per_seq 32k), minus the pool's offset, the allocation
  extent ``amax((encoded >= 0) * cols)`` into ``_ctx_block_counts``, then ``clamp(min=0) // divisor`` cast to int32
  into ``_ctx_block_tables`` (another copy);
* ``_store_step_context``'s index math: ``ctx_len[slots] (- ctx_base[slots]) + arange(K + 1)``, the clamp to the
  request's allocation ``counts[rows] * page - 1`` and to 0, the positions ``ctx_len + r``, and inside the RoPE the
  positions as int32 repeated per drafter layer;
* ``_store_context_kv_paged``: ``rows`` / ``positions`` to int32, and per drafter layer ``k[:, l].contiguous()``,
  ``v[:, l].contiguous()`` and FlashInfer's ``append_paged_kv_cache`` (``AppendPagedKVCacheKernel``) into that
  layer's ``[pages, K/V, Hkv, page, D]`` pool.

``ContextKvPagedPrep`` decodes the block-table rows, the extents, the columns and the positions in one launch (one
program per request); ``ContextKvPagedScatter`` writes K (post norm and RoPE) and V (read in place from the fused
K/V GEMM's output) of every drafter layer into the pages in one launch. Integer arithmetic and copies only: every
value written is the built-in's (rows whose columns clamp onto one slot of an exhausted allocation: the last row is
kept, which the built-in's append leaves unspecified).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

#: Generation requests one call takes (``context_kv_store.MAX_REQUESTS``).
MAX_REQUESTS = 64
#: Block-table columns per loop iteration of the decode.
DECODE_BLOCK = 1024


@triton.jit
def _paged_prep_kernel(
    src_ptr,
    src_row_stride,
    table_ptr,
    table_row_stride,
    counts_ptr,
    ctx_len_ptr,
    ctx_base_ptr,
    slots_ptr,
    positions_ptr,
    col_ptr,
    pos_i32_ptr,
    width,
    block_offset,
    divisor,
    page_size,
    ROWS: tl.constexpr,
    LAYERS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_W: tl.constexpr,
    HAS_BASE: tl.constexpr,
):
    g = tl.program_id(0)
    # --- _refresh_ctx_block_tables for batch row g ---
    w = tl.arange(0, BLOCK_W)
    extent = tl.zeros([BLOCK_W], dtype=tl.int64)
    g64 = g.to(tl.int64)
    for start in range(0, width, BLOCK_W):
        cols = start + w
        keep = cols < width
        encoded = tl.load(src_ptr + g64 * src_row_stride + cols, mask=keep, other=0).to(tl.int64) - block_offset
        # (encoded >= 0) * _ctx_block_cols[c], _ctx_block_cols = arange(1, width + 1)
        valid = tl.where((encoded >= 0) & keep, (cols + 1).to(tl.int64), 0)
        extent = tl.maximum(extent, valid)
        decoded = tl.maximum(encoded, 0) // divisor
        tl.store(table_ptr + g64 * table_row_stride + cols, decoded.to(tl.int32), mask=keep)
    count = tl.max(extent, axis=0)
    tl.store(counts_ptr + g, count)
    # --- _store_step_context's index math for request g (rows = batch positions = g) ---
    r = tl.arange(0, BLOCK_R)
    in_rows = r < ROWS
    slot = tl.load(slots_ptr + g)
    ctx = tl.load(ctx_len_ptr + slot)
    pos = ctx + r.to(tl.int64)
    if HAS_BASE:
        col = (ctx - tl.load(ctx_base_ptr + slot)) + r.to(tl.int64)
    else:
        col = pos
    col = tl.maximum(tl.minimum(col, count * page_size - 1), 0)
    tl.store(positions_ptr + g * ROWS + r, pos, mask=in_rows)
    tl.store(col_ptr + g * ROWS + r, col, mask=in_rows)
    pos32 = pos.to(tl.int32)
    for layer in tl.static_range(LAYERS):
        tl.store(pos_i32_ptr + (g * ROWS + r) * LAYERS + layer, pos32, mask=in_rows)


class ContextKvPagedPrep(OpWrapper):
    """The paged side chain's block-table decode, extents, columns and positions, in one launch.

    ``src`` the manager's encoded block offsets of this pool and K/V field 0, ``[gens, width]`` (any row stride, unit
    column stride; int32 or int64); writes ``table[:gens]`` (int32), ``counts[:gens]`` (int64) as
    ``_refresh_ctx_block_tables`` does, and ``positions`` / ``col`` (int64 ``[gens * rows]``) and ``pos_i32`` (int32
    ``[gens * rows * layers]``) as ``_store_step_context`` computes them (``rows`` of request ``g`` = batch row ``g``):

    * ``e = src - offset``; ``counts[g] = max_c((e >= 0) * (c + 1))``; ``table[g] = (max(e, 0) // divisor).to(int32)``;
    * ``positions[g * rows + r] = ctx_len[slots[g]] + r``;
    * ``col = clamp(min(ctx_len[slot] (- ctx_base[slot]) + r, counts[g] * page - 1), min=0)``;
    * ``pos_i32 = positions.to(int32).repeat_interleave(layers)``.
    """

    ARCHS = frozenset({Arch.SM_107})

    note = """
    Metadata consumed: the attention metadata's encoded block offsets (read). One Triton program per request
    (grid gens, 4 warps), the decode looping over the table width. Call site: ``context_kv_fork.ContextKvFork
    .side_chain`` under ``drafter_kv_side_chain__PAGED``, first launch of the side stream.
    Numerics: integer arithmetic; int32 is the low word, as ``.to(torch.int32)``. Hard limits: rows <= 64,
    gens <= MAX_REQUESTS; floor division of non-negative values only (the clamp precedes it).
    """

    def __call__(
        self, src, table, counts, ctx_len, ctx_base, slots, positions, col, pos_i32,
        rows: int, layers: int, block_offset: int, divisor: int, page_size: int,
    ):
        gens = slots.numel()
        _paged_prep_kernel[(gens,)](
            src,
            src.stride(0),
            table,
            table.stride(0),
            counts,
            ctx_len,
            ctx_len if ctx_base is None else ctx_base,
            slots,
            positions,
            col,
            pos_i32,
            src.shape[1],
            block_offset,
            divisor,
            page_size,
            ROWS=rows,
            LAYERS=layers,
            BLOCK_R=triton.next_power_of_2(rows),
            BLOCK_W=DECODE_BLOCK,
            HAS_BASE=ctx_base is not None,
            num_warps=4,
        )

    def reference(
        self, src, table, counts, ctx_len, ctx_base, slots, positions, col, pos_i32,
        rows: int, layers: int, block_offset: int, divisor: int, page_size: int,
    ):
        """The built-in statements (``_refresh_ctx_block_tables`` + ``_store_step_context``'s index math)."""
        gens = slots.numel()
        encoded = src[:gens].to(torch.int64).clone()
        encoded.sub_(block_offset)
        cols = torch.arange(1, encoded.size(1) + 1, dtype=torch.long, device=src.device)
        valid_cols = (encoded >= 0) * cols
        counts[:gens].copy_(valid_cols.amax(dim=1))
        decoded = encoded.clamp_(min=0).div_(divisor, rounding_mode="floor")
        table[:gens].copy_(decoded.to(torch.int32))
        offsets = torch.arange(rows, dtype=torch.long, device=src.device)
        ctx = ctx_len[slots]
        col_idx = (ctx if ctx_base is None else ctx - ctx_base[slots]).unsqueeze(1) + offsets.unsqueeze(0)
        gen_capacity = counts[torch.arange(gens, device=src.device)] * page_size
        col_idx = torch.minimum(col_idx, (gen_capacity - 1).unsqueeze(1)).clamp_(min=0)
        pos = ctx.unsqueeze(1) + offsets.unsqueeze(0)
        positions.copy_(pos.reshape(-1))
        col.copy_(col_idx.reshape(-1))
        pos_i32.copy_(pos.reshape(-1).to(torch.int32).repeat_interleave(layers))

    def is_valid(
        self, src, table, counts, ctx_len, ctx_base, slots, positions, col, pos_i32,
        rows: int, layers: int, block_offset: int, divisor: int, page_size: int,
    ):
        gens = slots.numel()
        assert src.dim() == 2 and src.stride(1) == 1 and src.shape[0] >= gens
        assert src.dtype in (torch.int32, torch.int64)
        assert table.dtype == torch.int32 and table.stride(1) == 1 and table.shape[1] == src.shape[1]
        assert counts.dtype == torch.int64 and ctx_len.dtype == torch.int64 and slots.dtype == torch.int64
        assert ctx_base is None or ctx_base.dtype == torch.int64
        n = gens * rows
        assert positions.dtype == torch.int64 and positions.numel() == n and positions.is_contiguous()
        assert col.dtype == torch.int64 and col.numel() == n and col.is_contiguous()
        assert pos_i32.dtype == torch.int32 and pos_i32.numel() == n * layers and pos_i32.is_contiguous()
        assert 1 <= rows <= 64 and 1 <= gens <= MAX_REQUESTS and divisor >= 1 and page_size >= 1


context_kv_paged_prep = ContextKvPagedPrep()


@triton.jit
def _paged_scatter_kernel(
    k_ptr,
    kv_ptr,
    kv_row_stride,
    col_ptr,
    counts_ptr,
    table_ptr,
    table_row_stride,
    layer_ptrs_ptr,
    page_stride,
    plane_stride,
    head_stride,
    token_stride,
    page_size,
    ROWS: tl.constexpr,
    LAYERS: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    n = tl.program_id(0)
    g = n // ROWS
    r = n % ROWS
    col = tl.load(col_ptr + n)
    cap = tl.load(counts_ptr + g) * page_size
    # Rows clamped onto the allocation's last slot (or onto 0): keep the last one.
    if (r + 1 < ROWS) & ((col >= cap - 1) | (col <= 0)):
        nxt = tl.load(col_ptr + n + 1)
        if nxt == col:
            return
    page = tl.load(table_ptr + g.to(tl.int64) * table_row_stride + col // page_size).to(tl.int64)
    tok = (col % page_size).to(tl.int64)
    h = tl.arange(0, BLOCK_H)
    d = tl.arange(0, BLOCK_D)
    m = (h < HEADS)[:, None] & (d < HEAD_DIM)[None, :]
    for layer in tl.static_range(LAYERS):
        base = tl.load(layer_ptrs_ptr + layer).to(tl.pointer_type(k_ptr.dtype.element_ty))
        k = tl.load(k_ptr + (n * LAYERS + layer) * (HEADS * HEAD_DIM) + h[:, None] * HEAD_DIM + d[None, :], mask=m)
        # The fused K/V GEMM's row: per layer [K heads | V heads].
        v = tl.load(
            kv_ptr + n * kv_row_stride + layer * (2 * HEADS * HEAD_DIM) + HEADS * HEAD_DIM
            + h[:, None] * HEAD_DIM + d[None, :],
            mask=m,
        )
        dst = base + page * page_stride + h[:, None].to(tl.int64) * head_stride + tok * token_stride + d[None, :]
        tl.store(dst, k, mask=m)
        tl.store(dst + plane_stride, v, mask=m)


class ContextKvPagedScatter(OpWrapper):
    """Write a step's context K and V rows of every drafter layer into the paged pools, as the built-in's per-layer
    ``append_paged_kv_cache`` calls do (page ``table[g, col // page]``, slot ``col % page``, HND pools).

    ``k`` ``[rows_total, layers, heads * head_dim]`` contiguous (post norm + RoPE); ``kv`` the fused K/V GEMM's
    ``[rows_total, layers * 2 * heads * head_dim]`` (V read in place); ``layer_ptrs`` int64 ``[layers]``: each layer
    pool's data pointer, the pools ``[pages, 2, heads, page, head_dim]`` with the strides given.
    """

    ARCHS = frozenset({Arch.SM_107})

    note = """
    Metadata consumed: the decoded block table and extents (``ContextKvPagedPrep``'s). One program per stored row
    (grid gens * rows, 1 warp), every layer's K and V. Call site: ``context_kv_fork`` under
    ``drafter_kv_side_chain__PAGED``, last launch of the side chain. Copies only.
    """

    def __call__(self, k, kv, col, counts, table, layer_ptrs, pool, rows: int, layers: int, heads: int, head_dim: int):
        total = col.numel()
        _paged_scatter_kernel[(total,)](
            k,
            kv,
            kv.stride(0),
            col,
            counts,
            table,
            table.stride(0),
            layer_ptrs,
            pool.stride(0),
            pool.stride(1),
            pool.stride(2),
            pool.stride(3),
            pool.shape[3],
            ROWS=rows,
            LAYERS=layers,
            HEADS=heads,
            HEAD_DIM=head_dim,
            BLOCK_H=triton.next_power_of_2(heads),
            BLOCK_D=triton.next_power_of_2(head_dim),
            num_warps=1,
        )

    def reference(self, k, kv, col, counts, table, layer_ptrs, pool, rows: int, layers: int, heads: int, head_dim: int):
        """Index assignment into ``pool`` (the layer-0 pool; the op writes every layer's through ``layer_ptrs``):
        what the built-in append writes for layer 0. Rows clamped onto one slot: the last row's values."""
        total = col.numel()
        gens = total // rows
        b = torch.arange(gens, device=col.device).repeat_interleave(rows)
        page = table[b, col // pool.shape[3]].long()
        tok = col % pool.shape[3]
        v = kv.view(total, layers, 2, heads, head_dim)[:, 0, 1]
        for i in range(total):
            pool[page[i], 0, :, tok[i]] = k[i, 0].view(heads, head_dim)
            pool[page[i], 1, :, tok[i]] = v[i]

    def is_valid(self, k, kv, col, counts, table, layer_ptrs, pool, rows: int, layers: int, heads: int, head_dim: int):
        total = col.numel()
        assert k.is_contiguous() and k.shape == (total, layers, heads * head_dim)
        assert kv.stride(1) == 1 and kv.shape[0] == total and kv.shape[1] == layers * 2 * heads * head_dim
        assert pool.dim() == 5 and pool.shape[1] == 2 and pool.shape[2] == heads and pool.shape[4] == head_dim
        assert pool.stride(4) == 1 and pool.dtype == k.dtype == kv.dtype
        assert layer_ptrs.dtype == torch.int64 and layer_ptrs.numel() == layers
        assert col.dtype == torch.int64 and counts.dtype == torch.int64 and table.dtype == torch.int32


context_kv_paged_scatter = ContextKvPagedScatter()
