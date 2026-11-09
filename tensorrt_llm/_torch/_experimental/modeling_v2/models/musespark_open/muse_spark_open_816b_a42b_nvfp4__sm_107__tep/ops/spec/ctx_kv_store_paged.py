# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash context K/V store into a paged drafter context: every drafter layer in one launch.

With the drafter's context K/V in pages (``TRTLLM`` drafter attention), the worker's
``_store_context_kv_paged`` (``_torch/speculative/dflash.py``) stores each drafter
layer with flashinfer's ``append_paged_kv_cache``, after copying that layer's K and V
slices contiguous: three launches per layer. Each entry ``i`` goes to page
``table[rows[i], positions[i] // page_size]`` at row ``positions[i] % page_size``
of the layer's HND pages (``[pages, 2, heads, page_size, head_dim]``, K then V).

This entry writes every layer's K and V of each entry in one program per entry, reading
the inputs in place. Copies only: bitwise the built-in.
"""

from __future__ import annotations

from typing import Sequence

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

#: Drafter layers one call takes (one pointer argument each).
MAX_LAYERS = 8


@triton.jit
def _ctx_kv_store_paged_kernel(
    k_ptr,
    v_ptr,
    stride_k_row,
    stride_k_layer,
    stride_k_head,
    stride_v_row,
    stride_v_layer,
    stride_v_head,
    rows_ptr,
    positions_ptr,
    table_ptr,
    stride_table_row,
    page_size,
    stride_page,
    stride_kv,
    stride_head,
    stride_entry,
    c0,
    c1,
    c2,
    c3,
    c4,
    c5,
    c6,
    c7,
    LAYERS: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    i = tl.program_id(0).to(tl.int64)
    row = tl.load(rows_ptr + i).to(tl.int64)
    pos = tl.load(positions_ptr + i).to(tl.int64)
    page = tl.load(table_ptr + row * stride_table_row + pos // page_size).to(tl.int64)
    entry = pos % page_size
    h = tl.arange(0, HEADS).to(tl.int64)
    d = tl.arange(0, HEAD_DIM).to(tl.int64)
    dst = page * stride_page + entry * stride_entry + h[:, None] * stride_head + d[None, :]
    for layer in tl.static_range(LAYERS):
        if layer == 0:
            cache = c0
        elif layer == 1:
            cache = c1
        elif layer == 2:
            cache = c2
        elif layer == 3:
            cache = c3
        elif layer == 4:
            cache = c4
        elif layer == 5:
            cache = c5
        elif layer == 6:
            cache = c6
        else:
            cache = c7
        key = tl.load(
            k_ptr
            + i * stride_k_row
            + layer * stride_k_layer
            + h[:, None] * stride_k_head
            + d[None, :]
        )
        value = tl.load(
            v_ptr
            + i * stride_v_row
            + layer * stride_v_layer
            + h[:, None] * stride_v_head
            + d[None, :]
        )
        tl.store(cache + dst, key)
        tl.store(cache + stride_kv + dst, value)


class CtxKvStorePaged(OpWrapper):
    """Store a step's drafter context K/V of every layer into its pages in one launch.

    Arguments (``reference`` takes the same ones):
        k, v           [n, layers, heads, head_dim], the drafter's K (post norm and RoPE)
                       and V; head_dim stride 1
        rows           [n] int64, the page-table row of each entry (batch positions with
                       the manager's tables, slots with the private one)
        positions      [n] int64, each entry's position in its context
        table          [rows, blocks] int32, the page table (``_ctx_block_tables`` or
                       ``_ctx_page_table``); column stride 1
        page_size      the pages' rows (``_ctx_page_size``)
        caches         the layers' HND pages (``_ctx_kv_buf``), [pages, 2, heads,
                       page_size, head_dim] each, the same strides for every layer, head_dim
                       stride 1, dtype of k and v
    """

    ARCHS = frozenset({Arch.SM_107})

    note = """
    Metadata consumed: the drafter's page table (read). One Triton program per entry
    (grid n, 1 warp), JIT-compiled per (layers, heads, head_dim); no
    programmatic-dependent-launch attributes.

    Call site: ``ContextKvForkDFlashWorker._store_context_kv_paged`` under the
    ``paged_ctx_kv_store`` fast path, in place of the worker's per-layer appends.

    Numerics: copies, bitwise the built-in method (the reference).

    Hard limits
      layers <= MAX_LAYERS; heads and head_dim powers of two.
    """

    def __call__(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        rows: torch.Tensor,
        positions: torch.Tensor,
        table: torch.Tensor,
        page_size: int,
        caches: Sequence[torch.Tensor],
    ) -> None:
        n, layers, heads, head_dim = k.shape
        first = caches[0]
        pointers = list(caches) + [first] * (MAX_LAYERS - layers)
        _ctx_kv_store_paged_kernel[(n,)](
            k,
            v,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            rows,
            positions,
            table,
            table.stride(0),
            page_size,
            first.stride(0),
            first.stride(1),
            first.stride(2),
            first.stride(3),
            *pointers,
            LAYERS=layers,
            HEADS=heads,
            HEAD_DIM=head_dim,
            num_warps=1,
        )

    def reference(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        rows: torch.Tensor,
        positions: torch.Tensor,
        table: torch.Tensor,
        page_size: int,
        caches: Sequence[torch.Tensor],
    ) -> None:
        """The worker's per-layer ``append_paged_kv_cache`` calls, as plain indexing."""
        pages = table[rows.long(), positions.long() // page_size].long()
        entries = positions.long() % page_size
        for layer, cache in enumerate(caches):
            cache[pages, 0, :, entries] = k[:, layer]
            cache[pages, 1, :, entries] = v[:, layer]

    def is_valid(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        rows: torch.Tensor,
        positions: torch.Tensor,
        table: torch.Tensor,
        page_size: int,
        caches: Sequence[torch.Tensor],
    ) -> None:
        if k.dim() != 4 or v.shape != k.shape or k.stride(3) != 1 or v.stride(3) != 1:
            raise ValueError("k and v must be [n, layers, heads, head_dim] with head_dim stride 1")
        n, layers, heads, head_dim = k.shape
        if not 1 <= layers <= MAX_LAYERS or len(caches) != layers:
            raise ValueError(f"1..{MAX_LAYERS} layers, one cache each")
        if heads & (heads - 1) or head_dim & (head_dim - 1):
            raise ValueError("heads and head_dim must be powers of two")
        if rows.dtype != torch.int64 or positions.dtype != torch.int64:
            raise ValueError("rows and positions must be int64")
        if (
            rows.shape != (n,)
            or positions.shape != (n,)
            or not rows.is_contiguous()
            or not positions.is_contiguous()
        ):
            raise ValueError("rows and positions must be contiguous [n]")
        if table.dtype != torch.int32 or table.dim() != 2 or table.stride(1) != 1:
            raise ValueError("table must be int32 [rows, blocks] with column stride 1")
        first = caches[0]
        for cache in caches:
            if (
                cache.dim() != 5
                or cache.dtype != k.dtype
                or cache.shape[1] != 2
                or cache.shape[2] != heads
                or cache.shape[3] != page_size
                or cache.shape[4] != head_dim
                or cache.stride() != first.stride()
                or cache.stride(4) != 1
            ):
                raise ValueError(
                    "caches must be [pages, 2, heads, page_size, head_dim] of one layout"
                )


ctx_kv_store_paged = CtxKvStorePaged()
