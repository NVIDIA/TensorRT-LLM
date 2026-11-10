# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A batch's pages of an e4m3 paged K/V pool, widened into a bf16 scratch pool, and the block
table that addresses them there: the context launches over a kv_fp8 pool read their requests'
pages through this scratch with the bf16 row-tile kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Elements a program widens per loop step.
_CHUNK = 4096

#: A dep4 rank's K/V geometry and QSA table at max_seq_len 262144 (8196 columns); `slot_pages`
#: role pages a slot (24 for the 12 trunk layers, 26 with the MTP draft layer's planes last).
_CHECKPOINT = dict(
    kv_heads=2,
    head_dim=256,
    tokens_per_block=32,
    width=8196,
    slots=32,
    slot_pages=24,
    layer=11,
    holes=False,
    straddle=False,
)


def _spec(kv_lens, **overrides) -> dict:
    """`kv_lens` is one KV length per listed request, in order."""
    return {**_CHECKPOINT, "kv_lens": tuple(kv_lens), **overrides}


# Per-step and per-engine sizes are not specialized on (none feeds a vector access).
@triton.jit(do_not_specialize=["num_pool_pages", "listed", "bt_stride", "table_stride"])
def _fp8_pages_to_bf16_kernel(
    k_cache,
    v_cache,
    block_table,
    slots,
    counts,
    offsets,
    table,
    k_out,
    v_out,
    num_pool_pages,
    listed,
    stride_page,
    bt_stride,
    table_stride,
    PAGE_ELEMS: tl.constexpr,
    CHUNK: tl.constexpr,
):
    request = tl.program_id(0)
    count = tl.load(counts + request)
    base = tl.load(offsets + request).to(tl.int64)
    slot = tl.load(slots + request).to(tl.int64)
    for page in range(tl.program_id(1), count, tl.num_programs(1)):
        dst = base + page
        if dst < listed:
            src = tl.load(block_table + slot * bt_stride + page).to(tl.int64)
            resident = (src >= 0) & (src < num_pool_pages)
            tl.store(table + slot * table_stride + page, tl.where(resident, dst, -1).to(tl.int32))
            safe = tl.where(resident, src, 0)
            for start in tl.static_range(0, PAGE_ELEMS, CHUNK):
                elems = start + tl.arange(0, CHUNK)
                k = tl.load(k_cache + safe * stride_page + elems, mask=resident, other=0.0)
                v = tl.load(v_cache + safe * stride_page + elems, mask=resident, other=0.0)
                tl.store(k_out + dst * PAGE_ELEMS + elems, k.to(tl.bfloat16))
                tl.store(v_out + dst * PAGE_ELEMS + elems, v.to(tl.bfloat16))


#: Programs per request; each strides the request's pages.
_PAGE_PROGRAMS = 1024  # capacity: fixed: a launch width; a request may list any page count


class QsaFp8PagesToBf16(OpWrapper):
    """K and V of listed requests' pages of an e4m3 HND paged pool, as bf16 scratch pools,
    and the block table that addresses them there.

    Request `r` lists its first `counts[r]` table pages, `block_table[slots[r], j]` for `j <
    counts[r]`, at scratch pages `offsets[r] + j` (those below `listed`, the scratch's page
    count)::

        k_out[offsets[r] + j] = bf16(k_cache[block_table[slots[r], j]])

    and `v_out` likewise. Each e4m3 code widens exactly to bf16 (no scale: the pool holds
    K/V at unit scale, as `qsa_paged_sparse_gqa` reads it), so a bf16 kernel over the scratch
    reads the values the e4m3 kernels read. `table[slots[r], j]` becomes `offsets[r] + j`,
    or -1 where the pool's entry is unmapped or out of the pool (whose scratch page is then
    zeros); the call writes no other entry of `table`, which the caller fills with -1.
    `offsets` is the exclusive prefix sum of `counts`. Returns `(k_out, v_out, table)`.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="one request, one page", spec=_spec((17,))),
        Cell(
            why="dep4 deep chunk: one request at 198160 tokens (6193 pages) over a pool "
            "straddling 2^31 elements of the trunk plane",
            spec=_spec((198160,), straddle=True),
        ),
        Cell(
            why="three requests, a chunked continuation and two short ones",
            spec=_spec((40000 + 3192, 3000, 900)),
        ),
        Cell(why="holes: unmapped and out-of-pool entries", spec=_spec((4000, 2000), holes=True)),
        Cell(
            why="a tp4 / tep4 rank's single KV head, [128, 8196] table",
            spec=_spec((9000, 2100), kv_heads=1, slots=128),
        ),
        Cell(
            why="ht_dep4 (dep4 + MTP): the trunk's 26 role pages a slot, the draft layer's "
            "planes (layer 12), one request at 196576 tokens straddling 2^31",
            spec=_spec((196576,), slot_pages=26, layer=12, straddle=True),
        ),
    )

    note = """
    Source: one Triton kernel (`_fp8_pages_to_bf16_kernel`), `(requests, min(width, 1024))`
    programs, each striding its request's pages. Page ids are widened to int64 before the
    stride product: the engine's trunk planes pass 2^31 elements.

    Preconditions: e4m3 role planes `[pages, Hkv, tpb, D]` with a page's heads, tokens and
    dims contiguous (`stride(1) == tpb * D`, `stride(2) == D`, unit last stride; any page
    stride), K's and V's at equal strides; `k_out` / `v_out` contiguous bf16 `[listed, Hkv,
    tpb, D]`; `table` int32 of `block_table`'s shape. Bitwise deterministic (a copy).
    """

    def __call__(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        slots: torch.Tensor,
        counts: torch.Tensor,
        offsets: torch.Tensor,
        table: torch.Tensor,
        k_out: torch.Tensor,
        v_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        requests, listed = slots.shape[0], k_out.shape[0]
        _, heads, tpb, dim = k_cache.shape
        if k_cache.dtype != torch.float8_e4m3fn or v_cache.dtype != k_cache.dtype:
            raise ValueError("the widening copy reads e4m3 K/V planes")
        if k_cache.stride() != v_cache.stride() or k_cache.stride()[1:] != (tpb * dim, dim, 1):
            raise ValueError("K/V planes must share strides with contiguous head pages")
        if not (k_out.is_contiguous() and v_out.is_contiguous()):
            raise ValueError("the bf16 scratch pools must be contiguous")
        if k_out.dtype != torch.bfloat16 or k_out.shape[1:] != k_cache.shape[1:]:
            raise ValueError("the scratch pools must be bf16 pages of the pool's geometry")
        page_elems = heads * tpb * dim
        if requests and listed:
            _fp8_pages_to_bf16_kernel[(requests, min(block_table.shape[1], _PAGE_PROGRAMS))](
                k_cache,
                v_cache,
                block_table,
                slots,
                counts,
                offsets,
                table,
                k_out,
                v_out,
                k_cache.shape[0],
                listed,
                k_cache.stride(0),
                block_table.stride(0),
                table.stride(0),
                PAGE_ELEMS=page_elems,
                CHUNK=min(_CHUNK, page_elems),
                num_warps=8,
            )
        return k_out, v_out, table

    def reference(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        slots: torch.Tensor,
        counts: torch.Tensor,
        offsets: torch.Tensor,
        table: torch.Tensor,
        k_out: torch.Tensor,
        v_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_ref, v_ref, table_ref = k_out.clone(), v_out.clone(), table.clone()
        listed, pool_pages = k_out.shape[0], k_cache.shape[0]
        for r in range(slots.shape[0]):
            slot, base = int(slots[r]), int(offsets[r])
            count = max(0, min(int(counts[r]), block_table.shape[1], listed - base))
            if count == 0:
                continue
            src = block_table[slot, :count].long()
            resident = (src >= 0) & (src < pool_pages)
            dst = torch.arange(base, base + count, device=src.device)
            table_ref[slot, :count] = torch.where(resident, dst, -1).to(table.dtype)
            safe = torch.where(resident, src, 0)
            keep = resident[:, None, None, None]
            k_ref[base : base + count] = torch.where(keep, k_cache[safe].float(), 0.0).to(
                k_out.dtype
            )
            v_ref[base : base + count] = torch.where(keep, v_cache[safe].float(), 0.0).to(
                v_out.dtype
            )
        return k_ref, v_ref, table_ref


qsa_fp8_pages_to_bf16 = QsaFp8PagesToBf16()
