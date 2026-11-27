# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TAILGLUE_A: the DSpark step's post-join drafter glue in one Triton launch.

After the fork joins, a forked decode step runs four launches before the drafter's first layer, each a few hundred
nanoseconds of work behind a launch gap:

* the DRAFTER_CTX_KV_STORE_FUSED context K / V store (``ops/dspark/dspark_ctx_store._ctx_store_kernel``, the column
derived from
  the position on a bound pool: ``COL_FROM_POS`` + ``BOUND``), one program per (row, layer, KV head);
* the DRAFTER_CTX_ADVANCE_FUSED context advance (``ops/dspark/dspark_ctx_advance._advance_only_kernel``), one program
per
  request, and its read-view copy (``_read_view_kernel``, ADV_PAR), one program per (request, column chunk);
* the DRAFTER_FORWARD_INDICES_FUSED forward indices of the drafter's paged block decode
(``ops/dspark/dspark_forward_indices``,
  this target's C++ op ``minimax_m31_dspark_indices`` / its Triton kernel).

This kernel is one launch of all four program kinds. The store and advance programs are those kernels' bodies line for
line (the same loads, the same arithmetic in the same order, the same stores), and the forward indices are written by
the advance program of the same request from the advertised length it just stored. The read-view programs cannot read
the rebase the advance program stores (no ordering between the programs of one launch); each recomputes it with the
advance's own integer lines from the request's context length before this step's advance, which is the request's first
context position (``positions[r * (K + 1)]``: the drafter inputs wrote ``ctx_len[slot] + j`` and nothing writes
``ctx_len`` until the advance). No program reads what another program of the launch writes: the store writes cache
pages (reading the absolute block table), the advance writes the context lengths and its outputs, the read view writes
the read table (reading the absolute block table). Integer copies and the store's element-wise bf16 / fp32 arithmetic,
no reduction: bitwise the four launches.
"""

import torch
import triton
import triton.language as tl

BLOCK_V = (
    512  # columns per read-view program (16 B per thread at 4 warps); does not change any value
)


# n, nch and the scalar bounds depend on the batch and the pool, never on the values: not specialized, so a new batch
# shape does not compile a variant at a value that happens to be 1 or a multiple of 16. Every other integer (the
# strides, page and block sizes, iblock, pages_per_slot, ipage) is fixed per deployment, and on a forked step
# (num_contexts == 0) every pointer is a fresh allocation or a view at offset 0, so one variant serves every step; it
# compiles at the CUDA-graph warmup's eager step, before the capture.
@triton.jit(do_not_specialize=["n", "nch", "max_ctx", "dummy_slot", "width"])
def _tail_glue_kernel(
    # DRAFTER_CTX_KV_STORE_FUSED store (dspark_ctx_store._ctx_store_kernel, COL_FROM_POS + BOUND), programs [0, NS)
    kv_ptr,  # [N, L * 2 * NKV * HD] the fused K / V GEMM output (V is read here)
    k_ptr,  # [N, L, NKV, HD] K after the k_norm (and, without FOLD_W, the weight multiply)
    w_ptr,  # [L, HD] bf16 k_norm weight (FOLD_W)
    cs_ptr,  # [max_pos, ROT] fp32 cos | sin
    pos_ptr,  # [N] int64 RoPE positions = context columns (request-major, KP1 rows per request)
    nacc_ptr,  # [n] accepted tokens per request
    rows_ptr,  # [n] int64 block-table row per request
    counts_ptr,  # [rows] int64 pages per block-table row
    table_ptr,  # int32 [rows, width] absolute page ids (the store's table and the read view's source)
    cache_ptr,  # stacked [L, pages, 2, NKV, PS, HD] cache, or an int64 array of L per-layer cache addresses
    kv_stride_row,
    table_stride,
    cache_stride_layer,
    # DRAFTER_CTX_ADVANCE_FUSED advance (dspark_ctx_advance._advance_only_kernel), programs [NS, NS + n)
    slots_ptr,  # [n] int64
    ctx_len_ptr,  # [slots] int64, in / out
    base_ptr,  # [slots] int64
    out_ptr,  # [n] int64 advertised context lengths
    max_ctx,
    dummy_slot,
    page,
    block_size,
    # DRAFTER_CTX_ADVANCE_FUSED read view (dspark_ctx_advance._read_view_kernel), programs [NS + n, NS + n + n * nch)
    read_ptr,  # int32 [rows, width]
    width,
    read_stride,
    nch,  # read-view programs per request: cdiv(width, BLOCK_V)
    # DRAFTER_FORWARD_INDICES_FUSED forward indices (dspark_forward_indices._indices_kernel), written by the advance
    # programs
    qpos_ptr,  # [n, iblock] query positions (int32 / int64), strided
    qpos_s0,
    qpos_s1,
    seqlens_ptr,  # out [n] int32
    batch_out_ptr,  # out [n] int32
    after_ptr,  # out [n] int32
    indptr_ptr,  # out [n + 1] int32
    last_ptr,  # out [n] int32
    append_ptr,  # out [n * iblock] int32
    qflat_ptr,  # out [n * iblock] int32
    iblock,  # the drafter's query block
    pages_per_slot,
    ipage,  # the drafter cache's page size
    n,
    KP1: tl.constexpr,
    L: tl.constexpr,
    NKV: tl.constexpr,
    HD: tl.constexpr,
    ROT: tl.constexpr,
    PS: tl.constexpr,
    PTR_LIST: tl.constexpr,
    FOLD_W: tl.constexpr,
    BLOCK_Q: tl.constexpr,  # next power of two of iblock
    BLOCK: tl.constexpr,  # BLOCK_V
):
    pid = tl.program_id(0)
    NS = n * (KP1 * L * NKV)
    half: tl.constexpr = HD // 2
    rhalf: tl.constexpr = ROT // 2
    # Every branch names its values apart (s_ / a_ / v_): one launch, three program kinds.
    if pid < NS:
        # ---- the store: _ctx_store_kernel's program (r, layer, h), COL_FROM_POS and BOUND
        s_r = pid // (L * NKV)
        s_layer = (pid // NKV) % L
        s_h = pid % NKV
        s_o = tl.arange(0, half)
        s_ro = tl.arange(0, rhalf)
        s_req = s_r // KP1
        s_m = tl.where((s_r % KP1) < tl.load(nacc_ptr + s_req), 1.0, 0.0)
        s_vbase = kv_ptr + s_r * kv_stride_row + (s_layer * 2 * NKV + NKV + s_h) * HD
        s_kbase = k_ptr + ((s_r * L + s_layer) * NKV + s_h) * HD
        s_a = tl.load(s_kbase + s_ro)
        s_b = tl.load(s_kbase + rhalf + s_ro)
        if FOLD_W:  # torch's bf16 * bf16: the fp32 product rounded to bf16
            s_wbase = w_ptr + s_layer * HD
            s_a = (s_a.to(tl.float32) * tl.load(s_wbase + s_ro).to(tl.float32)).to(tl.bfloat16)
            s_b = (s_b.to(tl.float32) * tl.load(s_wbase + rhalf + s_ro).to(tl.float32)).to(
                tl.bfloat16
            )
        s_a = s_a.to(tl.float32)
        s_b = s_b.to(tl.float32)
        s_p = tl.load(pos_ptr + s_r)
        s_c = tl.load(cs_ptr + s_p * ROT + s_ro)
        s_s = tl.load(cs_ptr + s_p * ROT + rhalf + s_ro)
        s_ra = (tl.fma(s_a, s_c, -s_b * s_s).to(tl.bfloat16).to(tl.float32) * s_m).to(tl.bfloat16)
        s_rb = (tl.fma(s_b, s_c, s_a * s_s).to(tl.bfloat16).to(tl.float32) * s_m).to(tl.bfloat16)
        s_v0 = (tl.load(s_vbase + s_o).to(tl.float32) * s_m).to(tl.bfloat16)
        s_v1 = (tl.load(s_vbase + half + s_o).to(tl.float32) * s_m).to(tl.bfloat16)
        s_row = tl.load(rows_ptr + s_req)
        s_last = (s_r % KP1) == KP1 - 1
        s_p_nxt = tl.load(pos_ptr + tl.where(s_last, s_r, s_r + 1))
        s_cap = tl.load(counts_ptr + s_row) * PS - 1
        s_col = tl.maximum(tl.minimum(s_p, s_cap), 0)
        s_nxt = tl.maximum(tl.minimum(s_p_nxt, s_cap), 0)
        s_pg = tl.load(table_ptr + s_row * table_stride + s_col // PS).to(tl.int64)
        if PTR_LIST:
            s_cache = tl.load(cache_ptr + s_layer).to(tl.pointer_type(tl.bfloat16))
        else:
            s_cache = cache_ptr + s_layer.to(tl.int64) * cache_stride_layer
        s_kdst = s_cache + ((s_pg * 2 * NKV + s_h) * PS + s_col % PS) * HD
        s_vdst = s_cache + ((s_pg * 2 * NKV + NKV + s_h) * PS + s_col % PS) * HD
        # Clamped rows of one request share a column; the last of them writes (the catalog kernel's rule).
        s_write = s_last | (s_nxt != s_col)
        tl.store(s_kdst + s_ro, s_ra, mask=s_write)
        tl.store(s_kdst + rhalf + s_ro, s_rb, mask=s_write)
        if ROT < HD:  # partial rotary: dims ROT..HD are not rotated, only masked
            s_po = ROT + tl.arange(0, HD - ROT)
            s_kp = tl.load(s_kbase + s_po)
            if FOLD_W:
                s_kp = (
                    s_kp.to(tl.float32) * tl.load(w_ptr + s_layer * HD + s_po).to(tl.float32)
                ).to(tl.bfloat16)
            s_kp = (s_kp.to(tl.float32) * s_m).to(tl.bfloat16)
            tl.store(s_kdst + s_po, s_kp, mask=s_write)
        tl.store(s_vdst + s_o, s_v0, mask=s_write)
        tl.store(s_vdst + half + s_o, s_v1, mask=s_write)
    elif pid < NS + n:
        # ---- the advance: _advance_only_kernel's program r, then the forward indices of request r
        a_r32 = pid - NS
        a_r = a_r32.to(tl.int64)
        a_slot = tl.load(slots_ptr + a_r)
        a_row = tl.load(rows_ptr + a_r)
        a_dummy = a_slot == dummy_slot
        a_old = tl.load(ctx_len_ptr + a_slot)
        a_new = tl.minimum(a_old + tl.load(nacc_ptr + a_r).to(tl.int64), max_ctx)
        if not a_dummy:
            tl.store(ctx_len_ptr + a_slot, a_new)
        if a_r == 0:
            tl.store(ctx_len_ptr + dummy_slot, tl.zeros((), dtype=tl.int64))
        a_new = tl.where(a_dummy, 0, a_new)
        a_alloc = tl.maximum(tl.load(counts_ptr + a_row) * page - block_size, 0)
        a_num = tl.minimum(a_new, a_alloc)
        a_base = tl.load(base_ptr + a_slot)
        a_out = tl.maximum(a_num - a_base * page, 0)
        tl.store(out_ptr + a_r, a_out)
        # _indices_kernel's lines for row r (num_ctx = the advertised length just stored, batch_idx = the row)
        a_seq = a_out.to(tl.int32)
        tl.store(seqlens_ptr + a_r, a_seq)
        tl.store(batch_out_ptr + a_r, a_row.to(tl.int32))
        a_after = a_seq + iblock
        tl.store(after_ptr + a_r, a_after)
        a_lp = (a_after - 1) % ipage
        a_lp = tl.where(
            a_lp < 0, a_lp + ipage, a_lp
        )  # torch's floored modulo for a positive divisor
        tl.store(last_ptr + a_r, a_lp + 1)
        tl.store(indptr_ptr + a_r32, a_r32 * pages_per_slot)
        if a_r32 == n - 1:
            tl.store(indptr_ptr + n, n * pages_per_slot)
        a_j = tl.arange(0, BLOCK_Q)
        a_mj = a_j < iblock
        tl.store(append_ptr + a_r32 * iblock + a_j, a_seq + a_j, mask=a_mj)
        a_q = tl.load(qpos_ptr + a_r * qpos_s0 + a_j * qpos_s1, mask=a_mj, other=0).to(tl.int32)
        tl.store(qflat_ptr + a_r32 * iblock + a_j, a_q, mask=a_mj)
    else:
        # ---- the read view: _read_view_kernel's program (r, chunk), with the advance's rebase recomputed
        v_i = pid - NS - n
        v_r = (v_i // nch).to(tl.int64)
        v_chunk = v_i % nch
        v_slot = tl.load(slots_ptr + v_r)
        v_row = tl.load(rows_ptr + v_r)
        v_dummy = v_slot == dummy_slot
        v_old = tl.load(pos_ptr + v_r * KP1)  # ctx_len[slot] before this step's advance
        v_new = tl.minimum(v_old + tl.load(nacc_ptr + v_r).to(tl.int64), max_ctx)
        v_new = tl.where(v_dummy, 0, v_new)
        v_alloc = tl.maximum(tl.load(counts_ptr + v_row) * page - block_size, 0)
        v_num = tl.minimum(v_new, v_alloc)
        v_base = tl.load(base_ptr + v_slot)
        v_rebase = tl.minimum(v_base, v_num // page)
        v_j = v_chunk * BLOCK + tl.arange(0, BLOCK)
        v_mk = v_j < width
        v_src = tl.minimum(v_j + v_rebase, width - 1)
        v_val = tl.load(table_ptr + v_row * table_stride + v_src, mask=v_mk)
        tl.store(read_ptr + v_row * read_stride + v_j, v_val, mask=v_mk)


def launch(
    store,
    slots: torch.Tensor,
    ctx_len: torch.Tensor,
    max_ctx: int,
    dummy_slot: int,
    base_pages: torch.Tensor,
    tables_read: torch.Tensor,
    page_size: int,
    block_size: int,
    query_positions: torch.Tensor,
    iblock: int,
    ipage: int,
):
    """One launch of the store (``store``: the DRAFTER_CTX_KV_STORE_FUSED store's operands,
    ``dspark_tailfork.StoreOperands`` with COL_FROM_POS and BOUND), the context advance + read view of ``store``'s
    requests and their forward indices.

    Returns (advertised context lengths [n] int64, (cache_seqlens, cache_batch_idx, seq_lens_after, kv_indptr,
    kv_last_page_len, append_positions, query_positions_flat) int32), the outputs of ``advance_ctx_and_read_view`` and
    of ``forward_indices(out, rows, query_positions, iblock, width, ipage)``. The caller checks every operand
    (``TailGlue.advance``)."""
    meta = store.meta
    rows = store.request_rows
    n = rows.shape[0]
    tables = store.block_table
    width = tables.shape[1]
    dev = ctx_len.device
    out = torch.empty(n, dtype=torch.int64, device=dev)
    i32 = dict(dtype=torch.int32, device=dev)
    seqlens = torch.empty(n, **i32)
    batch_out = torch.empty(n, **i32)
    after = torch.empty(n, **i32)
    indptr = torch.empty(n + 1, **i32)
    last = torch.empty(n, **i32)
    append = torch.empty(n * iblock, **i32)
    qflat = torch.empty(n * iblock, **i32)
    nch = triton.cdiv(width, BLOCK_V)
    ns = n * meta["KP1"] * meta["L"] * meta["NKV"]
    _tail_glue_kernel[(ns + n + n * nch,)](
        store.kv_flat,
        store.k,
        store.w,
        store.cos_sin,
        store.positions,
        store.num_accepted,
        rows,
        store.counts,
        tables,
        store.cache,
        store.kv_stride_row,
        store.table_stride,
        store.cache_stride_layer,
        slots,
        ctx_len,
        base_pages,
        out,
        int(max_ctx),
        int(dummy_slot),
        int(page_size),
        int(block_size),
        tables_read,
        width,
        tables_read.stride(0),
        nch,
        query_positions,
        query_positions.stride(0),
        query_positions.stride(1),
        seqlens,
        batch_out,
        after,
        indptr,
        last,
        append,
        qflat,
        int(iblock),
        width,
        int(ipage),
        n,
        KP1=meta["KP1"],
        L=meta["L"],
        NKV=meta["NKV"],
        HD=meta["HD"],
        ROT=meta["ROT"],
        PS=meta["PS"],
        PTR_LIST=meta["PTR_LIST"],
        FOLD_W=meta["FOLD_W"],
        BLOCK_Q=triton.next_power_of_2(int(iblock)),
        BLOCK=BLOCK_V,
        num_warps=4,
    )
    return out, (seqlens, batch_out, after, indptr, last, append, qflat)
