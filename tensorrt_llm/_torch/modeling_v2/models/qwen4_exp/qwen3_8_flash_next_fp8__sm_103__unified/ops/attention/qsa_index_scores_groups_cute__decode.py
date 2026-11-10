# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA index scores of a verify step's request groups: a CuTe DSL kernel with a cp.async ring.

A CTA (4 warps) walks `TILES` tiles (`BN` keys each) of one request group with `STAGES` tiles in
flight: the program's page numbers are looked up once into shared memory, each tile's keys go to
an XOR-swizzled shared ring with `cp.async` (four threads a key), and the scores come from
`mma.sync` m16n8k16 with the group's query columns as the A fragments (rows 4 t + h: verify row t,
index head h) and the keys as B. A key's score per row is the family's
(`qsa_paged_index_scores` with `rows_per_request`): ReLU of each head's dot, summed
(h0 + h1) + (h2 + h3), times 128^-0.5. The dot's accumulation order differs from the family
kernel's, so the scores differ in the last bits.

Precondition (`serves`): verify rows, `rows_per_request` (2 or 3) consecutive rows of one request
each, 4 bf16 index heads of 128; the index pool a `[pages, tokens, 1, 128]` view with token rows
256 B apart. Columns at and past a row's window are left unwritten.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from .qsa_sparse_gqa_splitk_cute__decode import (
    _asm,
    _cp16z,
    _f32_bits,
    _ir,
    _ldg1,
    _ldg4,
    _lds1,
    _lds4,
    _mma_bf16,
    _ptr,
    _stg1,
    _sts1,
)

HEADS = 4
HEAD_DIM = 128
#: Keys a tile: 64 keeps a 2-stage ring small enough for several CTAs an SM.
BN = 64
WARPS = 4
THREADS = 32 * WARPS
#: Bytes of a key row in the ring: unpadded, with 16-byte chunk `ch` of key `k` stored at chunk
#: `ch ^ 4 (k & 1)`, so the two keys a quarter-warp's 16-byte loads and cp.async writes touch fall
#: in disjoint bank halves without row padding.
K_LD = 256
#: Tiles a CTA walks, and tiles in flight (the fastest of the configurations measured on GB300).
TILES = 8
STAGES = 2


@dsl_user_op
def _stg2f(gaddr, a, b, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [
            _ir(gaddr, cutlass.Int64, loc, ip),
            _ir(_f32_bits(a), cutlass.Int32, loc, ip),
            _ir(_f32_bits(b), cutlass.Int32, loc, ip),
        ],
        "{ st.global.v2.b32 [$1], {$2, $3}; mov.u32 $0, 0; }",
        "=r,l,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def _shfl_xor_f(x, m, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [_ir(x, cutlass.Float32, loc, ip)],
            f"shfl.sync.bfly.b32 $0, $1, {m}, 31, 0xffffffff;",
            "=f,f",
            loc,
            ip,
            side=True,
        )
    )


class _IdxGroupsKernel:
    def __init__(self, *, rpr, tpb, cr, tiles, stages, bt_width, num_requests, pdl, bn=BN):
        assert 2 <= rpr <= 3 and tpb % cr == 0 and bn in (64, 128)
        self.T, self.tpb, self.cr = rpr, tpb, cr
        self.bn = bn
        self.tiles, self.stages = tiles, max(2, min(stages, tiles))
        self.bt_width, self.num_requests, self.pdl = bt_width, num_requests, pdl
        self.k_ld = K_LD
        self.o_pg = 0
        self.pg_bytes = tiles * bn * 4
        self.o_ring = -(-self.pg_bytes // 128) * 128
        self.smem = self.o_ring + self.stages * bn * self.k_ld + 128

    @cute.jit
    def __call__(
        self,
        q_ptr: cute.Pointer,
        cache_ptr: cute.Pointer,
        bt_ptr: cute.Pointer,
        pos_ptr: cute.Pointer,
        req_ptr: cute.Pointer,
        out_t: cute.Tensor,
        rows: cutlass.Int32,
        q_stride_row: cutlass.Int32,
        page_stride: cutlass.Int64,
        out_stride: cutlass.Int32,
        max_blocks: cutlass.Int32,
        groups: cutlass.Int32,
        col_progs: cutlass.Int32,
        stream,
    ):
        self.kernel(
            q_ptr.toint(),
            cache_ptr.toint(),
            bt_ptr.toint(),
            pos_ptr.toint(),
            req_ptr.toint(),
            # the output arrives as a tensor: the TVM-FFI entry takes its stream from it
            out_t.iterator.toint(),
            rows,
            q_stride_row,
            page_stride,
            out_stride,
            max_blocks,
            col_progs,
        ).launch(
            grid=[groups * col_progs, 1, 1],
            block=[THREADS, 1, 1],
            smem=self.smem,
            stream=stream,
            use_pdl=self.pdl,
        )

    @cute.jit
    def _issue(self, j, buf, tidx, s_pg, s_ring, cache_base, page_stride):
        """cp.async of tile `j`'s `bn` keys into ring slot `buf`, one commit group: thread t serves
        chunks q4, q4 + 4, q4 + 8, q4 + 12 of key t // 4 + 32 i (64 contiguous bytes for the four
        threads of a key); a key past the window (page -1) gets zeros."""
        tpb: cutlass.Constexpr = self.tpb
        cr: cutlass.Constexpr = self.cr
        bn: cutlass.Constexpr = self.bn
        k_ld: cutlass.Constexpr = self.k_ld
        q4 = tidx % 4
        dst0 = s_ring + buf * (bn * k_ld)
        for i in cutlass.range_constexpr(bn // (THREADS // 4)):
            k = tidx // 4 + (THREADS // 4) * i
            page = _lds1(s_pg + (j * bn + k) * 4)
            ok = page >= 0
            n = cutlass.Int32(cutlass.select_(ok, cutlass.Int32(16), cutlass.Int32(0)))
            tok = (k * cr + cr - 1) % tpb
            src = (
                cache_base
                + (
                    cutlass.Int64(cutlass.max(page, 0)) * page_stride
                    + cutlass.Int64(tok * HEAD_DIM)
                )
                * 2
            )
            for cc in cutlass.range_constexpr(4):
                ch = q4 + 4 * cc
                _cp16z(dst0 + k * k_ld + 16 * (ch ^ ((k & 1) << 2)), src + 16 * ch, n)
        cute.arch.cp_async_commit_group()

    @cute.kernel
    def kernel(
        self,
        q_base: cutlass.Int64,
        cache_base: cutlass.Int64,
        bt_base: cutlass.Int64,
        pos_base: cutlass.Int64,
        req_base: cutlass.Int64,
        out_base: cutlass.Int64,
        rows: cutlass.Int32,
        q_stride_row: cutlass.Int32,
        page_stride: cutlass.Int64,
        out_stride: cutlass.Int32,
        max_blocks: cutlass.Int32,
        col_progs: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        g = lane // 4
        c = lane % 4
        rpr: cutlass.Constexpr = self.T
        tpb: cutlass.Constexpr = self.tpb
        cr: cutlass.Constexpr = self.cr
        tiles: cutlass.Constexpr = self.tiles
        stages: cutlass.Constexpr = self.stages
        bn: cutlass.Constexpr = self.bn
        ntw: cutlass.Constexpr = self.bn // (8 * WARPS)  # n8 tiles a warp
        group = bidx // col_progs
        prog = bidx % col_progs

        smem = cutlass.utils.SmemAllocator()
        base_t = smem.allocate_tensor(
            cutlass.Int8, cute.make_layout((self.smem - 128,), stride=(1,)), byte_alignment=128
        )
        sbase = cutlass.Int32(base_t.iterator.toint())
        s_pg = sbase + self.o_pg
        s_ring = sbase + self.o_ring

        if const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        # ---- the group's rows: request, windows -------------------------------------------
        r0 = group * rpr
        lead = cutlass.Int64(_ldg1(req_base + cutlass.Int64(cutlass.min(r0, rows - 1)) * 4))
        safe_req = cutlass.min(
            cutlass.max(lead, cutlass.Int64(0)), cutlass.Int64(self.num_requests - 1)
        )
        vis = []
        max_vis = cutlass.Int32(0)
        for t in cutlass.range_constexpr(rpr):
            r = r0 + t
            p64 = cutlass.Int64(_ldg1(pos_base + cutlass.Int64(cutlass.min(r, rows - 1)) * 8))
            v = cutlass.Int32(
                cutlass.select_(r < rows, cutlass.Int32((p64 + 1) // cr), cutlass.Int32(0))
            )
            vis.append(v)
            max_vis = cutlass.max(max_vis, v)
        max_vis = cutlass.min(max_vis, max_blocks)
        first = prog * tiles
        n_t = cutlass.max(cutlass.min(cutlass.Int32(tiles), (max_vis + bn - 1) // bn - first), 0)

        # ---- the program's page numbers (page of each tile key, -1 past the window) ------
        for i in cutlass.range_constexpr(-(-(tiles * bn) // THREADS)):
            kk = tidx + THREADS * i
            blk = first * bn + kk
            lp = (blk * cr + cr - 1) // tpb
            ok = (blk < max_vis) & (lp < self.bt_width)
            pg = cutlass.Int32(-1)
            if ok:
                pg = _ldg1(bt_base + (safe_req * self.bt_width + cutlass.Int64(lp)) * 4)
            if kk < tiles * bn:
                _sts1(s_pg + kk * 4, pg)

        # ---- q of the group as A fragments: row m = 4 t + h; thread (g, c) holds dims
        # 32 m4 + 8 c .. + 7 of rows g and g + 8 (m4 = 0..3), the keys read the same dims. -----
        qa = cute.make_rmem_tensor(cute.make_layout((4, 4), stride=(4, 1)), cutlass.Int32)
        qb = cute.make_rmem_tensor(cute.make_layout((4, 4), stride=(4, 1)), cutlass.Int32)
        ta = g // 4
        tb = (g + 8) // 4
        ha = g % 4
        ra = cutlass.min(r0 + ta, rows - 1)
        rb = cutlass.min(r0 + tb, rows - 1)
        oka = (ta < rpr) & (r0 + ta < rows)
        okb = (tb < rpr) & (r0 + tb < rows)
        for m4 in cutlass.range_constexpr(4):
            off = cutlass.Int64((32 * m4 + 8 * c) * 2)
            wa = _ldg4(q_base + (cutlass.Int64(ra) * q_stride_row + ha * HEAD_DIM) * 2 + off)
            wb = _ldg4(q_base + (cutlass.Int64(rb) * q_stride_row + ha * HEAD_DIM) * 2 + off)
            for e in cutlass.range_constexpr(4):
                qa[m4, e] = cutlass.Int32(cutlass.select_(oka, wa[e], cutlass.Int32(0)))
                qb[m4, e] = cutlass.Int32(cutlass.select_(okb, wb[e], cutlass.Int32(0)))
        cute.arch.sync_threads()

        # one commit group a ring slot, empty past the program's tiles: the loop's wait_group(stages - 2)
        # then always leaves exactly the newer tiles in flight
        for s in cutlass.range_constexpr(stages - 1):
            if s < n_t:
                self._issue(s, s, tidx, s_pg, s_ring, cache_base, page_stride)
            else:
                cute.arch.cp_async_commit_group()

        scale = cutlass.Float32(HEAD_DIM**-0.5)
        out_a = out_base + cutlass.Int64(ra) * out_stride * 4
        out_b = out_base + cutlass.Int64(rb) * out_stride * 4
        vis_a = vis[0]
        vis_b = vis[0]
        for t in cutlass.range_constexpr(rpr):
            vis_a = cutlass.Int32(cutlass.select_(ta == t, vis[t], vis_a))
            vis_b = cutlass.Int32(cutlass.select_(tb == t, vis[t], vis_b))
        for j in cutlass.range(n_t):
            buf = j % stages
            cute.arch.cp_async_wait_group(stages - 2)
            cute.arch.sync_threads()
            if j + stages - 1 < n_t:
                self._issue(
                    j + stages - 1,
                    (j + stages - 1) % stages,
                    tidx,
                    s_pg,
                    s_ring,
                    cache_base,
                    page_stride,
                )
            else:
                cute.arch.cp_async_commit_group()
            rk = s_ring + buf * (bn * self.k_ld)
            for nt in cutlass.range_constexpr(ntw):
                key = warp * (8 * ntw) + nt * 8 + g
                zero = cutlass.Float32(0.0)
                d = [zero, zero, zero, zero]
                for m4 in cutlass.range_constexpr(4):
                    # chunk 4 m4 + c of key `key` (parity g & 1) sits at chunk (4 m4 + c) ^ 4 (g & 1)
                    b = _lds4(rk + key * self.k_ld + 16 * ((4 * m4 + c) ^ ((g & 1) << 2)))
                    a_lo = (qa[m4, 0], qb[m4, 0], qa[m4, 1], qb[m4, 1])
                    a_hi = (qa[m4, 2], qb[m4, 2], qa[m4, 3], qb[m4, 3])
                    d = _mma_bf16(a_lo, (b[0], b[1]), d)
                    d = _mma_bf16(a_hi, (b[2], b[3]), d)
                # d0, d1: row g (t = g // 4, head g % 4), keys 2c, 2c + 1 of the n8 tile;
                # d2, d3: row g + 8. A row's heads are lanes 4 apart: (h0 + h1) + (h2 + h3).
                rr = [cutlass.max(x, zero) for x in d]
                pr = [rr[e] + _shfl_xor_f(rr[e], 4) for e in range(4)]
                sc = [pr[e] + _shfl_xor_f(pr[e], 8) for e in range(4)]
                col = (first + j) * bn + warp * (8 * ntw) + nt * 8 + 2 * c
                if ha == 0:
                    # both keys inside the row's window: one 8-byte store; only the first: 4 bytes
                    if oka & (col + 1 < vis_a):
                        _stg2f(out_a + cutlass.Int64(col) * 4, sc[0] * scale, sc[1] * scale)
                    if oka & (col + 1 == vis_a):
                        _stg1(out_a + cutlass.Int64(col) * 4, _f32_bits(sc[0] * scale))
                    if okb & (col + 1 < vis_b):
                        _stg2f(out_b + cutlass.Int64(col) * 4, sc[2] * scale, sc[3] * scale)
                    if okb & (col + 1 == vis_b):
                        _stg1(out_b + cutlass.Int64(col) * 4, _f32_bits(sc[2] * scale))


_COMPILED: dict = {}


def serves(q: torch.Tensor, index_cache: torch.Tensor, rows_per_request: int) -> bool:
    """Verify rows this kernel serves: 2 or 3 rows a request, 4 bf16 index heads of 128 with unit
    dim stride and 16-byte aligned rows, a `[pages, tokens, 1, 128]` bf16 pool view with token rows
    256 B apart and 16-byte aligned pages."""
    return (
        rows_per_request in (2, 3)
        and q.ndim == 3
        and q.shape[1] == HEADS
        and q.shape[2] == HEAD_DIM
        and q.dtype == torch.bfloat16
        and q.stride(2) == 1
        and q.stride(1) == HEAD_DIM
        and (q.stride(0) * 2) % 16 == 0
        and q.data_ptr() % 16 == 0
        and q.shape[0] % rows_per_request == 0
        and index_cache.ndim == 4
        and index_cache.shape[2] == 1
        and index_cache.shape[3] == HEAD_DIM
        and index_cache.dtype == torch.bfloat16
        and index_cache.stride(3) == 1
        and index_cache.stride(1) == HEAD_DIM
        and (index_cache.stride(0) * 2) % 16 == 0
        and index_cache.data_ptr() % 16 == 0
    )


def index_scores_groups(
    q: torch.Tensor,
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_positions: torch.Tensor,
    request_indices: torch.Tensor,
    tokens_per_block: int,
    compress_ratio: int,
    use_pdl: bool,
    rows_per_request: int,
) -> torch.Tensor:
    """`qsa_paged_index_scores(..., only_visible_blocks=True, context_rows=False,
    rows_per_request=...)`'s scores to the last bits (`[rows, max_blocks]` fp32, columns at and
    past a row's window unspecified)."""
    rows = q.shape[0]
    assert block_table.dtype == torch.int32 and block_table.stride(1) == 1
    assert block_table.stride(0) == block_table.shape[1]
    assert query_positions.dtype == torch.int64 and request_indices.dtype == torch.int32
    max_blocks = block_table.shape[1] * tokens_per_block // compress_ratio
    out = torch.empty((rows, max_blocks), dtype=torch.float32, device=q.device)
    if rows == 0:
        return out
    groups = rows // rows_per_request
    col_progs = -(-max_blocks // (BN * TILES))
    pdl = bool(use_pdl and get_env_enable_pdl())
    key = (rows_per_request, tokens_per_block, compress_ratio, tuple(block_table.shape), pdl)
    args = (
        _ptr(q, cutlass.BFloat16),
        _ptr(index_cache, cutlass.BFloat16),
        _ptr(block_table, cutlass.Int32, 4),
        _ptr(query_positions, cutlass.Int64, 8),
        _ptr(request_indices, cutlass.Int32, 4),
        from_dlpack(out, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        cutlass.Int32(rows),
        cutlass.Int32(q.stride(0)),
        cutlass.Int64(index_cache.stride(0)),
        cutlass.Int32(out.stride(0)),
        cutlass.Int32(max_blocks),
        cutlass.Int32(groups),
        cutlass.Int32(col_progs),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )
    fn = _COMPILED.get(key)
    if fn is None:
        kern = _IdxGroupsKernel(
            rpr=rows_per_request,
            tpb=tokens_per_block,
            cr=compress_ratio,
            tiles=TILES,
            stages=STAGES,
            bt_width=block_table.shape[1],
            num_requests=block_table.shape[0],
            pdl=pdl,
        )
        fn = _COMPILED[key] = cute.compile(kern, *args, options="--opt-level 2 --enable-tvm-ffi")
    fn(
        q.data_ptr(),
        index_cache.data_ptr(),
        block_table.data_ptr(),
        query_positions.data_ptr(),
        request_indices.data_ptr(),
        out,
        rows,
        q.stride(0),
        index_cache.stride(0),
        out.stride(0),
        max_blocks,
        groups,
        col_progs,
    )
    return out
