# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoE decode route front above 32 tokens (512 experts, top-10).

After the stock router GEMM, routing + trtllm-gen pre-routed tables + NVFP4 quantization of the MoE input run in
one launch of thread-block clusters. :mod:`moe_decode_front` covers M <= 32 with a single routing CTA. Outputs are
bitwise: ids / BF16 weights as :mod:`moe_decode_front`, xq / xsf as ``fp4_quantize`` (linear scale layout), tables
(``prerouted_table_sizes`` order) as the Triton route kernel's up to the row order inside an expert's block.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op

from .moe_decode_front import (
    _SORT16,
    _add,
    _asm,
    _below,
    _bit,
    _div_bf16,
    _exp_tr,
    _f32,
    _gdc_launch,
    _gdc_wait,
    _i32,
    _i64,
    _key1,
    _key_val,
    _ld_f32,
    _ld_shared,
    _max_u32,
    _min_u32,
    _nvfp4_block,
    _popc,
    _red_or_shared,
    _redux_max_u32,
    _scan_add,
    _sel_eq,
    _sel_eq_f,
    _sel_lt,
    _smem_addr,
    _SmemAllocator,
    _st_b16,
    _st_b32,
    _st_shared,
    _st_u8,
    _st_v2,
    _struct,
    _sub,
    _unpack,
    max_num_tiles,
)


@dsl_user_op
def _keys2(word: Int32, idx: Int32, *, loc=None, ip=None):
    """Top-k keys of the two BF16 logits in ``word`` (expert ``idx`` in the low half, ``idx + 1`` in the high
    half): ``twiddle(bits) << 16 | (65535 - expert)``, unsigned order = value order, ties -> smaller expert.
    twiddle: negative -> bits ^ 0xFFFF, else bits | 0x8000."""
    out = _asm(
        _struct(2),
        [_i32(word, loc, ip), _i32(idx, loc, ip)],
        "{\n\t.reg .b32 lo, hi, t, m, bb;\n\t.reg .pred p;\n\t"
        "and.b32 lo, $2, 65535;\n\tshr.u32 hi, $2, 16;\n\t"
        "and.b32 t, lo, 32768;\n\tsetp.ne.u32 p, t, 0;\n\tselp.b32 m, 65535, 32768, p;\n\t"
        "xor.b32 lo, lo, m;\n\t"
        "and.b32 t, hi, 32768;\n\tsetp.ne.u32 p, t, 0;\n\tselp.b32 m, 65535, 32768, p;\n\t"
        "xor.b32 hi, hi, m;\n\t"
        "mov.b32 bb, 65535;\n\tsub.u32 bb, bb, $3;\n\t"
        "shl.b32 lo, lo, 16;\n\tor.b32 $0, lo, bb;\n\t"
        "sub.u32 bb, bb, 1;\n\tshl.b32 hi, hi, 16;\n\tor.b32 $1, hi, bb;\n\t}",
        "=r,=r,r,r",
        False,
        loc=loc,
        ip=ip,
    )
    return _unpack(out, 2, loc, ip)


ROUTE_CLUSTER = 8
ROUTE_THREADS = 1024
# Clusters of 16 (route mode; a non-portable size, which the DSL allows on every kernel) take 257..512 tokens with
# the same one-warp-per-token routing and 32-bit per-CTA token masks (TPC = ceil(M / 16) <= 32). The launcher then
# uses 3 quantization clusters, so the grid stays at 64 CTAs. 513..1024 tokens: TPC <= 64, two tokens per warp
# (the full-sort routing path) and two-word per-expert token masks.
ROUTE_CLUSTER_WIDE = 16


@dsl_user_op
def _cluster_arrive_release(*, loc=None, ip=None) -> None:
    _asm(None, [], "barrier.cluster.arrive.release.aligned;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _cluster_wait_acquire(*, loc=None, ip=None) -> None:
    _asm(None, [], "barrier.cluster.wait.acquire.aligned;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _ld_dsmem_b32(addr: Int32, rank: Int32, *, loc=None, ip=None) -> Int32:
    """The word at shared-memory address ``addr`` of cluster CTA ``rank``."""
    return Int32(
        _asm(
            T.i32(),
            [_i32(addr, loc, ip), _i32(rank, loc, ip)],
            "{\n\t.reg .b32 ra;\n\tmapa.shared::cluster.u32 ra, $1, $2;\n\t"
            "ld.shared::cluster.b32 $0, [ra];\n\t}",
            "=r,r,r,~{memory}",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_cg_v4w(addr: Int64, *, loc=None, ip=None):
    out = _asm(
        _struct(4),
        [_i64(addr, loc, ip)],
        "ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l,~{memory}",
        loc=loc,
        ip=ip,
    )
    return _unpack(out, 4, loc, ip)


@dsl_user_op
def _cluster_arrive_relaxed(*, loc=None, ip=None) -> None:
    _asm(None, [], "barrier.cluster.arrive.relaxed.aligned;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def _shfl_idx_f32(x: Float32, src: int, *, loc=None, ip=None) -> Float32:
    """``x`` of lane ``src`` (whole warp)."""
    return Float32(
        _asm(
            T.f32(),
            [_f32(x, loc, ip)],
            f"shfl.sync.idx.b32 $0, $1, {src}, 31, 0xffffffff;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _vote_any(x: Int32, *, loc=None, ip=None) -> Int32:
    """1 if ``x != 0`` in any lane of the warp, else 0."""
    return Int32(
        _asm(
            T.i32(),
            [_i32(x, loc, ip)],
            "{\n\t.reg .pred p, q;\n\tsetp.ne.u32 p, $1, 0;\n\tvote.sync.any.pred q, p, 0xffffffff;\n\t"
            "selp.u32 $0, 1, 0, q;\n\t}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


def _cas(a, b):
    """(max, min) of two keys (unsigned)."""
    return _max_u32(a, b), _min_u32(a, b)


def _merge22(x, y):
    """Two descending pairs -> the descending 4."""
    c0, m1 = _cas(x[0], y[0])
    m2, c3 = _cas(x[1], y[1])
    c1, c2 = _cas(m1, m2)
    return (c0, c1, c2, c3)


def _top4of8(a, b):
    """The 4 largest of two descending 4s, descending (bitonic: max(a_i, b_3-i), then a 4-element bitonic sort)."""
    t0 = _max_u32(a[0], b[3])
    t1 = _max_u32(a[1], b[2])
    t2 = _max_u32(a[2], b[1])
    t3 = _max_u32(a[3], b[0])
    u0, u2 = _cas(t0, t2)
    u1, u3 = _cas(t1, t3)
    v0, v1 = _cas(u0, u1)
    v2, v3 = _cas(u2, u3)
    return (v0, v1, v2, v3)


class MoeRouteFrontKernel:
    """Route-only big front: router logits [M, E] BF16 from the stock GEMM -> top-k ids / BF16 weights, the
    trtllm-gen pre-routed tables (``prerouted_table_sizes`` order, token order inside an expert's block) and the
    launch of ``1 + quant_clusters`` clusters of C CTAs (``threads`` each, default 1024; C = ``cluster``: 8, or 16
    for 257..1024 tokens):

    * cluster 0 routes: CTA c takes tokens [c TPC, (c + 1) TPC) (TPC = ceil(M / C) <= 32, or <= 64 with clusters of
      16), one warp per token (two interleaved when TPC > #warps; the routing arithmetic of :mod:`moe_decode_front`);
      the local picks set the token's bit in the CTA's per-expert token mask (shared memory; ceil(TPC / 32)
      words per expert). After a cluster barrier every routing CTA reads the
      cluster's per-expert counts through distributed shared memory, scans the CTAs per expert (``tile``) and writes
      the expanded -> permuted / permuted -> token entries of its own picks (CTA 0 also the CTA tables and totals);
    * the other clusters NVFP4-quantize x (``fp4_quantize`` arithmetic, linear scale layout).

    No CTA waits for another cluster, so nothing needs the clusters co-resident. PDL: everything after
    ``griddepcontrol.wait``; every CTA releases its dependents after a barrier that follows its last global
    access."""

    def __init__(
        self,
        *,
        num_rows: int,
        k_extent: int,
        num_experts: int,
        top_k: int,
        num_local: int,
        tile: int,
        a_stride: int,
        use_pdl: bool,
        quant_clusters: int = 7,
        threads: int = ROUTE_THREADS,
        cluster: int = ROUTE_CLUSTER,
    ) -> None:
        if threads % 32 or not 256 <= threads <= 1024:
            raise ValueError(f"threads={threads} must be a multiple of 32 in [256, 1024]")
        if cluster not in (ROUTE_CLUSTER, ROUTE_CLUSTER_WIDE):
            raise ValueError(f"cluster={cluster} must be {ROUTE_CLUSTER} or {ROUTE_CLUSTER_WIDE}")
        max_tpc = 64 if cluster == ROUTE_CLUSTER_WIDE else 32  # token-mask words per expert: 2 / 1
        if not 1 <= num_rows <= cluster * max_tpc:
            raise ValueError(
                f"num_rows={num_rows} must be in [1, {cluster * max_tpc}] with clusters of {cluster}"
            )
        if top_k != 10:
            raise ValueError("the renormalization tree is the route kernel's top-10 tree")
        if num_experts != 32 * 16:
            raise ValueError("one warp per token holds 16 logits per lane: 512 experts")
        if num_local % 32 or not 32 <= num_local <= min(num_experts, threads):
            raise ValueError(f"num_local={num_local} must be a multiple of 32 in [32, {threads}]")
        if tile & (tile - 1) or not 8 <= tile <= 256:
            raise ValueError(f"tile={tile} must be a power of two in [8, 256]")
        if k_extent % 16 or a_stride % 8 or a_stride < k_extent:
            raise ValueError("x rows must be 16-byte aligned blocks of 16")
        if quant_clusters < 1:
            raise ValueError("at least one quantization cluster")
        self.num_rows = num_rows
        self.k_extent = k_extent
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local = num_local
        self.tile = tile
        self.a_stride = a_stride
        self.use_pdl = use_pdl
        self.threads = threads
        self.cluster = cluster
        self.tpc = -(-num_rows // cluster)
        self.mw = -(-self.tpc // 32)  # token-mask words per expert (TPC > 32: two)
        if num_local * self.mw > threads:
            raise ValueError(
                "one thread per token-mask word: num_local * ceil(TPC / 32) <= threads"
            )
        self.warps = threads // 32
        self.tpw = -(-self.tpc // self.warps)
        if self.tpw > 2:
            raise ValueError("at most two tokens per warp")
        self.quant_clusters = quant_clusters
        self.ctas = cluster * (1 + quant_clusters)
        self.q_ctas = cluster * quant_clusters
        self.blocks = num_rows * (k_extent // 16)
        self.q_passes = -(-self.blocks // (self.q_ctas * threads))
        self.max_ctas = max_num_tiles(num_rows, top_k, num_local, tile)
        self.max_ctas_per_expert = -(-num_rows // tile)
        self.off_p2t = num_rows * top_k
        self.off_cte = self.off_p2t + (self.max_ctas + 1) * tile
        self.off_ctl = self.off_cte + self.max_ctas
        self.off_nct = self.off_ctl + self.max_ctas
        self.off_npd = self.off_nct + 1
        # shared memory words: mask [NL * MW], count, prefix (earlier CTAs), total, block offset [NL] each, warp
        # sums [32], picks [TPC * K]
        nl = num_local
        mw = self.mw
        self.o_mask = 0
        self.o_cnt = mw * nl
        self.o_pre = (mw + 1) * nl
        self.o_tot = (mw + 2) * nl
        self.o_offs = (mw + 3) * nl
        self.o_wsum = (mw + 4) * nl
        self.o_pick = (mw + 4) * nl + 32
        self.smem_words = self.o_pick + self.tpc * top_k + 16

    @cute.jit
    def __call__(
        self,
        logits_ptr: Int64,
        x_ptr: Int64,
        gs_ptr: Int64,
        xq_ptr: Int64,
        xsf_ptr: Int64,
        ids_ptr: Int64,
        wts_ptr: Int64,
        tab_ptr: Int64,
        local_start: Int32,
        stream,
    ) -> None:
        self.kernel(
            logits_ptr,
            x_ptr,
            gs_ptr,
            xq_ptr,
            xsf_ptr,
            ids_ptr,
            wts_ptr,
            tab_ptr,
            local_start,
        ).launch(
            grid=[self.ctas, 1, 1],
            block=[self.threads, 1, 1],
            cluster=[self.cluster, 1, 1],
            smem=self.smem_words * 4,
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        logits_ptr: Int64,
        x_ptr: Int64,
        gs_ptr: Int64,
        xq_ptr: Int64,
        xsf_ptr: Int64,
        ids_ptr: Int64,
        wts_ptr: Int64,
        tab_ptr: Int64,
        local_start: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        smem = _SmemAllocator()
        buf = smem.allocate_tensor(Int32, cute.make_layout((self.smem_words,)), byte_alignment=16)
        base = _smem_addr(buf.iterator)
        if bidx < self.cluster:
            self._route_cta(logits_ptr, ids_ptr, wts_ptr, tab_ptr, local_start, base, bidx, tidx)
        else:
            self._quant_cta(x_ptr, gs_ptr, xq_ptr, xsf_ptr, bidx - self.cluster, tidx)

    @cute.jit
    def _quant_cta(
        self, x_ptr: Int64, gs_ptr: Int64, xq_ptr: Int64, xsf_ptr: Int64, qc, tidx
    ) -> None:
        bpr: cutlass.Constexpr = self.k_extent // 16
        gs = _ld_f32(gs_ptr)  # static (the experts' input scale): before the wait
        if const_expr(self.use_pdl):
            _gdc_wait()
        for p in cutlass.range_constexpr(self.q_passes):
            blk = (p * self.q_ctas + qc) * self.threads + tidx
            if blk < self.blocks:
                row = blk // bpr
                cb = blk % bpr
                src = x_ptr + (Int64(row) * Int64(self.a_stride) + Int64(cb) * Int64(16)) * Int64(2)
                w0, w1, w2, w3 = _ld_cg_v4w(src)
                w4, w5, w6, w7 = _ld_cg_v4w(src + Int64(16))
                lo, hi, sf = _nvfp4_block(w0, w1, w2, w3, w4, w5, w6, w7, gs)
                _st_v2(
                    xq_ptr + Int64(row) * Int64(self.k_extent // 2) + Int64(cb) * Int64(8), lo, hi
                )
                _st_u8(xsf_ptr + Int64(row) * Int64(bpr) + Int64(cb), sf)
        cute.arch.sync_threads()
        if const_expr(self.use_pdl):
            _gdc_launch()

    @cute.jit
    def _route_cta(
        self,
        logits_ptr: Int64,
        ids_ptr: Int64,
        wts_ptr: Int64,
        tab_ptr: Int64,
        local_start: Int32,
        base: Int32,
        c,
        tidx,
    ) -> None:
        num_local: cutlass.Constexpr = self.num_local
        top_k: cutlass.Constexpr = self.top_k
        tile: cutlass.Constexpr = self.tile
        tpc: cutlass.Constexpr = self.tpc
        warp = tidx // 32
        lane = tidx % 32
        smask = base + Int32(self.o_mask * 4)
        scnt = base + Int32(self.o_cnt * 4)
        spre = base + Int32(self.o_pre * 4)
        stot = base + Int32(self.o_tot * 4)
        soffs = base + Int32(self.o_offs * 4)
        swsum = base + Int32(self.o_wsum * 4)
        spick = base + Int32(self.o_pick * 4)
        mw: cutlass.Constexpr = self.mw
        if tidx < num_local * mw:
            _st_shared(smask + tidx * 4, Int32(0))
        # this CTA's tokens [c TPC, c TPC + n_tok)
        t_first = c * tpc
        n_tok = cutlass.min(Int32(tpc), Int32(self.num_rows) - t_first)
        if const_expr(self.use_pdl):
            _gdc_wait()
        cute.arch.sync_threads()
        my_sel = Int32(0)
        my_w = Int32(0)
        if const_expr(self.tpw == 1):
            # one token per warp; its ids / weights are stored after the cluster barrier (the barrier's release
            # then only waits for shared memory)
            if warp < n_tok:
                my_sel, my_w = self._route_warp_fast(
                    logits_ptr, local_start, smask, spick, t_first, warp, lane
                )
        else:
            if warp < n_tok:
                self._route_warp(
                    logits_ptr,
                    ids_ptr,
                    wts_ptr,
                    local_start,
                    smask,
                    spick,
                    t_first,
                    n_tok,
                    warp,
                    lane,
                )
        cute.arch.sync_threads()
        if tidx < num_local:
            if const_expr(mw == 1):
                _st_shared(scnt + tidx * 4, _popc(_ld_shared(smask + tidx * 4)))
            else:
                _st_shared(
                    scnt + tidx * 4,
                    _popc(_ld_shared(smask + (tidx * 2) * 4))
                    + _popc(_ld_shared(smask + (tidx * 2 + 1) * 4)),
                )
        # The 8 CTAs' counts through distributed shared memory.
        _cluster_arrive_release()
        _cluster_wait_acquire()
        if const_expr(self.tpw == 1):
            if warp < n_tok:
                if lane < top_k:
                    o = Int64(t_first + warp) * Int64(top_k) + Int64(lane)
                    _st_b32(ids_ptr + o * Int64(4), my_sel)
                    _st_b16(wts_ptr + o * Int64(2), my_w)
        tot = Int32(0)
        nct = Int32(0)
        e = tidx
        if e < num_local:
            cnts = []
            for k in cutlass.range_constexpr(self.cluster):
                cnts.append(_ld_dsmem_b32(scnt + e * 4, Int32(k)))
            pre = Int32(0)
            for k in cutlass.range_constexpr(self.cluster):
                tot = tot + cnts[k]
                pre = pre + _sel_lt(Int32(k), c, cnts[k], Int32(0))
            _st_shared(spre + e * 4, pre)
            _st_shared(stot + e * 4, tot)
            nct = (tot + Int32(tile - 1)) // Int32(tile)
        # every peer's counts are read (the matching wait is at the end: this CTA's counts stay until then)
        _cluster_arrive_relaxed()
        # Exclusive scan of the per-expert CTA counts (one expert per thread).
        incl = nct
        for d in cutlass.range_constexpr(5):
            incl = _scan_add(incl, 1 << d)
        if lane == 31:
            _st_shared(swsum + warp * 4, incl)
        cute.arch.sync_threads()
        n_warps_exp: cutlass.Constexpr = -(-num_local // 32)  # warps holding experts
        wpre = Int32(0)
        total = Int32(0)
        for w in cutlass.range_constexpr(n_warps_exp):
            ws = _ld_shared(swsum + w * 4)
            wpre = wpre + _sel_lt(Int32(w), warp, ws, Int32(0))
            total = total + ws
        offs = incl - nct + wpre
        if e < num_local:
            _st_shared(soffs + e * 4, offs)
            if c == 0:
                for cc in cutlass.range_constexpr(self.max_ctas_per_expert):
                    if Int32(cc) < nct:
                        cta_i = offs + Int32(cc)
                        _st_b32(tab_ptr + Int64(self.off_cte + cta_i) * Int64(4), e)
                        _st_b32(
                            tab_ptr + Int64(self.off_ctl + cta_i) * Int64(4),
                            cutlass.min((cta_i + 1) * Int32(tile), offs * Int32(tile) + tot),
                        )
        if c == 0:
            if tidx == 0:
                _st_b32(tab_ptr + Int64(self.off_nct) * Int64(4), total)
                _st_b32(tab_ptr + Int64(self.off_npd) * Int64(4), total * Int32(tile))
        cute.arch.sync_threads()
        # This CTA's picks: expanded -> permuted, permuted -> token (token order inside a block).
        for u in cutlass.range_constexpr(-(-(tpc * top_k) // self.threads)):
            i = tidx + u * self.threads
            if i < n_tok * top_k:
                tl = i // top_k
                tok = t_first + tl
                pick = _ld_shared(spick + i * 4)
                perm = Int32(-1)
                if pick >= Int32(0):
                    if const_expr(mw == 1):
                        rank = _ld_shared(spre + pick * 4) + _popc(
                            _ld_shared(smask + pick * 4) & _below(tl)
                        )
                    else:
                        # tokens before tl: in word 0 (all of word 0 from tl 32 on), in word 1 below tl - 32
                        mw_w0 = _ld_shared(smask + (pick * 2) * 4)
                        mw_w1 = _ld_shared(smask + (pick * 2 + 1) * 4)
                        mw_lo = _sel_lt(tl, Int32(32), _popc(mw_w0 & _below(tl)), _popc(mw_w0))
                        mw_hi = _sel_lt(
                            tl, Int32(32), Int32(0), _popc(mw_w1 & _below(tl - Int32(32)))
                        )
                        rank = _ld_shared(spre + pick * 4) + mw_lo + mw_hi
                    perm = _ld_shared(soffs + pick * 4) * Int32(tile) + rank
                    _st_b32(tab_ptr + Int64(self.off_p2t + perm) * Int64(4), tok)
                _st_b32(
                    tab_ptr + (Int64(tok) * Int64(top_k) + Int64(i - tl * top_k)) * Int64(4), perm
                )
        _cluster_wait_acquire()  # every peer has read this CTA's counts
        cute.arch.sync_threads()
        if const_expr(self.use_pdl):
            _gdc_launch()

    @cute.jit
    def _route_warp_fast(
        self, logits_ptr: Int64, local_start: Int32, smask: Int32, spick: Int32, t_first, warp, lane
    ):
        """Token ``t_first + warp``: the same bits as :meth:`_route_warp`, but each lane keeps only its 4 largest
        keys; a lane that runs dry before the last round (holds >= 5 of the top 10) sends the warp to the full
        sort. Returns (expert id, BF16 weight bits) of pick ``lane`` (lanes < top_k)."""
        top_k: cutlass.Constexpr = self.top_k
        num_local: cutlass.Constexpr = self.num_local
        num_experts: cutlass.Constexpr = self.num_experts
        e0 = lane * 16
        src = logits_ptr + (Int64(t_first + warp) * Int64(num_experts) + Int64(e0)) * Int64(2)
        p0, p1, p2, p3 = _ld_cg_v4w(src)
        p4, p5, p6, p7 = _ld_cg_v4w(src + Int64(16))
        packed = (p0, p1, p2, p3, p4, p5, p6, p7)
        pairs = []
        for jw in cutlass.range_constexpr(8):
            klo, khi = _keys2(packed[jw], e0 + 2 * jw)
            pairs.append(_cas(klo, khi))
        quads = []
        for qd in cutlass.range_constexpr(4):
            quads.append(_merge22(pairs[2 * qd], pairs[2 * qd + 1]))
        c0, c1, c2, c3 = _top4of8(_top4of8(quads[0], quads[1]), _top4of8(quads[2], quads[3]))
        sel = cute.make_rmem_tensor(cute.make_layout((top_k,)), Int32)
        val = cute.make_rmem_tensor(cute.make_layout((top_k,)), Float32)
        zero = Int32(0)
        bad = Int32(0)
        for k in cutlass.range_constexpr(top_k):
            gmax = _redux_max_u32(c0)
            sel[k] = Int32(65535) - (gmax & Int32(65535))
            val[k] = _key_val(gmax)
            if const_expr(k < top_k - 1):
                # the owner lane's candidates move up by one (keys are unique)
                c0, c1, c2, c3 = (
                    _sel_eq(c0, gmax, c1, c0),
                    _sel_eq(c0, gmax, c2, c1),
                    _sel_eq(c0, gmax, c3, c2),
                    _sel_eq(c0, gmax, zero, c3),
                )
            if const_expr(k == top_k - 2):
                bad = _vote_any(_sel_eq(c0, zero, Int32(1), zero))
        if bad != 0:
            self._topk_full(packed, e0, sel, val)
        my_sel = sel[0]
        my_v = val[0]
        for k in cutlass.range_constexpr(top_k - 1):
            my_sel = _sel_eq(lane, Int32(k + 1), sel[k + 1], my_sel)
            my_v = _sel_eq_f(lane, Int32(k + 1), val[k + 1], my_v)
        my_e = _exp_tr(_sub(my_v, val[0]))
        ex = []
        for k in cutlass.range_constexpr(top_k):
            ex.append(_shfl_idx_f32(my_e, k))
        ssum = _add(
            _add(_add(_add(ex[0], ex[8]), ex[4]), _add(ex[2], ex[6])),
            _add(_add(_add(ex[1], ex[9]), ex[5]), _add(ex[3], ex[7])),
        )
        my_w = _div_bf16(my_e, ssum)
        if lane < top_k:
            lid = my_sel - local_start
            pick = _sel_lt(lid, Int32(num_local), lid, Int32(-1))
            if pick >= Int32(0):
                _red_or_shared(smask + pick * 4, _bit(warp))
            _st_shared(spick + (warp * top_k + lane) * 4, pick)
        return my_sel, my_w

    @cute.jit
    def _topk_full(self, packed, e0, sel: cute.Tensor, val: cute.Tensor) -> None:
        """Full per-lane sort + pop top-k into sel / val: fallback of :meth:`_route_warp_fast`."""
        top_k: cutlass.Constexpr = self.top_k
        keys = cute.make_rmem_tensor(cute.make_layout((16,)), Int32)
        for jw in cutlass.range_constexpr(8):
            keys[2 * jw] = _key1(packed[jw], e0 + 2 * jw)
            keys[2 * jw + 1] = _key1(packed[jw] >> Int32(16), e0 + 2 * jw + 1)
        for pi in cutlass.range_constexpr(len(_SORT16)):
            i, jj, imax = _SORT16[pi]
            x = keys[i]
            y = keys[jj]
            hi = _max_u32(x, y)
            lo = _min_u32(x, y)
            if const_expr(imax):
                keys[i] = hi
                keys[jj] = lo
            else:
                keys[i] = lo
                keys[jj] = hi
        for k in cutlass.range_constexpr(top_k):
            head = keys[0]
            gmax = _redux_max_u32(head)
            sel[k] = Int32(65535) - (gmax & Int32(65535))
            val[k] = _key_val(gmax)
            if const_expr(k < top_k - 1):
                for jk in cutlass.range_constexpr(top_k - 1):
                    keys[jk] = _sel_eq(head, gmax, keys[jk + 1], keys[jk])

    @cute.jit
    def _route_warp(
        self,
        logits_ptr: Int64,
        ids_ptr: Int64,
        wts_ptr: Int64,
        local_start: Int32,
        smask: Int32,
        spick: Int32,
        t_first,
        n_tok,
        warp,
        lane,
    ) -> None:
        """Tokens warp and warp + #warps (when TPC > #warps; clamped to the first, not stored when absent):
        logits from global memory (16 per lane), top-k / renormalization bitwise moe_decode_front, ids / BF16 weights
        out, local picks into the CTA's token masks and pick list."""
        top_k: cutlass.Constexpr = self.top_k
        num_local: cutlass.Constexpr = self.num_local
        num_experts: cutlass.Constexpr = self.num_experts
        tpw: cutlass.Constexpr = self.tpw
        e0 = lane * 16
        tl1 = warp
        if const_expr(tpw == 2):
            tl1 = _sel_lt(warp + Int32(self.warps), n_tok, warp + Int32(self.warps), warp)
        keys = cute.make_rmem_tensor(cute.make_layout((16 * tpw,)), Int32)
        for rr in cutlass.range_constexpr(tpw):
            tl = warp
            if const_expr(rr == 1):
                tl = tl1
            src = logits_ptr + (Int64(t_first + tl) * Int64(num_experts) + Int64(e0)) * Int64(2)
            p0, p1, p2, p3 = _ld_cg_v4w(src)
            p4, p5, p6, p7 = _ld_cg_v4w(src + Int64(16))
            packed = (p0, p1, p2, p3, p4, p5, p6, p7)
            for jw in cutlass.range_constexpr(8):
                keys[16 * rr + 2 * jw] = _key1(packed[jw], e0 + 2 * jw)
                keys[16 * rr + 2 * jw + 1] = _key1(packed[jw] >> Int32(16), e0 + 2 * jw + 1)
            for pi in cutlass.range_constexpr(len(_SORT16)):
                i, jj, imax = _SORT16[pi]
                x = keys[16 * rr + i]
                y = keys[16 * rr + jj]
                hi = _max_u32(x, y)
                lo = _min_u32(x, y)
                if const_expr(imax):
                    keys[16 * rr + i] = hi
                    keys[16 * rr + jj] = lo
                else:
                    keys[16 * rr + i] = lo
                    keys[16 * rr + jj] = hi
        sel = cute.make_rmem_tensor(cute.make_layout((top_k * tpw,)), Int32)
        val = cute.make_rmem_tensor(cute.make_layout((top_k * tpw,)), Float32)
        for k in cutlass.range_constexpr(top_k):
            for rr in cutlass.range_constexpr(tpw):
                head = keys[16 * rr]
                gmax = _redux_max_u32(head)
                sel[top_k * rr + k] = Int32(65535) - (gmax & Int32(65535))
                val[top_k * rr + k] = _key_val(gmax)
                if const_expr(k < top_k - 1):
                    for jk in cutlass.range_constexpr(top_k - 1):
                        keys[16 * rr + jk] = _sel_eq(
                            head, gmax, keys[16 * rr + jk + 1], keys[16 * rr + jk]
                        )
        for rr in cutlass.range_constexpr(tpw):
            tl = warp
            active = Int32(1)
            if const_expr(rr == 1):
                tl = tl1
                active = _sel_eq(tl1, warp, Int32(0), Int32(1))
            vmax = val[top_k * rr]
            ex = cute.make_rmem_tensor(cute.make_layout((top_k,)), Float32)
            for k in cutlass.range_constexpr(top_k):
                ex[k] = _exp_tr(_sub(val[top_k * rr + k], vmax))
            ssum = _add(
                _add(_add(_add(ex[0], ex[8]), ex[4]), _add(ex[2], ex[6])),
                _add(_add(_add(ex[1], ex[9]), ex[5]), _add(ex[3], ex[7])),
            )
            my_sel = sel[top_k * rr]
            my_e = ex[0]
            for k in cutlass.range_constexpr(top_k - 1):
                my_sel = _sel_eq(lane, Int32(k + 1), sel[top_k * rr + k + 1], my_sel)
                my_e = _sel_eq_f(lane, Int32(k + 1), ex[k + 1], my_e)
            if active != Int32(0):
                if lane < top_k:
                    o = Int64(t_first + tl) * Int64(top_k) + Int64(lane)
                    _st_b32(ids_ptr + o * Int64(4), my_sel)
                    _st_b16(wts_ptr + o * Int64(2), _div_bf16(my_e, ssum))
                    lid = my_sel - local_start
                    pick = _sel_lt(lid, Int32(num_local), lid, Int32(-1))
                    if pick >= Int32(0):
                        # token tl's bit in word tl // 32 of the expert's ceil(TPC / 32) mask words
                        _red_or_shared(
                            smask + (pick * Int32(self.mw) + tl // Int32(32)) * 4,
                            _bit(tl % Int32(32)),
                        )
                    _st_shared(spick + (tl * top_k + lane) * 4, pick)
