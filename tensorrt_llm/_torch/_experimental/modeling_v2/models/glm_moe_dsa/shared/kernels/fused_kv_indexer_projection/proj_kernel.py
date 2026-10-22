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
"""CuTe DSL kernel of the fused_kv_indexer_projection merged projection with staged weights.

Computes, for ``R <= 8`` token rows of a row-major BF16 activation
``a [R, K]``,

    y[t, j] = bf16(sum_k a[t, k] * W[j, k])

over the concatenation of two row-major BF16 weight tensors with the same
``K`` (``W0 [N0, K]`` = the fused ``[q_a | kv_a | k_pe]`` projection,
``W1 [N1, K]`` = the indexer ``wk | weights_proj`` rows of a full-indexer
layer, or none), writing ``out0 [R, N0]`` and ``out1 [R, N1]``. Both weight
tensors are read in place.

Accumulation order (the order of the native ``fused_a_gemm_kernel`` for
``nsplit = 4``): ``K`` is cut into ``nsplit`` contiguous slices; each slice
is one sequential chain of ``mma.sync.m16n8k16`` steps (weights as the A
operand, the activation as the B operand, K ascending, accumulator starting
at +0); the per-row slice partials are summed in slice order and the sum
``0 + s`` is rounded once to BF16.

Schedule (one CTA per SM, ``warps`` MMA warps):

* CTA ``c`` owns a contiguous range of output rows. Before
  ``griddepcontrol.wait`` (the weights are immutable) warp 0 copies up to
  ``smem_rows`` of them into shared memory with ``cp.async.bulk`` (row
  stride padded by 16 bytes) in sub-chunks ordered so that every warp's
  slice receives bytes at the same rate (``(w0, s0), (w1, s0), ..., (w0,
  s1), ...``), one single-phase mbarrier per sub-chunk. A CTA that owns one
  row more than it stages prefetches that row into L2 and reads it from
  global memory in the main loop (the "direct" row).
* After the wait, warp ``w`` loads its activation fragments from global
  memory through a rolling window of ``b_dist`` K steps and runs the chains
  of its K range for both m16 tiles of the CTA (rows 0-15 by
  ``ldmatrix.x4``, rows 16-23 by ``ldmatrix.x2``; rows the CTA does not
  stage read a zero chunk, and the lanes that hold the direct row take its
  fragment from global memory instead). Each finished slice partial is
  stored to a shared-memory reduction buffer.
* After a CTA barrier, thread ``(row, token)`` sums the slice partials in
  slice order and stores the BF16 result.

No global state is kept between launches (no counters, flags or scratch).
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32, Int64
from cutlass.utils.smem_allocator import SmemAllocator

from .ptx import (
    add_f32,
    bf16_bits,
    bulk_g2s,
    cp_async16,
    cp_async_mbar_arrive,
    fence_tensormap_acquire,
    globaltimer,
    ld_ca_b32,
    ld_ca_b32_if,
    ld_cg_b32,
    ld_cg_b32_if,
    ld_nc_b32,
    ld_shared_f32,
    ldsm_x2,
    ldsm_x4,
    mbar_arrive_expect_tx,
    mbar_fence_init,
    mbar_init,
    mbar_wait,
    min_i32,
    mma_bf16,
    prefetch_l2,
    saddr,
    sel_i32,
    sel_i64,
    st_global_b16,
    st_global_b64,
    st_shared_f32,
    st_shared_v4,
    sync_warp,
    tensor_g2s_3d,
    xor_i32,
)

MAX_TOKENS = 8
SMEM_LIMIT = 227 * 1024
# Rows held by one CTA's two m16 tiles: rows 0-15 (tile 0) and 16-23 (tile 1).
TILE_ROWS = 24
_MISC_BYTES = 1024
# Tensor (TMA) staging: 16 rows x 64 K elements (128 B, 128-byte swizzle) per
# K group; the rows of m16 tile 0 come from shared memory, rows 16.. of the
# CTA are read from global memory.
TMA_ROWS = 16
TMA_GROUP_K = 64
TMA_GROUP_BYTES = TMA_ROWS * TMA_GROUP_K * 2


def split_ctas(n0: int, n1: int, ctas: int):
    """CTA split of the tensor-staged schedule: ``(c0, c1)`` CTAs for the
    ``n0`` rows of ``W0`` and the ``n1`` rows of ``W1`` (no CTA straddles the
    two tensors), minimizing the largest row count; ``None`` if no split
    gives every CTA ``TMA_ROWS .. TILE_ROWS`` rows."""
    if n1 == 0:
        parts = [(0, ctas)]
    else:
        parts = [(c, ctas - c) for c in range(1, ctas)]
    best = None
    for c1, c0 in parts:
        groups = [(n0, c0)] + ([(n1, c1)] if n1 else [])
        lo = min(nn // cc for nn, cc in groups)
        hi = max(-(-nn // cc) for nn, cc in groups)
        if lo >= TMA_ROWS and hi <= TILE_ROWS and (best is None or hi < best[0]):
            best = (hi, c0, c1)
    return None if best is None else (best[1], best[2])


def _ordered_sum(vals, order: str, ftz: bool):
    """Trace-time helper: fp32 sum of ``vals`` in slice order (``seq``:
    ``((v0 + v1) + v2) + ...``; ``tree``: pairwise by index)."""
    vals = list(vals)
    if order == "seq":
        s = vals[0]
        for v in vals[1:]:
            s = add_f32(s, v, ftz)
        return s
    while len(vals) > 1:
        nxt = [add_f32(vals[2 * q], vals[2 * q + 1], ftz) for q in range(len(vals) // 2)]
        if len(vals) % 2:
            nxt.append(vals[-1])
        vals = nxt
    return vals[0]


@cute.jit
def _row_src(
    w0: Int64,
    w1: Int64,
    gr: Int32,
    K: cutlass.Constexpr[int],
    N0: cutlass.Constexpr[int],
    N1: cutlass.Constexpr[int],
) -> Int64:
    """Global address of row ``gr`` of the concatenation ``[W0; W1]``."""
    k2 = Int64(K * 2)
    src = w0 + Int64(gr) * k2
    if cutlass.const_expr(N1 > 0):
        src = sel_i64(Int32(gr >= Int32(N0)), w1 + Int64(gr - Int32(N0)) * k2, src)
    return src


@cute.jit
def _store_out(
    out0: Int64,
    out1: Int64,
    gr: Int32,
    tok: Int32,
    value: Float32,
    N0: cutlass.Constexpr[int],
    N1: cutlass.Constexpr[int],
):
    """BF16 store of ``value`` into ``out0[tok, gr]`` or ``out1[tok, gr - N0]``."""
    if gr < Int32(N0):
        st_global_b16(out0 + Int64(tok * Int32(N0) + gr) * Int64(2), bf16_bits(value))
    else:
        if cutlass.const_expr(N1 > 0):
            st_global_b16(
                out1 + Int64(tok * Int32(N1) + gr - Int32(N0)) * Int64(2), bf16_bits(value)
            )


@cute.jit
def _stamp(
    enabled: cutlass.Constexpr[bool],
    ts: Int64,
    bid: Int32,
    tid: Int32,
    k: cutlass.Constexpr[int],
    lead: cutlass.Constexpr[int] = 0,
):
    """Tests only: ``%globaltimer`` of CTA ``bid`` at point ``k`` (thread ``lead``)."""
    if cutlass.const_expr(enabled):
        if tid == Int32(lead):
            st_global_b64(ts + Int64((bid * Int32(8) + Int32(k)) * Int32(8)), globaltimer())


class FusedKvProjKernel:
    """Compile-time configuration of the merged projection (module docstring)."""

    def __init__(
        self,
        k: int,
        n0: int,
        n1: int,
        ctas: int,
        warps: int = 4,
        smem_rows: int = 18,
        nsplit: int = 4,
        nsub: int = 4,
        use_pdl: bool = True,
        trigger: str = "start",
        stage_before_wait: bool = True,
        l2_prefetch: bool = True,
        b_dist: int = 16,
        a_dist: int = 2,
        reduce: str = "seq",
        ftz: bool = False,
        producer_warp: bool = True,
        copy_method: str = "bulk",
        producer_warps: int = 1,
        copy_only: bool = False,
        debug_ts: bool = False,
        chunks: tuple = (),
        b_load: str = "cg",
        diag_no_act: bool = False,
        tma_groups: int = 1,
        prewait_chunks: int = -1,
    ):
        assert k % 16 == 0
        self.k = k
        self.n0 = n0
        self.n1 = n1
        self.ctas = ctas
        self.warps = warps
        # A dedicated producer warp issues the bulk copies (the issue blocks
        # while the copy engine is busy); otherwise MMA warp 0 does.
        # ``copy_method``: "bulk" (1-D cp.async.bulk, one lane per row of the
        # copy warp), "cpasync" (16-byte cp.async spread over
        # ``producer_warps`` dedicated warps, one mbarrier arrival per thread
        # and sub-chunk) or "tma" (3-D tensor copies of 16 rows x
        # ``tma_groups`` K groups each, 128-byte swizzle; see ``tma``).
        assert copy_method in ("bulk", "cpasync", "tma")
        self.copy_method = copy_method
        if copy_method == "cpasync":
            assert producer_warp and producer_warps >= 1
        else:
            producer_warps = 1 if producer_warp else 0
        self.producer_warp = producer_warp
        self.producer_warps = producer_warps
        self.copy_warp = warps if producer_warp else 0
        self.nt = 32 * (warps + producer_warps)
        # Tests only: stage the weights and exit (streaming floor).
        self.copy_only = copy_only
        steps = k // 16
        assert nsplit % warps == 0 and steps % nsplit == 0, (nsplit, warps)
        self.nsplit = nsplit
        self.split_steps = steps // nsplit
        self.warp_steps = steps // warps
        self.warp_k = k // warps
        # Staging chunks of each warp slice, in K steps (``chunks`` overrides
        # ``nsub`` equal chunks); one mbarrier per (chunk, warp).
        if chunks:
            chunks = tuple(int(c) for c in chunks)
        else:
            assert self.warp_steps % nsub == 0, (self.warp_steps, nsub)
            chunks = (self.warp_steps // nsub,) * nsub
        assert all(c > 0 for c in chunks) and sum(chunks) == self.warp_steps, chunks
        self.tma = copy_method == "tma"
        self.tma_groups = tma_groups
        if self.tma:
            assert self.warp_k % TMA_GROUP_K == 0
            assert tma_groups in (1, 3), "odd group counts keep ldmatrix conflict-free"
            assert all(c % (4 * tma_groups) == 0 for c in chunks), (chunks, tma_groups)
        self.chunks = chunks
        self.nsub = len(chunks)
        self.chunk_start = [sum(chunks[:i]) for i in range(len(chunks))]
        self.chunk_at = {st: i for i, st in enumerate(self.chunk_start)}
        self.nmbar = warps * self.nsub
        # Activation fragment loads: "cg" (L2 only), "ca" (L1 and L2), "nc"
        # (read-only path; benchmarks only). ``diag_no_act``: benchmarks only,
        # no activation loads at all (the output is meaningless).
        assert b_load in ("cg", "ca", "nc")
        self.b_load = b_load
        self.diag_no_act = diag_no_act
        assert 64 + 8 * self.nmbar <= _MISC_BYTES
        n = n0 + n1
        self.stride = k * 2 + 16
        if self.tma:
            # ``tma``: CTAs [0, c0) own W0 rows, [c0, ctas) own W1 rows; each
            # CTA stages its first 16 rows (m16 tile 0) by tensor copies into
            # one 2 KiB block per K group ([16 rows][128 B], 128-byte
            # swizzle), and the lanes of tile 1 read rows 16.. directly.
            split = split_ctas(n0, n1, ctas)
            assert split is not None, (n0, n1, ctas)
            self.c0, self.c1 = split
            groups = [(n0, self.c0)] + ([(n1, self.c1)] if n1 else [])
            self.max_rows = max(-(-nn // cc) for nn, cc in groups)
            self.smem_rows = TMA_ROWS
            self.direct = self.max_rows - TMA_ROWS
            self.wgroups = self.warp_k // TMA_GROUP_K
            self.w_bytes = warps * self.wgroups * TMA_GROUP_BYTES + 1024
        else:
            self.base_rows = n // ctas
            self.rem_rows = n % ctas
            self.max_rows = self.base_rows + (1 if self.rem_rows else 0)
            assert 1 <= self.base_rows and self.max_rows <= TILE_ROWS
            self.smem_rows = min(smem_rows, self.max_rows)
            self.direct = self.max_rows - self.smem_rows
            assert self.direct in (0, 1), "at most one directly-read row per CTA"
            if self.direct:
                # The direct row sits in m16 tile 1 (rows 16-23).
                assert 16 <= self.smem_rows < TILE_ROWS
            assert self.smem_rows <= 32, "the copy warp issues one row per lane"
            self.w_bytes = self.smem_rows * self.stride
        self.use_pdl = use_pdl
        assert trigger in ("start", "end", "none")
        self.trigger = trigger
        self.stage_before_wait = stage_before_wait
        # Chunks staged before the dependency wait (the rest after it):
        # all (-1) or the first ``prewait_chunks``; none without
        # ``stage_before_wait``.
        if not stage_before_wait:
            self.pre_chunks = 0
        elif prewait_chunks < 0:
            self.pre_chunks = self.nsub
        else:
            self.pre_chunks = min(prewait_chunks, self.nsub)
        self.l2_prefetch = l2_prefetch
        self.b_dist = max(1, b_dist)
        self.a_dist = max(1, a_dist)
        assert reduce in ("seq", "tree")
        self.reduce = reduce
        self.ftz = ftz
        self.debug_ts = debug_ts
        self.red_bytes = nsplit * TILE_ROWS * MAX_TOKENS * 4
        self.smem_bytes = self.w_bytes + _MISC_BYTES + self.red_bytes
        assert self.smem_bytes <= SMEM_LIMIT, f"{self.smem_bytes} B of shared memory"

    @cute.jit
    def __call__(
        self,
        act: Int64,
        w0: Int64,
        w1: Int64,
        out0: Int64,
        out1: Int64,
        ts: Int64,
        tmaps: Int64,
        anchor: cute.Tensor,
        rows: Int32,
        stream: CUstream,
    ):
        self.kernel(act, w0, w1, out0, out1, ts, tmaps, anchor, rows).launch(
            grid=[self.ctas, 1, 1], block=[self.nt, 1, 1], stream=stream, use_pdl=self.use_pdl
        )

    @cute.jit
    def _issue_copies(
        self,
        sW: Int32,
        sMbar: Int32,
        w0: Int64,
        w1: Int64,
        row0: Int32,
        nsm: Int32,
        warp: Int32,
        lane: Int32,
        tid: Int32,
        tmap: Int64,
        lrow0: Int32,
        nrows: Int32,
        part: cutlass.Constexpr[int],
    ):
        """Stage the CTA's rows sub-chunk by sub-chunk in arrival order.

        ``bulk``: the copy warp's lane 0 posts every sub-chunk's byte count,
        then lane ``r < nsm`` copies row ``r``. ``cpasync``: producer thread
        ``p`` copies 16-byte pieces ``p, p + P, ...`` of each sub-chunk and
        then arrives on the sub-chunk's mbarrier when its copies have landed.
        ``tma``: the copy warp's lane 0 posts every chunk's byte count and
        issues the chunk's tensor copies (box ``(64, tma_groups, 16)`` at K
        group ``gq``, local row ``lrow0``) into the group blocks; lanes 1..
        prefetch the directly-read rows into L2.
        ``part`` 0 issues the chunks staged before the dependency wait, 1 the
        others.
        """
        s_lo = self.pre_chunks * part
        s_hi = self.pre_chunks + (self.nsub - self.pre_chunks) * part
        first = (1 - part) + int(self.pre_chunks == 0) > 0
        if cutlass.const_expr(self.tma):
            if warp == Int32(self.copy_warp):
                if lane == Int32(0):
                    if cutlass.const_expr(first):
                        fence_tensormap_acquire(tmap)
                    for ci in cutlass.range_constexpr((s_hi - s_lo) * self.warps):
                        c = s_lo * self.warps + ci
                        mbar_arrive_expect_tx(
                            sMbar + Int32(8 * c),
                            Int32(self.chunks[c // self.warps] // 4 * TMA_GROUP_BYTES),
                        )
                    for si in cutlass.range_constexpr(s_hi - s_lo):
                        s = s_lo + si
                        for w in cutlass.range_constexpr(self.warps):
                            g0 = w * self.wgroups + self.chunk_start[s] // 4
                            for o in cutlass.range_constexpr(
                                self.chunks[s] // (4 * self.tma_groups)
                            ):
                                gq = g0 + o * self.tma_groups
                                tensor_g2s_3d(
                                    sW + Int32(gq * TMA_GROUP_BYTES),
                                    tmap,
                                    Int32(0),
                                    Int32(gq),
                                    lrow0,
                                    sMbar + Int32(8 * (s * self.warps + w)),
                                )
                if cutlass.const_expr(self.direct > 0 and self.l2_prefetch and first):
                    if lane >= Int32(1):
                        if lane <= nrows - Int32(TMA_ROWS):
                            prefetch_l2(
                                _row_src(
                                    w0,
                                    w1,
                                    row0 + Int32(TMA_ROWS - 1) + lane,
                                    self.k,
                                    self.n0,
                                    self.n1,
                                ),
                                Int32(self.k * 2),
                            )
        elif cutlass.const_expr(self.copy_method == "bulk"):
            if warp == Int32(self.copy_warp):
                if lane == Int32(0):
                    for ci in cutlass.range_constexpr((s_hi - s_lo) * self.warps):
                        c = s_lo * self.warps + ci
                        mbar_arrive_expect_tx(
                            sMbar + Int32(8 * c), nsm * Int32(32 * self.chunks[c // self.warps])
                        )
                sync_warp()
                if lane < nsm:
                    src = _row_src(w0, w1, row0 + lane, self.k, self.n0, self.n1)
                    dst = sW + lane * Int32(self.stride)
                    for si in cutlass.range_constexpr(s_hi - s_lo):
                        s = s_lo + si
                        for w in cutlass.range_constexpr(self.warps):
                            off = (w * self.warp_k + self.chunk_start[s] * 16) * 2
                            bulk_g2s(
                                dst + Int32(off),
                                src + Int64(off),
                                Int32(32 * self.chunks[s]),
                                sMbar + Int32(8 * (s * self.warps + w)),
                            )
        else:
            nprod = 32 * self.producer_warps
            if warp >= Int32(self.copy_warp):
                p = tid - Int32(32 * self.copy_warp)
                for si in cutlass.range_constexpr(s_hi - s_lo):
                    s = s_lo + si
                    per_row = self.chunks[s] * 2
                    total = nsm * Int32(per_row)
                    iters = (self.smem_rows * per_row + nprod - 1) // nprod
                    for w in cutlass.range_constexpr(self.warps):
                        base_off = (w * self.warp_k + self.chunk_start[s] * 16) * 2
                        for it in cutlass.range_constexpr(iters):
                            piece = p + Int32(nprod * it)
                            if piece < total:
                                r = piece // Int32(per_row)
                                off = Int32(base_off) + (piece % Int32(per_row)) * Int32(16)
                                cp_async16(
                                    sW + r * Int32(self.stride) + off,
                                    _row_src(w0, w1, row0 + r, self.k, self.n0, self.n1)
                                    + Int64(off),
                                )
                        cp_async_mbar_arrive(sMbar + Int32(8 * (s * self.warps + w)))

    @cute.jit
    def _mma(
        self,
        act: Int64,
        w0: Int64,
        w1: Int64,
        ts: Int64,
        sW: Int32,
        sMbar: Int32,
        sZero: Int32,
        sRed: Int32,
        bid: Int32,
        tid: Int32,
        warp: Int32,
        lane: Int32,
        rows: Int32,
        row0: Int32,
        nrows: Int32,
        nsm: Int32,
    ):
        """MMA warp ``warp``: the chains of its K range for both m16 tiles,
        each finished slice partial stored to ``sRed``."""
        k = self.k
        stride = self.stride
        # Fragment coordinates: token / row group g = lane / 4, pair t = lane % 4.
        g = lane // Int32(4)
        t4 = lane % Int32(4)
        tok_ok = Int32(g < rows)
        safe_g = min_i32(g, rows - Int32(1))
        kw0 = warp * Int32(self.warp_k)
        # B fragment of K step j: a[g, k0 + 2 t4 .. +1] and a[g, k0 + 8 + 2 t4 .. +1].
        b_lane = act + Int64((safe_g * Int32(k) + kw0 + t4 * Int32(2)) * Int32(2))
        # ldmatrix row addresses (per lane), advanced by 32 B per K step.
        r16 = lane % Int32(16)
        a_lane0 = sW + r16 * Int32(stride) + (lane // Int32(16)) * Int32(16) + kw0 * Int32(2)
        ok0 = Int32(r16 < nsm)
        r8 = Int32(16) + lane % Int32(8)
        a_lane1 = (
            sW + r8 * Int32(stride) + ((lane // Int32(8)) % Int32(2)) * Int32(16) + kw0 * Int32(2)
        )
        ok1 = Int32(r8 < nsm)
        hl = lane // Int32(16)
        t_off = []
        t_x = []
        t_base = sW
        if cutlass.const_expr(self.tma):
            # Tile 0 from the swizzled group blocks: in a box of G groups the
            # 128-byte line of (row r, group g) is u = r * G + g, and its
            # 16-byte chunk c sits at chunk c ^ (u % 8).
            t_rows = [r16 * Int32(self.tma_groups) + Int32(q) for q in range(self.tma_groups)]
            t_off = [u * Int32(128) for u in t_rows]
            t_x = [u % Int32(8) for u in t_rows]
            t_base = sW + warp * Int32(self.wgroups * TMA_GROUP_BYTES)
        dir_lane = Int32(0)
        d_lane = Int64(0)
        if cutlass.const_expr(self.direct > 0):
            if cutlass.const_expr(self.tma):
                # Lanes of fragment row group g of tile 1 hold CTA row 16 + g.
                dir_lane = Int32(g < nrows - Int32(TMA_ROWS))
                d_lane = _row_src(w0, w1, row0 + Int32(TMA_ROWS) + g, k, self.n0, self.n1) + Int64(
                    (kw0 + t4 * Int32(2)) * Int32(2)
                )
            else:
                # Lanes of fragment row group smem_rows - 16 of tile 1 hold the direct row.
                dir_lane = Int32(g == Int32(self.smem_rows - 16)) * Int32(nrows > nsm)
                d_lane = _row_src(
                    w0, w1, row0 + Int32(self.smem_rows), k, self.n0, self.n1
                ) + Int64((kw0 + t4 * Int32(2)) * Int32(2))

        zero = Int32(0)
        nsteps = self.warp_steps

        ld_b = {"cg": ld_cg_b32, "ca": ld_ca_b32, "nc": ld_nc_b32}[self.b_load]

        def load_b(j):
            off = Int64(32 * j)
            if cutlass.const_expr(self.diag_no_act):
                b0 = lane + Int32(j)
                b1 = lane - Int32(j)
            else:
                b0 = ld_b(b_lane + off)
                b1 = ld_b(b_lane + off + Int64(16))
            return sel_i32(tok_ok, b0, zero), sel_i32(tok_ok, b1, zero)

        ld_d = ld_ca_b32_if if self.b_load == "ca" else ld_cg_b32_if

        def load_d(j):
            res = (zero, zero)
            if cutlass.const_expr(self.direct > 0):
                off = Int64(32 * j)
                res = (ld_d(d_lane + off, dir_lane), ld_d(d_lane + off + Int64(16), dir_lane))
            return res

        def load_a(j):
            if cutlass.const_expr(j in self.chunk_at):
                mbar_wait(
                    sMbar + Int32(8 * self.chunk_at[j] * self.warps) + warp * Int32(8), Int32(0)
                )
                if cutlass.const_expr(self.chunk_at[j] == self.nsub - 1):
                    _stamp(self.debug_ts, ts, bid, tid, 7, 32 * (self.warps - 1))
            if cutlass.const_expr(self.tma):
                gk = j // 4
                q = gk % self.tma_groups
                ch = Int32(2 * (j % 4)) + hl
                addr = (
                    t_base
                    + Int32((gk - q) * TMA_GROUP_BYTES)
                    + t_off[q]
                    + xor_i32(ch, t_x[q]) * Int32(16)
                )
                return ldsm_x4(addr), (zero, zero)
            kb = Int32(32 * j)
            return (
                ldsm_x4(sel_i32(ok0, a_lane0 + kb, sZero)),
                ldsm_x2(sel_i32(ok1, a_lane1 + kb, sZero)),
            )

        c0 = Int32(2) * t4
        splits_per_warp = self.nsplit // self.warps

        def stash(acc0, acc1, i):
            """Slice partials of this warp's local slice ``i`` -> sRed."""
            sl = warp * Int32(splits_per_warp) + Int32(i)
            base_s = sRed + sl * Int32(TILE_ROWS * MAX_TOKENS * 4)
            st_shared_f32(base_s + (g * Int32(MAX_TOKENS) + c0) * Int32(4), acc0[0])
            st_shared_f32(base_s + (g * Int32(MAX_TOKENS) + c0 + Int32(1)) * Int32(4), acc0[1])
            st_shared_f32(base_s + ((g + Int32(8)) * Int32(MAX_TOKENS) + c0) * Int32(4), acc0[2])
            st_shared_f32(
                base_s + ((g + Int32(8)) * Int32(MAX_TOKENS) + c0 + Int32(1)) * Int32(4), acc0[3]
            )
            st_shared_f32(base_s + ((g + Int32(16)) * Int32(MAX_TOKENS) + c0) * Int32(4), acc1[0])
            st_shared_f32(
                base_s + ((g + Int32(16)) * Int32(MAX_TOKENS) + c0 + Int32(1)) * Int32(4), acc1[1]
            )

        bq = {}
        dq = {}
        aq = {}
        for j in cutlass.range_constexpr(min(self.b_dist, nsteps)):
            bq[j] = load_b(j)
            dq[j] = load_d(j)
        for j in cutlass.range_constexpr(min(self.a_dist, nsteps)):
            aq[j] = load_a(j)
        acc0 = [Float32(0.0)] * 4
        acc1 = [Float32(0.0)] * 4
        for j in cutlass.range_constexpr(nsteps):
            if cutlass.const_expr(j + self.b_dist < nsteps):
                bq[j + self.b_dist] = load_b(j + self.b_dist)
                dq[j + self.b_dist] = load_d(j + self.b_dist)
            if cutlass.const_expr(j + self.a_dist < nsteps):
                aq[j + self.a_dist] = load_a(j + self.a_dist)
            if cutlass.const_expr(j > 0 and j % self.split_steps == 0):
                stash(acc0, acc1, j // self.split_steps - 1)
                acc0 = [Float32(0.0)] * 4
                acc1 = [Float32(0.0)] * 4
            fa, fb = aq.pop(j)
            b0, b1 = bq.pop(j)
            fb0, fb1 = fb
            d0, d1 = dq.pop(j)
            if cutlass.const_expr(self.direct > 0):
                fb0 = sel_i32(dir_lane, d0, fb0)
                fb1 = sel_i32(dir_lane, d1, fb1)
            acc0[0], acc0[1], acc0[2], acc0[3] = mma_bf16(
                fa[0], fa[1], fa[2], fa[3], b0, b1, acc0[0], acc0[1], acc0[2], acc0[3]
            )
            acc1[0], acc1[1], acc1[2], acc1[3] = mma_bf16(
                fb0, zero, fb1, zero, b0, b1, acc1[0], acc1[1], acc1[2], acc1[3]
            )
        stash(acc0, acc1, splits_per_warp - 1)
        _stamp(self.debug_ts, ts, bid, tid, 6)
        _stamp(self.debug_ts, ts, bid, tid, 3, 32 * (self.warps - 1))

    @cute.kernel
    def kernel(
        self,
        act: Int64,
        w0: Int64,
        w1: Int64,
        out0: Int64,
        out1: Int64,
        ts: Int64,
        tmaps: Int64,
        anchor: cute.Tensor,
        rows: Int32,
    ):
        """``debug_ts`` stamps per CTA (int64 ``[ctas, 8]``): 0 entry, 1 copies
        issued (thread 0 passed the issue: copy warp 0 only), 2 after the
        dependency wait, 7 last chunk of the last MMA warp landed, 3 main loop
        done (last MMA warp), 6 main loop done (MMA warp 0), 4 rows stored,
        5 exit."""
        k = self.k
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        _stamp(self.debug_ts, ts, bid, tid, 0)

        smem = SmemAllocator()
        sW = saddr(smem.allocate(self.w_bytes, byte_alignment=128))
        if cutlass.const_expr(self.tma):
            # Swizzled tensor-copy destinations start on 1024-byte boundaries.
            sW = (sW + Int32(1023)) // Int32(1024) * Int32(1024)
        sMisc = saddr(smem.allocate(_MISC_BYTES, byte_alignment=128))
        sRed = saddr(smem.allocate(self.red_bytes, byte_alignment=128))
        # sMisc: [0, 32) zero chunk, [64, 64 + 8 * nmbar) mbarriers.
        sZero = sMisc
        sMbar = sMisc + Int32(64)

        # Rows of this CTA: [row0, row0 + nrows) of the concatenation [W0; W1]
        # (local row lrow0 of tensor map tmap for ``tma``).
        tmap = tmaps
        if cutlass.const_expr(self.tma):
            in1 = Int32(0)
            if cutlass.const_expr(self.n1 > 0):
                in1 = Int32(bid >= Int32(self.c0))
            b = bid - in1 * Int32(self.c0)
            base = sel_i32(in1, Int32(self.n1 // max(self.c1, 1)), Int32(self.n0 // self.c0))
            rem = sel_i32(in1, Int32(self.n1 % max(self.c1, 1)), Int32(self.n0 % self.c0))
            nrows = base + Int32(b < rem)
            lrow0 = b * base + min_i32(b, rem)
            row0 = lrow0 + in1 * Int32(self.n0)
            nsm = Int32(TMA_ROWS)
            tmap = tmaps + Int64(in1 * Int32(128))
        else:
            base = Int32(self.base_rows)
            rem = Int32(self.rem_rows)
            nrows = base + Int32(bid < rem)
            row0 = bid * base + min_i32(bid, rem)
            lrow0 = row0
            nsm = min_i32(nrows, Int32(self.smem_rows))

        if tid < Int32(2):
            st_shared_v4(sZero + tid * Int32(16), Int32(0), Int32(0), Int32(0), Int32(0))
        if tid == Int32(0):
            for c in cutlass.range_constexpr(self.nmbar):
                mbar_init(
                    sMbar + Int32(8 * c),
                    Int32(32 * self.producer_warps if self.copy_method == "cpasync" else 1),
                )
            mbar_fence_init()
        cute.arch.sync_threads()

        if cutlass.const_expr(self.pre_chunks > 0):
            self._issue_copies(sW, sMbar, w0, w1, row0, nsm, warp, lane, tid, tmap, lrow0, nrows, 0)
        if cutlass.const_expr(self.direct > 0 and self.l2_prefetch and not self.tma):
            if tid == Int32(0 if self.producer_warp else 32 % self.nt):
                if nrows > nsm:
                    prefetch_l2(_row_src(w0, w1, row0 + nsm, k, self.n0, self.n1), Int32(k * 2))
        _stamp(self.debug_ts, ts, bid, tid, 1, 32 * self.copy_warp)
        if cutlass.const_expr(self.use_pdl and self.trigger == "start"):
            cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        _stamp(self.debug_ts, ts, bid, tid, 2)
        if cutlass.const_expr(self.pre_chunks < self.nsub):
            self._issue_copies(sW, sMbar, w0, w1, row0, nsm, warp, lane, tid, tmap, lrow0, nrows, 1)

        if cutlass.const_expr(self.copy_only):
            if warp < Int32(self.warps):
                for sub in cutlass.range_constexpr(self.nsub):
                    mbar_wait(sMbar + Int32(8 * sub * self.warps) + warp * Int32(8), Int32(0))
        elif cutlass.const_expr(self.producer_warp):
            if warp < Int32(self.warps):
                self._mma(
                    act,
                    w0,
                    w1,
                    ts,
                    sW,
                    sMbar,
                    sZero,
                    sRed,
                    bid,
                    tid,
                    warp,
                    lane,
                    rows,
                    row0,
                    nrows,
                    nsm,
                )
        else:
            self._mma(
                act,
                w0,
                w1,
                ts,
                sW,
                sMbar,
                sZero,
                sRed,
                bid,
                tid,
                warp,
                lane,
                rows,
                row0,
                nrows,
                nsm,
            )

        cute.arch.sync_threads()
        if cutlass.const_expr(not self.copy_only):
            pairs = TILE_ROWS * MAX_TOKENS
            for i in cutlass.range_constexpr((pairs + self.nt - 1) // self.nt):
                idx = tid + Int32(self.nt * i)
                row = idx // Int32(MAX_TOKENS)
                tok = idx % Int32(MAX_TOKENS)
                if idx < Int32(pairs):
                    if row < nrows:
                        if tok < rows:
                            vals = [
                                ld_shared_f32(sRed + Int32(sl * pairs * 4) + idx * Int32(4))
                                for sl in range(self.nsplit)
                            ]
                            s = _ordered_sum(vals, self.reduce, self.ftz)
                            _store_out(
                                out0,
                                out1,
                                row0 + row,
                                tok,
                                add_f32(Float32(0.0), s, self.ftz),
                                self.n0,
                                self.n1,
                            )
        _stamp(self.debug_ts, ts, bid, tid, 4)
        if cutlass.const_expr(self.use_pdl and self.trigger == "end"):
            cute.arch.griddepcontrol_launch_dependents()
        _stamp(self.debug_ts, ts, bid, tid, 5)


_COMPILED = {}


def is_compiled(**cfg) -> bool:
    """Whether ``compiled_projection(**cfg)`` would be a cache hit."""
    return tuple(sorted(cfg.items())) in _COMPILED


def compiled_projection(**cfg):
    """Return the compiled projection for one configuration (compiles once, no GPU work)."""
    key = tuple(sorted(cfg.items()))
    compiled = _COMPILED.get(key)
    if compiled is None:
        from cutlass.cute import runtime as cute_runtime

        kernel = FusedKvProjKernel(**cfg)
        z64 = Int64(0)
        anchor = cute_runtime.make_fake_compact_tensor(
            Int32, (cute.sym_int(),), stride_order=(0,), assumed_align=4
        )
        compiled = cute.compile(
            kernel,
            z64,
            z64,
            z64,
            z64,
            z64,
            z64,
            z64,
            anchor,
            Int32(1),
            stream=cute_runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _COMPILED[key] = compiled
    return compiled


def projection_supported(
    k: int,
    n: int,
    ctas: int,
    warps: int,
    smem_rows: int,
    nsplit: int,
    nsub: int,
    chunks: tuple = (),
    n1: int = 0,
    copy_method: str = "bulk",
    tma_groups: int = 1,
) -> Optional[str]:
    """Return why the projection kernel cannot serve a shape, or ``None``."""
    steps = k // 16
    if k % 16 or nsplit % warps or steps % nsplit:
        return f"K={k} does not split into {nsplit} slices over {warps} warps"
    if chunks:
        if any(c <= 0 for c in chunks) or sum(chunks) != steps // warps:
            return f"chunks {chunks} do not cover the {steps // warps} K steps per warp"
    elif (steps // warps) % nsub:
        return f"{steps // warps} K steps per warp do not split into {nsub} sub-chunks"
    else:
        chunks = ((steps // warps) // nsub,) * nsub
    nsub = len(chunks)
    if 64 + 8 * warps * nsub > _MISC_BYTES:
        return f"{warps * nsub} mbarriers do not fit"
    if copy_method == "tma":
        if (k // warps) % TMA_GROUP_K:
            return f"K slice {k // warps} is not a multiple of {TMA_GROUP_K}"
        if tma_groups not in (1, 3) or any(c % (4 * tma_groups) for c in chunks):
            return f"chunks {chunks} are not whole boxes of {tma_groups} K groups"
        if split_ctas(n - n1, n1, ctas) is None:
            return f"no CTA split of {n - n1} + {n1} rows gives {TMA_ROWS}..{TILE_ROWS} rows"
        smem = (
            (k // TMA_GROUP_K) * TMA_GROUP_BYTES
            + 1024
            + _MISC_BYTES
            + nsplit * TILE_ROWS * MAX_TOKENS * 4
        )
        if smem > SMEM_LIMIT:
            return f"{smem} B of shared memory exceed the limit"
        return None
    base, rem = divmod(n, ctas)
    max_rows = base + (1 if rem else 0)
    if base < 1 or max_rows > TILE_ROWS:
        return f"{n} rows over {ctas} CTAs gives {base}..{max_rows} rows per CTA (1..24 supported)"
    staged = min(smem_rows, max_rows)
    if max_rows - staged > 1:
        return f"{max_rows} rows per CTA exceed {smem_rows} staged rows by more than one"
    if max_rows > staged and not 16 <= staged < TILE_ROWS:
        return "a directly-read row must fall into rows 16..23 of the CTA"
    smem = staged * (k * 2 + 16) + _MISC_BYTES + nsplit * TILE_ROWS * MAX_TOKENS * 4
    if smem > SMEM_LIMIT:
        return f"{smem} B of shared memory exceed the limit"
    return None
