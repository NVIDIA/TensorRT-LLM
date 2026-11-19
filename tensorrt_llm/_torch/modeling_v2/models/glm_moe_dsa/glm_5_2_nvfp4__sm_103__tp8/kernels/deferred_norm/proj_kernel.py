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
"""CuTe DSL kernel of the deferred_norm next-layer projection with staged weights.

Computes, for ``R <= 8`` token rows,

    y[t, j] = bf16(r[t] * sum_k a[t, k] * W[j, k])

over the concatenation of up to two row-major BF16 weight tensors with the
same ``K`` (``W0 [N0, K]`` = the fused ``[q_a | kv_a | k_pe]`` projection,
``W1 [N1, K]`` = the indexer ``wk | weights_proj`` rows on full-indexer
layers), writing ``out0 [R, N0]`` and ``out1 [R, N1]``. In deferred mode
``a`` is the gain-folded activation ``bf16(x * gamma)`` of the deferred_norm hop and
``r[t] = rsqrt.approx(div.rn(sum_i sumsq[t, i], H) + eps)`` with the square
sum taken in the hop's partial order (bit-identical to the post_moe_hop's
``rsqrt`` operand); in native-input mode ``a`` is an ordinary normalized
hidden state and ``r = 1``.

Schedule (one CTA per SM, ``NT`` threads):

* CTA ``c`` owns a contiguous range of 17-19 output rows. Before
  ``griddepcontrol.wait`` (the weights do not depend on the preceding hop)
  one thread issues ``cp.async.bulk`` copies of up to ``SMEM_ROWS`` of its
  rows into shared memory (row stride padded by 16 bytes), one mbarrier per
  K chunk, and prefetches every row it cannot stage into L2;
* after the wait every thread loads its activation fragments (``B`` of
  ``mma.m16n8k16``, the token rows) from global memory, and warp ``w`` runs
  the K steps ``s = w + NW * m`` of two ``m16n8`` tiles (rows 0-15 via
  ``ldmatrix.x4``, rows 16-23 via ``ldmatrix.x2``; rows the CTA does not own
  read a zero chunk);
* the ``NW`` per-warp partial sums are reduced in warp order through shared
  memory, scaled by ``r`` and rounded once to BF16;
* a CTA that owns more rows than it can stage processes the extra row after
  the main loop from a bulk copy of the (L2-prefetched) row.
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils.smem_allocator import SmemAllocator

from .ptx import (
    add,
    bf16_bits,
    bits_f32,
    cp_async16,
    cp_async_commit,
    cp_async_wait_group,
    inv_rms,
    ldsm_x4,
    mma_bf16,
    mul,
    saddr,
    st_global_b16,
    st_shared_v4,
)

MAX_TOKENS = 8
SMEM_LIMIT = 227 * 1024


def _asm(res, args, text, cons, side_effects=True, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        args,
        text,
        cons,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bulk_g2s(saddr: Int32, gaddr: Int64, nbytes: Int32, mbar: Int32, *, loc=None, ip=None):
    _asm(
        None,
        [saddr.ir_value(), gaddr.ir_value(), nbytes.ir_value(), mbar.ir_value()],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _prefetch_l2(gaddr: Int64, nbytes: Int32, *, loc=None, ip=None):
    _asm(
        None,
        [gaddr.ir_value(), nbytes.ir_value()],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_init(mbar: Int32, count: Int32, *, loc=None, ip=None):
    _asm(
        None,
        [mbar.ir_value(), count.ir_value()],
        "mbarrier.init.shared::cta.b64 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_fence_init(*, loc=None, ip=None):
    _asm(
        None,
        [],
        "fence.mbarrier_init.release.cluster;\n\tfence.proxy.async.shared::cta;",
        "",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_expect_tx(mbar: Int32, nbytes: Int32, *, loc=None, ip=None):
    _asm(
        None,
        [mbar.ir_value(), nbytes.ir_value()],
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mbar_wait(mbar: Int32, phase: Int32, *, loc=None, ip=None):
    _asm(
        None,
        [mbar.ir_value(), phase.ir_value()],
        "{\n\t.reg .pred p;\n\tNORM_PROJ_MBAR_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1;\n\t"
        "@!p bra NORM_PROJ_MBAR_WAIT;\n\t}",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ldsm_x2(saddr: Int32, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [saddr.ir_value()],
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {$0, $1}, [$2];",
        "=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(2))


@dsl_user_op
def _ld_nc_b32(gaddr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(T.i32(), [gaddr.ir_value()], "ld.global.nc.b32 $0, [$1];", "=r,l", loc=loc, ip=ip)
    )


@dsl_user_op
def _st_shared_f32(saddr: Int32, v: Float32, *, loc=None, ip=None):
    _asm(None, [saddr.ir_value(), v.ir_value()], "st.shared.f32 [$0], $1;", "r,f", loc=loc, ip=ip)


@dsl_user_op
def _ld_shared_f32(saddr: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(T.f32(), [saddr.ir_value()], "ld.shared.f32 $0, [$1];", "=f,r", loc=loc, ip=ip)
    )


@dsl_user_op
def _sel_i32(pred: Int32, a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [pred.ir_value(), a.ir_value(), b.ir_value()],
            "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $1, 0;\n\tselp.b32 $0, $2, $3, p;\n\t}",
            "=r,r,r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sel_i64(pred: Int32, a: Int64, b: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(
        _asm(
            T.i64(),
            [pred.ir_value(), a.ir_value(), b.ir_value()],
            "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $1, 0;\n\tselp.b64 $0, $2, $3, p;\n\t}",
            "=l,r,l,l",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _min_i32(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [a.ir_value(), b.ir_value()],
            "min.s32 $0, $1, $2;",
            "=r,r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fence_proxy_async(*, loc=None, ip=None):
    _asm(None, [], "fence.proxy.async.shared::cta;", "", loc=loc, ip=ip)


@dsl_user_op
def _ld_b32(gaddr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(T.i32(), [gaddr.ir_value()], "ld.global.b32 $0, [$1];", "=r,l", loc=loc, ip=ip)
    )


@dsl_user_op
def _ld_v2(gaddr: Int64, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [gaddr.ir_value()],
        "ld.global.v2.b32 {$0, $1}, [$2];",
        "=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(2))


@dsl_user_op
def _ld_v2_if(gaddr: Int64, pred: Int32, *, loc=None, ip=None):
    """8-byte load if ``pred`` else zeros."""
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [gaddr.ir_value(), pred.ir_value()],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $3, 0;\n\tmov.b32 $0, 0;\n\t"
        "mov.b32 $1, 0;\n\t@p ld.global.v2.b32 {$0, $1}, [$2];\n\t}",
        "=r,=r,l,r",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(2))


@dsl_user_op
def _shfl_idx_f32(v: Float32, src: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [v.ir_value(), src.ir_value()],
            "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=f,f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _globaltimer(*, loc=None, ip=None) -> Int64:
    return Int64(_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", loc=loc, ip=ip))


@dsl_user_op
def _st_global_b64(gaddr: Int64, v: Int64, *, loc=None, ip=None):
    _asm(None, [gaddr.ir_value(), v.ir_value()], "st.global.b64 [$0], $1;", "l,l", loc=loc, ip=ip)


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
        src = _sel_i64(Int32(gr >= Int32(N0)), w1 + Int64(gr - Int32(N0)) * k2, src)
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
    enabled: cutlass.Constexpr[bool], ts: Int64, bid: Int32, tid: Int32, k: cutlass.Constexpr[int]
):
    """Tests only: ``%globaltimer`` of CTA ``bid`` at point ``k``."""
    if cutlass.const_expr(enabled):
        if tid == Int32(0):
            _st_global_b64(ts + Int64((bid * Int32(8) + Int32(k)) * Int32(8)), _globaltimer())


class DeferredNormProjKernel:
    """Compile-time configuration of the staged projection (see the module docstring)."""

    def __init__(
        self,
        k: int,
        n0: int,
        n1: int,
        ctas: int,
        threads: int = 512,
        smem_rows: int = 18,
        nchunk: int = 4,
        deferred: bool = True,
        partials_per_row: int = 24,
        use_pdl: bool = True,
        trigger: str = "start",
        stage_before_wait: bool = True,
        l2_prefetch: bool = True,
        dist: int = 0,
        copy_threads: int = 1,
        copy_method: str = "bulk",
        wait_staged_first: bool = False,
        b_layout: str = "frag",
        prefetch_next: bool = False,
        debug_ts: bool = False,
    ):
        assert k % 16 == 0
        self.k = k
        self.n0 = n0
        self.n1 = n1
        self.ctas = ctas
        self.nt = threads
        self.nw = threads // 32
        self.steps = k // 16
        assert self.steps % self.nw == 0, "K steps must split evenly over the warps"
        self.rounds = self.steps // self.nw
        n = n0 + n1
        self.base_rows = n // ctas
        self.rem_rows = n % ctas
        self.max_rows = self.base_rows + (1 if self.rem_rows else 0)
        assert 16 < self.base_rows and self.max_rows <= 24, "17..24 rows per CTA supported"
        self.smem_rows = min(smem_rows, self.max_rows)
        self.overflow = self.max_rows - self.smem_rows
        assert self.overflow in (0, 1), "at most one row per CTA beyond the staged rows"
        assert self.smem_rows >= 17
        self.stride = k * 2 + 16
        self.nchunk = nchunk
        assert k % nchunk == 0 and (k // nchunk) % (16 * self.nw) == 0
        self.chunk_k = k // nchunk
        self.rounds_per_chunk = self.chunk_k // (16 * self.nw)
        self.deferred = deferred
        self.partials_per_row = partials_per_row
        # Each lane of the r warp loads a quarter of its token's partials.
        assert partials_per_row % 8 == 0
        self.use_pdl = use_pdl
        assert trigger in ("start", "end", "none")
        self.trigger = trigger
        self.stage_before_wait = stage_before_wait
        self.l2_prefetch = l2_prefetch
        self.debug_ts = debug_ts
        self.dist = dist
        self.copy_threads = copy_threads
        assert copy_method in ("bulk", "cpasync")
        self.copy_method = copy_method
        # "frag": the activation is in the MMA B-fragment layout of
        # packet.py (one 8-byte load per lane and K step); "plain": a
        # row-major [R, K] tensor (two 4-byte loads per lane and K step).
        assert b_layout in ("frag", "plain")
        self.b_layout = b_layout
        self.prefetch_next = prefetch_next
        # cp.async staging completes per thread: every thread waits for it
        # (then a CTA barrier) before the first MMA, so it is always "first".
        self.wait_staged_first = wait_staged_first or copy_method == "cpasync"
        self.red_bytes = self.nw * 24 * MAX_TOKENS * 4
        assert self.red_bytes <= 2 * self.stride
        self.smem_bytes = self.smem_rows * self.stride + 1024
        assert self.smem_bytes <= SMEM_LIMIT, f"{self.smem_bytes} B of shared memory"

    @cute.jit
    def __call__(
        self,
        act: Int64,
        sumsq: Int64,
        w0: Int64,
        w1: Int64,
        out0: Int64,
        out1: Int64,
        ts: Int64,
        pf0: Int64,
        pf0_bytes: Int64,
        pf1: Int64,
        pf1_bytes: Int64,
        pf2: Int64,
        pf2_bytes: Int64,
        anchor: cute.Tensor,
        rows: Int32,
        eps: Float32,
        stream: CUstream,
    ):
        self.kernel(
            act,
            sumsq,
            w0,
            w1,
            out0,
            out1,
            ts,
            pf0,
            pf0_bytes,
            pf1,
            pf1_bytes,
            pf2,
            pf2_bytes,
            anchor,
            rows,
            eps,
        ).launch(grid=[self.ctas, 1, 1], block=[self.nt, 1, 1], stream=stream, use_pdl=self.use_pdl)

    @cute.jit
    def _issue_copies(
        self, sW: Int32, sMbar: Int32, w0: Int64, w1: Int64, row0: Int32, nsm: Int32, tid: Int32
    ):
        """Bulk copies of the staged rows, one mbarrier per K chunk. With
        ``copy_threads > 1`` thread 0 posts every chunk's byte count first and
        thread ``i < nsm`` then copies row ``i`` (transaction counts may
        complete in any order)."""
        chunk_bytes = Int32(self.chunk_k * 2)
        if cutlass.const_expr(self.copy_method == "cpasync"):
            pieces = self.k * 2 // 16
            for i in cutlass.range_constexpr((self.smem_rows * pieces + self.nt - 1) // self.nt):
                c = tid + Int32(self.nt * i)
                r = c // Int32(pieces)
                off = c % Int32(pieces)
                if r < nsm:
                    cp_async16(
                        sW + r * Int32(self.stride) + off * Int32(16),
                        _row_src(w0, w1, row0 + r, self.k, self.n0, self.n1)
                        + Int64(off * Int32(16)),
                    )
            cp_async_commit()
        elif cutlass.const_expr(self.copy_threads == 1):
            if tid == Int32(0):
                for j in cutlass.range_constexpr(self.nchunk):
                    _mbar_expect_tx(sMbar + Int32(8 * j), nsm * chunk_bytes)
                    for i in cutlass.range(nsm):
                        _bulk_g2s(
                            sW + i * Int32(self.stride) + Int32(j * self.chunk_k * 2),
                            _row_src(w0, w1, row0 + i, self.k, self.n0, self.n1)
                            + Int64(j * self.chunk_k * 2),
                            chunk_bytes,
                            sMbar + Int32(8 * j),
                        )
        else:
            if tid == Int32(0):
                for j in cutlass.range_constexpr(self.nchunk):
                    _mbar_expect_tx(sMbar + Int32(8 * j), nsm * chunk_bytes)
            if tid < nsm:
                src = _row_src(w0, w1, row0 + tid, self.k, self.n0, self.n1)
                for j in cutlass.range_constexpr(self.nchunk):
                    _bulk_g2s(
                        sW + tid * Int32(self.stride) + Int32(j * self.chunk_k * 2),
                        src + Int64(j * self.chunk_k * 2),
                        chunk_bytes,
                        sMbar + Int32(8 * j),
                    )

    @cute.jit
    def _prefetch_slice(self, base: Int64, nbytes: Int64, bid: Int32):
        """``cp.async.bulk.prefetch.L2`` of slice ``bid`` of ``[base, base +
        nbytes)`` in 16 KB pieces (slices are 16-byte multiples)."""
        slice_bytes = ((nbytes + Int64(self.ctas * 16 - 1)) // Int64(self.ctas * 16)) * Int64(16)
        start = Int64(bid) * slice_bytes
        end = start + slice_bytes
        if end > nbytes:
            end = nbytes
        piece = Int64(16384)
        off = start
        for _ in cutlass.range_constexpr(4):
            if off < end:
                n = end - off
                if n > piece:
                    n = piece
                _prefetch_l2(base + off, Int32(n))
                off = off + n

    @cute.jit
    def _wait_staged(self, sMbar: Int32):
        """Every thread waits until all staged rows are in shared memory."""
        if cutlass.const_expr(self.copy_method == "cpasync"):
            cp_async_wait_group(0)
            cute.arch.sync_threads()
        else:
            for j in cutlass.range_constexpr(self.nchunk):
                _mbar_wait(sMbar + Int32(8 * j), Int32(0))

    @cute.kernel
    def kernel(
        self,
        act: Int64,
        sumsq: Int64,
        w0: Int64,
        w1: Int64,
        out0: Int64,
        out1: Int64,
        ts: Int64,
        pf0: Int64,
        pf0_bytes: Int64,
        pf1: Int64,
        pf1_bytes: Int64,
        pf2: Int64,
        pf2_bytes: Int64,
        anchor: cute.Tensor,
        rows: Int32,
        eps: Float32,
    ):
        """``debug_ts`` stamps per CTA (int64 ``[ctas, 8]``): 0 entry, 7 copies
        issued (before the wait when staging early), 6 every staged row
        landed (only with ``wait_staged_first``: all threads wait for it
        before the dependency wait), 1 after the dependency wait, 2
        activation fragments loaded, 3 main loop done, 4 rows stored, 5
        exit."""
        k = self.k
        nw = self.nw
        stride = self.stride
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        _stamp(self.debug_ts, ts, bid, tid, 0)

        smem = SmemAllocator()
        sW = saddr(smem.allocate(self.smem_rows * stride, byte_alignment=128))
        sMisc = saddr(smem.allocate(512, byte_alignment=128))
        # sMisc: [0, 32) zero chunk, [64, 64 + 8 * (nchunk + 1)) mbarriers,
        # [256, 288) r per token.
        sZero = sMisc
        sMbar = sMisc + Int32(64)
        sR = sMisc + Int32(256)

        # Rows of this CTA.
        base = Int32(self.base_rows)
        rem = Int32(self.rem_rows)
        extra = Int32(bid < rem)
        nrows = base + extra
        row0 = bid * base + _min_i32(bid, rem)
        nsm = _min_i32(nrows, Int32(self.smem_rows))

        if tid < Int32(2):
            st_shared_v4(sZero + tid * Int32(16), Int32(0), Int32(0), Int32(0), Int32(0))
        if tid == Int32(0):
            for j in cutlass.range_constexpr(self.nchunk + 1):
                _mbar_init(sMbar + Int32(8 * j), Int32(1))
            _mbar_fence_init()
        cute.arch.sync_threads()

        if cutlass.const_expr(self.stage_before_wait):
            self._issue_copies(sW, sMbar, w0, w1, row0, nsm, tid)
        _stamp(self.debug_ts, ts, bid, tid, 7)
        if cutlass.const_expr(self.prefetch_next):
            # L2 prefetch of the next kernels' weights (a hint; e.g. q_b and
            # W_UK of the same layer): CTA b covers slice b of each tensor.
            if tid < Int32(3):
                pf_addr = _sel_i64(
                    Int32(tid == Int32(0)), pf0, _sel_i64(Int32(tid == Int32(1)), pf1, pf2)
                )
                pf_bytes = _sel_i64(
                    Int32(tid == Int32(0)),
                    pf0_bytes,
                    _sel_i64(Int32(tid == Int32(1)), pf1_bytes, pf2_bytes),
                )
                self._prefetch_slice(pf_addr, pf_bytes, bid)
        if cutlass.const_expr(self.l2_prefetch):
            # Rows that are not staged into shared memory before the wait.
            first = Int32(0)
            if cutlass.const_expr(self.stage_before_wait):
                first = nsm
            if tid >= first:
                if tid < nrows:
                    _prefetch_l2(_row_src(w0, w1, row0 + tid, k, self.n0, self.n1), Int32(k * 2))
        if cutlass.const_expr(self.use_pdl and self.trigger == "start"):
            cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.stage_before_wait and self.wait_staged_first):
            # Every thread waits until the staged rows have landed before its
            # dependency wait (they do not depend on the preceding grid).
            self._wait_staged(sMbar)
            _stamp(self.debug_ts, ts, bid, tid, 6)
        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        _stamp(self.debug_ts, ts, bid, tid, 1)

        if cutlass.const_expr(not self.stage_before_wait):
            self._issue_copies(sW, sMbar, w0, w1, row0, nsm, tid)
            if cutlass.const_expr(self.copy_method == "cpasync"):
                self._wait_staged(sMbar)
        # Activation fragments: token g = lane / 4, K pair 2 * (lane % 4).
        g = lane // Int32(4)
        t4 = lane % Int32(4)
        tok_ok = Int32(g < rows)
        safe_g = _min_i32(g, rows - Int32(1))
        # Square-sum partials for r: lane 4g + q of the last warp loads
        # quarter q of token g's partials before its activation loads (they
        # are summed after the main loop, in the hop's partial order).
        pq = []
        if cutlass.const_expr(self.deferred):
            quarter = self.partials_per_row // 4
            p_ok = Int32(warp == Int32(nw - 1)) * tok_ok
            pbase = sumsq + Int64(
                (g * Int32(self.partials_per_row) + t4 * Int32(quarter)) * Int32(4)
            )
            for j in cutlass.range_constexpr(quarter // 2):
                lo, hi = _ld_v2_if(pbase + Int64(8 * j), p_ok)
                pq.append(bits_f32(lo))
                pq.append(bits_f32(hi))
        bfr = []
        if cutlass.const_expr(self.b_layout == "frag"):
            # Lanes of token rows >= R load nothing (zero fragments).
            a_lane = act + Int64(lane * Int32(8))
            for m in cutlass.range_constexpr(self.rounds):
                bfr.append(_ld_v2_if(a_lane + Int64((warp + Int32(nw * m)) * Int32(256)), tok_ok))
        else:
            a_row = act + Int64(safe_g * Int32(k) * Int32(2) + t4 * Int32(4))
            for m in cutlass.range_constexpr(self.rounds):
                off = Int64((warp + Int32(nw * m)) * Int32(32))
                b0 = _ld_b32(a_row + off)
                b1 = _ld_b32(a_row + off + Int64(16))
                bfr.append((_sel_i32(tok_ok, b0, Int32(0)), _sel_i32(tok_ok, b1, Int32(0))))
        _stamp(self.debug_ts, ts, bid, tid, 2)

        # ldmatrix row addresses (per lane), advanced by 32 B per K step.
        r16 = lane % Int32(16)
        a_lane0 = sW + r16 * Int32(stride) + (lane // Int32(16)) * Int32(16)
        ok0 = Int32(r16 < nsm)
        r8 = Int32(16) + lane % Int32(8)
        a_lane1 = sW + r8 * Int32(stride) + ((lane // Int32(8)) % Int32(2)) * Int32(16)
        ok1 = Int32(r8 < nsm)

        acc0 = [Float32(0.0)] * 4
        acc1 = [Float32(0.0)] * 4
        zero = Int32(0)
        rpc = self.rounds_per_chunk

        def load_a(n):
            if cutlass.const_expr(n % rpc == 0 and self.copy_method == "bulk"):
                _mbar_wait(sMbar + Int32(8 * (n // rpc)), Int32(0))
            kb = (warp + Int32(nw * n)) * Int32(32)
            return (
                ldsm_x4(_sel_i32(ok0, a_lane0 + kb, sZero)),
                _ldsm_x2(_sel_i32(ok1, a_lane1 + kb, sZero)),
            )

        frags = [load_a(n) for n in range(min(self.dist, self.rounds))]
        for m in cutlass.range_constexpr(self.rounds):
            if cutlass.const_expr(m + self.dist < self.rounds):
                frags.append(load_a(m + self.dist))
            fa, fb = frags[m]
            b0, b1 = bfr[m]
            acc0[0], acc0[1], acc0[2], acc0[3] = mma_bf16(
                fa[0], fa[1], fa[2], fa[3], b0, b1, acc0[0], acc0[1], acc0[2], acc0[3]
            )
            acc1[0], acc1[1], acc1[2], acc1[3] = mma_bf16(
                fb[0], zero, fb[1], zero, b0, b1, acc1[0], acc1[1], acc1[2], acc1[3]
            )
        _stamp(self.debug_ts, ts, bid, tid, 3)

        # r per token (published by the barrier below): the serial sum of
        # the 24 partials, quarter by quarter along the 4 lanes of token g.
        if cutlass.const_expr(self.deferred):
            if warp == Int32(nw - 1):
                s = Float32(0.0)
                for q in cutlass.range_constexpr(4):
                    mine = s
                    for v in pq:
                        mine = add(mine, v)
                    s = _shfl_idx_f32(mine, g * Int32(4) + Int32(q))
                if t4 == Int32(0):
                    _st_shared_f32(sR + g * Int32(4), inv_rms(s, Float32(float(k)), eps))
        else:
            if tid < Int32(MAX_TOKENS):
                _st_shared_f32(sR + tid * Int32(4), Float32(1.0))

        # Cross-warp reduction in warp order (the weight rows are dead now).
        cute.arch.sync_threads()
        if cutlass.const_expr(self.overflow > 0):
            if tid == Int32(0):
                if nrows > nsm:
                    _fence_proxy_async()
                    _mbar_expect_tx(sMbar + Int32(8 * self.nchunk), Int32(k * 2))
                    _bulk_g2s(
                        sW + Int32(2 * stride),
                        _row_src(w0, w1, row0 + nsm, k, self.n0, self.n1),
                        Int32(k * 2),
                        sMbar + Int32(8 * self.nchunk),
                    )
        sRed = sW
        wbase = sRed + warp * Int32(24 * MAX_TOKENS * 4)
        c0 = Int32(2) * t4
        _st_shared_f32(wbase + (g * Int32(MAX_TOKENS) + c0) * Int32(4), acc0[0])
        _st_shared_f32(wbase + (g * Int32(MAX_TOKENS) + c0 + Int32(1)) * Int32(4), acc0[1])
        _st_shared_f32(wbase + ((g + Int32(8)) * Int32(MAX_TOKENS) + c0) * Int32(4), acc0[2])
        _st_shared_f32(
            wbase + ((g + Int32(8)) * Int32(MAX_TOKENS) + c0 + Int32(1)) * Int32(4), acc0[3]
        )
        _st_shared_f32(wbase + ((g + Int32(16)) * Int32(MAX_TOKENS) + c0) * Int32(4), acc1[0])
        _st_shared_f32(
            wbase + ((g + Int32(16)) * Int32(MAX_TOKENS) + c0 + Int32(1)) * Int32(4), acc1[1]
        )
        cute.arch.sync_threads()
        if tid < Int32(24 * MAX_TOKENS):
            row = tid // Int32(MAX_TOKENS)
            tok = tid % Int32(MAX_TOKENS)
            if row < nsm:
                if tok < rows:
                    s = Float32(0.0)
                    for w in cutlass.range_constexpr(nw):
                        s = add(
                            s,
                            _ld_shared_f32(sRed + Int32(w * 24 * MAX_TOKENS * 4) + tid * Int32(4)),
                        )
                    _store_out(
                        out0,
                        out1,
                        row0 + row,
                        tok,
                        mul(_ld_shared_f32(sR + tok * Int32(4)), s),
                        self.n0,
                        self.n1,
                    )
        _stamp(self.debug_ts, ts, bid, tid, 4)

        if cutlass.const_expr(self.overflow > 0):
            if nrows > nsm:
                # The extra row: rows 0-7 of an m16 tile through ldmatrix.x2
                # with only row 0 real, K steps as in the main loop.
                _mbar_wait(sMbar + Int32(8 * self.nchunk), Int32(0))
                l8 = lane % Int32(8)
                okx = Int32(l8 == Int32(0))
                a_lanex = sW + Int32(2 * stride) + ((lane // Int32(8)) % Int32(2)) * Int32(16)
                accx = [Float32(0.0)] * 4
                for m in cutlass.range_constexpr(self.rounds):
                    kb = (warp + Int32(nw * m)) * Int32(32)
                    fx = _ldsm_x2(_sel_i32(okx, a_lanex + kb, sZero))
                    b0, b1 = bfr[m]
                    accx[0], accx[1], accx[2], accx[3] = mma_bf16(
                        fx[0], zero, fx[1], zero, b0, b1, accx[0], accx[1], accx[2], accx[3]
                    )
                sRedX = sW + Int32(3 * stride)
                if g == Int32(0):
                    _st_shared_f32(sRedX + (warp * Int32(MAX_TOKENS) + c0) * Int32(4), accx[0])
                    _st_shared_f32(
                        sRedX + (warp * Int32(MAX_TOKENS) + c0 + Int32(1)) * Int32(4), accx[1]
                    )
                cute.arch.sync_threads()
                if tid < rows:
                    s = Float32(0.0)
                    for w in cutlass.range_constexpr(nw):
                        s = add(
                            s, _ld_shared_f32(sRedX + Int32(w * MAX_TOKENS * 4) + tid * Int32(4))
                        )
                    _store_out(
                        out0,
                        out1,
                        row0 + nsm,
                        tid,
                        mul(_ld_shared_f32(sR + tid * Int32(4)), s),
                        self.n0,
                        self.n1,
                    )
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

        kernel = DeferredNormProjKernel(**cfg)
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
            z64,
            z64,
            z64,
            z64,
            z64,
            z64,
            anchor,
            Int32(1),
            Float32(1e-5),
            stream=cute_runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _COMPILED[key] = compiled
    return compiled


def projection_supported(
    k: int, n: int, ctas: int, threads: int, smem_rows: int, nchunk: int
) -> Optional[str]:
    """Return why the projection kernel cannot serve a shape, or ``None``."""
    nw = threads // 32
    if k % 16 or (k // 16) % nw:
        return f"K={k} does not split into {nw} warps of 16-wide steps"
    if k % nchunk or (k // nchunk) % (16 * nw):
        return f"K={k} does not split into {nchunk} chunks of whole warp rounds"
    base, rem = divmod(n, ctas)
    max_rows = base + (1 if rem else 0)
    if base <= 16 or max_rows > 24:
        return f"{n} rows over {ctas} CTAs gives {base}..{max_rows} rows per CTA (17..24 supported)"
    if max_rows - min(smem_rows, max_rows) > 1:
        return f"{max_rows} rows per CTA exceed {smem_rows} staged rows by more than one"
    if min(smem_rows, max_rows) * (k * 2 + 16) + 1024 > SMEM_LIMIT:
        return "staged rows exceed the shared-memory limit"
    return None
