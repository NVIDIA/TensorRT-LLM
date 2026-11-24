# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Split-boundary coefficients of large decode batches on tcgen05 tensor cores.

``MhcCoefficientTcKernel`` computes a lagged boundary's ``pre`` / ``post`` / ``comb`` from its BF16
residual. A work item is a tile of 128 tokens and one of ``SLICES`` K slices; persistent CTAs
stream each item's residual and weight k-blocks through shared memory into a tcgen05 accumulator
(the products of the TF32-rounded weight split into BF16 high and low parts, as
``prepare_coefficient_weight``; FP32 accumulation) while the epilogue warps sum the squares of their
tokens' residual rows from the same stages. Each token and slice stores one partial (the 24 mixes
and the sum of squares); the last of a 32-token group's partials (one arrival counter per group,
reset by its last arrival) sums them in slice order and runs the coefficients and the Sinkhorn
iterations, one token per lane. Results are deterministic.
"""

import functools

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import SmemAllocator

from tensorrt_llm.math_utils import ceil_div

from .mhc_boundary import (
    MIXES,
    MULT,
    MhcBoundaryKernel,
    _mat,
    atom_add_acq_rel,
    bf16x2_hi,
    bf16x2_lo,
    bulk_g2s,
    f32_bits,
    fence_gpu,
    l2_policy,
    ldg_cg_v4_f32,
    ldg_f32,
    lds_f32,
    lds_v4_u32,
    mbar_expect_tx,
    mbar_wait,
    rcp,
    shfl_idx_i32,
    sigmoid,
    stg_u32,
    stg_v4_u32,
    sts_f32,
)

Int32, Int64, Float32 = cutlass.Int32, cutlass.Int64, cutlass.Float32


@dsl_user_op
def lds_v4_f32(addr, *, loc=None, ip=None):
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(4))


TILE = 128  # tokens per work item and accumulator (UMMA M)
N = 2 * MIXES  # weight rows: 24 high, 24 low (UMMA N)
BK = 64  # K per stage: one 128-byte swizzle row
# Work items of one tile and a quarter of K on 40 CTAs: few SMs, so the overlapped sublayer keeps
# the rest.
SLICES, CTAS = 4, 40
STAGES = 8  # 22 KB each (residual tile + weight)
EPI_WARPS = 4
MMA_WARP = 4
TMA_WARP = 5
THREADS = 32 * (EPI_WARPS + 2)
CHUNKS = MIXES // 4 + 1  # float4 per token partial: the mixes, then the sum of squares
PART_BYTES = CHUNKS * 16
GROUP = 32  # tokens per arrival counter (one epilogue warp)
# The finalize stages a warp's partials in the idle residual buffers.
assert EPI_WARPS * SLICES * GROUP * PART_BYTES <= STAGES * TILE * BK * 2


class MhcCoefficientTcKernel(MhcBoundaryKernel):
    """``(pre, post, comb)`` of a boundary from its residual ``[M, 4, H]``."""

    def __init__(self, hidden: int, sinkhorn_iters: int):
        self.hidden = hidden
        self.sinkhorn_iters = sinkhorn_iters
        self.k = MULT * hidden
        self.kb = self.k // (SLICES * BK)  # k-blocks per item
        self.tiler = (TILE, N, BK)

    @staticmethod
    def supports(hidden: int) -> bool:
        return (MULT * hidden) % (SLICES * BK) == 0

    @cute.jit
    def __call__(
        self,
        residual,
        weight,
        scale,
        base,
        partials,
        counters,
        pre_out,
        post_out,
        comb_out,
        rms_eps: Float32,
        hc_eps: Float32,
        sinkhorn_eps: Float32,
        post_mult: Float32,
        ctas: Int32,
        stream,
    ):
        tokens = residual.shape[0]
        a = cute.make_tensor(
            residual.iterator,
            cute.make_layout((tokens, self.k, 1), stride=(self.k, 1, tokens * self.k)),
        )
        b = cute.make_tensor(
            weight.iterator, cute.make_layout((N, self.k, 1), stride=(self.k, 1, N * self.k))
        )
        mma = sm100_utils.make_trivial_tiled_mma(
            cutlass.BFloat16,
            cutlass.BFloat16,
            OperandMajorMode.K,
            OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            (TILE, N),
        )
        sA_l = sm100_utils.make_smem_layout_a(mma, self.tiler, cutlass.BFloat16, STAGES)
        sB_l = sm100_utils.make_smem_layout_b(mma, self.tiler, cutlass.BFloat16, STAGES)
        cluster = cute.tiled_divide(cute.make_layout((1, 1, 1)), (mma.thr_id.shape,))
        tma_a, gA = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A((1, 1), mma.thr_id),
            a,
            cute.slice_(sA_l, (None, None, None, 0)),
            self.tiler,
            mma,
            cluster.shape,
        )
        tma_b, gB = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B((1, 1), mma.thr_id),
            b,
            cute.slice_(sB_l, (None, None, None, 0)),
            self.tiler,
            mma,
            cluster.shape,
        )
        self.kernel(
            mma,
            tma_a,
            gA,
            tma_b,
            gB,
            sA_l,
            sB_l,
            scale,
            base,
            partials,
            counters,
            pre_out,
            post_out,
            comb_out,
            rms_eps,
            hc_eps,
            sinkhorn_eps,
            post_mult,
            ctas,
        ).launch(grid=[ctas, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        mma,
        tma_a,
        gA_tma,
        tma_b,
        gB_tma,
        sA_l,
        sB_l,
        scale,
        base,
        partials,
        counters,
        pre_out,
        post_out,
        comb_out,
        rms_eps: Float32,
        hc_eps: Float32,
        sinkhorn_eps: Float32,
        post_mult: Float32,
        ctas: Int32,
    ):
        tidx = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cta = cute.arch.block_idx()[0]
        tokens = pre_out.shape[0]
        items = (tokens + TILE - 1) // TILE * SLICES
        mine = (items - cta + ctas - 1) // ctas

        smem = SmemAllocator()
        sA = smem.allocate_tensor(cutlass.BFloat16, sA_l.outer, 1024, sA_l.inner)
        sB = smem.allocate_tensor(cutlass.BFloat16, sB_l.outer, 1024, sB_l.inner)
        bars = smem.allocate_array(Int64, 2 * STAGES + 4 + EPI_WARPS)
        full = bars
        empty = bars + STAGES
        acc_full = bars + 2 * STAGES
        acc_empty = bars + 2 * STAGES + 2
        bulk_bars = bars + 2 * STAGES + 4
        tmem_holder = smem.allocate_array(Int32, 4)
        s_hcw = smem.allocate(32 * 4, 16).toint()

        if warp == TMA_WARP:
            cpasync.prefetch_descriptor(tma_a)
            cpasync.prefetch_descriptor(tma_b)
        if tidx == 0:
            for i in cutlass.range_constexpr(STAGES):
                cute.arch.mbarrier_init(full + i, 1)
                # Released by the MMA commit and by each epilogue warp's square sums.
                cute.arch.mbarrier_init(empty + i, 1 + EPI_WARPS)
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(acc_full + i, 1)
                cute.arch.mbarrier_init(acc_empty + i, EPI_WARPS * 32)
            for i in cutlass.range_constexpr(EPI_WARPS):
                cute.arch.mbarrier_init(bulk_bars + i, 1)
        elif (tidx >= 32) & (tidx < 32 + 3 + MIXES):
            i = tidx - 32
            if i < 3:
                sts_f32(s_hcw + i * 4, ldg_f32(scale.iterator.toint() + i * 4))
            else:
                sts_f32(s_hcw + i * 4, ldg_f32(base.iterator.toint() + (i - 3) * 4))
        cute.arch.mbarrier_init_fence()
        # Two accumulator buffers of N columns.
        if warp == MMA_WARP:
            cute.arch.alloc_tmem(128, tmem_holder)
        cute.arch.sync_threads()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, 16, tmem_holder)
        acc_shape = mma.partition_shape_C((TILE, N))
        tAcc = cute.make_tensor(tmem_ptr, mma.make_fragment_C(cute.append(acc_shape, 2)).layout)

        if warp == TMA_WARP:
            gA = cute.local_tile(
                gA_tma, cute.slice_(self.tiler, (None, 0, None)), (None, None, None)
            )
            gB = cute.local_tile(
                gB_tma, cute.slice_(self.tiler, (0, None, None)), (None, None, None)
            )
            thr = mma.get_slice(0)
            tAs, tAg = cpasync.tma_partition(
                tma_a,
                0,
                cute.make_layout(1),
                cute.group_modes(sA, 0, 3),
                cute.group_modes(thr.partition_A(gA), 0, 3),
            )
            tBs, tBg = cpasync.tma_partition(
                tma_b,
                0,
                cute.make_layout(1),
                cute.group_modes(sB, 0, 3),
                cute.group_modes(thr.partition_B(gB), 0, 3),
            )
            # The residual keeps the L2 priority its writer gave it.
            keep = l2_policy("evict_unchanged")
            for j in cutlass.range(mine, unroll=1):
                item = cta + j * ctas
                tile = item // SLICES
                k0 = item % SLICES * self.kb
                for kb in cutlass.range(self.kb, unroll=1):
                    it = j * self.kb + kb
                    s = it % STAGES
                    cute.arch.mbarrier_wait(empty + s, ((it // STAGES) & 1) ^ 1)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(full + s, (TILE + N) * BK * 2)
                    cute.copy(
                        tma_a,
                        tAg[(None, tile, k0 + kb, 0)],
                        tAs[(None, s)],
                        tma_bar_ptr=full + s,
                        cache_policy=keep,
                    )
                    cute.copy(
                        tma_b, tBg[(None, 0, k0 + kb, 0)], tBs[(None, s)], tma_bar_ptr=full + s
                    )
        elif warp == MMA_WARP:
            tCrA = mma.make_fragment_A(sA)
            tCrB = mma.make_fragment_B(sB)
            for j in cutlass.range(mine, unroll=1):
                buf = j % 2
                cute.arch.mbarrier_wait(acc_empty + buf, ((j // 2) & 1) ^ 1)
                acc = tAcc[(None, 0, 0, buf)]
                for kb in cutlass.range(self.kb, unroll=1):
                    it = j * self.kb + kb
                    s = it % STAGES
                    cute.arch.mbarrier_wait(full + s, (it // STAGES) & 1)
                    for kk in cutlass.range_constexpr(cute.size(tCrA, mode=[2])):
                        if cutlass.const_expr(kk == 0):
                            mma.set(tcgen05.Field.ACCUMULATE, kb != 0)
                        else:
                            mma.set(tcgen05.Field.ACCUMULATE, True)
                        cute.gemm(mma, acc, tCrA[(None, 0, kk, s)], tCrB[(None, 0, kk, s)], acc)
                    with cute.arch.elect_one():
                        tcgen05.commit(empty + s)
                with cute.arch.elect_one():
                    tcgen05.commit(acc_full + buf)
        elif warp < EPI_WARPS:
            # Thread tidx owns TMEM lane tidx: token tidx of each accumulator's tile.
            lane = tidx % 32
            atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x16), Float32)
            for j in cutlass.range(mine, unroll=1):
                item = cta + j * ctas
                tile = item // SLICES
                sl = item % SLICES
                buf = j % 2
                sq = self.square_sums(sA.iterator.toint(), j, tidx, lane, full, empty)
                # No warp stages partials into the pipeline buffers while another still reads them.
                cute.arch.barrier(barrier_id=1, number_of_threads=EPI_WARPS * 32)
                cute.arch.mbarrier_wait(acc_full + buf, (j // 2) & 1)
                tmpl = cute.local_tile(tAcc[(None, None), 0, 0, buf], (TILE, N), (0, 0))
                t2r = tcgen05.make_tmem_copy(atom, tmpl)
                thr = t2r.get_slice(tidx)
                acc = cute.make_rmem_tensor_like(
                    thr.partition_D(cute.make_identity_tensor((TILE, N))), Float32
                )
                cute.copy(t2r, thr.partition_S(tmpl), acc)
                cute.arch.fence_view_async_tmem_load()
                cute.arch.mbarrier_arrive(acc_empty + buf)
                # high + low weight rows: the slice's partial of each mix
                vals = [acc[i] + acc[MIXES + i] for i in range(MIXES)]
                vals += [sq, Float32(0.0), Float32(0.0), Float32(0.0)]
                dst = partials.iterator.toint() + (
                    Int64((tile * SLICES + sl) * TILE + tidx) * PART_BYTES
                )
                for c in cutlass.range_constexpr(CHUNKS):
                    stg_v4_u32(dst + c * 16, [f32_bits(v) for v in vals[4 * c : 4 * c + 4]])
                fence_gpu()
                cute.arch.sync_warp()
                old = Int32(0)
                if (tile * TILE + warp * GROUP < tokens) & (lane == 0):
                    old = atom_add_acq_rel(
                        counters.iterator.toint() + Int64(tile * 4 + warp) * 4, 1
                    )
                last = shfl_idx_i32(old, 0) == SLICES - 1
                # After the CTA's last item the pipeline's shared memory is idle: the partials are
                # copied there in bulk instead of loaded lane by lane.
                staged = j == mine - 1
                if last:
                    self.finalize(
                        tile,
                        warp,
                        lane,
                        tokens,
                        partials,
                        counters,
                        s_hcw,
                        pre_out,
                        post_out,
                        comb_out,
                        rms_eps,
                        hc_eps,
                        sinkhorn_eps,
                        post_mult,
                        staged,
                        sA.iterator.toint() + warp * (SLICES * GROUP * PART_BYTES),
                        bulk_bars + warp,
                    )
        cute.arch.sync_threads()
        if warp == MMA_WARP:
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(tmem_ptr, 128)

    @cute.jit
    def square_sums(self, smem_a, j, tidx, lane, full, empty):
        """Thread ``tidx``'s sum of squares of its token's residual over item ``j``'s k-blocks, read
        from the pipeline stages before they are released (row ``tidx`` of a stage is its 128 bytes
        at ``tidx * 128``, 16-byte chunk ``c`` at ``c ^ (tidx % 8)``)."""
        sq = cute.make_rmem_tensor(cute.make_layout((8,)), Float32)
        sq.fill(0.0)
        swz = tidx % 8
        for kb in cutlass.range(self.kb, unroll=1):
            it = j * self.kb + kb
            s = it % STAGES
            cute.arch.mbarrier_wait(full + s, (it // STAGES) & 1)
            row = smem_a + (s * TILE + tidx) * (BK * 2)
            for c in cutlass.range_constexpr(8):
                # Logical chunk order: the eight lanes of a phase hit eight bank groups.
                w = lds_v4_u32(row + (c ^ swz) * 16)
                for x in cutlass.range_constexpr(4):
                    lo = bf16x2_lo(w[x])
                    hi = bf16x2_hi(w[x])
                    sq[c] = cute.math.fma(hi, hi, cute.math.fma(lo, lo, sq[c]))
            # The reads finish before the async proxy refills the stage.
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_warp()
            if lane == 0:
                cute.arch.mbarrier_arrive(empty + s)
        return ((sq[0] + sq[1]) + (sq[2] + sq[3])) + ((sq[4] + sq[5]) + (sq[6] + sq[7]))

    @cute.jit
    def finalize(
        self,
        tile,
        warp,
        lane,
        tokens,
        partials,
        counters,
        s_hcw,
        pre_out,
        post_out,
        comb_out,
        rms_eps: Float32,
        hc_eps: Float32,
        eps: Float32,
        post_mult: Float32,
        staged,
        region,
        bulk_bar,
    ):
        """Coefficients of the 32 tokens of ``(tile, warp)`` from their slice partials, one per lane.

        ``staged``: the partials go through the idle pipeline buffer ``region`` (bulk copies
        completing on ``bulk_bar``)."""
        fence_gpu()
        if lane == 0:
            stg_u32(counters.iterator.toint() + Int64(tile * 4 + warp) * 4, cutlass.Uint32(0))
        first = partials.iterator.toint() + Int64(tile * SLICES * TILE + warp * GROUP) * PART_BYTES
        if staged:
            self.coefficients32(
                self.staged_sums(first, lane, region, bulk_bar),
                tile * TILE + warp * GROUP + lane,
                tokens,
                s_hcw,
                pre_out,
                post_out,
                comb_out,
                rms_eps,
                hc_eps,
                eps,
                post_mult,
            )
        else:
            self.coefficients32(
                self.loaded_sums(first + Int64(lane) * PART_BYTES),
                tile * TILE + warp * GROUP + lane,
                tokens,
                s_hcw,
                pre_out,
                post_out,
                comb_out,
                rms_eps,
                hc_eps,
                eps,
                post_mult,
            )

    @cute.jit
    def staged_sums(self, first, lane, region, bulk_bar):
        """A lane's token sums in slice order, its 32-token group's partials copied in bulk to
        ``region`` (3.5 KB of one slice per copy)."""
        tot = [Float32(0.0)] * (4 * CHUNKS)
        # Earlier generic reads of the buffer finish before the async proxy refills it, and the
        # partials other CTAs stored (acquired through the arrival counter) are visible to it.
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.fence_proxy("async.global")
        cute.arch.sync_warp()
        bar = bulk_bar.toint()
        if lane == 0:
            mbar_expect_tx(bar, SLICES * GROUP * PART_BYTES)
            for s in cutlass.range_constexpr(SLICES):
                bulk_g2s(
                    region + s * GROUP * PART_BYTES,
                    first + Int64(s * TILE) * PART_BYTES,
                    GROUP * PART_BYTES,
                    bar,
                )
        mbar_wait(bar, 0)
        for s in cutlass.range_constexpr(SLICES):
            for c in cutlass.range_constexpr(CHUNKS):
                v = lds_v4_f32(region + (s * GROUP + lane) * PART_BYTES + c * 16)
                for x in cutlass.range_constexpr(4):
                    tot[4 * c + x] = tot[4 * c + x] + v[x]
        return tot

    @cute.jit
    def loaded_sums(self, src):
        """A lane's token sums in slice order from its partials at ``src`` (L2 loads, all issued
        before the sums)."""
        tot = [Float32(0.0)] * (4 * CHUNKS)
        vals = [
            ldg_cg_v4_f32(src + Int64(s * TILE) * PART_BYTES + c * 16)
            for s in range(SLICES)
            for c in range(CHUNKS)
        ]
        for s in cutlass.range_constexpr(SLICES):
            for c in cutlass.range_constexpr(CHUNKS):
                for x in cutlass.range_constexpr(4):
                    tot[4 * c + x] = tot[4 * c + x] + vals[s * CHUNKS + c][x]
        return tot

    @cute.jit
    def coefficients32(
        self,
        tot,
        row,
        tokens,
        s_hcw,
        pre_out,
        post_out,
        comb_out,
        rms_eps: Float32,
        hc_eps: Float32,
        eps: Float32,
        post_mult: Float32,
    ):
        """``coefficients8``'s arithmetic for this lane's token ``row`` from its mixes and sum of
        squares ``tot``."""
        rstd = cute.math.rsqrt(tot[MIXES] / Float32(MULT * self.hidden) + rms_eps)
        hcw = [lds_f32(s_hcw + i * 4) for i in range(3 + MIXES)]
        pre = [
            sigmoid(cute.math.fma(tot[i] * rstd, hcw[0], hcw[3 + i])) + hc_eps for i in range(MULT)
        ]
        post = [
            sigmoid(cute.math.fma(tot[MULT + i] * rstd, hcw[1], hcw[3 + MULT + i])) * post_mult
            for i in range(MULT)
        ]
        rows = []
        for i in cutlass.range_constexpr(MULT):
            row_v = [
                cute.math.fma(tot[8 + 4 * i + k] * rstd, hcw[2], hcw[11 + 4 * i + k])
                for k in range(MULT)
            ]
            row_max = cute.arch.fmax(
                cute.arch.fmax(row_v[0], row_v[1]), cute.arch.fmax(row_v[2], row_v[3])
            )
            row_v = [cute.math.exp(v - row_max) for v in row_v]
            inv = rcp(row_v[0] + row_v[1] + row_v[2] + row_v[3])
            rows.extend([cute.math.fma(v, inv, eps) for v in row_v])
        comb = self.sinkhorn(rows, eps)
        if row < tokens:
            gt = Int64(row)
            stg_v4_u32(pre_out.iterator.toint() + gt * 16, [f32_bits(v) for v in pre])
            stg_v4_u32(post_out.iterator.toint() + gt * 16, [f32_bits(v) for v in post])
            for i in cutlass.range_constexpr(MULT):
                stg_v4_u32(
                    comb_out.iterator.toint() + (gt * 16 + i * 4) * 4,
                    [f32_bits(v) for v in comb[4 * i : 4 * i + 4]],
                )


_COMPILED = set()
# Tokens one ``coefficient_counters`` buffer serves.
MAX_TOKENS = 8192


@functools.cache
def _compile(hidden: int, sinkhorn_iters: int):
    tokens = cute.sym_int()
    fn = cute.compile(
        MhcCoefficientTcKernel(hidden, sinkhorn_iters),
        _mat(cutlass.BFloat16, tokens, MULT, hidden),
        _mat(cutlass.BFloat16, N, MULT * hidden),
        _mat(Float32, 3),
        _mat(Float32, MIXES),
        _mat(Float32, cute.sym_int()),
        _mat(Int32, cute.sym_int()),
        _mat(Float32, tokens, MULT),
        _mat(Float32, tokens, MULT),
        _mat(Float32, tokens, MULT * MULT),
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )
    _COMPILED.add((hidden, sinkhorn_iters))
    return fn


def compiled(hidden: int, sinkhorn_iters: int) -> bool:
    """Whether ``mhc_coefficients_tc`` is compiled for this variant (not allowed inside graph capture)."""
    return (hidden, sinkhorn_iters) in _COMPILED


def coefficient_counters(device) -> torch.Tensor:
    """Zeroed arrival counters for ``mhc_coefficients_tc``; the kernel leaves them zeroed. Launches
    sharing a buffer must not overlap."""
    return torch.zeros(ceil_div(MAX_TOKENS, GROUP), dtype=torch.int32, device=device)


def mhc_coefficients_tc(
    residual,
    weight,
    scale,
    base,
    rms_eps,
    hc_eps,
    sinkhorn_eps,
    post_mult,
    sinkhorn_iters,
    counters,
):
    """``(pre, post, comb)`` of a boundary from its BF16 residual ``[B, 4, hidden]``; ``weight`` is
    ``prepare_coefficient_weight``'s and ``counters`` ``coefficient_counters``'s."""
    tokens, mult, hidden = residual.shape
    assert 0 < tokens <= MAX_TOKENS
    dev = residual.device
    items = ceil_div(tokens, TILE) * SLICES
    pre, post = torch.empty(2, tokens, mult, dtype=torch.float32, device=dev)
    comb = torch.empty(tokens, mult * mult, dtype=torch.float32, device=dev)
    partials = torch.empty(items * TILE * PART_BYTES // 4, dtype=torch.float32, device=dev)
    _compile(hidden, sinkhorn_iters)(
        residual,
        weight,
        scale,
        base,
        partials,
        counters,
        pre,
        post,
        comb,
        rms_eps,
        hc_eps,
        sinkhorn_eps,
        post_mult,
        min(CTAS, items),
    )
    return pre, post, comb
