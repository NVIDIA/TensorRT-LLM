# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-project/vllm@e3ddbdbd60 vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/gemm/splitk.py
# (itself adapted from FlashInfer dense_bf16_gemm_sm100_splitk.py); changed: the q_a RMSNorm runs as the activation
# prologue.
"""MR8's split-K BF16 GEMM with the q_a RMSNorm as its activation prologue (one launch).

``NormSplitKGemm`` is the vendored ``SplitKGemm`` (MR8 ``cutedsl/gemm/splitk.py`` in
``kernels/vendor/vllm_mr8``) with the activation warp's TMA replaced by the epilogue warps
(idle until the last MMA). For ``y = rmsnorm(x; gamma, eps) @ W^T`` with x [M, 1536] BF16
(any 16-byte row pitch), W [N, 1536] BF16 and y [M, N] BF16:

- before the PDL wait, the weight warp streams the CTA's K slice of its W rows (TMA,
  L2 evict_first, unchanged) and each norm thread loads its gamma chunks;
- after the wait, thread (row, t) of the 128 loads its row's 16-byte x chunks of the CTA's
  K slice, sums their squares in FP32 and the row's ``128 / mma_n`` threads reduce them
  (butterfly). The sum of squares of the whole row is either
  - exchanged (``full_rows`` off): the ``split_k`` CTAs of a cluster hold the whole row
    between them; each sends its per-row partials to every CTA of the cluster
    (``st.async`` into slot ``rank``) and every CTA sums the slots in rank order; or
  - read (``full_rows`` on): every CTA loads the whole rows (L2-resident) and sums the K
    slices in order, keeping only its own slice's chunks;
  either way every CTA of a cluster gets the same bits;
- ``r = rsqrt(sum / 1536 + eps)`` (``div.rn``, ``rsqrt.approx``: FlashInfer's RMSNorm
  formula), then ``bf16((x * r) * gamma)`` (FP32, round-to-nearest per product, one
  rounding to BF16, as FlashInfer's RMSNorm output) written as the same 128-byte
  swizzled K-major B stage the TMA would load; each thread fences the async proxy and
  arrives on the stage's full barrier;
- the MMAs, the FP32 cluster split-K reduction in rank order and the BF16 store are
  ``SplitKGemm``'s.

Rows past M load row M - 1 (their output columns are not stored). Every K tile of the
CTA's slice has its own ring stage (stages >= K tiles per CTA), so the B stages are
written once and never recycled. The launch takes only the dynamic shared memory the
tiles use (``splitk_ring.fit_smem``), so the CTAs can share SMs with the main stream's
kernels.
"""

from __future__ import annotations

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05

from ..vendor.vllm_mr8 import module

_splitk = module("cutedsl.gemm.splitk")
_ring = module("cutedsl.gemm.splitk_ring")
_prim = module("cutedsl.common.primitives")
_w8 = module("cutedsl.common.w8")

CTA_K, ATOM_K = _splitk.CTA_K, _splitk.ATOM_K
HIDDEN = 1536  # the q_a rank: ``query_norm_rsqrt`` divides by it
NORM_THREADS = 128
MAX_CHUNKS = 12  # 16-byte x chunks of its K slice a norm thread holds in registers
MAX_ROW_CHUNKS = 24  # 16-byte x chunks a norm thread loads with ``full_rows``
SMEM_LIMIT = 227 * 1024  # sm_100 / sm_103 / sm_107 opt-in dynamic shared memory per CTA


@_prim.asm(Uint32, pure=True)
def _pick(rank, p, a, b):
    """``a`` if ``rank == p`` else ``b``."""
    code = "{ .reg .pred q; setp.eq.s32 q,$1,$2; selp.b32 $0,$3,$4,q; }"
    return code, "=r,r,r,r,r", (Int32(rank), Int32(p), Uint32(a), Uint32(b))


def chunks_per_thread(mma_n: int, split_k: int, k: int = HIDDEN) -> int:
    """16-byte x chunks per norm thread: its row's K slice over the row's threads."""
    return k // split_k // 8 // (NORM_THREADS // mma_n)


def smem_bytes(tactic) -> int:
    """Dynamic shared memory of a launch: ``SplitKGemm``'s tiles plus the norm slots."""
    return _ring.fit_smem(_splitk.SplitKTactic(*tactic[:4])) + 1024


def supported(tactic, k: int = HIDDEN) -> bool:
    """Whether ``NormSplitKGemm`` builds for (mma_m, mma_n, split_k, stages[, full_rows])
    at K ``k``."""
    m, n, split, stages = tactic[:4]
    full_rows = bool(tactic[4]) if len(tactic) > 4 else False
    if m not in (64, 128) or n not in (8, 16, 32) or k != HIDDEN or k % (CTA_K * split):
        return False
    per_row = NORM_THREADS // n
    slice_chunks = k // split // 8
    chunks = chunks_per_thread(n, split, k)
    return (
        stages >= k // CTA_K // split
        and slice_chunks % per_row == 0
        and 16 % per_row == 0
        and chunks <= MAX_CHUNKS
        and (not full_rows or chunks * split <= MAX_ROW_CHUNKS)
        and smem_bytes(tactic) <= SMEM_LIMIT
    )


class NormSplitKGemm(_splitk.SplitKGemm):
    """``tactic`` (mma_m 64 / 128, mma_n 8 / 16 / 32, split_k, stages >= K tiles per CTA);
    ``full_rows``: every CTA reads the whole rows for the sum of squares (no exchange)."""

    def __init__(self, *, tactic, k: int = HIDDEN, full_rows: bool = False):
        super().__init__(tactic=tactic)
        tactic_fields = (tactic.mma_m, tactic.mma_n, tactic.split_k, tactic.ab_stages)
        self.full_rows = full_rows and self.split_k > 1
        if not supported((*tactic_fields, self.full_rows), k):
            raise ValueError(f"NormSplitKGemm: unsupported tactic {tactic_fields} for K {k}")
        self.k = k
        self.k_tiles = k // CTA_K // self.split_k
        self.per_row = NORM_THREADS // self.cta_n  # threads per activation row
        self.slice = k // self.split_k
        self.chunks = chunks_per_thread(self.cta_n, self.split_k, k)
        self.stage_chunks = (CTA_K // 8) // self.per_row  # a thread's chunks per K tile
        # FlashInfer's layout mirror of the SplitKGemm tiles + the norm slots and barrier.
        self.fit = smem_bytes(tactic_fields)

    def full_arrivals(self):
        """A stage's arrivals: the weight TMA's and one per norm thread."""
        return 1 + NORM_THREADS

    @cute.experimental.jit
    def __call__(
        self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, gamma: cutlass.Int64,
        a_map: cutlass.Int64, eps: cutlass.Float32, stream: _cuda.CUstream,
    ):  # fmt: skip
        self.kernel(a, b, c, gamma, a_map, eps).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.fit),
            stream=stream,
            use_pdl=True,
        )

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, gamma: cutlass.Int64,
        a_map: cutlass.Int64, eps: cutlass.Float32,
    ):  # fmt: skip
        """A (N, K, L) weights and B (M, K, L) activations K-major, C (N, M, L) M-major."""
        stages = self.num_ab_stage
        ab_dtype = mA.element_type
        majors = (utils.LayoutEnum.from_tensor(t).mma_major_mode() for t in (mA, mB))
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, *majors, self.acc_dtype, self.cta_group,
            self.mma_tiler_mn,
        )  # fmt: skip

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        shape = (tiled_mma, mnk_tiler, ab_dtype, stages)
        sA = _splitk._smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_a(*shape))
        sB = _splitk._smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_b(*shape))
        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma, self.mma_tiler_mn, acc_stage=1
        )
        bar_full = _splitk._smem(cutlass.Int64, stages, 8).iterator
        bar_empty = _splitk._smem(cutlass.Int64, stages, 8).iterator
        bar_tma_epilog = _splitk._smem(cutlass.Int64, 1, 8).iterator  # FlashInfer's bias
        bar_mma_epilog = _splitk._smem(cutlass.Int64, 1, 8).iterator
        bar_tmem_alloc = _splitk._smem(cutlass.Int64, 1, 8).iterator
        tmem_base_ptr = _splitk._smem(cutlass.Int32, 1, 4).iterator
        mailbox, bar_reduce = sA, bar_mma_epilog  # split 1: compile-time dead
        partials, bar_norm = sA, bar_mma_epilog
        if cutlass.const_expr(self.split_k > 1):
            mailbox = _splitk._smem(cutlass.Float32, self.mailbox_elements, 128)
            bar_reduce = _splitk._smem(cutlass.Int64, 1, 8).iterator
        if cutlass.const_expr(self.exchanges()):
            # Slot [rank][row]: cluster rank ``rank``'s sum of squares of tile row ``row``.
            partials = _splitk._smem(cutlass.Float32, self.split_k * self.cta_n, 16)
            bar_norm = _splitk._smem(cutlass.Int64, 1, 8).iterator

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, self.full_arrivals())
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
                if cutlass.const_expr(self.split_k > 1):
                    cute.arch.mbarrier_init(bar_reduce, 1)
                if cutlass.const_expr(self.exchanges()):
                    cute.arch.mbarrier_init(bar_norm, 1)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_arrive_relaxed()  # peers' barriers before their stores
        else:
            cute.arch.barrier()

        k_tile_count = cutlass.Int32(self.k_tiles)
        k_tile_start = split_rank * k_tile_count

        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_wait()

        # Warps 1 and 3 are idle; warps 4-7 normalize the activations, then run the epilogue.
        if warp_idx == 0:
            row0 = bidx * self.cta_m
            self.weight_warp(bar_full, bar_empty, sA, a_map, row0, k_tile_start, k_tile_count)
        elif warp_idx == 2:
            k_blocks = self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2])
            self.mma_warp(
                bar_full, bar_empty, bar_mma_epilog, bar_tmem_alloc, tiled_mma, sA, sB,
                tmem_base_ptr, acc_layout, k_blocks, k_tile_count,
            )  # fmt: skip
        elif warp_idx >= 4:
            tid = cute.arch.thread_idx()[0] - 128
            self.norm_rows(
                mB, gamma, eps, sB, bar_full, partials, bar_norm, split_rank, n_idx, tid
            )
            tile = cute.local_tile(mC, (self.cta_m, self.cta_n), (bidx, n_idx, l_idx))
            self.epilog_warp(
                bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr, acc_layout, tile, tid,
                mC.element_type, utils.LayoutEnum.from_tensor(mC), mailbox, bar_reduce,
                split_rank,
            )  # fmt: skip

    def exchanges(self) -> bool:
        return self.split_k > 1 and not self.full_rows

    @cute.experimental.jit
    def norm_rows(
        self, mB: cute.Tensor, gamma: cutlass.Int64, eps: cutlass.Float32,
        sB: cute.Tensor, bar_full, partials, bar_norm, split_rank: cutlass.Int32,
        n_idx: cutlass.Int32, tid: cutlass.Int32,
    ):  # fmt: skip
        """Thread ``tid`` (row ``tid // per_row`` of the N tile, part ``tid % per_row``):
        RMSNorm of its chunks of the CTA's K slice into the B stages."""
        per_row = self.per_row
        row = tid // per_row
        part = tid % per_row
        rows = cute.size(mB, mode=[0])
        src = n_idx * self.cta_n + row
        if src >= rows:
            src = rows - 1
        k0 = split_rank * self.slice
        # Chunk j of this thread: 8 elements at K offset k0 + 8 * (part + j * per_row).
        g_words = []
        for j in cutlass.range_constexpr(self.chunks):
            k = k0 + 8 * (part + j * per_row)
            address = gamma + Int64(k) * 2
            g_words.append(_prim.load_words(address, count=4))
        cute.arch.griddepcontrol_wait()
        x_words = []
        squares = Float32(0.0)
        if cutlass.const_expr(self.full_rows):
            # Every K slice of the row in order (the same sum in every CTA); keep this CTA's.
            row_words = []
            for p in cutlass.range_constexpr(self.split_k):
                for j in cutlass.range_constexpr(self.chunks):
                    k = p * self.slice + 8 * (part + j * per_row)
                    address = _prim.at(mB, src, k, 0)
                    row_words.append(_prim.load_words(address, count=4, mutable=True))
            for j in cutlass.range_constexpr(self.chunks):
                x_words.append(list(row_words[j]))
            for p in cutlass.range_constexpr(1, self.split_k):
                for j in cutlass.range_constexpr(self.chunks):
                    for w in cutlass.range_constexpr(4):
                        mine = row_words[p * self.chunks + j][w]
                        x_words[j][w] = _pick(split_rank, p, mine, x_words[j][w])
            for i in cutlass.range_constexpr(self.split_k * self.chunks):
                for w in cutlass.range_constexpr(4):
                    lo, hi = _prim.unpack_bf16(row_words[i][w])
                    squares = _prim.fma(lo, lo, squares)
                    squares = _prim.fma(hi, hi, squares)
        else:
            for j in cutlass.range_constexpr(self.chunks):
                k = k0 + 8 * (part + j * per_row)
                address = _prim.at(mB, src, k, 0)
                x_words.append(_prim.load_words(address, count=4, mutable=True))
            for j in cutlass.range_constexpr(self.chunks):
                for w in cutlass.range_constexpr(4):
                    lo, hi = _prim.unpack_bf16(x_words[j][w])
                    squares = _prim.fma(lo, lo, squares)
                    squares = _prim.fma(hi, hi, squares)
        squares = _prim.warp_sum(squares, per_row)
        total = squares
        if cutlass.const_expr(self.exchanges()):
            if part == 0:
                slot = partials.iterator + split_rank * self.cta_n + row
                for peer in cutlass.range_constexpr(self.split_k):
                    _prim.push_float(slot, squares, bar_norm, Int32(peer))
            if tid == 0:
                expected = self.split_k * self.cta_n * 4
                cute.arch.mbarrier_arrive_and_expect_tx(bar_norm, expected)
            _splitk._wait_cluster(bar_norm, Int32(0))
            total = partials[row]
            for peer in cutlass.range_constexpr(1, self.split_k):
                total = total + partials[peer * self.cta_n + row]
        rstd = _prim.query_norm_rsqrt(total, eps)
        for stage in cutlass.range_constexpr(self.k_tiles):
            tile = _prim.shared_address(sB[None, None, None, stage].iterator)
            for i in cutlass.range_constexpr(self.stage_chunks):
                j = stage * self.stage_chunks + i
                within = (part + j * per_row) % (CTA_K // 8)  # chunk in the K tile
                atom, chunk = within // (ATOM_K // 8), within % (ATOM_K // 8)
                words = []
                for w in cutlass.range_constexpr(4):
                    x = _prim.unpack_bf16(x_words[j][w])
                    g = _prim.unpack_bf16(g_words[j][w])
                    y = _prim.mul2(_prim.mul2(x, (rstd, rstd)), g)
                    words.append(_prim.bf16x2(y[0], y[1]))
                box = tile + atom * (self.cta_n * ATOM_K * 2)
                _w8.sts128(_w8.swizzled(box, row, chunk), words)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.mbarrier_arrive(bar_full + stage)
