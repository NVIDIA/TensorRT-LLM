# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from FlashInfer (Apache-2.0; upstream notice: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.):
# https://github.com/flashinfer-ai/flashinfer/blob/4b6f2ead50642001033179b1f77cb443c794a9cb/flashinfer/gemm/kernels/dense_bf16_gemm_sm100_splitk.py
"""Low-M BF16 cluster split-K GEMM whose weight is on chip before its grid wait (helix8 latent_up).

The kernel is a programmatic (PDL) dependent of the kernel that writes its activation. Its weight
does not depend on that kernel, so the weight warp loads it at launch, before the activation warp's
``griddepcontrol.wait``: every resident K tile of the CTA's chunk goes to shared memory at once (no
ring throttles the loads); the chunk's other K tiles stream from HBM as the MMA frees stages.
After the wait only the activation and those tiles are read.

Swap-AB, as FlashInfer's split-K kernel (``rubin_ull_k3_low_m_bf16_splitk``, whose DMA, MMA and
epilogue structure this subclass keeps): kernel-M is the weight's 7168 output features (128-row
tiles), kernel-N the tokens (one token tile of ``mma_n`` = 16 * ceil(M / 16) up to 128), and the
K tiles of 128 are split over a cluster of ``split_k`` CTAs in front-heavy chunks (the first
``K tiles % split_k`` ranks take one more, as ``splitk_chunked`` does). The weight and the
activation have separate stage rings: ``a_stages`` resident weight stages (recycled only past
them) and a short activation ring. The MMA warp releases the dependents after its last MMA.

Each rank accumulates its chunk in FP32 in K order; the peers send their FP32 partials to rank 0
through DSMEM, and rank 0 adds them in rank order and rounds once to BF16: the words of
FlashInfer's split-K over the same chunks, a sum taken in another order than a single-CTA GEMM's.
"""

from __future__ import annotations

import functools
from typing import NamedTuple, Tuple

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Int32
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05

from . import rubin_ull_k3_low_m_bf16_splitk as _sk

#: Cluster ranks along K.
SPLIT_K = 3

#: Weight rows (kernel-M) per CTA and K extent of one tile.
CTA_M = 128
CTA_K = _sk._CTA_K

#: Most rows (tokens) one token tile serves.
MAX_M = 128

#: Per-CTA shared memory the device grants on sm_107.
SMEM_CAPACITY_BYTES = utils.get_smem_capacity_in_bytes("sm_107")

#: Default shared-memory budget per CTA: below the capacity, so the small CTAs of the kernel that
#: follows (e.g. a PDL mix) can be resident beside it while it finishes.
DEFAULT_SMEM_BUDGET = 320 * 1024

#: Activation ring depth up to 64 tokens per tile, and above.
_B_STAGES_NARROW = 4
_B_STAGES_WIDE = 2

#: Bytes of one BF16 weight K tile, one FP32 value, one mbarrier.
_A_STAGE_BYTES = CTA_M * CTA_K * 2
_FP32_BYTES = 4
_MBARRIER_BYTES = 8

#: Shared memory kept free of the accounted layout (allocator padding).
_SMEM_SLACK_BYTES = 2048


class ResidentConfig(NamedTuple):
    """One compiled specialization."""

    mma_n: int
    split_k: int
    a_stages: int
    b_stages: int


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _mailbox_elements(mma_n: int, split_k: int) -> int:
    """FP32 values of the DSMEM mailbox: every peer's partial."""
    return (split_k - 1) * CTA_M * mma_n


def smem_bytes(config: ResidentConfig) -> int:
    """The shared memory the kernel allocates, in allocation order."""
    cursor = config.a_stages * _A_STAGE_BYTES
    cursor = (
        _align_up(cursor, _sk._AB_BUFFER_ALIGN_BYTES) + config.b_stages * config.mma_n * CTA_K * 2
    )
    barriers = 2 * config.a_stages + 2 * config.b_stages + 2
    cursor = _align_up(cursor, _MBARRIER_BYTES) + barriers * _MBARRIER_BYTES
    cursor = _align_up(cursor, _sk._TMEM_POINTER_BYTES) + _sk._TMEM_POINTER_BYTES
    elements = _mailbox_elements(config.mma_n, config.split_k)
    cursor = _align_up(cursor, _sk._MAILBOX_ALIGN_BYTES) + elements * _FP32_BYTES
    return _align_up(cursor, _MBARRIER_BYTES) + _MBARRIER_BYTES


def launch_smem_bytes(config: ResidentConfig) -> int:
    """The dynamic shared memory a CTA is launched with: the layout plus slack, in KiB."""
    return _align_up(smem_bytes(config) + _SMEM_SLACK_BYTES, 1024)


def k_chunks(k: int, split_k: int = SPLIT_K) -> list[tuple[int, int]]:
    """``(first, count)`` K tiles of each rank: front-heavy, as the kernel partitions them."""
    tiles = k // CTA_K
    base, extra = divmod(tiles, split_k)
    chunks, first = [], 0
    for rank in range(split_k):
        count = base + (1 if rank < extra else 0)
        chunks.append((first, count))
        first += count
    return chunks


def select_config(m: int, k: int, smem_budget: int = DEFAULT_SMEM_BUDGET) -> ResidentConfig:
    """The specialization for ``m`` tokens and ``k``: an activation ring of 4 stages up to 64
    tokens per tile (else 2) and as many resident weight stages as the largest chunk needs and
    ``smem_budget`` holds (at least 2)."""
    if not 1 <= m <= MAX_M:
        raise ValueError(f"the resident latent_up GEMM serves 1..{MAX_M} rows, got {m}")
    if k % CTA_K or k // CTA_K < SPLIT_K:
        raise ValueError(f"K={k} must be a multiple of {CTA_K} with at least {SPLIT_K} tiles")
    if not 0 < smem_budget <= SMEM_CAPACITY_BYTES:
        raise ValueError(f"smem_budget {smem_budget} outside (0, {SMEM_CAPACITY_BYTES}]")
    mma_n = 16 * ((m + 15) // 16)
    chunk = max(count for _, count in k_chunks(k))
    b_stages = min(chunk, _B_STAGES_NARROW if mma_n <= 64 else _B_STAGES_WIDE)
    best = None
    for a_stages in range(1, chunk + 1):
        config = ResidentConfig(mma_n, SPLIT_K, a_stages, b_stages)
        if launch_smem_bytes(config) <= smem_budget:
            best = config
    if best is None or best.a_stages < 2:
        raise ValueError(f"no resident config for M={m}, K={k} within {smem_budget} B")
    return best


class LatentUpResidentKernel(_sk.SplitKDenseGemmKernel):
    """FlashInfer's cluster split-K GEMM with a resident weight ring loaded before the grid wait."""

    def __init__(self, config: ResidentConfig, smem_launch_bytes: int):
        super().__init__(
            tactic=_sk.SplitKTactic(CTA_M, config.mma_n, config.split_k, config.a_stages),
            use_pdl=True,
            has_bias=False,
        )
        self.a_stages = config.a_stages
        self.b_stages = config.b_stages
        self.smem_launch_bytes = smem_launch_bytes
        self.mailbox_elements = _mailbox_elements(config.mma_n, config.split_k)
        self.expected_transaction_bytes = self.mailbox_elements * _FP32_BYTES

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        bias: cute.Tensor,
        stream: _cuda.CUstream,
    ):
        # Grid-y packs the token tile and the cluster rank.
        self.kernel(a, b, c, bias).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.smem_launch_bytes),
            stream=stream,
            use_pdl=True,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major: the weight
        mB: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major: the activation
        mC: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L), M-major
        mBias: cute.Tensor,  # dead (no bias)
    ):
        """Allocate storage and dispatch the specialized warps."""
        ab_dtype = mA.element_type
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            utils.LayoutEnum.from_tensor(mA).mma_major_mode(),
            utils.LayoutEnum.from_tensor(mB).mma_major_mode(),
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        sA = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, self.a_stages),
            alignment=_sk._AB_BUFFER_ALIGN_BYTES,
        )
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, self.b_stages),
            alignment=_sk._AB_BUFFER_ALIGN_BYTES,
        )

        acc_layout = cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)
        c_tiler_mn = (self.cta_m, self.cta_n)

        bar_full_a = self._barriers(self.a_stages)
        bar_empty_a = self._barriers(self.a_stages)
        bar_full_b = self._barriers(self.b_stages)
        bar_empty_b = self._barriers(self.b_stages)
        bar_mma_epilog = self._barriers(1)
        bar_tmem_alloc = self._barriers(1)
        tmem_base_ptr = cute_ext.allocate(
            cutlass.Int32,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_sk._TMEM_POINTER_BYTES,
        ).iterator
        mailbox = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(self.mailbox_elements),
            alignment=_sk._MAILBOX_ALIGN_BYTES,
        )
        bar_reduce = self._barriers(1)

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(self.a_stages):
                    cute.arch.mbarrier_init(bar_full_a + i, 1)
                    cute.arch.mbarrier_init(bar_empty_a + i, 1)
                for i in range(self.b_stages):
                    cute.arch.mbarrier_init(bar_full_b + i, 1)
                    cute.arch.mbarrier_init(bar_empty_b + i, 1)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
                # Owner arrival plus the senders' transaction bytes.
                cute.arch.mbarrier_init(bar_reduce, 1)

        cute.arch.mbarrier_init_fence()
        # Publish the barriers before any peer's remote store.
        cute.arch.cluster_arrive_relaxed()

        # Front-heavy K chunks: the first (K tiles % split) ranks take one 128-wide K tile more.
        k_dim = cute.size(mA, mode=[1])
        k_tiles = k_dim // self.cta_k
        base = k_tiles // self.split_k
        extra = k_tiles - base * self.split_k
        k_tile_count = base
        k_tile_start = split_rank * base + extra
        if split_rank < extra:
            k_tile_count = base + 1
            k_tile_start = split_rank * base + split_rank

        cute.arch.cluster_wait()

        # Warp 3 is idle; warps 4-7 run the epilogue.
        if warp_idx == 0:
            self.load_warp(
                bar_full_a,
                bar_empty_a,
                cute.local_tile(mA, (self.cta_m, self.cta_k), (bidx, None, l_idx)),
                sA,
                cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A"),
                k_tile_start,
                k_tile_count,
                True,
            )
        elif warp_idx == 1:
            self.load_warp(
                bar_full_b,
                bar_empty_b,
                cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx)),
                sB,
                cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B"),
                k_tile_start,
                k_tile_count,
                False,
            )
        elif warp_idx == 2:
            self.resident_mma_warp(
                bar_full_a,
                bar_empty_a,
                bar_full_b,
                bar_empty_b,
                bar_mma_epilog,
                bar_tmem_alloc,
                tiled_mma,
                sA,
                sB,
                tmem_base_ptr,
                acc_layout,
                self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2]),
                k_tile_count,
            )
        elif warp_idx >= 4:
            gD_tile = cute.local_tile(mC, c_tiler_mn, (bidx, n_idx, l_idx))
            epi_tid = cute.arch.thread_idx()[0] - 128
            self.split_epilog_warp(
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                gD_tile,
                epi_tid,
                mC.element_type,
                utils.LayoutEnum.from_tensor(mC),
                mailbox,
                bar_reduce,
                split_rank,
            )

    def _barriers(self, count: int):
        """``count`` mbarriers in shared memory (allocated while the kernel is traced)."""
        return cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(count),
            alignment=_MBARRIER_BYTES,
        ).iterator

    @cute.experimental.jit
    def load_warp(
        self,
        bar_full,
        bar_empty,
        g_tile: cute.Tensor,
        s_tile: cute.Tensor,
        cta_v_map: cute.Layout,
        k_tile_start: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        is_weight: cutlass.Constexpr,
    ):
        """The weight warp loads its first ``a_stages`` K tiles at once without waiting for the
        grid, then refills stages from HBM as the MMA frees them. The activation warp waits for
        the grid, then streams its ring."""
        stages = self.a_stages if is_weight else self.b_stages
        if cutlass.const_expr(not is_weight):
            cute.arch.griddepcontrol_wait()

        stage_bytes = cute.size_in_bytes(
            s_tile.element_type, cute.slice_(s_tile.layout, (None, None, None, 0))
        )
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bar_full + stage, stage_bytes)
            cute_ext.tma_load(
                g_tile[None, None, k_tile_start + k_tile],
                s_tile[None, None, None, stage],
                (bar_full + stage).value,
                cta_v_map=cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

        for tail in cutlass.range(stages, unroll=1):
            stage = (tail + k_tile_count) % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

    @cute.experimental.jit
    def resident_mma_warp(
        self,
        bar_full_a,
        bar_empty_a,
        bar_full_b,
        bar_empty_b,
        bar_mma_epilog,
        bar_tmem_alloc,
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,
        sB: cute.Tensor,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        mma_inst_tile_k: cutlass.Constexpr,
        k_tile_count: cutlass.Int32,
    ):
        num_tmem_cols = 256
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_ptr, is_two_cta=False)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        accumulator = cute.make_tensor(tmem_ptr, acc_layout)[None, None, None, 0]
        mma_atom = cute.make_mma_atom(tiled_mma.op)

        phase_a = cutlass.Int32(0)
        phase_b = cutlass.Int32(0)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage_a = k_tile % self.a_stages
            stage_b = k_tile % self.b_stages
            cute.arch.mbarrier_wait(bar_full_a + stage_a, phase_a)
            cute.arch.mbarrier_wait(bar_full_b + stage_b, phase_b)
            for k_block in range(mma_inst_tile_k):
                if k_block == 0:
                    mma_atom.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                else:
                    mma_atom.set(tcgen05.Field.ACCUMULATE, True)
                cute_ext.dot(
                    mma_atom,
                    cute.append_ones(sA[None, None, k_block, stage_a], up_to_rank=3),
                    cute.append_ones(sB[None, None, k_block, stage_b], up_to_rank=3),
                    accumulator,
                )
            with cute.arch.elect_one():
                tcgen05.commit(bar_empty_a + stage_a, None, self.cta_group)
                tcgen05.commit(bar_empty_b + stage_b, None, self.cta_group)
            if stage_a == self.a_stages - 1:
                phase_a = phase_a ^ 1
            if stage_b == self.b_stages - 1:
                phase_b = phase_b ^ 1

        # Every MMA is issued: the next kernel's CTAs may start (its grid wait still waits for this
        # grid to finish).
        cute.arch.griddepcontrol_launch_dependents()
        with cute.arch.elect_one():
            tcgen05.commit(bar_mma_epilog, None, self.cta_group)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols, is_two_cta=False)

    @cute.experimental.jit
    def split_epilog_warp(
        self,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,
        epi_tid: cutlass.Int32,
        c_dtype: cutlass.Constexpr,
        d_layout: cutlass.Constexpr,
        mailbox,
        bar_reduce,
        split_rank: cutlass.Int32,
    ):
        """FlashInfer's split-K epilogue (the peers' FP32 partials into rank 0's mailbox,
        added there in rank order, one rounding), with the mailbox words of (thread, value) in
        thread-interleaved groups of four, so the peers' 16-byte stores and rank 0's loads cover
        consecutive banks."""
        # Wait until the MMA warp publishes the TMEM base pointer.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)

        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
            acc_layout,
        )[((None, None), 0, 0, 0)]
        epi_tile = (self.cta_m, self.cta_n)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.cta_m, self.cta_n, self.cta_k),
                d_layout,
                c_dtype,
                self.acc_dtype,
                epi_tile,
                False,
            ),
            acc_view,
        )
        gD_epi = cute.flat_divide(gD_tile, epi_tile)
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rAcc = cute_ext.allocate(self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        rD = cute_ext.allocate(c_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)
        values_per_thread = cutlass.const_expr(cute.size(rmem_layout))
        values_per_peer = cutlass.const_expr(self.epilog_threads * values_per_thread)
        group_words = cutlass.const_expr(self.epilog_threads * 4)

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        if split_rank != 0:
            for value_idx in cutlass.range_constexpr(0, values_per_thread, 4):
                _sk._store_shared_remote_v4(
                    rAcc[value_idx],
                    rAcc[value_idx + 1],
                    rAcc[value_idx + 2],
                    rAcc[value_idx + 3],
                    mailbox.iterator
                    + (split_rank - Int32(1)) * values_per_peer
                    + (value_idx // 4) * group_words
                    + epi_tid * 4,
                    bar_reduce,
                    Int32(0),
                )
        else:
            if epi_tid == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(bar_reduce, self.expected_transaction_bytes)
            cute.arch.mbarrier_wait(bar_reduce, 0)
            for peer in cutlass.range_constexpr(self.split_k - 1):
                for value_idx in cutlass.range_constexpr(values_per_thread):
                    rAcc[value_idx] = (
                        rAcc[value_idx]
                        + mailbox[
                            peer * values_per_peer
                            + (value_idx // 4) * group_words
                            + epi_tid * 4
                            + value_idx % 4
                        ]
                    )
            rD.store(rAcc.load().to(c_dtype))
            # Preserve TMEM coordinates; the copy predicates output tails.
            cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])


@cute.experimental.jit
def _bmm_resident(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    stream: _cuda.CUstream,
):
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        c,
        cute.make_tensor(c.iterator, cute.select(c.layout, mode=[0, 1, 2])),
        stream,
    )


@functools.cache
def _compiled(
    config: ResidentConfig, smem_launch_bytes: int, leading_dims: Tuple[int, int, int], device: int
):
    """The kernel compiled for ``config`` on ``device`` (one JIT per key; before graph capture)."""
    kernel = LatentUpResidentKernel(config, smem_launch_bytes)
    compile_tensors = _sk._make_compile_repr_tensors(torch.bfloat16, False, *leading_dims)
    stream = _cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute_ext.compile(_bmm_resident, kernel, *compile_tensors[:3], stream)


def check_operands(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> None:
    """Raise ValueError unless ``x [M, K]``, ``w [N, K]`` and ``out [M, N]`` are contiguous BF16 on
    one CUDA device, 32-byte aligned, ``N % 128 == 0`` and ``1 <= M <= MAX_M``."""
    tensors = (x, w, out)
    if any(t.dtype != torch.bfloat16 or t.dim() != 2 for t in tensors):
        raise ValueError("x, w and out must be 2-D BF16")
    if any(not t.is_cuda or t.device != x.device for t in tensors):
        raise ValueError("x, w and out must be on one CUDA device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("x, w and out must be contiguous row-major")
    if any(t.data_ptr() % 32 for t in tensors):
        raise ValueError("x, w and out must be 32-byte aligned")
    (m, k), (n, kw) = x.shape, w.shape
    if kw != k or tuple(out.shape) != (m, n):
        raise ValueError(f"shapes x {tuple(x.shape)}, w {tuple(w.shape)}, out {tuple(out.shape)}")
    if n % CTA_M:
        raise ValueError(f"N={n} must be a multiple of {CTA_M}")
    if not 1 <= m <= MAX_M:
        raise ValueError(f"M={m} outside 1..{MAX_M}")


def run_latent_up_resident(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    *,
    smem_budget: int = DEFAULT_SMEM_BUDGET,
) -> torch.Tensor:
    """``out = x @ w.T`` on the current stream.

    Args:
        x: ``[M, K]`` BF16 activation, contiguous (``1 <= M <= 128``).
        w: ``[N, K]`` BF16 weight, contiguous (``N % 128 == 0``).
        out: ``[M, N]`` BF16 output, contiguous; only its ``M x N`` values are stored.
        smem_budget: Shared memory per CTA the resident stages may use.

    The kernel launches as a programmatic (PDL) dependent of the previous kernel on the stream;
    after a kernel that is not one, its grid wait returns at once.

    Returns:
        ``out``.
    """
    check_operands(x, w, out)
    config = select_config(x.shape[0], x.shape[1], smem_budget)
    cute_tensors = _sk._to_cute_swap(x, w.t(), out, None)
    compiled = _compiled(config, launch_smem_bytes(config), cute_tensors[4], x.device.index)
    compiled(
        *cute_tensors[:3],
        _cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
    )
    return out


__all__ = [
    "DEFAULT_SMEM_BUDGET",
    "MAX_M",
    "SMEM_CAPACITY_BYTES",
    "SPLIT_K",
    "LatentUpResidentKernel",
    "ResidentConfig",
    "check_operands",
    "k_chunks",
    "launch_smem_bytes",
    "run_latent_up_resident",
    "select_config",
    "smem_bytes",
]
