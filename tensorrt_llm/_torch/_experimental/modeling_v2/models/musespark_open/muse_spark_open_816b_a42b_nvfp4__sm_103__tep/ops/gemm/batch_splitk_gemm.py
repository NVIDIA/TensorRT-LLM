# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blackwell BF16 cluster split-K GEMM for decode steps of 33..256 rows (``batch_splitk``).

flashinfer's low-M split-K kernel (``flashinfer/gemm/kernels/dense_bf16_gemm_sm100_splitk.py``,
flashinfer-python 0.6.18; ``dense_gemm``'s ``splitk`` family) serves at most 32 public rows: its
kernel-N tile, which carries the public rows, is 8, 16 or 32 wide, and its owner CTA receives every
peer's whole FP32 tile through DSMEM. Above 32 rows the built-in ``Linear`` sends a TP8 rank's
projections to cuBLAS, which picks a split-K kernel plus a separate ``cublasLt::splitKreduce``
launch for most of them.

This is the same kernel with the public rows in one kernel-N tile of up to 256 (any multiple of 8
at ``mma_m`` 64, of 16 at ``mma_m`` 128: the tcgen05 MMA shapes), so each weight slab is streamed
from HBM once whatever the row count, and with the reduce-scatter epilogue of this target's
upstream copy of that kernel (TensorRT-LLM ``feat/musespark-open-agentx``,
``muse_spark_open_816b_a42b_nvfp4__sm_103__tep/ops/gemm/splitk_gemm.py``, ``rs_split_k``): each of
the ``split_k`` CTAs of a cluster owns ``128 / split_k`` of the epilogue's threads, receives those
threads' FP32 partials from the other ranks through DSMEM, sums the ``split_k`` partials in
cluster-rank order, rounds once to BF16 and stores them. A CTA's mailbox is ``(split_k - 1) /
split_k`` of a tile, so the wide tiles keep a deep pipeline, and no separate reduction kernel runs.
It also carries flashinfer's ordering fixes of the TMEM and DSMEM hand-offs (flashinfer PR #7443,
854a896b: ``tcgen05.fence`` around the mbarrier hand-offs, the TMEM release arrival in phase 1, an
``.acquire.cluster`` wait on the reduction barrier).

Kernel coordinates: the public ``x[M, K] @ weight[N, K]^T`` is swapped, so kernel-M carries the
public N (weight rows, ``mma_m``) and kernel-N the public M (rows, ``mma_n``). Each CTA of a
cluster of ``split_k`` accumulates one equal K slice in FP32 in TMEM.

Numerics: every output element is the FP32 sum of the ``split_k`` K-slice partials in cluster-rank
order, each slice accumulated by the tensor core tile by tile along K, rounded once to BF16. It is
deterministic and does not depend on the row count or ``mma_n`` (a row's sum never involves
another row), only on ``split_k``; it is not cuBLAS's summation order, so it can differ from the
built-in's output in the last BF16 place.

Stores: each output element is stored exactly once, by the CTA that owns it, after the
reduction, which follows the kernel's dependency wait (the activation is loaded only after
``griddepcontrol.wait``); there is no read-back of the output. So the output may be a slot of a
multicast transport, as flashinfer's split-K kernel's output is (``unpool_direct``).

Programmatic dependent launch (``pdl``): the weight's TMA loads are issued before
``griddepcontrol.wait`` (immutable bytes), the activation's after it; the weight warp triggers
the dependents once it has issued its last load.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Optional, Tuple

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

#: Per-CTA shared memory CuTe DSL reports on SM100 / SM103.
SMEM_CAPACITY_BYTES = utils.get_smem_capacity_in_bytes("sm_100")
#: K extent of one CTA k-tile.
CTA_K = 128
#: Kernel-M tiles (public N per CTA).
SUPPORTED_MMA_M = (64, 128)
#: Kernel-N (public rows per CTA): multiples of these, up to ``MAX_MMA_N`` (tcgen05 MMA shapes).
MMA_N_STEP = {64: 8, 128: 16}
MAX_MMA_N = 256
#: Cluster sizes along K (portable cluster sizes; 128 epilogue threads split evenly).
SUPPORTED_SPLIT_K = (1, 2, 4, 8)
#: Largest public row count a call may have (several kernel-N tiles above ``mma_n``).
MAX_ROWS = 256
#: FP32 accumulator values each epilogue thread holds: at most this many (registers).
MAX_VALUES_PER_THREAD = 128
MAX_AB_STAGES = 16

_EPILOG_THREADS = 128
_FP32_BYTES = 4
_AB_ELEMENT_BYTES = 2
_AB_BUFFER_ALIGN_BYTES = 1024
_MAILBOX_ALIGN_BYTES = 128
_MBARRIER_BYTES = 8
_TMEM_POINTER_BYTES = 4
_TMEM_COLUMNS = 256


@dataclasses.dataclass(frozen=True)
class BatchSplitKTactic:
    """One specialization: ``mma_m`` weight rows x ``mma_n`` public rows per CTA, ``split_k`` CTAs
    per cluster along K, ``ab_stages`` k-tiles in flight."""

    mma_m: int
    mma_n: int
    split_k: int
    ab_stages: int


def parse(kernel: str) -> BatchSplitKTactic:
    """``"bsplitk:<mma_m>,<mma_n>,<split_k>,<stages>"`` -> the tactic.

    Raises:
        ValueError: for any other string.
    """
    family, _, rest = str(kernel).partition(":")
    fields = rest.split(",")
    if family == FAMILY and len(fields) == 4 and all(f.strip().isdigit() for f in fields):
        return BatchSplitKTactic(*(int(f) for f in fields))
    raise ValueError(
        f"unknown batch split-K kernel {kernel!r}; expected '{FAMILY}:<mma_m>,<mma_n>,<split_k>,<stages>'"
    )


#: The kernel string family (``parse``).
FAMILY = "bsplitk"


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def values_per_thread(tactic: BatchSplitKTactic) -> int:
    """FP32 accumulator values one epilogue thread holds (``mma_m * mma_n / 128``)."""
    return tactic.mma_m * tactic.mma_n // _EPILOG_THREADS


def mailbox_bytes(tactic: BatchSplitKTactic) -> int:
    """Bytes of FP32 partials a CTA receives: its thread group's values from each other rank."""
    if tactic.split_k == 1:
        return 0
    group = _EPILOG_THREADS // tactic.split_k
    return (tactic.split_k - 1) * group * values_per_thread(tactic) * _FP32_BYTES


def smem_bytes(tactic: BatchSplitKTactic, ab_stages: Optional[int] = None) -> int:
    """Shared memory the kernel's allocator lays out (A ring, B ring, barriers, mailbox)."""
    stages = tactic.ab_stages if ab_stages is None else ab_stages
    cursor = _align_up(tactic.mma_m * CTA_K * _AB_ELEMENT_BYTES * stages, _AB_BUFFER_ALIGN_BYTES)
    cursor += tactic.mma_n * CTA_K * _AB_ELEMENT_BYTES * stages
    cursor = _align_up(cursor, _MBARRIER_BYTES) + (2 * stages + 3) * _MBARRIER_BYTES
    cursor = _align_up(cursor, _TMEM_POINTER_BYTES) + _TMEM_POINTER_BYTES
    if tactic.split_k == 1:
        return cursor
    cursor = _align_up(cursor, _MAILBOX_ALIGN_BYTES) + mailbox_bytes(tactic)
    return _align_up(cursor, _MBARRIER_BYTES) + _MBARRIER_BYTES


def max_stages(tactic: BatchSplitKTactic, smem_capacity: int = SMEM_CAPACITY_BYTES) -> int:
    """The deepest ring that fits ``smem_capacity`` (0 if none)."""
    return next(
        (s for s in range(MAX_AB_STAGES, 0, -1) if smem_bytes(tactic, s) <= smem_capacity), 0
    )


def validate(
    tactic: BatchSplitKTactic, m: int, n: int, k: int, smem_capacity: int = SMEM_CAPACITY_BYTES
) -> None:
    """Reject a tactic that cannot serve public ``x[m, k] @ weight[n, k]^T``.

    Raises:
        ValueError: naming the first violated bound.
    """
    if tactic.mma_m not in SUPPORTED_MMA_M:
        raise ValueError(f"unsupported mma_m={tactic.mma_m}")
    step = MMA_N_STEP[tactic.mma_m]
    if not step <= tactic.mma_n <= MAX_MMA_N or tactic.mma_n % step:
        raise ValueError(
            f"mma_n={tactic.mma_n} must be a multiple of {step} in [{step}, {MAX_MMA_N}] at "
            f"mma_m={tactic.mma_m}"
        )
    if values_per_thread(tactic) > MAX_VALUES_PER_THREAD:
        raise ValueError(
            f"tile {tactic.mma_m}x{tactic.mma_n} gives {values_per_thread(tactic)} FP32 values per "
            f"epilogue thread (at most {MAX_VALUES_PER_THREAD})"
        )
    if tactic.split_k not in SUPPORTED_SPLIT_K:
        raise ValueError(f"unsupported split_k={tactic.split_k} (one of {SUPPORTED_SPLIT_K})")
    if not 1 <= tactic.ab_stages <= MAX_AB_STAGES:
        raise ValueError(f"ab_stages must be in [1, {MAX_AB_STAGES}], got {tactic.ab_stages}")
    if not 1 <= m <= MAX_ROWS:
        raise ValueError(f"this kernel serves 1 <= M <= {MAX_ROWS}, got {m}")
    if n <= 0:
        raise ValueError(f"N must be positive, got {n}")
    if k <= 0 or k % (CTA_K * tactic.split_k):
        raise ValueError(
            f"K={k} does not split into {tactic.split_k} slices of {CTA_K}-wide k-tiles"
        )
    if tactic.ab_stages > k // CTA_K // tactic.split_k:
        raise ValueError(
            f"ab_stages={tactic.ab_stages} exceeds the {k // CTA_K // tactic.split_k} k-tiles of a "
            "CTA's K slice"
        )
    need = smem_bytes(tactic)
    if need > smem_capacity:
        raise ValueError(
            f"tactic {tactic} needs {need} B of shared memory, {smem_capacity} B available; at most "
            f"{max_stages(tactic, smem_capacity)} stages fit"
        )


@dsl_user_op
def _map_shared_rank(smem_ptr, peer_cta_rank_in_cluster, *, loc=None, ip=None):
    """Map an SMEM pointer into a peer CTA's address space."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr.toint(loc=loc, ip=ip).ir_value(), peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _store_shared_remote_v4(
    value0,
    value1,
    value2,
    value3,
    smem_ptr,
    mbar_ptr,
    peer_cta_rank_in_cluster,
    *,
    loc=None,
    ip=None,
):
    """Publish four FP32 partials into a peer's SMEM, crediting 16 bytes to its mbarrier."""
    llvm.inline_asm(
        None,
        [
            _map_shared_rank(smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip).ir_value(),
            value0.bitcast(Int32).ir_value(loc=loc, ip=ip),
            value1.bitcast(Int32).ir_value(loc=loc, ip=ip),
            value2.bitcast(Int32).ir_value(loc=loc, ip=ip),
            value3.bitcast(Int32).ir_value(loc=loc, ip=ip),
            _map_shared_rank(mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip).ir_value(),
        ],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];",
        "r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _mbarrier_wait_acquire_cluster(mbar_ptr, phase, *, loc=None, ip=None) -> None:
    """``mbarrier.try_wait.parity`` with ``.acquire.cluster``: the peers' ``st.async`` complete-tx
    operations release at cluster scope (flashinfer #7443)."""
    llvm.inline_asm(
        None,
        [mbar_ptr.toint(loc=loc, ip=ip).ir_value(), phase.ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred P1;\n\tLAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [$0], $1, 10000000;\n\t"
        "@P1 bra DONE;\n\tbra LAB_WAIT;\n\tDONE:\n\t}",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _tcgen05_fence(*, before: bool, loc=None, ip=None) -> None:
    """``tcgen05.fence::{before,after}_thread_sync`` around an mbarrier hand-off of tcgen05 work."""
    llvm.inline_asm(
        None,
        [],
        f"tcgen05.fence::{'before' if before else 'after'}_thread_sync;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class BatchSplitKGemmKernel:
    """BF16 GEMM with a cluster-local reduce-scatter split-K reduction (see the module docstring)."""

    def __init__(self, *, tactic: BatchSplitKTactic, use_pdl: bool) -> None:
        self.acc_dtype = cutlass.Float32
        self.cta_m = tactic.mma_m
        self.cta_n = tactic.mma_n
        self.cta_k = CTA_K
        self.num_ab_stage = tactic.ab_stages
        self.split_k = tactic.split_k
        self.use_pdl = use_pdl
        self.threads_per_cta = 256
        self.epilog_threads = _EPILOG_THREADS
        self.mma_tiler_mn = (tactic.mma_m, tactic.mma_n)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.tma_op = cute_ext.OperationTypeEnum.SM90_TMA_LOAD
        self.cluster_shape = (1, tactic.split_k, 1)
        vpt = values_per_thread(tactic)
        if vpt % 4:
            raise ValueError(
                f"tile {tactic.mma_m}x{tactic.mma_n}: {vpt} values per thread, not a multiple of 4"
            )
        # Reduce-scatter: epilogue thread group g (rs_group threads) of every CTA belongs to cluster
        # rank g, which receives that group's values from the other split_k - 1 ranks.
        self.rs_group = self.epilog_threads // tactic.split_k
        self.rs_mailbox_elements = max(1, (tactic.split_k - 1) * self.rs_group * vpt)
        self.rs_expected_bytes = (tactic.split_k - 1) * self.rs_group * vpt * _FP32_BYTES

    @cute.experimental.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream: _cuda.CUstream):
        # Grid-y packs the kernel-N (public row) tile and the cluster rank.
        self.kernel(a, b, c).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(SMEM_CAPACITY_BYTES),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major: the weight
        mB: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major: the activation
        mC: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L), M-major: the output
    ):
        stages = self.num_ab_stage
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
            sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )
        acc_layout = cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)
        c_tiler_mn = (self.cta_m, self.cta_n)

        bar_full = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_empty = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_mma_epilog = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=_MBARRIER_BYTES
        ).iterator
        bar_tmem_alloc = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=_MBARRIER_BYTES
        ).iterator
        bar_spare = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=_MBARRIER_BYTES
        ).iterator
        tmem_base_ptr = cute_ext.allocate(
            cutlass.Int32,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_TMEM_POINTER_BYTES,
        ).iterator
        if cutlass.const_expr(self.split_k > 1):
            mailbox = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(self.rs_mailbox_elements),
                alignment=_MAILBOX_ALIGN_BYTES,
            )
            bar_reduce = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(1),
                alignment=_MBARRIER_BYTES,
            ).iterator
        else:
            # Dummy operands for the compile-time-elided reduction.
            mailbox = sA
            bar_reduce = bar_spare

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
                if cutlass.const_expr(self.split_k > 1):
                    # The receiving group's arrival plus the peers' transaction-byte credits.
                    cute.arch.mbarrier_init(bar_reduce, 1)
        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.split_k > 1):
            # Publish every CTA's barriers before any cross-CTA store.
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        # Host validation guarantees an equal, tail-free K partition.
        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        k_tile_start = split_rank * k_tile_count

        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_wait()

        # Warp 0 loads the weight, warp 1 the activation, warp 2 issues the MMAs, warp 3 idles,
        # warps 4-7 run the epilogue.
        if warp_idx == 0:
            self.dma_warp(
                bar_full,
                bar_empty,
                cute.local_tile(mA, (self.cta_m, self.cta_k), (bidx, None, l_idx)),
                sA,
                cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A"),
                k_tile_start,
                k_tile_count,
                True,
            )
        elif warp_idx == 1:
            self.dma_warp(
                bar_full,
                bar_empty,
                cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx)),
                sB,
                cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B"),
                k_tile_start,
                k_tile_count,
                False,
            )
        elif warp_idx == 2:
            self.mma_warp(
                bar_full,
                bar_empty,
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
            self.epilog_warp(
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                cute.local_tile(mC, c_tiler_mn, (bidx, n_idx, l_idx)),
                cute.arch.thread_idx()[0] - 128,
                mC.element_type,
                utils.LayoutEnum.from_tensor(mC),
                mailbox,
                bar_reduce,
                split_rank,
            )

    @cute.experimental.jit
    def dma_warp(
        self,
        bar_full,
        bar_empty,
        g_tile: cute.Tensor,
        s_tile: cute.Tensor,
        cta_v_map: cute.Layout,
        k_tile_start: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        is_a: cutlass.Constexpr,
    ):
        stages = self.num_ab_stage
        if cutlass.const_expr(not is_a and self.use_pdl):
            # The activation is the predecessor's output: load it only after the dependency wait.
            cute.arch.griddepcontrol_wait()
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    cute.size_in_bytes(
                        s_tile.element_type, cute.slice_(s_tile.layout, (None, None, None, 0))
                    ),
                )
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
        if cutlass.const_expr(is_a and self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        self._drain_producer(bar_empty, empty_phase, k_tile_count)

    @cute.experimental.jit
    def _drain_producer(self, bar_empty, empty_phase: cutlass.Int32, k_tile_count: cutlass.Int32):
        stages = self.num_ab_stage
        for tail in cutlass.range(stages, unroll=1):
            stage = (tail + k_tile_count) % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

    @cute.experimental.jit
    def mma_warp(
        self,
        bar_full,
        bar_empty,
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
        cute.arch.alloc_tmem(_TMEM_COLUMNS, tmem_base_ptr, is_two_cta=False)
        _tcgen05_fence(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        accumulator = cute.make_tensor(tmem_ptr, acc_layout)[None, None, None, 0]
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        full_phase = cutlass.Int32(0)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % self.num_ab_stage
            cute.arch.mbarrier_wait(bar_full + stage, full_phase)
            _tcgen05_fence(before=False)
            for k_block in range(mma_inst_tile_k):
                if k_block == 0:
                    mma_atom.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                else:
                    mma_atom.set(tcgen05.Field.ACCUMULATE, True)
                cute_ext.dot(
                    mma_atom,
                    cute.append_ones(sA[None, None, k_block, stage], up_to_rank=3),
                    cute.append_ones(sB[None, None, k_block, stage], up_to_rank=3),
                    accumulator,
                )
            with cute.arch.elect_one():
                tcgen05.commit(bar_empty + stage, None, self.cta_group)
            if stage == self.num_ab_stage - 1:
                full_phase = full_phase ^ 1

        with cute.arch.elect_one():
            tcgen05.commit(bar_mma_epilog, None, self.cta_group)
        # The release arrival must land in phase 1 (flashinfer #7443).
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        _tcgen05_fence(before=False)
        cute.arch.dealloc_tmem(tmem_ptr, _TMEM_COLUMNS, is_two_cta=False)

    @cute.experimental.jit
    def epilog_warp(
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
        # Wait until the MMA warp publishes the TMEM base pointer.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        _tcgen05_fence(before=False)

        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr), acc_layout
        )[((None, None), 0, 0, 0)]
        epi_tile = (self.cta_m, self.cta_n)
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
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
        # Match each epilogue thread's TMEM partition in RMEM.
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rAcc = cute_ext.allocate(self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        rD = cute_ext.allocate(c_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        _tcgen05_fence(before=False)
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        # Make tcgen05.ld visible before the TMEM release and the RMEM use.
        cute.arch.fence_view_async_tmem_load()
        _tcgen05_fence(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        if cutlass.const_expr(self.split_k > 1):
            # Reduce-scatter: thread group g = epi_tid // rs_group of every CTA sends its FP32
            # partial to cluster rank g (unless it is rank g); rank g's group g sums the split_k
            # partials of its values in cluster-rank order, rounds once and stores them.
            vpt = cutlass.const_expr(cute.size(rmem_layout))
            group_values = cutlass.const_expr(self.rs_group * vpt)
            dest = epi_tid // self.rs_group
            lt = epi_tid % self.rs_group
            if dest != split_rank:
                # This sender's slot in rank dest's mailbox: ranks below dest keep their index,
                # ranks above shift down by one.
                j = split_rank - cutlass.select_(split_rank > dest, Int32(1), Int32(0))
                for value_idx in cutlass.range_constexpr(0, vpt, 4):
                    slot = (value_idx // 4) * (self.rs_group * 4) + lt * 4
                    _store_shared_remote_v4(
                        rAcc[value_idx],
                        rAcc[value_idx + 1],
                        rAcc[value_idx + 2],
                        rAcc[value_idx + 3],
                        mailbox.iterator + j * group_values + slot,
                        bar_reduce,
                        dest,
                    )
            else:
                if lt == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(bar_reduce, self.rs_expected_bytes)
                _mbarrier_wait_acquire_cluster(bar_reduce, Int32(0))
                for value_idx in cutlass.range_constexpr(vpt):
                    slot = (value_idx // 4) * (self.rs_group * 4) + lt * 4 + value_idx % 4
                    own = rAcc[value_idx]
                    acc = own
                    for r in cutlass.range_constexpr(self.split_k):
                        # Rank r's partial: this CTA's own, or its slot in the mailbox (the index
                        # clamped where it is unused).
                        jr = cutlass.min(
                            Int32(r) - cutlass.select_(Int32(r) > split_rank, Int32(1), Int32(0)),
                            Int32(self.split_k - 2),
                        )
                        term = cutlass.select_(
                            Int32(r) == split_rank, own, mailbox[jr * group_values + slot]
                        )
                        if cutlass.const_expr(r == 0):
                            acc = term
                        else:
                            acc = acc + term
                    rAcc[value_idx] = acc
                rD.store(rAcc.load().to(c_dtype))
                # Preserve the TMEM coordinates; the copy predicates the output's tails.
                cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])
        else:
            rD.store(rAcc.load().to(c_dtype))
            cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])
        # The reduction mbarrier covers the remote stores; no cluster barrier is needed.


@cute.experimental.jit
def _bmm(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    stream: _cuda.CUstream,
):
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0])),
        stream,
    )


def _from_dlpack_dynamic(tensor: torch.Tensor, leading_dim: int, assumed_align: int = 32):
    return from_dlpack(tensor.detach(), assumed_align=assumed_align).mark_layout_dynamic(
        leading_dim=leading_dim
    )


def _detect_leading_dim(tensor: torch.Tensor) -> int:
    for dim, stride in enumerate(tensor.stride()[1:], start=1):
        if stride == 1:
            return dim
    raise ValueError("tensor has no stride-1 dimension")


def _make_layout_tensor(shape, dtype, leading_dim: int) -> torch.Tensor:
    permutation = [dim for dim in range(len(shape)) if dim != leading_dim] + [leading_dim]
    return torch.empty(tuple(shape[d] for d in permutation), dtype=dtype, device="cuda").permute(
        [permutation.index(d) for d in range(len(shape))]
    )


def _swapped(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor):
    """The kernel's operands: A = the weight [1, N, K], B = x as [1, K, M], C = out as [1, N, M]."""
    tensors = (
        weight.unsqueeze(0),
        x.unsqueeze(0).transpose(-2, -1),
        out.unsqueeze(0).transpose(-2, -1),
    )
    leading = tuple(_detect_leading_dim(t) for t in tensors)
    return tuple(_from_dlpack_dynamic(t, d) for t, d in zip(tensors, leading)), leading


@functools.lru_cache(maxsize=None)
def _compiled(
    dtype, tactic: BatchSplitKTactic, use_pdl: bool, leading: Tuple[int, int, int], device: int
):
    # Compile-time stand-ins (flashinfer's): every extent and stride but the unit one is dynamic.
    m, n, k = 64, 8, CTA_K
    with torch.cuda.device(device):
        reprs = tuple(
            _from_dlpack_dynamic(_make_layout_tensor(shape, dtype, d), d)
            for shape, d in (
                ((1, n, k), leading[0]),
                ((1, k, m), leading[1]),
                ((1, n, m), leading[2]),
            )
        )
        return cute_ext.compile(
            _bmm,
            BatchSplitKGemmKernel(tactic=tactic, use_pdl=use_pdl),
            *reprs,
            _cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )


def run(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    tactic: BatchSplitKTactic,
    pdl: bool = True,
) -> torch.Tensor:
    """``out[M, N] = x[M, K] @ weight[N, K]^T`` in BF16: row-major, contiguous, 32-byte aligned.

    Raises:
        ValueError: for operands or a tactic the kernel does not take.
    """
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError("batch split-K takes BF16 operands only")
    if x.dim() != 2 or weight.dim() != 2 or out.dim() != 2:
        raise ValueError("batch split-K takes 2-D x [M, K], weight [N, K], out [M, N]")
    m, k = (int(d) for d in x.shape)
    n = int(weight.shape[0])
    if int(weight.shape[1]) != k or tuple(out.shape) != (m, n):
        raise ValueError(
            f"shapes x {tuple(x.shape)}, weight {tuple(weight.shape)}, out {tuple(out.shape)}"
        )
    if not (x.is_contiguous() and weight.is_contiguous() and out.is_contiguous()):
        raise ValueError("batch split-K takes contiguous row-major operands")
    if any(t.data_ptr() % 32 for t in (x, weight, out)) or not (
        x.is_cuda and weight.device == x.device == out.device
    ):
        raise ValueError("batch split-K takes 32-byte aligned CUDA operands on one device")
    validate(tactic, m, n, k)
    with torch.cuda.device(x.device):
        tensors, leading = _swapped(x, weight, out)
        compiled = _compiled(x.dtype, tactic, bool(pdl), leading, x.get_device())
        compiled(*tensors, _cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream))
    return out
