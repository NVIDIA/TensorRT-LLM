# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The drafter MLP's ``gate_up_proj`` and ``silu(gate) * up`` in one launch.

``GatedMLP`` runs ``gate_up_proj`` (on the drafter's 9..32-row blocks flashinfer's cluster split-K
kernel at the fixed tactic of ``fixed_low_m_gemm``, ``drafter_fixed_gemms``) and then the built-in
Triton ``silu_and_mul_kernel`` (``modules/swiglu.py``) as its own launch before ``down_proj``.

This entry is that GEMM on ``batch_splitk_gemm``'s copy of the same kernel (same tiles, same
``split_k``, so the same bits: each output element is the FP32 sum of the ``split_k`` K-slice
partials in cluster-rank order, rounded once to BF16) over a row-permuted copy of the weight in
which gate row ``c`` and up row ``c`` are adjacent (``interleave``: rows ``2c``, ``2c + 1``). A
row permutation changes no output element's K-ordered sum. After the reduction the CTA that owns
a share of the tile rounds it to BF16 (the GEMM output's values), stages it widened back to FP32
(exact) in shared memory, and applies the Triton kernel's statements to each (gate, up) pair, as
the PTX that kernel compiles to (``SILU_PTX``: ``a * -log2(e)``, ``ex2.approx.f32``, ``+ 1``,
``div.full.f32``, ``* a``, ``b + 0.0``, the product), rounding the result to BF16 to nearest even
as its store does. So the output is bitwise ``silu_and_mul(fixed_low_m_gemm(x, weight))``. The
GEMM output itself is never stored.

Ownership: the reduce-scatter epilogue gives cluster rank ``r`` the values of epilogue thread group
``r`` (``128 / split_k`` threads, whole warps); a warp's TMEM sub-partition holds a contiguous
quarter of the tile's kernel-M rows, so rank ``r`` owns kernel-M rows ``[r M / s, (r + 1) M / s)``
of every public row: whole (gate, up) pairs, whatever the tactic. One interleaved copy serves
every tactic.

Programmatic dependent launch: as ``batch_splitk_gemm`` (weight before the wait, activation after
it, dependents released once the weight warp has issued its last load). The epilogue reads no
global memory; it stores after the reduction, which follows the dependency wait.

The SwiGLU statements are the ones upstream's sm107 drafter folds compile into their split-K
GEMM's epilogue (TensorRT-LLM ``feat/musespark-open-agentx`` #21754,
``muse_spark_open_816b_a42b_nvfp4__sm_107__tep/ops/gemm/dr3_fold_splitk_gemm.py``, ``SILU_PTX``;
flashinfer's split-K kernel at ``mma_n`` 8 / 16 with a whole-tile owner). This copy runs on the
reduce-scatter kernel at any ``mma_n`` the tactic tables use (up to 32 at 32 rows).
"""

from __future__ import annotations

import functools
from typing import Tuple

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Int32, Uint32
from cutlass.cute import experimental as cute_ext
from cutlass.cutlass_dsl import dsl_user_op

from ..allreduce.drafter_hop import _asm, _f32_asm, _i64, _pack_bf16x2, _u32
from . import batch_splitk_gemm as _bsk

#: Output columns per epilogue item (one 16-byte BF16 vector): 16 kernel-M rows (8 pairs).
_VECTOR = 8
_STAGE_ALIGN_BYTES = 128

#: The Triton ``silu_and_mul_kernel`` statements for one element (``swiglu_alpha`` 1.0,
#: ``swiglu_beta`` 0.0), as that kernel compiles: ``1.0 * a`` and the negation fold into one
#: multiply by ``-log2(e)``, ``exp`` is ``ex2.approx.f32``, ``1 / (1 + e)`` is ``div.full.f32``,
#: then ``a * sigmoid``, ``b + 0.0`` (keeps the sign of zero) and their product. No two of them
#: can contract (no product feeds an add). From upstream's ``SILU_PTX`` (#21754).
SILU_PTX = (
    "{\n"
    ".reg .f32 t, e, d, s, p, q;\n"
    "mul.f32 t, $1, 0fBFB8AA3B;\n"
    "ex2.approx.f32 e, t;\n"
    "add.f32 d, e, 0f3F800000;\n"
    "div.full.f32 s, 0f3F800000, d;\n"
    "mul.f32 p, $1, s;\n"
    "add.f32 q, $2, 0f00000000;\n"
    "mul.f32 $0, p, q;\n"
    "}"
)


@dsl_user_op
def _silu_and_mul(a, b, *, loc=None, ip=None):
    """``a * sigmoid(a) * (b + 0.0)`` in FP32, the Triton kernel's instructions."""
    return _f32_asm(SILU_PTX, (a, b), loc, ip)


@dsl_user_op
def _store_x4_if(address, enabled, w0, w1, w2, w3, *, loc=None, ip=None):
    """Plain 16-byte global store when ``enabled``; else nothing."""
    _asm(
        None,
        [_i64(address, loc, ip), _u32(enabled, loc, ip)]
        + [_u32(w, loc, ip) for w in (w0, w1, w2, w3)],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $1, 0;\n@p st.global.v4.b32 [$0], {$2, $3, $4, $5};\n}",
        "l,r,r,r,r,r",
        loc,
        ip,
    )


def interleave(weight: torch.Tensor) -> torch.Tensor:
    """The kernel's weight: ``[gate; up]`` rows ``[2d, K]`` -> ``[d, 2, K]`` (gate ``c``, up ``c``
    adjacent). Returns a new contiguous tensor (the module's weight is not modified)."""
    d = weight.shape[0] // 2
    return torch.stack((weight[:d], weight[d:]), dim=1).reshape(weight.shape).contiguous()


def stage_bytes(tactic: _bsk.BatchSplitKTactic) -> int:
    return tactic.mma_m * tactic.mma_n * 4


def smem_bytes(tactic: _bsk.BatchSplitKTactic, ab_stages: int | None = None) -> int:
    """``batch_splitk_gemm``'s layout plus the FP32 staging tile."""
    cursor = _bsk.smem_bytes(tactic, ab_stages)
    return _bsk._align_up(cursor, _STAGE_ALIGN_BYTES) + stage_bytes(tactic)


def fitted(tactic: _bsk.BatchSplitKTactic) -> _bsk.BatchSplitKTactic:
    """``tactic`` with as many of its stages as fit beside the staging tile (pipeline depth only:
    the stages do not change any output bit)."""
    stages = tactic.ab_stages
    while stages > 1 and smem_bytes(tactic, stages) > _bsk.SMEM_CAPACITY_BYTES:
        stages -= 1
    return _bsk.BatchSplitKTactic(tactic.mma_m, tactic.mma_n, tactic.split_k, stages)


def tactic_error(tactic: _bsk.BatchSplitKTactic, m: int, n: int, k: int) -> str | None:
    """Why the fused kernel cannot run ``x[m, k] @ weight[n, k]^T`` at ``tactic`` (stages as
    ``fitted``), or None. ``n`` is the GEMM's width (gate and up)."""
    tactic = fitted(tactic)
    try:
        _bsk.validate(tactic, m, n, k)
    except ValueError as error:
        return str(error)
    if tactic.split_k > 4:
        return f"split_k={tactic.split_k}: an owner group is under one warp"
    rows_owned = tactic.mma_m // tactic.split_k
    if rows_owned % (2 * _VECTOR):
        return (
            f"tactic {tactic}: a rank's {rows_owned} kernel-M rows are not whole 8-column vectors"
        )
    if n % tactic.mma_m:
        return f"mma_m={tactic.mma_m} does not tile the {n} rows"
    if smem_bytes(tactic) > _bsk.SMEM_CAPACITY_BYTES:
        return f"tactic {tactic} with its staging tile exceeds shared memory"
    return None


def items(tactic: _bsk.BatchSplitKTactic) -> Tuple[int, int, int]:
    """``(vectors per public row, items per tile, items per owner thread)`` of an owner group."""
    rows_owned = tactic.mma_m // tactic.split_k
    per_row = rows_owned // (2 * _VECTOR)
    total = per_row * tactic.mma_n
    group = 128 // tactic.split_k
    return per_row, total, (total + group - 1) // group


class GateUpSiluGemmKernel(_bsk.BatchSplitKGemmKernel):
    """``batch_splitk_gemm``'s kernel whose owners store ``silu(gate) * up`` of their share."""

    def __init__(self, *, tactic: _bsk.BatchSplitKTactic, use_pdl: bool) -> None:
        super().__init__(tactic=tactic, use_pdl=use_pdl)
        self.rows_owned = tactic.mma_m // tactic.split_k
        self.per_row, self.total_items, self.items = items(tactic)

    @cute.experimental.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream: _cuda.CUstream):
        # The parent's grid over the GEMM's kernel-M (the interleaved weight's rows: twice the
        # output's), its cluster, shared-memory request and PDL flag.
        self.kernel(a, b, c).launch(
            grid=(
                cute.ceil_div(a.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(_bsk.SMEM_CAPACITY_BYTES),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major: the interleaved weight
        mB: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major: the activation
        mC: cute.Tensor,  # (Gemm_M / 2, Gemm_N, Gemm_L), M-major: the SwiGLU output
    ):
        # batch_splitk_gemm.BatchSplitKGemmKernel.kernel, the staging tile and the epilogue aside.
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
            alignment=_bsk._AB_BUFFER_ALIGN_BYTES,
        )
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_bsk._AB_BUFFER_ALIGN_BYTES,
        )
        acc_layout = cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)

        bar_full = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_bsk._MBARRIER_BYTES,
        ).iterator
        bar_empty = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_bsk._MBARRIER_BYTES,
        ).iterator
        bar_mma_epilog = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_bsk._MBARRIER_BYTES,
        ).iterator
        bar_tmem_alloc = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_bsk._MBARRIER_BYTES,
        ).iterator
        bar_spare = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_bsk._MBARRIER_BYTES,
        ).iterator
        tmem_base_ptr = cute_ext.allocate(
            cutlass.Int32,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_bsk._TMEM_POINTER_BYTES,
        ).iterator
        if cutlass.const_expr(self.split_k > 1):
            mailbox = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(self.rs_mailbox_elements),
                alignment=_bsk._MAILBOX_ALIGN_BYTES,
            )
            bar_reduce = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(1),
                alignment=_bsk._MBARRIER_BYTES,
            ).iterator
        else:
            mailbox = sA
            bar_reduce = bar_spare
        # The owner's BF16-rounded GEMM values of the tile, as FP32, kernel-M fastest.
        stage = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.cta_m)),
            alignment=_STAGE_ALIGN_BYTES,
        )

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
                if cutlass.const_expr(self.split_k > 1):
                    cute.arch.mbarrier_init(bar_reduce, 1)
        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        k_tile_start = split_rank * k_tile_count

        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_wait()

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
            self.epilog_silu_warp(
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                cute.arch.thread_idx()[0] - 128,
                utils.LayoutEnum.from_tensor(mC),
                mailbox,
                bar_reduce,
                split_rank,
                stage,
                bidx,
                n_idx,
                l_idx,
                mC,
            )

    @cute.experimental.jit
    def epilog_silu_warp(
        self,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        epi_tid: cutlass.Int32,
        d_layout: cutlass.Constexpr,
        mailbox,
        bar_reduce,
        split_rank: cutlass.Int32,
        stage: cute.Tensor,
        bidx: cutlass.Int32,
        n_idx: cutlass.Int32,
        l_idx: cutlass.Int32,
        mC: cute.Tensor,
    ):
        # batch_splitk_gemm.BatchSplitKGemmKernel.epilog_warp up to the owner's rounding.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        _bsk._tcgen05_fence(before=False)

        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr), acc_layout
        )[((None, None), 0, 0, 0)]
        epi_tile = (self.cta_m, self.cta_n)
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.cta_m, self.cta_n, self.cta_k),
                d_layout,
                cutlass.BFloat16,
                self.acc_dtype,
                epi_tile,
                False,
            ),
            acc_view,
        )
        stage_epi = cute.flat_divide(stage, epi_tile)
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, stage_epi, epi_tid)
        rAcc = cute_ext.allocate(self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        rD = cute_ext.allocate(cutlass.BFloat16, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        dest = Int32(0)
        lt = epi_tid
        if cutlass.const_expr(self.split_k > 1):
            dest = epi_tid // self.rs_group
            lt = epi_tid % self.rs_group

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        _bsk._tcgen05_fence(before=False)
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        cute.arch.fence_view_async_tmem_load()
        _bsk._tcgen05_fence(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        if cutlass.const_expr(self.split_k > 1):
            vpt = cutlass.const_expr(cute.size(rmem_layout))
            group_values = cutlass.const_expr(self.rs_group * vpt)
            if dest != split_rank:
                j = split_rank - cutlass.select_(split_rank > dest, Int32(1), Int32(0))
                for value_idx in cutlass.range_constexpr(0, vpt, 4):
                    slot = (value_idx // 4) * (self.rs_group * 4) + lt * 4
                    _bsk._store_shared_remote_v4(
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
                _bsk._mbarrier_wait_acquire_cluster(bar_reduce, Int32(0))
                for value_idx in cutlass.range_constexpr(vpt):
                    slot = (value_idx // 4) * (self.rs_group * 4) + lt * 4 + value_idx % 4
                    own = rAcc[value_idx]
                    acc = own
                    for r in cutlass.range_constexpr(self.split_k):
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
                # The GEMM output's values (BF16), widened back exactly.
                rD.store(rAcc.load().to(cutlass.BFloat16))
                rAcc.store(rD.load().to(self.acc_dtype))
                cute_ext.partition_and_copy(thr_t2r, rAcc, stage_epi[None, None, 0, 0])
        else:
            rD.store(rAcc.load().to(cutlass.BFloat16))
            rAcc.store(rD.load().to(self.acc_dtype))
            cute_ext.partition_and_copy(thr_t2r, rAcc, stage_epi[None, None, 0, 0])

        if dest == split_rank:
            # The owner group's staged rows are complete for every one of its threads.
            cute.arch.barrier(barrier_id=1, number_of_threads=self.rs_group)
            self._silu(stage, lt, split_rank, bidx, n_idx, l_idx, mC)

    def _silu(self, stage, lt, split_rank, bidx, n_idx, l_idx, mC):
        """``silu(gate) * up`` of this thread's items: 8 output columns of one public row each,
        from 16 staged kernel-M rows (gate, up pairs); rows past the call's are not stored. Traced
        from the jit epilogue: its Python loops unroll."""
        rows = cute.size(mC, mode=[1])
        m_first = split_rank * self.rows_owned
        for i in range(self.items):
            item = lt + i * self.rs_group
            vec = item % self.per_row
            n = item // self.per_row
            t = n_idx * self.cta_n + n
            m0 = m_first + vec * (2 * _VECTOR)
            column = (bidx * self.cta_m + m0) // 2
            valid = Uint32(t < rows) & Uint32(item < self.total_items)
            # Clamp the stage coordinates of an item past the tile (never stored).
            n_safe = cutlass.min(n, Int32(self.cta_n - 1))
            words = []
            for k in range(_VECTOR // 2):
                lo = _silu_and_mul(stage[m0 + 4 * k, n_safe], stage[m0 + 4 * k + 1, n_safe])
                hi = _silu_and_mul(stage[m0 + 4 * k + 2, n_safe], stage[m0 + 4 * k + 3, n_safe])
                words.append(_pack_bf16x2(lo, hi))
            address = (
                mC.iterator + cute.crd2idx((column, cutlass.min(t, rows - 1), l_idx), mC.layout)
            ).toint()
            _store_x4_if(address, valid, words[0], words[1], words[2], words[3])


@cute.experimental.jit
def _bmm_silu(
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


@functools.lru_cache(maxsize=None)
def _compiled(
    tactic: _bsk.BatchSplitKTactic, use_pdl: bool, leading: Tuple[int, int, int], device: int
):
    m, n, k = 64, 8, _bsk.CTA_K
    dtype = torch.bfloat16
    with torch.cuda.device(device):
        reprs = tuple(
            _bsk._from_dlpack_dynamic(_bsk._make_layout_tensor(shape, dtype, d), d)
            for shape, d in (
                ((1, n, k), leading[0]),
                ((1, k, m), leading[1]),
                ((1, n, m), leading[2]),
            )
        )
        return cute_ext.compile(
            _bmm_silu,
            GateUpSiluGemmKernel(tactic=tactic, use_pdl=use_pdl),
            *reprs,
            _cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )


def run(
    x: torch.Tensor,
    weight: torch.Tensor,
    tactic: _bsk.BatchSplitKTactic,
    pdl: bool = True,
) -> torch.Tensor:
    """``silu(x @ gate.T) * (x @ up.T)`` -> ``[M, d]`` BF16 for an ``interleave``d weight ``[2d, K]``.

    Bitwise ``silu_and_mul(fixed_low_m_gemm(x, [gate; up], tactic))``. ``tactic`` is taken with its
    stages cut to fit the staging tile (``fitted``). The first call per tactic and layout compiles
    (CuTe-DSL JIT) and must not be inside a CUDA-graph capture.

    Raises:
        ValueError: for operands or a tactic the kernel does not take (``tactic_error``).
    """
    tactic = fitted(tactic)
    if not (x.dtype == weight.dtype == torch.bfloat16):
        raise ValueError("the SwiGLU epilogue takes BF16 operands only")
    if x.dim() != 2 or weight.dim() != 2 or int(weight.shape[1]) != int(x.shape[1]):
        raise ValueError(f"shapes x {tuple(x.shape)}, weight {tuple(weight.shape)}")
    if not (x.is_contiguous() and weight.is_contiguous()):
        raise ValueError("the SwiGLU epilogue takes contiguous operands")
    if any(t.data_ptr() % 32 for t in (x, weight)):
        raise ValueError("the SwiGLU epilogue takes 32-byte aligned operands")
    if not (x.is_cuda and weight.device == x.device):
        raise ValueError("the SwiGLU epilogue takes CUDA operands on one device")
    m, k = (int(s) for s in x.shape)
    n = int(weight.shape[0])
    error = tactic_error(tactic, m, n, k)
    if error is not None:
        raise ValueError(f"the SwiGLU epilogue cannot run {tactic}: {error}")
    out = torch.empty((m, n // 2), dtype=x.dtype, device=x.device)
    with torch.cuda.device(x.device):
        tensors, leading = _bsk._swapped(x, weight[: n // 2], out)
        # ``_swapped`` gives A from the half-height view; the kernel reads the whole weight.
        tensors = (_bsk._from_dlpack_dynamic(weight.unsqueeze(0), leading[0]),) + tensors[1:]
        compiled = _compiled(tactic, bool(pdl), leading, x.get_device())
        compiled(*tensors, _cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream))
    return out
