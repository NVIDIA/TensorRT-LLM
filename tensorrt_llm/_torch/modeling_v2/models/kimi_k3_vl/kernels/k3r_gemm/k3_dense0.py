# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3's dense MLP (layer 0) on one rank's slice of a ``world``-rank tensor-parallel split, as
two low-M BF16 split-K GEMMs (:mod:`low_m_bf16_splitk`):

* gate/up + SiTU (:class:`SituSplitKGemmKernel`): ``hidden[M, I] = situ(x @ Wg^T, x @ Wu^T)``.
  The weight is the slice's gate and up rows interleaved in 64-row tiles (32 gate rows, then the
  matching 32 up rows: :func:`fuse_gate_up`), so the owner CTA of a cluster (after the split-K
  reduction) holds both halves of 32 hidden columns. It rounds them to BF16 (the gate/up GEMM's
  output type), stages the tile token-major in shared memory and applies
  ``beta tanh(g / beta) sigmoid(g) * lb tanh(u / lb)`` in FP32 (``k3_shared_persist``'s device
  math: ``ex2.approx``), storing BF16 pairs into ``hidden``.
* down + push (:class:`FitPushGemmKernel`): ``hidden @ Wd^T`` pushed into the Lamport slots of an
  all-reduce (:class:`dspark_draft_gemm_push.SplitKArPushGemmKernel`, same arithmetic) whose
  consumer finishes it. ``K = I`` padded to a whole number of 128-wide K tiles: the padded weight
  columns are zero and the hidden buffer's padding columns stay zero (never written).

Both launch with the shared memory they use (not the whole carve-out), so their CTAs can share an
SM with a predecessor's. PDL: the weight warp streams its first stages before
``griddepcontrol.wait`` and the dependents are released right after the activation warp's wait
(``SplitKArPushGemmKernel.dma_warp``); the SiTU epilogue only writes ``hidden`` (after the MMA,
which follows the wait).

Not bitwise the module path (cuBLASLt split-K + the Triton SiTU + cuBLASLt + the C++ all-reduce
over the module's own split): different split and summation order, BF16 partials per rank.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass.cute import experimental as cute_ext

from ..k3r_spec.dspark_draft_gemm_push import (
    SplitKArPushGemmKernel,
    _bmm_push,
    _make_layout_tensor,
    _swapped,
    ensure_cute_runtime,
    smem_bytes,
    validate_push_tactic,
)
from ..k3r_comm.k3_lamport_ar import acc_bf16x2, pack_bf16x2, st_u32
from ..k3r_moe.k3_moe_tail1 import _lds_b32
from ..k3r_moe.k3_shared_persist import _div, _mul, _sigmoid, _tanh
from .low_m_bf16_splitk import (
    _CTA_K,
    _SMEM_CAPACITY_BYTES,
    MAX_M,
    OWNER_RANK,
    SplitKTactic,
    _from_dlpack_dynamic,
    _store_shared_remote_v4,
    validate_tactic,
)

__all__ = [
    "GU_TILE",
    "HIDDEN",
    "MAX_M",
    "FitPushGemmKernel",
    "SituSplitKGemmKernel",
    "default_tactics",
    "fuse_gate_up",
    "launch_smem",
    "pad_k",
    "parse_tactic",
    "run_down",
    "run_down_push",
    "run_situ_gate_up",
    "slice_weights",
    "validate_tactics",
]

#: Hidden size the kernels serve.
HIDDEN = 7168
#: Weight rows per gate/up tile: 32 gate rows, then the matching 32 up rows.
GU_TILE = 64
_HALF = GU_TILE // 2
#: Launch shared memory above the allocator's layout (``smem_bytes``), for alignment slack.
_SMEM_SLACK = 1024


def pad_k(k: int) -> int:
    """``k`` rounded up to whole 128-wide K tiles."""
    return (k + _CTA_K - 1) // _CTA_K * _CTA_K


def launch_smem(tactic: SplitKTactic) -> int:
    """Dynamic shared memory the kernels launch with: what they allocate, plus slack."""
    return min(smem_bytes(tactic) + _SMEM_SLACK, _SMEM_CAPACITY_BYTES)


def parse_tactic(text: str) -> SplitKTactic:
    """``"mma_m.mma_n.split_k.stages"`` -> :class:`SplitKTactic`."""
    parts = [int(p) for p in text.split(".")]
    if len(parts) != 4:
        raise ValueError(f"tactic {text!r} is not mma_m.mma_n.split_k.stages")
    return SplitKTactic(*parts)


def default_tactics(rows: int) -> Tuple[SplitKTactic, SplitKTactic]:
    """``(gate/up, down)`` tactics for ``rows`` token rows (TP16 slice: gate/up N 4224 K 7168,
    down N 7168 K 2176). The tiles of vLLM's Rubin TP16 dense MLP (``moe/dense.py``: gate/up
    64-row tiles split 4 / 2 ways, the ring down 64.8.1.10 at up to 8 rows)."""
    if rows <= 8:
        return SplitKTactic(64, 8, 4, 8), SplitKTactic(64, 8, 1, 10)
    if rows <= 16:
        return SplitKTactic(64, 16, 2, 8), SplitKTactic(128, 16, 1, 6)
    return SplitKTactic(64, 32, 2, 8), SplitKTactic(128, 32, 1, 5)


def validate_tactics(rows: int, inter: int, gu: SplitKTactic, down: SplitKTactic) -> None:
    """Reject tactics the kernels cannot run for ``rows`` rows of an ``inter``-column slice."""
    if gu.mma_m != GU_TILE or gu.mma_n % 8:
        raise ValueError(f"gate/up tactic {gu}: mma_m must be {GU_TILE}, mma_n a multiple of 8")
    validate_tactic(gu, rows, 2 * inter, HIDDEN)
    if launch_smem(gu) < smem_bytes(gu):
        raise ValueError(f"gate/up tactic {gu} exceeds shared memory with its staging tile")
    validate_push_tactic(down, rows, HIDDEN, pad_k(inter))
    if launch_smem(down) < smem_bytes(down):
        raise ValueError(f"down tactic {down} exceeds shared memory with its staging tile")


# ---------------------------------------------------------------------------
# Host-side weight layout
# ---------------------------------------------------------------------------


def fuse_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``[I, K]`` gate rows and ``[I, K]`` up rows -> ``[2 I, K]`` in 64-row tiles of 32 gate rows
    then their 32 up rows (``I`` a multiple of 32)."""
    inter, k = gate.shape
    if up.shape != gate.shape or inter % _HALF:
        raise ValueError(f"gate {tuple(gate.shape)} / up {tuple(up.shape)}: not [32 n, K] pairs")
    tiles = inter // _HALF
    pairs = torch.stack((gate.reshape(tiles, _HALF, k), up.reshape(tiles, _HALF, k)), dim=1)
    return pairs.reshape(2 * inter, k).contiguous()


def slice_weights(
    gate_up: torch.Tensor, down: torch.Tensor, index: int, count: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sub-slice ``index`` of ``count`` of a tensor-parallel GatedMLP shard: ``gate_up``
    ``[2 I, H]`` (gate rows then up rows) and ``down`` ``[H, I]`` -> the kernels' fused gate/up
    ``[2 I / count, H]`` and K-padded down ``[H, pad_k(I / count)]`` (new tensors)."""
    rows, hidden = gate_up.shape
    inter = rows // 2
    if rows % 2 or down.shape != (hidden, inter) or inter % count:
        raise ValueError(f"gate_up {tuple(gate_up.shape)} / down {tuple(down.shape)} / {count}")
    sub = inter // count
    lo, hi = index * sub, (index + 1) * sub
    fused = fuse_gate_up(gate_up[lo:hi], gate_up[inter + lo : inter + hi])
    padded = torch.zeros(hidden, pad_k(sub), dtype=down.dtype, device=down.device)
    padded[:, :sub].copy_(down[:, lo:hi])
    return fused, padded


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


class FitPushGemmKernel(SplitKArPushGemmKernel):
    """:class:`SplitKArPushGemmKernel` (same arithmetic, same pushed bits) launched with the shared
    memory it uses."""

    def __init__(self, *, tactic: SplitKTactic, **kwargs) -> None:
        super().__init__(tactic=tactic, **kwargs)
        self.launch_smem = launch_smem(tactic)

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        peer_ptrs: Int64,
        mc_base: Int64,
        flags: Int64,
        rank: Int32,
        m: Int32,
        stream: _cuda.CUstream,
    ):
        self.kernel(a, b, c, peer_ptrs, mc_base, flags, rank, m).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.launch_smem),
            stream=stream,
            use_pdl=self.use_pdl,
        )


class SituSplitKGemmKernel(FitPushGemmKernel):
    """The gate/up GEMM with the SiTU epilogue (module docstring).

    Launch arguments of :class:`SplitKArPushGemmKernel`, reinterpreted: ``mc_base`` is the address
    of ``hidden`` (BF16 ``[M, pitch]``), ``rank`` its row pitch in elements, ``m`` the rows;
    ``peer_ptrs`` / ``flags`` are unused; ``c`` only sets the grid (``[2 I, M]``, never written).
    """

    def __init__(
        self, *, tactic: SplitKTactic, use_pdl: bool, beta: float, linear_beta: Optional[float]
    ) -> None:
        if tactic.mma_m != GU_TILE:
            raise ValueError(f"the SiTU epilogue needs {GU_TILE}-row tiles, got {tactic}")
        super().__init__(
            tactic=tactic,
            use_pdl=use_pdl,
            world=1,
            hidden=HIDDEN,
            one_shot=True,
            store_out=False,
            push=True,
        )
        if self.epilog_threads != 128 or self.cta_n % 8:
            raise ValueError("the SiTU epilogue maps 128 threads to 8 tokens x 32 columns")
        self.beta = float(beta)
        self.lbeta = None if linear_beta is None else float(linear_beta)

    @cute.experimental.jit
    def push_epilog_warp(
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
        stage: cute.Tensor,
        bidx: cutlass.Int32,
        n_idx: cutlass.Int32,
        peer_ptrs: Int64,
        mc_base: Int64,
        flags: Int64,
        rank: Int32,
        m: Int32,
    ):
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
            acc_layout,
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
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rAcc = cute_ext.allocate(self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        rD = cute_ext.allocate(c_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        if cutlass.const_expr(self.split_k > 1):
            values_per_thread = cutlass.const_expr(cute.size(rmem_layout))
            values_per_peer = cutlass.const_expr(self.epilog_threads * values_per_thread)
            if split_rank != OWNER_RANK:
                for value_idx in cutlass.range_constexpr(0, values_per_thread, 4):
                    _store_shared_remote_v4(
                        rAcc[value_idx],
                        rAcc[value_idx + 1],
                        rAcc[value_idx + 2],
                        rAcc[value_idx + 3],
                        mailbox.iterator
                        + (split_rank - Int32(1)) * values_per_peer
                        + epi_tid * values_per_thread
                        + value_idx,
                        bar_reduce,
                        Int32(OWNER_RANK),
                    )
            else:
                if epi_tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bar_reduce, self.expected_transaction_bytes
                    )
                cute.arch.mbarrier_wait(bar_reduce, 0)
                for peer in cutlass.range_constexpr(self.split_k - 1):
                    for value_idx in cutlass.range_constexpr(values_per_thread):
                        rAcc[value_idx] = (
                            rAcc[value_idx]
                            + mailbox[
                                peer * values_per_peer + epi_tid * values_per_thread + value_idx
                            ]
                        )

        if split_rank == OWNER_RANK:
            # BF16 gate/up tile (the GEMM's output type), token-major in shared memory.
            rD.store(rAcc.load().to(c_dtype))
            cute_ext.partition_and_copy(thr_t2r, rD, stage)
            cute.arch.barrier(barrier_id=1, number_of_threads=self.epilog_threads)
            stage_base = stage.iterator.toint()
            # Thread: two adjacent hidden columns of one token per pass, 8 tokens per pass.
            col = (epi_tid % 16) * 2
            for it in cutlass.range_constexpr(self.cta_n // 8):
                t = epi_tid // 16 + it * 8
                tok = n_idx * self.cta_n + t
                if tok < m:
                    g = acc_bf16x2(
                        _lds_b32(stage_base + (t * self.cta_m + col) * 2),
                        Float32(0.0),
                        Float32(0.0),
                    )
                    u = acc_bf16x2(
                        _lds_b32(stage_base + (t * self.cta_m + _HALF + col) * 2),
                        Float32(0.0),
                        Float32(0.0),
                    )
                    beta = Float32(self.beta)
                    a0 = _mul(_mul(beta, _tanh(_div(g[0], beta))), _sigmoid(g[0]))
                    a1 = _mul(_mul(beta, _tanh(_div(g[1], beta))), _sigmoid(g[1]))
                    u0 = u[0]
                    u1 = u[1]
                    if cutlass.const_expr(self.lbeta is not None):
                        lb = Float32(self.lbeta)
                        u0 = _mul(lb, _tanh(_div(u[0], lb)))
                        u1 = _mul(lb, _tanh(_div(u[1], lb)))
                    st_u32(
                        mc_base + Int64(tok * rank + bidx * _HALF + col) * 2,
                        pack_bf16x2(_mul(a0, u0), _mul(a1, u1)),
                    )


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def _fake_operands(dtype):
    """Compile-time stand-ins of the swapped ``x [M, K] @ w[N, K]^T`` operands (dynamic layouts):
    A = w (batch, N, K) K-major, B = x^T (batch, K, M) K-major, C = out^T (batch, N, M) N-major."""
    m, n, k = 8, 64, 128
    return tuple(
        _from_dlpack_dynamic(_make_layout_tensor(shape, dtype, leading), leading)
        for shape, leading in (((1, n, k), 2), ((1, k, m), 1), ((1, n, m), 1))
    )


@functools.cache
def _compiled_situ(dtype, tactic: SplitKTactic, use_pdl: bool, beta: float, linear_beta):
    kernel = SituSplitKGemmKernel(
        tactic=tactic, use_pdl=use_pdl, beta=beta, linear_beta=linear_beta
    )
    stream = _cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    a = Int64(1 << 20)
    return cute_ext.compile(
        _bmm_push, kernel, *_fake_operands(dtype), a, a, a, Int32(0), Int32(1), stream
    )


@functools.cache
def _compiled_push(dtype, tactic: SplitKTactic, use_pdl: bool, world: int, one_shot: bool,
                   store_out: bool, push: bool):  # fmt: skip
    kernel = FitPushGemmKernel(
        tactic=tactic,
        use_pdl=use_pdl,
        world=world,
        hidden=HIDDEN,
        one_shot=one_shot,
        store_out=store_out,
        push=push,
    )
    stream = _cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    a = Int64(1 << 20)
    return cute_ext.compile(
        _bmm_push, kernel, *_fake_operands(dtype), a, a, a, Int32(0), Int32(1), stream
    )


def _stream(device) -> _cuda.CUstream:
    return _cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


def run_situ_gate_up(
    x: torch.Tensor,
    weight: torch.Tensor,
    hidden: torch.Tensor,
    grid_out: torch.Tensor,
    tactic: SplitKTactic,
    beta: float,
    linear_beta: Optional[float],
    *,
    pdl: bool = True,
) -> torch.Tensor:
    """``hidden[:, :I] = situ(x @ Wg^T, x @ Wu^T)`` with ``weight`` the fused ``[2 I, H]`` gate/up
    rows (:func:`fuse_gate_up`); ``x`` ``[M, H]`` BF16 contiguous; ``hidden`` BF16 ``[M, P]``
    (``P >= I``, contiguous rows; columns ``>= I`` untouched); ``grid_out`` any BF16 ``[M, 2 I]``
    contiguous tensor (sets the launch, never written). Returns ``hidden``."""
    m, k = x.shape
    n = weight.shape[0]
    if k != HIDDEN or weight.shape[1] != k or n % GU_TILE:
        raise ValueError(f"x {tuple(x.shape)} / weight {tuple(weight.shape)}")
    if hidden.shape[0] != m or hidden.shape[1] < n // 2 or hidden.stride(1) != 1:
        raise ValueError(f"hidden {tuple(hidden.shape)} for {m} rows of {n // 2} columns")
    if grid_out.shape != (m, n) or not grid_out.is_contiguous():
        raise ValueError(f"grid_out {tuple(grid_out.shape)}, expected {(m, n)}")
    validate_tactic(tactic, m, n, k)
    ensure_cute_runtime()
    _compiled_situ(x.dtype, tactic, pdl, float(beta), linear_beta)(
        *_swapped(x, weight, grid_out),
        Int64(0),
        Int64(hidden.data_ptr()),
        Int64(0),
        Int32(hidden.stride(0)),
        Int32(m),
        _stream(x.device),
    )
    return hidden


def run_down(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, tactic: SplitKTactic, *,
    pdl: bool = True,
) -> torch.Tensor:  # fmt: skip
    """``out = x [M, K] @ weight[H, K]^T`` with the push kernel's store-only variant (the values the
    push variant pushes). Returns ``out``."""
    m, k = x.shape
    validate_push_tactic(tactic, m, weight.shape[0], k)
    ensure_cute_runtime()
    _compiled_push(x.dtype, tactic, pdl, 2, True, True, False)(
        *_swapped(x, weight, out),
        Int64(0),
        Int64(0),
        Int64(0),
        Int32(0),
        Int32(m),
        _stream(x.device),
    )
    return out


def run_down_push(
    x: torch.Tensor, weight: torch.Tensor, target, tactic: SplitKTactic, *, pdl: bool = True
) -> None:
    """Push ``x [M, K] @ weight[H, K]^T`` (BF16) into ``target`` (a ``LamportPushTarget`` of an
    all-reduce of ``[M, H]``; ``target.output`` sets the launch) and set ``target.pushed``."""
    m, k = x.shape
    validate_push_tactic(tactic, m, weight.shape[0], k)
    ensure_cute_runtime()
    ws = target.workspace
    _compiled_push(x.dtype, tactic, pdl, ws.world, target.one_shot, False, True)(
        *_swapped(x, weight, target.output),
        Int64(ws.peer_ptrs.data_ptr()),
        Int64(ws.mc_base),
        Int64(ws.flags.data_ptr()),
        Int32(ws.rank),
        Int32(m),
        _stream(x.device),
    )
    target.pushed = True
