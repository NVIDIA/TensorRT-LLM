# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from FlashInfer (Apache-2.0; upstream notice: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.):
# flashinfer/gemm/kernels/dense_bf16_gemm_sm100_splitk.py
"""Low-M BF16 GEMM that pushes its output into a Lamport call: the K3.1 low-latency MoE entries on
BF16 latent weights (SM107).

The GEMM is the split-K kernel of ``spec/rubin_draft_gemm_push.py`` (``out[M, N] =
x[M, K] @ w[N, K]^T`` for ``M <= 32``: FlashInfer's dense BF16 kernel, the cluster split-K
reduction in FP32, one BF16 rounding). Only the owner CTA's push differs:

* ``AG``: row ``tok`` of the call's all-gather rows, which follow its ``m x W`` slots
  (``rubin_lamport_ar.ag_row_addr``), at feature ``col0 + j`` for output column ``j``: the
  column-parallel MoE tail.
* ``SLOT``: row ``tok`` of this rank's slot of a call over rows of ``row`` features, as
  ``rubin_lamport_ar.push_slot`` addresses it, at feature ``col0 + j``: slot ``(tok, rank)`` of
  every rank's buffer (the sharded MoE front's latent slots).

Both stores go through the multicast address (the one-shot transport).

Values are sanitized (``-0`` sent as ``+0``). The weight warp streams before
``griddepcontrol.wait``; the push reads the call's rotation state after it, as the base kernel
does. A finish of the call (``rubin_lamport_ar.run_lamport_finish``, or a mix that finishes it)
then gives the all-reduce of the pushed values.

:func:`run_ull_front_bf16` composes the sharded MoE front for a BF16 latent_down: this GEMM
pushes the rank's latent tiles into the front workspace's slots, the router logits are copied
into the logits rows of the call's half, and the front's route kernel (``rubin_ull_front``) polls
both and writes the routing and the NVFP4 latent. No torch op is registered here.
"""

from __future__ import annotations

import struct
from typing import Tuple

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Int32, Int64
from cutlass.cute import experimental as cute_ext
from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
    OWNER_RANK,
    SplitKTactic,
    _from_dlpack_dynamic,
    _store_shared_remote_v4,
)

from ....kernels.comm import rubin_lamport_ar as lamport
from ....kernels.moe.rubin_fused_router import sort_buffer_sizes
from ..spec.rubin_draft_gemm_push import (
    SplitKArPushGemmKernel,
    _bmm_push,
    _lds_v4,
    _make_layout_tensor,
    _swapped,
    ensure_cute_runtime,
    validate_push_tactic,
)
from . import rubin_ull_front as front

#: Push targets.
SLOT = "slot"
AG = "ag"
#: Largest rows of one call (the low-M tactics' range).
MAX_M = 32


class UllPushGemmKernel(SplitKArPushGemmKernel):
    """:class:`SplitKArPushGemmKernel` whose owner CTAs push into a ``SLOT`` or ``AG`` target at
    feature offset ``col0`` of rows of ``row`` features (see the module docstring). ``n`` is the
    GEMM's N: whole tiles of ``mma_m`` columns inside the row."""

    def __init__(
        self,
        *,
        tactic: SplitKTactic,
        world: int,
        n: int,
        row: int,
        col0: int,
        target: str,
    ) -> None:
        if target not in (SLOT, AG):
            raise ValueError(f"unknown push target {target!r}")
        if n % tactic.mma_m or col0 % 8 or row % 8 or col0 + n > row:
            raise ValueError(
                f"columns [{col0}, {col0 + n}) in rows of {row} features with tiles of "
                f"{tactic.mma_m} columns: whole tiles of 8-aligned features only"
            )
        super().__init__(
            tactic=tactic,
            use_pdl=True,
            world=world,
            hidden=row,
            one_shot=True,
        )
        self.row = row
        self.col0 = col0
        self.target = target

    def _push(self, mc_base, cur, nbytes, tok, rank, m, col, words):
        """Store eight sanitized BF16 values of row ``tok``, output column ``col``."""
        s = [lamport.sanitize_bf16x2(w) for w in words]
        feat = Int64(col + self.col0) * 2
        row_bytes = self.row * 2
        if self.target == AG:
            addr = (
                mc_base
                + Int64(cur) * Int64(nbytes)
                + Int64(m * self.world + tok) * row_bytes
                + feat
            )
            lamport.multimem_st_v4(addr, s[0], s[1], s[2], s[3])
        else:
            addr = (
                mc_base
                + Int64(cur) * Int64(nbytes)
                + Int64(tok * self.world + rank) * row_bytes
                + feat
            )
            lamport.multimem_st_v4(addr, s[0], s[1], s[2], s[3])

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
        """The base epilogue (TMEM load, split-K reduction into the owner) with this class's push:
        the feature offset and the target."""
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
            rD.store(rAcc.load().to(c_dtype))
            cute_ext.partition_and_copy(thr_t2r, rD, stage)
            # The four epilogue warps only.
            cute.arch.barrier(barrier_id=1, number_of_threads=self.epilog_threads)
            cute.arch.griddepcontrol_wait()
            state = lamport.ld_volatile_v4(flags)
            stage_base = stage.iterator.toint()
            vecs = self.cta_n * self.vecs_per_token
            for it in cutlass.range_constexpr(
                (vecs + self.epilog_threads - 1) // self.epilog_threads
            ):
                v = epi_tid + it * self.epilog_threads
                t = v // self.vecs_per_token
                col = (v % self.vecs_per_token) * 8
                tok = n_idx * self.cta_n + t
                if v < vecs:
                    if tok < m:
                        words = _lds_v4(stage_base + (t * self.cta_m + col) * 2)
                        self._push(
                            mc_base,
                            state[0],
                            state[2],
                            tok,
                            rank,
                            m,
                            bidx * self.cta_m + col,
                            words,
                        )


#: Compiled launches by their arguments.
_COMPILED: dict = {}


def _compiled(tactic, world, n, row, col0, target):
    key = (tactic, world, n, row, col0, target)
    fn = _COMPILED.get(key)
    if fn is None:
        fn = _compile(*key)
        _COMPILED[key] = fn
    return fn


def _compile(tactic, world, n, row, col0, target):
    kernel = UllPushGemmKernel(
        tactic=tactic,
        world=world,
        n=n,
        row=row,
        col0=col0,
        target=target,
    )
    # The swapped operands of ``x [M, K] @ w[N, K]^T``: A = w (batch, N, K) K-major,
    # B = x^T (batch, K, M) K-major, C = out^T (batch, N, M) N-major.
    m, k = 8, 128
    tensors = tuple(
        _from_dlpack_dynamic(_make_layout_tensor(shape, torch.bfloat16, leading), leading)
        for shape, leading in (((1, n, k), 2), ((1, k, m), 1), ((1, n, m), 1))
    )
    stream = _cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    a = Int64(1 << 20)
    return cute_ext.compile(_bmm_push, kernel, *tensors, a, a, a, Int32(0), Int32(1), stream)


def tactic_of(t: Tuple[int, int, int, int]) -> SplitKTactic:
    """The ``SplitKTactic`` of a ``(mma_m, mma_n, split_k, ab_stages)`` tuple."""
    return SplitKTactic(*t)


def check_push(m: int, n: int, k: int, tactic: SplitKTactic, row: int, col0: int) -> None:
    """Raise ``ValueError`` unless the kernel serves the call."""
    if not 1 <= m <= MAX_M:
        raise ValueError(f"rows {m} outside [1, {MAX_M}]")
    validate_push_tactic(tactic, m, n, k)
    if n % tactic.mma_m or col0 % 8 or row % 8 or col0 + n > row:
        raise ValueError(f"columns [{col0}, {col0 + n}) of rows of {row} features")


def run_ull_push(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    *,
    peer_ptrs: int,
    mc_base: int,
    flags: int,
    rank: int,
    world: int,
    tactic: SplitKTactic,
    target: str,
    row: int,
    col0: int = 0,
) -> None:
    """Push ``x [m, k] @ weight[n, k]^T`` (BF16, contiguous) into the call whose rotation state is
    at ``flags`` (see the module docstring). ``out``: a contiguous BF16 ``[m, n]`` tensor that
    gives the launch its shape; the kernel never writes it."""
    m, k = x.shape
    n = weight.shape[0]
    check_push(m, n, k, tactic, row, col0)
    ensure_cute_runtime()
    _compiled(tactic, world, n, row, col0, target)(
        *_swapped(x, weight, out),
        Int64(peer_ptrs),
        Int64(mc_base),
        Int64(flags),
        Int32(rank),
        Int32(m),
        _cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
    )


# ---------------------------------------------------------------------------
# The sharded MoE front on a BF16 latent_down
# ---------------------------------------------------------------------------


def front_logits_views(ws) -> Tuple[torch.Tensor, torch.Tensor]:
    """FP32 ``[M_MAX, 896]`` views of the logits rows of the front workspace's two halves on this
    rank (built once, outside capture; the route kernel polls them)."""
    nbytes = front.half_bytes(ws.world)
    buf = ws.handle.get_uc_buffer(ws.rank, (2 * nbytes // 4,), torch.float32, 0)
    off = front.logits_offset(ws.world) // 4
    size = front.M_MAX * front.NUM_EXPERTS
    return tuple(
        buf[h * nbytes // 4 + off : h * nbytes // 4 + off + size].view(
            front.M_MAX, front.NUM_EXPERTS
        )
        for h in (0, 1)
    )


def run_ull_front_bf16(
    x: torch.Tensor,
    logits: torch.Tensor,
    latent_rows: torch.Tensor,
    logits_views: Tuple[torch.Tensor, torch.Tensor],
    bias: torch.Tensor,
    routed_scaling_factor: float,
    global_sf: torch.Tensor,
    local_offset: int,
    local_num: int,
    ws,
    *,
    tactic: SplitKTactic,
    zero: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    """The front of one MoE layer on this rank, for the BF16 ``latent_rows`` (this rank's
    ``[128 * tiles, 7168]`` rows of latent_down) and the FP32 router ``logits [m, 896]``. Returns
    ``(routing, codes, scales)``: ``rubin_route_tail``'s eight outputs for the expert window
    ``[local_offset, local_offset + local_num)``, and the NVFP4 latent of
    ``bf16(x @ latent_down^T)`` under ``global_sf`` (uint8 ``[m, 1792]`` codes and uint8
    ``[m, 224]`` linear-layout scales).

    Launches on the current stream: the push GEMM of the rank's latent tiles into every rank's
    slot ``(token, rank)`` of the call's half (one-shot), the copy of ``logits`` into this rank's
    logits rows of that half (adding ``+0`` sends ``-0`` as ``+0``, which the poll needs), and
    the front's route kernel. Collective over the workspace's group: every rank makes the same
    calls in the same order (the halves alternate call by call)."""
    m = x.shape[0]
    world = ws.world
    if not 1 <= m <= front.M_MAX or logits.shape != (m, front.NUM_EXPERTS):
        raise ValueError(f"front call of {m} rows with logits {tuple(logits.shape)}")
    half = ws.calls & 1
    flags = ws.next_flags()
    run_ull_push(
        x,
        latent_rows,
        torch.empty((m, latent_rows.shape[0]), dtype=torch.bfloat16, device=x.device),
        peer_ptrs=ws.peer_ptrs.data_ptr(),
        mc_base=ws.mc_base,
        flags=flags.data_ptr(),
        rank=ws.rank,
        world=world,
        tactic=tactic,
        target=SLOT,
        row=front.slot_cols(world),
        col0=0,
    )
    torch.add(logits, 0.0, out=logits_views[half][:m])
    return route_front(
        m,
        ws,
        flags,
        bias,
        routed_scaling_factor,
        global_sf,
        local_offset,
        local_num,
        zero=zero,
    )


def route_front(
    m: int,
    ws,
    flags: torch.Tensor,
    bias: torch.Tensor,
    routed_scaling_factor: float,
    global_sf: torch.Tensor,
    local_offset: int,
    local_num: int,
    *,
    zero: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    """The front's route kernel on the call's half (``flags``): polls the logits rows and every
    rank's latent slots, writes ``rubin_route_tail``'s eight outputs and the NVFP4 latent."""
    device = bias.device
    max_tiles, max_permuted = sort_buffer_sizes(m, local_num, front.TILE)
    i32 = dict(dtype=torch.int32, device=device)
    routing = (
        torch.empty((m, front.TOP_K), **i32),
        torch.empty((m, front.TOP_K), dtype=torch.float32, device=device),
        torch.empty((max_tiles,), **i32),
        torch.empty((max_tiles,), **i32),
        torch.empty((m, front.TOP_K), **i32),
        torch.empty((max_permuted,), **i32),
        torch.empty((1,), **i32),
        torch.empty((1,), **i32),
    )
    codes = torch.empty((m, front.LATENT // 2), dtype=torch.uint8, device=device)
    scales = torch.empty((m, front.LATENT // front._FP4_VEC), dtype=torch.uint8, device=device)
    tile_expert, tile_limit, e2p, p2e, total, num_tiles = (t.data_ptr() for t in routing[2:])
    scale_bits = struct.unpack("<q", struct.pack("<d", float(routed_scaling_factor)))[0]
    num_ctas, team, warps = front.route_config(m)
    route = front._compiled_route(num_ctas, warps, team, front._local_cap(local_num), ws.world)
    route(
        Int32(m),
        Int64(ws.uc_local),
        Int64(flags.data_ptr()),
        Int64(bias.data_ptr()),
        Int64(routing[0].data_ptr()),
        Int64(routing[1].data_ptr()),
        Int64(e2p),
        Int64(p2e),
        Int64(tile_expert),
        Int64(tile_limit),
        Int64(total),
        Int64(num_tiles),
        Int32(local_offset),
        Int32(local_num),
        Int64(scale_bits),
        Int64(codes.data_ptr()),
        Int64(scales.data_ptr()),
        Int64(global_sf.data_ptr()),
        Int64(zero.data_ptr()),
        Int32(zero.numel()),
        _cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
    )
    return routing, codes, scales


__all__ = [
    "AG",
    "MAX_M",
    "SLOT",
    "UllPushGemmKernel",
    "check_push",
    "front_logits_views",
    "route_front",
    "run_ull_front_bf16",
    "run_ull_push",
    "tactic_of",
]
