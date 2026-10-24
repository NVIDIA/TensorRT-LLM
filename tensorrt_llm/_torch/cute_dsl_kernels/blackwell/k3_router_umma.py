# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 router GEMM on ``tcgen05`` UMMA with a small grid (SM100/SM103/SM107).

Computes split-K partial router logits ``partials[s, t, e] = sum_{k in slice s} h[t, k] W[e, k]``
(FP32) for the routing tail (:func:`k3_route_tail.run_k3_route_tail` sums the ``S`` splits in
order). It is the GEMM phase of ``k3_route`` for decode batches: a fixed grid of ``7 * S`` CTAs
(35 by default) streams the 12.8 MB BF16 weight, so it runs on the SMs the latent
down-projection (112 CTAs) leaves free, and the tensor-core MMA keeps it bandwidth-bound for
every token count up to 256 (the ``mma.sync`` GEMV becomes compute-bound above about 16 tokens
on so few SMs).

* CTA ``(g, s)`` owns experts ``[128 g, 128 g + 128)`` (the UMMA M side; tokens are N, padded
  to a multiple of 16) and the ``s``-th slice of K in 64-wide chunks.
* Warp 0 streams (weight chunk, activation chunk) pairs through a ring of TMA stages: the
  weight halves of the first ``stages`` chunks are issued before ``griddepcontrol.wait`` (the
  weight never depends on the previous grid), the activations after it.
* Warp 1 allocates tensor memory and issues the UMMAs (FP32 accumulation in tensor memory);
  warps 4-7 copy the accumulator out, one expert row per thread, so a warp stores 32
  consecutive experts of a token (128 contiguous bytes).
* Within a CTA the K order is fixed (chunk by chunk, 16-wide UMMA K steps), so the partials and
  the logits the tail sums from them are deterministic.
"""

from __future__ import annotations

import functools
import os

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Int32, Int64
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import make_ptr

from .k3_fused_router import (
    _EVICT_FIRST,
    HIDDEN_SIZE,
    NUM_EXPERTS,
    _griddepcontrol_launch_dependents,
    _griddepcontrol_wait,
    _lds_b32,
    _st_global_f32,
    _tmem_fence_before_sync,
    _tmem_ld16,
)

#: Experts per CTA (UMMA M).
_ROWS = 128
#: Expert groups (CTAs per K split).
_GROUPS = NUM_EXPERTS // _ROWS
#: K extent of one ring stage (one 128-byte swizzle atom of BF16).
_K_CHUNK = 64
_NUM_K_CHUNKS = HIDDEN_SIZE // _K_CHUNK
_A_CHUNK_BYTES = _ROWS * _K_CHUNK * 2
#: Largest token count (UMMA N).
MAX_M = 256
#: Default K splits: ``7 * 5 = 35`` CTAs. Measured on GB300 (148 SMs), where the GEMM shares the
#: SMs with latent_down. On Rubin (SM107, 212 SMs; MoE entry replica of one layer, M 18-30) 5 and 7
#: splits are within the noise of each other, 3 is slower and 9 is not faster.
DEFAULT_SPLITS = int(os.environ.get("TRTLLM_K3_ROUTER_UMMA_SPLITS", "5"))
#: Shared memory for the ring (the rest holds barriers and alignment slack).
_RING_BUDGET = 200 * 1024
_THREADS = 256

assert NUM_EXPERTS % _ROWS == 0 and HIDDEN_SIZE % _K_CHUNK == 0


class K3RouterUmmaKernel:
    """Router GEMM ``logits[M, 896] = h[M, 7168] @ W^T`` as FP32 split-K partials, ``M <= 256``.

    Args:
        n_pad: Padded token count (UMMA N), a multiple of 16 in ``[16, 256]``.
        num_splits: K splits ``S``; the grid is ``7 * S`` CTAs.
        use_pdl: Launch with programmatic dependent launch.
    """

    def __init__(self, n_pad: int, num_splits: int, use_pdl: bool):
        if n_pad % 16 or not 16 <= n_pad <= MAX_M:
            raise ValueError(f"n_pad={n_pad} must be a multiple of 16 in [16, {MAX_M}]")
        if not 1 <= num_splits <= _NUM_K_CHUNKS:
            raise ValueError(f"num_splits={num_splits} must be in [1, {_NUM_K_CHUNKS}]")
        self.n_pad = n_pad
        self.num_splits = num_splits
        self.num_ctas = _GROUPS * num_splits
        self.use_pdl = use_pdl
        self.max_chunks = (_NUM_K_CHUNKS + num_splits - 1) // num_splits
        self.b_chunk_bytes = n_pad * _K_CHUNK * 2
        self.stages = max(
            1, min(self.max_chunks, _RING_BUDGET // (_A_CHUNK_BYTES + self.b_chunk_bytes))
        )
        self.tmem_cols = max(32, 1 << (n_pad - 1).bit_length())
        self.mma_tiler = (_ROWS, n_pad, _K_CHUNK)

    @cute.jit
    def __call__(
        self,
        w_ptr: cute.Pointer,
        h_ptr: cute.Pointer,
        m: Int32,
        out_addr: Int64,
        stream: cuda.CUstream,
    ):
        mW = cute.make_tensor(
            w_ptr,
            cute.make_layout(
                (NUM_EXPERTS, HIDDEN_SIZE, 1), stride=(HIDDEN_SIZE, 1, NUM_EXPERTS * HIDDEN_SIZE)
            ),
        )
        mH = cute.make_tensor(
            h_ptr, cute.make_layout((m, HIDDEN_SIZE, 1), stride=(HIDDEN_SIZE, 1, m * HIDDEN_SIZE))
        )
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            cutlass.BFloat16,
            cutlass.BFloat16,
            OperandMajorMode.K,
            OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler[:2],
        )
        a_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, cutlass.BFloat16, self.stages
        )
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, cutlass.BFloat16, self.stages
        )
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        tma_atom_a, tma_a = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            mW,
            cute.slice_(a_layout, (None, None, None, 0)),
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_atom_b, tma_b = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            mH,
            cute.slice_(b_layout, (None, None, None, 0)),
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )

        @cute.struct
        class SharedStorage:
            full: cute.struct.MemRange[cutlass.Int64, self.stages]
            empty: cute.struct.MemRange[cutlass.Int64, self.stages]
            acc_full: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_ready: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_free: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding: cute.struct.MemRange[cutlass.Int32, 1]
            sA: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(a_layout.outer)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(b_layout.outer)], 1024
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma, tma_atom_a, tma_a, tma_atom_b, tma_b, a_layout, b_layout, m, out_addr
        ).launch(
            grid=(self.num_ctas, 1, 1),
            block=(_THREADS, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        m: Int32,
        out_addr: Int64,
    ):
        S = self.num_splits
        R = self.stages
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        group = bidx % _GROUPS
        split = bidx // _GROUPS
        base_chunks = _NUM_K_CHUNKS // S
        rem_chunks = _NUM_K_CHUNKS % S
        chunk_count = Int32(base_chunks)
        chunk_start = split * base_chunks
        if cutlass.const_expr(rem_chunks > 0):
            if split < rem_chunks:
                chunk_count = chunk_count + 1
                chunk_start = chunk_start + split
            else:
                chunk_start = chunk_start + rem_chunks

        storage = utils.SmemAllocator().allocate(self.shared_storage)
        full = storage.full.data_ptr()
        empty = storage.empty.data_ptr()
        acc_full = storage.acc_full.data_ptr()
        tmem_ready = storage.tmem_ready.data_ptr()
        tmem_free = storage.tmem_free.data_ptr()
        tmem_holding = storage.tmem_holding.data_ptr()
        if warp == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(R):
                    cute.arch.mbarrier_init(full + i, 1)
                    cute.arch.mbarrier_init(empty + i, 1)
                cute.arch.mbarrier_init(acc_full, 1)
                cute.arch.mbarrier_init(tmem_ready, 1)
                cute.arch.mbarrier_init(tmem_free, 4)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        if warp == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            sA = storage.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
            sB = storage.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)
            gA = cute.local_tile(
                mA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
            )
            gB = cute.local_tile(
                mB, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
            )
            thr_mma = tiled_mma.get_slice(0)
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                0,
                cute.make_layout(1),
                cute.group_modes(sA, 0, 3),
                cute.group_modes(thr_mma.partition_A(gA), 0, 3),
            )
            tBsB, tBgB = cpasync.tma_partition(
                tma_atom_b,
                0,
                cute.make_layout(1),
                cute.group_modes(sB, 0, 3),
                cute.group_modes(thr_mma.partition_B(gB), 0, 3),
            )
            stage_bytes = _A_CHUNK_BYTES + self.b_chunk_bytes
            # The weight never depends on the previous grid: fill the ring's weight halves first.
            prologue = chunk_count
            if prologue > R:
                prologue = Int32(R)
            for c in cutlass.range(prologue, unroll=1):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(full + c, stage_bytes)
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, group, chunk_start + c, 0)],
                    tAsA[(None, c)],
                    tma_bar_ptr=full + c,
                    cache_policy=Int64(_EVICT_FIRST),
                )
            # Dependents may launch once this CTA holds its tensor memory (a dependent CTA that
            # took the columns first and then waited on this grid would deadlock it).
            cute.arch.mbarrier_wait(tmem_ready, 0)
            if cutlass.const_expr(self.use_pdl):
                _griddepcontrol_launch_dependents()
                _griddepcontrol_wait()
            for c in cutlass.range(prologue, unroll=1):
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, 0, chunk_start + c, 0)],
                    tBsB[(None, c)],
                    tma_bar_ptr=full + c,
                )
            for c in cutlass.range(prologue, chunk_count, unroll=1):
                stage = c % R
                cute.arch.mbarrier_wait(empty + stage, ((c // R) & 1) ^ 1)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(full + stage, stage_bytes)
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, group, chunk_start + c, 0)],
                    tAsA[(None, stage)],
                    tma_bar_ptr=full + stage,
                    cache_policy=Int64(_EVICT_FIRST),
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, 0, chunk_start + c, 0)],
                    tBsB[(None, stage)],
                    tma_bar_ptr=full + stage,
                )
        elif warp == 1:
            cute.arch.alloc_tmem(self.tmem_cols, tmem_holding)
            cute.arch.relinquish_tmem_alloc_permit()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(tmem_ready)
            acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
                cutlass.Float32, alignment=16, ptr_to_buffer_holding_addr=tmem_holding
            )
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
            tCtAcc = cute.make_tensor(acc_tmem_ptr, tiled_mma.make_fragment_C(acc_shape).layout)
            sA = storage.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
            sB = storage.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)
            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)
            num_kphases = cute.size(tCrA, mode=[2])
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for c in cutlass.range(chunk_count, unroll=1):
                stage = c % R
                cute.arch.mbarrier_wait(full + stage, (c // R) & 1)
                for kp in cutlass.range_constexpr(num_kphases):
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[(None, None, kp, stage)],
                        tCrB[(None, None, kp, stage)],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage)
            with cute.arch.elect_one():
                tcgen05.commit(acc_full)
            cute.arch.mbarrier_wait(tmem_free, 0)
            cute.arch.dealloc_tmem(acc_tmem_ptr, self.tmem_cols)
        elif warp >= 4:
            # TMEM lane = expert row: warp 4 + i reads lanes [32 i, 32 i + 32).
            cute.arch.mbarrier_wait(tmem_ready, 0)
            tmem_base = _lds_b32(tmem_holding.toint())
            if cutlass.const_expr(self.use_pdl):
                # The partial buffer may still be read by the previous grid.
                _griddepcontrol_wait()
            row = tidx - 128
            expert = group * _ROWS + row
            split_base = Int64(split) * Int64(m)
            lane_base = tmem_base + (((warp - 4) * 32) << 16)
            cute.arch.mbarrier_wait(acc_full, 0)
            for j in cutlass.range_constexpr(self.n_pad // 16):
                v = _tmem_ld16(lane_base + j * 16)
                for i in cutlass.range_constexpr(16):
                    token = j * 16 + i
                    if token < m:
                        off = ((split_base + token) * NUM_EXPERTS + expert) * 4
                        _st_global_f32(out_addr + off, v[i])
            _tmem_fence_before_sync()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(tmem_free)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


@functools.cache
def _compiled(n_pad: int, num_splits: int, use_pdl: bool):
    kernel = K3RouterUmmaKernel(n_pad, num_splits, use_pdl)
    dummy = 1 << 20
    return cute.compile(
        kernel,
        make_ptr(cutlass.BFloat16, dummy, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.BFloat16, dummy, cute.AddressSpace.gmem, assumed_align=16),
        Int32(n_pad),
        Int64(dummy),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


@functools.cache
def _is_supported_device(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) in ((10, 0), (10, 3), (10, 7))


def can_run(hidden_states: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether :func:`run_k3_router_umma` serves these inputs (non-raising check)."""
    return (
        hidden_states.is_cuda
        and _is_supported_device(hidden_states.device.index)
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.dim() == 2
        and hidden_states.is_contiguous()
        and tuple(hidden_states.shape[1:]) == (HIDDEN_SIZE,)
        and 1 <= hidden_states.shape[0] <= MAX_M
        and hidden_states.data_ptr() % 16 == 0
        and weight.dtype == torch.bfloat16
        and tuple(weight.shape) == (NUM_EXPERTS, HIDDEN_SIZE)
        and weight.is_contiguous()
        and weight.data_ptr() % 16 == 0
    )


def run_k3_router_umma(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    partials: torch.Tensor,
    *,
    use_pdl: bool = True,
) -> None:
    """Launch the router GEMM: ``partials[s] = split s of hidden_states @ weight^T`` (FP32).

    Args:
        hidden_states: ``bf16 [M, 7168]``, ``1 <= M <= 256``, contiguous, 16-byte aligned.
        weight: ``bf16 [896, 7168]``, contiguous, 16-byte aligned.
        partials: Output ``fp32 [S, M, 896]``; ``S`` is the number of K splits (``7 * S`` CTAs).
    """
    m = hidden_states.shape[0]
    compiled = _compiled(16 * ((m + 15) // 16), partials.shape[0], use_pdl)
    compiled(
        make_ptr(cutlass.BFloat16, weight.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(
            cutlass.BFloat16, hidden_states.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        ),
        Int32(m),
        Int64(partials.data_ptr()),
        cuda.CUstream(torch.cuda.current_stream(hidden_states.device).cuda_stream),
    )


__all__ = ["DEFAULT_SPLITS", "MAX_M", "K3RouterUmmaKernel", "can_run", "run_k3_router_umma"]
