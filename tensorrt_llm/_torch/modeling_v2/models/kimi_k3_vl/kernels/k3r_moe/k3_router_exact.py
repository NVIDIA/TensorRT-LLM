# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 router logits (SM100/SM103/SM107, ``M <= 16`` tokens), bit-identical to
``trtllm::dsv3_router_gemm_op``, with the router weight read before ``griddepcontrol.wait``.

``dsv3_router_gemm_op`` serves ``M <= 16`` tokens of the 896 x 7168 BF16 router with
``router_gemm_mma_kernel``: CTA ``b`` owns experts ``[8 b, 8 b + 8)``, its warp ``s`` of 14 owns
the K slice ``[512 s, 512 s + 512)`` and chains 32 ``mma.sync.m16n8k16`` from a zero accumulator
(two per 32-wide chunk: lane ``(g, q)`` holds k ``chunk + 8 q .. + 8`` of activation rows ``g`` and
``g + 8`` and of expert ``g``, fed as k pairs ``{0,1},{2,3}`` then ``{4,5},{6,7}``), and the CTA sums
the 14 slice partials in slice order starting from 0.

:class:`K3RouterExactKernel` computes the same 14 slice partials with the same fragments, MMAs
and order, and sums them the same way, so every logit has the same bits. The work is cut
differently: a cluster of 2 CTAs owns 16 experts (two n8 tiles); CTA ``k`` computes slices
``7 k .. 7 k + 6`` (warp ``w`` = slice ``7 k + w``, one MMA chain per tile), sums its seven partials
in slice order, and CTA 0 hands its running sum to CTA 1 over distributed shared memory, which
adds slices 7..13 in order and stores the logits. A CTA's 112 KB weight slice is copied into shared
memory by TMA (128-byte swizzle, bank-conflict-free fragment loads) **before**
``griddepcontrol.wait``: with PDL the copy runs while the previous kernel finishes. Each warp loads
its slice of the activation (16 KB) straight into registers in one round trip after the wait.

PDL: the kernel triggers its dependents right after its own ``griddepcontrol.wait``; before the
wait it reads only the router weight.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

from .k3_fused_router import (
    _EVICT_FIRST,
    HIDDEN_SIZE,
    NUM_EXPERTS,
    _asm,
    _fv,
    _iv,
    _mma_bf16,
    _st_global_f32,
    _unpack,
)
from .k3_route_tail import _griddepcontrol_launch_dependents, _griddepcontrol_wait

#: Largest token count (one m16 tile, like ``router_gemm_mma_kernel``).
MAX_M = 16
#: K slices of ``router_gemm_mma_kernel`` (its warps) and their width.
NUM_SLICES = 14
SLICE_K = HIDDEN_SIZE // NUM_SLICES
#: 32-wide K chunks per slice.
CHUNKS = SLICE_K // 32
#: Cluster: 2 CTAs x 7 slices; 2 n8 expert tiles (16 experts) per cluster.
CLUSTER = 2
SLICES_PER_CTA = NUM_SLICES // CLUSTER
TILES = 2
NUM_WARPS = SLICES_PER_CTA
THREADS = 32 * NUM_WARPS
CLUSTER_EXPERTS = 8 * TILES
NUM_CTAS = (NUM_EXPERTS // CLUSTER_EXPERTS) * CLUSTER
#: Weight TMA box: 8 rows x 64 BF16 (one 1 KB 128-byte-swizzle atom).
_BOX_K = 64
_BOXES_PER_SLICE = SLICE_K // _BOX_K
_W_BOX_BYTES = 8 * _BOX_K * 2
#: One fragment set of the two tiles: 2 tiles x 32 lanes x 4 floats.
_FRAG_BYTES = TILES * 32 * 4 * 4

assert HIDDEN_SIZE % (NUM_SLICES * 32) == 0 and NUM_SLICES % CLUSTER == 0
assert NUM_EXPERTS % CLUSTER_EXPERTS == 0


@dsl_user_op
def _lds_v4_b32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(addr, loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _st_async_v4_f32(local_addr, v0, v1, v2, v3, local_mbar, peer, *, loc=None, ip=None):
    """Store four FP32 values into a peer CTA's SMEM, crediting its mbarrier."""
    _asm(
        None,
        [_iv(local_addr, loc=loc, ip=ip)]
        + [_fv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)]
        + [_iv(local_mbar, loc=loc, ip=ip), _iv(peer, loc=loc, ip=ip)],
        "{\n"
        ".reg .b32 ra, rb;\n"
        "mapa.shared::cluster.u32 ra, $0, $6;\n"
        "mapa.shared::cluster.u32 rb, $5, $6;\n"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [ra], {$1, $2, $3, $4}, [rb];\n"
        "}",
        "r,f,f,f,f,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ldg_v4_b32_if(valid, addr, *, loc=None, ip=None):
    """16 bytes of global memory as four 32-bit words if ``valid != 0``, else zeros."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(valid, loc=loc, ip=ip), Int64(addr).ir_value(loc=loc, ip=ip)],
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.s32 p, $4, 0;\n"
        "mov.b32 $0, 0;\nmov.b32 $1, 0;\nmov.b32 $2, 0;\nmov.b32 $3, 0;\n"
        "@p ld.global.v4.b32 {$0, $1, $2, $3}, [$5];\n"
        "}",
        "=r,=r,=r,=r,r,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _sts_v4_f32(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)] + [_fv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.shared.v4.f32 [$0], {$1, $2, $3, $4};",
        "r,f,f,f,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_v4_f32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_iv(addr, loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 4, loc=loc, ip=ip)


class K3RouterExactKernel:
    """``logits[M, 896]``, bit-identical to ``router_gemm_mma_kernel`` (see the module docstring)."""

    def __init__(self, use_pdl: bool):
        self.use_pdl = use_pdl

    @cute.jit
    def __call__(
        self,
        w_ptr: cute.Pointer,
        x_addr: Int64,
        m: Int32,
        out_addr: Int64,
        stream: cuda.CUstream,
    ):
        mW = cute.make_tensor(
            w_ptr, cute.make_layout((NUM_EXPERTS, HIDDEN_SIZE), stride=(HIDDEN_SIZE, 1))
        )
        atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
        )
        # Weight boxes [8 rows, 64 k, (slice, tile, box)].
        w_boxes = SLICES_PER_CTA * TILES * _BOXES_PER_SLICE
        w_layout = cute.tile_to_shape(atom, (8, _BOX_K, w_boxes), order=(0, 1, 2))
        tma_atom_w, tma_w = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mW,
            cute.tile_to_shape(atom, (8, _BOX_K), order=(0, 1)),
            (8, _BOX_K),
        )

        @cute.struct
        class SharedStorage:
            w_full: cute.struct.MemRange[cutlass.Int64, NUM_WARPS]
            mbox_full: cutlass.Int64
            sW: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, cute.cosize(w_layout.outer)], 1024
            ]
            # Slice partials [warp][tile][lane][4], then the peer's running sum [tile][lane][4].
            red: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, NUM_WARPS * _FRAG_BYTES // 4], 16
            ]
            mbox: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, _FRAG_BYTES // 4], 16]

        self.shared_storage = SharedStorage
        self.kernel(tma_atom_w, tma_w, w_layout, x_addr, m, out_addr).launch(
            grid=(NUM_CTAS, 1, 1),
            block=(THREADS, 1, 1),
            cluster=(CLUSTER, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_w: cute.CopyAtom,
        mWt: cute.Tensor,
        w_layout: cute.ComposedLayout,
        x_addr: Int64,
        m: Int32,
        out_addr: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        group = cta // CLUSTER
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        w_full = storage.w_full.data_ptr()
        mbox_full = storage.mbox_full
        w_base = storage.sW.data_ptr().toint()
        red_base = storage.red.data_ptr().toint()
        mbox_base = storage.mbox.data_ptr().toint()
        s = rank * SLICES_PER_CTA + warp

        with cute.arch.elect_one():
            cute.arch.mbarrier_init(w_full + warp, 1)
        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mbox_full, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        # The peer writes into this CTA's mailbox only after both CTAs initialized their barriers.
        cute.arch.cluster_arrive_relaxed()

        # This warp's slice of the cluster's two weight tiles, before the wait.
        sW = storage.sW.get_tensor(w_layout.outer, swizzle=w_layout.inner)
        gW = cute.local_tile(mWt, (8, _BOX_K), (None, None))
        tWs, tWg = cpasync.tma_partition(
            tma_atom_w,
            0,
            cute.make_layout(1),
            cute.group_modes(sW, 0, 2),
            cute.group_modes(gW, 0, 2),
        )
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(
                w_full + warp, TILES * _BOXES_PER_SLICE * _W_BOX_BYTES
            )
        for t in cutlass.range_constexpr(TILES):
            for j in cutlass.range_constexpr(_BOXES_PER_SLICE):
                cute.copy(
                    tma_atom_w,
                    tWg[(None, group * TILES + t, s * _BOXES_PER_SLICE + j)],
                    tWs[(None, (warp * TILES + t) * _BOXES_PER_SLICE + j)],
                    tma_bar_ptr=w_full + warp,
                    cache_policy=Int64(_EVICT_FIRST),
                )

        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
            # Safe trigger point: everything up to the previous grid is complete.
            _griddepcontrol_launch_dependents()

        # router_gemm_mma_kernel fragments: lane (g, q) holds activation rows g and g + 8 and
        # weight row g at k = chunk + 8 q .. + 8. The whole slice of the activation is loaded
        # into registers at once.
        g = lane >> 2
        q = lane & 3
        row0 = Int32(0)
        if g < m:
            row0 = Int32(1)
        row1 = Int32(0)
        if g + 8 < m:
            row1 = Int32(1)
        a0_base = x_addr + Int64(g * (HIDDEN_SIZE * 2) + (s * SLICE_K + q * 8) * 2)
        a1_base = a0_base + Int64(8 * HIDDEN_SIZE * 2)
        acts = []
        for cc in cutlass.range_constexpr(CHUNKS):
            a_off = Int64(cc * 64)
            acts.append(
                (_ldg_v4_b32_if(row0, a0_base + a_off), _ldg_v4_b32_if(row1, a1_base + a_off))
            )
        w_warp = w_base + warp * (TILES * _BOXES_PER_SLICE * _W_BOX_BYTES)
        cute.arch.mbarrier_wait(w_full + warp, 0)
        accs = [[Float32(0.0)] * 4 for _ in range(TILES)]
        nxt = [self._wfrag(w_warp, g, q, t, 0) for t in range(TILES)]
        for cc in cutlass.range_constexpr(CHUNKS):
            cur = nxt
            if cc + 1 < CHUNKS:
                nxt = [self._wfrag(w_warp, g, q, t, cc + 1) for t in range(TILES)]
            a0, a1 = acts[cc]
            for t in cutlass.range_constexpr(TILES):
                bw = cur[t]
                acc = _mma_bf16((a0[0], a1[0], a0[1], a1[1]), bw[0], bw[1], accs[t])
                accs[t] = list(_mma_bf16((a0[2], a1[2], a0[3], a1[3]), bw[2], bw[3], acc))
        for t in cutlass.range_constexpr(TILES):
            acc = accs[t]
            _sts_v4_f32(
                red_base + warp * _FRAG_BYTES + (t * 32 + lane) * 16, acc[0], acc[1], acc[2], acc[3]
            )
        cute.arch.barrier()

        # Thread t < 64 owns fragment (tile t / 32, lane t % 32): the running sum over this
        # CTA's slices in order (CTA 0 starts from 0, CTA 1 from CTA 0's sum).
        cute.arch.cluster_wait()
        if rank == 1:
            if tidx == 0:
                # One arrival plus CTA 0's bytes complete the mailbox phase.
                cute.arch.mbarrier_arrive_and_expect_tx(mbox_full, _FRAG_BYTES)
        if tidx < TILES * 32:
            off = tidx * 16
            tot = [Float32(0.0)] * 4
            if rank == 1:
                cute.arch.mbarrier_wait(mbox_full, 0)
                tot = list(_lds_v4_f32(mbox_base + off))
            for w in cutlass.range_constexpr(NUM_WARPS):
                v = _lds_v4_f32(red_base + w * _FRAG_BYTES + off)
                for i in cutlass.range_constexpr(4):
                    tot[i] = tot[i] + v[i]
            if rank == 0:
                _st_async_v4_f32(
                    mbox_base + off, tot[0], tot[1], tot[2], tot[3], mbox_full.toint(), Int32(1)
                )
            else:
                t_idx = tidx >> 5
                ln = tidx & 31
                for i in cutlass.range_constexpr(4):
                    row = (ln >> 2) + (i >> 1) * 8
                    col = group * CLUSTER_EXPERTS + t_idx * 8 + (ln & 3) * 2 + (i & 1)
                    if row < m:
                        _st_global_f32(out_addr + Int64((row * NUM_EXPERTS + col) * 4), tot[i])
        # No CTA leaves while its peer may still write into its mailbox.
        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

    @cute.jit
    def _wfrag(self, w_warp, g, q, t: int, cc: int):
        """Weight row ``g`` of tile ``t`` at chunk ``cc`` of this warp's slice (swizzled box)."""
        box = t * _BOXES_PER_SLICE + cc // 2
        unit = (4 * (cc % 2) + q) ^ g
        return _lds_v4_b32(w_warp + box * _W_BOX_BYTES + g * 128 + (unit << 4))


@functools.cache
def _compiled(use_pdl: bool):
    kernel = K3RouterExactKernel(use_pdl)
    return cute.compile(
        kernel,
        make_ptr(cutlass.BFloat16, 1 << 20, cute.AddressSpace.gmem, assumed_align=16),
        Int64(1 << 20),
        Int32(MAX_M),
        Int64(1 << 20),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


@functools.cache
def _is_supported_device(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) in ((10, 0), (10, 3), (10, 7))


def can_run(hidden_states: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether :func:`run_k3_router_exact` serves these inputs (non-raising check)."""
    return (
        hidden_states.is_cuda
        and _is_supported_device(hidden_states.device.index)
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.dim() == 2
        and hidden_states.is_contiguous()
        and hidden_states.shape[1] == HIDDEN_SIZE
        and 1 <= hidden_states.shape[0] <= MAX_M
        and hidden_states.data_ptr() % 16 == 0
        and weight.dtype == torch.bfloat16
        and tuple(weight.shape) == (NUM_EXPERTS, HIDDEN_SIZE)
        and weight.is_contiguous()
        and weight.data_ptr() % 16 == 0
    )


def run_k3_router_exact(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    logits: torch.Tensor,
    *,
    use_pdl: bool = True,
) -> None:
    """Launch on the current stream: ``logits`` (``fp32 [M, 896]``) =
    ``dsv3_router_gemm_op(hidden_states, weight.t())`` bit for bit."""
    m = hidden_states.shape[0]
    _compiled(use_pdl)(
        make_ptr(cutlass.BFloat16, weight.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        Int64(hidden_states.data_ptr()),
        Int32(m),
        Int64(logits.data_ptr()),
        cuda.CUstream(torch.cuda.current_stream(hidden_states.device).cuda_stream),
    )


def k3_router_logits(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """FP32 ``[M, 896]`` router logits of ``M <= 16`` tokens, bit-identical to
    ``dsv3_router_gemm_op(hidden_states, weight.t())``; the weight is copied into shared memory
    before ``griddepcontrol.wait``."""
    if not can_run(hidden_states, weight):
        raise ValueError("unsupported router inputs")
    logits = torch.empty(
        (hidden_states.shape[0], NUM_EXPERTS), dtype=torch.float32, device=hidden_states.device
    )
    run_k3_router_exact(hidden_states, weight, logits)
    return logits


__all__ = [
    "MAX_M",
    "K3RouterExactKernel",
    "can_run",
    "k3_router_logits",
    "run_k3_router_exact",
]
