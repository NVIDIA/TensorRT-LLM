# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 MoE front for small decode steps (``M <= 32`` rows) over the TP group.

Replaces, per MoE layer and rank, the router GEMM (``k3_router_exact`` / ``dsv3_router_gemm_op``),
the routing tail (``k3_route_tail``) and the replicated latent down-projection with its NVFP4
quantization (``k3_latent_down_fp4``, the side stream it runs on and its fork / join) by two
kernels on one stream:

* **Head** (:class:`K3UllFrontHeadKernel`, one grid, cluster size = the latent GEMM's split-K):

  - *Latent CTAs*: the ``k3_fp8w_lowm_gemm`` split-K MXFP8 GEMM on this rank's 128-row tiles of
    latent_down (28 tiles over ``W`` ranks, rank ``r`` owning tiles ``[t0(r), t1(r))``), same
    tactic as the replicated latent_down of ``k3_latent_down_fp4``. Its all-reduce push epilogue
    (one-shot) multicasts the BF16 rows into slot ``(token, r)`` of the front workspace of every
    rank (``-0`` sent as ``+0``).
  - *Router CTAs*: ``k3_router_exact``'s arithmetic on all 896 experts (every rank computes all
    logits, 12.8 MB of BF16 router weight streamed before the wait, and writes them into its own
    workspace only, so the routing does not wait for an exchange). A cluster owns
    ``8 * ROUTER_TILES`` experts; its CTAs split the 14 K slices of 512 and each warp runs one
    ``mma.sync`` chain per (slice, 16-row tile, 8-expert tile) exactly as
    ``router_gemm_mma_kernel`` does; the cluster's last CTA collects the 14 slice partials over
    distributed shared memory, sums them in slice order from 0 and stores the FP32 logits into
    ``[token][expert]``. Each 16-row tile therefore gets the bits ``dsv3_router_gemm_op`` gives
    for those 16 rows.

  Both roles read their weights before ``griddepcontrol.wait`` and release the dependent right
  after it.

* **Route** (:class:`K3UllFrontRouteKernel`, PDL dependent of the head, launched while the head
  runs and never waiting for it before it polls): a ``k3_route_tail`` cluster whose logits loads
  poll the workspace (sentinel ``0x80000000``) instead of reading a tensor, so the top-16 and the
  ``moe_sort`` tables of this rank are ``k3_route_tail``'s (bit-identical to ``noaux_tc_op`` +
  ``moe_sort`` on the same logits); and gather CTAs that poll the latent slots of every rank and
  write the NVFP4 quantization of the gathered BF16 latent (``fp4_quantize``'s arithmetic, linear
  scale layout), i.e. what ``k3_latent_down_fp4`` returns. Every reader re-arms what it read.

Workspace (:class:`UllFrontWorkspace`): an MNNVL multicast buffer of two halves; consecutive calls
alternate between them (the half is a launch argument through a constant flags tensor, so a CUDA
graph replays the same alternation). Rank ``r`` writes half ``h`` of a peer only after every rank
finished the call that last used ``h``: between two calls on a half there is the other half's call,
whose route kernel on ``r`` waited for data every peer pushed after its own previous call.
"""

from __future__ import annotations

import dataclasses
import functools
import os
import struct
import threading
from typing import Dict, Optional, Tuple

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

from . import k3_fp8w_lowm_gemm as lowm
from . import k3_lamport_ar as lamport
from .k3_fused_router import _EVICT_FIRST, _mma_bf16, sort_buffer_sizes
from .k3_latent_fp8_gemm import (
    _abs_max_f32x16,
    _e2m1x8,
    _ldg_f32,
    _nvfp4_scales_x2,
    _rcp_approx_ftz,
    _stg_u16,
    _stg_v4_b32,
)
from .k3_route_tail import (
    MAX_CTAS,
    TILE,
    K3RouteTailKernel,
    _cluster_arrive_release,
    _cluster_wait_acquire,
    _griddepcontrol_launch_dependents,
    _griddepcontrol_wait,
    _ld_nc_v4_f32,
    _lds_b32,
    _st_dsmem_b32,
    _sts_b32,
    _sts_v4_f32,
    _syncwarp,
)
from .k3_router_exact import _ldg_v4_b32_if, _lds_v4_b32, _lds_v4_f32, _st_async_v4_f32

#: Kimi-K3 shapes.
HIDDEN = 7168
LATENT = 3584
NUM_EXPERTS = 896
TOP_K = 16
#: 128-row tiles of latent_down (its output columns), split over the ranks.
LATENT_TILES = LATENT // TILE
#: Largest rows per call.
M_MAX = 32
#: Router K slices (``router_gemm_mma_kernel``'s warps) and their width.
NUM_SLICES = 14
SLICE_K = HIDDEN // NUM_SLICES
CHUNKS = SLICE_K // 32
_BOX_K = 64
_BOXES_PER_SLICE = SLICE_K // _BOX_K
_BOX_BYTES = 8 * _BOX_K * 2
#: One 8-expert x 16-row fragment set: 32 lanes x 4 FP32.
_FRAG_BYTES = 32 * 4 * 4
#: 8-expert tiles per router cluster (28 clusters, 84 CTAs at split 3).
ROUTER_TILES = 4
#: Gather clusters of the route kernel (of ``num_ctas`` CTAs each).
GATHER_CLUSTERS = 1
SENTINEL = lamport.SENTINEL
#: NVFP4 group (fp4_quantize's scaling vector) and the gather item (32 latent columns).
_FP4_VEC = 16
_ITEM_COLS = 32

assert HIDDEN % (NUM_SLICES * 32) == 0


# ---------------------------------------------------------------------------
# Rank geometry
# ---------------------------------------------------------------------------


def latent_tiles_of(rank: int, world: int) -> Tuple[int, int]:
    """``(first tile, tile count)`` of latent_down owned by ``rank`` (the first ``28 % W`` ranks
    own one tile more)."""
    base, rem = divmod(LATENT_TILES, world)
    count = base + (1 if rank < rem else 0)
    return rank * base + min(rank, rem), count


def slot_cols(world: int) -> int:
    """Features of one (token, rank) latent slot: the largest per-rank tile count x 128."""
    base, rem = divmod(LATENT_TILES, world)
    return (base + (1 if rem else 0)) * TILE


def experts_per_rank(world: int) -> int:
    return NUM_EXPERTS // world


def supports_world(world: int) -> bool:
    return (
        world >= 2
        and NUM_EXPERTS % world == 0
        and (NUM_EXPERTS // world) % 8 == 0
        and LATENT_TILES >= world
    )


def _align(a: int, b: int) -> int:
    return (a + b - 1) // b * b


def logits_offset(world: int) -> int:
    """Byte offset of the logits ``fp32 [M_MAX, 896]`` in a workspace half (after the latent
    slots ``bf16 [M_MAX, W, slot_cols]``)."""
    return _align(M_MAX * world * slot_cols(world) * 2, 1024)


def half_bytes(world: int) -> int:
    return _align(logits_offset(world) + M_MAX * NUM_EXPERTS * 4, 1 << 16)


# ---------------------------------------------------------------------------
# PTX helpers
# ---------------------------------------------------------------------------


def _asm(res, operands, text, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_volatile_b32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.b32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_poll_v4(addr, *, loc=None, ip=None):
    """16 bytes of the workspace with the poll's load."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.volatile.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _red_add_release_gpu(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_sentinel_b32(addr, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b32 t;\nmov.b32 t, 0x80000000;\nst.global.b32 [$0], t;\n}",
        "l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sanitize_f32_bits(word, *, loc=None, ip=None):
    """``-0.0`` (the sentinel) sent as ``+0.0``."""
    return Int32(
        _asm(
            T.i32(),
            [Int32(word).ir_value(loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.eq.b32 p, $1, 0x80000000;\nselp.b32 $0, 0, $1, p;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16x2_to_f32x2(word, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Int32(word).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b32 l, h;\nshl.b32 l, $2, 16;\nand.b32 h, $2, 0xffff0000;\n"
        "mov.b32 $0, l;\nmov.b32 $1, h;\n}",
        "=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


@cute.jit
def _poll_words(base: Int64, step: int, n: int):
    """``n`` FP32 words at ``base + i * step`` bytes, polled (every load of a sweep in flight at
    once) until none holds the sentinel."""
    words = cute.make_rmem_tensor((n,), Int32)
    ready = Int32(0)
    while ready == 0:
        ready = Int32(1)
        for i in cutlass.range_constexpr(n):
            words[i] = _ld_volatile_b32(base + Int64(i * step))
        for i in cutlass.range_constexpr(n):
            if words[i] == Int32(SENTINEL):
                ready = Int32(0)
    return [_i2f(words[i]) for i in range(n)]


@cute.jit
def _poll_raw4(base: Int64, step: int, n: int):
    """``n`` 16-byte vectors at ``base + i * step`` bytes polled until none holds the sentinel;
    returns the ``4 n`` 32-bit words."""
    words = cute.make_rmem_tensor((4 * n,), Int32)
    ready = Int32(0)
    while ready == 0:
        ready = Int32(1)
        for i in cutlass.range_constexpr(n):
            w = _ld_poll_v4(base + Int64(i * step))
            for j in cutlass.range_constexpr(4):
                words[4 * i + j] = w[j]
            ready = ready & lamport.clean4(*w)
    return [words[i] for i in range(4 * n)]


@cute.jit
def _poll_words4(base: Int64, step: int, n: int):
    """``n`` 16-byte vectors at ``base + i * step`` bytes, polled like :func:`_poll_words`; returns
    the ``4 n`` words as FP32."""
    words = cute.make_rmem_tensor((4 * n,), Int32)
    ready = Int32(0)
    while ready == 0:
        ready = Int32(1)
        for i in cutlass.range_constexpr(n):
            w = _ld_poll_v4(base + Int64(i * step))
            for j in cutlass.range_constexpr(4):
                words[4 * i + j] = w[j]
            ready = ready & lamport.clean4(*w)
    return [_i2f(words[i]) for i in range(4 * n)]


# ---------------------------------------------------------------------------
# Head: latent GEMM CTAs + router CTAs
# ---------------------------------------------------------------------------


class K3UllFrontHeadKernel:
    """The head grid: the latent GEMM's ``n_tiles * split_k`` CTAs, then the router clusters.

    Args:
        gemm: The latent GEMM (``K3Fp8wLowMGemmKernel`` with the one-shot push, ``ar_slot_n`` =
            :func:`slot_cols`) of this rank's tiles.
        router_tiles: 8-expert tiles per router cluster.
        m_tiles: 16-row tiles of the router (``ceil(M_max / 16)``).
        logits_off: Byte offset of the logits in a workspace half.
    """

    def __init__(
        self,
        gemm: "lowm.K3Fp8wLowMGemmKernel",
        router_tiles: int,
        m_tiles: int,
        logits_off: int,
        use_pdl: bool,
    ):
        self.gemm = gemm
        self.S = gemm.split_k
        self.threads = gemm.threads
        self.num_warps = self.threads // 32
        self.rt = router_tiles
        self.m_tiles = m_tiles
        self.logits_off = logits_off
        self.use_pdl = use_pdl
        self.n_tiles_r = NUM_EXPERTS // 8
        self.router_clusters = (self.n_tiles_r + router_tiles - 1) // router_tiles
        self.n_gemm_ctas = gemm.n_tiles * self.S
        self.grid = self.n_gemm_ctas + self.router_clusters * self.S
        # Slices of cluster rank r: [NUM_SLICES * r // S, NUM_SLICES * (r + 1) // S).
        self.max_slices = max(
            NUM_SLICES * (r + 1) // self.S - NUM_SLICES * r // self.S for r in range(self.S)
        )
        last = self.S - 1
        self.last_slices = NUM_SLICES - NUM_SLICES * last // self.S
        if self.max_slices > gemm.a_stages:
            raise ValueError("the router uses one GEMM weight-stage barrier per local slice")
        self.sw_bytes = self.max_slices * router_tiles * _BOXES_PER_SLICE * _BOX_BYTES
        self.part_bytes = NUM_SLICES * m_tiles * router_tiles * _FRAG_BYTES
        self.stage_bytes = M_MAX * 8 * router_tiles * 4
        # The router CTAs keep their buffers in the GEMM's shared storage from its weight ring on
        # (one allocation for both roles): weights, slice partials, logits (checked at launch).
        self.router_bytes = self.sw_bytes + self.part_bytes + self.stage_bytes
        if m_tiles * router_tiles * 32 > self.threads:
            raise ValueError("the router's combine needs one thread per fragment")

    @cute.jit
    def __call__(
        self,
        w_ptr: cute.Pointer,
        x_ptr: cute.Pointer,
        ws_ptr: cute.Pointer,
        m: Int32,
        peer_ptrs: Int64,
        mc_base: Int64,
        flags: Int64,
        rank: Int32,
        rw_ptr: cute.Pointer,
        x_addr: Int64,
        stream: cuda.CUstream,
    ):
        gemm_args = self.gemm.kernel_args(
            w_ptr,
            x_ptr,
            None,
            None,
            ws_ptr,
            x_ptr,
            x_ptr,
            x_ptr,
            x_ptr,
            m,
            peer_ptrs,
            mc_base,
            flags,
            rank,
        )
        tma_atom_r, tma_r, r_layout = self.router_args(rw_ptr)
        # The weight ring starts at the first 1 KB boundary after the barriers.
        avail = self.gemm.shared_storage.size_in_bytes() - 1024
        if cutlass.const_expr(self.router_bytes > avail):
            raise ValueError(
                f"router buffers ({self.router_bytes} B) exceed the GEMM storage ({avail} B)"
            )
        self.kernel(*gemm_args, tma_atom_r, tma_r, r_layout, x_addr).launch(
            grid=(self.grid, 1, 1),
            block=(self.threads, 1, 1),
            cluster=(self.S, 1, 1),
            smem=self.gemm.shared_storage.size_in_bytes(),
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.jit
    def router_args(self, rw_ptr: cute.Pointer):
        """``(tma_atom, tma_tensor, smem_layout)`` of the router weight slice (8-row x 64-column
        128-byte-swizzled boxes, as ``k3_router_exact``)."""
        mR = cute.make_tensor(rw_ptr, cute.make_layout((NUM_EXPERTS, HIDDEN), stride=(HIDDEN, 1)))
        atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
        )
        boxes = self.max_slices * self.rt * _BOXES_PER_SLICE
        r_layout = cute.tile_to_shape(atom, (8, _BOX_K, boxes), order=(0, 1, 2))
        tma_atom_r, tma_r = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mR,
            cute.tile_to_shape(atom, (8, _BOX_K), order=(0, 1)),
            (8, _BOX_K),
        )
        return tma_atom_r, tma_r, r_layout

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_x: cute.CopyAtom,
        mXt: cute.Tensor,
        mWS: cute.Tensor,
        mOut: cute.Tensor,
        out1_ptr: cute.Pointer,
        aux0_ptr: cute.Pointer,
        aux1_ptr: cute.Pointer,
        m: Int32,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        sfa_layout: cute.Layout,
        sfb_layout: cute.Layout,
        x_layout: cute.Layout,
        tma_atom_g: Optional[cute.CopyAtom],
        mGt: Optional[cute.Tensor],
        ar_peers: Optional[Int64],
        ar_mc: Optional[Int64],
        ar_flags: Optional[Int64],
        ar_rank: Optional[Int32],
        tma_atom_r: cute.CopyAtom,
        mRt: cute.Tensor,
        r_layout: cute.ComposedLayout,
        x_addr: Int64,
    ):
        bidx, _, _ = cute.arch.block_idx()
        storage = utils.SmemAllocator().allocate(self.gemm.shared_storage)
        if bidx < self.n_gemm_ctas:
            self.gemm.cta(
                tiled_mma,
                tma_atom_a,
                mA,
                tma_atom_x,
                mXt,
                mWS,
                mOut,
                out1_ptr,
                aux0_ptr,
                aux1_ptr,
                m,
                a_layout,
                b_layout,
                sfa_layout,
                sfb_layout,
                x_layout,
                tma_atom_g,
                mGt,
                ar_peers,
                ar_mc,
                ar_flags,
                ar_rank,
                storage,
            )
        else:
            self.router_cta(
                (bidx - self.n_gemm_ctas) // self.S,
                storage,
                tma_atom_r,
                mRt,
                r_layout,
                x_addr,
                m,
                ar_flags,
                ar_peers,
                ar_rank,
            )

    @cute.jit
    def router_cta(
        self,
        cl: Int32,
        storage,
        tma_atom_r: cute.CopyAtom,
        mRt: cute.Tensor,
        r_layout: cute.ComposedLayout,
        x_addr: Int64,
        m: Int32,
        flags: Int64,
        peer_ptrs: Int64,
        ar_rank: Int32,
    ):
        """Router cluster ``cl``: experts ``[8 * rt * cl, 8 * rt * (cl + 1))`` of the router,
        written to this rank's workspace only."""
        S = self.S
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        crank = Int32(0)
        if cutlass.const_expr(S > 1):
            crank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        # Barriers of the GEMM's storage (a_full: one per local slice) and its weight ring.
        w_full = storage.a_full.data_ptr()
        mbox_full = storage.mbox_full
        ring = storage.sA.data_ptr()
        w_base = ring.toint()
        part_base = w_base + self.sw_bytes
        stage_base = part_base + self.part_bytes
        lo = (crank * NUM_SLICES) // S
        hi = ((crank + 1) * NUM_SLICES) // S
        n_sl = hi - lo
        is_last = crank == S - 1
        tile0 = cl * self.rt

        if warp == 0:
            with cute.arch.elect_one():
                for s in cutlass.range_constexpr(self.max_slices):
                    cute.arch.mbarrier_init(w_full + s, 1)
                cute.arch.mbarrier_init(mbox_full, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        if warp == 0:
            with cute.arch.elect_one():
                if is_last:
                    # One arrival plus the other cluster ranks' slice partials.
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbox_full,
                        (NUM_SLICES - self.last_slices) * self.m_tiles * self.rt * _FRAG_BYTES,
                    )
        if cutlass.const_expr(S > 1):
            # Peers write into the last CTA's partials only after its barriers are set up.
            cute.arch.cluster_arrive_relaxed()

        # The weight slices of this CTA, before the wait.
        sW = cute.make_tensor(
            cute.recast_ptr(ring, swizzle_=r_layout.inner, dtype=cutlass.BFloat16), r_layout.outer
        )
        gW = cute.local_tile(mRt, (8, _BOX_K), (None, None))
        tWs, tWg = cpasync.tma_partition(
            tma_atom_r,
            0,
            cute.make_layout(1),
            cute.group_modes(sW, 0, 2),
            cute.group_modes(gW, 0, 2),
        )
        if warp == 0:
            for ls in cutlass.range(n_sl, unroll=1):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        w_full + ls, self.rt * _BOXES_PER_SLICE * _BOX_BYTES
                    )
                for t in cutlass.range_constexpr(self.rt):
                    # Tiles past this rank's experts read zero rows (TMA bounds) and are not pushed.
                    for j in cutlass.range_constexpr(_BOXES_PER_SLICE):
                        cute.copy(
                            tma_atom_r,
                            tWg[(None, tile0 + t, (lo + ls) * _BOXES_PER_SLICE + j)],
                            tWs[(None, (ls * self.rt + t) * _BOXES_PER_SLICE + j)],
                            tma_bar_ptr=w_full + ls,
                            cache_policy=Int64(_EVICT_FIRST),
                        )

        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
            # Every earlier grid has completed: the dependent (the route kernel) may launch.
            _griddepcontrol_launch_dependents()

        if cutlass.const_expr(S > 1):
            cute.arch.cluster_wait()

        # router_gemm_mma_kernel fragments: lane (g, q) holds activation rows g and g + 8 of the
        # 16-row tile and weight row g at k = chunk + 8 q .. + 8.
        g = lane >> 2
        q = lane & 3
        mt_n = self.m_tiles
        for item in cutlass.range(warp, n_sl * mt_n, self.num_warps, unroll=1):
            ls = item // mt_n
            mt = item - ls * mt_n
            s = lo + ls
            r0 = mt * 16 + g
            v0 = Int32(0)
            if r0 < m:
                v0 = Int32(1)
            v1 = Int32(0)
            if r0 + 8 < m:
                v1 = Int32(1)
            a0_base = x_addr + Int64(r0 * (HIDDEN * 2) + (s * SLICE_K + q * 8) * 2)
            a1_base = a0_base + Int64(8 * HIDDEN * 2)
            acts = []
            for cc in cutlass.range_constexpr(CHUNKS):
                a_off = Int64(cc * 64)
                acts.append(
                    (_ldg_v4_b32_if(v0, a0_base + a_off), _ldg_v4_b32_if(v1, a1_base + a_off))
                )
            cute.arch.mbarrier_wait(w_full + ls, 0)
            accs = [[Float32(0.0)] * 4 for _ in range(self.rt)]
            for cc in cutlass.range_constexpr(CHUNKS):
                a0, a1 = acts[cc]
                for t in cutlass.range_constexpr(self.rt):
                    bw = self._wfrag(w_base, ls, t, g, q, cc)
                    acc = _mma_bf16((a0[0], a1[0], a0[1], a1[1]), bw[0], bw[1], accs[t])
                    accs[t] = list(_mma_bf16((a0[2], a1[2], a0[3], a1[3]), bw[2], bw[3], acc))
            for t in cutlass.range_constexpr(self.rt):
                acc = accs[t]
                off = ((s * mt_n + mt) * self.rt + t) * _FRAG_BYTES + lane * 16
                if is_last:
                    _sts_v4_f32(part_base + off, acc[0], acc[1], acc[2], acc[3])
                else:
                    _st_async_v4_f32(
                        part_base + off,
                        acc[0],
                        acc[1],
                        acc[2],
                        acc[3],
                        mbox_full.toint(),
                        Int32(S - 1),
                    )

        if is_last:
            cute.arch.mbarrier_wait(mbox_full, 0)
            cute.arch.barrier()
            # Thread (m tile, expert tile, lane): the 14 slice partials in slice order from 0.
            frags = mt_n * self.rt * 32
            if tidx < frags:
                mt = tidx // (self.rt * 32)
                t = (tidx // 32) - mt * self.rt
                ln = tidx & 31
                tot = [Float32(0.0)] * 4
                for s in cutlass.range_constexpr(NUM_SLICES):
                    off = ((s * mt_n + mt) * self.rt + t) * _FRAG_BYTES + ln * 16
                    v = _lds_v4_f32(part_base + off)
                    for i in cutlass.range_constexpr(4):
                        tot[i] = tot[i] + v[i]
                for i in cutlass.range_constexpr(4):
                    row = mt * 16 + (ln >> 2) + (i >> 1) * 8
                    col = t * 8 + (ln & 3) * 2 + (i & 1)
                    _sts_f32(stage_base + (row * (8 * self.rt) + col) * 4, tot[i])
            cute.arch.barrier()
            cur, _, nbytes, _, _, _, _, _ = lamport.lamport_state(flags)
            vecs = 2 * self.rt
            for p in cutlass.range(tidx, m * vecs, self.threads, unroll=1):
                row = p // vecs
                v = p - row * vecs
                local = tile0 * 8 + v * 4
                if local < NUM_EXPERTS:
                    w = _lds_v4_b32(stage_base + (row * (8 * self.rt) + v * 4) * 4)
                    s0 = _sanitize_f32_bits(w[0])
                    s1 = _sanitize_f32_bits(w[1])
                    s2 = _sanitize_f32_bits(w[2])
                    s3 = _sanitize_f32_bits(w[3])
                    off = Int64((row * NUM_EXPERTS + local) * 4)
                    rel = Int64(cur) * Int64(nbytes) + Int64(self.logits_off) + off
                    lamport.st_v4(
                        lamport.ld_nc_u64(peer_ptrs + Int64(ar_rank) * 8) + rel,
                        s0,
                        s1,
                        s2,
                        s3,
                    )
        if cutlass.const_expr(S > 1):
            # No CTA leaves while a peer may still write into its shared memory.
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

    @cute.jit
    def _wfrag(self, w_base: Int32, ls: Int32, t: int, g: Int32, q: Int32, cc: int):
        """Weight row ``g`` of expert tile ``t`` at chunk ``cc`` of local slice ``ls``."""
        box = (ls * self.rt + t) * _BOXES_PER_SLICE + cc // 2
        unit = (4 * (cc % 2) + q) ^ g
        return _lds_v4_b32(w_base + box * _BOX_BYTES + g * 128 + (unit << 4))


@dsl_user_op
def _sts_f32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Float32(value).ir_value(loc=loc, ip=ip)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------
# Route: routing tail on the polled logits + latent gather / NVFP4
# ---------------------------------------------------------------------------


class K3UllFrontRouteKernel(K3RouteTailKernel):
    """``k3_route_tail``'s cluster (top-16 + ``moe_sort`` of one rank) on the logits the head
    pushes, then ``gather_clusters`` clusters that gather and quantize the latent.

    The grid is ``(1 + gather_clusters) * num_ctas`` CTAs; cluster 0 routes.

    ``publish`` (the experts kernel builds its tables itself, ``k3_moe_ull`` ``moeblk``): no
    ``moe_sort`` tables and no cluster mask exchange; after each token's ids / weights are stored
    the team adds 1 to the 32-bit word at ``num_tiles_addr`` (GPU-scope release), and every CTA
    releases the dependent grid at its start, so the experts' CTAs can be resident and polling
    before the top-k is published (they poll that word; their reads of the latent / output still
    wait for this grid's end).
    """

    def __init__(
        self,
        num_ctas: int,
        num_warps: int,
        warps_per_token: int,
        local_cap: int,
        world: int,
        use_pdl: bool,
        publish: bool = False,
    ):
        super().__init__(num_ctas, num_warps, warps_per_token, 1, not publish, use_pdl, local_cap)
        self.publish = publish
        # k3_route_tail's approximate candidate pass (``tanh.approx`` scores pick the candidates,
        # ``noaux_tc``'s exact score ranks them; the same selection, see ROUTE_APPROX).
        self.approx = True
        self.world = world
        self.slot_n = slot_cols(world)
        self.logits_off = logits_offset(world)
        self.tiles_base, self.tiles_rem = divmod(LATENT_TILES, world)
        self.gather_clusters = GATHER_CLUSTERS
        self.grid = (1 + GATHER_CLUSTERS) * num_ctas

    def load_logits(self, row: Int64, g: Int32, stride: int, n: int):
        """The logits polled from the workspace (all ``n`` loads in flight per sweep)."""
        return _poll_words(row + Int64(g * 4), stride * 4, n)

    def load_logits4(self, row: Int64, lane: Int32, n: int):
        return _poll_words4(row + Int64(lane * 16), 128 * 4, n)

    @cute.jit
    def __call__(
        self,
        m: Int32,
        uc_local: Int64,
        flags: Int64,
        bias_addr: Int64,
        ids_addr: Int64,
        vals_addr: Int64,
        e2p_addr: Int64,
        p2e_addr: Int64,
        tile_expert_addr: Int64,
        tile_limit_addr: Int64,
        total_addr: Int64,
        num_tiles_addr: Int64,
        local_offset: Int32,
        local_num: Int32,
        scale_bits: Int64,
        codes_addr: Int64,
        sf_addr: Int64,
        gsf_addr: Int64,
        zero_addr: Int64,
        zero_words: Int32,
        stream: cuda.CUstream,
    ):
        teams = self.teams

        @cute.struct
        class SharedStorage:
            bias: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, NUM_EXPERTS], 16]
            mask: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, self.local_cap], 16]
            masks: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, self.num_ctas * self.local_cap], 16
            ]
            base: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, self.local_cap], 16]
            ids: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 32 * TOP_K], 16]
            cand: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, self.num_warps * self.region_slots * 4], 16
            ]
            top: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, teams * TOP_K * 2], 16]
            xchg: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, teams * 16], 16]
            warp_tot: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, self.num_warps], 16]

        self.shared_storage = SharedStorage
        self.kernel(
            m,
            uc_local,
            flags,
            bias_addr,
            ids_addr,
            vals_addr,
            e2p_addr,
            p2e_addr,
            tile_expert_addr,
            tile_limit_addr,
            total_addr,
            num_tiles_addr,
            local_offset,
            local_num,
            scale_bits,
            codes_addr,
            sf_addr,
            gsf_addr,
            zero_addr,
            zero_words,
        ).launch(
            grid=(self.grid, 1, 1),
            block=(self.threads, 1, 1),
            cluster=(self.num_ctas, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        m: Int32,
        uc_local: Int64,
        flags: Int64,
        bias_addr: Int64,
        ids_addr: Int64,
        vals_addr: Int64,
        e2p_addr: Int64,
        p2e_addr: Int64,
        tile_expert_addr: Int64,
        tile_limit_addr: Int64,
        total_addr: Int64,
        num_tiles_addr: Int64,
        local_offset: Int32,
        local_num: Int32,
        scale_bits: Int64,
        codes_addr: Int64,
        sf_addr: Int64,
        gsf_addr: Int64,
        zero_addr: Int64,
        zero_words: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.publish and self.use_pdl):
            # the experts poll the published top-k and wait for this grid before the latent
            _griddepcontrol_launch_dependents()
        # The halves' flags never change: no ordering needed to read them.
        cur, _, nbytes, _, _, _, _, _ = lamport.lamport_state(flags)
        half = uc_local + Int64(cur) * Int64(nbytes)
        if bidx < self.num_ctas:
            self.route_cta(
                bidx,
                tidx,
                m,
                half + Int64(self.logits_off),
                bias_addr,
                ids_addr,
                vals_addr,
                e2p_addr,
                p2e_addr,
                tile_expert_addr,
                tile_limit_addr,
                total_addr,
                num_tiles_addr,
                local_offset,
                local_num,
                scale_bits,
            )
        else:
            self.gather_cta(
                bidx - self.num_ctas,
                tidx,
                m,
                half,
                codes_addr,
                sf_addr,
                gsf_addr,
                zero_addr,
                zero_words,
            )
        if cutlass.const_expr(self.use_pdl):
            # Complete only after the head (whose CTAs are done pushing by now).
            _griddepcontrol_wait()

    @cute.jit
    def route_cta(
        self,
        rank: Int32,
        tidx: Int32,
        m: Int32,
        logits_addr: Int64,
        bias_addr: Int64,
        ids_addr: Int64,
        vals_addr: Int64,
        e2p_addr: Int64,
        p2e_addr: Int64,
        tile_expert_addr: Int64,
        tile_limit_addr: Int64,
        total_addr: Int64,
        num_tiles_addr: Int64,
        local_offset: Int32,
        local_num: Int32,
        scale_bits: Int64,
    ):
        """``K3RouteTailKernel.kernel`` without the grid-dependency wait before the logits (they
        are polled), re-arming each token's logits once its team is done with them."""
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        bias_smem = storage.bias.data_ptr().toint()
        mask_smem = storage.mask.data_ptr().toint()
        masks_smem = storage.masks.data_ptr().toint()
        base_smem = storage.base.data_ptr().toint()
        ids_smem = storage.ids.data_ptr().toint()
        cand0 = storage.cand.data_ptr().toint()
        top0 = storage.top.data_ptr().toint()
        xchg0 = storage.xchg.data_ptr().toint()
        warp_tot_smem = storage.warp_tot.data_ptr().toint()
        team = warp // self.team_warps
        team_warp = warp - team * self.team_warps
        cand_smem = cand0 + team * (self.team_warps * self.region_slots * 16)
        top_smem = top0 + team * (TOP_K * 8)
        xchg_smem = xchg0 + team * 64

        per_cta = (m + self.num_ctas - 1) // self.num_ctas
        t_begin = rank * per_cta
        t_end = t_begin + per_cta
        if t_end > m:
            t_end = m

        for j in cutlass.range(tidx, local_num, self.threads, unroll=1):
            _sts_b32(mask_smem + j * 4, Int32(0))
        for e4 in cutlass.range(tidx, NUM_EXPERTS // 4, self.threads, unroll=1):
            v = _ld_nc_v4_f32(bias_addr + Int64(e4 * 16))
            _sts_v4_f32(bias_smem + e4 * 16, v[0], v[1], v[2], v[3])
        cute.arch.barrier()

        stride = self.team_lanes
        gl = team_warp * 32 + lane
        for token in cutlass.range(t_begin + team, t_end, self.teams, unroll=1):
            self.topk_token(
                token,
                token - t_begin,
                team,
                team_warp,
                lane,
                Int32(M_MAX),
                logits_addr,
                bias_smem,
                ids_addr,
                vals_addr,
                scale_bits,
                Int64(0),
                cand_smem,
                top_smem,
                xchg_smem,
                ids_smem,
                mask_smem,
                local_offset,
                local_num,
            )
            # Every lane of the team is done with the token's logits: re-arm them.
            self.team_sync(team)
            row = logits_addr + Int64(token) * Int64(NUM_EXPERTS * 4)
            for s in cutlass.range_constexpr(self.slots):
                _st_sentinel_b32(row + Int64((s * stride + gl) * 4))
            if cutlass.const_expr(self.publish):
                # the team's first warp stored the ids / weights (lanes < 16): publish the token
                if team_warp == 0:
                    _syncwarp()
                    if lane == 0:
                        _red_add_release_gpu(num_tiles_addr, 1)
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_launch_dependents()
        if cutlass.const_expr(not self.publish):
            cute.arch.barrier()
            if cutlass.const_expr(self.num_ctas > 1):
                for j in cutlass.range(tidx, local_num, self.threads, unroll=1):
                    word = _lds_b32(mask_smem + j * 4)
                    dst = masks_smem + (rank * self.local_cap + j) * 4
                    for c in cutlass.range_constexpr(self.num_ctas):
                        _st_dsmem_b32(dst, Int32(c), word)
                _cluster_arrive_release()
                _cluster_wait_acquire()
            self.sort(
                tidx,
                warp,
                lane,
                rank,
                t_begin,
                t_end,
                mask_smem,
                masks_smem,
                base_smem,
                ids_smem,
                warp_tot_smem,
                e2p_addr,
                p2e_addr,
                tile_expert_addr,
                tile_limit_addr,
                total_addr,
                num_tiles_addr,
                local_offset,
                local_num,
            )

    @cute.jit
    def gather_cta(
        self,
        gidx: Int32,
        tidx: Int32,
        m: Int32,
        half: Int64,
        codes_addr: Int64,
        sf_addr: Int64,
        gsf_addr: Int64,
        zero_addr: Int64,
        zero_words: Int32,
    ):
        """Items of 32 latent columns of one token: poll the owner rank's slot, quantize two
        16-element NVFP4 groups (``fp4_quantize``), store, re-arm. First, ``zero_words`` 32-bit
        words at ``zero_addr`` are zeroed (the routed experts' output and counters, which FC12
        then does not clear itself)."""
        nthreads = self.gather_clusters * self.num_ctas * self.threads
        gtid = gidx * self.threads + tidx
        for i in cutlass.range(gtid, zero_words // 4, nthreads, unroll=1):
            lamport.st_v4(zero_addr + Int64(i) * 16, Int32(0), Int32(0), Int32(0), Int32(0))
        tail = zero_words // 4 * 4
        if gtid < zero_words - tail:
            _st_global_b32(zero_addr + Int64(tail + gtid) * 4, Int32(0))
        gsf = _ldg_f32(gsf_addr)
        rgsf = _rcp_approx_ftz(gsf)
        rcp6 = _rcp_approx_ftz(Float32(6.0))
        items_per_row = LATENT // _ITEM_COLS
        big = self.tiles_base + 1
        split = self.tiles_rem * big
        for p in cutlass.range(gtid, m * items_per_row, nthreads, unroll=1):
            t = p // items_per_row
            j = p - t * items_per_row
            tile = j >> 2
            owner = Int32(0)
            t0 = Int32(0)
            if tile < split:
                owner = tile // big
                t0 = owner * big
            else:
                owner = self.tiles_rem + (tile - split) // self.tiles_base
                t0 = split + (owner - self.tiles_rem) * self.tiles_base
            lcol = j * _ITEM_COLS - t0 * TILE
            src = half + Int64(((t * self.world + owner) * self.slot_n + lcol) * 2)
            words = _poll_raw4(src, 16, 4)
            vals = []
            for i in cutlass.range_constexpr(16):
                lo_f, hi_f = _bf16x2_to_f32x2(words[i])
                vals.extend((lo_f, hi_f))
            amax0 = _abs_max_f32x16(vals[:16])
            amax1 = _abs_max_f32x16(vals[16:])
            sf2, os0, os1 = _nvfp4_scales_x2(amax0, amax1, gsf, rgsf, rcp6)
            _stg_v4_b32(
                codes_addr + Int64(t) * (LATENT // 2) + Int64(j * (_ITEM_COLS // 2)),
                _e2m1x8(vals[0:8], os0),
                _e2m1x8(vals[8:16], os0),
                _e2m1x8(vals[16:24], os1),
                _e2m1x8(vals[24:32], os1),
            )
            _stg_u16(
                sf_addr + Int64(t) * (LATENT // _FP4_VEC) + Int64(j * (_ITEM_COLS // _FP4_VEC)),
                sf2,
            )
            for v in cutlass.range_constexpr(4):
                lamport.st_sentinel_v4(src + Int64(v * 16))
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_launch_dependents()


@dsl_user_op
def _st_global_b32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _i2f(word, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [Int32(word).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,r",
            loc=loc,
            ip=ip,
        )
    )


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class UllFrontWorkspace:
    """The front's MNNVL multicast workspace of one TP group: two halves of
    :func:`half_bytes`; ``flags[h]`` is the constant rotation state (``k3_lamport_ar`` layout) that
    points the kernels at half ``h``."""

    world: int
    rank: int
    uc_local: int
    mc_base: int
    peer_ptrs: torch.Tensor
    flags: Tuple[torch.Tensor, torch.Tensor]
    half_bytes: int
    handle: object = None
    calls: int = 0

    def next_flags(self) -> torch.Tensor:
        """The flags of the next call (the halves alternate call by call)."""
        flags = self.flags[self.calls & 1]
        self.calls += 1
        return flags


_WORKSPACES: Dict[object, UllFrontWorkspace] = {}
_lock = threading.Lock()


def ull_front_workspace(mapping, comm=None) -> UllFrontWorkspace:
    """The front workspace of ``mapping``'s TP group, created on first use (collective: every rank
    of the group calls it at the same point). ``comm``: the group's MPI communicator (default: the
    one of the MNNVL all-reduce workspace)."""
    ws = _WORKSPACES.get(mapping)
    if ws is not None:
        return ws
    from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace
    from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer

    from ..comm.lamport import mnnvl_workspace_mpi_comm

    if comm is None:
        comm = mnnvl_workspace_mpi_comm(
            get_or_scale_allreduce_mnnvl_workspace(mapping, torch.bfloat16)
        )
    world, rank = mapping.tp_size, mapping.tp_rank
    if not supports_world(world):
        raise ValueError(f"k3_ull_front does not serve TP world {world}")
    nbytes = half_bytes(world)
    use_fabric = os.environ.get("TRTLLM_FORCE_MNNVL_AR", "0") == "1" or mapping.is_multi_node()
    handle = McastGPUBuffer(2 * nbytes, world, rank, mapping.local_rank, use_fabric, comm.py2f())
    buf = handle.get_uc_buffer(rank, (2 * nbytes // 4,), torch.int32, 0)
    buf.fill_(SENTINEL)
    torch.cuda.synchronize()
    comm.Barrier()
    peers = [handle.get_uc_buffer(r, (1,), torch.uint8, 0).data_ptr() for r in range(world)]
    flags = tuple(
        torch.tensor([h, 0, nbytes, 0, 0, 0, 0, 0, 0], dtype=torch.int32, device=buf.device)
        for h in (0, 1)
    )
    ws = UllFrontWorkspace(
        world=world,
        rank=rank,
        uc_local=buf.data_ptr(),
        mc_base=handle.get_mc_buffer((1,), torch.uint8, 0).data_ptr(),
        peer_ptrs=torch.tensor(peers, dtype=torch.int64, device=buf.device),
        flags=flags,
        half_bytes=nbytes,
        handle=handle,
    )
    _WORKSPACES[mapping] = ws
    return ws


@dataclasses.dataclass
class UllFrontPlan:
    """One MoE layer's front operands on one rank (views of the model's weights plus the packed
    scale of the rank's latent_down tiles)."""

    world: int
    rank: int
    latent_w: torch.Tensor  # FP8 [128 * tiles of the rank, 7168]
    latent_ws: torch.Tensor  # DeepGEMM-packed UE8M0 scale of those rows (MN-major int32)
    router_w: torch.Tensor  # BF16 [896, 7168]
    bias: torch.Tensor  # FP32 [896]
    routed_scaling_factor: float
    global_sf: torch.Tensor  # FP32 [1]: the NVFP4 global scale of the routed experts' input
    local_offset: int  # this rank's routed experts (moe_sort tables)
    local_num: int


def make_plan(
    latent_w: torch.Tensor,
    latent_scale: torch.Tensor,
    router_w: torch.Tensor,
    bias: torch.Tensor,
    routed_scaling_factor: float,
    global_sf: torch.Tensor,
    world: int,
    rank: int,
    local_offset: int,
    local_num: int,
) -> UllFrontPlan:
    """The front plan of one layer: ``latent_w`` FP8 ``[3584, 7168]`` with its 128x128 block scale
    in either layout of ``_Fp8BlockScaleWeightReadLinear`` (Rubin MXFP8 R128c4 or DeepGEMM-packed),
    ``router_w`` BF16 ``[896, 7168]``."""
    if not supports_world(world):
        raise ValueError(f"k3_ull_front does not serve TP world {world}")
    if tuple(latent_w.shape) != (LATENT, HIDDEN) or latent_w.dtype != torch.float8_e4m3fn:
        raise ValueError("latent_w must be FP8 [3584, 7168]")
    if tuple(router_w.shape) != (NUM_EXPERTS, HIDDEN) or router_w.dtype != torch.bfloat16:
        raise ValueError("router_w must be BF16 [896, 7168]")
    t0, nt = latent_tiles_of(rank, world)
    return UllFrontPlan(
        world=world,
        rank=rank,
        latent_w=latent_w[t0 * TILE : (t0 + nt) * TILE],
        latent_ws=lowm.pack_ue8m0_exponents(
            lowm.block_exponents_of(latent_scale, LATENT, HIDDEN)[t0 : t0 + nt].contiguous(),
            nt * TILE,
            HIDDEN,
        ),
        router_w=router_w,
        bias=bias.contiguous(),
        routed_scaling_factor=float(routed_scaling_factor),
        global_sf=global_sf.reshape(-1)[:1],
        local_offset=local_offset,
        local_num=local_num,
    )


@functools.cache
def latent_tactic(m: int) -> "lowm.K3LowMTactic":
    """The latent GEMM's tactic: the low-M kernel's split-K tactic ``k3_latent_down_fp4`` uses at
    ``M <= 16`` (``k3_latent_fp8_gemm.route_tactic``, 3 cluster ranks); above, the low-M kernel
    with the mid-M route's 4 ranks (the mid-M kernel has no all-gather layout)."""
    return lowm.split_tactic(m, LATENT, HIDDEN, 3 if m <= 16 else 4, mid=False)


def _m_tiles(mma_n: int) -> int:
    return (mma_n + 15) // 16


@functools.cache
def _compiled_head(n: int, tactic, world: int, use_pdl: bool):
    gemm = lowm.K3Fp8wLowMGemmKernel(
        n,
        HIDDEN,
        tactic,
        use_pdl,
        ar_push=lowm.AR_PUSH_ONE_SHOT,
        ar_world=world,
        ar_slot_n=slot_cols(world),
    )
    head = K3UllFrontHeadKernel(
        gemm,
        ROUTER_TILES,
        _m_tiles(tactic.mma_n),
        logits_offset(world),
        use_pdl,
    )
    dummy = 1 << 20
    addr = Int64(dummy)
    return cute.compile(
        head,
        make_ptr(cutlass.Float8E4M3FN, dummy, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.BFloat16, dummy, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(cutlass.Uint8, dummy, cute.AddressSpace.gmem, assumed_align=16),
        Int32(1),
        addr,
        addr,
        addr,
        Int32(0),
        make_ptr(cutlass.BFloat16, dummy, cute.AddressSpace.gmem, assumed_align=16),
        addr,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


#: Local-expert capacity of the routing tail's sort build (as ``k3_route_tail``): 224 (EP >= 4),
#: else every expert (e.g. TP8 x EP2: 448 local experts).
_SMALL_LOCAL_CAP = 224


def _local_cap(local_num: int) -> int:
    return _SMALL_LOCAL_CAP if local_num <= _SMALL_LOCAL_CAP else NUM_EXPERTS


#: Warps routing one token (measured best at M 8 / 16 / 32 on Rubin).
_TEAM = 2


def route_config(m: int) -> Tuple[int, int, int]:
    """``(num_ctas, warps_per_token, num_warps)`` of the route cluster for ``m`` rows: eight CTAs,
    one team per token of the CTA (``ceil(m / 8)`` tokens)."""
    per_cta = (m + MAX_CTAS - 1) // MAX_CTAS
    return MAX_CTAS, _TEAM, max(_TEAM * per_cta, 8)


@functools.cache
def _compiled_route(
    num_ctas: int,
    num_warps: int,
    warps_per_token: int,
    local_cap: int,
    world: int,
    use_pdl: bool,
    publish: bool = False,
):
    kernel = K3UllFrontRouteKernel(
        num_ctas,
        num_warps,
        warps_per_token,
        local_cap,
        world,
        use_pdl,
        publish,
    )
    addr = Int64(1 << 20)
    return cute.compile(
        kernel,
        Int32(1),
        *([addr] * 11),
        Int32(0),
        Int32(NUM_EXPERTS),
        Int64(0),
        addr,
        addr,
        addr,
        addr,
        Int32(0),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


#: Launch shapes compiled so far (see :func:`is_compiled`).
_READY: set = set()


def _shape_key(m: int, plan: UllFrontPlan, publish: bool = False) -> tuple:
    return (
        publish,
        plan.latent_w.shape[0],
        latent_tactic(m),
        plan.world,
        _local_cap(plan.local_num),
        route_config(m),
    )


def is_compiled(m: int, plan: UllFrontPlan, publish: bool = False) -> bool:
    """Whether a call of ``m`` rows with this plan's shapes runs without compiling (required
    inside CUDA-graph capture)."""
    return _shape_key(m, plan, publish) in _READY


def _gptr(dtype, t: torch.Tensor):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def can_run(x: torch.Tensor, plan: UllFrontPlan) -> bool:
    return (
        x.is_cuda
        and torch.cuda.get_device_capability(x.device) in ((10, 0), (10, 3), (10, 7))
        and x.dtype == torch.bfloat16
        and x.dim() == 2
        and x.is_contiguous()
        and x.shape[1] == HIDDEN
        and 1 <= x.shape[0] <= M_MAX
        and x.data_ptr() % 16 == 0
        and 1 <= plan.local_num
        and plan.local_offset + plan.local_num <= NUM_EXPERTS
    )


def run_k3_ull_front(
    x: torch.Tensor,
    plan: UllFrontPlan,
    ws: UllFrontWorkspace,
    *,
    zero: Optional[torch.Tensor] = None,
    after_head=None,
    use_pdl: bool = True,
    publish: int = 0,
) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    """Launch the head and the route kernel on the current stream.

    Returns ``(routing, codes, scales)``: ``routing`` = ``k3_route_tail``'s 8 outputs (top-k ids
    and weights, then the ``moe_sort`` tables of ``plan.local_offset`` / ``local_num`` at tile 128),
    ``codes`` uint8 ``[M, 1792]`` / ``scales`` uint8 ``[M, 224]`` the NVFP4 latent (linear scale
    layout), i.e. ``k3_latent_down_fp4``'s outputs. ``zero``: an int32 tensor the route kernel
    zeroes (the routed experts' output + FC12 counters, ``Fc12ZeroedOutput.buffer``).
    ``after_head``: called between the two launches (e.g. to fork a side stream after the head).
    ``publish``: address of an int32 word (zero) the route kernel adds 1 to per routed token
    (release) instead of building the ``moe_sort`` tables (``routing[2:]`` are then not written):
    the ULL experts kernel's ``moeblk`` mode builds them itself. The first call of each shape
    compiles (outside CUDA-graph capture)."""
    m = x.shape[0]
    if not can_run(x, plan):
        raise ValueError(f"k3_ull_front: unsupported call x {tuple(x.shape)} {x.dtype}")
    world = plan.world
    tactic = latent_tactic(m)
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    flags = ws.next_flags()
    head = _compiled_head(plan.latent_w.shape[0], tactic, world, use_pdl)
    head(
        _gptr(cutlass.Float8E4M3FN, plan.latent_w),
        _gptr(cutlass.BFloat16, x),
        _gptr(cutlass.Uint8, plan.latent_ws),
        Int32(m),
        Int64(ws.peer_ptrs.data_ptr()),
        Int64(ws.mc_base),
        Int64(flags.data_ptr()),
        Int32(ws.rank),
        _gptr(cutlass.BFloat16, plan.router_w),
        Int64(x.data_ptr()),
        stream,
    )
    if after_head is not None:
        after_head()
    device = x.device
    max_tiles, max_permuted = sort_buffer_sizes(m, plan.local_num, TILE)
    i32 = dict(dtype=torch.int32, device=device)
    routing = (
        torch.empty((m, TOP_K), **i32),
        torch.empty((m, TOP_K), dtype=torch.float32, device=device),
        torch.empty((max_tiles,), **i32),
        torch.empty((max_tiles,), **i32),
        torch.empty((m, TOP_K), **i32),
        torch.empty((max_permuted,), **i32),
        torch.empty((1,), **i32),
        torch.empty((1,), **i32),
    )
    codes = torch.empty((m, LATENT // 2), dtype=torch.uint8, device=device)
    scales = torch.empty((m, LATENT // _FP4_VEC), dtype=torch.uint8, device=device)
    tile_expert, tile_limit, e2p, p2e, total, num_tiles = (t.data_ptr() for t in routing[2:])
    if publish:
        num_tiles = publish
    scale_bits = struct.unpack("<q", struct.pack("<d", plan.routed_scaling_factor))[0]
    num_ctas, team, warps = route_config(m)
    route = _compiled_route(
        num_ctas,
        warps,
        team,
        _local_cap(plan.local_num),
        world,
        use_pdl,
        bool(publish),
    )
    route(
        Int32(m),
        Int64(ws.uc_local),
        Int64(flags.data_ptr()),
        Int64(plan.bias.data_ptr()),
        Int64(routing[0].data_ptr()),
        Int64(routing[1].data_ptr()),
        Int64(e2p),
        Int64(p2e),
        Int64(tile_expert),
        Int64(tile_limit),
        Int64(total),
        Int64(num_tiles),
        Int32(plan.local_offset),
        Int32(plan.local_num),
        Int64(scale_bits),
        Int64(codes.data_ptr()),
        Int64(scales.data_ptr()),
        Int64(plan.global_sf.data_ptr()),
        Int64(zero.data_ptr() if zero is not None else 0),
        Int32(zero.numel() if zero is not None else 0),
        stream,
    )
    _READY.add(_shape_key(m, plan, bool(publish)))
    return routing, codes, scales


__all__ = [
    "M_MAX",
    "UllFrontPlan",
    "UllFrontWorkspace",
    "can_run",
    "latent_tactic",
    "latent_tiles_of",
    "make_plan",
    "run_k3_ull_front",
    "slot_cols",
    "supports_world",
    "is_compiled",
    "ull_front_workspace",
]
