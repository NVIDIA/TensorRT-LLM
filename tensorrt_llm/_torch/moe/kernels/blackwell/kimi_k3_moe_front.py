# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kimi-K3 MoE decode front sharded over the routed all-reduce's group (``M <= 32`` rows).

Computes, per MoE layer and rank, the router logits (as ``kimi_k3_router_logits``), the routing
(as ``kimi_k3_routing``) and the latent_down projection with its NVFP4 output (as
``fp8_block_scale_low_m_gemm.run_low_m_gemm_nvfp4``) in two kernels on one stream, the latent GEMM
sharded over the routed all-reduce's group:

* **Head** (:class:`KimiK3MoEFrontHeadKernel`, one grid, cluster size = the latent GEMM's split-K):

  - *Latent CTAs*: the FP8 block-scale low-M GEMM (``Fp8BlockScaleLowMGemmKernel.cta``) on this
    rank's 128-row tiles of latent_down (28 tiles over ``W`` ranks, :func:`latent_tiles_of`), at
    the split-K of the replicated latent_down. Its one-shot all-reduce push epilogue multicasts
    the BF16 rows into slot ``(token, rank)`` of the front workspace of every rank (``-0`` sent as
    ``+0``).
  - *Router CTAs*: the router logits of all 896 experts on every rank (no exchange to wait for),
    written into this rank's workspace only. A cluster owns ``8 * ROUTER_TILES`` experts; its CTAs
    split the 14 K slices of 512 and each warp runs one ``mma.sync`` chain per (slice, 16-row tile,
    8-expert tile) as ``router_gemm_mma_kernel`` does; the cluster's last CTA sums the 14 slice
    partials in slice order from 0 and stores the FP32 logits. Each 16-row tile gets the bits
    ``dsv3_router_gemm_op`` gives those 16 rows (``-0`` stored as ``+0``, which routes the same).

  Both roles read only weights before ``griddepcontrol.wait`` and release the dependent after it.

* **Route** (:class:`KimiK3MoEFrontRouteKernel`, PDL dependent of the head): the routing kernel's
  cluster (top-16 + ``moe_sort`` tables of this rank, ``NoauxTcRouteSortKernel``) on logits it
  polls from the workspace (sentinel ``0x80000000``) instead of reading a tensor, and gather CTAs
  that poll every rank's latent slots and write the NVFP4 quantization of the gathered BF16 latent
  (``fp4_quantize``'s arithmetic, linear scale layout). Every reader re-arms what it read.
  With ``publish`` the route kernel writes no ``moe_sort`` tables: it releases its dependent (the
  routed experts) at entry and publishes each token's top-k (stores, then a GPU-scope release add
  of 1 to a word of the experts), which build their tables from it while this grid still runs.

Workspace (:class:`MoEFrontWorkspace`): an MNNVL multicast buffer of two halves. Rank ``r`` may
write half ``h`` of a peer only after every rank finished the call that last used ``h``. Two
orderings provide that:

* Alternation: consecutive calls on a stream use the other half (the half is a launch argument
  through a constant flags tensor, picked by a host counter), and between two calls on a half
  there is the other half's call, whose route kernel on ``r`` waited for data every peer pushed
  after its own previous call. A CUDA graph replays the halves it was captured with, so the
  alternation holds across replays only for a graph of an even number of calls captured at the
  counter's parity its replays follow.
* A collective in between: the caller runs, between two calls, a collective over the same group
  that consumes every peer's output of the earlier call (the model's routed all-reduce of each
  MoE layer, whose input the routed experts produce after the route kernel's grid completes).
  This holds for any halves, so the model's calls need no particular count per graph.
"""

from __future__ import annotations

import dataclasses
import struct
from typing import Callable, Dict, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

from ....kernels.blackwell import fp8_block_scale_low_m_gemm as low_m
from ....kernels.blackwell import mnnvl_pushed_allreduce as pushed_ar
from ....kernels.blackwell.raw_ptx import (
    bf16x2_to_f32x2,
    ld_volatile_v4,
    rcp_approx_ftz,
    st_u16,
    st_v4,
)
from ....kernels.blackwell.raw_ptx import mma_bf16 as _mma_bf16
from . import raw_ptx as ptx
from .kimi_k3_router_logits import _ldg_v4_b32_if, _st_async_v4_f32
from .kimi_k3_routing import (
    _MAX_TOKENS_PER_CTA,
    _SMALL_LOCAL_CAP,
    CLUSTER_SIZE,
    TILE_SIZE,
    TOP_K,
    NoauxTcRouteSortKernel,
    _config,
    sort_buffer_sizes,
)

#: Kimi-K3 shapes.
HIDDEN = 7168
LATENT = 3584
NUM_EXPERTS = 896
#: 128-row tiles of latent_down (its output columns), split over the ranks.
LATENT_TILES = LATENT // TILE_SIZE
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
#: 8-expert tiles per router cluster.
ROUTER_TILES = 4
#: Gather clusters of the route kernel (of ``CLUSTER_SIZE`` CTAs each).
GATHER_CLUSTERS = 1
#: Empty-slot word of the workspace (``mnnvl_pushed_allreduce``'s sentinel, ``-0.0f``).
SENTINEL = -(1 << 31)
#: NVFP4 group (fp4_quantize's scaling vector) and the gather item (32 latent columns).
_FP4_VEC = 16
_ITEM_COLS = 32
#: L2 eviction hint for the streamed router weight (CacheHintSm90::EVICT_FIRST).
_EVICT_FIRST = 0x12F0000000000000

assert HIDDEN % (NUM_SLICES * 32) == 0 and M_MAX <= _MAX_TOKENS_PER_CTA * CLUSTER_SIZE


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
    return (base + (1 if rem else 0)) * TILE_SIZE


def supports_world(world: int) -> bool:
    """Whether the front serves a group of ``world`` ranks (every rank owns a latent tile)."""
    return world in pushed_ar.WORLD_SIZES and LATENT_TILES >= world


def _align(a: int, b: int) -> int:
    return (a + b - 1) // b * b


def logits_offset(world: int) -> int:
    """Byte offset of the logits ``fp32 [M_MAX, 896]`` in a workspace half (after the latent
    slots ``bf16 [M_MAX, W, slot_cols]``)."""
    return _align(M_MAX * world * slot_cols(world) * 2, 1024)


def half_bytes(world: int) -> int:
    """Bytes of one workspace half."""
    return _align(logits_offset(world) + M_MAX * NUM_EXPERTS * 4, 1 << 16)


# ---------------------------------------------------------------------------
# PTX helpers
# ---------------------------------------------------------------------------


@dsl_user_op
def _ld_volatile_b32(addr, *, loc=None, ip=None):
    return Int32(
        ptx.inline_asm(
            T.i32(),
            [ptx.i64(addr, loc=loc, ip=ip)],
            "ld.volatile.global.b32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_nc_u64(addr, *, loc=None, ip=None):
    """A 64-bit word of a tensor no kernel writes (the peer pointer table)."""
    return Int64(
        ptx.inline_asm(
            T.i64(),
            [ptx.i64(addr, loc=loc, ip=ip)],
            "ld.global.nc.u64 $0, [$1];",
            "=l,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_sentinel_b32(addr, *, loc=None, ip=None):
    ptx.inline_asm(
        None,
        [ptx.i64(addr, loc=loc, ip=ip)],
        "{\n.reg .b32 t;\nmov.b32 t, 0x80000000;\nst.global.b32 [$0], t;\n}",
        "l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_sentinel_v4(addr, *, loc=None, ip=None):
    ptx.inline_asm(
        None,
        [ptx.i64(addr, loc=loc, ip=ip)],
        "{\n.reg .b32 t;\nmov.b32 t, 0x80000000;\nst.global.v4.b32 [$0], {t, t, t, t};\n}",
        "l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _red_add_release_gpu(addr, value, *, loc=None, ip=None):
    ptx.inline_asm(
        None,
        [ptx.i64(addr, loc=loc, ip=ip), ptx.i32(value, loc=loc, ip=ip)],
        "red.release.gpu.global.add.s32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sanitize_f32_bits(word, *, loc=None, ip=None):
    """``-0.0`` (the sentinel) sent as ``+0.0``."""
    return Int32(
        ptx.inline_asm(
            T.i32(),
            [ptx.i32(word, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.eq.b32 p, $1, 0x80000000;\nselp.b32 $0, 0, $1, p;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _i2f(word, *, loc=None, ip=None):
    return Float32(
        ptx.inline_asm(
            T.f32(), [ptx.i32(word, loc=loc, ip=ip)], "mov.b32 $0, $1;", "=f,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _sts_f32(addr, value, *, loc=None, ip=None):
    ptx.inline_asm(
        None,
        [ptx.i32(addr, loc=loc, ip=ip), ptx.f32(value, loc=loc, ip=ip)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        loc=loc,
        ip=ip,
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
def _poll_vectors(base: Int64, step: int, n: int):
    """``n`` 16-byte vectors at ``base + i * step`` bytes, polled until none holds the sentinel;
    returns their ``4 n`` 32-bit words."""
    words = cute.make_rmem_tensor((4 * n,), Int32)
    ready = Int32(0)
    while ready == 0:
        ready = Int32(1)
        for i in cutlass.range_constexpr(n):
            w = ld_volatile_v4(base + Int64(i * step))
            for j in cutlass.range_constexpr(4):
                words[4 * i + j] = w[j]
            ready = ready & pushed_ar.arrived_v4(*w)
    return [words[i] for i in range(4 * n)]


# ---------------------------------------------------------------------------
# Head: latent GEMM CTAs + router CTAs
# ---------------------------------------------------------------------------


class KimiK3MoEFrontHeadKernel:
    """The head grid: the latent GEMM's ``n_tiles * split_k`` CTAs, then the router clusters.

    Args:
        gemm: The latent GEMM of this rank's tiles (``Fp8BlockScaleLowMGemmKernel`` with the
            one-shot all-reduce push, ``push_slot_n`` = :func:`slot_cols`).
        router_tiles: 8-expert tiles per router cluster.
        m_tiles: 16-row tiles of the router (``ceil(mma_n / 16)``).
        logits_off: Byte offset of the logits in a workspace half.
    """

    def __init__(
        self,
        gemm: low_m.Fp8BlockScaleLowMGemmKernel,
        router_tiles: int,
        m_tiles: int,
        logits_off: int,
    ):
        self.gemm = gemm
        self.S = gemm.split_k
        self.threads = gemm.threads
        self.num_warps = self.threads // 32
        self.rt = router_tiles
        self.m_tiles = m_tiles
        self.logits_off = logits_off
        self.router_clusters = (NUM_EXPERTS // 8 + router_tiles - 1) // router_tiles
        self.n_gemm_ctas = gemm.n_tiles * self.S
        self.grid = self.n_gemm_ctas + self.router_clusters * self.S
        # Slices of cluster rank r: [NUM_SLICES * r // S, NUM_SLICES * (r + 1) // S).
        self.max_slices = max(
            NUM_SLICES * (r + 1) // self.S - NUM_SLICES * r // self.S for r in range(self.S)
        )
        self.last_slices = NUM_SLICES - NUM_SLICES * (self.S - 1) // self.S
        if self.max_slices > gemm.a_stages:
            raise ValueError("the router uses one GEMM weight-stage barrier per local slice")
        self.sw_bytes = self.max_slices * router_tiles * _BOXES_PER_SLICE * _BOX_BYTES
        self.part_bytes = NUM_SLICES * m_tiles * router_tiles * _FRAG_BYTES
        self.stage_bytes = M_MAX * 8 * router_tiles * 4
        # The router CTAs keep their buffers in the GEMM's shared storage from its weight ring on
        # (one allocation for both roles): weights, slice partials, logits (checked at trace).
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
        (
            tiled_mma,
            tma_atom_a,
            mA,
            tma_atom_x,
            mXt,
            mWS,
            mOut,
            m_arg,
            a_layout,
            b_layout,
            sfa_layout,
            sfb_layout,
            x_layout,
            tma_atom_g,
            mGt,
            mSF,
            mGSF,
            push,
        ) = self.gemm.kernel_args(
            w_ptr,
            x_ptr,
            None,
            None,
            ws_ptr,
            x_ptr,
            None,
            None,
            m,
            (peer_ptrs, mc_base, flags, rank),
        )
        tma_atom_r, tma_r, r_layout = self.router_args(rw_ptr)
        # The weight ring starts at the first 1 KB boundary after the barriers.
        avail = self.gemm.shared_storage.size_in_bytes() - 1024
        if cutlass.const_expr(self.router_bytes > avail):
            raise ValueError(
                f"router buffers ({self.router_bytes} B) exceed the GEMM storage ({avail} B)"
            )
        self.kernel(
            tiled_mma,
            tma_atom_a,
            mA,
            tma_atom_x,
            mXt,
            mWS,
            mOut,
            m_arg,
            a_layout,
            b_layout,
            sfa_layout,
            sfb_layout,
            x_layout,
            tma_atom_g,
            mGt,
            mSF,
            mGSF,
            push,
            tma_atom_r,
            tma_r,
            r_layout,
            x_addr,
        ).launch(
            grid=(self.grid, 1, 1),
            block=(self.threads, 1, 1),
            cluster=(self.S, 1, 1),
            smem=self.gemm.shared_storage.size_in_bytes(),
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=True,
        )

    @cute.jit
    def router_args(self, rw_ptr: cute.Pointer):
        """``(tma_atom, tma_tensor, smem_layout)`` of the router weight slices (8-row x 64-column
        128-byte-swizzled boxes, as ``kimi_k3_router_logits``)."""
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
        m: Int32,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        sfa_layout: cute.Layout,
        sfb_layout: cute.Layout,
        x_layout: cute.Layout,
        tma_atom_g: Optional[cute.CopyAtom],
        mGt: Optional[cute.Tensor],
        mSF: Optional[cute.Tensor],
        mGSF: Optional[cute.Tensor],
        push,
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
                m,
                a_layout,
                b_layout,
                sfa_layout,
                sfb_layout,
                x_layout,
                tma_atom_g,
                mGt,
                mSF,
                mGSF,
                push,
                storage,
            )
        else:
            peer_ptrs, _, flags, rank = push
            self.router_cta(
                (bidx - self.n_gemm_ctas) // self.S,
                storage,
                tma_atom_r,
                mRt,
                r_layout,
                x_addr,
                m,
                flags,
                peer_ptrs,
                rank,
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
        rank: Int32,
    ):
        """Router cluster ``cl``: experts ``[8 * rt * cl, 8 * rt * (cl + 1))``, written to this
        rank's workspace only."""
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
                    # Tiles past the last expert read zero rows (TMA bounds) and are not stored.
                    for j in cutlass.range_constexpr(_BOXES_PER_SLICE):
                        cute.copy(
                            tma_atom_r,
                            tWg[(None, tile0 + t, (lo + ls) * _BOXES_PER_SLICE + j)],
                            tWs[(None, (ls * self.rt + t) * _BOXES_PER_SLICE + j)],
                            tma_bar_ptr=w_full + ls,
                            cache_policy=Int64(_EVICT_FIRST),
                        )

        cute.arch.griddepcontrol_wait()
        # Every earlier grid has completed: the dependent (the route kernel) may launch.
        cute.arch.griddepcontrol_launch_dependents()

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
                    ptx.st_shared_v4_f32(part_base + off, acc[0], acc[1], acc[2], acc[3])
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
                    v = ptx.ld_shared_v4_f32(part_base + off)
                    for i in cutlass.range_constexpr(4):
                        tot[i] = tot[i] + v[i]
                for i in cutlass.range_constexpr(4):
                    row = mt * 16 + (ln >> 2) + (i >> 1) * 8
                    col = t * 8 + (ln & 3) * 2 + (i & 1)
                    _sts_f32(stage_base + (row * (8 * self.rt) + col) * 4, tot[i])
            cute.arch.barrier()
            cur, _, nbytes, _, _, _, _, _ = pushed_ar.lamport_state(flags)
            vecs = 2 * self.rt
            own = _ld_nc_u64(peer_ptrs + Int64(rank) * 8)
            for p in cutlass.range(tidx, m * vecs, self.threads, unroll=1):
                row = p // vecs
                v = p - row * vecs
                local = tile0 * 8 + v * 4
                if local < NUM_EXPERTS:
                    w = ptx.ld_shared_v4_b32(stage_base + (row * (8 * self.rt) + v * 4) * 4)
                    off = Int64((row * NUM_EXPERTS + local) * 4)
                    rel = Int64(cur) * Int64(nbytes) + Int64(self.logits_off) + off
                    st_v4(
                        own + rel,
                        _sanitize_f32_bits(w[0]),
                        _sanitize_f32_bits(w[1]),
                        _sanitize_f32_bits(w[2]),
                        _sanitize_f32_bits(w[3]),
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
        return ptx.ld_shared_v4_b32(w_base + box * _BOX_BYTES + g * 128 + (unit << 4))


# ---------------------------------------------------------------------------
# Route: routing on the polled logits + latent gather / NVFP4
# ---------------------------------------------------------------------------


class KimiK3MoEFrontRouteKernel(NoauxTcRouteSortKernel):
    """The routing kernel's cluster (top-16 + ``moe_sort`` of one rank) on the logits the head
    stores, then ``GATHER_CLUSTERS`` clusters that gather and quantize the latent. The grid is
    ``(1 + GATHER_CLUSTERS) * CLUSTER_SIZE`` CTAs; cluster 0 routes.

    Args:
        num_warps, warps_per_token, local_cap: as ``NoauxTcRouteSortKernel``.
        world: Ranks of the group.
        publish: Publish each token's top-k (release add to the word at ``num_tiles_addr``) and
            release the dependent at entry instead of writing the ``moe_sort`` tables.
    """

    def __init__(
        self, num_warps: int, warps_per_token: int, local_cap: int, world: int, publish: bool
    ):
        super().__init__(NUM_EXPERTS, num_warps, warps_per_token, local_cap)
        self.publish = publish
        self.world = world
        self.slot_n = slot_cols(world)
        self.logits_off = logits_offset(world)
        self.tiles_base, self.tiles_rem = divmod(LATENT_TILES, world)
        self.grid = (1 + GATHER_CLUSTERS) * self.num_ctas

    def load_logits(self, row: Int64, g: Int32, stride: int, n: int):
        """The logits polled from the workspace (all ``n`` loads in flight per sweep)."""
        return _poll_words(row + Int64(g * 4), stride * 4, n)

    def load_logits4(self, row: Int64, lane: Int32, n: int):
        return [_i2f(w) for w in _poll_vectors(row + Int64(lane * 16), 128 * 4, n)]

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
            ids: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, _MAX_TOKENS_PER_CTA * TOP_K], 16
            ]
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
        ).launch(
            grid=(self.grid, 1, 1),
            block=(self.threads, 1, 1),
            cluster=(self.num_ctas, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=True,
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
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.publish):
            # The experts poll the published top-k; their reads of the latent wait for this grid.
            cute.arch.griddepcontrol_launch_dependents()
        # The halves' flags never change: no ordering needed to read them.
        cur, _, nbytes, _, _, _, _, _ = pushed_ar.lamport_state(flags)
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
            self.gather_cta(bidx - self.num_ctas, tidx, m, half, codes_addr, sf_addr, gsf_addr)
        # Complete only after the head (whose CTAs are done pushing by now).
        cute.arch.griddepcontrol_wait()

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
        """``NoauxTcRouteSortKernel.kernel`` without the grid-dependency wait before the logits
        (they are polled), re-arming each token's logits once its team is done with them."""
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        bias_smem = storage.bias.data_ptr().toint()
        mask_smem = storage.mask.data_ptr().toint()
        masks_smem = storage.masks.data_ptr().toint()
        base_smem = storage.base.data_ptr().toint()
        ids_smem = storage.ids.data_ptr().toint()
        team = warp // self.team_warps
        team_warp = warp - team * self.team_warps
        cand_smem = storage.cand.data_ptr().toint() + team * (
            self.team_warps * self.region_slots * 16
        )
        top_smem = storage.top.data_ptr().toint() + team * (TOP_K * 8)
        xchg_smem = storage.xchg.data_ptr().toint() + team * 64
        warp_tot_smem = storage.warp_tot.data_ptr().toint()

        per_cta = (m + self.num_ctas - 1) // self.num_ctas
        t_begin = rank * per_cta
        t_end = t_begin + per_cta
        if t_end > m:
            t_end = m

        for j in cutlass.range(tidx, local_num, self.threads, unroll=1):
            ptx.st_shared_b32(mask_smem + j * 4, Int32(0))
        for e4 in cutlass.range(tidx, NUM_EXPERTS // 4, self.threads, unroll=1):
            v = ptx.ld_global_nc_v4_f32(bias_addr + Int64(e4 * 16))
            ptx.st_shared_v4_f32(bias_smem + e4 * 16, v[0], v[1], v[2], v[3])
        if cutlass.const_expr(not self.publish):
            # Every CTA of the cluster must be running before a peer writes its shared memory.
            cute.arch.cluster_arrive_relaxed()
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
                logits_addr,
                bias_smem,
                ids_addr,
                vals_addr,
                scale_bits,
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
                # The team's first warp stored the ids / weights (lanes < 16): publish the token.
                if team_warp == 0:
                    cute.arch.sync_warp()
                    if lane == 0:
                        _red_add_release_gpu(num_tiles_addr, Int32(1))
        cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(not self.publish):
            self.exchange_and_sort(
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
    def exchange_and_sort(
        self,
        tidx: Int32,
        warp: Int32,
        lane: Int32,
        rank: Int32,
        t_begin: Int32,
        t_end: Int32,
        mask_smem: Int32,
        masks_smem: Int32,
        base_smem: Int32,
        ids_smem: Int32,
        warp_tot_smem: Int32,
        e2p_addr: Int64,
        p2e_addr: Int64,
        tile_expert_addr: Int64,
        tile_limit_addr: Int64,
        total_addr: Int64,
        num_tiles_addr: Int64,
        local_offset: Int32,
        local_num: Int32,
    ):
        """The token-mask exchange over the cluster and the ``moe_sort`` tables."""
        cute.arch.barrier()
        cute.arch.cluster_wait()
        # Broadcast this CTA's token masks into every CTA of the cluster.
        for j in cutlass.range(tidx, local_num, self.threads, unroll=1):
            word = ptx.ld_shared_b32(mask_smem + j * 4)
            dst = masks_smem + (rank * self.local_cap + j) * 4
            for c in cutlass.range_constexpr(self.num_ctas):
                ptx.st_dsmem_b32(dst, Int32(c), word)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
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
    ):
        """Items of 32 latent columns of one token: poll the owner rank's slot, quantize two
        16-element NVFP4 groups (``fp4_quantize``), store, re-arm."""
        nthreads = GATHER_CLUSTERS * self.num_ctas * self.threads
        gtid = gidx * self.threads + tidx
        gsf = ptx.ld_global_f32(gsf_addr)
        rgsf = rcp_approx_ftz(gsf)
        rcp6 = rcp_approx_ftz(Float32(6.0))
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
            lcol = j * _ITEM_COLS - t0 * TILE_SIZE
            src = half + Int64(((t * self.world + owner) * self.slot_n + lcol) * 2)
            words = _poll_vectors(src, 16, 4)
            vals = []
            for i in cutlass.range_constexpr(16):
                lo_f, hi_f = bf16x2_to_f32x2(words[i])
                vals.extend((lo_f, hi_f))
            amax0 = low_m.abs_max_f32x16(vals[:16])
            amax1 = low_m.abs_max_f32x16(vals[16:])
            sf2, os0, os1 = low_m.nvfp4_scales_x2(amax0, amax1, gsf, rgsf, rcp6)
            st_v4(
                codes_addr + Int64(t) * (LATENT // 2) + Int64(j * (_ITEM_COLS // 2)),
                low_m.e2m1x8(vals[0:8], os0),
                low_m.e2m1x8(vals[8:16], os0),
                low_m.e2m1x8(vals[16:24], os1),
                low_m.e2m1x8(vals[24:32], os1),
            )
            st_u16(
                sf_addr + Int64(t) * (LATENT // _FP4_VEC) + Int64(j * (_ITEM_COLS // _FP4_VEC)),
                sf2,
            )
            for v in cutlass.range_constexpr(4):
                _st_sentinel_v4(src + Int64(v * 16))
        cute.arch.griddepcontrol_launch_dependents()


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MoEFrontWorkspace:
    """The front's MNNVL multicast workspace of one group: two halves of :func:`half_bytes`;
    ``flags[h]`` is the constant rotation state (``mnnvl_pushed_allreduce.lamport_state``
    layout) that points the kernels at half ``h``."""

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
        """The flags of the next call (the halves alternate call by call on the host; see the
        module docstring for when that orders the ranks)."""
        flags = self.flags[self.calls & 1]
        self.calls += 1
        return flags


_WORKSPACES: Dict[object, MoEFrontWorkspace] = {}


def moe_front_workspace(mapping) -> MoEFrontWorkspace:
    """The front workspace of ``mapping``'s TP group (the MNNVL all-reduce's group), created on
    first use. Collective: every rank of the group calls it at the same point, outside CUDA graph
    capture."""
    ws = _WORKSPACES.get(mapping)
    if ws is not None:
        return ws
    from ....distributed import ops as dist_ops

    world, rank = mapping.tp_size, mapping.tp_rank
    if not supports_world(world):
        raise ValueError(f"the Kimi-K3 MoE front does not serve a group of {world} ranks")
    comm = dist_ops.get_or_scale_allreduce_mnnvl_workspace(mapping, torch.bfloat16)["comm"]
    nbytes = half_bytes(world)
    handle = dist_ops._make_mnnvl_mcast_buffer(
        comm, 2 * nbytes, mapping, dist_ops.use_mnnvl_fabric_handle(mapping)
    )
    buf = handle.get_uc_buffer(rank, (2 * nbytes // 4,), torch.int32, 0)
    buf.fill_(SENTINEL)
    torch.cuda.synchronize()
    # Every rank's workspace is armed before any rank pushes into it.
    if not dist_ops._mnnvl_workspace_all_succeeded(comm, True):
        raise RuntimeError("the Kimi-K3 MoE front workspace failed on a peer rank")
    peers = [handle.get_uc_buffer(r, (1,), torch.uint8, 0).data_ptr() for r in range(world)]
    flags = tuple(
        torch.tensor([h, 0, nbytes, 0, 0, 0, 0, 0, 0], dtype=torch.int32, device=buf.device)
        for h in (0, 1)
    )
    ws = MoEFrontWorkspace(
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
class MoEFrontPlan:
    """One MoE layer's front operands on one rank (views of the model's weights plus the packed
    scale of the rank's latent_down tiles)."""

    world: int
    rank: int
    latent_w: torch.Tensor  # FP8 [128 * tiles of the rank, 7168]
    latent_ws: torch.Tensor  # DeepGEMM-packed UE8M0 scale of those rows, strides (1, N)
    router_w: torch.Tensor  # BF16 [896, 7168]
    bias: torch.Tensor  # FP32 [896]
    routed_scaling_factor: float
    local_offset: int  # this rank's routed experts (moe_sort tables)
    local_num: int


def make_plan(
    latent_w: torch.Tensor,
    latent_scale: torch.Tensor,
    router_w: torch.Tensor,
    bias: torch.Tensor,
    routed_scaling_factor: float,
    world: int,
    rank: int,
    local_offset: int,
    local_num: int,
) -> MoEFrontPlan:
    """The front plan of one layer.

    Args:
        latent_w: FP8 ``[3584, 7168]`` latent_down weight.
        latent_scale: Its DeepGEMM-packed UE8M0 scale (the low-M GEMM's ``w_scale``).
        router_w: BF16 ``[896, 7168]`` router weight.
        bias: FP32 ``[896]`` score-correction bias.
        routed_scaling_factor: Multiplier of the renormalized top-k weights.
        world, rank: The group and this rank.
        local_offset, local_num: This rank's routed experts.
    """
    if not supports_world(world):
        raise ValueError(f"the Kimi-K3 MoE front does not serve a group of {world} ranks")
    if tuple(latent_w.shape) != (LATENT, HIDDEN) or latent_w.dtype != torch.float8_e4m3fn:
        raise ValueError("latent_w must be FP8 [3584, 7168]")
    if tuple(router_w.shape) != (NUM_EXPERTS, HIDDEN) or router_w.dtype != torch.bfloat16:
        raise ValueError("router_w must be BF16 [896, 7168]")
    t0, nt = latent_tiles_of(rank, world)
    rows = slice(t0 * TILE_SIZE, (t0 + nt) * TILE_SIZE)
    # Rows of an MN-major scale, kept MN-major with the slice's own stride.
    rows_scale = latent_scale[rows].t().contiguous().t()
    return MoEFrontPlan(
        world=world,
        rank=rank,
        latent_w=latent_w[rows],
        latent_ws=rows_scale,
        router_w=router_w.contiguous(),
        bias=bias.float().contiguous(),
        routed_scaling_factor=float(routed_scaling_factor),
        local_offset=local_offset,
        local_num=local_num,
    )


def latent_tactic(m: int) -> low_m.LowMTactic:
    """The latent GEMM's tactic: the low-M kernel at the split-K of the latent_down it replaces
    (its NVFP4-output route where that runs the low-M kernel, else the low-M default; 4 where the
    route runs the mid-M kernel, which has no all-gather push)."""
    tactic = low_m.nvfp4_out_tactic(m, LATENT, HIDDEN)
    if isinstance(tactic, low_m.LowMTactic):
        split = tactic.split_k
    elif tactic is not None:
        split = 4
    else:
        split = low_m.default_tactic(m, LATENT, HIDDEN).split_k
    return low_m.split_tactic(m, LATENT, HIDDEN, split, mid=False)


#: Compiled kernels by ``(builder, compile arguments)``; :func:`is_compiled` reads it.
_COMPILED: Dict[tuple, object] = {}


def _compiled(build: Callable, *args):
    key = (build, *args)
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = _COMPILED[key] = build(*args)
    return compiled


def _compile_head(n: int, tactic: low_m.LowMTactic, world: int):
    gemm = low_m.Fp8BlockScaleLowMGemmKernel(
        n,
        HIDDEN,
        tactic,
        True,
        all_reduce_world=world,
        one_shot=True,
        push_slot_n=slot_cols(world),
    )
    head = KimiK3MoEFrontHeadKernel(
        gemm, ROUTER_TILES, (tactic.mma_n + 15) // 16, logits_offset(world)
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


def _local_cap(local_num: int) -> int:
    return _SMALL_LOCAL_CAP if local_num <= _SMALL_LOCAL_CAP else NUM_EXPERTS


def _compile_route(num_warps: int, warps_per_token: int, local_cap: int, world: int, publish: bool):
    kernel = KimiK3MoEFrontRouteKernel(num_warps, warps_per_token, local_cap, world, publish)
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
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def _head_args(m: int, plan: MoEFrontPlan) -> tuple:
    return plan.latent_w.shape[0], latent_tactic(m), plan.world


def _route_args(m: int, plan: MoEFrontPlan, publish: bool) -> tuple:
    warps_per_token, num_warps = _config(m)
    return num_warps, warps_per_token, _local_cap(plan.local_num), plan.world, publish


def is_compiled(m: int, plan: MoEFrontPlan, publish: bool = False) -> bool:
    """Whether a call of ``m`` rows with this plan's shapes runs without compiling (required
    inside CUDA graph capture)."""
    return (_compile_head, *_head_args(m, plan)) in _COMPILED and (
        _compile_route,
        *_route_args(m, plan, publish),
    ) in _COMPILED


def _gptr(dtype, t: torch.Tensor):
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def can_run(x: torch.Tensor, plan: MoEFrontPlan, global_sf: torch.Tensor) -> bool:
    """Whether :func:`run_moe_front` serves ``x`` (contiguous BF16 ``[M, 7168]``, ``1 <= M <=
    M_MAX``) with ``plan`` and the FP32 one-element NVFP4 global scale ``global_sf``."""
    return (
        x.is_cuda
        and low_m.is_supported_device(x.device.index)
        and x.dtype == torch.bfloat16
        and x.dim() == 2
        and x.is_contiguous()
        and x.shape[1] == HIDDEN
        and 1 <= x.shape[0] <= M_MAX
        and x.data_ptr() % 16 == 0
        and global_sf.dtype == torch.float32
        and global_sf.numel() == 1
        and global_sf.device == x.device
        and 1 <= plan.local_num
        and plan.local_offset + plan.local_num <= NUM_EXPERTS
    )


def run_moe_front(
    x: torch.Tensor,
    plan: MoEFrontPlan,
    ws: MoEFrontWorkspace,
    global_sf: torch.Tensor,
    publish_word: Optional[torch.Tensor] = None,
    after_head: Optional[Callable[[], None]] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, ...]], torch.Tensor, torch.Tensor
]:
    """Launch the head and the route kernel on the current stream (every rank of the group with
    the same ``M``).

    Returns ``(topk_ids, topk_weights, sort_outputs, codes, scales)``: ``kimi_k3_routing``'s
    outputs for the logits (int32 / FP32 ``[M, 16]`` and the six ``moe_sort`` tables of
    ``plan.local_offset`` / ``local_num`` at 128-row tiles), and the NVFP4 latent, ``codes``
    uint8 ``[M, 1792]`` and ``scales`` uint8 ``[M, 224]`` (linear layout) under ``global_sf``.
    ``publish_word``: an int32 word (zero) the route kernel adds 1 to per routed token (release)
    instead of writing the tables (``sort_outputs`` is then ``None``); the next launch on the
    stream, which polls it, must be the routed experts. ``after_head``: called between the two
    launches (e.g. to fork a side stream after the head). The first call of each shape compiles
    (outside CUDA graph capture)."""
    m = x.shape[0]
    if not can_run(x, plan, global_sf):
        raise ValueError(f"Kimi-K3 MoE front: unsupported call x {tuple(x.shape)} {x.dtype}")
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    flags = ws.next_flags()
    _compiled(_compile_head, *_head_args(m, plan))(
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
    max_tiles, max_permuted = sort_buffer_sizes(m, plan.local_num)
    i32 = dict(dtype=torch.int32, device=device)
    topk_ids = torch.empty((m, TOP_K), **i32)
    topk_weights = torch.empty((m, TOP_K), dtype=torch.float32, device=device)
    codes = torch.empty((m, LATENT // 2), dtype=torch.uint8, device=device)
    scales = torch.empty((m, LATENT // _FP4_VEC), dtype=torch.uint8, device=device)
    publish = publish_word is not None
    sort_outputs = None
    if publish:
        tile_expert = tile_limit = e2p = p2e = total = 0
        num_tiles = publish_word.data_ptr()
    else:
        sort_outputs = (
            torch.empty((max_tiles,), **i32),
            torch.empty((max_tiles,), **i32),
            torch.empty((m, TOP_K), **i32),
            torch.empty((max_permuted,), **i32),
            torch.empty((1,), **i32),
            torch.empty((1,), **i32),
        )
        tile_expert, tile_limit, e2p, p2e, total, num_tiles = (t.data_ptr() for t in sort_outputs)
    scale_bits = struct.unpack("<q", struct.pack("<d", plan.routed_scaling_factor))[0]
    _compiled(_compile_route, *_route_args(m, plan, publish))(
        Int32(m),
        Int64(ws.uc_local),
        Int64(flags.data_ptr()),
        Int64(plan.bias.data_ptr()),
        Int64(topk_ids.data_ptr()),
        Int64(topk_weights.data_ptr()),
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
        Int64(global_sf.data_ptr()),
        stream,
    )
    return topk_ids, topk_weights, sort_outputs, codes, scales


__all__ = [
    "M_MAX",
    "KimiK3MoEFrontHeadKernel",
    "KimiK3MoEFrontRouteKernel",
    "MoEFrontPlan",
    "MoEFrontWorkspace",
    "can_run",
    "is_compiled",
    "latent_tactic",
    "latent_tiles_of",
    "make_plan",
    "moe_front_workspace",
    "run_moe_front",
    "slot_cols",
    "supports_world",
]
