# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 / K3.1 MoE front for small decode steps (``M <= 32`` rows) over the TP group: the route
kernel and its workspace.

Each rank's latent rows (its 128-row tiles of latent_down, rank ``r`` owning tiles
``[t0(r), t1(r))`` of 28) arrive in slot ``(token, r)`` of the front workspace of every rank, and
the FP32 router logits in this rank's logits rows; both producers run before this kernel
(``kernels/moe/rubin_ull_push_gemm.py``: the push GEMM, then the logits copy).

**Route** (:class:`K3UllFrontRouteKernel`, a PDL dependent that polls rather than waits): a
``rubin_route_tail`` cluster whose logits loads poll the workspace (sentinel ``0x80000000``)
instead of reading a tensor, so the top-16 and the ``moe_sort`` tables of this rank are
``rubin_route_tail``'s (bit-identical to ``noaux_tc_op`` + ``moe_sort`` on the same logits); and
gather CTAs that poll the latent slots of every rank and write the NVFP4 quantization of the
gathered BF16 latent (``fp4_quantize``'s arithmetic, linear scale layout): uint8 ``[m, 1792]``
codes and uint8 ``[m, 224]`` scales of ``bf16(x @ latent_down^T)``. Every reader re-arms what it
read.

Workspace (:class:`UllFrontWorkspace`): an MNNVL multicast buffer of two halves; consecutive calls
alternate between them (the half is a launch argument through a constant flags tensor, so a CUDA
graph replays the same alternation). Rank ``r`` writes half ``h`` of a peer only after every rank
finished the call that last used ``h``: between two calls on a half there is the other half's call,
whose route kernel on ``r`` waited for data every peer pushed after its own previous call.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Dict, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ....kernels.comm import rubin_lamport_ar as lamport
from ....kernels.moe.rubin_route_tail import (
    MAX_CTAS,
    TILE,
    K3RouteTailKernel,
    _cluster_arrive_relaxed,
    _cluster_arrive_release,
    _cluster_wait_acquire,
    _griddepcontrol_launch_dependents,
    _griddepcontrol_wait,
    _ld_nc_v4_f32,
    _lds_b32,
    _st_dsmem_b32,
    _sts_b32,
    _sts_v4_f32,
)
from ..gemm.rubin_latent_fp8_gemm import (
    _abs_max_f32x16,
    _e2m1x8,
    _ldg_f32,
    _nvfp4_scales_x2,
    _rcp_approx_ftz,
    _stg_u16,
    _stg_v4_b32,
)

#: Kimi K3 / K3.1 shapes.
LATENT = 3584
NUM_EXPERTS = 896
TOP_K = 16
#: 128-row tiles of latent_down (its output columns), split over the ranks.
LATENT_TILES = LATENT // TILE
#: Largest rows per call.
M_MAX = 32
#: Gather clusters of the route kernel (of ``num_ctas`` CTAs each).
GATHER_CLUSTERS = 1
SENTINEL = lamport.SENTINEL
#: NVFP4 group (fp4_quantize's scaling vector) and the gather item (32 latent columns).
_FP4_VEC = 16
_ITEM_COLS = 32


# ---------------------------------------------------------------------------
# Rank geometry
# ---------------------------------------------------------------------------


def slot_cols(world: int) -> int:
    """Features of one (token, rank) latent slot: the largest per-rank tile count x 128."""
    base, rem = divmod(LATENT_TILES, world)
    return (base + (1 if rem else 0)) * TILE


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
# Route: routing tail on the polled logits + latent gather / NVFP4
# ---------------------------------------------------------------------------


class K3UllFrontRouteKernel(K3RouteTailKernel):
    """``rubin_route_tail``'s cluster (top-16 + ``moe_sort`` of one rank) on the logits polled
    from the workspace (written by ``run_ull_front_bf16``'s logits copy), then ``gather_clusters``
    clusters that gather and quantize the latent.

    The grid is ``(1 + gather_clusters) * num_ctas`` CTAs; cluster 0 routes.
    """

    def __init__(
        self,
        num_ctas: int,
        num_warps: int,
        warps_per_token: int,
        local_cap: int,
        world: int,
    ):
        super().__init__(num_ctas, num_warps, warps_per_token, True, local_cap)
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
        zero_addr: Int64,
        zero_words: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
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
        # Complete only after the preceding grids (the push GEMM and the logits copy).
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
        # Distributed shared memory may only be accessed once every CTA of the cluster runs: arrive here, wait
        # before the mask broadcast.
        _cluster_arrive_relaxed()
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
        _griddepcontrol_launch_dependents()
        cute.arch.barrier()
        # Every CTA of the cluster has started (arrive after the bias staging).
        _cluster_wait_acquire()
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
    """The front's MNNVL multicast workspace of one TP group: two halves of :func:`half_bytes`;
    ``flags[h]`` is the constant rotation state (``rubin_lamport_ar`` layout) that points the
    kernels at half ``h``."""

    world: int
    rank: int
    uc_local: int
    mc_base: int
    peer_ptrs: torch.Tensor
    flags: Tuple[torch.Tensor, torch.Tensor]
    handle: object = None
    calls: int = 0

    def next_flags(self) -> torch.Tensor:
        """The flags of the next call (the halves alternate call by call)."""
        flags = self.flags[self.calls & 1]
        self.calls += 1
        return flags


_WORKSPACES: Dict[object, UllFrontWorkspace] = {}


def ull_front_workspace(mapping) -> UllFrontWorkspace:
    """The front workspace of ``mapping``'s TP group, created on first use (collective: every rank
    of the group calls it at the same point, over the MPI communicator of the MNNVL all-reduce
    workspace)."""
    ws = _WORKSPACES.get(mapping)
    if ws is not None:
        return ws
    from tensorrt_llm._torch.distributed.ops import get_or_scale_allreduce_mnnvl_workspace
    from tensorrt_llm.bindings.internal.runtime import McastGPUBuffer

    comm = get_or_scale_allreduce_mnnvl_workspace(mapping, torch.bfloat16)["comm"]
    world, rank = mapping.tp_size, mapping.tp_rank
    if not supports_world(world):
        raise ValueError(f"k3_ull_front does not serve TP world {world}")
    nbytes = half_bytes(world)
    # Fabric handles across nodes; POSIX FD handles are valid on one node.
    use_fabric = mapping.is_multi_node()
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
        handle=handle,
    )
    _WORKSPACES[mapping] = ws
    return ws


#: Local-expert capacity of the routing tail's sort build (as ``rubin_route_tail``): 224 (EP >= 4),
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
):
    kernel = K3UllFrontRouteKernel(
        num_ctas,
        num_warps,
        warps_per_token,
        local_cap,
        world,
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


__all__ = [
    "M_MAX",
    "UllFrontWorkspace",
    "slot_cols",
    "supports_world",
    "ull_front_workspace",
]
