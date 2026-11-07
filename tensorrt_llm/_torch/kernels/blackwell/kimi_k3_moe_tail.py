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
"""Kimi-K3 MoE back half in one kernel for at most 8 rows (``enable_min_latency``).

After the routed experts, the MoE block of a decode step on the column-parallel tail needs, per
token ``t``, ``y = latent_up(rmsnorm(sum_r p_r))`` where ``p_r`` is rank ``r``'s routed latent
partial ``[M, 3584]`` (BF16); the next layer's pre-attention mix adds ``y`` and the shared
expert's all-reduce to the residual (``KimiK3MoERuntime._column_parallel_tail``).
:class:`KimiK3MoeTailKernel` computes this rank's ``HIDDEN / W`` columns of ``y`` (rows
``[c0, c0 + HIDDEN / W)`` of latent_up) straight from the partials the routed experts push and
stores them into the all-gather rows of the tail's call, which that mix finishes:

* Input. The routed experts push their BF16 latent partial rows into slot ``(token, rank)`` of
  every rank's current buffer of a one-shot MNNVL all-reduce (``mnnvl_pushed_allreduce``, the
  small-batch kernel's output push). This kernel finishes that all-reduce: it re-arms the previous
  call's buffer, polls the slots and sums them in FP32 in the C++ one-shot kernel's order (chunks
  of eight ranks), rounded to BF16 (bitwise the latent all-reduce's sum), and the last CTA
  advances the rotation state after its ``griddepcontrol.wait``.
* Grid. ``G`` clusters of 16 CTAs x 256 threads. Cluster ``g`` owns ``RG`` output rows
  (:func:`cluster_rows`: 128 where the rank's rows are whole 128-row blocks, else 112); CTA ``q``
  of the cluster owns latent columns ``[224 q, 224 q + 224)``.
  Before the poll each lane loads its FP8 latent_up fragments (the 128x128 UE8M0 block scales
  applied, BF16 exact) into registers.
* RMSNorm late. A token's partial sum of squares goes to its two owner CTAs (``st.async``); the up
  projection runs on ``bf16(x * gamma)`` and the owners scale the summed products by
  ``rsqrt(sum / 3584 + eps)`` before rounding.
* Up projection. ``mma.sync`` m16n8k16 BF16 with FP32 accumulation over the CTA's 224 columns.
* Owners. Output line (token ``t``, row half ``h``) is owned by CTA ``t + 8 h`` of the cluster; the
  owner sums the 16 CTAs' partials in CTA order, scales, rounds to BF16, replaces ``-0`` by ``+0``
  and multicasts the row segment into the tail call's all-gather rows
  (``mnnvl_pushed_allreduce.all_gather_row``). Where the tail call follows on the latent
  all-reduce's workspace (``ag_follows``) its buffer is the next one in the rotation.

PDL: the dependent is released at entry. Weights, gamma, the block exponents and both rotation
states are read before any wait; the kernel stores nothing a predecessor reads.

Numerics (not bitwise the column-parallel tail): the latent sum is the all-reduce's; the RMS uses
FP32 squares of it; the projection's activation is ``bf16(x * gamma)`` in BF16 instead of the
MXFP8 ``bf16((x * r) * gamma)``, and ``r`` multiplies the FP32 product sum before its rounding.
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
from cutlass.cutlass_dsl import T, dsl_user_op

from . import mnnvl_pushed_allreduce as pushed_ar
from .mnnvl_pushed_allreduce import arrived_v4 as clean4
from .mnnvl_pushed_allreduce import clear_dirty_stages
from .raw_ptx import (
    acc_bf16x2,
    add_f32,
    ld_volatile_v4,
    multimem_st_v4,
    pack_bf16x2,
    sanitize_bf16x2,
    st_v4,
)
from .raw_ptx import asm as _asm
from .raw_ptx import bar_sync as _bar_sync
from .raw_ptx import bf16x2_to_f32x2 as _bf16x2_f32
from .raw_ptx import cluster_arrive_relaxed as _cluster_arrive_relaxed
from .raw_ptx import fence_mbar_init as _fence_mbar_init
from .raw_ptx import fv as _fv
from .raw_ptx import iv as _iv
from .raw_ptx import ld_nc_u8 as _ld_nc_u8
from .raw_ptx import ld_nc_v2 as _ld_nc_v2
from .raw_ptx import ld_nc_v4 as _ld_nc_v4
from .raw_ptx import lds_f32 as _lds_f32
from .raw_ptx import lds_v4_f32 as _lds_v4_f32
from .raw_ptx import mbar_expect_tx as _mbar_expect_tx
from .raw_ptx import mbar_init as _mbar_init
from .raw_ptx import mma_bf16 as _mma_bf16
from .raw_ptx import mul_f32 as _fmul
from .raw_ptx import shfl_bfly_f32 as _shfl_bfly_f32
from .raw_ptx import sts_f32 as _sts_f32
from .raw_ptx import sts_v4 as _sts_v4
from .raw_ptx import unpack as _unpack
from .utils import griddepcontrol_launch_dependents, griddepcontrol_wait

__all__ = [
    "HIDDEN",
    "LATENT",
    "MAX_M",
    "ROWS_PER_CLUSTER",
    "KimiK3MoeTailKernel",
    "block_exponents",
    "can_run",
    "cluster_rows",
    "rank_rows",
    "run_moe_tail",
]

#: Latent (latent_up's K) and hidden (latent_up's N) widths.
LATENT = 3584
HIDDEN = 7168
#: CTAs per cluster; each owns ``LATENT / CLUSTER`` latent columns.
CLUSTER = 16
KS = LATENT // CLUSTER
#: 16-byte column chunks of a token's K slice (one polling lane each).
CHUNKS = KS // 8
#: Threads per CTA (warp ``t`` polls token ``t``).
THREADS = 256
WARPS = THREADS // 32
#: Largest row count (one warp per token).
MAX_M = 8
#: MMA K steps (16) over a CTA's K slice.
K_STEPS = KS // 16
#: 16-row m-tiles a warp may hold in registers (56 registers each).
MAX_MTILES_PER_WARP = 2
#: Output rows of one 16-CTA cluster (four clusters at 16 ranks).
ROWS_PER_CLUSTER = 112


def _k_col(j: int):
    """``(base, lane stride)`` of K step ``j``'s first column for lane group ``t`` (column
    ``base + stride * t``, four consecutive columns per step): steps ``4 i .. 4 i + 3`` (i < 3) of
    lane group ``t`` are the 16 columns ``64 i + 16 t ..``, steps 12-13 the 8 columns
    ``192 + 8 t ..``; a bijection of the 224 columns onto (step, lane group, element)."""
    if j < 12:
        return 64 * (j // 4) + 4 * (j % 4), 16
    return 192 + 4 * (j - 12), 8


#: Bytes of a token row of ``bf16(x * gamma)`` in shared memory (16 bytes of padding).
XS_PITCH = KS * 2 + 16
#: FP32 words per token row of a warp's staged MMA partial (16 rows + padding: conflict-free).
STAGE_PITCH = 20
STAGE_WARP_BYTES = MAX_M * STAGE_PITCH * 4

assert KS % 8 == 0 and KS == 224 and 56 * 4 == KS


@dsl_user_op
def _deq_bf16x2(word, scale, high: bool, *, loc=None, ip=None):
    """Two E4M3 codes (the low or high 16 bits of ``word``, low byte first) times ``scale`` (a power
    of two) as a BF16 pair, low value in the low half. Exact for latent_up's weights: E4M3 -> F16 ->
    F32 are exact and the power-of-two scale keeps the 4 significant bits."""
    half = "h" if high else "l"
    return Int32(
        _asm(
            T.i32(),
            [_iv(word, loc=loc, ip=ip), _fv(scale, loc=loc, ip=ip)],
            "{\n.reg .b16 l, h, p0, p1;\n.reg .b32 p;\n.reg .f32 f0, f1;\n"
            "mov.b32 {l, h}, $1;\n"
            f"cvt.rn.f16x2.e4m3x2 p, {half};\n"
            "mov.b32 {p0, p1}, p;\n"
            "cvt.f32.f16 f0, p0;\ncvt.f32.f16 f1, p1;\n"
            "mul.rn.f32 f0, f0, $2;\nmul.rn.f32 f1, f1, $2;\n"
            "cvt.rn.bf16x2.f32 $0, f1, f0;\n}",
            "=r,r,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _exp2_of_ue8m0(e, *, loc=None, ip=None):
    """``2^(e - 127)`` of a UE8M0 exponent ``1 <= e <= 254`` (the FP32 word ``e << 23``)."""
    return Float32(
        _asm(
            T.f32(),
            [_iv(e, loc=loc, ip=ip)],
            "{\n.reg .b32 b;\nshl.b32 b, $1, 23;\nmov.b32 $0, b;\n}",
            "=f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fma(a, b, c, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip), _fv(c, loc=loc, ip=ip)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _rms_scale(total, eps, *, loc=None, ip=None):
    """``rsqrt(total / 3584 + eps)`` (IEEE division, then add, approximate reciprocal root)."""
    return Float32(
        _asm(
            T.f32(),
            [_fv(total, loc=loc, ip=ip), _fv(eps, loc=loc, ip=ip)],
            "{\n.reg .f32 q;\ndiv.rn.f32 q, $1, 0f45600000;\nadd.rn.f32 q, q, $2;\n"
            "rsqrt.approx.f32 $0, q;\n}",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _lds_v2(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_iv(addr, loc=loc, ip=ip)],
        "ld.shared.v2.b32 {$0, $1}, [$2];",
        "=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 2, loc=loc, ip=ip)


@dsl_user_op
def _mbar_try_wait_cluster(addr, *, loc=None, ip=None):
    """1 once phase 0 of the local mbarrier at ``addr`` (completed by peers' ``st.async``) has
    completed, else 0 (cluster-scope acquire)."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(addr, loc=loc, ip=ip)],
            "{\n.reg .pred p;\n"
            "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [$1], 0;\n"
            "selp.u32 $0, 1, 0, p;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_async_f32(local_addr, peer, local_mbar, value, *, loc=None, ip=None):
    """One FP32 into CTA ``peer``'s copy of ``local_addr``, counted by its copy of
    ``local_mbar``."""
    _asm(
        None,
        [
            _iv(local_addr, loc=loc, ip=ip),
            _iv(peer, loc=loc, ip=ip),
            _iv(local_mbar, loc=loc, ip=ip),
            _fv(value, loc=loc, ip=ip),
        ],
        "{\n.reg .b32 ra, rm;\nmapa.shared::cluster.u32 ra, $0, $1;\n"
        "mapa.shared::cluster.u32 rm, $2, $1;\n"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [ra], $3, [rm];\n}",
        "r,r,r,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_async_v4_f32(local_addr, peer, local_mbar, v0, v1, v2, v3, *, loc=None, ip=None):
    """16 bytes into CTA ``peer``'s copy of ``local_addr``, counted by its copy of
    ``local_mbar``."""
    _asm(
        None,
        [_iv(x, loc=loc, ip=ip) for x in (local_addr, peer, local_mbar)]
        + [_fv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "{\n.reg .b32 ra, rm;\nmapa.shared::cluster.u32 ra, $0, $1;\n"
        "mapa.shared::cluster.u32 rm, $2, $1;\n"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 "
        "[ra], {$3, $4, $5, $6}, [rm];\n}",
        "r,r,r,f,f,f,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _cluster_wait(*, loc=None, ip=None):
    _asm(None, [], "barrier.cluster.wait.aligned;", "", loc=loc, ip=ip)


class KimiK3MoeTailKernel:
    """The MoE back half of ``m <= 8`` rows on one rank of ``world`` (see the module docstring).

    :param world: Ranks of the latent / tail workspaces.
    :param rows: Output rows of this rank (:func:`cluster_rows` per cluster; warp ``w`` multiplies
        the 16-row m-tiles ``w, w + 8, ..`` of its cluster's rows).
    :param m: Tokens of the call.
    :param mc_out: Store the output through the multicast mapping (``multimem.st``); False stores
        to the given address (single-GPU tests).
    :param ag_follows: The tail call follows the latent all-reduce on the same workspace (its
        buffer is the next one in the rotation), else its workspace's state says.
    """

    def __init__(
        self, world: int, rows: int, m: int, mc_out: bool = True, ag_follows: bool = False
    ):
        rg = cluster_rows(rows)
        if not rg:
            raise ValueError(f"{rows} rows do not split into clusters")
        if not 1 <= m <= MAX_M:
            raise ValueError(f"m={m} rows: 1..{MAX_M}")
        self.world = world
        self.m = m
        self.rg = rg
        self.groups = rows // rg
        self.nctas = self.groups * CLUSTER
        self.mtiles = rg // 16
        self.mpw = (self.mtiles + WARPS - 1) // WARPS
        assert self.mpw <= MAX_MTILES_PER_WARP
        # Two owners per token (row halves of the cluster's rows), 8-row lines per lane.
        self.ro = rg // 2
        self.mc_out = mc_out
        self.ag_follows = ag_follows
        self.part_words = CLUSTER * self.ro
        self.rank_chunk = pushed_ar.reduce_chunk(world, True)

    @cute.jit
    def __call__(
        self,
        lat_uc: Int64,
        lat_flags: Int64,
        w_addr: Int64,
        wexp_addr: Int64,
        gamma_addr: Int64,
        ag_base: Int64,
        ag_flags: Int64,
        c0: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            bars: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 2], 16]
            sq: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, CLUSTER], 16]
            xs: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, MAX_M * XS_PITCH], 16]
            stage: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, WARPS * STAGE_WARP_BYTES], 16
            ]
            part: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, self.part_words], 16]

        self.shared_storage = SharedStorage
        self.kernel(
            lat_uc,
            lat_flags,
            w_addr,
            wexp_addr,
            gamma_addr,
            ag_base,
            ag_flags,
            c0,
            eps,
        ).launch(  # fmt: skip
            grid=(self.nctas, 1, 1),
            block=(THREADS, 1, 1),
            cluster=(CLUSTER, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            min_blocks_per_mp=1,
            # The largest carveout: the next kernels' CTAs may land beside this grid's.
            preferred_smem_carveout=100,
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        lat_uc: Int64,
        lat_flags: Int64,
        w_addr: Int64,
        wexp_addr: Int64,
        gamma_addr: Int64,
        ag_base: Int64,
        ag_flags: Int64,
        c0: Int32,
        eps: Float32,
    ):
        W = self.world
        m = self.m
        KJ = K_STEPS
        RO = self.ro
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        q = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        gcl = bidx // CLUSTER
        gq = lane // 4
        tq = lane % 4

        griddepcontrol_launch_dependents()

        storage = utils.SmemAllocator().allocate(self.shared_storage)
        bars = storage.bars.data_ptr().toint()
        bar_sq = bars
        bar_part = bars + 8
        sq_s = storage.sq.data_ptr().toint()
        xs_s = storage.xs.data_ptr().toint()
        stage_s = storage.stage.data_ptr().toint()
        part_s = storage.part.data_ptr().toint()
        k0 = q * KS
        rbase = c0 + gcl * self.rg

        # ---- latent_up loads first (before any wait; their conversions follow the setup) ----
        # Per m-tile and weight row: three 16-byte loads (K steps 0-11) and one 8-byte load (steps
        # 12-13) of FP8 codes, and the UE8M0 exponents of the K slice's (up to) three 128-column
        # scale blocks.
        b0 = k0 // 128
        raw = []
        exps = []
        for s in cutlass.range_constexpr(self.mpw):
            mt = cutlass.min(warp + WARPS * s, self.mtiles - 1)
            r0 = rbase + mt * 16 + gq
            for h in cutlass.range_constexpr(2):
                w_row = w_addr + Int64(r0 + 8 * h) * LATENT + Int64(k0)
                for i in cutlass.range_constexpr(3):
                    raw += list(_ld_nc_v4(w_row + Int64(64 * i + 16 * tq)))
                raw += list(_ld_nc_v2(w_row + Int64(192 + 8 * tq)))
                e_row = wexp_addr + Int64(((r0 + 8 * h) // 128) * (LATENT // 128))
                for bi in cutlass.range_constexpr(3):
                    blk = cutlass.min(b0 + bi, LATENT // 128 - 1)
                    exps.append(_ld_nc_u8(e_row + Int64(blk)))
        lat_lane = (warp < m) & (lane < CHUNKS)
        gcol = k0 + cutlass.select_(lane < CHUNKS, lane, Int32(0)) * 8
        gw = _ld_nc_v4(gamma_addr + Int64(gcol) * 2)

        # ---- setup: barriers, rotation states (every thread reads them), the arrival count ----
        # Owner role: line (token q % 8, row half q // 8) of the cluster's rows.
        own_t = q % MAX_M
        own_h = q // MAX_M
        # Rotation states: final, the previous calls on both workspaces completed before the
        # experts (whose launch released this grid) passed their own wait.
        f = ld_volatile_v4(lat_flags)
        fc = ld_volatile_v4(lat_flags + Int64(16))
        a = ld_volatile_v4(ag_flags)
        cur, dirty, nbytes, dstages = f[0], f[1], f[2], f[3]
        ag_cur, ag_bytes = a[0], a[2]
        if cutlass.const_expr(self.ag_follows):
            ag_cur = cutlass.select_(cur + 1 == pushed_ar.NUM_BUFFERS, Int32(0), cur + 1)
            ag_bytes = nbytes
        if tidx == 0:
            _mbar_init(bar_sq, 1)
            _mbar_init(bar_part, 1)
            if own_t < m:
                _mbar_expect_tx(bar_sq, CLUSTER * 4)
                _mbar_expect_tx(bar_part, CLUSTER * RO * 4)
            _fence_mbar_init()
        # Every thread of the CTA has read the state before its count.
        _bar_sync(1, THREADS)
        # The CTAs of a cluster start together; the wait (every peer's barriers initialized) comes
        # before the first remote store.
        _cluster_arrive_relaxed()

        # ---- latent_up fragments: BF16 of the FP8 codes times the block scale ----
        afr = cute.make_rmem_tensor((self.mpw * KJ * 4,), Int32)
        scales = [_exp2_of_ue8m0(e) for e in exps]
        for s in cutlass.range_constexpr(self.mpw):
            for j in cutlass.range_constexpr(KJ):
                cb, cs = _k_col(j)
                bi = (k0 + cb + cs * tq) // 128 - b0
                sc = []
                for h in cutlass.range_constexpr(2):
                    e3 = (s * 2 + h) * 3
                    hi = cutlass.select_(bi == 1, scales[e3 + 1], scales[e3 + 2])
                    sc.append(cutlass.select_(bi == 0, scales[e3], hi))
                w0 = raw[(s * 2) * KJ + j]
                w1 = raw[(s * 2 + 1) * KJ + j]
                base = (s * KJ + j) * 4
                afr[base + 0] = _deq_bf16x2(w0, sc[0], False)
                afr[base + 1] = _deq_bf16x2(w1, sc[1], False)
                afr[base + 2] = _deq_bf16x2(w0, sc[0], True)
                afr[base + 3] = _deq_bf16x2(w1, sc[1], True)

        # The previous call's buffer (complete before this grid launched; nobody reads it).
        clear_dirty_stages(
            lat_uc, dirty, nbytes, dstages, fc[0], fc[1], fc[2], fc[3],
            bidx * THREADS + tidx, self.nctas * THREADS,
        )  # fmt: skip

        # ---- poll token `warp`'s slots of every rank (lanes < 28: 8 columns each) ----
        words = cute.make_rmem_tensor((W * 4,), Int32)
        row0 = lat_uc + Int64(cur) * Int64(nbytes) + Int64((warp * W) * LATENT + gcol) * 2
        pending = Int32(0)
        for e in cutlass.range_constexpr(W * 4):
            words[e] = Int32(0)
        # Every pass issues all W loads before it checks any (they are in flight together).
        if lat_lane:
            loaded = []
            for s in cutlass.range_constexpr(W):
                loaded.append(list(ld_volatile_v4(row0 + Int64(s * LATENT * 2))))
            for s in cutlass.range_constexpr(W):
                for e in cutlass.range_constexpr(4):
                    words[s * 4 + e] = loaded[s][e]
                clean = clean4(loaded[s][0], loaded[s][1], loaded[s][2], loaded[s][3])
                pending = pending | ((Int32(1) - clean) << s)
        while cute.arch.vote_any_sync(pending != 0):
            if pending != 0:
                loaded = []
                for s in cutlass.range_constexpr(W):
                    loaded.append(list(ld_volatile_v4(row0 + Int64(s * LATENT * 2))))
                pending = Int32(0)
                for s in cutlass.range_constexpr(W):
                    for e in cutlass.range_constexpr(4):
                        words[s * 4 + e] = loaded[s][e]
                    clean = clean4(loaded[s][0], loaded[s][1], loaded[s][2], loaded[s][3])
                    pending = pending | ((Int32(1) - clean) << s)

        # Rank-order sum (the C++ one-shot chunks), BF16; squares; bf16(x * gamma) to smem.
        acc = [Float32(0.0)] * 8
        for cb in cutlass.range_constexpr(0, W, self.rank_chunk):
            part = [Float32(0.0)] * 8
            for s in cutlass.range_constexpr(cb, cb + self.rank_chunk):
                for j in cutlass.range_constexpr(4):
                    part[2 * j], part[2 * j + 1] = acc_bf16x2(
                        words[s * 4 + j], part[2 * j], part[2 * j + 1]
                    )
            for i in cutlass.range_constexpr(8):
                acc[i] = add_f32(acc[i], part[i])
        s2 = Float32(0.0)
        xg = []
        for j in cutlass.range_constexpr(4):
            red = pack_bf16x2(acc[2 * j], acc[2 * j + 1])
            x_lo, x_hi = _bf16x2_f32(red)
            g_lo, g_hi = _bf16x2_f32(gw[j])
            s2 = _fma(x_lo, x_lo, s2)
            s2 = _fma(x_hi, x_hi, s2)
            xg.append(pack_bf16x2(_fmul(x_lo, g_lo), _fmul(x_hi, g_hi)))
        if lane < CHUNKS:
            if warp < m:
                _sts_v4(xs_s + warp * XS_PITCH + lane * 16, xg[0], xg[1], xg[2], xg[3])
            else:
                _sts_v4(xs_s + warp * XS_PITCH + lane * 16, Int32(0), Int32(0), Int32(0), Int32(0))
        s2 = cutlass.select_(lat_lane, s2, Float32(0.0))
        for off in (16, 8, 4, 2, 1):
            s2 = s2 + _shfl_bfly_f32(s2, off)
        _cluster_wait()
        if warp < m:
            if lane < 2:
                # This CTA's sum of squares of token `warp` to its owners (row halves 0 and 1).
                _st_async_f32(sq_s + q * 4, warp + lane * MAX_M, bar_sq, s2)
        _bar_sync(1, THREADS)

        # ---- up projection: m-tiles w, w + 8, ..; 4-row partials to the line owners ----
        xrow = xs_s + gq * XS_PITCH
        bfr = []
        for j in cutlass.range_constexpr(KJ):
            cb, cs = _k_col(j)
            bfr += list(_lds_v2(xrow + (cb + cs * tq) * 2))
        stg = stage_s + warp * STAGE_WARP_BYTES
        for s in cutlass.range_constexpr(self.mpw):
            mt = warp + WARPS * s
            if mt < self.mtiles:
                # Two accumulator chains (even / odd K steps), added at the end.
                acc2 = [[Float32(0.0)] * 4, [Float32(0.0)] * 4]
                for j in cutlass.range_constexpr(KJ):
                    base = (s * KJ + j) * 4
                    ch = acc2[j % 2]
                    acc2[j % 2] = list(_mma_bf16(
                        (afr[base], afr[base + 1], afr[base + 2], afr[base + 3]),
                        bfr[2 * j], bfr[2 * j + 1], ch,
                    ))  # fmt: skip
                d0 = add_f32(acc2[0][0], acc2[1][0])
                d1 = add_f32(acc2[0][1], acc2[1][1])
                d2 = add_f32(acc2[0][2], acc2[1][2])
                d3 = add_f32(acc2[0][3], acc2[1][3])
                # d0 (row gq, token 2 tq), d1 (row gq, token 2 tq + 1), d2 / d3 (row gq + 8).
                _sts_f32(stg + ((2 * tq) * STAGE_PITCH + gq) * 4, d0)
                _sts_f32(stg + ((2 * tq + 1) * STAGE_PITCH + gq) * 4, d1)
                _sts_f32(stg + ((2 * tq) * STAGE_PITCH + gq + 8) * 4, d2)
                _sts_f32(stg + ((2 * tq + 1) * STAGE_PITCH + gq + 8) * 4, d3)
                cute.arch.sync_warp()
                tk = lane // 4
                r4 = (lane % 4) * 4
                if tk < m:
                    v0, v1, v2, v3 = _lds_v4_f32(stg + (tk * STAGE_PITCH + r4) * 4)
                    rloc = mt * 16 + r4
                    h = rloc // RO
                    dst = part_s + (q * RO + rloc - h * RO) * 4
                    _st_async_v4_f32(dst, tk + h * MAX_M, bar_part, v0, v1, v2, v3)
                cute.arch.sync_warp()

        # ---- owner: sum the CTAs' partials, scale by the token's RMS reciprocal, multicast ----
        if own_t < m:
            if warp == 0:
                done = _mbar_try_wait_cluster(bar_sq)
                while done == 0:
                    done = _mbar_try_wait_cluster(bar_sq)
                done = _mbar_try_wait_cluster(bar_part)
                while done == 0:
                    done = _mbar_try_wait_cluster(bar_part)
                if lane < RO // 8:
                    total = Float32(0.0)
                    for r in cutlass.range_constexpr(CLUSTER):
                        total = add_f32(total, _lds_f32(sq_s + r * 4))
                    inv = _rms_scale(total, eps)
                    parts = []
                    for sl in cutlass.range_constexpr(CLUSTER):
                        parts.append(_lds_v4_f32(part_s + (sl * RO + lane * 8) * 4))
                        parts.append(_lds_v4_f32(part_s + (sl * RO + lane * 8 + 4) * 4))
                    y = [Float32(0.0)] * 8
                    for sl in cutlass.range_constexpr(CLUSTER):
                        for i in cutlass.range_constexpr(4):
                            y[i] = add_f32(y[i], parts[2 * sl][i])
                            y[4 + i] = add_f32(y[4 + i], parts[2 * sl + 1][i])
                    ow = [
                        sanitize_bf16x2(pack_bf16x2(_fmul(y[2 * j], inv), _fmul(y[2 * j + 1], inv)))
                        for j in range(4)
                    ]
                    col = c0 + gcl * self.rg + own_h * RO + lane * 8
                    dst = (
                        pushed_ar.all_gather_row(
                            ag_base, ag_cur, ag_bytes, Int32(m), Int32(own_t), W, HIDDEN
                        )
                        + Int64(col) * 2
                    )
                    if cutlass.const_expr(self.mc_out):
                        multimem_st_v4(dst, ow[0], ow[1], ow[2], ow[3])
                    else:
                        st_v4(dst, ow[0], ow[1], ow[2], ow[3])

        # ---- advance the latent all-reduce's state once the experts (which read it) completed ----
        griddepcontrol_wait()
        if tidx == 0:
            pushed_ar.arrive_finish(
                lat_flags, Int32(self.nctas), cur, nbytes, Int32(m), W, LATENT, True
            )


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def rank_rows(world: int, rank: int):
    """``(first row, rows)`` of latent_up's output this rank computes: ``HIDDEN / world`` each."""
    n = HIDDEN // world
    return rank * n, n


def cluster_rows(rows: int) -> int:
    """Output rows per cluster for a rank's ``rows`` (0: not served): 128 for whole 128-row
    blocks, else :data:`ROWS_PER_CLUSTER`."""
    for rg in (128, ROWS_PER_CLUSTER):
        if rows > 0 and rows % rg == 0:
            return rg
    return 0


def can_run(m: int, world: int, rows: int = 0) -> bool:
    """Whether the kernel serves ``m`` tokens on ``world`` ranks with ``rows`` output rows on this
    rank (0: ``HIDDEN / world``)."""
    rows = rows or (HIDDEN // world if HIDDEN % world == 0 else 0)
    return 1 <= m <= MAX_M and world in pushed_ar.WORLD_SIZES and cluster_rows(rows) > 0


@functools.cache
def _compiled(world: int, rows: int, m: int, mc_out: bool, ag_follows: bool, device_index: int):
    kernel = KimiK3MoeTailKernel(world, rows, m, mc_out, ag_follows)
    z = Int64(0)
    with torch.cuda.device(device_index):
        compiled = cute.compile(
            kernel, z, z, z, z, z, z, z, Int32(0), Float32(1e-6),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )  # fmt: skip
    return compiled


def block_exponents(packed_scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """UE8M0 block exponents ``uint8 [n / 128, k / 128]`` of a weight from its DeepGEMM-packed
    scale (``fp8_block_scale_low_m_gemm``'s ``w_scale``: int32 ``[n, ceil(k / 512)]``, strides
    ``(1, n)``, byte ``kb % 4`` of word ``(row, kb // 4)``)."""
    rows = packed_scale[::128].contiguous()
    return rows.view(torch.uint8).view(n // 128, -1)[:, : k // 128].contiguous()


def run_moe_tail(
    *,
    m: int,
    world: int,
    lat_uc: int,
    lat_flags: torch.Tensor,
    weight: torch.Tensor,
    wexp: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    ag_base: int,
    ag_flags: torch.Tensor,
    c0: int,
    rows: int = 0,
    mc_out: bool = True,
    ag_follows: bool = False,
) -> None:
    """Launch :class:`KimiK3MoeTailKernel` on the current stream.

    Args:
        m: Tokens (1..8).
        world: Ranks of both workspaces.
        lat_uc / lat_flags: The latent all-reduce's workspace (local unicast base, uint32 [9]
            rotation state) the routed experts push their partials into; this kernel finishes it.
        weight: latent_up ``[HIDDEN, LATENT]`` FP8 E4M3 (all rows; the kernel reads rows
            ``[c0, c0 + rows)``).
        wexp: latent_up's UE8M0 block exponents ``uint8 [HIDDEN / 128, LATENT / 128]`` (1..254).
        gamma: The routed latent RMSNorm weight ``[LATENT]`` BF16.
        ag_base / ag_flags: The tail call's workspace (multicast base, or with ``mc_out=False``
            a local base) and its rotation state; the kernel writes the call's all-gather rows.
        c0: This rank's first output row.
        rows: This rank's output rows (0: ``HIDDEN / world``, :func:`rank_rows`).
        ag_follows: The tail call is the next call on the latent all-reduce's workspace.
    """
    rows = rows or HIDDEN // world
    if not can_run(m, world, rows):
        raise ValueError(f"the Kimi-K3 MoE tail kernel does not serve m={m} world={world}")
    if weight.dtype != torch.float8_e4m3fn or tuple(weight.shape) != (HIDDEN, LATENT):
        raise ValueError("weight must be the [7168, 3584] FP8 E4M3 latent_up")
    if wexp.dtype != torch.uint8 or tuple(wexp.shape) != (HIDDEN // 128, LATENT // 128):
        raise ValueError("wexp must be uint8 [56, 28]")
    if gamma.dtype != torch.bfloat16 or gamma.numel() != LATENT:
        raise ValueError("gamma must be BF16 [3584]")
    device = weight.device
    index = device.index if device.index is not None else torch.cuda.current_device()
    _compiled(world, rows, m, mc_out, ag_follows, index)(
        Int64(lat_uc),
        Int64(lat_flags.data_ptr()),
        Int64(weight.data_ptr()),
        Int64(wexp.data_ptr()),
        Int64(gamma.data_ptr()),
        Int64(ag_base),
        Int64(ag_flags.data_ptr()),
        Int32(c0),
        Float32(eps),
        cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
    )
