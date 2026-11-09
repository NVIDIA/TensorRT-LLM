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
"""Kimi-K3 shared expert of at most 8 rows on a few SMs beside the routed experts
(``enable_min_latency``).

The shared expert of a decode step is a TP-sharded gated MLP: this rank computes
``act = situ(x @ Wg^T, x @ Wu^T)`` for its ``I = 6144 / W`` intermediate columns and the
row-parallel partial ``act @ Wd^T`` (``[M, 7168]``), which the column-parallel MoE tail pushes into
its all-gather call (or the kernel stores straight into that call's slots: the output push).
:class:`KimiK3SharedExpertKernel` does all of it with ``N`` persistent CTAs on the shared stream,
beside the routed experts' grid:

* Weights. Each CTA streams its weights once, in consumption order, through a ring of shared
  memory stages (``cp.async.bulk``, one producer warp, full / empty mbarriers, a proxy fence
  between the compute warps' reads of a slot and its refill): the gate/up rows of its ``I / N``
  intermediate columns, then the down rows of its 16-row m-tiles (m-tile ``T`` of the 448 goes to
  CTA ``T mod N``). The fragments are prepared once on the host (:func:`persist_fragments`) in the
  ``mma.sync`` A order.
* Gate/up. Eight compute warps split K (BF16 ``mma.sync`` m16n8k16, FP32 accumulation); the
  partials are summed in warp order, rounded to BF16 and the SiTU activation (FP32, rounded to
  BF16) gives the CTA's activation columns.
* Exchange. Each CTA writes its columns into a global ``[8, I]`` BF16 buffer and counts itself
  (release); every CTA then waits for all ``N`` (acquire) and copies the whole activation into
  shared memory; the last CTA to have copied it re-arms the two counters.
* Down. Warp ``w`` multiplies m-tile ``8 g + w`` of the CTA's list (``K = I``) and stores the
  16 x M outputs as BF16 into the partial ``[M, 7168]``, or with the output push rounds them to
  BF16 (``-0`` as ``+0``) and multicasts them into slot ``(token, rank)`` of the all-gather call's
  current buffer (``mnnvl_pushed_allreduce.push_slot``, one-shot), the bits
  ``PushedAllReduceCopyKernel`` would push.

Numerics: the products are those of the BF16 GEMMs (FP32 accumulation, BF16 outputs) in a
different summation order, and the activation evaluates tanh / sigmoid with ``ex2.approx``: not
bitwise. Launched without PDL; consecutive launches on a stream reuse the exchange buffer and
counters (the stream serializes them). The output push reads the call's rotation state before the
down projection: the call must be the current one on its workspace when the kernel starts.
"""

from __future__ import annotations

import functools
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op

from . import mnnvl_pushed_allreduce as pushed_ar
from .raw_ptx import asm as _asm
from .raw_ptx import (
    atom_add_acq_rel_u32,
    ld_acquire_u32,
    ld_volatile_v4,
    multimem_st_v4,
    pack_bf16x2,
    sanitize_bf16x2,
    st_u32,
    st_v4,
)
from .raw_ptx import bar_sync as _bar_sync
from .raw_ptx import bulk_g2s as _bulk_g2s
from .raw_ptx import fence_mbar_init as _fence_mbar_init
from .raw_ptx import fence_proxy_async as _fence_proxy_async
from .raw_ptx import fv as _fv
from .raw_ptx import iv as _iv
from .raw_ptx import lds_b32 as _lds_b32
from .raw_ptx import lds_f32 as _lds_f32
from .raw_ptx import lds_v4 as _lds_v4
from .raw_ptx import lds_v4_f32 as _lds_v4_f32
from .raw_ptx import lv as _lv
from .raw_ptx import mbar_arrive as _mbar_arrive
from .raw_ptx import mbar_expect_tx as _mbar_expect_tx
from .raw_ptx import mbar_init as _mbar_init
from .raw_ptx import mma_bf16 as _mma_bf16
from .raw_ptx import mul_f32 as _mul
from .raw_ptx import sts_f32 as _sts_f32
from .raw_ptx import sts_v4 as _sts_v4

__all__ = [
    "HIDDEN",
    "INTER",
    "MAX_M",
    "KimiK3SharedExpertKernel",
    "PersistWorkspace",
    "can_run",
    "persist_fragments",
    "run_shared_expert",
]

HIDDEN = 7168
INTER = 6144
MAX_M = 8
#: Compute warps (K split of gate/up, m-tiles of down) and the producer warp.
WARPS = 8
THREADS = (WARPS + 1) * 32
#: Bytes of one ``mma.sync`` A fragment (32 lanes x 16 bytes): one (m-tile, K step).
FRAG = 512
#: Gate/up K steps per warp (K = HIDDEN, eight warps).
GU_STEPS = HIDDEN // 16 // WARPS
#: 16-row m-tiles of the down projection's output.
DOWN_TILES = HIDDEN // 16
#: Shared memory the kernel may use (bytes).
SMEM_BUDGET = 220 * 1024
#: Words per token of the gate/up partials in shared memory (16 rows).
RED_PITCH = 16
#: FP32 words per token row of a warp's staged down tile (16 rows + padding).
STAGE_PITCH = 20


@dsl_user_op
def _mbar_try_wait(addr, parity, *, loc=None, ip=None):
    """1 once the phase of parity ``parity`` of the mbarrier at ``addr`` has completed."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(addr, loc=loc, ip=ip), _iv(parity, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2;\n"
            "selp.u32 $0, 1, 0, p;\n}",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fence_acq_rel_gpu(*, loc=None, ip=None):
    _asm(None, [], "fence.acq_rel.gpu;", "", loc=loc, ip=ip)


@dsl_user_op
def _ld_b32(addr, *, loc=None, ip=None):
    """A 32-bit word of the layer input (complete before the launch)."""
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.global.nc.b32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )  # fmt: skip
    )


@dsl_user_op
def _tanh(x, *, loc=None, ip=None):
    """``tanh(x)`` in FP32: an odd Taylor polynomial below ``|x| = 0.25``, else
    ``(1 - e) / (1 + e)`` with ``e = 2^(-2 |x| log2(e))`` (``ex2.approx``) and the sign."""
    return Float32(
        _asm(
            T.f32(),
            [_fv(x, loc=loc, ip=ip)],
            "{\n.reg .f32 a, x2, p, e, n, d, t;\n.reg .pred small, neg;\n"
            "abs.f32 a, $1;\n"
            "setp.lt.f32 small, a, 0f3E800000;\n"
            "setp.lt.f32 neg, $1, 0f00000000;\n"
            # polynomial: x (1 + x2 (-1/3 + x2 (2/15 + x2 (-17/315 + x2 62/2835))))
            "mul.rn.f32 x2, $1, $1;\n"
            "fma.rn.f32 p, x2, 0f3CB327A4, 0fBD5D0DD1;\n"
            "fma.rn.f32 p, p, x2, 0f3E088889;\n"
            "fma.rn.f32 p, p, x2, 0fBEAAAAAB;\n"
            "mul.rn.f32 p, p, x2;\n"
            "fma.rn.f32 p, p, $1, $1;\n"
            # exponential form
            "mul.rn.f32 e, a, 0fC038AA3B;\n"
            "ex2.approx.f32 e, e;\n"
            "sub.rn.f32 n, 0f3F800000, e;\n"
            "add.rn.f32 d, 0f3F800000, e;\n"
            "div.rn.f32 t, n, d;\n"
            "@neg neg.f32 t, t;\n"
            "selp.f32 $0, p, t, small;\n}",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sigmoid(x, *, loc=None, ip=None):
    """``1 / (1 + exp(-x))`` in FP32 (``ex2.approx``, IEEE division)."""
    return Float32(
        _asm(
            T.f32(),
            [_fv(x, loc=loc, ip=ip)],
            "{\n.reg .f32 e, d;\n"
            "mul.rn.f32 e, $1, 0fBFB8AA3B;\n"
            "ex2.approx.f32 e, e;\n"
            "add.rn.f32 d, 0f3F800000, e;\n"
            "div.rn.f32 $0, 0f3F800000, d;\n}",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _div(a, b, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16_round(x, *, loc=None, ip=None):
    """``x`` rounded to BF16 (nearest even), as FP32."""
    return Float32(
        _asm(
            T.f32(),
            [_fv(x, loc=loc, ip=ip)],
            "{\n.reg .b16 h;\n.reg .b32 w;\ncvt.rn.bf16.f32 h, $1;\n"
            "mov.b32 w, {0, h};\nmov.b32 $0, w;\n}",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


class _Plan:
    """Static sizes of one configuration (``world`` ranks, ``sms`` CTAs)."""

    def __init__(self, world: int, sms: int):
        if INTER % world or (INTER // world) % 16:
            raise ValueError(f"world={world}: the intermediate shard is not whole K steps")
        self.world = world
        self.sms = sms
        self.inter = INTER // world
        if self.inter % sms or (self.inter // sms) % 8:
            raise ValueError(f"{self.inter} intermediate columns do not split into {sms} CTAs")
        self.ic = self.inter // sms
        self.gt = 2 * self.ic // 16
        self.kd = self.inter // 16
        if self.kd % self.gt:
            raise ValueError(f"{self.kd} down K steps do not split into stages of {self.gt}")
        self.stage_bytes = self.gt * WARPS * FRAG
        self.gu_stages = GU_STEPS
        self.dt = -(-DOWN_TILES // sms)
        self.groups = -(-self.dt // WARPS)
        self.dpg = self.kd // self.gt
        self.stages = self.gu_stages + self.groups * self.dpg
        self.cta_bytes = self.stages * self.stage_bytes
        self.red_bytes = WARPS * self.gt * MAX_M * RED_PITCH * 4
        self.act_pitch = self.inter * 2 + 16
        self.act_bytes = MAX_M * self.act_pitch
        self.out_bytes = WARPS * MAX_M * STAGE_PITCH * 4
        fixed = self.red_bytes + self.act_bytes + self.out_bytes + 1024
        self.ring = min(self.stages, (SMEM_BUDGET - fixed) // self.stage_bytes)
        if self.ring < 2:
            raise ValueError(f"world={world} sms={sms}: no room for a 2-stage weight ring")
        if self.gt > 12:
            raise ValueError(f"{self.gt} gate/up m-tiles per CTA: too many accumulators")


class KimiK3SharedExpertKernel:
    """The shared expert of ``m <= 8`` rows of one rank of ``world`` on ``sms`` CTAs (see the
    module docstring). ``push``: the output push into a ``world``-rank all-gather call (multicast,
    or with ``mc_out=False`` plain stores through a local base)."""

    def __init__(
        self,
        world: int,
        m: int,
        sms: int,
        beta: float,
        lbeta: Optional[float],
        push: bool = False,
        mc_out: bool = True,
    ):
        if not 1 <= m <= MAX_M:
            raise ValueError(f"m={m}: 1..{MAX_M}")
        self.p = _Plan(world, sms)
        self.world = world
        self.m = m
        self.sms = sms
        self.beta = float(beta)
        self.lbeta = None if lbeta is None else float(lbeta)
        self.push = push
        self.mc_out = mc_out

    @cute.jit
    def __call__(
        self,
        x_addr: Int64,
        w_addr: Int64,
        xch_addr: Int64,
        ctr_addr: Int64,
        out_addr: Int64,
        out_flags: Int64,
        rank: Int32,
        stream: cuda.CUstream,
    ):
        p = self.p

        @cute.struct
        class SharedStorage:
            full: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, p.ring], 16]
            empty: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, p.ring], 16]
            ring: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, p.ring * p.stage_bytes], 128
            ]
            red: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, p.red_bytes], 16]
            act: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, p.act_bytes], 16]
            out: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, p.out_bytes], 16]

        self.shared_storage = SharedStorage
        self.kernel(x_addr, w_addr, xch_addr, ctr_addr, out_addr, out_flags, rank).launch(
            grid=(self.sms, 1, 1),
            block=(THREADS, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            min_blocks_per_mp=1,
            # The largest carveout: the next kernels' CTAs may land beside this grid's.
            preferred_smem_carveout=100,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x_addr: Int64,
        w_addr: Int64,
        xch_addr: Int64,
        ctr_addr: Int64,
        out_addr: Int64,
        out_flags: Int64,
        rank: Int32,
    ):
        p = self.p
        m = self.m
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        gq = lane // 4
        tq = lane % 4
        storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage)
        full = storage.full.data_ptr().toint()
        empty = storage.empty.data_ptr().toint()
        ring = storage.ring.data_ptr().toint()
        red = storage.red.data_ptr().toint()
        act_s = storage.act.data_ptr().toint()
        out_s = storage.out.data_ptr().toint()
        if tidx == 0:
            for s in cutlass.range_constexpr(p.ring):
                _mbar_init(full + 8 * s, 1)
                _mbar_init(empty + 8 * s, WARPS)
            _fence_mbar_init()
        cute.arch.sync_threads()
        w_cta = w_addr + Int64(cta) * Int64(p.cta_bytes)

        if warp == WARPS:
            # ---- producer: every stage of this CTA's weight stream through the ring ----
            if lane == 0:
                for s in cutlass.range(p.stages, unroll=1):
                    slot = s % p.ring
                    if s >= p.ring:
                        par = ((s // p.ring) - 1) % 2
                        done = _mbar_try_wait(empty + 8 * slot, par)
                        while done == 0:
                            done = _mbar_try_wait(empty + 8 * slot, par)
                        # The compute warps' reads of the slot (generic proxy) before the bulk
                        # copy's writes (async proxy).
                        _fence_proxy_async()
                    _mbar_expect_tx(full + 8 * slot, p.stage_bytes)
                    _bulk_g2s(
                        ring + slot * p.stage_bytes, w_cta + Int64(s) * p.stage_bytes,
                        p.stage_bytes, full + 8 * slot,
                    )  # fmt: skip
        else:
            # ---- gate/up: K steps 8 j + warp of every m-tile ----
            xb = cute.make_rmem_tensor((2 * GU_STEPS,), Int32)
            tok_ok = gq < m
            xrow = x_addr + Int64(cutlass.select_(tok_ok, gq, Int32(0)) * HIDDEN) * 2
            for j in cutlass.range_constexpr(GU_STEPS):
                col = 16 * (8 * j) + 16 * warp + 2 * tq
                v0 = _ld_b32(xrow + Int64(col) * 2)
                v1 = _ld_b32(xrow + Int64(col + 8) * 2)
                xb[2 * j] = cutlass.select_(tok_ok, v0, Int32(0))
                xb[2 * j + 1] = cutlass.select_(tok_ok, v1, Int32(0))
            acc = [[Float32(0.0)] * 4 for _ in range(p.gt)]
            for j in cutlass.range_constexpr(GU_STEPS):
                slot = j % p.ring
                par = (j // p.ring) % 2
                done = _mbar_try_wait(full + 8 * slot, Int32(par))
                while done == 0:
                    done = _mbar_try_wait(full + 8 * slot, Int32(par))
                base = ring + slot * p.stage_bytes
                for mt in cutlass.range_constexpr(p.gt):
                    a4 = _lds_v4(base + ((mt * WARPS + warp) * 32 + lane) * 16)
                    c = acc[mt]
                    acc[mt] = list(_mma_bf16(a4, xb[2 * j], xb[2 * j + 1], c))
                cute.arch.sync_warp()
                if lane == 0:
                    _mbar_arrive(empty + 8 * slot)
            # The warp's partials: red[warp][mt][token][row].
            for mt in cutlass.range_constexpr(p.gt):
                rb = red + ((warp * p.gt + mt) * MAX_M) * RED_PITCH * 4
                _sts_f32(rb + ((2 * tq) * RED_PITCH + gq) * 4, acc[mt][0])
                _sts_f32(rb + ((2 * tq + 1) * RED_PITCH + gq) * 4, acc[mt][1])
                _sts_f32(rb + ((2 * tq) * RED_PITCH + gq + 8) * 4, acc[mt][2])
                _sts_f32(rb + ((2 * tq + 1) * RED_PITCH + gq + 8) * 4, acc[mt][3])
            _bar_sync(1, WARPS * 32)

            # ---- activation of this CTA's columns: warp-order sums, BF16, SiTU, BF16 ----
            ic = p.ic
            for i0 in cutlass.range_constexpr(0, ic * MAX_M, WARPS * 32):
                item = tidx + i0
                if item < ic * MAX_M:
                    t = item // ic
                    col = item - t * ic
                    g = Float32(0.0)
                    u = Float32(0.0)
                    for w in cutlass.range_constexpr(WARPS):
                        gr = col
                        ur = ic + col
                        g = g + _lds_f32(
                            red + (((w * p.gt + gr // 16) * MAX_M + t) * RED_PITCH + gr % 16) * 4
                        )
                        u = u + _lds_f32(
                            red + (((w * p.gt + ur // 16) * MAX_M + t) * RED_PITCH + ur % 16) * 4
                        )
                    g = _bf16_round(g)
                    u = _bf16_round(u)
                    a = _mul(_mul(Float32(self.beta), _tanh(_div(g, Float32(self.beta)))),
                             _sigmoid(g))  # fmt: skip
                    ub = u
                    if cutlass.const_expr(self.lbeta is not None):
                        ub = _mul(Float32(self.lbeta), _tanh(_div(u, Float32(self.lbeta))))
                    r = _mul(a, ub)
                    if t < m:
                        xa = xch_addr + Int64(t * p.inter + cta * ic + col) * 2
                        _st_bf16_global(xa, r)
            _bar_sync(1, WARPS * 32)
            if tidx == 0:
                _fence_acq_rel_gpu()
                atom_add_acq_rel_u32(ctr_addr, Int32(1))
                arrived = ld_acquire_u32(ctr_addr)
                while arrived < self.sms:
                    arrived = ld_acquire_u32(ctr_addr)
            _bar_sync(1, WARPS * 32)
            # The whole activation [m, I] into shared memory (tokens >= m: zeros).
            vecs = MAX_M * (p.inter // 8)
            for v0 in cutlass.range_constexpr(0, vecs, WARPS * 32):
                v = tidx + v0
                if v < vecs:
                    t = v // (p.inter // 8)
                    c8 = (v - t * (p.inter // 8)) * 8
                    dst = act_s + t * p.act_pitch + c8 * 2
                    if t < m:
                        w4 = ld_volatile_v4(xch_addr + Int64(t * p.inter + c8) * 2)
                        _sts_v4(dst, w4[0], w4[1], w4[2], w4[3])
                    else:
                        _sts_v4(dst, Int32(0), Int32(0), Int32(0), Int32(0))
            _bar_sync(1, WARPS * 32)
            if tidx == 0:
                # Every CTA has copied the activation once this count completes: re-arm.
                old = atom_add_acq_rel_u32(ctr_addr + Int64(4), Int32(1))
                if old == self.sms - 1:
                    st_u32(ctr_addr, Int32(0))
                    st_u32(ctr_addr + Int64(4), Int32(0))

            # ---- down: m-tile 8 g + warp of this CTA's list, K = I ----
            cur = Int32(0)
            nbytes = Int32(0)
            if cutlass.const_expr(self.push):
                # The all-gather call's buffer and its bytes (``lamport_state``).
                f = ld_volatile_v4(out_flags)
                cur, nbytes = f[0], f[2]
            arow = act_s + gq * p.act_pitch
            stg = out_s + warp * MAX_M * STAGE_PITCH * 4
            for g in cutlass.range_constexpr(p.groups):
                acc2 = [[Float32(0.0)] * 4, [Float32(0.0)] * 4]
                for hh in cutlass.range_constexpr(p.dpg):
                    s = GU_STEPS + g * p.dpg + hh
                    slot = s % p.ring
                    par = (s // p.ring) % 2
                    done = _mbar_try_wait(full + 8 * slot, Int32(par))
                    while done == 0:
                        done = _mbar_try_wait(full + 8 * slot, Int32(par))
                    base = ring + slot * p.stage_bytes
                    for kk in cutlass.range_constexpr(p.gt):
                        k = hh * p.gt + kk
                        a4 = _lds_v4(base + ((kk * WARPS + warp) * 32 + lane) * 16)
                        b0 = _lds_b32(arow + (16 * k + 2 * tq) * 2)
                        b1 = _lds_b32(arow + (16 * k + 2 * tq + 8) * 2)
                        c = acc2[k % 2]
                        acc2[k % 2] = list(_mma_bf16(a4, b0, b1, c))
                    cute.arch.sync_warp()
                    if lane == 0:
                        _mbar_arrive(empty + 8 * slot)
                d = [acc2[0][i] + acc2[1][i] for i in range(4)]
                _sts_f32(stg + ((2 * tq) * STAGE_PITCH + gq) * 4, d[0])
                _sts_f32(stg + ((2 * tq + 1) * STAGE_PITCH + gq) * 4, d[1])
                _sts_f32(stg + ((2 * tq) * STAGE_PITCH + gq + 8) * 4, d[2])
                _sts_f32(stg + ((2 * tq + 1) * STAGE_PITCH + gq + 8) * 4, d[3])
                cute.arch.sync_warp()
                tile = (WARPS * g + warp) * self.sms + cta
                tk = lane // 2
                h8 = (lane % 2) * 8
                if (lane < 2 * m) & (tile < DOWN_TILES):
                    e0, e1, e2, e3 = _lds_v4_f32(stg + (tk * STAGE_PITCH + h8) * 4)
                    e4, e5, e6, e7 = _lds_v4_f32(stg + (tk * STAGE_PITCH + h8 + 4) * 4)
                    o = [
                        pack_bf16x2(e0, e1),
                        pack_bf16x2(e2, e3),
                        pack_bf16x2(e4, e5),
                        pack_bf16x2(e6, e7),
                    ]
                    if cutlass.const_expr(self.push):
                        o = [sanitize_bf16x2(word) for word in o]
                        dst = pushed_ar.push_slot(
                            Int64(0), out_addr, cur, nbytes, tk, rank, tile * 16 + h8,
                            self.world, HIDDEN, True,
                        )  # fmt: skip
                        if cutlass.const_expr(self.mc_out):
                            multimem_st_v4(dst, o[0], o[1], o[2], o[3])
                        else:
                            st_v4(dst, o[0], o[1], o[2], o[3])
                    else:
                        dst = out_addr + (Int64(tk) * HIDDEN + Int64(tile * 16 + h8)) * 2
                        st_v4(dst, o[0], o[1], o[2], o[3])
                cute.arch.sync_warp()


@dsl_user_op
def _st_bf16_global(addr, x, *, loc=None, ip=None):
    """Store ``x`` rounded to BF16 (nearest even) into global memory."""
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _fv(x, loc=loc, ip=ip)],
        "{\n.reg .b16 h;\ncvt.rn.bf16.f32 h, $1;\nst.global.b16 [$0], h;\n}",
        "l,f",
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def can_run(m: int, world: int, sms: int) -> bool:
    """Whether the kernel serves ``m`` rows of the ``world``-rank shard on ``sms`` CTAs."""
    if not 1 <= m <= MAX_M or sms < 1:
        return False
    try:
        _Plan(world, sms)
    except ValueError:
        return False
    return True


def persist_fragments(gate_up: torch.Tensor, down: torch.Tensor, sms: int) -> torch.Tensor:
    """This rank's shared expert weights in the kernel's stream order: ``gate_up`` BF16
    ``[2 I, 7168]`` (gate rows, then up rows; the GatedMLP shard) and ``down`` BF16 ``[7168, I]``;
    returns int32 ``[sms, stages * stage_bytes / 4]`` (CTA ``c``: its gate/up stages, then its
    down stages; padding m-tiles zero)."""
    two_i, k = gate_up.shape
    inter = two_i // 2
    if k != HIDDEN or tuple(down.shape) != (HIDDEN, inter):
        raise ValueError(f"expected [2 I, {HIDDEN}] and [{HIDDEN}, I], got {gate_up.shape} "
                         f"and {down.shape}")  # fmt: skip
    if gate_up.dtype != torch.bfloat16 or down.dtype != torch.bfloat16:
        raise ValueError("BF16 weights expected")
    p = _Plan(INTER // inter, sms)
    ic, gt = p.ic, p.gt
    out = []
    for c in range(sms):
        lo, hi = c * ic, (c + 1) * ic
        rows = torch.cat((gate_up[lo:hi], gate_up[inter + lo : inter + hi]))
        # [gt mt, 2 h, 8 gq, 56 j, 8 w, 2 hk, 4 tq, 2 e] -> [j, mt, w, gq, tq, hk, h, e]
        v = rows.contiguous().view(gt, 2, 8, GU_STEPS, WARPS, 2, 4, 2)
        gu = v.permute(3, 0, 4, 2, 6, 5, 1, 7).reshape(-1)
        tiles = torch.zeros(p.groups * WARPS * 16, inter, dtype=torch.bfloat16, device=down.device)
        for i in range(p.groups * WARPS):
            t = i * sms + c
            if t < DOWN_TILES:
                tiles[16 * i : 16 * i + 16] = down[16 * t : 16 * t + 16]
        # [g, 8 w, 2 h, 8 gq, dpg hh, gt kk, 2 hk, 4 tq, 2 e] -> [g, hh, kk, w, gq, tq, hk, h, e]
        v = tiles.view(p.groups, WARPS, 2, 8, p.dpg, gt, 2, 4, 2)
        dn = v.permute(0, 4, 5, 1, 3, 7, 6, 2, 8).reshape(-1)
        out.append(torch.cat((gu, dn)).view(torch.int32))
    res = torch.stack(out)
    assert res.shape[1] * 4 == p.cta_bytes, (res.shape, p.cta_bytes)
    return res.contiguous()


class PersistWorkspace:
    """The kernel's exchange buffer (BF16 ``[8, I]``) and its two counters (zero between
    launches)."""

    def __init__(self, world: int, device):
        self.xch = torch.zeros(MAX_M, INTER // world, dtype=torch.bfloat16, device=device)
        self.ctr = torch.zeros(2, dtype=torch.int32, device=device)


@functools.cache
def _compiled(
    world: int,
    m: int,
    sms: int,
    beta: float,
    lbeta: Optional[float],
    push: bool,
    mc_out: bool,
    dev: int,
):
    kernel = KimiK3SharedExpertKernel(world, m, sms, beta, lbeta, push, mc_out)
    z = Int64(0)
    with torch.cuda.device(dev):
        compiled = cute.compile(
            kernel, z, z, z, z, z, z, Int32(0),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )  # fmt: skip
    return compiled


def run_shared_expert(
    *,
    x: torch.Tensor,
    frags: torch.Tensor,
    world: int,
    sms: int,
    ws: PersistWorkspace,
    out: Optional[torch.Tensor],
    beta: float,
    linear_beta: Optional[float],
    push_base: Optional[int] = None,
    push_flags: Optional[torch.Tensor] = None,
    rank: int = 0,
    mc_out: bool = True,
) -> Optional[torch.Tensor]:
    """Launch :class:`KimiK3SharedExpertKernel` on the current stream; returns ``out``.

    Args:
        x: The shared expert's input ``[m, 7168]`` BF16 (contiguous, complete).
        frags: :func:`persist_fragments` of this rank's weights for ``sms`` CTAs.
        world: The shared expert's TP size (``I = 6144 / world``).
        ws: The exchange buffer and counters (:class:`PersistWorkspace`).
        out: The partial ``[m, 7168]`` BF16 (contiguous; every element written); None with the
            output push.
        beta / linear_beta: SiTU's ``beta`` and ``linear_beta`` (``None``: no up clamp).
        push_base / push_flags / rank: The output push: a ``world``-rank all-gather call of
            ``m`` rows (``MnnvlPushTarget``; multicast base, or with ``mc_out=False`` a local
            base, and its rotation state), current on its workspace when the kernel starts; the
            kernel writes slot ``(token, rank)`` of every token.
    """
    m = x.shape[0]
    if x.dtype != torch.bfloat16 or tuple(x.shape[1:]) != (HIDDEN,) or not x.is_contiguous():
        raise ValueError("x must be contiguous BF16 [m, 7168]")
    if not can_run(m, world, sms):
        raise ValueError(f"m={m} world={world} sms={sms} not supported")
    lbeta = None if linear_beta is None else float(linear_beta)
    push = push_flags is not None
    if push:
        if out is not None or push_base is None:
            raise ValueError("the output push takes push_base and push_flags instead of out")
    elif out is None or (
        out.dtype != torch.bfloat16 or tuple(out.shape) != (m, HIDDEN) or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous BF16 [m, 7168]")
    _compiled(world, m, sms, float(beta), lbeta, push, push and mc_out, x.device.index)(
        Int64(x.data_ptr()),
        Int64(frags.data_ptr()),
        Int64(ws.xch.data_ptr()),
        Int64(ws.ctr.data_ptr()),
        Int64(push_base if push else out.data_ptr()),
        Int64(push_flags.data_ptr() if push else 0),
        Int32(rank),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    return out
