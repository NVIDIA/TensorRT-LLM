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
"""Residual + RMSNorm finish of a pushed MNNVL all-reduce (SM100 / SM103 / SM107).

The finish of :mod:`mnnvl_pushed_allreduce` for ``AllReduceFusionOp.RESIDUAL_RMS_NORM``: a
producer kernel pushed its BF16 ``[m, hidden]`` rows into the Lamport slots of the call
(``push_slot`` / ``push_v4``), and :class:`PushedAllReduceNormFinishKernel` runs the rest:

* every thread owns eight consecutive values of a row; the rows are split over a cluster of CTAs
  as the C++ launcher splits them (:func:`grid_config`, ``adjustGridConfig``);
* it polls the slots of its values (one-shot: every rank's slot of the row; two-shot: the row's
  owner sums its scatter slots and multicasts the row, the others poll the broadcast row), sums
  the ranks in FP32 in rank order and rounds to BF16;
* adds the residual in BF16 (``add.rn.bf16x2``) and stores the new residual;
* RMSNorm: BF16 squares summed in element order, warp butterflies (a partial last warp adds zero
  for its missing lanes), the warp-0 butterfly over the warp sums, the CTA sums of the row in CTA
  order, ``rsqrt.approx(div.rn(sum, hidden) + eps)``, then the output as the one-shot kernel
  (``bf16((x * r) * gamma)``) or as ``rmsNormLamport`` after the two-shot all-reduce
  (``bf16((gamma * x) * r)``).

The outputs are bitwise ``mnnvl_fusion_allreduce(RESIDUAL_RMS_NORM)`` of the pushed input. Launch
and ordering follow :class:`mnnvl_pushed_allreduce.PushedAllReduceFinishKernel`: the kernel
releases its dependents first, reads the rotation state and polls without waiting for its
producer (which releases it only after its own ``griddepcontrol.wait``); the residual and gamma
it reads are older than the producer. CTA (0, 0) alone waits for the producer before it advances
the rotation state, after every CTA has counted itself; the CTAs re-arm the previous call's buffer.
"""

from __future__ import annotations

import functools
import struct
from typing import TYPE_CHECKING

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass.cutlass_dsl import T, dsl_user_op

from .mnnvl_pushed_allreduce import (
    FLAG_ARRIVALS,
    NUM_BUFFERS,
    WORLD_SIZES,
    clear_dirty_stages,
    finish_two_shot_v4,
    lamport_state,
    reduce_chunk,
    reduce_slots_v4,
    st_v4,
    stage_addr,
)
from .raw_ptx import add_f32 as _add_f32
from .raw_ptx import asm as _asm
from .raw_ptx import bf16x2_to_f32x2 as _bf16x2_f32
from .raw_ptx import cluster_sync as _cluster_sync_release_acquire
from .raw_ptx import fv as _fv
from .raw_ptx import iv as _iv
from .raw_ptx import ld_acquire_u32 as _ld_acquire_u32
from .raw_ptx import ld_v4 as _ld_v4
from .raw_ptx import lds_f32 as _lds_f32
from .raw_ptx import mul_f32 as _fmul
from .raw_ptx import pack_bf16x2 as _pack_bf16x2
from .raw_ptx import red_release_add as _red_release_add
from .raw_ptx import st_u32 as _st_u32
from .raw_ptx import sts_f32 as _sts_f32
from .utils import griddepcontrol_launch_dependents, griddepcontrol_wait

if TYPE_CHECKING:
    from ...distributed.ops import LamportWorkspaceView

#: Values per thread (one 16-byte access).
ELTS = 8


def grid_config(num_tokens: int, hidden: int, sm_count: int) -> tuple[int, int]:
    """``(threads per CTA, CTAs per row)`` of the C++ one-shot kernel and of ``rmsNormLamport``
    for BF16 rows (``adjustGridConfig`` with eight values per thread), or ``(0, 0)`` where it
    takes more than one load per thread."""
    threads = -(-hidden // ELTS)
    cluster = 8
    while threads % cluster and cluster > 1:
        cluster //= 2
    max_cluster = cluster
    block = -(-threads // cluster)
    while block < 128 and cluster >= 2:
        block *= 2
        cluster //= 2
    while num_tokens * cluster > sm_count and cluster > 1 and block <= 512:
        block *= 2
        cluster //= 2
    while cluster < max_cluster:
        cand = cluster * 2
        cand_block = -(-threads // cand)
        if cand_block < 64 or num_tokens * cand > sm_count:
            break
        cluster, block = cand, cand_block
    if block > 1024 or block * cluster * ELTS != hidden:
        return 0, 0
    return block, cluster


# ---------------------------------------------------------------------------
# Inline PTX helpers
# ---------------------------------------------------------------------------


@dsl_user_op
def _add_bf16x2(a, b, *, loc=None, ip=None):
    """Elementwise BF16 sum of two pairs, rounded to nearest (``__hadd2``)."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(a, loc=loc, ip=ip), _iv(b, loc=loc, ip=ip)],
            "add.rn.bf16x2 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sq_bf16x2(word, *, loc=None, ip=None):
    """Elementwise BF16 square of a pair (``mul.rn.bf16x2``)."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(word, loc=loc, ip=ip)],
            "mul.rn.bf16x2 $0, $1, $1;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_add(value, offset: int, lanes: int, lane, *, loc=None, ip=None):
    """One butterfly step of the C++ warp sums over the first ``lanes`` lanes of a warp:
    ``value + (partner < lanes ? partner's value : 0)``."""
    mask = (1 << lanes) - 1 if lanes < 32 else 0xFFFFFFFF
    return Float32(
        _asm(
            T.f32(),
            [_fv(value, loc=loc, ip=ip), _iv(lane, loc=loc, ip=ip)],
            "{\n.reg .pred p;\n.reg .b32 t;\n.reg .f32 o;\n"
            f"shfl.sync.bfly.b32 o, $1, {offset}, 0x1f, {mask};\n"
            f"xor.b32 t, $2, {offset};\nsetp.lt.u32 p, t, {lanes};\n"
            "selp.f32 o, o, 0f00000000, p;\nadd.rn.f32 $0, $1, o;\n}",
            "=f,f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _rms_scale(total, eps, hidden_hex: str, *, loc=None, ip=None):
    """``rsqrtf(total / hidden + eps)`` as the C++ kernels compute it (IEEE division, then add)."""
    return Float32(
        _asm(
            T.f32(),
            [_fv(total, loc=loc, ip=ip), _fv(eps, loc=loc, ip=ip)],
            "{\n.reg .f32 q;\n"
            f"div.rn.f32 q, $1, 0f{hidden_hex};\n"
            "add.rn.f32 q, q, $2;\nrsqrt.approx.f32 $0, q;\n}",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_cluster_f32(local_addr, peer, value, *, loc=None, ip=None):
    """Store ``value`` at ``local_addr`` of cluster CTA ``peer``'s shared memory."""
    _asm(
        None,
        [_iv(local_addr, loc=loc, ip=ip), _iv(peer, loc=loc, ip=ip), _fv(value, loc=loc, ip=ip)],
        "{\n.reg .b32 ra;\nmapa.shared::cluster.u32 ra, $0, $1;\n"
        "st.shared::cluster.f32 [ra], $2;\n}",
        "r,r,f",
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class PushedAllReduceNormFinishKernel:
    """``norm(allreduce(pushed) + residual)`` and the new residual for ``m`` rows of ``hidden``
    BF16 values on ``world`` ranks (see the module docstring)."""

    def __init__(
        self, world: int, hidden: int, threads: int, cluster: int, one_shot: bool, use_pdl: bool
    ):
        if threads * cluster * ELTS != hidden:
            raise ValueError(f"{cluster} x {threads} threads do not cover {hidden} values")
        if world not in WORLD_SIZES:
            raise ValueError(f"world={world} is not one of {WORLD_SIZES}")
        self.world = world
        self.hidden = hidden
        self.threads = threads
        self.cluster = cluster
        self.one_shot = one_shot
        self.use_pdl = use_pdl
        self.warps = (threads + 31) // 32
        self.last_lanes = threads - 32 * (self.warps - 1)
        self.chunk = reduce_chunk(world, one_shot)
        self.hidden_hex = struct.pack(">f", float(hidden)).hex()

    @cute.jit
    def __call__(
        self,
        uc_local: Int64,
        mc_base: Int64,
        flags: Int64,
        res_in: Int64,
        gamma: Int64,
        out: Int64,
        res_out: Int64,
        m: Int32,
        rank: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            warp_sum: cute.struct.MemRange[cutlass.Float32, 32]
            block_sum: cute.struct.MemRange[cutlass.Float32, 1]
            cta_sum: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 8], 16]

        self.shared_storage = SharedStorage
        self.kernel(uc_local, mc_base, flags, res_in, gamma, out, res_out, m, rank, eps).launch(
            grid=(m, self.cluster, 1),
            block=(self.threads, 1, 1),
            cluster=(1, self.cluster, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        uc_local: Int64,
        mc_base: Int64,
        flags: Int64,
        res_in: Int64,
        gamma: Int64,
        out: Int64,
        res_out: Int64,
        m: Int32,
        rank: Int32,
        eps: Float32,
    ):
        W, H, threads = self.world, self.hidden, self.threads
        tidx, _, _ = cute.arch.thread_idx()
        tok, cy, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        warp_sum = storage.warp_sum.data_ptr().toint()
        block_sum = storage.block_sum.data_ptr().toint()
        cta_sum = storage.cta_sum.data_ptr().toint()
        arrivals = flags + Int64(FLAG_ARRIVALS * 4)

        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_launch_dependents()
        cur, dirty, nbytes, dstages, c0, c1, c2, c3 = lamport_state(flags)
        # Count this CTA once every thread has read the state.
        cute.arch.barrier()
        if tidx == 0:
            _red_release_add(arrivals, Int32(1))

        col = (cy * threads + tidx) * ELTS
        row_off = Int64(tok * H + col) * 2
        rw = _ld_v4(res_in + row_off)
        gw = _ld_v4(gamma + Int64(col) * 2)

        # Re-arm the previous call's buffer (every stage it used) while the rows travel.
        clear_dirty_stages(
            uc_local,
            dirty,
            nbytes,
            dstages,
            c0,
            c1,
            c2,
            c3,
            (tok * self.cluster + cy) * threads + tidx,
            m * self.cluster * threads,
        )

        red = [Int32(0)] * 4
        if cutlass.const_expr(self.one_shot):
            buf = stage_addr(uc_local, cur, Int32(0), 1, nbytes)
            acc = reduce_slots_v4(buf + Int64(tok * W * H + col) * 2, H * 2, W, self.chunk)
            red = [_pack_bf16x2(acc[2 * j], acc[2 * j + 1]) for j in range(4)]
        else:
            w0, w1, w2, w3 = finish_two_shot_v4(
                uc_local, mc_base, cur, nbytes, tok, col * 2, rank, Int32(1), W, H
            )
            red = [w0, w1, w2, w3]

        y = [_add_bf16x2(red[j], rw[j]) for j in range(4)]
        st_v4(res_out + row_off, y[0], y[1], y[2], y[3])

        # RMSNorm: this thread's BF16 squares in element order.
        thread_sum = Float32(0.0)
        for j in cutlass.range_constexpr(4):
            lo, hi = _bf16x2_f32(_sq_bf16x2(y[j]))
            thread_sum = _add_f32(thread_sum, lo)
            thread_sum = _add_f32(thread_sum, hi)
        v = thread_sum
        if cutlass.const_expr(self.last_lanes < 32):
            if warp == self.warps - 1:
                for off in (16, 8, 4, 2, 1):
                    v = _shfl_add(v, off, self.last_lanes, lane)
            else:
                for off in (16, 8, 4, 2, 1):
                    v = _shfl_add(v, off, 32, lane)
        else:
            for off in (16, 8, 4, 2, 1):
                v = _shfl_add(v, off, 32, lane)
        if lane == 0:
            _sts_f32(warp_sum + warp * 4, v)
        cute.arch.barrier()
        if warp == 0:
            b = Float32(0.0)
            if lane < self.warps:
                b = _lds_f32(warp_sum + lane * 4)
            for off in (16, 8, 4, 2, 1):
                b = _shfl_add(b, off, 32, lane)
            if lane == 0:
                _sts_f32(block_sum, b)
        cute.arch.barrier()
        total = _lds_f32(block_sum)
        if cutlass.const_expr(self.cluster > 1):
            # The row's CTA sums: CTA c's into every peer's slot c, then added in CTA order.
            if tidx < self.cluster:
                _st_cluster_f32(cta_sum + cy * 4, tidx, total)
            _cluster_sync_release_acquire()
            total = Float32(0.0)
            for c in cutlass.range_constexpr(self.cluster):
                total = _add_f32(total, _lds_f32(cta_sum + c * 4))
        scale = _rms_scale(total, eps, self.hidden_hex)

        ow = []
        for j in cutlass.range_constexpr(4):
            x_lo, x_hi = _bf16x2_f32(y[j])
            g_lo, g_hi = _bf16x2_f32(gw[j])
            if cutlass.const_expr(self.one_shot):
                ow.append(
                    _pack_bf16x2(_fmul(_fmul(x_lo, scale), g_lo), _fmul(_fmul(x_hi, scale), g_hi))
                )
            else:
                ow.append(
                    _pack_bf16x2(_fmul(_fmul(g_lo, x_lo), scale), _fmul(_fmul(g_hi, x_hi), scale))
                )
        st_v4(out + row_off, ow[0], ow[1], ow[2], ow[3])

        # CTA (0, 0): the next call's state, once the producer (which reads the state after its
        # own wait) has completed and every CTA has read the state.
        if tok == 0:
            if cy == 0:
                if tidx == 0:
                    if cutlass.const_expr(self.use_pdl):
                        griddepcontrol_wait()
                    total_ctas = m * self.cluster
                    arrived = _ld_acquire_u32(arrivals)
                    while arrived < total_ctas:
                        arrived = _ld_acquire_u32(arrivals)
                    row_bytes = H * 2
                    if cutlass.const_expr(self.one_shot):
                        st_v4(flags, (cur + 1) % NUM_BUFFERS, cur, nbytes, Int32(1))
                        st_v4(flags + Int64(16), m * (W * row_bytes), Int32(0), Int32(0), Int32(0))
                    else:
                        st_v4(flags, (cur + 1) % NUM_BUFFERS, cur, nbytes, Int32(2))
                        st_v4(
                            flags + Int64(16),
                            ((m + (W - 1)) // W) * (W * row_bytes),
                            m * row_bytes,
                            Int32(0),
                            Int32(0),
                        )
                    _st_u32(arrivals, Int32(0))


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def _stream() -> cuda.CUstream:
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


@functools.cache
def _compiled_norm_finish(world, hidden, threads, cluster, one_shot, use_pdl):
    kernel = PushedAllReduceNormFinishKernel(world, hidden, threads, cluster, one_shot, use_pdl)
    a = Int64(1 << 20)
    return cute.compile(kernel, a, a, a, a, a, a, a, Int32(1), Int32(0), Float32(1e-6), _stream())


@functools.cache
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def norm_finish_config(num_tokens: int, hidden: int, device: torch.device) -> tuple[int, int]:
    """``(threads, cluster)`` of :class:`PushedAllReduceNormFinishKernel`, ``(0, 0)`` if none."""
    return grid_config(num_tokens, hidden, _sm_count(device.index))


def run_finish_residual_rms_norm(
    view: LamportWorkspaceView,
    one_shot: bool,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    residual_out: torch.Tensor,
    use_pdl: bool = True,
) -> None:
    """Finish the all-reduce every rank pushed into ``view`` with the ``one_shot`` (else
    two-shot) transport, add ``residual`` and RMSNorm with ``gamma``: ``out`` / ``residual_out``
    (contiguous BF16 ``[m, hidden]``) as ``mnnvl_fusion_allreduce(RESIDUAL_RMS_NORM)`` returns
    them."""
    from .mnnvl_pushed_allreduce import check_transport

    m, hidden = out.shape
    check_transport(view, m, hidden, one_shot)
    threads, cluster = norm_finish_config(m, hidden, out.device)
    if not threads:
        raise ValueError(f"no single-load launch shape for {m} x {hidden}")
    _compiled_norm_finish(view.world, hidden, threads, cluster, one_shot, use_pdl)(
        Int64(view.uc_local),
        Int64(view.mc_base),
        Int64(view.flags.data_ptr()),
        Int64(residual.data_ptr()),
        Int64(gamma.data_ptr()),
        Int64(out.data_ptr()),
        Int64(residual_out.data_ptr()),
        Int32(m),
        Int32(view.rank),
        Float32(eps),
        _stream(),
    )


__all__ = [
    "PushedAllReduceNormFinishKernel",
    "grid_config",
    "norm_finish_config",
    "run_finish_residual_rms_norm",
]
