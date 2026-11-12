# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 routed latent all-reduce + RMSNorm (MNNVL one-shot, SM100/SM103/SM107) with an early
PDL trigger, bit-identical to ``oneshotAllreduceFusionKernel<WS, bf16, kARRMSNorm>``.

The C++ kernel (``mnnvlAllreduceKernels.cu``) serves the one-shot sizes of the latent all-reduce
(``M`` rows of 3584 BF16; ``M <= 38`` launches clusters of 4 CTAs x 112 threads per row). It
triggers its programmatic dependents only after the cross-rank reduction, so latent_up cannot
start loading its weight while the all-reduce waits for the other ranks.
:class:`K3LatentArNormKernel` does the same work with the same arithmetic and triggers right after
its own ``griddepcontrol.wait``:

* Thread ``t`` of cluster CTA ``c`` owns the eight values ``[8 (112 c + t), + 8)`` of its row. It
  replaces BF16 ``-0`` by ``+0`` (the Lamport sentinel), stores the 16 bytes into slot
  ``(row, rank)`` of every rank's current Lamport buffer through the multicast address, clears
  this rank's dirty buffer (the bytes the previous launch on the workspace left to clear), and
  polls its eight values of all ``WS`` slots in the local buffer. Ranks are summed in FP32 in
  chunks of eight like the C++ kernel (each chunk from 0, chunks added in rank order) and rounded
  to BF16.
* RMSNorm: BF16 squares (``mul.rn.bf16x2``) summed in element order; butterfly warp sums (the
  half warp first adds 0 for its missing upper half); the block sum ``(w0 + w2) + (w1 + w3)`` of
  the reference's warp-0 butterfly over four warp sums; the four CTA sums of the row added in CTA
  order from 0 through distributed shared memory; ``rsqrt.approx(div.rn(sum, 3584) + eps)``; then
  ``bf16((x * r) * gamma)``.
* Lamport state: the same flag words as ``LamportFlags`` (current / dirty buffer, buffer bytes,
  dirty stages, bytes to clear per stage, arrival counter), so the C++ one- and two-shot kernels
  can use the same workspace before or after it. Each warp counts one arrival once its lanes
  have read the flags; at the end, thread 0 of CTA (0, 0) waits for all warps of the grid, rotates
  the buffers and resets the counter.

Optionally the kernel also stores the MXFP8 form of its output for latent_up
(``k3_fp8w_lowm_mxfp8``): per 1x32 group (four consecutive threads) the low-M FP8 GEMM's own
amax / UE8M0 exponent / E4M3 conversion, so that GEMM on it is bitwise the low-M GEMM on the BF16
output.

PDL: ``griddepcontrol.wait`` first, and the dependents are triggered right after it (they may
only start reading what they do not wait for, e.g. latent_up its weight), not after the reduction
like the C++ kernel.

:class:`K3LatentArRowKernel` computes the same result, bit for bit,
with one 512-thread CTA per row: four groups of 128 threads play the four cluster CTAs (same warp
boundaries and reduction trees; every thread forms the group sums after one CTA barrier instead of
a cluster barrier); the rank's own slot is summed from registers instead of being polled back
from the multicast loopback, after the first polling round only the slots that have not arrived
are loaded again, and CTA 0 rotates the buffers while it waits for the other ranks instead of at
the end.
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

from ..k3r_gemm.k3_fp8w_lowm_gemm import _bf16x2_inv_scale, _quant_bf16x8_to_e4m3, _ue8m0_of_amax
from ..k3r_moe.k3_route_tail import _griddepcontrol_launch_dependents, _griddepcontrol_wait

#: Row width (the latent) and values per thread (one 16-byte access).
HIDDEN = 3584
ELTS = 8
#: Cluster of CTAs per row and threads per CTA, as the C++ launcher picks them for 3584 BF16 values
#: and ``M <= 38`` rows (adjustGridConfig): 4 x 112.
CLUSTER = 4
THREADS = HIDDEN // ELTS // CLUSTER
#: Warps per CTA (the last one is a half warp).
WARPS = (THREADS + 31) // 32
#: Largest row count with that launch shape.
MAX_M = 38
#: Ranks summed per chunk (``accumulateLamportRanksChunked``).
RANK_CHUNK = 8
#: World sizes the C++ launcher dispatches.
WORLD_SIZES = (2, 4, 8, 16, 32, 64)
#: Lamport buffers per workspace.
NUM_LAMPORT_BUFFERS = 3

assert THREADS * CLUSTER * ELTS == HIDDEN and THREADS % 16 == 0


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


def _iv(value, *, loc=None, ip=None):
    return Int32(value).ir_value(loc=loc, ip=ip)


def _lv(value, *, loc=None, ip=None):
    return Int64(value).ir_value(loc=loc, ip=ip)


def _fv(value, *, loc=None, ip=None):
    return Float32(value).ir_value(loc=loc, ip=ip)


def _unpack(res, ty, cls, n, *, loc=None, ip=None):
    return tuple(cls(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(n))


@dsl_user_op
def _ld_v4(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _ld_vol_v4(addr, *, loc=None, ip=None):
    """16 bytes of the Lamport buffer, read past L1 (peers write it over NVLink)."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.volatile.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def _st_v4(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)] + [_iv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_sentinel_v4(addr, *, loc=None, ip=None):
    """Four FP32 ``-0`` words (``getPackedLamportInit<float4, float>``)."""
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip)],
        "{\n.reg .b32 t;\nmov.b32 t, 0x80000000;\nst.global.v4.b32 [$0], {t, t, t, t};\n}",
        "l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sanitize(word, *, loc=None, ip=None):
    """A BF16 pair with each ``-0`` (0x8000) replaced by ``+0``."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(word, loc=loc, ip=ip)],
            "{\n.reg .pred p;\n.reg .b32 lo, hi;\n"
            "and.b32 lo, $1, 0xffff;\nsetp.eq.u32 p, lo, 0x8000;\nselp.b32 lo, 0, lo, p;\n"
            "and.b32 hi, $1, 0xffff0000;\nsetp.eq.u32 p, hi, 0x80000000;\nselp.b32 hi, 0, hi, p;\n"
            "or.b32 $0, lo, hi;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _clean4(w0, w1, w2, w3, *, loc=None, ip=None):
    """1 if none of the four words is the sentinel (``isLamportDirty`` is false)."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(w, loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
            "{\n.reg .pred p;\n"
            "setp.ne.b32 p, $1, 0x80000000;\nsetp.ne.and.b32 p, $2, 0x80000000, p;\n"
            "setp.ne.and.b32 p, $3, 0x80000000, p;\nsetp.ne.and.b32 p, $4, 0x80000000, p;\n"
            "selp.u32 $0, 1, 0, p;\n}",
            "=r,r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16x2_f32(word, *, loc=None, ip=None):
    """(low, high) BF16 of a pair as FP32."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [_iv(word, loc=loc, ip=ip)],
        "{\n.reg .b32 lo, hi;\nshl.b32 lo, $2, 16;\nand.b32 hi, $2, 0xffff0000;\n"
        "mov.b32 $0, lo;\nmov.b32 $1, hi;\n}",
        "=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 2, loc=loc, ip=ip)


@dsl_user_op
def _pack_bf16x2(lo, hi, *, loc=None, ip=None):
    """Round two FP32 values to a BF16 pair (``lo`` in the low half)."""
    return Int32(
        _asm(
            T.i32(),
            [_fv(lo, loc=loc, ip=ip), _fv(hi, loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
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
def _fadd(a, b, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fmul(a, b, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _rms_scale(total, eps, *, loc=None, ip=None):
    """``rsqrtf(total / 3584 + eps)`` as the C++ kernel computes it (IEEE division, then add)."""
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
def _shfl_bfly(value, offset: int, mask: int, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(value, loc=loc, ip=ip)],
            f"shfl.sync.bfly.b32 $0, $1, {offset}, 0x1f, {mask};",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _amax_bf16x8(w0, w1, w2, w3, *, loc=None, ip=None):
    """|x| max of eight BF16 values (four packed words) as a packed BF16 pair (garbage sign bit,
    masked by ``_ue8m0_of_amax``)."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(w, loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
            "{\n.reg .b32 m0, m1;\n"
            "max.xorsign.abs.bf16x2 m0, $1, $2;\n"
            "max.xorsign.abs.bf16x2 m1, $3, $4;\n"
            "max.xorsign.abs.bf16x2 $0, m0, m1;\n}",
            "=r,r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _amax_pair(a, b, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(a, loc=loc, ip=ip), _iv(b, loc=loc, ip=ip)],
            "max.xorsign.abs.bf16x2 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_bfly_b32(value, offset: int, mask: int, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(value, loc=loc, ip=ip)],
            f"shfl.sync.bfly.b32 $0, $1, {offset}, 0x1f, {mask};",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_v2(addr, v0, v1, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(v0, loc=loc, ip=ip), _iv(v1, loc=loc, ip=ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_u8(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.global.u8 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_f32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip), _fv(value, loc=loc, ip=ip)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.f32 $0, [$1];", "=f,r", loc=loc, ip=ip
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


@dsl_user_op
def _cluster_sync_release_acquire(*, loc=None, ip=None):
    _asm(
        None,
        [],
        "barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;",
        "",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _warp_sync(half_mask: int, is_half, *, loc=None, ip=None):
    """``bar.warp.sync`` over the warp's existing lanes (the last warp of the CTA may be a half
    warp: ``half_mask``)."""
    _asm(
        None,
        [_iv(is_half, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.s32 p, $0, 0;\n"
        f"@p bar.warp.sync {half_mask};\n@!p bar.warp.sync 0xffffffff;\n}}",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bar_sync(*, loc=None, ip=None):
    _asm(None, [], "bar.sync 0;", "", loc=loc, ip=ip)


@dsl_user_op
def _red_add_release_gpu(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "red.release.gpu.global.add.u32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_acquire_gpu_u32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_lv(addr, loc=loc, ip=ip)],
            "ld.acquire.gpu.global.u32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_u32(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.global.u32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


class K3LatentArNormKernel:
    """One-shot MNNVL all-reduce + RMSNorm of ``M`` rows of 3584 BF16 (see the module docstring)."""

    def __init__(
        self,
        world_size: int,
        use_pdl: bool,
        norm: bool = True,
        mxfp8: bool = False,
    ):
        if world_size not in WORLD_SIZES:
            raise ValueError(f"unsupported world size {world_size}")
        if mxfp8 and not norm:
            raise ValueError("the MXFP8 output is the normalized output's")
        self.world_size = world_size
        self.use_pdl = use_pdl
        # Without ``norm`` the output is the BF16 sum (``kAllReduce``), e.g. to check the rank order.
        self.norm = norm
        # Also store the MXFP8 form of the output (E4M3 codes, one UE8M0 exponent per 1x32 group).
        self.mxfp8 = mxfp8
        self.rank_chunk = min(RANK_CHUNK, world_size)

    @cute.jit
    def __call__(
        self,
        x_addr: Int64,
        gamma_addr: Int64,
        out_addr: Int64,
        xq_addr: Int64,
        xs_addr: Int64,
        uc_addr: Int64,
        mc_addr: Int64,
        flags_addr: Int64,
        rank: Int32,
        m: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            # Warp sums, the block sum, and the row's CTA sums (written by the cluster peers).
            warp_sum: cute.struct.MemRange[cutlass.Float32, WARPS]
            block_sum: cute.struct.MemRange[cutlass.Float32, 1]
            cta_sum: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, CLUSTER], 16]

        self.shared_storage = SharedStorage
        self.kernel(
            x_addr,
            gamma_addr,
            out_addr,
            xq_addr,
            xs_addr,
            uc_addr,
            mc_addr,
            flags_addr,
            rank,
            m,
            eps,
        ).launch(
            grid=(m, CLUSTER, 1),
            block=(THREADS, 1, 1),
            cluster=(1, CLUSTER, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        x_addr: Int64,
        gamma_addr: Int64,
        out_addr: Int64,
        xq_addr: Int64,
        xs_addr: Int64,
        uc_addr: Int64,
        mc_addr: Int64,
        flags_addr: Int64,
        rank: Int32,
        m: Int32,
        eps: Float32,
    ):
        ws = self.world_size
        tidx, _, _ = cute.arch.thread_idx()
        row, cy, _ = cute.arch.block_idx()
        crank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        warp_sum = storage.warp_sum.data_ptr().toint()
        block_sum = storage.block_sum.data_ptr().toint()
        cta_sum = storage.cta_sum.data_ptr().toint()

        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
            _griddepcontrol_launch_dependents()

        # LamportFlags: [current, dirty, bytes per buffer, dirty stages], bytes to clear x4.
        f0, f1, f2, f3 = _ld_v4(flags_addr)
        c0, c1, c2, c3 = _ld_v4(flags_addr + Int64(16))
        # One arrival per warp once its lanes hold the flags: CTA (0, 0) rotates them only after
        # every warp of the grid has read them.
        counter = flags_addr + Int64(32)
        _warp_sync(0xFFFF if THREADS % 32 else 0xFFFFFFFF, warp == WARPS - 1)
        if lane == 0:
            _red_add_release_gpu(counter, Int32(1))
        cur = Int64(f0)
        dirty = Int64(f1)
        buf_bytes = Int64(f2) & Int64(0xFFFFFFFF)
        dirty_stages = f3
        col = (crank * THREADS + tidx) * ELTS
        cur_off = cur * buf_bytes

        # Broadcast this rank's eight values into slot (row, rank) of every rank's buffer.
        xw = _ld_v4(x_addr + Int64((row * HIDDEN + col) * 2))
        if cutlass.const_expr(self.norm):
            gw = _ld_v4(gamma_addr + Int64(col * 2))
        sw = [_sanitize(w) for w in xw]
        slot = Int64(((row * ws + rank) * HIDDEN + col) * 2)
        _st_v4(mc_addr + cur_off + slot, sw[0], sw[1], sw[2], sw[3])

        # Clear this rank's dirty buffer (all stages the previous launch used).
        n_threads = m * CLUSTER * THREADS
        gtid = (row * CLUSTER + cy) * THREADS + tidx
        clear = (c0, c1, c2, c3)
        for s in cutlass.range_constexpr(4):
            if s < dirty_stages:
                stage_bytes = buf_bytes // Int64(dirty_stages)
                base = uc_addr + (dirty * Int64(dirty_stages) + Int64(s)) * stage_bytes
                n_vec = (clear[s] + 15) >> 4
                for p in cutlass.range(gtid, n_vec, n_threads, unroll=1):
                    _st_sentinel_v4(base + Int64(p) * Int64(16))

        # Sum all ranks' values in chunks of RANK_CHUNK, polling until no word is the sentinel.
        acc = [Float32(0.0)] * ELTS
        row_base = uc_addr + cur_off + Int64((row * ws * HIDDEN + col) * 2)
        for rb in cutlass.range_constexpr(0, ws, self.rank_chunk):
            chunk = [Float32(0.0)] * ELTS
            ready = Int32(0)
            while ready == 0:
                ready = Int32(1)
                for i in cutlass.range_constexpr(ELTS):
                    chunk[i] = Float32(0.0)
                for rr in cutlass.range_constexpr(self.rank_chunk):
                    w = _ld_vol_v4(row_base + Int64((rb + rr) * HIDDEN * 2))
                    ready = ready & _clean4(w[0], w[1], w[2], w[3])
                    for j in cutlass.range_constexpr(4):
                        lo, hi = _bf16x2_f32(w[j])
                        chunk[2 * j] = _fadd(chunk[2 * j], lo)
                        chunk[2 * j + 1] = _fadd(chunk[2 * j + 1], hi)
            for i in cutlass.range_constexpr(ELTS):
                acc[i] = _fadd(acc[i], chunk[i])
        red = [_pack_bf16x2(acc[2 * j], acc[2 * j + 1]) for j in range(4)]

        if cutlass.const_expr(self.norm):
            # RMSNorm: this thread's BF16 squares in element order.
            thread_sum = Float32(0.0)
            for j in cutlass.range_constexpr(4):
                lo, hi = _bf16x2_f32(_sq_bf16x2(red[j]))
                thread_sum = _fadd(thread_sum, lo)
                thread_sum = _fadd(thread_sum, hi)
            # Warp butterflies; the half warp first adds 0 for its missing upper half.
            v = thread_sum
            if warp == WARPS - 1:
                v = _fadd(v, Float32(0.0))
                for off in (8, 4, 2, 1):
                    v = _fadd(v, _shfl_bfly(v, off, 0xFFFF))
            else:
                for off in (16, 8, 4, 2, 1):
                    v = _fadd(v, _shfl_bfly(v, off, 0xFFFFFFFF))
            if lane == 0:
                _sts_f32(warp_sum + warp * 4, v)
            _bar_sync()
            if warp == 0:
                b = Float32(0.0)
                if lane < WARPS:
                    b = _lds_f32(warp_sum + lane * 4)
                for off in (16, 8, 4, 2, 1):
                    b = _fadd(b, _shfl_bfly(b, off, 0xFFFFFFFF))
                if lane == 0:
                    _sts_f32(block_sum, b)
            _bar_sync()
            bsum = _lds_f32(block_sum)
            # Row sum: CTA c's sum into every peer's slot c, then the four slots in CTA order.
            if tidx < CLUSTER:
                _st_cluster_f32(cta_sum + crank * 4, tidx, bsum)
            _cluster_sync_release_acquire()
            total = Float32(0.0)
            for c in cutlass.range_constexpr(CLUSTER):
                total = _fadd(total, _lds_f32(cta_sum + c * 4))
            scale = _rms_scale(total, eps)

            ow = []
            for j in cutlass.range_constexpr(4):
                x_lo, x_hi = _bf16x2_f32(red[j])
                g_lo, g_hi = _bf16x2_f32(gw[j])
                ow.append(
                    _pack_bf16x2(_fmul(_fmul(x_lo, scale), g_lo), _fmul(_fmul(x_hi, scale), g_hi))
                )
            _st_v4(out_addr + Int64((row * HIDDEN + col) * 2), ow[0], ow[1], ow[2], ow[3])
            if cutlass.const_expr(self.mxfp8):
                # MXFP8 form of the stored row, as the low-M FP8 GEMM quantizes its BF16
                # activation: four consecutive threads hold one 1x32 group (the half warp's
                # quads use its 16-lane mask).
                amax = _amax_bf16x8(ow[0], ow[1], ow[2], ow[3])
                if warp == WARPS - 1:
                    amax = _amax_pair(amax, _shfl_bfly_b32(amax, 1, 0xFFFF))
                    amax = _amax_pair(amax, _shfl_bfly_b32(amax, 2, 0xFFFF))
                else:
                    amax = _amax_pair(amax, _shfl_bfly_b32(amax, 1, 0xFFFFFFFF))
                    amax = _amax_pair(amax, _shfl_bfly_b32(amax, 2, 0xFFFFFFFF))
                e = _ue8m0_of_amax(amax)
                q0, q1 = _quant_bf16x8_to_e4m3(ow[0], ow[1], ow[2], ow[3], _bf16x2_inv_scale(e))
                _st_v2(xq_addr + Int64(row * HIDDEN + col), q0, q1)
                if (tidx & 3) == 0:
                    _st_u8(xs_addr + Int64(row * (HIDDEN // 32) + (col >> 5)), e)
        else:
            _st_v4(out_addr + Int64((row * HIDDEN + col) * 2), red[0], red[1], red[2], red[3])

        # CTA (0, 0): rotate the buffers once every warp of the grid has read the flags.
        if crank == 0:
            if tidx == 0:
                if row == 0:
                    arrived = _ld_acquire_gpu_u32(counter)
                    while arrived < m * (CLUSTER * WARPS):
                        arrived = _ld_acquire_gpu_u32(counter)
                    nxt = f0 + 1
                    if nxt == NUM_LAMPORT_BUFFERS:
                        nxt = Int32(0)
                    _st_v4(flags_addr, nxt, f0, f2, Int32(1))
                    _st_v4(
                        flags_addr + Int64(16), m * (HIDDEN * ws * 2), Int32(0), Int32(0), Int32(0)
                    )
                    _st_u32(counter, Int32(0))


@functools.cache
def _compiled(world_size: int, use_pdl: bool, norm: bool, mxfp8: bool):
    kernel = K3LatentArNormKernel(world_size, use_pdl, norm, mxfp8)
    return cute.compile(
        kernel,
        Int64(0),
        Int64(0),
        Int64(0),
        Int64(0),
        Int64(0),
        Int64(0),
        Int64(0),
        Int64(0),
        Int32(0),
        Int32(1),
        Float32(1e-6),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


@functools.cache
def _is_supported_device(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) in ((10, 0), (10, 3), (10, 7))


def can_run(x: torch.Tensor, gamma: torch.Tensor, world_size: int) -> bool:
    """Whether :func:`run_k3_latent_ar_norm` serves this all-reduce (non-raising check)."""
    return (
        x.is_cuda
        and _is_supported_device(x.device.index)
        and world_size in WORLD_SIZES
        and x.dtype == torch.bfloat16
        and x.dim() == 2
        and x.shape[1] == HIDDEN
        and 1 <= x.shape[0] <= MAX_M
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
        and gamma.dtype == torch.bfloat16
        and tuple(gamma.shape) == (HIDDEN,)
        and gamma.is_contiguous()
        and gamma.data_ptr() % 16 == 0
    )


def run_k3_latent_ar_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    *,
    uc_addr: int,
    mc_addr: int,
    flags: torch.Tensor,
    rank: int,
    world_size: int,
    use_pdl: bool = True,
    norm: bool = True,
    mxfp8_out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Launch on the current stream: ``out`` (``[M, 3584]`` BF16) = RMSNorm(``gamma``, ``eps``) of
    the sum of every rank's ``x`` over the MNNVL Lamport workspace (local unicast base
    ``uc_addr``, multicast base ``mc_addr``, state ``flags``), bit-identical to the C++ one-shot
    kernel (``norm=False``: the BF16 sum alone). Every rank of the group launches it with its
    ``rank``. ``mxfp8_out = (xq, xs)`` also stores the MXFP8 form of ``out`` (E4M3 ``xq [M, 3584]``,
    UE8M0 ``xs [M, 112]``), bitwise the low-M FP8 GEMM's in-kernel quantization of ``out``."""
    mxfp8 = mxfp8_out is not None
    _compiled(world_size, use_pdl, norm, mxfp8)(
        Int64(x.data_ptr()),
        Int64(gamma.data_ptr()),
        Int64(out.data_ptr()),
        Int64(mxfp8_out[0].data_ptr() if mxfp8 else 0),
        Int64(mxfp8_out[1].data_ptr() if mxfp8 else 0),
        Int64(uc_addr),
        Int64(mc_addr),
        Int64(flags.data_ptr()),
        Int32(rank),
        Int32(x.shape[0]),
        Float32(eps),
        cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
    )


# ---------------------------------------------------------------------------
# One CTA per row
# ---------------------------------------------------------------------------

#: Row kernel: four groups of 128 threads per CTA; threads ``t < 112`` of group ``g`` play thread
#: ``t`` of cluster CTA ``g`` of :class:`K3LatentArNormKernel`, the others shadow lane ``t - 16`` of
#: their (half) warp and never store.
ROW_GROUPS = CLUSTER
ROW_GROUP_THREADS = 128
ROW_THREADS = ROW_GROUPS * ROW_GROUP_THREADS
ROW_WARPS = ROW_THREADS // 32
#: Largest row count of the row kernel (one CTA per row): 128 covers the helix8 batch-16
#: DSpark-5 verify step (96 rows); the kernel itself has no row limit.
ROW_MAX_M = 128
#: World sizes of the row kernel (one bit per rank of the arrival mask).
ROW_WORLD_SIZES = (2, 4, 8, 16)

assert ROW_GROUP_THREADS - THREADS == 16


@dsl_user_op
def _ld_vol_v4_pred(addr, pred, o0, o1, o2, o3, *, loc=None, ip=None):
    """16 bytes of the Lamport buffer (read past L1) if ``pred`` != 0, else ``(o0, o1, o2, o3)``
    (a predicated load: no branch)."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(pred, loc=loc, ip=ip)]
        + [_iv(v, loc=loc, ip=ip) for v in (o0, o1, o2, o3)]
        + [_lv(addr, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.b32 p, $4, 0;\nmov.b32 $0, $5;\nmov.b32 $1, $6;\n"
        "mov.b32 $2, $7;\nmov.b32 $3, $8;\n"
        "@p ld.volatile.global.v4.b32 {$0, $1, $2, $3}, [$9];\n}",
        "=&r,=&r,=&r,=&r,r,r,r,r,r,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


class K3LatentArRowKernel:
    """:class:`K3LatentArNormKernel` with one 512-thread CTA per row (same result, bit for bit).

    * Group ``g`` (threads ``128 g ...``) does what cluster CTA ``g`` does, with the same warp
      boundaries (its fourth warp is the half warp). After one CTA barrier every thread forms the
      four group sums from the sixteen warp sums as the cluster kernel's warp butterfly does,
      ``(w0 + w2) + (w1 + w3)``, and adds them in group order (no cluster barrier, no second CTA
      barrier).
    * Every round polls all ranks' slots of the thread's eight values at once. The rank's own slot
      is not read back: the sanitized values the thread stored through the multicast address are
      summed from its registers, so the last rank to arrive does not wait for its own data to
      come back from the switch. After the first round only the slots that have not arrived are
      loaded again (predicated loads).
    * The previous call's buffer is cleared after the push, like the C++ kernels.
    * CTA 0 rotates the buffers right after its first polling round when every warp of the grid
      has counted itself by then (as a rule), so that the acquire load and the state stores are
      not on the grid's tail; otherwise it waits for the count at the end, like the cluster kernel.
    * ``early_state``: the rotation state read, the warps' arrival count and the
      re-arm of the previous call's buffer (nobody polls it any more) run before
      ``griddepcontrol.wait``, while the producer of ``x`` still runs: the previous call on the
      workspace completed before that producer started (stream order), so the state is final.
      The producer may read the state (``prepushed``): the rotation still follows the wait. Same
      bits.
    """

    def __init__(
        self,
        world_size: int,
        use_pdl: bool,
        norm: bool = True,
        mxfp8: bool = False,
        sf_blocks: int = 0,
        prepushed: bool = False,
        early_state: bool = False,
    ):
        # prepushed: the producer already stored this rank's sanitized rows into slot (row, rank) of
        # every rank's current buffer (an in-kernel push); the kernel only polls and sums
        # (``x`` still supplies the own slot's values, so the result is bitwise the same).
        self.prepushed = prepushed
        self.early_state = early_state and use_pdl
        if world_size not in ROW_WORLD_SIZES:
            raise ValueError(f"unsupported world size {world_size}")
        if mxfp8 and not norm:
            raise ValueError("the MXFP8 output is the normalized output's")
        if sf_blocks and not mxfp8:
            raise ValueError("sf_blocks sets the layout of the MXFP8 output's scales")
        self.world_size = world_size
        self.use_pdl = use_pdl
        self.norm = norm
        self.mxfp8 = mxfp8
        # 0 stores the UE8M0 exponents row-major [M, 3584 / 32] (the low-M MXFP8 GEMM's
        # operand); > 0 in the mid-M GEMM's tcgen05 atoms [3584 / 128][sf_blocks][512 B]
        # (k3_fp8w_lowm_gemm.K3MxFp8QuantKernel's layout), rows M .. 128 sf_blocks - 1 neutral.
        self.sf_blocks = sf_blocks
        self.rank_chunk = min(RANK_CHUNK, world_size)

    @cute.jit
    def __call__(
        self,
        x_addr: Int64,
        gamma_addr: Int64,
        out_addr: Int64,
        xq_addr: Int64,
        xs_addr: Int64,
        uc_addr: Int64,
        mc_addr: Int64,
        flags_addr: Int64,
        rank: Int32,
        m: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            warp_sum: cute.struct.MemRange[cutlass.Float32, ROW_WARPS]

        self.shared_storage = SharedStorage
        self.kernel(
            x_addr,
            gamma_addr,
            out_addr,
            xq_addr,
            xs_addr,
            uc_addr,
            mc_addr,
            flags_addr,
            rank,
            m,
            eps,
        ).launch(
            grid=(m, 1, 1),
            block=(ROW_THREADS, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        x_addr: Int64,
        gamma_addr: Int64,
        out_addr: Int64,
        xq_addr: Int64,
        xs_addr: Int64,
        uc_addr: Int64,
        mc_addr: Int64,
        flags_addr: Int64,
        rank: Int32,
        m: Int32,
        eps: Float32,
    ):
        ws = self.world_size
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        grp = tidx // ROW_GROUP_THREADS
        t = tidx - grp * ROW_GROUP_THREADS
        active = t < THREADS
        # The group's fourth warp is the half warp (lanes 0-15 active).
        half = (warp % WARPS) == WARPS - 1
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        warp_sum = storage.warp_sum.data_ptr().toint()

        if cutlass.const_expr(self.use_pdl and not self.early_state):
            _griddepcontrol_wait()
            _griddepcontrol_launch_dependents()

        f0, f1, f2, f3 = _ld_v4(flags_addr)
        c0, c1, c2, c3 = _ld_v4(flags_addr + Int64(16))
        counter = flags_addr + Int64(32)
        cute.arch.sync_warp()
        if lane == 0:
            _red_add_release_gpu(counter, Int32(1))
        cur = Int64(f0)
        dirty = Int64(f1)
        buf_bytes = Int64(f2) & Int64(0xFFFFFFFF)
        dirty_stages = f3
        # An inactive thread loads and polls the columns of lane t - 16 of its half warp.
        tc = t
        if t >= THREADS:
            tc = t - (ROW_GROUP_THREADS - THREADS)
        col = (grp * THREADS + tc) * ELTS
        cur_off = cur * buf_bytes

        if cutlass.const_expr(self.early_state):
            # early_state: the re-arm of the previous call's buffer also before the wait, then the
            # wait.
            if cutlass.const_expr(self.norm):
                gw = _ld_v4(gamma_addr + Int64(col * 2))
            n_threads = m * ROW_THREADS
            gtid = row * ROW_THREADS + tidx
            clear = (c0, c1, c2, c3)
            for s in cutlass.range_constexpr(4):
                if s < dirty_stages:
                    stage_bytes = buf_bytes // Int64(dirty_stages)
                    base = uc_addr + (dirty * Int64(dirty_stages) + Int64(s)) * stage_bytes
                    n_vec = (clear[s] + 15) >> 4
                    for p in cutlass.range(gtid, n_vec, n_threads, unroll=1):
                        _st_sentinel_v4(base + Int64(p) * Int64(16))
            _griddepcontrol_wait()
            _griddepcontrol_launch_dependents()

        # Broadcast this rank's eight values into slot (row, rank) of every rank's buffer.
        xw = _ld_v4(x_addr + Int64((row * HIDDEN + col) * 2))
        if cutlass.const_expr(self.norm and not self.early_state):
            gw = _ld_v4(gamma_addr + Int64(col * 2))
        sw = [_sanitize(w) for w in xw]
        if cutlass.const_expr(not self.prepushed):
            if active:
                slot = Int64(((row * ws + rank) * HIDDEN + col) * 2)
                _st_v4(mc_addr + cur_off + slot, sw[0], sw[1], sw[2], sw[3])

        if cutlass.const_expr(not self.early_state):
            # Clear the previous call's buffer.
            n_threads = m * ROW_THREADS
            gtid = row * ROW_THREADS + tidx
            clear = (c0, c1, c2, c3)
            for s in cutlass.range_constexpr(4):
                if s < dirty_stages:
                    stage_bytes = buf_bytes // Int64(dirty_stages)
                    base = uc_addr + (dirty * Int64(dirty_stages) + Int64(s)) * stage_bytes
                    n_vec = (clear[s] + 15) >> 4
                    for p in cutlass.range(gtid, n_vec, n_threads, unroll=1):
                        _st_sentinel_v4(base + Int64(p) * Int64(16))

        # Poll every slot; the own slot comes from the registers. Bit r of `have`: slot r arrived.
        row_base = uc_addr + cur_off + Int64((row * ws * HIDDEN + col) * 2)
        words = cute.make_rmem_tensor((ws * 4,), Int32)
        have = Int32(0)
        for rr in cutlass.range_constexpr(ws):
            w = _ld_vol_v4(row_base + Int64(rr * HIDDEN * 2))
            for e in cutlass.range_constexpr(4):
                words[rr * 4 + e] = w[e]
            if rr == rank:
                for e in cutlass.range_constexpr(4):
                    words[rr * 4 + e] = sw[e]
            have = have | (
                _clean4(words[rr * 4], words[rr * 4 + 1], words[rr * 4 + 2], words[rr * 4 + 3])
                << rr
            )
        # CTA 0 rotates the buffers once every warp of the grid has read the flags: here, while it
        # waits for the other ranks, if they all have by the end of its first polling round (as a
        # rule), else at the end.
        rotated = Int32(0)
        if tidx == 0:
            if row == 0:
                if _ld_acquire_gpu_u32(counter) >= m * ROW_WARPS:
                    nxt0 = f0 + 1
                    if nxt0 == NUM_LAMPORT_BUFFERS:
                        nxt0 = Int32(0)
                    _st_v4(flags_addr, nxt0, f0, f2, Int32(1))
                    _st_v4(
                        flags_addr + Int64(16), m * (HIDDEN * ws * 2), Int32(0), Int32(0), Int32(0)
                    )
                    _st_u32(counter, Int32(0))
                    rotated = Int32(1)
        full = Int32((1 << ws) - 1)
        while have != full:
            missing = have
            for rr in cutlass.range_constexpr(ws):
                w = _ld_vol_v4_pred(
                    row_base + Int64(rr * HIDDEN * 2),
                    ((missing >> rr) & Int32(1)) ^ Int32(1),
                    words[rr * 4],
                    words[rr * 4 + 1],
                    words[rr * 4 + 2],
                    words[rr * 4 + 3],
                )
                for e in cutlass.range_constexpr(4):
                    words[rr * 4 + e] = w[e]
            for rr in cutlass.range_constexpr(ws):
                have = have | (
                    _clean4(words[rr * 4], words[rr * 4 + 1], words[rr * 4 + 2], words[rr * 4 + 3])
                    << rr
                )
        # Sum in chunks of RANK_CHUNK ranks (each chunk from 0, chunks in rank order).
        acc = [Float32(0.0)] * ELTS
        for rb in cutlass.range_constexpr(0, ws, self.rank_chunk):
            chunk = [Float32(0.0)] * ELTS
            for rr in cutlass.range_constexpr(self.rank_chunk):
                for j in cutlass.range_constexpr(4):
                    lo, hi = _bf16x2_f32(words[(rb + rr) * 4 + j])
                    chunk[2 * j] = _fadd(chunk[2 * j], lo)
                    chunk[2 * j + 1] = _fadd(chunk[2 * j + 1], hi)
            for i in cutlass.range_constexpr(ELTS):
                acc[i] = _fadd(acc[i], chunk[i])
        red = [_pack_bf16x2(acc[2 * j], acc[2 * j + 1]) for j in range(4)]

        if cutlass.const_expr(self.norm):
            thread_sum = Float32(0.0)
            for j in cutlass.range_constexpr(4):
                lo, hi = _bf16x2_f32(_sq_bf16x2(red[j]))
                thread_sum = _fadd(thread_sum, lo)
                thread_sum = _fadd(thread_sum, hi)
            v = thread_sum
            if half:
                if lane < 16:
                    v = _fadd(v, Float32(0.0))
                    for off in (8, 4, 2, 1):
                        v = _fadd(v, _shfl_bfly(v, off, 0xFFFF))
            else:
                for off in (16, 8, 4, 2, 1):
                    v = _fadd(v, _shfl_bfly(v, off, 0xFFFFFFFF))
            if lane == 0:
                _sts_f32(warp_sum + warp * 4, v)
            _bar_sync()
            # The cluster kernel's butterfly over a group's four warp sums (lanes past them add 0)
            # yields (w0 + w2) + (w1 + w3); every thread forms it for each group and adds the
            # group sums in order, so one CTA barrier is enough.
            total = Float32(0.0)
            for c in cutlass.range_constexpr(ROW_GROUPS):
                w4 = [_lds_f32(warp_sum + (c * WARPS + k) * 4) for k in range(WARPS)]
                total = _fadd(total, _fadd(_fadd(w4[0], w4[2]), _fadd(w4[1], w4[3])))
            scale = _rms_scale(total, eps)
            if active:
                ow = []
                for j in cutlass.range_constexpr(4):
                    x_lo, x_hi = _bf16x2_f32(red[j])
                    g_lo, g_hi = _bf16x2_f32(gw[j])
                    ow.append(
                        _pack_bf16x2(
                            _fmul(_fmul(x_lo, scale), g_lo), _fmul(_fmul(x_hi, scale), g_hi)
                        )
                    )
                _st_v4(out_addr + Int64((row * HIDDEN + col) * 2), ow[0], ow[1], ow[2], ow[3])
                if cutlass.const_expr(self.mxfp8):
                    amax = _amax_bf16x8(ow[0], ow[1], ow[2], ow[3])
                    if half:
                        amax = _amax_pair(amax, _shfl_bfly_b32(amax, 1, 0xFFFF))
                        amax = _amax_pair(amax, _shfl_bfly_b32(amax, 2, 0xFFFF))
                    else:
                        amax = _amax_pair(amax, _shfl_bfly_b32(amax, 1, 0xFFFFFFFF))
                        amax = _amax_pair(amax, _shfl_bfly_b32(amax, 2, 0xFFFFFFFF))
                    e = _ue8m0_of_amax(amax)
                    q0, q1 = _quant_bf16x8_to_e4m3(ow[0], ow[1], ow[2], ow[3], _bf16x2_inv_scale(e))
                    _st_v2(xq_addr + Int64(row * HIDDEN + col), q0, q1)
                    if cutlass.const_expr(self.sf_blocks == 0):
                        if (t & 3) == 0:
                            _st_u8(xs_addr + Int64(row * (HIDDEN // 32) + (col >> 5)), e)
                    else:
                        if (t & 3) == 0:
                            # tcgen05 scale atom of (row, 32-group sf_g), as K3MxFp8QuantKernel.
                            sf_g = col >> 5
                            sf_r = row & 127
                            sf_off = ((sf_g >> 2) * self.sf_blocks + (row >> 7)) * 512
                            sf_off = sf_off + (sf_r & 31) * 16 + (sf_r >> 5) * 4 + (sf_g & 3)
                            _st_u8(xs_addr + Int64(sf_off), e)
                            # The padding rows M .. 128 sf_blocks - 1 have no CTA: neutral 127.
                            for pad in cutlass.range(row + m, self.sf_blocks * 128, m, unroll=1):
                                pad_r = pad & 127
                                pad_off = ((sf_g >> 2) * self.sf_blocks + (pad >> 7)) * 512
                                pad_off = (
                                    pad_off + (pad_r & 31) * 16 + (pad_r >> 5) * 4 + (sf_g & 3)
                                )
                                _st_u8(xs_addr + Int64(pad_off), Int32(127))
        else:
            if active:
                _st_v4(out_addr + Int64((row * HIDDEN + col) * 2), red[0], red[1], red[2], red[3])

        # CTA 0: rotate the buffers now if it could not after its first polling round.
        if tidx == 0:
            if row == 0:
                if rotated == 0:
                    arrived = _ld_acquire_gpu_u32(counter)
                    while arrived < m * ROW_WARPS:
                        arrived = _ld_acquire_gpu_u32(counter)
                    nxt = f0 + 1
                    if nxt == NUM_LAMPORT_BUFFERS:
                        nxt = Int32(0)
                    _st_v4(flags_addr, nxt, f0, f2, Int32(1))
                    _st_v4(
                        flags_addr + Int64(16), m * (HIDDEN * ws * 2), Int32(0), Int32(0), Int32(0)
                    )
                    _st_u32(counter, Int32(0))


@functools.cache
def _compiled_row(
    world_size: int,
    use_pdl: bool,
    norm: bool,
    mxfp8: bool,
    sf_blocks: int = 0,
    prepushed: bool = False,
    early_state: bool = False,
):
    kernel = K3LatentArRowKernel(
        world_size,
        use_pdl,
        norm,
        mxfp8,
        sf_blocks,
        prepushed=prepushed,
        early_state=early_state,
    )
    a = Int64(0)
    return cute.compile(
        kernel,
        a,
        a,
        a,
        a,
        a,
        a,
        a,
        a,
        Int32(0),
        Int32(1),
        Float32(1e-6),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def can_run_row(
    x: torch.Tensor, gamma: torch.Tensor, world_size: int, buffer_bytes: int | None = None
) -> bool:
    """Whether :func:`run_k3_latent_ar_norm_row` serves this all-reduce (non-raising check); with
    ``buffer_bytes``, also that one Lamport buffer of that size holds every rank's rows."""
    if buffer_bytes is not None and x.dim() == 2 and x.numel() * world_size * 2 > buffer_bytes:
        return False
    return (
        x.is_cuda
        and _is_supported_device(x.device.index)
        and world_size in ROW_WORLD_SIZES
        and x.dtype == torch.bfloat16
        and x.dim() == 2
        and x.shape[1] == HIDDEN
        and 1 <= x.shape[0] <= ROW_MAX_M
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
        and gamma.dtype == torch.bfloat16
        and tuple(gamma.shape) == (HIDDEN,)
        and gamma.is_contiguous()
        and gamma.data_ptr() % 16 == 0
    )


def run_k3_latent_ar_norm_row(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    *,
    uc_addr: int,
    mc_addr: int,
    flags: torch.Tensor,
    rank: int,
    world_size: int,
    use_pdl: bool = True,
    norm: bool = True,
    mxfp8_out: tuple[torch.Tensor, torch.Tensor] | None = None,
    sf_blocks: int = 0,
    prepushed: bool = False,
    early_state: bool = False,
) -> None:
    """:func:`run_k3_latent_ar_norm` with :class:`K3LatentArRowKernel` (same result, bit for bit,
    for up to :data:`ROW_MAX_M` rows). Each Lamport buffer of the workspace must hold the rows of
    every rank (``M x world_size x 3584 x 2`` bytes, as the one-shot sizing of the MNNVL
    workspace guarantees for the one-shot sizes; see :func:`can_run_row`). ``sf_blocks`` > 0
    stores ``mxfp8_out``'s scales ``[(3584 / 128) * sf_blocks * 512]`` in the mid-M GEMM's
    layout (``k3_fp8w_lowm_gemm.mid_prequant_plan``) instead of row-major ``[M, 112]``.
    ``prepushed``: the producer of ``x`` already stored it into this rank's slot of every rank's
    current buffer (an in-kernel all-reduce push); the kernel skips its push.
    ``early_state``: the state read, the arrival count and the re-arm of the previous
    buffer run before the kernel's ``griddepcontrol.wait`` (see :class:`K3LatentArRowKernel`)."""
    mxfp8 = mxfp8_out is not None
    if sf_blocks and not (mxfp8 and x.shape[0] <= sf_blocks * 128):
        raise ValueError(f"sf_blocks={sf_blocks} needs the MXFP8 output and M <= {sf_blocks * 128}")
    _compiled_row(world_size, use_pdl, norm, mxfp8, sf_blocks, prepushed, early_state)(
        Int64(x.data_ptr()),
        Int64(gamma.data_ptr()),
        Int64(out.data_ptr()),
        Int64(mxfp8_out[0].data_ptr() if mxfp8 else 0),
        Int64(mxfp8_out[1].data_ptr() if mxfp8 else 0),
        Int64(uc_addr),
        Int64(mc_addr),
        Int64(flags.data_ptr()),
        Int32(rank),
        Int32(x.shape[0]),
        Float32(eps),
        cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
    )


__all__ = [
    "HIDDEN",
    "MAX_M",
    "ROW_MAX_M",
    "ROW_WORLD_SIZES",
    "WORLD_SIZES",
    "K3LatentArNormKernel",
    "K3LatentArRowKernel",
    "can_run",
    "can_run_row",
    "run_k3_latent_ar_norm",
    "run_k3_latent_ar_norm_row",
]
