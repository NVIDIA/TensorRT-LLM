# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 decode attention-residual mix + RMSNorm (+ residual add) with a PDL prologue.

Runs on SM100, SM103 and SM107: the kernel uses no tensor-core instructions, at most four named
barriers per CTA and portable cluster sizes.

For ``M`` tokens of hidden size 7168, ``N - 1`` snapshot rows ``block[n]`` and the layer row
``x``, one launch computes, per token,

* ``rsigma_n = rsqrt(mean(v_n^2) + eps)``, ``logit_n = rsigma_n * <v_n, res_w * rms_w>``,
* ``p = softmax(logit)``, ``mixed = bf16(sum_n p_n v_n)``,
* ``out = bf16(bf16(mixed * rsqrt(mean(mixed^2) + out_eps)) * out_w)``,

where ``x`` is the given row, ``bf16(prefix + addend)`` (``trtllm::attn_res_add_rmsnorm_fwd``'s
fused add), or ``bf16(residual + bf16(routed + shared))`` (a MoE layer's residual update, the
two BF16 adds of its routed and shared outputs); the updated prefix ``x`` is stored as well in
the last two cases, bit-identical to those adds.

The kernel is launched with programmatic dependent launch and does everything that does not
depend on the previous kernel before ``griddepcontrol.wait``: it loads the weights and the
snapshot rows (and the prefix sum, which is older than the previous kernel), reduces the
snapshots' statistics over the cluster and folds their share of the softmax and of the mixture
(``sum_n exp(logit_n - m) v_n``). After the wait it loads the dependent row, reduces its two
statistics (one cluster round), rescales the mixture, reduces its sum of squares (a second
round) and stores. Cluster partials travel by ``st.async`` into every peer's shared memory and
are counted by a per-round mbarrier (one one-way message per round).

A cluster of ``c`` CTAs (:func:`cluster_size`) serves one token, one 16-byte vector per thread.
The statistics are summed in another order than the C++ attention-residual kernels, so the
normalized output can differ from theirs by one or two BF16 rounding steps in a small fraction
of the elements; the updated prefix sum, an elementwise add, is bit-identical.

Snapshot rows and the prefix sum may be read before the wait because they are older than the
previous kernel: the bank rows are written by plain copy kernels at least one layer earlier, and
the prefix sum by the previous attention-residual launch, which is separated from this one by a
grid that waits for its predecessor before it triggers its dependents (an all-reduce, or a KDA /
MLA attention kernel).

The kernel releases its dependent grid right after its own ``griddepcontrol.wait``, so the next
kernel (the decode projection GEMM or the router) can become resident and stream its weights
while this one finishes. Every grid upstream has completed when a dependent starts; only this
grid may still run. The grids that can start before it completes are its dependents and,
through dependents that trigger before their own wait (the FP8 low-M GEMMs), theirs; before
their wait these read only weights, weight scales, routing metadata and recurrent-state pools
and write no global memory, and they read this kernel's outputs (normed row, updated prefix)
only after a wait. The next attention-residual launch, which reads the prefix this kernel stores
before its wait, starts only after a grid between them waited for this one to complete (see
above).
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

#: Hidden size the kernel is specialized for.
HIDDEN_SIZE = 7168
#: Candidate counts (snapshots + the layer row) the kernel is instantiated for.
MIN_CANDIDATES, MAX_CANDIDATES = 2, 12
#: Largest token count the kernel is selected for: above it the persistent C++ kernel is as fast
#: (measured at 128 tokens with eight or more candidates).
MAX_TOKENS = 64
#: Inputs of the layer row: the row itself; ``bf16(prefix + addend)`` (the C++ fused add);
#: ``bf16(residual + bf16(routed + shared))`` (the MoE residual update).
ADD_NONE, ADD_PREFIX, ADD_MOE = 0, 1, 2

_LOG2E = "0f3FB8AA3B"
_NEG_FLT_MAX = "0fFF7FFFFF"
_HIDDEN_F32 = "0f45E00000"  # 7168.0

# ---------------------------------------------------------------------------
# Inline PTX helpers (the fast-math instructions of the C++ attention-residual kernels)
# ---------------------------------------------------------------------------


def _asm(res, operands, text, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _asm_se(res, operands, text, constraints, *, loc=None, ip=None):
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


def _f32_op(text, *args, loc=None, ip=None):
    constraints = "=f," + ",".join("f" for _ in args)
    return Float32(
        _asm(T.f32(), [_fv(a, loc=loc, ip=ip) for a in args], text, constraints, loc=loc, ip=ip)
    )


@dsl_user_op
def _fma(a, b, c, *, loc=None, ip=None):
    return _f32_op("fma.rn.f32 $0, $1, $2, $3;", a, b, c, loc=loc, ip=ip)


@dsl_user_op
def _add(a, b, *, loc=None, ip=None):
    return _f32_op("add.rn.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def _add_ftz(a, b, *, loc=None, ip=None):
    return _f32_op("add.ftz.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def _mul_ftz(a, b, *, loc=None, ip=None):
    return _f32_op("mul.ftz.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def _zero_f32(*, loc=None, ip=None):
    return Float32(_asm(T.f32(), [], "mov.b32 $0, 0f00000000;", "=f", loc=loc, ip=ip))


@dsl_user_op
def _rsigma(sum_sq, eps, *, loc=None, ip=None):
    """``rsqrtf(sum_sq / 7168 + eps)`` as nvcc --use_fast_math emits it."""
    return _f32_op(
        "{\n.reg .f32 t;\n"
        f"div.approx.ftz.f32 t, $1, {_HIDDEN_F32};\n"
        "add.ftz.f32 t, t, $2;\nrsqrt.approx.ftz.f32 $0, t;\n}",
        sum_sq,
        eps,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _max_ftz(a, b, *, loc=None, ip=None):
    return _f32_op("max.ftz.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def _neg_flt_max(*, loc=None, ip=None):
    return Float32(_asm(T.f32(), [], f"mov.b32 $0, {_NEG_FLT_MAX};", "=f", loc=loc, ip=ip))


@dsl_user_op
def _rcp_ftz(a, *, loc=None, ip=None):
    return _f32_op("rcp.approx.ftz.f32 $0, $1;", a, loc=loc, ip=ip)


@dsl_user_op
def _round_bf16(a, *, loc=None, ip=None):
    """``float(bf16_rn(a))``."""
    return _f32_op(
        "{\n.reg .b16 t;\ncvt.rn.bf16.f32 t, $1;\ncvt.f32.bf16 $0, t;\n}", a, loc=loc, ip=ip
    )


@dsl_user_op
def _shfl_idx(value, lane: int, *, loc=None, ip=None):
    return Float32(
        _asm_se(
            T.f32(),
            [_fv(value, loc=loc, ip=ip)],
            f"shfl.sync.idx.b32 $0, $1, {lane}, 0x1f, 0xffffffff;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sts_f32(addr, a, *, loc=None, ip=None):
    _asm_se(
        None,
        [_iv(addr, loc=loc, ip=ip), _fv(a, loc=loc, ip=ip)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm_se(
            T.f32(), [_iv(addr, loc=loc, ip=ip)], "ld.shared.f32 $0, [$1];", "=f,r", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _bar_sync(barrier_id: int, num_threads: int, *, loc=None, ip=None):
    _asm_se(None, [], f"bar.sync {barrier_id}, {num_threads};", "", loc=loc, ip=ip)


@dsl_user_op
def _mbar_init(addr, count: int, *, loc=None, ip=None):
    _asm_se(
        None,
        [_iv(addr, loc=loc, ip=ip)],
        f"mbarrier.init.shared::cta.b64 [$0], {count};",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fence_mbar_init(*, loc=None, ip=None):
    _asm_se(None, [], "fence.mbarrier_init.release.cluster;", "", loc=loc, ip=ip)


@dsl_user_op
def _mbar_arrive_expect_tx(addr, nbytes: int, *, loc=None, ip=None):
    _asm_se(
        None,
        [_iv(addr, loc=loc, ip=ip)],
        f"mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], {nbytes};",
        "r",
        loc=loc,
        ip=ip,
    )


_WAIT_LABEL = [0]


@dsl_user_op
def _mbar_wait_cluster(addr, *, loc=None, ip=None):
    """Wait for phase 0 of a local mbarrier completed by peers' ``st.async`` (cluster acquire)."""
    _WAIT_LABEL[0] += 1
    lab = f"K3AR_WAIT{_WAIT_LABEL[0]}"
    _asm_se(
        None,
        [_iv(addr, loc=loc, ip=ip)],
        "{\n.reg .pred p;\n"
        f"{lab}:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [$0], 0;\n"
        f"@!p bra {lab};\n}}",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_async_b32(local_addr, rank, local_mbar, a, *, loc=None, ip=None):
    _asm_se(
        None,
        [
            _iv(local_addr, loc=loc, ip=ip),
            _iv(rank, loc=loc, ip=ip),
            _iv(local_mbar, loc=loc, ip=ip),
            _fv(a, loc=loc, ip=ip),
        ],
        "{\n.reg .b32 ra, rm;\nmapa.shared::cluster.u32 ra, $0, $1;\n"
        "mapa.shared::cluster.u32 rm, $2, $1;\n"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [ra], $3, [rm];\n}",
        "r,r,r,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _cluster_arrive_relaxed(*, loc=None, ip=None):
    _asm_se(None, [], "barrier.cluster.arrive.relaxed.aligned;", "", loc=loc, ip=ip)


@dsl_user_op
def _cluster_wait(*, loc=None, ip=None):
    _asm_se(None, [], "barrier.cluster.wait.acquire.aligned;", "", loc=loc, ip=ip)


@dsl_user_op
def _griddepcontrol_wait(*, loc=None, ip=None):
    _asm_se(None, [], "griddepcontrol.wait;", "", loc=loc, ip=ip)


@dsl_user_op
def _griddepcontrol_launch_dependents(*, loc=None, ip=None):
    _asm_se(None, [], "griddepcontrol.launch_dependents;", "", loc=loc, ip=ip)


@dsl_user_op
def _ldg_v4(addr, nc: bool, *, loc=None, ip=None):
    """16-byte global load of four packed bf16x2 words."""
    space = "global.nc" if nc else "global"
    res = _asm_se(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        f"ld.{space}.v4.b32 {{$0, $1, $2, $3}}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _ldg_v4_if(valid, addr, nc: bool, *, loc=None, ip=None):
    """:func:`_ldg_v4` predicated on ``valid != 0`` (zeros otherwise)."""
    space = "global.nc" if nc else "global"
    res = _asm_se(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(valid, loc=loc, ip=ip), _lv(addr, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $4, 0;\n"
        "mov.b32 $0, 0;\nmov.b32 $1, 0;\nmov.b32 $2, 0;\nmov.b32 $3, 0;\n"
        f"@p ld.{space}.v4.b32 {{$0, $1, $2, $3}}, [$5];\n}}",
        "=&r,=&r,=&r,=&r,r,l",
        loc=loc,
        ip=ip,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _stg_v4_if(valid, addr, words, *, loc=None, ip=None):
    _asm_se(
        None,
        [_iv(valid, loc=loc, ip=ip), _lv(addr, loc=loc, ip=ip)]
        + [_iv(w, loc=loc, ip=ip) for w in words],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $0, 0;\n@p st.global.v4.b32 [$1], {$2, $3, $4, $5};\n}",
        "r,l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _unpack_bf16x2(word, *, loc=None, ip=None):
    """``(float(lo), float(hi))`` of a packed bf16x2 word."""
    res = _asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [_iv(word, loc=loc, ip=ip)],
        "{\n.reg .b32 t;\nshl.b32 t, $2, 16;\nmov.b32 $0, t;\n"
        "and.b32 t, $2, 0xffff0000;\nmov.b32 $1, t;\n}",
        "=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _pack_bf16x2(lo, hi, *, loc=None, ip=None):
    """bf16x2 word ``{bf16_rn(hi), bf16_rn(lo)}``."""
    return Int32(
        _asm(
            T.i32(),
            [_fv(hi, loc=loc, ip=ip), _fv(lo, loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_xor(value, offset: int, *, loc=None, ip=None):
    return Float32(
        _asm_se(
            T.f32(),
            [_fv(value, loc=loc, ip=ip)],
            f"shfl.sync.bfly.b32 $0, $1, {offset}, 0x1f, 0xffffffff;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _selp(cond, a, b, *, loc=None, ip=None):
    """``a if cond != 0 else b``."""
    return Float32(
        _asm(
            T.f32(),
            [_iv(cond, loc=loc, ip=ip), _fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.u32 p, $1, 0;\nselp.f32 $0, $2, $3, p;\n}",
            "=f,r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ex2_scaled(x, *, loc=None, ip=None):
    """``exp(x)`` as ``ex2.approx.ftz(x * log2(e))``."""
    return _f32_op(
        f"{{\n.reg .f32 t;\nmul.ftz.f32 t, $1, {_LOG2E};\nex2.approx.ftz.f32 $0, t;\n}}",
        x,
        loc=loc,
        ip=ip,
    )


def _warp_transpose_sum(values, lane):
    """Sums every value over the warp in ``len(values) - 1 + 5 - log2(len)`` shuffles.

    ``len(values)`` must be a power of two <= 32. Afterwards lane ``l`` holds the warp total of
    ``values[l >> (5 - log2(len(values)))]``.
    """
    offset = 16
    while len(values) > 1:
        half = len(values) // 2
        upper = Int32((lane & offset) != 0)
        reduced = []
        for i in range(half):
            mine = _selp(upper, values[i + half], values[i])
            send = _selp(upper, values[i], values[i + half])
            reduced.append(_add(mine, _shfl_xor(send, offset)))
        values = reduced
        offset //= 2
    total = values[0]
    while offset >= 1:
        total = _add(total, _shfl_xor(total, offset))
        offset //= 2
    return total


def _warp_sum(value):
    for offset in (16, 8, 4, 2, 1):
        value = _add(value, _shfl_xor(value, offset))
    return value


def _warp_max(value):
    for offset in (16, 8, 4, 2, 1):
        value = _max_ftz(value, _shfl_xor(value, offset))
    return value


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class KimiK3AttnResKernel:
    """A cluster of ``c`` CTAs per token, one 16-byte vector per thread.

    Each CTA owns ``7168 / c`` contiguous elements and each thread eight of them (one 16-byte
    load per row).
    """

    def __init__(self, num_candidates: int, add_mode: int, use_pdl: bool, cluster: int):
        if not MIN_CANDIDATES <= num_candidates <= MAX_CANDIDATES:
            raise ValueError(f"num_candidates={num_candidates} not in [2, 12]")
        if add_mode not in (ADD_NONE, ADD_PREFIX, ADD_MOE) or cluster not in (1, 2, 4, 8):
            raise ValueError(f"add_mode={add_mode} cluster={cluster}")
        self.n = num_candidates
        self.k = num_candidates - 1
        self.c = cluster
        self.add_mode = add_mode
        self.use_pdl = use_pdl
        self.vecs = HIDDEN_SIZE // 8 // cluster  # 16-byte vectors per CTA, one per thread
        self.threads = (self.vecs + 31) // 32 * 32
        self.warps = self.threads // 32
        self.parts = cluster * self.warps  # warp partials per token
        assert self.parts <= 32
        # Snapshot statistics (sq, dot) per warp, padded to a power of two for the transpose sum.
        self.nvals = 2 * self.k
        self.width = 1 << (self.nvals - 1).bit_length()
        self.shift = 5 - (self.width.bit_length() - 1)

    @cute.jit
    def __call__(
        self,
        layer_addr: Int64,
        add_addr: Int64,
        add2_addr: Int64,
        block_addr: Int64,
        res_addr: Int64,
        rms_addr: Int64,
        out_w_addr: Int64,
        out_addr: Int64,
        upd_addr: Int64,
        m: Int32,
        eps: Float32,
        out_eps: Float32,
        stream: cuda.CUstream,
    ):
        parts, nvals = self.parts, self.nvals

        @cute.struct
        class SharedStorage:
            snap: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, parts * nvals], 16]
            last: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, parts * 2], 16]
            outsq: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, parts], 16]
            mbar: cute.struct.Align[cute.struct.MemRange[cutlass.Int64, 4], 16]

        self.shared_storage = SharedStorage
        kernel = self.kernel(
            layer_addr,
            add_addr,
            add2_addr,
            block_addr,
            res_addr,
            rms_addr,
            out_w_addr,
            out_addr,
            upd_addr,
            m,
            eps,
            out_eps,
        )
        if cutlass.const_expr(self.c > 1):
            kernel.launch(
                grid=(m * self.c, 1, 1),
                block=(self.threads, 1, 1),
                cluster=(self.c, 1, 1),
                smem=self.shared_storage.size_in_bytes(),
                stream=stream,
                use_pdl=self.use_pdl,
            )
        else:
            kernel.launch(
                grid=(m, 1, 1),
                block=(self.threads, 1, 1),
                smem=self.shared_storage.size_in_bytes(),
                stream=stream,
                use_pdl=self.use_pdl,
            )

    @cute.kernel
    def kernel(
        self,
        layer_addr: Int64,
        add_addr: Int64,
        add2_addr: Int64,
        block_addr: Int64,
        res_addr: Int64,
        rms_addr: Int64,
        out_w_addr: Int64,
        out_addr: Int64,
        upd_addr: Int64,
        m: Int32,
        eps: Float32,
        out_eps: Float32,
    ):
        c, k, parts, nvals = self.c, self.k, self.parts, self.nvals
        warps, shift = self.warps, self.shift
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        snap_smem = storage.snap.data_ptr().toint()
        last_smem = storage.last.data_ptr().toint()
        outsq_smem = storage.outsq.data_ptr().toint()
        mbar_smem = storage.mbar.data_ptr().toint()

        token = bidx // c
        rank = bidx - token * c
        if cutlass.const_expr(c > 1):
            if tidx == 0:
                for r in cutlass.range_constexpr(3):
                    _mbar_init(mbar_smem + r * 8, 1)
                _fence_mbar_init()
            _cluster_arrive_relaxed()

        valid = Int32(1)
        if cutlass.const_expr(self.vecs != self.threads):
            valid = Int32(tidx < self.vecs)
        col = Int64(rank * self.vecs + tidx) * 16
        row = Int64(token) * (HIDDEN_SIZE * 2)

        def load(addr, nc):
            if cutlass.const_expr(self.vecs != self.threads):
                return _ldg_v4_if(valid, addr, nc)
            return _ldg_v4(addr, nc)

        def unpack(words):
            out = []
            for w in words:
                lo, hi = _unpack_bf16x2(w)
                out.append(lo)
                out.append(hi)
            return out

        # ---- prologue: everything that does not depend on the previous grid ----
        res_w = unpack(load(res_addr + col, True))
        rms_w = unpack(load(rms_addr + col, True))
        out_w = load(out_w_addr + col, True)
        plane = Int64(m) * (HIDDEN_SIZE * 2)
        snaps = [load(block_addr + plane * n + row + col, True) for n in range(k)]
        prefix = []
        if cutlass.const_expr(self.add_mode != ADD_NONE):
            prefix = load(layer_addr + row + col, True)
        q = [_mul_ftz(res_w[e], rms_w[e]) for e in range(8)]

        snap_f = [unpack(snaps[n]) for n in range(k)]
        stats = []
        for n in cutlass.range_constexpr(k):
            sq = _zero_f32()
            dot = _zero_f32()
            for e in cutlass.range_constexpr(8):
                sq = _fma(snap_f[n][e], snap_f[n][e], sq)
                dot = _fma(snap_f[n][e], q[e], dot)
            stats.append(sq)
            stats.append(dot)
        for _ in cutlass.range_constexpr(self.width - nvals):
            stats.append(_zero_f32())
        warp_stat = _warp_transpose_sum(stats, lane)
        vidx = lane >> shift
        sub = lane & ((1 << shift) - 1)
        if cutlass.const_expr(c == 1):
            if sub == 0:
                if vidx < nvals:
                    _sts_f32(snap_smem + (warp * nvals + vidx) * 4, warp_stat)
            _bar_sync(1, self.threads)
        else:
            _cluster_wait()
            if tidx == 0:
                _mbar_arrive_expect_tx(mbar_smem, parts * nvals * 4)
                _mbar_arrive_expect_tx(mbar_smem + 8, parts * 2 * 4)
                _mbar_arrive_expect_tx(mbar_smem + 16, parts * 4)
            slot = snap_smem + ((rank * warps + warp) * nvals + vidx) * 4
            for r in cutlass.range_constexpr(c):
                if vidx < nvals:
                    if sub == r % (1 << shift):
                        _st_async_b32(slot, r, mbar_smem, warp_stat)
            _mbar_wait_cluster(mbar_smem)

        # Snapshot totals (lane n < k), logits and the snapshot part of the softmax.
        logit = _neg_flt_max()
        if lane < k:
            tot_sq = _zero_f32()
            tot_dot = _zero_f32()
            for j in cutlass.range_constexpr(parts):
                tot_sq = _add(tot_sq, _lds_f32(snap_smem + (j * nvals + 2 * lane) * 4))
                tot_dot = _add(tot_dot, _lds_f32(snap_smem + (j * nvals + 2 * lane + 1) * 4))
            logit = _mul_ftz(tot_dot, _rsigma(tot_sq, eps))
        snap_max = _warp_max(logit)
        weight = _zero_f32()
        if lane < k:
            weight = _ex2_scaled(_add(logit, -snap_max))
        snap_sum = _warp_sum(weight)
        premix = [_zero_f32() for _ in range(8)]
        for n in cutlass.range_constexpr(k):
            w_n = _shfl_idx(weight, n)
            for e in cutlass.range_constexpr(8):
                premix[e] = _fma(w_n, snap_f[n][e], premix[e])

        # ---- wait for the previous grid, then release the dependent grid ----
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
            _griddepcontrol_launch_dependents()

        if cutlass.const_expr(self.add_mode == ADD_PREFIX):
            addend = unpack(load(add_addr + row + col, False))
            pre = unpack(prefix)
            x = [_round_bf16(_add_ftz(pre[e], addend[e])) for e in range(8)]
        elif cutlass.const_expr(self.add_mode == ADD_MOE):
            routed = unpack(load(add_addr + row + col, False))
            shared = unpack(load(add2_addr + row + col, False))
            pre = unpack(prefix)
            x = [
                _round_bf16(_add(pre[e], _round_bf16(_add(routed[e], shared[e])))) for e in range(8)
            ]
        else:
            x = unpack(load(layer_addr + row + col, False))
        if cutlass.const_expr(self.add_mode != ADD_NONE):
            words = [_pack_bf16x2(x[2 * i], x[2 * i + 1]) for i in range(4)]
            _stg_v4_if(valid, upd_addr + row + col, words)

        sq = _zero_f32()
        dot = _zero_f32()
        for e in cutlass.range_constexpr(8):
            sq = _fma(x[e], x[e], sq)
            dot = _fma(x[e], q[e], dot)
        # Lanes 0-15 hold the warp's sum of squares, lanes 16-31 its dot product.
        pair = _warp_transpose_sum([sq, dot], lane)
        half = lane & 15
        if cutlass.const_expr(c == 1):
            if half == 0:
                _sts_f32(last_smem + (warp * 2 + (lane >> 4)) * 4, pair)
            _bar_sync(2, self.threads)
        else:
            if half < c:
                _st_async_b32(
                    last_smem + ((rank * warps + warp) * 2 + (lane >> 4)) * 4,
                    half,
                    mbar_smem + 8,
                    pair,
                )
            _mbar_wait_cluster(mbar_smem + 8)
        part_sq = _zero_f32()
        part_dot = _zero_f32()
        if lane < parts:
            part_sq = _lds_f32(last_smem + lane * 8)
            part_dot = _lds_f32(last_smem + lane * 8 + 4)
        tot = _warp_transpose_sum([part_sq, part_dot], lane)
        tot_sq = _shfl_idx(tot, 0)
        tot_dot = _shfl_idx(tot, 16)
        last_logit = _mul_ftz(tot_dot, _rsigma(tot_sq, eps))

        # Softmax over all candidates: p_n = exp(logit_n - m_s) * a / den, p_last = b / den.
        new_max = _max_ftz(snap_max, last_logit)
        a = _ex2_scaled(_add(snap_max, -new_max))
        b = _ex2_scaled(_add(last_logit, -new_max))
        inv = _rcp_ftz(_fma(snap_sum, a, b))
        scale = _mul_ftz(a, inv)
        p_last = _mul_ftz(b, inv)
        mixed = [_round_bf16(_fma(p_last, x[e], _mul_ftz(scale, premix[e]))) for e in range(8)]
        out_sq = _zero_f32()
        for e in cutlass.range_constexpr(8):
            out_sq = _fma(mixed[e], mixed[e], out_sq)
        out_sq = _warp_sum(out_sq)
        if cutlass.const_expr(c == 1):
            if lane == 0:
                _sts_f32(outsq_smem + warp * 4, out_sq)
            _bar_sync(3, self.threads)
        else:
            if lane < c:
                _st_async_b32(outsq_smem + (rank * warps + warp) * 4, lane, mbar_smem + 16, out_sq)
            _mbar_wait_cluster(mbar_smem + 16)
        part = _zero_f32()
        if lane < parts:
            part = _lds_f32(outsq_smem + lane * 4)
        out_rs = _rsigma(_warp_sum(part), out_eps)

        w_out = unpack(out_w)
        y = [_mul_ftz(_round_bf16(_mul_ftz(mixed[e], out_rs)), w_out[e]) for e in range(8)]
        words = [_pack_bf16x2(y[2 * i], y[2 * i + 1]) for i in range(4)]
        _stg_v4_if(valid, out_addr + row + col, words)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def cluster_size(num_tokens: int, num_candidates: int) -> int:
    """CTAs per token.

    Measured on GB300 behind a PDL producer: eight CTAs per token are fastest for a handful of
    tokens with at least three snapshots (smaller per-CTA prologue), four from ~32 tokens on
    (eight-CTA clusters then fill the machine) and for two or three candidates. On SM107 (212
    SMs) this choice is within 0.25 us of the fastest cluster size for up to 64 tokens.
    """
    return 8 if num_tokens <= 16 and num_candidates >= 4 else 4


@functools.cache
def _compiled(num_candidates: int, add_mode: int, use_pdl: bool, cluster: int):
    kernel = KimiK3AttnResKernel(num_candidates, add_mode, use_pdl, cluster)
    addr = Int64(1 << 20)
    return cute.compile(
        kernel,
        *([addr] * 9),
        Int32(1),
        Float32(1e-6),
        Float32(1e-6),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


@functools.cache
def _is_supported_device(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) in ((10, 0), (10, 3), (10, 7))


def can_run(
    layer: torch.Tensor,
    addends: tuple[torch.Tensor, ...],
    block: torch.Tensor,
    *weights: torch.Tensor,
) -> bool:
    """Whether :func:`run_attn_res` serves ``layer [M, H]``, 0-2 addends ``[M, H]``,
    ``block [N - 1, M, H]`` and the ``[H]`` weights: BF16, contiguous, ``H == 7168``,
    ``1 <= M <= MAX_TOKENS``, ``2 <= N <= 12``, on SM100 / SM103 / SM107."""
    if not layer.is_cuda or not _is_supported_device(layer.device.index or 0):
        return False
    if layer.dim() != 2 or layer.shape[1] != HIDDEN_SIZE or not 1 <= layer.shape[0] <= MAX_TOKENS:
        return False
    m = layer.shape[0]
    if block.dim() != 3 or tuple(block.shape[1:]) != (m, HIDDEN_SIZE):
        return False
    if not MIN_CANDIDATES <= block.shape[0] + 1 <= MAX_CANDIDATES or len(addends) > 2:
        return False
    for t in (layer, block, *weights, *addends):
        if t.dtype != torch.bfloat16 or not t.is_contiguous() or t.device != layer.device:
            return False
    if any(a.shape != layer.shape for a in addends):
        return False
    return all(w.numel() == HIDDEN_SIZE for w in weights)


def run_attn_res(
    layer: torch.Tensor,
    addends: tuple[torch.Tensor, ...],
    block: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    out_weight: torch.Tensor,
    rms_eps: float,
    out_eps: float,
    output: torch.Tensor,
    updated: torch.Tensor | None,
    *,
    use_pdl: bool = True,
    cluster: int | None = None,
) -> None:
    """Launch on the current stream (inputs are not re-validated; see :func:`can_run`).

    Args:
        layer: ``bf16 [M, 7168]``: the layer row without addends, else the prefix sum.
        addends: ``()``; ``(addend,)`` for ``bf16(layer + addend)``; ``(routed, shared)`` for
            ``bf16(layer + bf16(routed + shared))`` (all ``bf16 [M, 7168]``).
        block: ``bf16 [N - 1, M, 7168]`` snapshot rows, ``2 <= N <= 12``.
        res_weight, rms_weight: Score projection and its RMSNorm weight, ``bf16 [7168]``.
        out_weight: Weight of the trailing RMSNorm, ``bf16 [7168]``.
        rms_eps, out_eps: Epsilons of the two norms.
        output: ``bf16 [M, 7168]`` normalized mixture.
        updated: ``bf16 [M, 7168]`` updated prefix sum; required with addends.
        use_pdl: Launch with programmatic dependent launch.
        cluster: CTAs per token (1, 2, 4 or 8); :func:`cluster_size` if None.
    """
    add_mode = len(addends)
    if cluster is None:
        cluster = cluster_size(layer.shape[0], block.shape[0] + 1)
    compiled = _compiled(block.shape[0] + 1, add_mode, use_pdl, cluster)
    compiled(
        Int64(layer.data_ptr()),
        Int64(addends[0].data_ptr() if add_mode >= 1 else 0),
        Int64(addends[1].data_ptr() if add_mode == 2 else 0),
        Int64(block.data_ptr()),
        Int64(res_weight.data_ptr()),
        Int64(rms_weight.data_ptr()),
        Int64(out_weight.data_ptr()),
        Int64(output.data_ptr()),
        Int64(updated.data_ptr() if add_mode else 0),
        Int32(layer.shape[0]),
        Float32(rms_eps),
        Float32(out_eps),
        cuda.CUstream(torch.cuda.current_stream(layer.device).cuda_stream),
    )


__all__ = [
    "ADD_MOE",
    "ADD_NONE",
    "ADD_PREFIX",
    "HIDDEN_SIZE",
    "MAX_CANDIDATES",
    "MAX_TOKENS",
    "MIN_CANDIDATES",
    "KimiK3AttnResKernel",
    "can_run",
    "cluster_size",
    "run_attn_res",
]
