# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused KDA prefill short convolution + SiLU + indexed state update + Q/K/V layout.

One CuTe DSL kernel. The packed QKV source is token-major and the requested
outputs are token-major [T, H*128] with the same channel order as each source
section, so no transpose is needed: each thread owns 8 consecutive channels
(one 16-byte vector) and walks a run of tokens, carrying the 3-token causal
history in registers.

Alignment: vector source loads require a 16-byte-aligned base and a row
stride divisible by 8 BF16 elements. Misaligned sources use elementwise
loads of channels [cc, cc+8), including history and final-state loads.
Rounding a misaligned pointer down can read uninitialized storage before
the tensor view or in row padding, even when it stays inside the allocation.

SiLU: the accumulator is h = x/2 (pre-halved weights, exact). The fast form
silu(x) = h + h*tanh.approx(h) has relative error <= 2^-11 * |tanh(h)| / (1 + tanh(h)),
which is <= 1.3% for h >= -2 but grows without bound as h -> -inf (cancellation).
Each lane keeps the minimum h of the current 8-token group; one warp vote per group
flags groups where some h < -2, and flagged groups are recomputed after the loop with
x*e/(1+e), e = exp(-|x|) (ex2/rcp, subnormal-preserving), stored over the fast results
by the same thread in program order.

cp.async ring (aligned sources): per lane, the prologue packs ASYNC_GROUP adjacent
tokens into each commit group (empty copies are omitted). Each token pack waits until
at most D/ASYNC_GROUP-1 groups remain, consumes its distinct slots, refills those slots
for tokens i+D..., then commits the refills as one group. Thus pack p retires group p,
including every token it consumes. A lane refills a slot only after its own lds_v4 of
that slot in program order; partial tails still commit one final group.

Launch: one-warp CTAs on grid (channel blocks, runs, sequences). Every CTA
grid-strides over the runs of its sequence with the run count taken from the
device-side length (starts), so max_seqlen only sizes grid.y (parallelism), never
which tokens are processed: an understated or overstated bound still covers every
token. grid.y is also capped at 65535. Only when B > 65535 does a shape-selected
specialization grid-stride over sequences as well. A max_seqlen that is a CUDA tensor
is never read on the host (T is used as the bound).

State pool: run 0 of each sequence (one warp per 256 channels) owns the slot's rows for
its channels: it reads the old history, then writes the final three raw inputs after
its own loop (same thread, program order). With a dense [C, 3] slot layout, 16-byte
aligned base and slot stride % 8 == 0, each lane moves its 8 channels x 3 taps as
3 x 16-byte loads/stores; otherwise elementwise.

Slots: -1 skips the state pool (raw copy); -pool <= slot < -1 index the pool from the
end, as in the reference's Python indexing.

Caller contract (stated by the problem: "Slot -1 bypasses ... Other slots are unique and
valid"; not checked on device): slots other than -1 are unique after wrapping and lie in
[-pool, pool). Duplicate slots are outside that contract: the reference's sequential
semantics would need a cross-sequence ordering of state reads and writes that one
launch does not provide.
"""

from collections.abc import Callable
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_tensor
from cutlass.cutlass_dsl import T, dsl_user_op

WARPS = 4
THREADS = WARPS * 32
VEC = 8  # channels per lane
WARP_CH = 32 * VEC  # channels per warp
WIDTH = 4
ASYNC_GROUP = 4  # adjacent token copies per cp.async commit/wait group
SLOT = 32 * 16  # bytes per ring slot: one 16-byte granule per lane
# Fast SiLU only while every accumulator h = x/2 of the warp step is >= this.
FAST_MIN_H = -2.0
H_INIT = 3.0e38  # running-min start value
NEG_2LOG2E = -2.8853900817779268  # -2*log2(e)


def _asm(res, args, text, cons, side=False, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        args,
        text,
        cons,
        has_side_effects=side,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_v4(addr, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return [cutlass.Int32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def ld_v4_first(addr, *, loc=None, ip=None):
    """ld.global.nc.v4 kept in program order with the cp.async issue (volatile asm), so the
    halo rows are requested before the ring prologue."""
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        side=True,
        loc=loc,
        ip=ip,
    )
    return [cutlass.Int32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def st_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [
            cutlass.Int64(addr).ir_value(loc=loc, ip=ip),
            cutlass.Int32(a).ir_value(loc=loc, ip=ip),
            cutlass.Int32(b).ir_value(loc=loc, ip=ip),
            cutlass.Int32(c).ir_value(loc=loc, ip=ip),
            cutlass.Int32(d).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.b32 [$1], {$2, $3, $4, $5};\n\tmov.b32 $0, 0;",
        "=r,l,r,r,r,r",
        side=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_v4_rw(addr, *, loc=None, ip=None):
    """Coherent 16-byte load (for the state pool, which this kernel also writes)."""
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [cutlass.Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        side=True,
        loc=loc,
        ip=ip,
    )
    return [cutlass.Int32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def prmt(a, b, sel: int, *, loc=None, ip=None):
    """Byte permute of the pair (a: bytes 0-3, b: bytes 4-7) with an immediate selector."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [cutlass.Int32(a).ir_value(loc=loc, ip=ip), cutlass.Int32(b).ir_value(loc=loc, ip=ip)],
            f"prmt.b32 $0, $1, $2, {sel:#x};",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def unpack_bf16x2(w, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [cutlass.Int32(w).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 t;\n\tshl.b32 t, $2, 16;\n\tmov.b32 $0, t;\n\t"
        "and.b32 t, $2, 0xFFFF0000;\n\tmov.b32 $1, t;\n\t}",
        "=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return (
        cutlass.Float32(llvm.extractvalue(T.f32(), out, [0], loc=loc, ip=ip)),
        cutlass.Float32(llvm.extractvalue(T.f32(), out, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def pack_bf16x2(lo, hi, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            T.i32(),
            [
                cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
                cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def tanh_approx(x, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fma_rn(a, b, c, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [
                cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                cutlass.Float32(b).ir_value(loc=loc, ip=ip),
                cutlass.Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cp_async16(saddr, gaddr, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [
            cutlass.Int32(saddr).ir_value(loc=loc, ip=ip),
            cutlass.Int64(gaddr).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global [$1], [$2], 16;\n\tmov.b32 $0, 0;",
        "=r,r,l",
        side=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def lds_v4(saddr, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [cutlass.Int32(saddr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        side=True,
        loc=loc,
        ip=ip,
    )
    return [cutlass.Int32(llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def smem_u32(ptr, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(
            T.i32(),
            [cutlass.Int64(ptr).ir_value(loc=loc, ip=ip)],
            "cvta.to.shared.u32 $0, $1;",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bf_lo(w, *, loc=None, ip=None):
    """FP32 value of the low BF16 half of a packed word."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Int32(w).ir_value(loc=loc, ip=ip)],
            "shl.b32 $0, $1, 16;",
            "=f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bf_hi(w, *, loc=None, ip=None):
    """FP32 value of the high BF16 half of a packed word."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Int32(w).ir_value(loc=loc, ip=ip)],
            "and.b32 $0, $1, 0xFFFF0000;",
            "=f,r",
            loc=loc,
            ip=ip,
        )
    )


def _pair_op(text, n_in, loc, ip, args):
    regs = ", ".join(f"p{i}" for i in range(n_in))
    body = "{\n\t.reg .b64 " + regs + ", d;\n\t"
    for i in range(n_in):
        body += f"mov.b64 p{i}, {{${2 + 2 * i}, ${3 + 2 * i}}};\n\t"
    body += text.format(*(f"p{i}" for i in range(n_in))) + "\n\tmov.b64 {$0, $1}, d;\n\t}"
    out = _asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [cutlass.Float32(a).ir_value(loc=loc, ip=ip) for a in args],
        body,
        "=f,=f," + ",".join(["f"] * (2 * n_in)),
        loc=loc,
        ip=ip,
    )
    return (
        cutlass.Float32(llvm.extractvalue(T.f32(), out, [0], loc=loc, ip=ip)),
        cutlass.Float32(llvm.extractvalue(T.f32(), out, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def mul2(a0, a1, b0, b1, *, loc=None, ip=None):
    return _pair_op("mul.rn.f32x2 d, {0}, {1};", 2, loc, ip, [a0, a1, b0, b1])


@dsl_user_op
def fma2(a0, a1, b0, b1, c0, c1, *, loc=None, ip=None):
    return _pair_op("fma.rn.f32x2 d, {0}, {1}, {2};", 3, loc, ip, [a0, a1, b0, b1, c0, c1])


@dsl_user_op
def fmin3(a, b, c, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Float32(x).ir_value(loc=loc, ip=ip) for x in (a, b, c)],
            "min.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def silu2_exact(h0, h1, *, loc=None, ip=None):
    """For h = x/2, silu(x) on a pair: e = exp(-|x|) in (0, 1], so 1 + e in (1, 2].

    x >= 0: x / (1 + e); x < 0: (x * e) / (1 + e). The reciprocal never sees a
    large or subnormal argument and e keeps subnormals (no FTZ on ex2), so deep
    negative outputs keep their relative accuracy.
    """
    out = _asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [cutlass.Float32(a).ir_value(loc=loc, ip=ip) for a in (h0, h1, NEG_2LOG2E)],
        "{\n\t.reg .f32 a<2>, e<2>, r<2>, x<2>, n<2>;\n\t.reg .pred p<2>;\n\t"
        "abs.f32 a0, $2;\n\tabs.f32 a1, $3;\n\t"
        "mul.rn.f32 a0, a0, $4;\n\tmul.rn.f32 a1, a1, $4;\n\t"
        "ex2.approx.f32 e0, a0;\n\tex2.approx.f32 e1, a1;\n\t"
        "add.rn.f32 r0, e0, 0f3F800000;\n\tadd.rn.f32 r1, e1, 0f3F800000;\n\t"
        "rcp.approx.ftz.f32 r0, r0;\n\trcp.approx.ftz.f32 r1, r1;\n\t"
        "add.rn.f32 x0, $2, $2;\n\tadd.rn.f32 x1, $3, $3;\n\t"
        "mul.rn.f32 n0, x0, e0;\n\tmul.rn.f32 n1, x1, e1;\n\t"
        "setp.lt.f32 p0, $2, 0f00000000;\n\tsetp.lt.f32 p1, $3, 0f00000000;\n\t"
        "selp.f32 n0, n0, x0, p0;\n\tselp.f32 n1, n1, x1, p1;\n\t"
        "mul.rn.f32 $0, n0, r0;\n\tmul.rn.f32 $1, n1, r1;\n\t}",
        "=f,=f,f,f,f",
        loc=loc,
        ip=ip,
    )
    return (
        cutlass.Float32(llvm.extractvalue(T.f32(), out, [0], loc=loc, ip=ip)),
        cutlass.Float32(llvm.extractvalue(T.f32(), out, [1], loc=loc, ip=ip)),
    )


class KdaConv:
    def __init__(
        self,
        heads,
        tokens_per_warp,
        unroll,
        src_vec,
        w_vec,
        out_vec,
        row_stride,
        state_stride,
        seq_loop,
        depth=8,
        warps=4,
        st_vec=False,
    ):
        self.W = warps
        self.MB = FORCE_MB
        self.heads = heads
        self.P = heads * 128
        self.C = 3 * heads * 128
        self.L = tokens_per_warp
        self.U = unroll
        self.src_vec = src_vec
        self.w_vec = w_vec
        self.out_vec = out_vec
        self.SR = row_stride
        self.SS = state_stride
        self.seq_loop = seq_loop
        self.st_vec = st_vec
        self.D = depth
        self.U = 8
        self.RUN = warps * tokens_per_warp  # tokens per CTA (power of two)
        self.RUN_LOG2 = self.RUN.bit_length() - 1
        assert self.RUN == 1 << self.RUN_LOG2
        assert self.D % ASYNC_GROUP == 0 and self.U % ASYNC_GROUP == 0

    @cute.jit
    def loadw(self, src, src_base, row, cc, xw: cute.Tensor, off: cutlass.Constexpr):
        """Load source channels [cc, cc+8) of `row` as 4 packed BF16x2 words into xw[off:off+4]."""
        if cutlass.const_expr(self.src_vec):
            w = ld_v4(src_base + (cutlass.Int64(row) * self.SR + cc) * 2)
            for j in cutlass.range_constexpr(4):
                xw[off + j] = w[j]
        else:
            for j in cutlass.range_constexpr(4):
                xw[off + j] = pack_bf16x2(
                    cutlass.Float32(src[cutlass.Int64(row), cc + 2 * j]),
                    cutlass.Float32(src[cutlass.Int64(row), cc + 2 * j + 1]),
                )

    @cute.jit
    def load8(self, src, src_base, row, cc, xs: cute.Tensor, off: cutlass.Constexpr):
        """Load source channels [cc, cc+8) of `row` as FP32 into xs[off:off+8]."""
        xw = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Int32)
        self.loadw(src, src_base, row, cc, xw, 0)
        for j in cutlass.range_constexpr(4):
            xlo, xhi = unpack_bf16x2(xw[j])
            xs[off + 2 * j] = xlo
            xs[off + 2 * j + 1] = xhi

    @cute.jit
    def issue(self, src_base, row, cc, sl):
        """cp.async the 16-byte granule holding channels [cc, cc+8) of `row` into sl."""
        cp_async16(sl, src_base + (cutlass.Int64(row) * self.SR + cc) * 2)

    @cute.jit
    def storew(self, q, k, v, sec, row, inner, yw: cute.Tensor):
        """Elementwise store of 8 channels for non-contiguous or misaligned q/k/v."""
        h = inner // 128
        d = inner - h * 128
        row64 = cutlass.Int64(row)
        for j in cutlass.range_constexpr(4):
            olo, ohi = unpack_bf16x2(yw[j])
            for e in cutlass.range_constexpr(2):
                val = cutlass.BFloat16(olo)
                if cutlass.const_expr(e == 1):
                    val = cutlass.BFloat16(ohi)
                if sec == 0:
                    q[0, row64, h, d + 2 * j + e] = val
                elif sec == 1:
                    k[0, row64, h, d + 2 * j + e] = val
                else:
                    v[0, row64, h, d + 2 * j + e] = val

    @cute.kernel
    def kernel(self, src, weight, state, starts, slots, has_init, q, k, v):
        tid, _, _ = cute.arch.thread_idx()
        b0, b1, b2 = cute.arch.block_idx()
        _, gdy, gdz = cute.arch.grid_dim()
        smem = cutlass.utils.SmemAllocator()
        ring = smem.allocate_tensor(cutlass.Int8, cute.make_layout(self.W * self.D * SLOT), 16)
        ring_warp = cutlass.Int32(ring.iterator.toint()) + (tid // 32) * (self.D * SLOT)
        if cutlass.const_expr(self.seq_loop):
            # B > 65535 (grid.z limit): grid-stride over sequences too.
            for seq in cutlass.range(b2, slots.shape[0], gdz, unroll=1):
                self.runs(
                    seq,
                    b0,
                    b1,
                    gdy,
                    tid,
                    ring_warp,
                    src,
                    weight,
                    state,
                    starts,
                    slots,
                    has_init,
                    q,
                    k,
                    v,
                )
        else:
            self.runs(
                b2,
                b0,
                b1,
                gdy,
                tid,
                ring_warp,
                src,
                weight,
                state,
                starts,
                slots,
                has_init,
                q,
                k,
                v,
            )

    @cute.jit
    def runs(
        self, seq, b0, b1, gdy, tid, ring_warp, src, weight, state, starts, slots, has_init, q, k, v
    ):
        """All runs b1, b1 + gdy, ... of sequence `seq` for channel block b0.

        Run b1 is straight-line (tile() skips it when it starts past the sequence end,
        except run 0, which owns the state update, also for empty sequences). The cold
        loop covers runs past grid.y, using the run count from the device-side length:
        it is empty whenever max_seqlen bounds every length, and otherwise (understated
        bound, or the 65535 cap) it still processes every token. Each tile drains its
        ring before returning.
        """
        begin, length, slot, init = self.meta(seq, state, starts, slots, has_init)
        # Only this bit stays live across the hot tile; the cold path reloads the metadata.
        more = b1 + gdy < ((length + (self.RUN - 1)) >> self.RUN_LOG2)
        self.tile(b1, b0, tid, ring_warp, begin, length, slot, init, src, weight, state, q, k, v)
        if more:
            begin, length, slot, init = self.meta(seq, state, starts, slots, has_init)
            nrun = (length + (self.RUN - 1)) >> self.RUN_LOG2
            for bx in cutlass.range(b1 + gdy, nrun, gdy, unroll=1):
                self.tile(
                    bx, b0, tid, ring_warp, begin, length, slot, init, src, weight, state, q, k, v
                )

    @cute.jit
    def meta(self, seq, state, starts, slots, has_init):
        begin = starts[seq]
        length = starts[seq + 1] - begin
        slot = slots[seq]
        # Issued with starts/slots so the old-state load is one round trip from launch.
        init = has_init[seq]
        # Python indexing, as in the reference: -pool <= slot < -1 counts from the end.
        if slot < -1:
            slot = slot + state.shape[0]
        return begin, length, slot, init

    @cute.jit
    def tile(self, bx, by, tid, ring_warp, begin, length, slot, init, src, weight, state, q, k, v):
        """One warp: run t0 = (bx * W + warp) * L of one sequence, channel block `by`."""
        C, P, L, U = self.C, self.P, self.L, self.U
        lane = tid % 32
        warp = tid // 32
        state_slot = cute.make_tensor(
            state.iterator + cutlass.Int64(slot) * self.SS[0],
            cute.make_layout((C, WIDTH - 1), stride=(self.SS[1], self.SS[2])),
        )
        t0 = (bx * self.W + warp) * L
        c = by * WARP_CH + lane * VEC
        if cutlass.const_expr(C % WARP_CH == 0):
            cvalid = True
            cc = c
        else:
            cvalid = c < C
            cc = cutlass.min(c, C - VEC)
        sec = cc // P
        inner = cc - sec * P
        src_base = src.iterator.toint()
        ob = q.iterator.toint()
        if sec == 1:
            ob = k.iterator.toint()
        if sec == 2:
            ob = v.iterator.toint()

        D = self.D

        if t0 < length or t0 == 0:
            count = cutlass.max(cutlass.min(length - t0, L), 0)
            row0 = begin + t0
            ring_lane = ring_warp + lane * 16
            # Halo rows t0-3..t0-1 (needed by token 0) are requested before the ring prologue.
            hw = cute.make_rmem_tensor(cute.make_layout(4 * (WIDTH - 1)), cutlass.Int32)
            if cutlass.const_expr(self.src_vec):
                if t0 != 0:
                    for j in cutlass.range_constexpr(WIDTH - 1):
                        w = ld_v4_first(
                            src_base + (cutlass.Int64(row0 - (WIDTH - 1) + j) * self.SR + cc) * 2
                        )
                        for e in cutlass.range_constexpr(4):
                            hw[4 * j + e] = w[e]
            if cutlass.const_expr(self.src_vec):
                for pg in cutlass.range_constexpr(D // ASYNC_GROUP):
                    for pe in cutlass.range_constexpr(ASYNC_GROUP):
                        d = ASYNC_GROUP * pg + pe
                        if d < count:
                            self.issue(src_base, row0 + d, cc, ring_lane + d * SLOT)
                    cute.arch.cp_async_commit_group()
            wt = cute.make_rmem_tensor(cute.make_layout(WIDTH * VEC), cutlass.Float32)
            hist = cute.make_rmem_tensor(cute.make_layout((WIDTH - 1) * VEC), cutlass.Float32)
            xs = cute.make_rmem_tensor(cute.make_layout(VEC), cutlass.Float32)
            fin = cute.make_rmem_tensor(cute.make_layout(4 * (WIDTH - 1)), cutlass.Int32)
            # Weights: weight[cc + i, w] -> wt[w * 8 + i]
            if cutlass.const_expr(self.w_vec):
                wbase = weight.iterator.toint() + cutlass.Int64(cc) * (WIDTH * 2)
                for j in cutlass.range_constexpr(4):
                    ww = ld_v4(wbase + 16 * j)
                    for jj in cutlass.range_constexpr(4):
                        ch = 2 * j + jj // 2
                        tap = 2 * (jj % 2)
                        wlo, whi = unpack_bf16x2(ww[jj])
                        wt[tap * VEC + ch] = wlo * 0.5
                        wt[(tap + 1) * VEC + ch] = whi * 0.5
            else:
                for i in cutlass.range_constexpr(VEC):
                    for w in cutlass.range_constexpr(WIDTH):
                        wt[w * VEC + i] = cutlass.Float32(weight[cc + i, w]) * 0.5

            # Causal history for token t0: tokens t0-3 .. t0-1.
            if t0 == 0:
                for j in cutlass.range_constexpr((WIDTH - 1) * VEC):
                    hist[j] = cutlass.Float32(0)
                if slot >= 0:
                    if init:
                        if cutlass.const_expr(self.st_vec):
                            # The lane's 8 channels x 3 taps are 48 contiguous, 16-byte-aligned
                            # bytes: element e = 3 * i + j is half e % 2 of word e // 2.
                            sw = []
                            for m in cutlass.range_constexpr(3):
                                sw.extend(ld_v4_rw(self.state_addr(state, slot, cc) + 16 * m))
                            for i in cutlass.range_constexpr(VEC):
                                for j in cutlass.range_constexpr(WIDTH - 1):
                                    e = 3 * i + j
                                    if cutlass.const_expr(e % 2 == 0):
                                        hist[j * VEC + i] = bf_lo(sw[e // 2])
                                    else:
                                        hist[j * VEC + i] = bf_hi(sw[e // 2])
                        else:
                            for i in cutlass.range_constexpr(VEC):
                                for j in cutlass.range_constexpr(WIDTH - 1):
                                    hist[j * VEC + i] = cutlass.Float32(state_slot[cc + i, j])
                    # New state = last three raw inputs, older ones from the
                    # initial history. Gathered (packed BF16x2) now so the loads
                    # overlap the main loop; written after the loop.
                    for j in cutlass.range_constexpr(WIDTH - 1):
                        rel = length - (WIDTH - 1) + j
                        if rel >= 0:
                            self.loadw(src, src_base, begin + rel, cc, fin, 4 * j)
                        else:
                            hidx = rel + (WIDTH - 1)
                            for i in cutlass.range_constexpr(VEC):
                                val = hist[i]
                                if hidx == 1:
                                    val = hist[VEC + i]
                                if hidx == 2:
                                    val = hist[2 * VEC + i]
                                xs[i] = val
                            for w in cutlass.range_constexpr(4):
                                fin[4 * j + w] = pack_bf16x2(xs[2 * w], xs[2 * w + 1])
            else:
                if cutlass.const_expr(self.src_vec):
                    for j in cutlass.range_constexpr(WIDTH - 1):
                        for e in cutlass.range_constexpr(4):
                            xlo, xhi = unpack_bf16x2(hw[4 * j + e])
                            hist[j * VEC + 2 * e] = xlo
                            hist[j * VEC + 2 * e + 1] = xhi
                else:
                    for j in cutlass.range_constexpr(WIDTH - 1):
                        self.load8(src, src_base, begin + t0 - (WIDTH - 1) + j, cc, hist, j * VEC)

            yw = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Int32)
            gbase = src_base + (cutlass.Int64(row0) * self.SR + cc) * 2
            sbase = ob + (cutlass.Int64(row0) * P + inner) * 2
            # hm: running min of the SiLU arguments h of the current token group.
            # flag bit g: some lane of the warp had h < FAST_MIN_H in group g
            # (bit ngroup: the tail). Flagged groups are recomputed after the loop.
            hm = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Float32)
            flag = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Int32)
            flag[0] = cutlass.Int32(0)
            if slot >= 0:
                ngroup = count // U
                for g in cutlass.range(ngroup, unroll=1):
                    hm[0] = cutlass.Float32(H_INIT)
                    # Group base addresses: per-token offsets below are immediates.
                    gg = gbase + cutlass.Int64(g * U) * (self.SR * 2)
                    sg = sbase + cutlass.Int64(g * U) * (P * 2)
                    for p in cutlass.range_constexpr(U // ASYNC_GROUP):
                        if cutlass.const_expr(self.src_vec):
                            cute.arch.cp_async_wait_group(D // ASYNC_GROUP - 1)
                        for pe in cutlass.range_constexpr(ASYNC_GROUP):
                            u = ASYNC_GROUP * p + pe
                            self.step(
                                g * U + u,
                                u,
                                True,
                                src,
                                src_base,
                                row0,
                                cc,
                                ring_lane,
                                count,
                                wt,
                                hist,
                                yw,
                                hm,
                                cvalid,
                                q,
                                k,
                                v,
                                sec,
                                inner,
                                gg,
                                sg,
                            )
                        if cutlass.const_expr(self.src_vec):
                            cute.arch.cp_async_commit_group()
                    deep = cute.arch.vote_any_sync(hm[0] < FAST_MIN_H)
                    flag[0] = flag[0] | (cutlass.Int32(deep) << g)
                hm[0] = cutlass.Float32(H_INIT)
                first = ngroup * U
                npack = (count - first + (ASYNC_GROUP - 1)) // ASYNC_GROUP
                for p in cutlass.range(npack, unroll=1):
                    i = first + ASYNC_GROUP * p
                    if cutlass.const_expr(self.src_vec):
                        cute.arch.cp_async_wait_group(D // ASYNC_GROUP - 1)
                    gg = gbase + cutlass.Int64(i) * (self.SR * 2)
                    sg = sbase + cutlass.Int64(i) * (P * 2)
                    self.step(
                        i,
                        0,
                        True,
                        src,
                        src_base,
                        row0,
                        cc,
                        ring_lane,
                        count,
                        wt,
                        hist,
                        yw,
                        hm,
                        cvalid,
                        q,
                        k,
                        v,
                        sec,
                        inner,
                        gg,
                        sg,
                    )
                    for pe in cutlass.range_constexpr(1, ASYNC_GROUP):
                        if i + pe < count:
                            self.step(
                                i + pe,
                                pe,
                                True,
                                src,
                                src_base,
                                row0,
                                cc,
                                ring_lane,
                                count,
                                wt,
                                hist,
                                yw,
                                hm,
                                cvalid,
                                q,
                                k,
                                v,
                                sec,
                                inner,
                                gg,
                                sg,
                            )
                    if cutlass.const_expr(self.src_vec):
                        cute.arch.cp_async_commit_group()
                deep = cute.arch.vote_any_sync(hm[0] < FAST_MIN_H)
                flag[0] = flag[0] | (cutlass.Int32(deep) << ngroup)
                # Rare, warp-uniform: exact SiLU for flagged groups. hist is dead here.
                if flag[0] != 0:
                    for g in cutlass.range(ngroup + 1, unroll=1):
                        if (flag[0] >> g) & 1:
                            first = g * U
                            self.redo(
                                first,
                                cutlass.min(count - first, U),
                                src,
                                src_base,
                                begin,
                                t0,
                                init,
                                state_slot,
                                row0,
                                cc,
                                wt,
                                hist,
                                yw,
                                cvalid,
                                q,
                                k,
                                v,
                                sec,
                                inner,
                                sbase,
                            )
            else:
                npack = (count + (ASYNC_GROUP - 1)) // ASYNC_GROUP
                for p in cutlass.range(npack, unroll=1):
                    i = ASYNC_GROUP * p
                    if cutlass.const_expr(self.src_vec):
                        cute.arch.cp_async_wait_group(D // ASYNC_GROUP - 1)
                    gg = gbase + cutlass.Int64(i) * (self.SR * 2)
                    sg = sbase + cutlass.Int64(i) * (P * 2)
                    self.step(
                        i,
                        0,
                        False,
                        src,
                        src_base,
                        row0,
                        cc,
                        ring_lane,
                        count,
                        wt,
                        hist,
                        yw,
                        hm,
                        cvalid,
                        q,
                        k,
                        v,
                        sec,
                        inner,
                        gg,
                        sg,
                    )
                    for pe in cutlass.range_constexpr(1, ASYNC_GROUP):
                        if i + pe < count:
                            self.step(
                                i + pe,
                                pe,
                                False,
                                src,
                                src_base,
                                row0,
                                cc,
                                ring_lane,
                                count,
                                wt,
                                hist,
                                yw,
                                hm,
                                cvalid,
                                q,
                                k,
                                v,
                                sec,
                                inner,
                                gg,
                                sg,
                            )
                    if cutlass.const_expr(self.src_vec):
                        cute.arch.cp_async_commit_group()
            if cutlass.const_expr(self.src_vec):
                cute.arch.cp_async_wait_group(0)
            if t0 == 0:
                if slot >= 0:
                    if cvalid:
                        if cutlass.const_expr(self.st_vec):
                            # Element e = 3 * i + j (channel i, tap j) is half i % 2 of
                            # fin[4 * j + i // 2]; pack element pairs, then 3 x 16-byte stores.
                            ow = []
                            for mw in cutlass.range_constexpr(12):
                                srcw = []
                                sel = 0
                                for hh in cutlass.range_constexpr(2):
                                    e = 2 * mw + hh
                                    i, j = e // 3, e % 3
                                    srcw.append(fin[4 * j + i // 2])
                                    b = 2 * (i % 2) + 4 * hh
                                    sel |= (b | ((b + 1) << 4)) << (8 * hh)
                                ow.append(prmt(srcw[0], srcw[1], sel))
                            sa = self.state_addr(state, slot, cc)
                            for m in cutlass.range_constexpr(3):
                                st_v4(
                                    sa + 16 * m,
                                    ow[4 * m],
                                    ow[4 * m + 1],
                                    ow[4 * m + 2],
                                    ow[4 * m + 3],
                                )
                        else:
                            for j in cutlass.range_constexpr(WIDTH - 1):
                                for w in cutlass.range_constexpr(4):
                                    slo, shi = unpack_bf16x2(fin[4 * j + w])
                                    state_slot[cc + 2 * w, j] = cutlass.BFloat16(slo)
                                    state_slot[cc + 2 * w + 1, j] = cutlass.BFloat16(shi)

    @cute.jit
    def state_addr(self, state, slot, cc):
        """Byte address of state[slot, cc, 0] (vector path: 16-byte aligned by the host check)."""
        return state.iterator.toint() + (cutlass.Int64(slot) * self.SS[0] + cc * self.SS[1]) * 2

    @cute.jit
    def step(
        self,
        i,
        u: cutlass.Constexpr,
        conv: cutlass.Constexpr,
        src,
        src_base,
        row0,
        cc,
        ring_lane,
        count,
        wt: cute.Tensor,
        hist: cute.Tensor,
        yw: cute.Tensor,
        hm: cute.Tensor,
        cvalid,
        q,
        k,
        v,
        sec,
        inner,
        gb,
        sb,
    ):
        """Consume token i of this warp's run: ring slot -> conv+SiLU (or raw copy) -> store.

        gb/sb are the source/output addresses of token i - u (u is a compile-time offset)."""
        D = self.D
        sl = ring_lane + (i % D) * SLOT
        xt = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Int32)
        if cutlass.const_expr(self.src_vec):
            w4 = lds_v4(sl)
            for j in cutlass.range_constexpr(4):
                xt[j] = w4[j]
        else:
            self.loadw(src, src_base, row0 + i, cc, xt, 0)
        if cutlass.const_expr(conv):
            xs = []
            for e2 in cutlass.range_constexpr(4):
                xs.append(bf_lo(xt[e2]))
                xs.append(bf_hi(xt[e2]))
            hs = []
            for j in cutlass.range_constexpr(4):
                a, b = 2 * j, 2 * j + 1
                acc0, acc1 = mul2(wt[a], wt[b], hist[a], hist[b])
                acc0, acc1 = fma2(
                    wt[VEC + a], wt[VEC + b], hist[VEC + a], hist[VEC + b], acc0, acc1
                )
                acc0, acc1 = fma2(
                    wt[2 * VEC + a],
                    wt[2 * VEC + b],
                    hist[2 * VEC + a],
                    hist[2 * VEC + b],
                    acc0,
                    acc1,
                )
                acc0, acc1 = fma2(wt[3 * VEC + a], wt[3 * VEC + b], xs[a], xs[b], acc0, acc1)
                hs.append(acc0)
                hs.append(acc1)
                for e3 in cutlass.range_constexpr(2):
                    ch = a + e3
                    hist[ch] = hist[VEC + ch]
                    hist[VEC + ch] = hist[2 * VEC + ch]
                    hist[2 * VEC + ch] = xs[ch]
            # acc = conv/2 (weights pre-halved): silu(2h) = h + h * tanh(h)
            for j in cutlass.range_constexpr(4):
                h0, h1 = hs[2 * j], hs[2 * j + 1]
                y0, y1 = fma2(h0, h1, tanh_approx(h0), tanh_approx(h1), h0, h1)
                yw[j] = pack_bf16x2(y0, y1)
            # That form cancels for strongly negative h: track the group minimum
            # (the caller recomputes the group exactly if it is below the bound).
            hm[0] = fmin3(
                fmin3(hs[0], hs[1], hs[2]), fmin3(hs[3], hs[4], hs[5]), fmin3(hs[6], hs[7], hm[0])
            )
        else:
            for j in cutlass.range_constexpr(4):
                yw[j] = xt[j]
        # Refill this slot in program order after its same-thread lds_v4. The
        # caller commits one adjacent-token pack of refills as one cp.async group.
        if cutlass.const_expr(self.src_vec):
            if i + D < count:
                cp_async16(sl, gb + (u + D) * (self.SR * 2))
        if cvalid:
            if cutlass.const_expr(self.out_vec):
                st_v4(sb + u * (self.P * 2), yw[0], yw[1], yw[2], yw[3])
            else:
                self.storew(q, k, v, sec, row0 + i, inner, yw)

    @cute.jit
    def redo(
        self,
        first,
        n,
        src,
        src_base,
        begin,
        t0,
        init,
        state_slot,
        row0,
        cc,
        wt: cute.Tensor,
        hx: cute.Tensor,
        yr: cute.Tensor,
        cvalid,
        q,
        k,
        v,
        sec,
        inner,
        sbase,
    ):
        """Rare path: recompute tokens [first, first+n) of this run with exact SiLU and store
        over the fast results (same thread, same addresses, program order).

        Inputs are re-read from global memory (just read, so L2-resident); the history
        before `first` comes from src or, at the sequence start, from the initial state,
        which this thread only overwrites after this. hx is scratch (the caller's dead
        rolling-history registers).
        """
        for j in cutlass.range_constexpr(WIDTH - 1):
            r = t0 + first - (WIDTH - 1) + j  # sequence-relative token index
            if r >= 0:
                self.load8(src, src_base, begin + r, cc, hx, j * VEC)
            else:
                for i in cutlass.range_constexpr(VEC):
                    hx[j * VEC + i] = cutlass.Float32(0)
                if init:
                    for i in cutlass.range_constexpr(VEC):
                        hx[j * VEC + i] = cutlass.Float32(state_slot[cc + i, r + (WIDTH - 1)])
        xr = cute.make_rmem_tensor(cute.make_layout(VEC), cutlass.Float32)
        for u in cutlass.range(n, unroll=1):
            self.load8(src, src_base, row0 + first + u, cc, xr, 0)
            for j in cutlass.range_constexpr(4):
                a, b = 2 * j, 2 * j + 1
                acc0, acc1 = mul2(wt[a], wt[b], hx[a], hx[b])
                acc0, acc1 = fma2(wt[VEC + a], wt[VEC + b], hx[VEC + a], hx[VEC + b], acc0, acc1)
                acc0, acc1 = fma2(
                    wt[2 * VEC + a], wt[2 * VEC + b], hx[2 * VEC + a], hx[2 * VEC + b], acc0, acc1
                )
                acc0, acc1 = fma2(wt[3 * VEC + a], wt[3 * VEC + b], xr[a], xr[b], acc0, acc1)
                y0, y1 = silu2_exact(acc0, acc1)
                yr[j] = pack_bf16x2(y0, y1)
                for e3 in cutlass.range_constexpr(2):
                    ch = a + e3
                    hx[ch] = hx[VEC + ch]
                    hx[VEC + ch] = hx[2 * VEC + ch]
                    hx[2 * VEC + ch] = xr[ch]
            self.store(first + u, cvalid, q, k, v, sec, row0, inner, sbase, yr)

    @cute.jit
    def store(self, i, cvalid, q, k, v, sec, row0, inner, sbase, yw: cute.Tensor):
        if cvalid:
            if cutlass.const_expr(self.out_vec):
                st_v4(sbase + cutlass.Int64(i) * (self.P * 2), yw[0], yw[1], yw[2], yw[3])
            else:
                self.storew(q, k, v, sec, row0 + i, inner, yw)

    @cute.jit
    def __call__(
        self,
        src,
        weight,
        state,
        starts,
        slots,
        has_init,
        q,
        k,
        v,
        grid_y: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        channel_blocks = (self.C + WARP_CH - 1) // WARP_CH
        grid = (channel_blocks, grid_y, min(slots.shape[0], MAX_GRID_YZ))
        self.kernel(src, weight, state, starts, slots, has_init, q, k, v).launch(
            grid=grid, block=(self.W * 32, 1, 1), min_blocks_per_mp=self.MB, stream=stream
        )


MAX_GRID_YZ = 65535  # grid.y/z launch limit; the kernel grid-strides past it
FORCE_L = 32
FORCE_D = 8
FORCE_W = 1
FORCE_MB = 16


def _pick_tokens(heads, max_seqlen, batch):
    # Enough warps to fill the machine; longer runs amortise the 3-token halo.
    lanes_c = (3 * heads * 128 + WARP_CH - 1) // WARP_CH
    for L in (32, 16):
        warps = lanes_c * batch * ((max_seqlen + L - 1) // L)
        if warps >= 160 * 48:
            return L
    return 8


def _align(t):
    return t.data_ptr() % 16


def _layout_key(t, *, dynamic_dim=None, dynamic_stride=None):
    """Keep only layout properties that specialize the convolution."""
    shape = tuple(None if i == dynamic_dim else n for i, n in enumerate(t.shape))
    stride = tuple(None if i == dynamic_stride else s for i, s in enumerate(t.stride()))
    return (shape, stride, t.dtype)


@cache
def compile_prefill_conv(
    device: torch.device,
    layouts: tuple[tuple[tuple[int | None, ...], tuple[int | None, ...], torch.dtype], ...],
    tokens_per_warp: int,
    unroll: int,
    depth: int,
    warps: int,
    min_blocks_per_mp: int,
    src_vec: bool,
    w_vec: bool,
    out_vec: bool,
    st_vec: bool,
    seq_loop: bool,
) -> Callable:
    """Compile and cache a convolution layout without retaining caller-owned tensors."""
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm KDA convolution layout before CUDA graph capture")
        dtypes = {
            torch.bfloat16: (cutlass.BFloat16, 2),
            torch.int32: (cutlass.Int32, 4),
            torch.bool: (cutlass.Boolean, 1),
        }
        tokens, batch, pool, offsets = (cute.sym_int() for _ in range(4))
        # starts has B + 1 entries; its length is validated by the caller.
        extents = (tokens, None, pool, offsets, batch, batch, tokens, tokens, tokens)
        args = [
            make_fake_tensor(
                dtypes[dtype][0],
                tuple(extent if n is None else n for n in shape),
                stride=tuple(cute.sym_int() if s is None else s for s in stride),
                assumed_align=dtypes[dtype][1],
            )
            for (shape, stride, dtype), extent in zip(layouts, extents)
        ]
        op = KdaConv(
            layouts[1][0][0] // 384,
            tokens_per_warp,
            unroll,
            src_vec,
            w_vec,
            out_vec,
            layouts[0][1][0],
            layouts[2][1],
            seq_loop,
            depth,
            warps,
            st_vec,
        )
        op.MB = min_blocks_per_mp
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        # Grid size and stream remain runtime arguments, not cache-key inputs.
        return cute.compile(op, *args, cutlass.Int32(1), stream, options="--enable-tvm-ffi")


def fused_prefill_conv(src, weight, conv_state, starts, slots, has_init, max_seqlen, q, k, v):
    """Convolve raw packed QKV into contiguous, unnormalized BF16 Q/K/V.

    Starts must be monotone, with unique nonnegative slots in the pool. The
    scheduler prepares both arrays once per batch, before layer execution.
    Outputs and the state pool must not alias the immutable inputs.
    """
    tensors = (src, weight, conv_state, starts, slots, has_init, q, k, v)
    if any(t.device != src.device for t in tensors) or src.device.type != "cuda":
        raise ValueError("KDA convolution requires tensors on one CUDA device")
    if any(t.dtype != torch.bfloat16 for t in (src, weight, conv_state, q, k, v)):
        raise ValueError("KDA convolution requires BF16 inputs, outputs and state")
    channels = weight.shape[0]
    if (
        weight.shape != (channels, 4)
        or channels % 384
        or src.ndim != 2
        or src.shape[1] < channels
        or conv_state.ndim != 3
        or conv_state.shape[1:] != (channels, 3)
    ):
        raise ValueError(
            "KDA convolution requires width 4, head dimension 128 and three state taps"
        )
    shape = (1, src.shape[0], channels // 384, 128)
    if any(t.shape != shape or not t.is_contiguous() for t in (q, k, v)):
        raise ValueError("KDA convolution outputs must be contiguous [1, T, H, 128]")
    if (
        starts.dtype != torch.int32
        or starts.shape != (slots.numel() + 1,)
        or slots.dtype != torch.int32
        or slots.ndim != 1
        or has_init.dtype != torch.bool
        or has_init.shape != slots.shape
        or any(not t.is_contiguous() for t in (starts, slots, has_init))
    ):
        raise ValueError("Prepare contiguous int32 starts/slots and bool initial-state flags")
    # The scalar fallback uses CuTe layout arithmetic with int32 extents.
    # Reject overflowing layouts before launch rather than wrap an address.
    if any(
        sum((n - 1) * stride for n, stride in zip(t.shape, t.stride())) >= 2**31 for t in tensors
    ):
        raise ValueError("KDA convolution tensor offsets must fit int32")
    writable = (conv_state, q, k, v)
    for i, output in enumerate(writable):
        for other in (src, weight, starts, slots, has_init, *writable[:i]):
            if output.untyped_storage().data_ptr() == other.untyped_storage().data_ptr():
                raise ValueError("KDA convolution outputs/state must use disjoint storage")
    heads = weight.shape[0] // 384
    batch = slots.shape[0]
    if batch == 0 or heads == 0:
        return
    if isinstance(max_seqlen, torch.Tensor) and max_seqlen.device.type != "cpu":
        # Reading a device bound would sync (and break graph capture); T bounds every length.
        max_seqlen = src.shape[0]
    max_seqlen = int(max_seqlen)
    L = FORCE_L or _pick_tokens(heads, max(max_seqlen, 1), batch)
    D = FORCE_D
    W = FORCE_W or 4
    U = min(8, L)
    sr = src.stride(0)
    # Vector (cp.async) source path only for a 16-byte-aligned base and row stride.
    src_vec = src.stride(1) == 1 and sr % 8 == 0 and _align(src) == 0
    w_vec = _align(weight) == 0 and weight.is_contiguous()
    out_vec = all(_align(t) == 0 and t.is_contiguous() for t in (q, k, v))
    # 16-byte state rows per lane (8 channels x 3 taps) need a dense [C, 3] slot layout,
    # a 16-byte-aligned base and a slot stride divisible by 8 elements.
    ss = conv_state.stride()
    st_vec = ss[1] == 3 and ss[2] == 1 and ss[0] % 8 == 0 and _align(conv_state) == 0
    stream = cuda.CUstream(torch.cuda.current_stream(src.device).cuda_stream)
    # grid.y only sets parallelism (CTAs loop over the device-side run count): no
    # sequence is longer than T, so a looser bound only adds idle CTAs and an
    # understated one only makes CTAs loop.
    seqlen_bound = max(1, min(max_seqlen, src.shape[0]))
    run_tokens = L * W
    grid_y = min((seqlen_bound + run_tokens - 1) // run_tokens, MAX_GRID_YZ)
    seq_loop = batch > MAX_GRID_YZ  # shape-only choice
    # Token count, batch size and pool capacity are runtime extents. The
    # singleton leading output stride also varies with the token count.
    layouts = (
        _layout_key(src, dynamic_dim=0),
        _layout_key(weight),
        _layout_key(conv_state, dynamic_dim=0),
        _layout_key(starts, dynamic_dim=0),
        _layout_key(slots, dynamic_dim=0),
        _layout_key(has_init, dynamic_dim=0),
        *(_layout_key(t, dynamic_dim=1, dynamic_stride=0) for t in (q, k, v)),
    )
    fn = compile_prefill_conv(
        src.device,
        layouts,
        L,
        U,
        D,
        W,
        FORCE_MB,
        src_vec,
        w_vec,
        out_vec,
        st_vec,
        seq_loop,
    )
    fn(src, weight, conv_state, starts, slots, has_init, q, k, v, grid_y, stream)
