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
"""Inline-PTX helpers of the deferred_norm kernels (CuTe DSL).

deferred_norm owns every device helper its kernels use; it imports none from another
package. Two groups follow a contract defined elsewhere and must keep its
exact semantics:

* the arithmetic of the post_moe_hop (finalize FMA chain, bf16 packing/rounding,
  sanitize, square sums, ``rsqrt`` operand), which makes the deferred_norm residual,
  gain-folded activation, square-sum partials and ``r`` bit-identical to the
  post_moe_hop's;
* the post_moe_hop workspace protocol (``post_moe_hop/kernels.py`` and
  ``post_moe_hop/workspace.py``), which the deferred_norm hop shares with the post_moe_hop hops of
  the same TP group: Lamport sentinel, timer-bounded poll, per-buffer error words with sticky
  poison, timeout poll count, epoch/arrival/call counters. Every state-word byte
  offset is derived here from post_moe_hop's exported word indices, so the two hops
  address the same words.

The tests check both: bitwise equality with the post_moe_hop on the same workspace
(also interleaved with post_moe_hop calls, in CUDA graphs, under scrambled store
order), and the error protocol across both hop types.
"""

import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ..post_moe_hop.kernels import (
    ERROR_LAMPORT_TIMEOUT,
    STATE_ARRIVALS,
    STATE_CALLS,
    STATE_EPOCH,
    STATE_ERROR_WORDS,
    STATE_PREV_ROWS,
    STATE_TIMEOUT_POLLS,
    STATE_WORDS,
)

# The hop loads the state words with two 16-byte vectors.
assert STATE_WORDS == 8, "deferred_norm loads the post_moe_hop state words as two 16-byte vectors"

SENTINEL_WORD = -2147450880  # 0x80008000: two bf16 -0.0 (empty Lamport element)
FP32_SENTINEL = -2147483648  # 0x80000000: fp32 -0.0 (empty row-sum partial)
# Polls between two deadline checks; the clock starts at the first check.
POLLS_PER_CLOCK_CHECK = 1024

_OFF_EPOCH = 4 * STATE_EPOCH
_OFF_PREV_ROWS = 4 * STATE_PREV_ROWS
_OFF_ARRIVALS = 4 * STATE_ARRIVALS
_OFF_CALLS = 4 * STATE_CALLS
_OFF_TIMEOUT_POLLS = 4 * STATE_TIMEOUT_POLLS
ERROR_WORD_OFFSETS = tuple(4 * w for w in STATE_ERROR_WORDS)


def _asm(res, args, text, cons, side_effects=True, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        args,
        text,
        cons,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _v(x, loc, ip):
    return x.ir_value(loc=loc, ip=ip)


def _struct(n, ty):
    return llvm.StructType.get_literal([ty] * n)


def _unpack(res, n, cls, ty, loc, ip):
    return tuple(cls(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(n))


# --------------------------------------------------------------------------
# Global memory and the post_moe_hop workspace protocol.
# --------------------------------------------------------------------------


@dsl_user_op
def ld_v4(addr: Int64, *, loc=None, ip=None):
    res = _asm(
        _struct(4, T.i32()),
        [_v(addr, loc, ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, 4, Int32, T.i32(), loc, ip)


@dsl_user_op
def st_v4(addr: Int64, a: Int32, b: Int32, c: Int32, d: Int32, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [_v(addr, loc, ip)] + [_v(x, loc, ip) for x in (a, b, c, d)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


def load_state(state: Int64):
    """The 8 state words of the rank's post_moe_hop workspace (two 16-byte loads).

    Returns ``(epoch, prev_rows, error_words)`` with the three per-buffer
    error words in buffer order. Called after the kernel's dependency wait;
    the words were written by the previous hop call of this workspace, which
    is complete at that point (kernel boundary).
    """
    words = ld_v4(state) + ld_v4(state + Int64(16))
    return (words[STATE_EPOCH], words[STATE_PREV_ROWS], [words[w] for w in STATE_ERROR_WORDS])


@dsl_user_op
def poll8(base: Int64, stride: Int64, timeout_ns: Int64, *, loc=None, ip=None):
    """Spin until 8 rank slots (16 bytes each, ``stride`` apart) hold no sentinel.

    Volatile 16-byte loads of all 8 slots per round; the ``%globaltimer``
    deadline is checked every ``POLLS_PER_CLOCK_CHECK`` rounds and its clock
    starts at the first check, so a poll that completes within that many
    rounds never reads the timer. Returns the 32 words, a nonzero flag when
    the deadline expired, and the round count. Same loop as the post_moe_hop cluster
    kernel's poll.
    """
    loads = []
    checks = []
    for r in range(8):
        o = 4 * r
        loads.append(
            f"ld.volatile.global.v4.b32 {{${o}, ${o + 1}, ${o + 2}, ${o + 3}}}, [a];\n"
            "add.s64 a, a, $35;\n"
        )
        for w in range(4):
            checks.append(
                f"and.b32 t, ${o + w}, 0xFFFF0000;\n"
                "setp.eq.u32 p, t, 0x80000000;\n"
                "selp.b32 t, 1, 0, p;\n"
                "or.b32 d, d, t;\n"
            )
    period = POLLS_PER_CLOCK_CHECK
    asm = (
        "{\n.reg .pred p;\n.reg .b32 t, d, n;\n.reg .b64 a, t0, now;\n"
        "mov.b32 n, 0;\nmov.b64 t0, 0;\n"
        "NORM_POLL:\nmov.b64 a, $34;\nmov.b32 d, 0;\n"
        + "".join(loads)
        + "".join(checks)
        + "add.u32 n, n, 1;\nsetp.eq.u32 p, d, 0;\n@p bra NORM_POLL_DONE;\n"
        f"and.b32 t, n, {period - 1};\nsetp.ne.u32 p, t, 0;\n@p bra NORM_POLL;\n"
        "mov.u64 now, %globaltimer;\n"
        f"setp.eq.u32 p, n, {period};\n@p mov.b64 t0, now;\n"
        "sub.s64 now, now, t0;\nsetp.lt.s64 p, now, $36;\n@p bra NORM_POLL;\n"
        "NORM_POLL_DONE:\nmov.b32 $32, d;\nmov.b32 $33, n;\n}\n"
    )
    constraints = ",".join(["=r"] * 34 + ["l", "l", "l"])
    res = _asm(
        _struct(34, T.i32()),
        [_v(base, loc, ip), _v(stride, loc, ip), _v(timeout_ns, loc, ip)],
        asm,
        constraints,
        loc=loc,
        ip=ip,
    )
    return _unpack(res, 34, Int32, T.i32(), loc, ip)


@dsl_user_op
def select3(values: list, index: Int32, *, loc=None, ip=None) -> Int32:
    """``values[index]`` for ``index`` in 0..2 (e.g. the error word of a buffer)."""
    return Int32(
        _asm(
            T.i32(),
            [_v(index, loc, ip)] + [_v(w, loc, ip) for w in values],
            "{\n.reg .pred p;\nsetp.eq.u32 p, $1, 1;\nselp.b32 $0, $3, $2, p;\n"
            "setp.eq.u32 p, $1, 2;\nselp.b32 $0, $4, $0, p;\n}\n",
            "=r,r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


def error_word_address(state: Int64, buffer: Int32) -> Int64:
    """Address of the error word of Lamport buffer ``buffer`` (0..2)."""
    return state + Int64(select3([Int32(o) for o in ERROR_WORD_OFFSETS], buffer))


@dsl_user_op
def flag_error(word: Int64, flag: Int32, bits: Int32, *, loc=None, ip=None) -> None:
    """OR ``bits`` into the error word at ``word`` when ``flag`` is nonzero."""
    _asm(
        None,
        [_v(word, loc, ip), _v(flag, loc, ip), _v(bits, loc, ip)],
        "{\n.reg .pred p;\nsetp.ne.s32 p, $1, 0;\n@p red.relaxed.gpu.global.or.b32 [$0], $2;\n}\n",
        "l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def flag_timeout(
    word: Int64, state: Int64, flag: Int32, polls: Int32, *, loc=None, ip=None
) -> None:
    """The Lamport-timeout bit into the error word and the largest poll count
    at expiry into the ``timeout_polls`` state word."""
    _asm(
        None,
        [_v(word, loc, ip), _v(state, loc, ip), _v(flag, loc, ip), _v(polls, loc, ip)],
        "{\n.reg .pred p;\nsetp.ne.s32 p, $2, 0;\n"
        f"@p red.relaxed.gpu.global.or.b32 [$0], {ERROR_LAMPORT_TIMEOUT};\n"
        f"@p red.relaxed.gpu.global.max.u32 [$1+{_OFF_TIMEOUT_POLLS}], $3;\n}}\n",
        "l,l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def arrive(
    state: Int64,
    is_leader: Int32,
    num_ctas: Int32,
    num_rows: Int32,
    next_epoch: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Leader thread of each CTA counts in; the last CTA records the row count
    to clear next, resets the arrival count, counts the call and publishes
    the next epoch (post_moe_hop's arrival protocol)."""
    _asm(
        None,
        [
            _v(state, loc, ip),
            _v(is_leader, loc, ip),
            _v(num_ctas, loc, ip),
            _v(num_rows, loc, ip),
            _v(next_epoch, loc, ip),
        ],
        "{\n.reg .pred p, q;\n.reg .b32 old, c;\n"
        "setp.ne.s32 p, $1, 0;\nmov.b32 old, 0;\n"
        f"@p atom.acq_rel.gpu.global.add.u32 old, [$0+{_OFF_ARRIVALS}], 1;\n"
        "add.u32 c, old, 1;\nsetp.eq.u32 q, c, $2;\nand.pred q, q, p;\n"
        f"@q st.global.u32 [$0+{_OFF_PREV_ROWS}], $3;\n"
        f"@q st.global.u32 [$0+{_OFF_ARRIVALS}], 0;\n"
        f"@q red.relaxed.gpu.global.add.u32 [$0+{_OFF_CALLS}], 1;\n"
        f"@q st.global.u32 [$0+{_OFF_EPOCH}], $4;\n}}\n",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fault_delay(a: Int32, b: Int32, c: Int32, max_ns: int, site: int, *, loc=None, ip=None) -> None:
    """Test-only fault injection: about one thread in four spins before ``site``.

    The delay (below ``max_ns``) is a hash of ``a``, ``b`` and ``c`` (post_moe_hop's
    hash), so the individual 16-byte stores of one call land in a scrambled
    order.
    """
    asm = (
        "{\n.reg .pred p;\n.reg .b32 h, t;\n.reg .b64 t0, now, d;\n"
        "mul.lo.u32 h, $0, 0x9E3779B1;\nmul.lo.u32 t, $1, 0x85EBCA77;\nxor.b32 h, h, t;\n"
        "mul.lo.u32 t, $2, 0xC2B2AE3D;\nxor.b32 h, h, t;\nshr.u32 t, h, 15;\n"
        "xor.b32 h, h, t;\nmul.lo.u32 h, h, 0x2C1B3C6D;\nshr.u32 t, h, 13;\n"
        "xor.b32 h, h, t;\nand.b32 t, h, 3;\nsetp.ne.u32 p, t, 0;\n"
        f"@p bra NORM_FAULT_END_{site};\nshr.u32 t, h, 2;\nrem.u32 t, t, {max_ns};\n"
        "cvt.u64.u32 d, t;\nmov.u64 t0, %globaltimer;\n"
        f"NORM_FAULT_LOOP_{site}:\nmov.u64 now, %globaltimer;\nsub.s64 now, now, t0;\n"
        f"setp.lt.s64 p, now, d;\n@p bra NORM_FAULT_LOOP_{site};\nNORM_FAULT_END_{site}:\n}}\n"
    )
    _asm(None, [_v(a, loc, ip), _v(b, loc, ip), _v(c, loc, ip)], asm, "r,r,r", loc=loc, ip=ip)


# --------------------------------------------------------------------------
# Arithmetic of the post_moe_hop.
# --------------------------------------------------------------------------


@dsl_user_op
def unpack_bf16x2(word: Int32, *, loc=None, ip=None):
    res = _asm(
        _struct(2, T.f32()),
        [_v(word, loc, ip)],
        "{\n.reg .b32 t;\nshl.b32 t, $2, 16;\nmov.b32 $0, t;\n"
        "and.b32 t, $2, 0xFFFF0000;\nmov.b32 $1, t;\n}\n",
        "=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, 2, Float32, T.f32(), loc, ip)


@dsl_user_op
def pack_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int32:
    """Round-to-nearest-even of two fp32 values into one bf16x2 word (``lo`` low)."""
    return Int32(
        _asm(
            T.i32(),
            [_v(lo, loc, ip), _v(hi, loc, ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def round_bf16(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(x, loc, ip)],
            "{\n.reg .b32 t;\ncvt.rn.bf16x2.f32 t, $1, $1;\n"
            "and.b32 t, t, 0xFFFF0000;\nmov.b32 $0, t;\n}\n",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def sanitize(word: Int32, *, loc=None, ip=None) -> Int32:
    """bf16 -0.0 (the Lamport sentinel) -> +0.0 in both halves."""
    return Int32(
        _asm(
            T.i32(),
            [_v(word, loc, ip)],
            "{\n.reg .b32 t;\n.reg .pred p, q;\n"
            "and.b32 t, $1, 0xFFFF;\nsetp.eq.u32 p, t, 0x8000;\n"
            "and.b32 t, $1, 0xFFFF0000;\nsetp.eq.u32 q, t, 0x80000000;\n"
            "mov.b32 $0, $1;\n@p and.b32 $0, $0, 0xFFFF0000;\n"
            "@q and.b32 $0, $0, 0xFFFF;\n}\n",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def poison(word: Int32, flag: Int32, *, loc=None, ip=None) -> Int32:
    """Two bf16 NaNs instead of ``word`` when ``flag`` is nonzero."""
    return Int32(
        _asm(
            T.i32(),
            [_v(word, loc, ip), _v(flag, loc, ip)],
            "{\n.reg .pred p;\nsetp.ne.s32 p, $2, 0;\nselp.b32 $0, 0x7FC07FC0, $1, p;\n}\n",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def perm_check(index: Int32, rows: Int32, *, loc=None, ip=None):
    """Return ``(valid, safe_index, out_of_range)`` of one permutation entry.

    ``valid`` is ``0 <= index < rows``; ``safe_index`` is ``index`` when valid
    and 0 otherwise; ``out_of_range`` flags ``index >= rows`` (negative
    entries are the native "no local row" marker, not an error).
    """
    res = _asm(
        _struct(3, T.i32()),
        [_v(index, loc, ip), _v(rows, loc, ip)],
        "{\n.reg .pred p, q;\nsetp.lt.u32 p, $3, $4;\nselp.b32 $0, 1, 0, p;\n"
        "selp.b32 $1, $3, 0, p;\nsetp.ge.s32 q, $3, $4;\nselp.b32 $2, 1, 0, q;\n}\n",
        "=r,=r,=r,r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, 3, Int32, T.i32(), loc, ip)


@dsl_user_op
def fma(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(a, loc, ip), _v(b, loc, ip), _v(c, loc, ip)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def add(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(a, loc, ip), _v(b, loc, ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def mul(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(a, loc, ip), _v(b, loc, ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def select_f32(pred: Int32, a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(pred, loc, ip), _v(a, loc, ip), _v(b, loc, ip)],
            "{\n.reg .pred p;\nsetp.ne.s32 p, $1, 0;\nselp.f32 $0, $2, $3, p;\n}\n",
            "=f,r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bits_f32(word: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), [_v(word, loc, ip)], "mov.b32 $0, $1;", "=f,r", loc=loc, ip=ip))


@dsl_user_op
def inv_rms(square_sum: Float32, hidden: Float32, eps: Float32, *, loc=None, ip=None) -> Float32:
    """``rsqrt.approx(div.rn(square_sum, hidden) + eps)`` (post_moe_hop's ``r``)."""
    return Float32(
        _asm(
            T.f32(),
            [_v(square_sum, loc, ip), _v(hidden, loc, ip), _v(eps, loc, ip)],
            "{\n.reg .f32 t;\ndiv.rn.f32 t, $1, $2;\nadd.rn.f32 t, t, $3;\n"
            "rsqrt.approx.f32 $0, t;\n}\n",
            "=f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def shfl_xor(x: Float32, mask: int, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [_v(x, loc, ip)],
            f"shfl.sync.bfly.b32 $0, $1, {mask}, 0x1f, 0xffffffff;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


# --------------------------------------------------------------------------
# Projection kernel: shared memory, cp.async and MMA.
# --------------------------------------------------------------------------


@dsl_user_op
def saddr(p: cute.Pointer, *, loc=None, ip=None) -> Int32:
    """32-bit shared-window address of a shared-memory pointer."""
    return Int32(p.toint(loc=loc, ip=ip))


@dsl_user_op
def cp_async16(smem: Int32, gaddr: Int64, *, loc=None, ip=None):
    _asm(
        None,
        [smem.ir_value(), gaddr.ir_value()],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_commit(*, loc=None, ip=None):
    _asm(None, [], "cp.async.commit_group;", "", loc=loc, ip=ip)


@dsl_user_op
def cp_async_wait_group(n: int, *, loc=None, ip=None):
    """Wait until at most ``n`` committed cp.async groups are pending."""
    _asm(None, [], f"cp.async.wait_group {n};", "", loc=loc, ip=ip)


@dsl_user_op
def st_global_b16(gaddr: Int64, v: Int32, *, loc=None, ip=None):
    _asm(
        None,
        [gaddr.ir_value(), v.ir_value()],
        "{\n\t.reg .b16 lo;\n\tcvt.u16.u32 lo, $1;\n\tst.global.b16 [$0], lo;\n\t}",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_shared_v4(smem: Int32, a: Int32, b: Int32, c: Int32, d: Int32, *, loc=None, ip=None):
    _asm(
        None,
        [smem.ir_value(), a.ir_value(), b.ir_value(), c.ir_value(), d.ir_value()],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ldsm_x4(smem: Int32, *, loc=None, ip=None):
    out = _asm(
        _struct(4, T.i32()),
        [smem.ir_value()],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(out, 4, Int32, T.i32(), loc, ip)


@dsl_user_op
def mma_bf16(
    a0: Int32,
    a1: Int32,
    a2: Int32,
    a3: Int32,
    b0: Int32,
    b1: Int32,
    c0: Float32,
    c1: Float32,
    c2: Float32,
    c3: Float32,
    *,
    loc=None,
    ip=None,
):
    """``mma.sync.m16n8k16`` BF16 x BF16 + FP32 (row.col)."""
    out = _asm(
        _struct(4, T.f32()),
        [
            a0.ir_value(),
            a1.ir_value(),
            a2.ir_value(),
            a3.ir_value(),
            b0.ir_value(),
            b1.ir_value(),
            c0.ir_value(),
            c1.ir_value(),
            c2.ir_value(),
            c3.ir_value(),
        ],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return _unpack(out, 4, Float32, T.f32(), loc, ip)


@dsl_user_op
def bf16_bits(x: Float32, *, loc=None, ip=None) -> Int32:
    """Round-to-nearest-even of one fp32 value to bf16 (in the low half)."""
    return Int32(
        _asm(
            T.i32(),
            [x.ir_value()],
            "{\n\t.reg .b16 h;\n\tcvt.rn.bf16.f32 h, $1;\n\tcvt.u32.u16 $0, h;\n\t}",
            "=r,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )
