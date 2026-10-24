# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Portions adapted from vLLM (https://github.com/vllm-project/vllm) under the Apache-2.0
# license: the Kimi K3 specialized decode mailbox primitives,
# vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/common/{primitives,mailbox,twoshot}.py.
# Original copyright: Copyright contributors to the vLLM project
"""Lamport mailbox primitives for Kimi K3 Rubin CuTe DSL kernels (inline PTX).

A mailbox is device memory whose 32-bit words are *armed* with the sentinel ``0x80000000`` (an
FP32 ``-0.0``, the same word ``cpp/tensorrt_llm/common/lamportUtils.cuh`` clears the MNNVL
all-reduce buffers with). A producer publishes data with 16-byte stores; a consumer polls the
16-byte fragment until none of its four words is the sentinel, uses it, and **re-arms** it (stores
the sentinel back) in the same thread. A published word can never equal the sentinel because
producers turn every BF16 ``-0`` into ``+0`` (:func:`sanitize_bf16x2`): the sentinel is the BF16
pair (high ``-0``, low ``+0``).

Ordering contract (see ``tensorrt_llm/_torch/distributed/k3_lamport_mailbox.py``):

* a kernel publishes only after its ``griddepcontrol.wait`` (or without PDL);
* every fragment that is published is polled and re-armed by its reader in the same mailbox use;
* two consecutive mailbox uses of a TP group never share a parity region.

Mailbox stores (publishes, pushes, re-arms) are ``st.relaxed.sys`` (or ``multimem.st.relaxed.sys``)
and polls are ``ld.volatile`` (relaxed, system scope): morally strong pairs in the PTX memory model.
Each 32-bit word is single-copy atomic, so a fragment is complete once none of its words is the
sentinel. ``TRTLLM_K3_LAMPORT_SPIN_LIMIT=N`` (debug) compiles every poll with a pass counter that
traps after ``N`` failed passes, so a hang becomes a CUDA error; by default polls spin until the
data arrives, like the MNNVL all-reduce's.

The helpers are ``dsl_user_op`` wrappers around fixed PTX so that the numerics and the memory
operations are exactly the ones written here (no fast math, no contraction): FP32 sums use
``add.rn.f32`` (denormals kept, like the MNNVL kernels, which are built without fast math), BF16
rounding uses ``cvt.rn.bf16x2.f32`` (round to nearest even, like ``__float2bfloat16``).
"""

from __future__ import annotations

import itertools
import os

from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ..blackwell.utils import griddepcontrol_launch_dependents, griddepcontrol_wait

#: An armed (unpublished) mailbox word: FP32 -0.0.
SENTINEL = 0x80000000
#: The sentinel as a signed 32-bit value (for torch.int32 fills and checks).
SENTINEL_I32 = -0x80000000
#: Bytes of one mailbox fragment (one ``.v4.b32`` access, 8 BF16 values).
FRAGMENT_BYTES = 16
#: BF16 elements per fragment.
FRAGMENT_ELEMS = 8
#: Default ``spin_limit`` of :func:`poll_slots_v4` (``TRTLLM_K3_LAMPORT_SPIN_LIMIT``, 0 = unbounded).
SPIN_LIMIT = int(os.environ.get("TRTLLM_K3_LAMPORT_SPIN_LIMIT", "0"))

_LABELS = itertools.count()


def _label(prefix: str) -> str:
    # PTX labels must be unique in the function; every inlined poll gets its own.
    return f"K3LMP_{prefix}{next(_LABELS)}"


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


def _words(res, count, *, loc=None, ip=None):
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(count)]


# ---------------------------------------------------------------------------------------------
# Plain loads and stores of one fragment
# ---------------------------------------------------------------------------------------------


@dsl_user_op
def load_v4(addr, *, loc=None, ip=None):
    """Coherent 16-byte global load (data of the previous grid: issue after the PDL wait)."""
    res = _asm_se(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return _words(res, 4, loc=loc, ip=ip)


@dsl_user_op
def store_v4(addr, words, *, loc=None, ip=None):
    """Plain 16-byte global store (kernel outputs; not for mailboxes)."""
    _asm_se(
        None,
        [_lv(addr, loc=loc, ip=ip)] + [_iv(w, loc=loc, ip=ip) for w in words],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mailbox_store_v4(addr, words, *, loc=None, ip=None):
    """16-byte ``st.relaxed.sys`` store to a mailbox (polled by other GPUs); to a multicast address
    it reaches every rank."""
    _asm_se(
        None,
        [_lv(addr, loc=loc, ip=ip)] + [_iv(w, loc=loc, ip=ip) for w in words],
        "st.relaxed.sys.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def multimem_store_v4(mc_addr, words, *, loc=None, ip=None):
    """16-byte ``multimem.st.relaxed.sys`` to a multicast address (bit-exact: the words are
    moved into ``.f32`` registers, never converted)."""
    _asm_se(
        None,
        [_lv(mc_addr, loc=loc, ip=ip)] + [_iv(w, loc=loc, ip=ip) for w in words],
        "{\n.reg .f32 a, b, c, d;\nmov.b32 a, $1;\nmov.b32 b, $2;\nmov.b32 c, $3;\nmov.b32 d, $4;\n"
        "multimem.st.relaxed.sys.global.v4.f32 [$0], {a, b, c, d};\n}",
        "l,r,r,r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


#: Publish kinds of :func:`publish_v4`.
PUBLISH_ST = "st"
PUBLISH_MULTIMEM = "multimem"


def publish_v4(addr, words, kind: str = PUBLISH_ST):
    """Publish one sanitized fragment at ``addr`` (a multicast address for an all-rank publish,
    or a peer's unicast address for a point-to-point push). ``kind`` selects
    ``st.relaxed.sys.global`` (to the multicast VA, as the MNNVL one-shot stores) or
    ``multimem.st.relaxed.sys``; the latter needs a multicast address."""
    if kind == PUBLISH_MULTIMEM:
        multimem_store_v4(addr, words)
    elif kind == PUBLISH_ST:
        mailbox_store_v4(addr, words)
    else:
        raise ValueError(f"publish kind {kind!r}")


@dsl_user_op
def rearm_slots_v4(addr, count: int, stride: int, *, loc=None, ip=None):
    """Store the sentinel into ``count`` fragments at ``addr + i * stride`` (bytes)."""
    lines = ["{", ".reg .b32 s;", f"mov.b32 s, {SENTINEL:#x};"]
    for i in range(count):
        lines.append(f"st.relaxed.sys.global.v4.b32 [$0+{i * stride}], {{s, s, s, s}};")
    lines.append("}")
    _asm_se(None, [_lv(addr, loc=loc, ip=ip)], "\n".join(lines), "l,~{memory}", loc=loc, ip=ip)


# ---------------------------------------------------------------------------------------------
# Sentinel handling and polling
# ---------------------------------------------------------------------------------------------


@dsl_user_op
def sanitize_bf16x2(word, *, loc=None, ip=None):
    """``word`` with each negative-zero BF16 half turned into +0 (so it is never the sentinel).
    Same rule as ``sanitizeLamportPayload`` of the MNNVL kernels."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(word, loc=loc, ip=ip)],
            "{\n.reg .b32 h;\n.reg .pred p;\nmov.b32 $0, $1;\n"
            "and.b32 h, $1, 0xffff;\nsetp.eq.b32 p, h, 0x8000;\n@p and.b32 $0, $0, 0xffff0000;\n"
            "shr.u32 h, $1, 16;\nsetp.eq.b32 p, h, 0x8000;\n@p and.b32 $0, $0, 0x0000ffff;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


def sanitize_v4(words):
    return [sanitize_bf16x2(w) for w in words]


def _poll_text(count: int, stride: int, backoff_ns: int, spin_limit: int, vote: bool) -> str:
    """PTX of a Lamport poll of ``count`` fragments at ``[$addr + i * stride]``.

    Every pass issues all ``count`` 16-byte volatile loads before any check (so the round trips
    overlap) and checks all words; it repeats until no word is the sentinel. ``vote``: the warp
    leaves together (``vote.sync.any``), every lane of the warp must execute the poll.
    ``backoff_ns``: ``nanosleep`` between passes. ``spin_limit``: trap after that many failed
    passes (debug; 0 = unbounded)."""
    loop, done = _label("POLL"), _label("DONE")
    addr = f"${4 * count}"
    lines = ["{", ".reg .pred d;"]
    if vote:
        lines.append(".reg .pred v;")
    if spin_limit:
        lines += [".reg .u32 n;", ".reg .pred q;", "mov.u32 n, 0;"]
    lines.append(f"{loop}:")
    for i in range(count):
        regs = ", ".join(f"${4 * i + j}" for j in range(4))
        lines.append(f"ld.volatile.global.v4.b32 {{{regs}}}, [{addr}+{i * stride}];")
    for k in range(4 * count):
        if k == 0:
            lines.append(f"setp.eq.b32 d, ${k}, {SENTINEL:#x};")
        else:
            lines.append(f"setp.eq.or.b32 d, ${k}, {SENTINEL:#x}, d;")
    if vote:
        lines.append("vote.sync.any.pred v, d, 0xffffffff;")
        lines.append(f"@!v bra {done};")
    else:
        lines.append(f"@!d bra {done};")
    if backoff_ns:
        lines.append(f"nanosleep.u32 {int(backoff_ns)};")
    if spin_limit:
        lines += ["add.u32 n, n, 1;", f"setp.gt.u32 q, n, {int(spin_limit)};", "@q trap;"]
    lines.append(f"bra {loop};")
    lines.append(f"{done}:")
    lines.append("}")
    return "\n".join(lines)


@dsl_user_op
def poll_slots_v4(
    addr,
    count: int,
    stride: int,
    backoff_ns: int = 0,
    spin_limit: int | None = None,
    vote: bool = False,
    *,
    loc=None,
    ip=None,
):
    """Lamport-poll ``count`` fragments at ``addr + i * stride`` bytes; returns ``count`` lists of
    four words, all published. ``stride`` must be a compile-time constant. ``spin_limit``: None
    takes :data:`SPIN_LIMIT` (kernels that cache compiled code must key on the value used)."""
    if spin_limit is None:
        spin_limit = SPIN_LIMIT
    text = _poll_text(count, stride, backoff_ns, spin_limit, vote)
    # Early-clobber outputs: the loop reads the address after writing them.
    constraints = ",".join(["=&r"] * (4 * count) + ["l", "~{memory}"])
    res = _asm_se(
        llvm.StructType.get_literal([T.i32()] * (4 * count)),
        [_lv(addr, loc=loc, ip=ip)],
        text,
        constraints,
        loc=loc,
        ip=ip,
    )
    words = _words(res, 4 * count, loc=loc, ip=ip)
    return [words[4 * i : 4 * i + 4] for i in range(count)]


@dsl_user_op
def poll_slots_v4_if(
    valid,
    addr,
    count: int,
    stride: int,
    spin_limit: int | None = None,
    *,
    loc=None,
    ip=None,
):
    """:func:`poll_slots_v4` predicated on ``valid != 0``: a thread with ``valid == 0`` issues no
    load and gets zeros (lanes of one warp may differ; no warp vote)."""
    if spin_limit is None:
        spin_limit = SPIN_LIMIT
    loop, done = _label("PIF"), _label("PIFD")
    n = 4 * count
    lines = ["{", ".reg .pred d, v;"]
    if spin_limit:
        lines += [".reg .u32 n;", ".reg .pred q;", "mov.u32 n, 0;"]
    lines += [f"mov.b32 ${k}, 0;" for k in range(n)]
    lines += [f"setp.ne.u32 v, ${n}, 0;", f"@!v bra {done};", f"{loop}:"]
    for i in range(count):
        regs = ", ".join(f"${4 * i + j}" for j in range(4))
        lines.append(f"ld.volatile.global.v4.b32 {{{regs}}}, [${n + 1}+{i * stride}];")
    for k in range(n):
        if k == 0:
            lines.append(f"setp.eq.b32 d, ${k}, {SENTINEL:#x};")
        else:
            lines.append(f"setp.eq.or.b32 d, ${k}, {SENTINEL:#x}, d;")
    lines.append(f"@!d bra {done};")
    if spin_limit:
        lines += ["add.u32 n, n, 1;", f"setp.gt.u32 q, n, {int(spin_limit)};", "@q trap;"]
    lines += [f"bra {loop};", f"{done}:", "}"]
    constraints = ",".join(["=&r"] * n + ["r", "l", "~{memory}"])
    res = _asm_se(
        llvm.StructType.get_literal([T.i32()] * n),
        [_iv(valid, loc=loc, ip=ip), _lv(addr, loc=loc, ip=ip)],
        "\n".join(lines),
        constraints,
        loc=loc,
        ip=ip,
    )
    words = _words(res, n, loc=loc, ip=ip)
    return [words[4 * i : 4 * i + 4] for i in range(count)]


@dsl_user_op
def rearm_slots_v4_if(valid, addr, count: int, stride: int, *, loc=None, ip=None):
    """:func:`rearm_slots_v4` predicated on ``valid != 0``."""
    lines = ["{", ".reg .b32 s;", ".reg .pred v;", f"mov.b32 s, {SENTINEL:#x};"]
    lines.append("setp.ne.u32 v, $0, 0;")
    for i in range(count):
        lines.append(f"@v st.relaxed.sys.global.v4.b32 [$1+{i * stride}], {{s, s, s, s}};")
    lines.append("}")
    _asm_se(
        None,
        [_iv(valid, loc=loc, ip=ip), _lv(addr, loc=loc, ip=ip)],
        "\n".join(lines),
        "r,l,~{memory}",
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------------------------
# Arithmetic (fixed instructions)
# ---------------------------------------------------------------------------------------------


@dsl_user_op
def zero_f32(*, loc=None, ip=None):
    return Float32(_asm(T.f32(), [], "mov.b32 $0, 0f00000000;", "=f", loc=loc, ip=ip))


@dsl_user_op
def add_rn(a, b, *, loc=None, ip=None):
    """IEEE ``a + b`` (round to nearest even, denormals kept)."""
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
def unpack_bf16x2(word, *, loc=None, ip=None):
    """``(float(lo), float(hi))`` of a packed BF16x2 word (exact)."""
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
def pack_bf16x2(lo, hi, *, loc=None, ip=None):
    """BF16x2 word ``{bf16_rn(hi), bf16_rn(lo)}``."""
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


def rank_order_sum(slots, chunk: int):
    """FP32 sum of ``len(slots)`` fragments (lists of four BF16x2 words) in rank order, rounded
    once to BF16x2 words: ``acc = 0; for each chunk c: s = 0; s += slot r for r in c; acc += s``.

    ``chunk`` reproduces the MNNVL all-reduce order (mnnvlAllreduceKernels.cu,
    ``accumulateLamportRanksChunked`` / ``reduceOneshotDeterministic``): the one-shot uses one
    chunk up to 8 ranks and chunks of 8 above; the two-shot one chunk up to 16 ranks."""
    world = len(slots)
    if chunk <= 0 or world % chunk:
        raise ValueError(f"chunk {chunk} must divide {world}")
    zero = zero_f32()
    acc = [zero] * 8
    for first in range(0, world, chunk):
        part = [zero] * 8
        for r in range(first, first + chunk):
            for j in range(4):
                lo, hi = unpack_bf16x2(slots[r][j])
                part[2 * j] = add_rn(part[2 * j], lo)
                part[2 * j + 1] = add_rn(part[2 * j + 1], hi)
        acc = [add_rn(acc[e], part[e]) for e in range(8)]
    return [pack_bf16x2(acc[2 * j], acc[2 * j + 1]) for j in range(4)]


def oneshot_chunk(world: int) -> int:
    """Rank chunk of the MNNVL one-shot sum order (fast path up to 8 ranks, chunks of 8 above)."""
    return world if world <= 8 else 8


#: Most slots one poll of :func:`poll_sum_slots_v4` holds in registers (4 words each).
POLL_GROUP = 8


def poll_sum_slots_v4(valid, addr, world: int, stride: int):
    """Poll, sum and re-arm the ``world`` slots of one one-shot fragment.

    Slot ``r`` (rank ``r``'s 16-byte publish) is at ``addr + r * stride``. Returns the four BF16x2
    words of the FP32 rank-order sum in the MNNVL one-shot order (:func:`rank_order_sum` with
    :func:`oneshot_chunk`), bit-identical to the one-shot ``MNNVLAllReduce`` output of the same
    payloads; zeros (and no memory access) where ``valid == 0``. The slots are polled in groups of
    at most :data:`POLL_GROUP` (register bound), each group inside one MNNVL rank chunk, and are
    re-armed by this thread after the sum (rule 3 of the mailbox contract)."""
    chunk = oneshot_chunk(world)
    group = min(chunk, POLL_GROUP)
    if world % chunk or chunk % group:
        raise ValueError(f"world {world}: chunk {chunk} / poll group {group} do not divide")
    zero = zero_f32()
    acc = [zero] * 8
    part = [zero] * 8
    for first in range(0, world, group):
        if first % chunk == 0:
            part = [zero] * 8
        slots = poll_slots_v4_if(valid, addr + first * stride, group, stride)
        for slot in slots:
            for j in range(4):
                lo, hi = unpack_bf16x2(slot[j])
                part[2 * j] = add_rn(part[2 * j], lo)
                part[2 * j + 1] = add_rn(part[2 * j + 1], hi)
        if (first + group) % chunk == 0:
            acc = [add_rn(acc[e], part[e]) for e in range(8)]
    rearm_slots_v4_if(valid, addr, world, stride)
    return [pack_bf16x2(acc[2 * j], acc[2 * j + 1]) for j in range(4)]


def twoshot_chunk(world: int) -> int:
    """Rank chunk of the MNNVL two-shot sum order (``WorldSize < 16 ? WorldSize : 16``)."""
    return world if world < 16 else 16


__all__ = [
    "FRAGMENT_BYTES",
    "FRAGMENT_ELEMS",
    "POLL_GROUP",
    "PUBLISH_MULTIMEM",
    "PUBLISH_ST",
    "SENTINEL",
    "SENTINEL_I32",
    "SPIN_LIMIT",
    "add_rn",
    "griddepcontrol_launch_dependents",
    "griddepcontrol_wait",
    "load_v4",
    "mailbox_store_v4",
    "multimem_store_v4",
    "oneshot_chunk",
    "pack_bf16x2",
    "poll_slots_v4",
    "poll_slots_v4_if",
    "poll_sum_slots_v4",
    "publish_v4",
    "rank_order_sum",
    "rearm_slots_v4",
    "rearm_slots_v4_if",
    "sanitize_bf16x2",
    "sanitize_v4",
    "store_v4",
    "twoshot_chunk",
    "unpack_bf16x2",
    "zero_f32",
]
