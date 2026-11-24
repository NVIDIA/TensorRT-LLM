# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The q mailbox polled by validity-reporting copies (Rubin).

``cp.async.bulk{.tensor} ... .mbarrier::report::validity::per_16bytes::80000000``
copies global memory into shared memory like any bulk / tensor copy, and the
layout::v1 mbarrier it completes also reports whether any copied 16 B chunk was
still armed. Only the FIRST 32-bit word of each 16 B chunk is checked against
0x80000000 (measured, also for local data).

The writer contract (``MlaQuery(valid=True)``; checked by a delayed-store stress test
and a torn-writer control):

- every 16 B chunk of a q row is written with ONE aligned 16 B store (multicast);
  never two 8 B halves or four 4 B words: a chunk whose first word has landed passes
  the report while the rest may still be armed (the torn-writer control trips on
  every hop);
- no chunk's first word is 0x80000000 (``off_sentinel``: FP8 (+0, +0, +0, -0) ->
  four +0, the same values).

The arm pattern (``ARMED``): each 16 B chunk holds (0x80000000, Q_SENT x 3), so the
LSU pollers that check every word against Q_SENT (0xFFFFFFFF: FP8 bytes are never
0xFF) keep working on the same mailbox, and a validity copy sees the first word.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32

from ..common.primitives import asm

VALID_SENT = 0x80000000
Q_SENT = 0xFFFFFFFF  # mla.dcp.Q_SENT
ARMED = (VALID_SENT, Q_SENT, Q_SENT, Q_SENT)  # one 16 B chunk of an armed q row


@asm(Uint32, pure=True)
def off_sentinel(word):
    """``word``, or 0 when it is 0x80000000 (the first word of a q chunk)."""
    code = "{ .reg .pred p; setp.eq.u32 p,$1,0x80000000; selp.b32 $0,0,$1,p; }"
    return code, "=r,r", (Uint32(word),)


@asm()
def push_words(pointer, words, barrier, peer):
    """DSMEM store of 2 or 4 words into cluster CTA ``peer``'s shared ``pointer``,
    completing on its ``barrier`` (``st.async``)."""
    n = len(words)
    registers = "{" + ",".join(f"${1 + i}" for i in range(n)) + "}"
    b, p = f"${1 + n}", f"${2 + n}"
    code = f"{{ .reg .b32 dst,bar; mapa.shared::cluster.u32 dst,$0,{p}; "
    code += f"mapa.shared::cluster.u32 bar,{b},{p}; "
    code += f"st.async.shared::cluster.mbarrier::complete_tx::bytes.v{n}.b32 "
    code += f"[dst],{registers},[bar]; }}"
    constraints = "r" + ",r" * n + ",r,r,~{memory}"
    return code, constraints, (pointer, *map(Uint32, words), barrier, Int32(peer))


@asm()
def init_valid(barrier):
    """layout::v1 mbarrier with one arrival (the caller fences the init)."""
    return (
        "mbarrier.init.layout::v1.shared::cta.b64 [$0], 1;",
        "r,~{memory}",
        (barrier,),
    )


@asm()
def expect_valid(barrier, nbytes):
    code = "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;"
    return code, "r,r,~{memory}", (barrier, Int32(nbytes))


@asm()
def copy_valid(dst, src, nbytes, barrier):
    """Bulk copy of ``nbytes`` (a multiple of 16) from global ``src`` into this CTA's
    shared ``dst``, reporting armed 16 B chunks on the layout::v1 ``barrier``."""
    code = "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
    code += ".mbarrier::report::validity::per_16bytes::80000000 [$0], [$1], $2, [$3];"
    return code, "r,l,r,r,~{memory}", (dst, src, Int32(nbytes), barrier)


@asm()
def tma_valid3(dst, descriptor, column, row, depth, barrier):
    """3-D tensor TMA box at (``column``, ``row``, ``depth``) into this CTA's shared
    ``dst`` (the descriptor's swizzle), reporting armed 16 B chunks on the layout::v1
    ``barrier``; out-of-bounds rows are zero-filled and report clean."""
    code = "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
    code += ".mbarrier::complete_tx::bytes"
    code += ".mbarrier::report::validity::per_16bytes::80000000"
    code += " [$0], [$1, {$2, $3, $4}], [$5];"
    operands = (
        dst,
        Int64(descriptor),
        Int32(column),
        Int32(row),
        Int32(depth),
        barrier,
    )
    return code, "r,l,r,r,r,r,~{memory}", operands


@asm()
def signal_word(pointer, barrier, peer):
    """One 4 B DSMEM store into cluster CTA ``peer`` completing 4 bytes on its
    ``barrier`` (st.async: its complete-tx is a release at cluster scope)."""
    code = "{ .reg .b32 dst,bar,one; mapa.shared::cluster.u32 dst,$0,$2; "
    code += "mapa.shared::cluster.u32 bar,$1,$2; mov.b32 one,1; "
    code += (
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [dst],one,[bar]; }"
    )
    return code, "r,r,r,~{memory}", (pointer, barrier, Int32(peer))


@asm()
def spin_ns(nanoseconds):
    """Spin on %globaltimer for ``nanoseconds`` (no memory traffic: a reader that
    polls while the rows land slows their landing)."""
    code = "{ .reg .u64 t0, t1; .reg .pred p; mov.u64 t0, %globaltimer; SPIN_NS: "
    code += "mov.u64 t1, %globaltimer; sub.u64 t1, t1, t0; setp.lt.u64 p, t1, $0; "
    code += "@p bra SPIN_NS; }"
    return code, "l,~{memory}", (Int64(nanoseconds),)


@asm((Int32, Int32))
def try_wait_valid(barrier, phase):
    """(phase complete, some copied chunk was armed) of a layout::v1 barrier."""
    code = "{ .reg .pred p, q; mbarrier.try_wait.parity.phase_type::primary.shared::cta"
    code += ".b64 p|q, [$2], $3; selp.u32 $0, 1, 0, p; selp.u32 $1, 1, 0, q; }"
    return code, "=r,=r,r,r,~{memory}", (barrier, Int32(phase))


@cute.jit
def wait_valid(barrier, phase):
    """Wait for ``phase`` of a layout::v1 barrier; nonzero if a chunk was armed."""
    done, armed = try_wait_valid(barrier, phase)
    while done == 0:
        done, armed = try_wait_valid(barrier, phase)
    return armed


@cute.jit
def poll_valid(barrier, phase, nbytes: cutlass.Constexpr[int], issue):
    """One thread: rounds of ``issue(barrier)`` (bulk copies of ``nbytes`` in all)
    until no copied chunk was armed; every copy of a round is re-issued (the data
    may land between the copies). Returns the next phase."""
    armed = Int32(1)
    while armed != 0:
        expect_valid(barrier, nbytes)
        issue(barrier)
        armed = wait_valid(barrier, phase)
        phase = phase ^ 1
    return phase
