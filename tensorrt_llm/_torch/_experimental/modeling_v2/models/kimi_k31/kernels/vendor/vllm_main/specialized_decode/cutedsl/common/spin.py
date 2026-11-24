# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded cross-rank mailbox polls (TensorRT-LLM local module of the vendored tree).

``poll_words`` replaces an unbounded ``while not published(words): words = load_volatile(...)``
loop. It takes the first load's words; if none is the sentinel it returns them at once (no timer
read on the fast path), else it reloads with the same volatile vector load until no word is the
sentinel or ``%globaltimer`` has advanced ``SPIN_NS`` since the first miss. It returns the last
words and ``late`` (1 on a timeout). ``count_late`` adds 1 to a device counter for a timed-out
poll (relaxed GPU-scope reduction), so a stuck peer shows up in the host's step statistics
instead of hanging the step. ``refresh_words`` reloads a first load once without waiting, so
the reloads of several slots overlap before their polls.

The bound must never fire in a healthy run: a timed-out reader proceeds with what it has and
re-arms the slot, so a publish landing afterwards would be taken for the next use's. The
deadline (2^26 us, about 67 s) is the key-shard exchange's poll deadline, longer than any compile
or host stall between two ranks' launches.
"""

from cutlass import Float32, Uint32

from .primitives import _vector, asm

SPIN_NS = (1 << 26) * 1000


@asm()
def poll_words(pointer, words, sentinel, limit_ns=SPIN_NS):
    """``(*words, late)``: ``words`` (a first volatile load of ``len(words)`` 32-bit words at
    global ``pointer``) once none is ``sentinel``, reloading for at most ``limit_ns``."""
    n = len(words)
    sent = f"{int(sentinel) & 0xFFFFFFFF:#x}"
    regs = [f"w{i}" for i in range(n)]
    vector, _ = _vector(n)
    loaded = "{" + ",".join(regs) + "}" if n > 1 else regs[0]
    check = f"setp.ne.u32 p, w0, {sent}; "
    for r in regs[1:]:
        check += f"setp.ne.and.u32 p, {r}, {sent}, p; "
    first = n + 1  # operands: n + 1 outputs, then the words, then the pointer
    code = "{ .reg .pred p, q; .reg .u64 t0, t; .reg .b32 late, " + ", ".join(regs) + "; "
    code += "".join(f"mov.b32 w{i}, ${first + i}; " for i in range(n))
    code += "mov.b32 late, 0; " + check + "@p bra POLL_DONE_${:uid}; "
    code += "mov.u64 t0, %globaltimer; "
    code += f"POLL_LOOP_${{:uid}}: ld.volatile.global{vector}.u32 {loaded}, [${first + n}]; "
    code += check + "@p bra POLL_DONE_${:uid}; "
    code += f"mov.u64 t, %globaltimer; sub.u64 t, t, t0; setp.lt.u64 q, t, {int(limit_ns)}; "
    code += "@q bra POLL_LOOP_${:uid}; mov.b32 late, 1; "
    code += "POLL_DONE_${:uid}: "
    code += "".join(f"mov.b32 ${i}, w{i}; " for i in range(n)) + f"mov.b32 ${n}, late; }}"
    constraints = ",".join(["=r"] * (n + 1) + ["r"] * n + ["l", "~{memory}"])
    operands = (*(Uint32(w) for w in words), pointer)
    return code, constraints, operands, (Uint32,) * (n + 1)


@asm()
def count_late(counter, late):
    """``*counter += 1`` (u32, relaxed GPU scope) if ``late``."""
    code = "{ .reg .pred p; setp.ne.u32 p, $1, 0; @p red.relaxed.gpu.global.add.u32 [$0], 1; }"
    return code, "l,r,~{memory}", (counter, Uint32(late))



@asm()
def refresh_words(pointer, words, sentinel, weight):
    """``words`` (a first volatile load of ``len(words)`` 32-bit words at global ``pointer``),
    reloaded once by the same volatile vector load if ``weight`` >= 0 and a word is ``sentinel``.
    Nothing here waits for the reload: called for many slots in a row, the reloads of the slots
    still unpublished at their first load are in flight together, and ``poll_words`` on each then
    starts from words at most one round trip old."""
    n = len(words)
    sent = f"{int(sentinel) & 0xFFFFFFFF:#x}"
    regs = [f"w{i}" for i in range(n)]
    vector, _ = _vector(n)
    loaded = "{" + ",".join(regs) + "}" if n > 1 else regs[0]
    code = "{ .reg .pred p, q; .reg .b32 " + ", ".join(regs) + "; "
    code += "".join(f"mov.b32 w{i}, ${n + i}; " for i in range(n))
    code += f"setp.eq.u32 p, w0, {sent}; "
    code += "".join(f"setp.eq.or.u32 p, {r}, {sent}, p; " for r in regs[1:])
    code += f"setp.ge.f32 q, ${2 * n}, 0f00000000; and.pred p, p, q; "
    code += f"@p ld.volatile.global{vector}.u32 {loaded}, [${2 * n + 1}]; "
    code += "".join(f"mov.b32 ${i}, w{i}; " for i in range(n)) + "}"
    constraints = ",".join(["=r"] * n + ["r"] * n + ["f", "l", "~{memory}"])
    operands = (*(Uint32(w) for w in words), Float32(weight), pointer)
    return code, constraints, operands, (Uint32,) * n
