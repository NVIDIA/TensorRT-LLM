# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A time bound for device spin waits: past it the waiting thread records the wait and traps.

A wait ``while cond: body`` becomes::

    spins, since = spin_trap.start()
    while cond:
        spins, since = spin_trap.tick(spins, since, SITE, RECORD, a, b, c)
        body

with ``RECORD = spin_trap.record_address()`` taken on the host before the kernel is compiled
(a compile-time constant). ``tick`` adds 1 to a spin counter and, once every ``CHECK_EVERY``
spins, reads ``%globaltimer`` (the first read starts the clock). Past ``BOUND_NS`` (lane 0 of a
warp; the other lanes after twice that) the thread writes one record into mapped pinned host
memory and traps, so a peer that never publishes ends the process with a CUDA error instead of
hanging it. ``expired(flag, SITE, RECORD, a, b, c)`` does the same for a wait with its own
bound that reports a nonzero ``flag`` when it ran out.

The record (``int64 [8]``) is claimed by the first trapping thread with a system-scope
compare-and-swap of word 0 (``site << 48 | block.x << 32 | thread.x``), which then stores the
site's three values ``a b c``, its spins, the ns waited and ``%globaltimer`` (volatile stores,
then ``fence.sc.sys``). Host memory stays readable after the trap kills the CUDA context:
``recorded()`` returns the record and ``log_trap()`` prints one ``K31_POLL_TIMEOUT`` line (once
per process). A daemon thread started with the record reads its claim word every
``WATCH_PERIOD_S`` (a host load from pinned memory: no CUDA call, no stream, nothing captured)
and logs the record as soon as a trap claims it, so the line is printed whatever the process
does next (a fatal-error path that ends in ``MPI_Abort`` / SIGKILL runs no atexit hook); the
hook is kept for a record that the watcher has not printed yet. No device printf: the trap path
needs no stack.
"""

import atexit
import os
import socket
import sys
import threading
import time

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from ..vendor.vllm_mr8 import module

_prim = module("cutedsl.common.primitives")

#: About 69 s: above any wait of a healthy step, including the rank skew of the eager warmup
#: (CuTe compiles), where a 1 s bound already trips cross-rank waits.
BOUND_NS = 1 << 36
#: Spins between two timer reads (a power of two).
CHECK_EVERY = 1024
#: The record's words (``recorded``).
FIELDS = ("claim", "a", "b", "c", "spins", "waited_ns", "gtimer")
#: Seconds between two reads of the claim word by the watcher thread: well inside the executor's
#: rank-crash grace (10 s) before it aborts every rank.
WATCH_PERIOD_S = 0.5

_record: torch.Tensor | None = None
_record_device_address = 0
_where = ""
_logged = threading.Event()
_log_lock = threading.Lock()


def record_address() -> int:
    """Device address of this process's trap record (mapped pinned host memory, allocated on
    first use; call before graph capture). The first call also starts the watcher thread."""
    global _record, _record_device_address, _where
    if _record is None:
        from cuda.bindings import runtime

        record = torch.zeros(8, dtype=torch.int64, pin_memory=True)
        err, address = runtime.cudaHostGetDevicePointer(record.data_ptr(), 0)
        if err != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaHostGetDevicePointer failed: {err}")
        _where = f"host={socket.gethostname()} pid={os.getpid()} device={torch.cuda.current_device()}"
        _record, _record_device_address = record, int(address)
        atexit.register(log_trap)
        threading.Thread(
            target=_watch, args=(record.numpy(),), name="k31-spin-trap-watch", daemon=True
        ).start()
    return _record_device_address


def _watch(words) -> None:
    """Log the record once a trap has claimed it (``words``: the record's host view)."""
    while not _logged.is_set():
        if int(words[0]) != 0:
            # The claimant stores the other words right after its claim, then fences.
            time.sleep(0.01)
            log_trap()
            return
        time.sleep(WATCH_PERIOD_S)


def recorded() -> dict | None:
    """The trap record (``FIELDS`` plus the decoded site / block / thread), or None."""
    if _record is None:
        return None
    words = [int(w) for w in _record.tolist()]
    if words[0] == 0:
        return None
    out = dict(zip(FIELDS, words))
    claim = words[0] & 0xFFFFFFFFFFFFFFFF
    out.update(site=claim >> 48, block=(claim >> 32) & 0xFFFF, thread=claim & 0xFFFFFFFF)
    return out


def log_trap() -> None:
    """Print the trap record as one ``K31_POLL_TIMEOUT`` line (nothing without a record; once
    per process: the watcher thread and the exit hook may both call it)."""
    with _log_lock:
        r = recorded()
        if r is None or _logged.is_set():
            return
        _logged.set()
        print(
            f"K31_POLL_TIMEOUT site={r['site']} block={r['block']} thread={r['thread']} "
            f"a={r['a']} b={r['b']} c={r['c']} spins={r['spins']} "
            f"waited_ns={r['waited_ns']} gtimer={r['gtimer']} {_where}",
            file=sys.stderr,
            flush=True,
        )


@_prim.asm()
def _record_and_trap(record, claim, a, b, c, spins, waited):
    code = (
        "{ .reg .b64 old, t; .reg .pred p; "
        "atom.global.sys.cas.b64 old, [$0], 0, $1; "
        "setp.eq.b64 p, old, 0; "
        "@p st.volatile.global.b64 [$0+8], $2; "
        "@p st.volatile.global.b64 [$0+16], $3; "
        "@p st.volatile.global.b64 [$0+24], $4; "
        "@p st.volatile.global.b64 [$0+32], $5; "
        "@p st.volatile.global.b64 [$0+40], $6; "
        "mov.u64 t, %globaltimer; "
        "@p st.volatile.global.b64 [$0+48], t; "
        "fence.sc.sys; "
        "trap; }"
    )
    operands = tuple(Int64(v) for v in (record, claim, a, b, c, spins, waited))
    return code, "l,l,l,l,l,l,l,~{memory}", operands


@dsl_user_op
def globaltimer(*, loc=None, ip=None):
    return Uint64(
        llvm.inline_asm(
            Uint64.mlir_type,
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def report(
    site: cutlass.Constexpr[int], record: cutlass.Constexpr[int], a, b, c, spins, waited
):
    """Write the trap record (first thread to claim it), then trap."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    claim = Int64(site << 48) + Int64(bidx) * Int64(1 << 32) + Int64(tidx)
    _record_and_trap(Int64(record), claim, Int64(a), Int64(b), Int64(c), Int64(spins), Int64(waited))


@cute.jit
def start():
    return Int32(0), Uint64(0)


@cute.jit
def tick(spins, since, site: cutlass.Constexpr[int], record: cutlass.Constexpr[int], a=0, b=0, c=0):
    spins = spins + 1
    if spins % CHECK_EVERY == 0:
        now = globaltimer()
        if since == Uint64(0):
            since = now
        tidx, _, _ = cute.arch.thread_idx()
        bound = Uint64(2 * BOUND_NS)
        if tidx % 32 == 0:
            bound = Uint64(BOUND_NS)
        if now - since > bound:
            report(site, record, a, b, c, spins, now - since)
    return spins, since


@cute.jit
def expired(flag, site: cutlass.Constexpr[int], record: cutlass.Constexpr[int], a=0, b=0, c=0):
    """``flag`` != 0 (a wait's own bound ran out): write the trap record, then trap."""
    if flag != 0:
        report(site, record, a, b, c, Int32(-1), Uint64(0))


@cute.jit
def expired_at(flag, site, record: cutlass.Constexpr[int], a=0, b=0, c=0):
    """``expired`` with a runtime ``site``: one trap point for several waits of a kernel, the
    record naming the one that ran out."""
    if flag != 0:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        claim = Int64(site) * Int64(1 << 48) + Int64(bidx) * Int64(1 << 32) + Int64(tidx)
        _record_and_trap(Int64(record), claim, Int64(a), Int64(b), Int64(c), Int64(-1), Int64(0))
