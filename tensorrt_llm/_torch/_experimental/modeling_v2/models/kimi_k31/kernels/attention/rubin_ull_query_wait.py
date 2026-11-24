# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-project/vllm@e3ddbdbd60 vllm/models/kimi_k3/nvidia/specialized_decode/cutedsl/dsa/overlap.py
# (QueryWait); modified by NVIDIA: release_after_poll, bounded polls with spin_trap.
"""The DSA query mailbox poll as its own small kernel (vLLM ``dsa.overlap.QueryWait``).

Polls every 16-byte chunk of a gathered FP8 query mailbox (``chunks`` chunks) with volatile
loads until none is the sentinel (``0xFFFFFFFF`` words: four FP8 0xFF bytes, never published);
at most 32 CTAs of 256 threads, each thread up to 8 chunks in flight. It first waits for the
kernels before it on the stream (PDL), then releases its dependents: at once (the vendored
kernel's), or with ``release_after_poll`` only after every chunk of the CTA was seen published.

``release_after_poll`` is for a dependent that reads the mailbox without ``griddepcontrol.wait``
(the DSA stack's speculative decode): the join of the query side
into the decode's stream is a programmatic edge (the decode launches with PDL), so with the early
release the decode can start while this kernel still polls, read rows not yet published, and
re-arm them (its end-of-kernel sentinel stores) before a late publish is seen here: the poll then
times out (site 181). Released after the polls, the decode starts once every chunk was seen on
every CTA (a CTA barrier, then the trigger), so its reads see the published rows and its re-arm
comes after the last poll.

Each poll is bounded (``common.spin.poll_words``: ``SPIN_NS``, about 67 s, after its first
miss): a chunk still unpublished then adds 1 to ``spins[SPIN_QUERY]`` and, with a ``record``
(``kernels.spin_trap.record_address()``), writes the trap record (site ``SITE_QUERY``: CTA,
thread, chunk step) and traps instead of reading a stale query; without one it moves on.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute

from ..vendor.vllm_mr8 import module
from ..vendor.vllm_main import module as main_module
from ..common import rubin_ull_spin_trap as spin_trap

_prim = module("cutedsl.common.primitives")
_spin = main_module("cutedsl.common.spin")
Q_SENT = module("cutedsl.mla.query").Q_SENT
#: ``spins`` slot of the query mailbox polls (DcpValue's polls use slots 1 and 2).
SPIN_QUERY = 0
#: ``spin_trap`` site of a timed-out query mailbox poll.
SITE_QUERY = 181


class QueryWait:
    def __init__(self, chunks: int, record: int = 0, release_after_poll: bool = False):
        self.chunks = chunks
        self.record = record
        self.release_after_poll = release_after_poll
        self.ctas = min(32, -(-chunks // (256 * 8)))

    @cute.jit
    def __call__(self, mailbox: cute.Tensor, spins: cute.Tensor, stream):
        self.kernel(mailbox, spins).launch(
            grid=(self.ctas, 1, 1), block=(256, 1, 1), stream=stream, use_pdl=True
        )

    @cute.kernel
    def kernel(self, mailbox: cute.Tensor, spins: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(not self.release_after_poll):
            cute.arch.griddepcontrol_launch_dependents()
        threads: cutlass.Constexpr = self.ctas * 256
        chunks: cutlass.Constexpr = self.chunks
        first = cta * 256 + tid
        last = chunks - 1  # threads past the end re-poll it
        steps: cutlass.Constexpr = -(-chunks // threads)
        counter = spins.iterator + SPIN_QUERY
        for i0 in cutlass.range_constexpr(0, steps, 8):
            n: cutlass.Constexpr = min(8, steps - i0)
            sources = [
                mailbox.iterator + min(first + (i0 + i) * threads, last) * 16 for i in range(n)
            ]
            loads = [_prim.load_volatile(source, count=4) for source in sources]
            for i in cutlass.range_constexpr(n):  # the loads above are in flight
                *_, late = _spin.poll_words(sources[i], loads[i], Q_SENT)
                _spin.count_late(counter, late)
                if cutlass.const_expr(self.record != 0):
                    spin_trap.expired(late, SITE_QUERY, self.record, cta, tid, i0 + i)
        if cutlass.const_expr(self.release_after_poll):
            # Every thread of the CTA saw its chunks published, then the CTA releases.
            cute.arch.sync_threads()
            cute.arch.griddepcontrol_launch_dependents()
