# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One bounded wait on a flag word, built only from ``rubin_ull_spin_trap``: the form every
bounded wait of the K3.1 ULL target takes, as a kernel of its own (``ops.common.spin_trap_wait``).

``mode="tick"``: every thread of one CTA waits ``while flag == 0`` (gpu-scope acquire loads) with
``spin_trap.tick`` (site ``SITE_TICK``, a / b = thread / flag), the shape of the fused KDA head's
F_A waits. ``mode="expired"``: every thread reads the flag once and, if it is nonzero, records it
with ``spin_trap.expired`` (site ``SITE_EXPIRED``, a / b = thread / flag) and traps, the shape of
a wait with its own bound that ran out. ``mode="expired_at"``: the same with ``spin_trap.expired_at``
and a runtime site, ``SITE_EXPIRED_AT + flag - 1`` (the key-shard exchange's one trap point for
two waits). Writes nothing else.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64

from ..vendor.vllm_mr8 import module
from . import rubin_ull_spin_trap as spin_trap

_acquire_load = module("cutedsl.kda.head")._acquire_load
stream = module("cutedsl.common.host").stream

#: ``spin_trap`` sites of the two modes (outside the target's 171-212).
SITE_TICK = 901
SITE_EXPIRED = 902
SITE_EXPIRED_AT = 903
MODES = ("tick", "expired", "expired_at")


class FlagWait:
    def __init__(self, threads: int, mode: str, record: int):
        if mode not in MODES:
            raise ValueError(f"FlagWait mode {mode!r}: one of {MODES}")
        self.threads, self.mode, self.record = threads, mode, record

    @cute.jit
    def __call__(self, flag: cutlass.Int64, stream):
        self.kernel(flag).launch(grid=(1, 1, 1), block=(self.threads, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, flag: cutlass.Int64):
        tid, _, _ = cute.arch.thread_idx()
        seen = Int32(_acquire_load(flag))
        if cutlass.const_expr(self.mode == "expired"):
            spin_trap.expired(seen, SITE_EXPIRED, self.record, tid, seen)
        elif cutlass.const_expr(self.mode == "expired_at"):
            site = Int64(SITE_EXPIRED_AT - 1) + Int64(seen)
            spin_trap.expired_at(seen, site, self.record, tid, seen)
        else:
            spins, since = spin_trap.start()
            while seen == 0:
                spins, since = spin_trap.tick(spins, since, SITE_TICK, self.record, tid, seen)
                seen = Int32(_acquire_load(flag))


@cache
def compile_flag_wait(threads: int, mode: str, record: int):
    """``FlagWait`` compiled for ``record`` (``spin_trap.record_address()``); call it with the flag
    word's device address (``cutlass.Int64``) and a CUstream."""
    from cutlass.cute.runtime import make_fake_stream

    return cute.compile(FlagWait(threads, mode, record), cutlass.Int64(0), make_fake_stream())
