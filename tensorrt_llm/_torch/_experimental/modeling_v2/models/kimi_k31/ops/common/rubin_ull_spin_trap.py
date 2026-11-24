# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The time bound of the K3.1 ULL target's device waits (``kernels/common/rubin_ull_spin_trap.py``)
on one flag word (``kernels/common/rubin_ull_spin_trap_wait.py``): a wait returns once its flag is
published; past the bound the waiting thread writes the process's trap record (mapped pinned host
memory) and traps instead of hanging."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.common import rubin_ull_spin_trap as _spin_trap
from ...kernels.common import rubin_ull_spin_trap_wait as _wait


def _cell(why: str, mode: str, publish: str, threads: int = 32) -> Cell:
    return Cell(why=why, spec=dict(mode=mode, publish=publish, threads=threads))


class SpinTrapWait(OpWrapper):
    """Wait on ``flag`` (an int32 CUDA word) with the target's bound, on the current stream.

    ``mode="tick"``: every thread of one ``threads``-thread CTA spins ``while flag == 0`` with
    ``spin_trap.tick``; past ``spin_trap.BOUND_NS`` of spinning it records ``SITE_TICK`` (a / b =
    thread / flag) and traps. ``mode="expired"``: every thread reads ``flag`` once and, if it is
    nonzero (a wait with its own bound that ran out), records ``SITE_EXPIRED`` and traps;
    ``mode="expired_at"``: the same through ``spin_trap.expired_at`` with the runtime site
    ``SITE_EXPIRED_AT + flag - 1``. Writes nothing on the way that returns; returns None. The
    record is ``spin_trap.record_address()``'s (``spin_trap.recorded()`` reads it on the host
    after the CUDA error).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("tick: flag published before the wait", "tick", "before"),
        _cell("tick: 256 threads, flag published before the wait", "tick", "before", 256),
        _cell("tick: flag published by another stream while the kernel spins", "tick", "late"),
        _cell("tick: never published, traps past the bound with the record", "tick", "never"),
        _cell("expired: zero flag (the wait made it), no trap", "expired", "before"),
        _cell(
            "expired: nonzero flag (the wait ran out), traps with the record", "expired", "never"
        ),
        _cell("expired_at: zero flag, no trap", "expired_at", "before"),
        _cell("expired_at: flag 2, traps with the runtime site", "expired_at", "never"),
    )

    note = """
    The bound the K3.1 ULL target's waits inline (``kernels/common/rubin_ull_spin_trap.py``,
    derived from r10 7da6f70b27): the fused KDA head's F_A waits use ``tick`` (sites 211 / 212);
    the DSA query mailbox poll (181) and DcpValue's LSE / partial-row polls (191-194) keep their
    own bound (``common.spin``) and call ``expired`` when it runs out, as do the key-shard waits
    (171 / 172, and 173 / 174 through one ``expired_at`` trap point). The target calls the kernels
    that inline them, not this one; this entry certifies the bound and the record on their
    behalf. A trap ends the CUDA context: the trapping cells run in a child process.
    """

    def __call__(self, flag: torch.Tensor, mode: str = "tick", threads: int = 32) -> None:
        import cutlass

        wait = _wait.compile_flag_wait(threads, mode, _spin_trap.record_address())
        wait(cutlass.Int64(flag.data_ptr()), _wait.stream(flag.device))

    def reference(self, flag: torch.Tensor, mode: str = "tick", threads: int = 32) -> None:
        """Nothing is computed: the op returns once ``flag`` is nonzero (``tick``) or at once
        when it is zero (``expired``), and leaves ``flag`` and the trap record unchanged."""
        return None

    def is_valid(self, flag: torch.Tensor, mode: str = "tick", threads: int = 32) -> None:
        """Raise where the kernel would read another word or not compile."""
        if flag.dtype != torch.int32 or not flag.is_cuda or flag.numel() < 1:
            raise ValueError("spin_trap_wait: flag must be an int32 CUDA word")
        if mode not in _wait.MODES:
            raise ValueError(f"spin_trap_wait: mode {mode!r} is not one of {_wait.MODES}")
        if not 1 <= threads <= 1024 or threads % 32:
            raise ValueError("spin_trap_wait: threads must be whole warps, at most 1024")


spin_trap_wait = SpinTrapWait()
