# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DSA query mailbox poll (``kernels/attention/rubin_ull_query_wait.py`` ``QueryWait``, after vLLM's
``dsa.overlap.QueryWait``): returns once every 16-byte chunk of the gathered FP8 query rows has been
published by its rank."""

from __future__ import annotations

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Mailbox row bytes: 96 query heads x 576 FP8 bytes; a chunk is 16 bytes.
ROW_BYTES = 96 * 576
#: The never-published word: four FP8 0xFF bytes (``mla/query.py`` ``Q_SENT``).
SENTINEL = 0xFF


def _cell(why: str, rows: int, late: bool = False) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, late=late))


class DsaQueryWait(OpWrapper):
    """Poll ``mailbox`` (the gathered query rows of one parity, ``[rows, 96, 576]`` uint8, as the
    compiled kernel's CuTe view) until no 32-bit word of it is ``0xFFFFFFFF``; nothing is written
    unless a poll times out (``common.spin.SPIN_NS`` after its first miss), which adds 1 to
    ``spins[0]``, writes the trap record and traps. ``wait``: ``QueryWait(rows * 96 * 36,
    spin_trap.record_address())`` compiled for this mailbox view (``modeling.query_wait_kernel``);
    ``spins``: the int32 [4] timeout counters' view; ``stream``:
    the current stream's CUstream. Waits for the kernels before it (PDL), then releases its
    dependents. Returns None.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        _cell("served b1: 8 rows, published before the wait", 8),
        _cell("served b4: 32 rows (several chunks a thread)", 32),
        _cell("q1 row", 1),
        _cell("8 rows published by another stream while the kernel polls", 8, late=True),
    )

    note = """
    The K3.1 ULL generation target's wait for its peers' query heads (the publish-only MLA query
    kernel multicasts each rank's 6 heads into every rank's mailbox), on the side stream after the
    FP8 append and before SparseDcpDecode reads the rows. Every CTA must be able to see every
    peer's stores: the publishers' grids run on other GPUs, so nothing here is resident-bound.
    The poll is bounded (``common/spin.py``, about 67 s): a timed-out chunk is counted, then
    writes the ``kernels/common/rubin_ull_spin_trap.py`` record (site 181: CTA, thread, chunk
    step) and traps instead of reading a stale query.
    """

    def __call__(self, wait, mailbox, spins, stream) -> None:
        wait(mailbox, spins, stream)

    def reference(self, wait, mailbox, spins, stream) -> None:
        """Nothing is computed: the op's effect is that it returns after every chunk is published
        (and that ``spins`` is unchanged when none timed out)."""
        return None


dsa_query_wait = DsaQueryWait()
