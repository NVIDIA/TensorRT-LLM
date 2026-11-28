# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The adaptive load balancer's route in a MegaMoE dispatch router.

A router built with ``adaptive_route_experts`` > 0 reads the staged picks as expert ids of that many experts: it
counts the picks of the forward's live rows into the layer's statistics row while the recording flag is set, and maps
each pick to a slot through the layer's packed route table row (``counts | starts`` per expert, ``slots |
cumulative_shares`` per slot, at a fixed device address, so a captured CUDA graph follows an in-place table update).
The metadata-push router mixes this in. A replicated expert ``e``'s pick in bucket row ``t`` on
EP rank ``r`` takes copy ``k`` when ``p = e * K_EXPERT + r * K_RANK + t * K_TOKEN mod 2^32`` lies in
``[c[k-1], c[k])`` of the expert's cumulative shares (``c[-1] = 0``; a bound of ``2^32 - 1`` and the last copy reach
``2^32``): each copy takes exactly its Q32 share of the points, a share of 0 none, a share of 1 all.

The live-row word (int32) says which of the launch's token rows are live: bits 0-15 ``real``, bits 16-31 ``requests``.
``requests`` = 0 counts no row; ``real`` < ``requests`` with token rows a multiple of ``requests`` counts the first
``rows / requests * real`` rows (padding requests at the tail of the bucket); anything else counts every row. It is
the framework's per-forward word (TensorRT-LLM's ``live_rows.py``, whose route staging drops rows by the same rule),
and only the statistics read it: a word outside the rule counts the padding too and never changes a route. The
statistics row holds one float32 per expert and the forward count one int64; both only accumulate. A float32 entry
is exact below 2^24 and a launch adds at most ``stats_tokens`` to it (``stats_tokens * topk`` if a row repeats an
id), so the caller zeroes the row before the ``stats_tokens`` of the launches since the last zeroing reach 2^24.

The table is read at execution time, also under CUDA graph replay; it has no version, and the helper READY gate does
not cover the slots it names. The caller orders a table switch (docs/DYNAMIC_LOAD_BALANCE.md, Adaptive route):
writes are stream-ordered between the launches that read the table; slot ``s``'s weights are complete on the rank that
owns them before any rank's table names ``s``, and stay until no launch on any rank still reads a table naming ``s``;
the row passes :func:`validate_adaptive_route_table`. tests/adaptive_route_reference.py is the plain-integer
definition of the route.
"""

from typing import ClassVar, Optional

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int32, Int64, T, dsl_user_op


@dsl_user_op
def _ld_nc_s32(address: Int64, *, loc=None, ip=None) -> Int32:
    """A read-only (non-coherent cache) load of words no launch of the forward writes: the route tables, the
    recording flag and the live-row word change only between launches. No side effect, so it issues with the
    loads around it instead of in program order."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [address.ir_value()],
            "ld.global.nc.b32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _red_add_f32(address: Int64, value: cutlass.Float32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [address.ir_value(), value.ir_value()],
        "red.relaxed.gpu.global.add.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _red_add_u64(address: Int64, value: Int64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [address.ir_value(), value.ir_value()],
        "red.relaxed.gpu.global.add.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


class AdaptiveRouteMixin:
    """Device code of the adaptive route, mixed into the MegaMoE dispatch routers.

    Attributes:
        adaptive_route_experts: Experts whose picks the router maps and counts, set by the router; 0 builds no
            adaptive route.
        expert_count: The slots the picks map to, set by the router.
    """

    # The multipliers of the point that picks a replicated expert's copy (K_EXPERT, K_RANK, K_TOKEN above).
    adaptive_expert_multiplier: ClassVar[int] = 0x632BE5AB
    adaptive_rank_multiplier: ClassVar[int] = 0x85EBCA6B
    adaptive_token_multiplier: ClassVar[int] = 0x9E3779B9

    @cute.jit
    def _adaptive_counted_rows(self, adaptive: tuple, cta_idx: Int32, thread_idx: Int32) -> Int32:
        """The token rows whose picks this launch counts: the live rows of the forward's rows by its live-row
        word (module docstring), none while the recording flag is clear. Thread 0 of CTA 0 also
        adds the forward to the forward count on the forward's first call."""
        _, _, forwards, stats_enabled, word_address, stats_tokens, stats_first = adaptive
        word = _ld_nc_s32(word_address)
        recording = _ld_nc_s32(stats_enabled)
        real = word & Int32(0xFFFF)
        requests = (word >> Int32(16)) & Int32(0xFFFF)
        counted_rows = stats_tokens
        if requests == Int32(0):
            counted_rows = Int32(0)
        elif real < requests:
            if stats_tokens % requests == Int32(0):
                counted_rows = stats_tokens // requests * real
        if recording == Int32(0):
            counted_rows = Int32(0)
        elif cta_idx == Int32(0):
            if thread_idx == Int32(0):
                if stats_first != Int32(0):
                    _red_add_u64(forwards, Int64(1))
        return counted_rows

    @cute.jit
    def _adaptive_pick(
        self,
        adaptive: tuple,
        counted_rows: Int32,
        expert: Int32,
        token: Int32,
        ep_rank: Int32,
        staged: Optional[cute.Tensor] = None,
    ) -> Int32:
        """One pick of ``expert`` (a valid id) in token row ``token``: counted into the statistics row when the
        row is counted, and routed to its slot (-1 for a malformed table entry); ``staged``: the table's leading
        entries in shared memory (:meth:`_adaptive_table_load`)."""
        tables, rows = adaptive[0], adaptive[1]
        if token < counted_rows:
            _red_add_f32(rows + Int64(expert) * Int64(4), cutlass.Float32(1.0))
        return self._adaptive_slot(tables, expert, token, ep_rank, staged)

    @cute.jit
    def _adaptive_table_load(self, tables: Int64, thread_idx: Int32, threads: int) -> cute.Tensor:
        """This thread's share of the table's ``counts | starts | slots`` (one load per entry a CTA's threads
        split), in registers: the CTA stores them to shared memory (:meth:`_adaptive_table_store`) once, so a
        pick reads them there instead of chaining dependent global loads."""
        entries = cute.make_rmem_tensor((self._adaptive_table_rounds(threads),), Int32)
        for round_ in cutlass.range_constexpr(self._adaptive_table_rounds(threads)):
            index = Int32(round_ * threads) + thread_idx
            if index < Int32(self.adaptive_table_entries):
                entries[round_] = _ld_nc_s32(tables + Int64(index) * Int64(4))
        return entries

    @cute.jit
    def _adaptive_table_store(
        self, entries: cute.Tensor, staged: cute.Tensor, thread_idx: Int32, threads: int
    ) -> None:
        for round_ in cutlass.range_constexpr(self._adaptive_table_rounds(threads)):
            index = Int32(round_ * threads) + thread_idx
            if index < Int32(self.adaptive_table_entries):
                staged[index] = entries[round_]

    @property
    def adaptive_table_entries(self) -> int:
        """Entries of the table a pick reads unconditionally: ``counts | starts`` per expert, ``slots``."""
        return 2 * self.adaptive_route_experts + self.expert_count

    def _adaptive_table_rounds(self, threads: int) -> int:
        return (self.adaptive_table_entries + threads - 1) // threads

    @cute.jit
    def _adaptive_slot(
        self,
        tables: Int64,
        expert: Int32,
        token: Int32,
        ep_rank: Int32,
        staged: Optional[cute.Tensor] = None,
    ) -> Int32:
        """The slot of a pick of ``expert`` (a valid id) in token row ``token``: its only copy, or for a replicated
        expert the copy whose cumulative-share range holds the pick's point (module docstring), -1
        for a malformed entry. Tables: ``counts | starts`` per expert, ``slots | cumulative_shares`` per slot;
        with ``staged``, all but the shares are read from it."""
        experts = self.adaptive_route_experts
        slot_count = self.expert_count
        if cutlass.const_expr(staged is not None):
            copies = staged[expert]
            start = staged[Int32(experts) + expert]
        else:
            copies = _ld_nc_s32(tables + Int64(expert) * Int64(4))
            start = _ld_nc_s32(tables + (Int64(experts) + Int64(expert)) * Int64(4))
        slots_base = tables + Int64(2 * experts) * Int64(4)
        shares_base = tables + Int64(2 * experts + slot_count) * Int64(4)
        slot = Int32(-1)
        # No sum of two loaded words: start + copies wraps in Int32 (start = 2^31 - 1 and copies = 1 would pass).
        if (
            copies > Int32(0)
            and copies <= Int32(slot_count)
            and start >= Int32(0)
            and start <= Int32(slot_count) - copies
        ):
            k = Int32(0)
            if copies > Int32(1):
                point = (
                    cutlass.Uint32(expert) * cutlass.Uint32(self.adaptive_expert_multiplier)
                    + cutlass.Uint32(ep_rank) * cutlass.Uint32(self.adaptive_rank_multiplier)
                    + cutlass.Uint32(token) * cutlass.Uint32(self.adaptive_token_multiplier)
                )
                searching = cutlass.Boolean(True)
                while searching:
                    searching = cutlass.Boolean(False)
                    if k < copies - Int32(1):
                        bound = cutlass.Uint32(
                            _ld_nc_s32(shares_base + Int64(start + k) * Int64(4))
                        )
                        # A bound of 2^32 - 1 ends the range: no point passes it.
                        if bound != cutlass.Uint32(0xFFFFFFFF):
                            if point >= bound:
                                k = k + Int32(1)
                                searching = cutlass.Boolean(True)
            if cutlass.const_expr(staged is not None):
                slot = staged[Int32(2 * experts) + start + k]
            else:
                slot = _ld_nc_s32(slots_base + Int64(start + k) * Int64(4))
        return slot


def adaptive_route_buffers(experts: int, slots: int) -> dict:
    """The device buffers behind the address operands of one layer's adaptive route, ``{operand: (dtype,
    entries)}``; each is contiguous and aligned to its element size. ``experts`` is ``adaptive_route_experts``
    and ``slots`` the kernel's physical slot count (``expert_count`` after any helper slots are added), not
    necessarily the expert count the caller passed."""
    return {
        "tables": ("int32", 2 * experts + 2 * slots),
        "stats": ("float32", experts),
        "forwards": ("int64", 1),
        "stats_enabled": ("int32", 1),
        "live_word": ("int32", 1),
    }


def validate_adaptive_route_table(row, experts: int, slots: int) -> None:
    """Raise ValueError unless ``row`` (``counts | starts`` per expert, ``slots | cumulative_shares`` per slot; int32
    values, e.g. a tensor's ``tolist()``) is a table the router routes as written: every expert has a copy, its run
    ``[start, start + copies)`` lies inside the slots, every routed slot is in ``[0, slots)`` and under one expert
    only (so no two runs overlap), and an expert's cumulative shares (uint32) do not decrease.

    The router only keeps its reads in bounds. A row outside this contract can route two picks of one token to one
    slot, which the destination pool is not sized for, and a pick it drops for a malformed entry is still summed by
    the top-k reduce, which sees only the expert id."""
    row = [int(value) for value in row]
    if len(row) != 2 * experts + 2 * slots:
        raise ValueError(f"A route table row has 2 * {experts} + 2 * {slots} entries, got {len(row)}.")
    counts, starts = row[:experts], row[experts:2 * experts]
    slot_ids = row[2 * experts:2 * experts + slots]
    bounds = [value & 0xFFFFFFFF for value in row[2 * experts + slots:]]
    slot_owner = {}
    for expert, (copies, start) in enumerate(zip(counts, starts)):
        if copies <= 0 or start < 0 or start + copies > slots:
            raise ValueError(f"Expert {expert}: {copies} copies from {start} is not a run inside the {slots} slots.")
        for slot in slot_ids[start:start + copies]:
            if not 0 <= slot < slots:
                raise ValueError(f"Expert {expert} routes to slot {slot}, outside [0, {slots}).")
            if slot in slot_owner:
                raise ValueError(f"Slot {slot} is under experts {slot_owner[slot]} and {expert}.")
            slot_owner[slot] = expert
        run = bounds[start:start + copies - 1]
        if any(later < earlier for earlier, later in zip(run, run[1:])):
            raise ValueError(f"Expert {expert}: the cumulative shares decrease.")
