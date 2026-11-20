# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The adaptive load balancer's route in a MegaMoE dispatch router.

A router built with ``adaptive_route_experts`` > 0 reads the staged picks as expert ids of that many experts: it
counts the picks of the forward's live rows into the layer's statistics row while the recording flag is set, and maps
each pick to a slot through the layer's packed route table row (``counts | starts`` per expert, ``slots |
cumulative_shares`` per slot, at a fixed device address). Both dispatch routers mix this in, so the routes and the
statistics are the same whichever router a launch bucket uses. A replicated expert ``e``'s pick in bucket row ``t`` on
EP rank ``r`` takes copy ``k`` when ``p = e * K_EXPERT + r * K_RANK + t * K_TOKEN mod 2^32`` lies in
``[c[k-1], c[k])`` of the expert's cumulative shares (``c[-1] = 0``; a bound of ``2^32 - 1`` and the last copy reach
``2^32``): each copy takes exactly its Q32 share of the points, a share of 0 none, a share of 1 all.
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
        word (adaptiveEplbKernels.cu liveIdCount), none while the recording flag is clear. Thread 0 of CTA 0 also
        adds the forward to the forward count on the forward's first call."""
        _, _, forwards, enabled, word_address, num_tokens, first = adaptive
        word = _ld_nc_s32(word_address)
        recording = _ld_nc_s32(enabled)
        real = word & Int32(0xFFFF)
        requests = (word >> Int32(16)) & Int32(0xFFFF)
        counted_rows = num_tokens
        if requests == Int32(0):
            counted_rows = Int32(0)
        elif real < requests:
            if num_tokens % requests == Int32(0):
                counted_rows = num_tokens // requests * real
        if recording == Int32(0):
            counted_rows = Int32(0)
        elif cta_idx == Int32(0):
            if thread_idx == Int32(0):
                if first != Int32(0):
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
        if copies > Int32(0) and start >= Int32(0) and start + copies <= Int32(slot_count):
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
