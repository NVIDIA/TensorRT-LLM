# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rotating-lane delayed release helpers for done-counter publishing."""

import dataclasses
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int64

from .ptx_helpers import red_add_release_gpu_s32, red_add_release_sys_s32_raw


@dataclasses.dataclass(frozen=True)
class FlagBatchTrackerBase:
    """Loop-carried per-thread state for batched release-counter publishing."""

    flag_addr: Int64  # per-lane counter-slot address (0 == null)
    cumulated_flags: cutlass.Int32  # current batch fill count (uniform)
    phase: cutlass.Int32  # current accumulated phase (uniform)
    tid: cutlass.Int32  # lane/thread id within the rotating group

    @cute.jit
    def _make(
        self,
        flag_addr: Int64,
        cumulated_flags: cutlass.Int32,
        phase: cutlass.Int32,
    ) -> "FlagBatchTrackerBase":
        return FlagBatchTrackerBase(
            flag_addr=flag_addr,
            cumulated_flags=cumulated_flags,
            phase=phase,
            tid=self.tid,
        )

    @cute.jit
    def fire(self) -> None:
        """Publish this lane's pending slot. Subclasses define the scope."""
        raise NotImplementedError

    @cute.jit
    def accumulate(
        self,
        next_phase: Any,
        flush_threshold: int,
        flag_addr: Int64,
        no_fire: bool = False,
    ) -> "FlagBatchTrackerBase":
        if cutlass.const_expr(flush_threshold == 1):
            if cutlass.const_expr(not no_fire):
                per_lane_addr = Int64(0)
                if self.tid == 0:
                    per_lane_addr = flag_addr
                self._make(
                    flag_addr=per_lane_addr,
                    cumulated_flags=cutlass.Int32(1),
                    phase=self.phase,
                ).fire()
            return self._make(
                flag_addr=Int64(0),
                cumulated_flags=cutlass.Int32(0),
                phase=cutlass.Int32(next_phase),
            )

        cur_addr = self.flag_addr
        cumulated = self.cumulated_flags
        if self.tid == cumulated:
            cur_addr = flag_addr
        cumulated = cumulated + 1

        if cumulated == flush_threshold or next_phase != self.phase:
            if not no_fire:
                self._make(
                    flag_addr=cur_addr,
                    cumulated_flags=cumulated,
                    phase=self.phase,
                ).fire()
            cumulated = cutlass.Int32(0)
            cur_addr = Int64(0)

        return self._make(
            flag_addr=cur_addr,
            cumulated_flags=cumulated,
            phase=cutlass.Int32(next_phase),
        )


@dataclasses.dataclass(frozen=True)
class GpuReleaseFlagBatchTracker(FlagBatchTrackerBase):
    """Batched done-counter publisher using GPU-scope release reductions."""

    @cute.jit
    def _make(
        self,
        flag_addr: Int64,
        cumulated_flags: cutlass.Int32,
        phase: cutlass.Int32,
    ) -> "GpuReleaseFlagBatchTracker":
        return GpuReleaseFlagBatchTracker(
            flag_addr=flag_addr,
            cumulated_flags=cumulated_flags,
            phase=phase,
            tid=self.tid,
        )

    @cute.jit
    def fire(self) -> None:
        if self.flag_addr != Int64(0):
            ptr = cute.make_ptr(
                cutlass.Int32,
                self.flag_addr,
                AddressSpace.gmem,
                assumed_align=4,
            )
            red_add_release_gpu_s32(ptr, cutlass.Int32(1))


@dataclasses.dataclass(frozen=True)
class SysReleaseFlagBatchTracker(FlagBatchTrackerBase):
    """Batched done-counter publisher using system-scope release reductions."""

    @cute.jit
    def _make(
        self,
        flag_addr: Int64,
        cumulated_flags: cutlass.Int32,
        phase: cutlass.Int32,
    ) -> "SysReleaseFlagBatchTracker":
        return SysReleaseFlagBatchTracker(
            flag_addr=flag_addr,
            cumulated_flags=cumulated_flags,
            phase=phase,
            tid=self.tid,
        )

    @cute.jit
    def fire(self) -> None:
        if self.flag_addr != Int64(0):
            red_add_release_sys_s32_raw(self.flag_addr, cutlass.Int32(1))
