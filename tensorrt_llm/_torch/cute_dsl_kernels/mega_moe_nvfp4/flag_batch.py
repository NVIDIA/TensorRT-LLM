# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rotating-lane delayed release helper for done-counter publishing."""

import dataclasses
from typing import Any, Union

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int64

from .ptx_helpers import red_add_release_gpu_s32, red_async_add_release_gpu_s32


@dataclasses.dataclass(frozen=True)
class GpuReleaseFlagBatchTracker:
    """Batched done-counter publisher using GPU-scope release reductions.

    Carries the loop-carried per-thread accumulation state.
    """

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
    ) -> "GpuReleaseFlagBatchTracker":
        return GpuReleaseFlagBatchTracker(
            flag_addr=flag_addr,
            cumulated_flags=cumulated_flags,
            phase=phase,
            tid=self.tid,
        )

    @cute.jit
    def fire(self) -> None:
        """Publish this lane's pending slot."""
        if self.flag_addr != Int64(0):
            ptr = cute.make_ptr(
                cutlass.Int32,
                self.flag_addr,
                AddressSpace.gmem,
                assumed_align=4,
            )
            red_add_release_gpu_s32(ptr, cutlass.Int32(1))

    @cute.jit
    def accumulate(
        self,
        next_phase: Any,
        flush_threshold: int,
        flag_addr: Int64,
        no_fire: bool = False,
    ) -> "GpuReleaseFlagBatchTracker":
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
class GpuAsyncReleaseFlagBatchTracker:
    """Warp-distributed asynchronous release-counter batching for SM107."""

    flag_addr: Int64
    cumulated_flags: cutlass.Int32
    phase: cutlass.Int32
    warp_idx: cutlass.Int32

    @cute.jit
    def _make(
        self,
        flag_addr: Int64,
        cumulated_flags: cutlass.Int32,
        phase: cutlass.Int32,
    ) -> "GpuAsyncReleaseFlagBatchTracker":
        return GpuAsyncReleaseFlagBatchTracker(
            flag_addr=flag_addr,
            cumulated_flags=cumulated_flags,
            phase=phase,
            warp_idx=self.warp_idx,
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
            with cute.arch.elect_one():
                red_async_add_release_gpu_s32(ptr, cutlass.Int32(1))

    @cute.jit
    def accumulate(
        self,
        next_phase: Any,
        flush_threshold: int,
        flag_addr: Int64,
        no_fire: bool = False,
    ) -> "GpuAsyncReleaseFlagBatchTracker":
        if cutlass.const_expr(flush_threshold == 1):
            if cutlass.const_expr(not no_fire):
                per_warp_addr = Int64(0)
                if self.warp_idx == 0:
                    per_warp_addr = flag_addr
                self._make(
                    flag_addr=per_warp_addr,
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
        if self.warp_idx == cumulated:
            cur_addr = flag_addr
        cumulated = cumulated + 1

        if cumulated == flush_threshold or next_phase != self.phase:
            if cutlass.const_expr(not no_fire):
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


@cute.jit
def make_flag_batch_tracker(
    use_async: bool,
    *,
    flag_addr: Int64,
    cumulated_flags: cutlass.Int32,
    phase: cutlass.Int32,
    tid: cutlass.Int32,
) -> Union[GpuReleaseFlagBatchTracker, GpuAsyncReleaseFlagBatchTracker]:
    """Construct the architecture-selected completion publisher."""
    if cutlass.const_expr(use_async):
        return GpuAsyncReleaseFlagBatchTracker(
            flag_addr=flag_addr,
            cumulated_flags=cumulated_flags,
            phase=phase,
            warp_idx=cute.arch.make_warp_uniform(tid // cutlass.Int32(32)),
        )
    return GpuReleaseFlagBatchTracker(
        flag_addr=flag_addr,
        cumulated_flags=cumulated_flags,
        phase=phase,
        tid=tid,
    )
