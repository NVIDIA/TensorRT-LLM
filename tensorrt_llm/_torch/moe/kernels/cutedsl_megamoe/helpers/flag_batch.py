"""Lane- and warp-distributed GPU release-counter batching."""

import dataclasses
from typing import Any, Union

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64

from .ptx_helpers import red_add_release_gpu_s32, red_async_add_release_gpu_s32


@dataclasses.dataclass(frozen=True)
class GpuReleaseFlagBatchTracker:
    """Lane-distributed delayed publication state for synchronous GPU release counters."""

    flag_address: Int64
    accumulated_flags: Int32
    phase: Int32
    thread_idx: Int32

    @cute.jit
    def _make(self, flag_address: Int64, accumulated_flags: Int32, phase: Int32) -> "GpuReleaseFlagBatchTracker":
        return type(self)(
            flag_address=flag_address, accumulated_flags=accumulated_flags, phase=phase, thread_idx=self.thread_idx
        )

    @cute.jit
    def fire(self) -> None:
        if self.flag_address != Int64(0):
            pointer = cute.make_ptr(cutlass.Int32, self.flag_address, AddressSpace.gmem, assumed_align=4)
            red_add_release_gpu_s32(pointer, Int32(1))

    @cute.jit
    def accumulate(
        self, next_phase: Any, flush_threshold: int, flag_address: Int64, no_fire: bool = False
    ) -> "GpuReleaseFlagBatchTracker":
        if cutlass.const_expr(flush_threshold == 1):
            if cutlass.const_expr(not no_fire):
                lane_address = Int64(0)
                if self.thread_idx == Int32(0):
                    lane_address = flag_address
                self._make(flag_address=lane_address, accumulated_flags=Int32(1), phase=self.phase).fire()
            return self._make(flag_address=Int64(0), accumulated_flags=Int32(0), phase=Int32(next_phase))

        current_address = self.flag_address
        accumulated_flags = self.accumulated_flags
        if self.thread_idx == accumulated_flags:
            current_address = flag_address
        accumulated_flags = accumulated_flags + Int32(1)

        if accumulated_flags == Int32(flush_threshold) or next_phase != self.phase:
            if cutlass.const_expr(not no_fire):
                self._make(flag_address=current_address, accumulated_flags=accumulated_flags, phase=self.phase).fire()
            accumulated_flags = Int32(0)
            current_address = Int64(0)

        return self._make(flag_address=current_address, accumulated_flags=accumulated_flags, phase=Int32(next_phase))


@dataclasses.dataclass(frozen=True)
class GpuAsyncReleaseFlagBatchTracker:
    """Loop-carried warp-uniform state for asynchronous GPU release counters."""

    flag_address: Int64
    accumulated_flags: Int32
    phase: Int32
    warp_idx: Int32

    @cute.jit
    def _make(self, flag_address: Int64, accumulated_flags: Int32, phase: Int32) -> "GpuAsyncReleaseFlagBatchTracker":
        return type(self)(
            flag_address=flag_address, accumulated_flags=accumulated_flags, phase=phase, warp_idx=self.warp_idx
        )

    @cute.jit
    def fire(self) -> None:
        if self.flag_address != Int64(0):
            pointer = cute.make_ptr(cutlass.Int32, self.flag_address, AddressSpace.gmem, assumed_align=4)
            with cute.arch.elect_one():
                red_async_add_release_gpu_s32(pointer, Int32(1))

    @cute.jit
    def accumulate(
        self, next_phase: Any, flush_threshold: int, flag_address: Int64, no_fire: bool = False
    ) -> "GpuAsyncReleaseFlagBatchTracker":
        if cutlass.const_expr(flush_threshold == 1):
            if cutlass.const_expr(not no_fire):
                warp_address = Int64(0)
                if self.warp_idx == Int32(0):
                    warp_address = flag_address
                self._make(flag_address=warp_address, accumulated_flags=Int32(1), phase=self.phase).fire()
            return self._make(flag_address=Int64(0), accumulated_flags=Int32(0), phase=Int32(next_phase))

        current_address = self.flag_address
        accumulated_flags = self.accumulated_flags
        if self.warp_idx == accumulated_flags:
            current_address = flag_address
        accumulated_flags = accumulated_flags + Int32(1)

        if accumulated_flags == Int32(flush_threshold) or next_phase != self.phase:
            if cutlass.const_expr(not no_fire):
                self._make(flag_address=current_address, accumulated_flags=accumulated_flags, phase=self.phase).fire()
            accumulated_flags = Int32(0)
            current_address = Int64(0)

        return self._make(flag_address=current_address, accumulated_flags=accumulated_flags, phase=Int32(next_phase))


@cute.jit
def make_flag_batch_tracker(
    use_async: bool, *, flag_address: Int64, accumulated_flags: Int32, phase: Int32, thread_idx: Int32
) -> Union[GpuReleaseFlagBatchTracker, GpuAsyncReleaseFlagBatchTracker]:
    """Construct a lane-batched synchronous or warp-batched asynchronous tracker.

    ``thread_idx`` must be zero-based within the caller's cooperating publisher
    group. The async tracker maps each contiguous group of 32 thread indices to
    one warp-uniform publisher.
    """
    if cutlass.const_expr(use_async):
        return GpuAsyncReleaseFlagBatchTracker(
            flag_address=flag_address,
            accumulated_flags=accumulated_flags,
            phase=phase,
            warp_idx=cute.arch.make_warp_uniform(thread_idx // Int32(32)),
        )
    return GpuReleaseFlagBatchTracker(
        flag_address=flag_address, accumulated_flags=accumulated_flags, phase=phase, thread_idx=thread_idx
    )


__all__ = ["GpuAsyncReleaseFlagBatchTracker", "GpuReleaseFlagBatchTracker", "make_flag_batch_tracker"]
