# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Generation-phase kernel tail: one arrival counter and, only where it is needed, one cross-rank barrier.

The tail has to guarantee before the next call can touch this rank:

  1. every CTA of this rank has finished (its counters are no longer read),
  2. the tail-reset span is zero again (rank-local state: only this rank's CTAs touch it),
  3. in-kernel reduce builds only: every peer has finished its reduce-adds into this rank's output.
     That is a real cross-rank dependency (``cross_rank_barrier``). Standalone-reduce builds have none:
     the reduce polls its own cells, and the cross-rank ready counters are not in the tail-reset span
     (they count in two slots that alternate direction and are never reset), so no peer has to be told
     that a reset happened.

Instead of two grid-wide syncs around a CTA-parallel reset, every CTA arrives once on a local counter
and leaves; the CTA whose arrival completes the count (it has acquired every other arrival) resets the
small tail span itself, then either runs the sense-reversing all-rank barrier alone (in-kernel reduce) or
just advances the epoch (standalone reduce). The barrier alternates between two signal slots and between
counting up and down, keyed by a per-call epoch that lives in the workspace like the baseline's phase
counter, so its signals never need resetting.
"""

from typing import ClassVar

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64

from .....helpers.device_workspace import DeviceWorkspace
from .....helpers.ptx_helpers import red_add_relaxed_sys_s32


class GenphaseTail:
    done_counter_region: ClassVar[str] = "rubin.genphase_tail.done_counter"
    epoch_region: ClassVar[str] = "rubin.genphase_tail.epoch"
    signal_region: ClassVar[str] = "rubin.genphase_tail.signal"

    def __init__(self, *, world_size: int) -> None:
        self.world_size = world_size

    def register_device_workspace(self, workspace: DeviceWorkspace) -> None:
        workspace.register(
            self.done_counter_region, cutlass.Int32, (1,), buffer_space="local", byte_alignment=16, reset="tail_reset"
        )
        workspace.register(
            self.epoch_region,
            cutlass.Int32,
            (1,),
            buffer_space="local",
            byte_alignment=16,
            reset="zero_on_first_allocate",
        )
        workspace.register(
            self.signal_region,
            cutlass.Int32,
            (2,),
            buffer_space="shared",
            byte_alignment=16,
            reset="zero_on_first_allocate",
        )

    @cute.jit
    def read_epoch(self, workspace: DeviceWorkspace) -> Int32:
        """This call's epoch. Only the last arriver advances it, after every CTA has arrived."""
        return cute.arch.load(workspace.ptr(self.epoch_region), Int32, sem="relaxed", scope="gpu")

    @cute.jit
    def arrive(self, workspace: DeviceWorkspace, cta_count: Int32) -> cutlass.Boolean:
        """One thread per CTA. Returns True on the CTA whose arrival completes the count."""
        old = cute.arch.atomic_add(workspace.ptr(self.done_counter_region), Int32(1), sem="acq_rel", scope="gpu")
        return old == cta_count - Int32(1)

    @cute.jit
    def reset_tail_space(self, workspace: DeviceWorkspace, lane_idx: Int32, lane_count: int) -> None:
        workspace.reset_tail_space("shared", lane_idx, lane_count)
        workspace.reset_tail_space("local", lane_idx, lane_count)

    @cute.jit
    def cross_rank_barrier(self, workspace: DeviceWorkspace, peer_rank_ptr_mapper, epoch: Int32) -> None:
        """Single thread: release everything this rank did, signal every rank, wait for every rank."""
        status = epoch & Int32(3)
        signal_phase = status & Int32(1)
        signal_delta = Int32(1)
        signal_target = Int32(self.world_size)
        if (status >> Int32(1)) != Int32(0):
            signal_delta = Int32(-1)
            signal_target = Int32(0)
        cute.arch.fence_acq_rel_sys()
        signal_address = workspace.ptr(self.signal_region).toint()
        for destination_rank in cutlass.range_constexpr(self.world_size):
            red_add_relaxed_sys_s32(
                peer_rank_ptr_mapper.map(signal_address, Int32(destination_rank), Int64(signal_phase * Int32(4))),
                signal_delta,
            )
        cute.arch.store(workspace.ptr(self.epoch_region), epoch + Int32(1), sem="relaxed", scope="gpu")
        local_signal = workspace.ptr(self.signal_region) + signal_phase
        while cute.arch.load(local_signal, Int32, sem="acquire", scope="sys") != signal_target:
            pass

    @cute.jit
    def advance_epoch(self, workspace: DeviceWorkspace, epoch: Int32) -> None:
        """Single thread: publish this rank's tail as complete for call ``epoch`` (standalone-reduce builds).

        Nothing is signalled to the peers. A peer never touches this rank's tail-reset span: the cross-rank
        ready counters count in two slots that alternate direction (see ``GenphaseTokenComm.ready_counter_slot``)
        and are never reset, so no peer has to learn that a reset happened. The epoch is released behind a
        gpu-scope fence because its readers are on this GPU: the next call's communication kernel (entry wait),
        and the standalone reduce's last CTA.
        """
        cute.arch.fence_acq_rel_gpu()
        cute.arch.store(workspace.ptr(self.epoch_region), epoch + Int32(1), sem="relaxed", scope="gpu")

    @cute.jit
    def previous_done_observation(self, workspace: DeviceWorkspace) -> Int32:
        """An early, unordered read of the slot the previous call's done signals count
        into, to be confirmed by ``wait_previous_done`` right before it matters."""
        previous = self.read_epoch(workspace) - Int32(1)
        signal_phase = previous & Int32(1)
        # Acquire, so that when this early value already meets the target (the usual case) the
        # adds into the peer's tail-reset counters are ordered after the peer's reset without
        # another load; the latency is overlapped either way.
        return cute.arch.load(workspace.ptr(self.signal_region) + signal_phase, Int32, sem="acquire", scope="sys")

    @cute.jit
    def wait_previous_done(self, workspace: DeviceWorkspace, observed: Int32) -> None:
        """Spin until every rank has signaled the previous call.

        On the first call (epoch 0) the previous slot is phase 1 counting down to 0, and
        the zero-initialized slot already reads 0, so no guard is needed; the same
        formula also covers the epoch's 32-bit wrap, which a guard on ``epoch != 0``
        would have skipped.
        """
        epoch = self.read_epoch(workspace)
        previous = epoch - Int32(1)
        status = previous & Int32(3)
        signal_phase = status & Int32(1)
        signal_target = Int32(self.world_size)
        if (status >> Int32(1)) != Int32(0):
            signal_target = Int32(0)
        local_signal = workspace.ptr(self.signal_region) + signal_phase
        while observed != signal_target:
            observed = cute.arch.load(local_signal, Int32, sem="acquire", scope="sys")

