# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Issue order of the DWDP expert prefetch.

The MoE layers of consecutive forward passes form one cyclic sequence of positions: position
``p`` is MoE order ``p % num_layers`` and uses that order's prefetch slot (the slot its composite
VA maps, ``WeightBuffer.buffer_index_for_layer``).  A position is issued (its copy enqueued) once
every earlier position is issued and its slot no longer holds a position that has not finished.
A slot is released right after the MoE that read it is enqueued, so each MoE layer is gathered
once per pass, the copy into a slot overlaps everything up to the next MoE that reads the slot,
and the last MoE layers of a pass already gather the first layers of the next pass, through the
gap between executor steps.

``PrefetchSchedule`` only does the host bookkeeping.  Each MoE layer goes through ``bind`` (before
its MoE reads the slot) and ``finish`` (right after the MoE is enqueued); a pass starts with
``start_forward``.  Every call returns a ``ScheduleStep`` that the caller enqueues in field order.
A pass that stops inside the sequence, or a bind out of sequence, restarts the sequence at that
layer; a pass without ``start_forward`` (e.g. a warmup forward) issues its copies on the first
bind.

A repeated MoE order runs several times in a row within a pass (the one-model MTP-Eagle layer,
reused for every draft step).  Its slot is held after each finish and released only by the next
call that is not a bind of the same order, so every repeat binds the copy that is still in the
slot instead of gathering the layer again.
"""

from __future__ import annotations

from typing import Iterable, List, NamedTuple, Optional, Sequence


class ScheduleStep(NamedTuple):
    """What the caller enqueues for one schedule call, in field order.

    Attributes:
        synchronize: Synchronize the device first: a layer bound by an unfinished pass may still
            be reading its slot, and its release was never recorded.
        release: Slot whose consume event is recorded now on the compute stream, if any.
        prefetch: MoE orders whose copies are enqueued now, in issue order.
    """

    synchronize: bool
    release: Optional[int]
    prefetch: List[int]


class PrefetchSchedule:
    """Host-side issue order of the DWDP expert prefetch over consecutive forward passes.

    Args:
        slots: Prefetch slot of each MoE order of a forward pass, as the weight buffer maps it.
        repeated_orders: MoE orders that may be bound several times in a row within a pass.

    Raises:
        ValueError: If ``slots`` is empty or negative, or a repeated order is out of range.
    """

    def __init__(self, slots: Sequence[int], repeated_orders: Iterable[int] = ()) -> None:
        if not slots:
            raise ValueError("a prefetch schedule needs at least one MoE layer")
        if min(slots) < 0:
            raise ValueError(f"negative prefetch slot in {list(slots)}")
        self._slots = list(slots)
        self._num_layers = len(self._slots)
        self._repeated = frozenset(repeated_orders)
        for order in self._repeated:
            self._check_order(order)
        num_slots = max(self._slots) + 1
        # Next position to bind and number of positions issued so far.
        self._next_bind = 0
        self._issued = 0
        # Position holding each slot until it finishes (None: free).
        self._slot_owner: List[Optional[int]] = [None] * num_slots
        # Position bound and not finished yet.
        self._bound: Optional[int] = None
        # Finished position of a repeated order whose slot is still held for a repeat.
        self._held: Optional[int] = None
        self._num_restarts = 0

    @property
    def num_restarts(self) -> int:
        """How often a pass broke the sequence (stopped inside it or bound out of order)."""
        return self._num_restarts

    def slot(self, order: int) -> int:
        """Prefetch slot of MoE order ``order``."""
        return self._slots[order]

    def start_forward(self) -> ScheduleStep:
        """Start a forward pass: issue its first layers unless they are already in flight."""
        release = self._release_held()
        if self._bound is None and self._next_bind % self._num_layers == 0:
            return ScheduleStep(False, release, self._issue())
        self._num_restarts += 1
        synchronize = self._restart(0)
        return ScheduleStep(synchronize, release, self._issue())

    def bind(self, order: int) -> ScheduleStep:
        """Bind MoE order ``order``; its MoE reads the slot only after the returned step.

        A bind out of sequence restarts the sequence at ``order``.  A repeated order bound again
        right after its finish reads the copy still held in its slot.

        Raises:
            ValueError: If ``order`` is out of range.
        """
        self._check_order(order)
        if (
            self._bound is None
            and self._held is not None
            and order == self._held % self._num_layers
        ):
            self._bound = self._held
            return ScheduleStep(False, None, [])
        release = self._release_held()
        synchronize = False
        if self._bound is not None or order != self._next_bind % self._num_layers:
            self._num_restarts += 1
            synchronize = self._restart(order)
        prefetch = self._issue()
        if self._issued <= self._next_bind:
            raise RuntimeError(
                f"DWDP prefetch schedule: MoE order {order} bound before its copy was issued"
            )
        self._bound = self._next_bind
        return ScheduleStep(synchronize, release, prefetch)

    def finish(self, order: int) -> ScheduleStep:
        """Finish the bound MoE order ``order`` (its MoE has just been enqueued).

        Raises:
            ValueError: If ``order`` is out of range.
            RuntimeError: If ``order`` is not the bound layer.
        """
        self._check_order(order)
        if self._bound is None or order != self._bound % self._num_layers:
            bound = None if self._bound is None else self._bound % self._num_layers
            raise RuntimeError(
                f"DWDP prefetch schedule: finish of MoE order {order} without a matching bind "
                f"(bound={bound})"
            )
        self._next_bind = self._bound + 1
        if order in self._repeated:
            self._held = self._bound
            self._bound = None
            return ScheduleStep(False, None, [])
        self._bound = None
        slot = self.slot(order)
        self._slot_owner[slot] = None
        return ScheduleStep(False, slot, self._issue())

    def invalidate(self) -> ScheduleStep:
        """Forget every issued prefetch; the next pass issues its first layers again."""
        release = self._release_held()
        return ScheduleStep(self._restart(0), release, [])

    def _check_order(self, order: int) -> None:
        if not 0 <= order < self._num_layers:
            raise ValueError(f"MoE order {order} out of range [0, {self._num_layers})")

    def _release_held(self) -> Optional[int]:
        """Release the slot held for a repeat, if any; returns the slot."""
        if self._held is None:
            return None
        slot = self.slot(self._held % self._num_layers)
        self._slot_owner[slot] = None
        self._held = None
        return slot

    def _restart(self, order: int) -> bool:
        """Restart the sequence at ``order``; returns whether a bound layer was abandoned."""
        abandoned = self._bound is not None
        self._slot_owner = [None] * len(self._slot_owner)
        self._bound = None
        self._next_bind = order
        self._issued = order
        return abandoned

    def _issue(self) -> List[int]:
        orders: List[int] = []
        while True:
            order = self._issued % self._num_layers
            slot = self.slot(order)
            if self._slot_owner[slot] is not None:
                return orders
            self._slot_owner[slot] = self._issued
            self._issued += 1
            orders.append(order)
