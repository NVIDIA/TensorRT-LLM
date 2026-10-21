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
``p`` is MoE order ``p % num_layers`` and uses the prefetch slot ``order % num_slots`` (the slot
its composite VA maps).  A position is issued (its copy enqueued) once every earlier position is
issued and its slot no longer holds a position that has not finished.  A slot is released right
after the MoE that read it is enqueued, so each MoE layer is gathered once per pass, the copy into
a slot overlaps everything up to the next MoE that reads the slot, and the last MoE layers of a
pass already gather the first layers of the next pass, through the gap between executor steps.

``PrefetchSchedule`` only does the host bookkeeping.  Each MoE layer goes through ``bind`` (before
its MoE reads the slot) and ``finish`` (right after the MoE is enqueued); a pass starts with
``start_forward``.  Every call returns a ``ScheduleStep`` that the caller enqueues in field order.
A pass that stops inside the sequence, or a bind out of sequence, restarts the sequence at that
layer; a pass without ``start_forward`` (e.g. a warmup forward) issues its copies on the first
bind.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional


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
        num_layers: Number of MoE layers of one forward pass.
        num_slots: Number of prefetch buffer slots; MoE order ``o`` uses slot ``o % num_slots``.
    """

    def __init__(self, num_layers: int, num_slots: int) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_slots < 2:
            raise ValueError(f"num_slots must be at least 2, got {num_slots}")
        self._num_layers = num_layers
        self._num_slots = num_slots
        # Next position to bind and number of positions issued so far.
        self._next_bind = 0
        self._issued = 0
        # Position holding each slot until it finishes (None: free).
        self._slot_owner: List[Optional[int]] = [None] * num_slots
        # Position bound and not finished yet.
        self._bound: Optional[int] = None
        self._num_restarts = 0

    @property
    def num_restarts(self) -> int:
        """How often a pass broke the sequence (stopped inside it or bound out of order)."""
        return self._num_restarts

    def slot(self, order: int) -> int:
        """Prefetch slot of MoE order ``order``."""
        return order % self._num_slots

    def start_forward(self) -> ScheduleStep:
        """Start a forward pass: issue its first layers unless they are already in flight."""
        if self._bound is None and self._next_bind % self._num_layers == 0:
            return ScheduleStep(False, None, self._issue())
        self._num_restarts += 1
        synchronize = self._restart(0)
        return ScheduleStep(synchronize, None, self._issue())

    def bind(self, order: int) -> ScheduleStep:
        """Bind MoE order ``order``; its MoE reads the slot only after the returned step.

        A bind out of sequence restarts the sequence at ``order``.

        Raises:
            ValueError: If ``order`` is out of range.
        """
        self._check_order(order)
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
        return ScheduleStep(synchronize, None, prefetch)

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
        self._bound = None
        slot = self.slot(order)
        self._slot_owner[slot] = None
        return ScheduleStep(False, slot, self._issue())

    def invalidate(self) -> ScheduleStep:
        """Forget every issued prefetch; the next pass issues its first layers again."""
        return ScheduleStep(self._restart(0), None, [])

    def _check_order(self, order: int) -> None:
        if not 0 <= order < self._num_layers:
            raise ValueError(f"MoE order {order} out of range [0, {self._num_layers})")

    def _restart(self, order: int) -> bool:
        """Restart the sequence at ``order``; returns whether a bound layer was abandoned."""
        abandoned = self._bound is not None
        self._slot_owner = [None] * self._num_slots
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
