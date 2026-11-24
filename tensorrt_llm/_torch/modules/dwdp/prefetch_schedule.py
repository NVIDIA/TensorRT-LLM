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
"""Issue order of the cross-step DWDP prefetch (``TRTLLM_DWDP_CROSS_STEP_PREFETCH=1``).

The MoE layers of consecutive forward passes form one cyclic sequence of positions: position ``p``
is MoE order ``p % num_layers`` and lands in prefetch slot ``(p % num_layers) % num_slots`` (the
static slot ``WeightBuffer.buffer_index_for_layer`` maps the layer to).  A position is issued once
every earlier position is issued and its slot no longer holds a position whose consume is still
pending.  The last MoE layers of a forward pass therefore prefetch the first layers of the next
pass, and every layer is gathered exactly once per pass.

The class only does host bookkeeping; the caller enqueues the copies and the events.  Each layer
goes through ``bind`` (before its MoE reads the slot) and ``consume`` (after the MoE is enqueued and
the slot's consume event is recorded).  A pass that stops between the two, or a bind out of
sequence, restarts the sequence; positions issued before the restart are abandoned (their copies
are overwritten in stream order by the re-issued ones).
"""

from __future__ import annotations

from typing import List, Optional, Tuple


class CyclicPrefetchSchedule:
    """Host-side prefetch order over the MoE layers of consecutive forward passes.

    Args:
        num_layers: Number of MoE layers of one forward pass.
        num_slots: Number of prefetch buffer slots; MoE order ``o`` uses slot ``o % num_slots``.
    """

    def __init__(self, num_layers: int, num_slots: int) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {num_slots}")
        self._num_layers = num_layers
        self._num_slots = num_slots
        self._next_bind = 0
        self._issued = 0
        self._slot_owner: List[Optional[int]] = [None] * num_slots
        self._bound: Optional[int] = None
        self._num_restarts = 0

    @property
    def num_layers(self) -> int:
        """Number of MoE layers of one forward pass."""
        return self._num_layers

    @property
    def num_slots(self) -> int:
        """Number of prefetch buffer slots."""
        return self._num_slots

    @property
    def num_restarts(self) -> int:
        """How often a pass broke the sequence (stopped inside it or bound out of order)."""
        return self._num_restarts

    def slot(self, order: int) -> int:
        """Prefetch slot of MoE order ``order``."""
        return order % self._num_slots

    def pending(self) -> List[int]:
        """MoE orders issued and not yet bound, in issue order."""
        first = self._next_bind if self._bound is None else self._next_bind + 1
        return [p % self._num_layers for p in range(first, self._issued)]

    def start_forward(self) -> Tuple[List[int], bool]:
        """Prepare a forward pass: issue its first layers unless they are already in flight.

        Returns:
            The MoE orders to prefetch now, and whether a layer bound by an unfinished pass was
            abandoned (its MoE may still be reading the slot).
        """
        if self._bound is None and self._next_bind % self._num_layers == 0:
            return self._issue(), False
        self._num_restarts += 1
        abandoned_read = self._restart(0)
        return self._issue(), abandoned_read

    def bind(self, order: int) -> Tuple[List[int], bool]:
        """Bind MoE order ``order``; it must read the slot only after the returned prefetches.

        A bind out of sequence restarts the sequence at ``order``.

        Returns:
            The MoE orders to prefetch before the bind (``order`` first if it was not issued yet),
            and whether a layer bound by an unfinished pass was abandoned.
        """
        self._check_order(order)
        abandoned_read = False
        if self._bound is not None or order != self._next_bind % self._num_layers:
            self._num_restarts += 1
            abandoned_read = self._restart(order)
        orders = self._issue()
        self._bound = self._next_bind
        return orders, abandoned_read

    def consume(self, order: int) -> List[int]:
        """Release the slot of the bound MoE order ``order``.

        Returns:
            The MoE orders whose prefetch the released slot unblocks, in issue order.

        Raises:
            RuntimeError: If ``order`` is not the bound layer.
        """
        self._check_order(order)
        if self._bound is None or order != self._bound % self._num_layers:
            raise RuntimeError(
                f"DWDP prefetch schedule: consume of MoE order {order} without a matching bind "
                f"(bound={None if self._bound is None else self._bound % self._num_layers})"
            )
        self._slot_owner[self.slot(order)] = None
        self._bound = None
        self._next_bind += 1
        return self._issue()

    def invalidate(self) -> bool:
        """Forget every issued prefetch; the next pass issues its first layers again.

        Returns:
            Whether a layer bound by an unfinished pass was abandoned.
        """
        return self._restart(0)

    def _check_order(self, order: int) -> None:
        if not 0 <= order < self._num_layers:
            raise ValueError(f"MoE order {order} out of range [0, {self._num_layers})")

    def _restart(self, order: int) -> bool:
        abandoned_read = self._bound is not None
        self._slot_owner = [None] * self._num_slots
        self._bound = None
        self._next_bind = order
        self._issued = order
        return abandoned_read

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
