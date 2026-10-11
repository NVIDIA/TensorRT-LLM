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
"""The backend contract: what a holder hands a backend, and what it gets back.

A holder holds a lease from the KV cache lender and submits the lease's view to a backend. The
obligations stated on each type are part of the contract, checked or not. Methods beyond
:class:`Backend` come as optional protocols, which a backend declares by implementing them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, FrozenSet, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

__all__ = [
    "Attempt",
    "Backend",
    "CacheKind",
    "Capabilities",
    "Hint",
    "InvalidSubmission",
    "LayerGroupRows",
    "Outcome",
    "SlotRegion",
    "SubmissionRejected",
    "TokenRange",
    "View",
    "WakeFn",
]

# Native's token range and layer-group kind, read by its per-request pieces and its page table;
# not part of the backend contract.


@dataclass
class TokenRange:
    """Half-open token range [start, end) within one request. An empty range is valid."""

    start: int
    end: int  # exclusive

    def __post_init__(self):
        if self.start < 0 or self.end < 0:
            raise ValueError("Token indices must be non-negative")
        if self.start > self.end:
            raise ValueError(f"Invalid range: [{self.start}, {self.end})")


class CacheKind(IntEnum):
    """How one layer group's pool slot ids are read, and with them its share of a token range.

    PAGED: a positional pool-slot table covering a span, whose end is an upper bound -- any prefix
           of it is valid cache (attention).
    STATE: one pool slot id standing for the request as a whole, whose end is an exact checkpoint
           -- a slot rolled to another position is not a prefix of this one (recurrent state).

    A backend that cannot tell the two apart reads a slot id as a block id and a checkpoint as a
    bound, and neither mistake announces itself.
    """

    PAGED = 0
    STATE = 1


Hint = Any
"""The caller's hint for the other end of a submission (its source or destination), handed to
the backend unread by the holder."""

WakeFn = Callable[[], None]
"""Handed to a backend by the assembly and called from any thread at least when an outcome lands or
an access ends, and on any further occasion an optional protocol names: thread-safe, reentrant,
never blocks or raises, harmless after shutdown."""


@runtime_checkable
class SlotRegion(Protocol):
    """A contiguous memory region divided into equal-sized slots, one row per slot.

    A backend registers the region once as ``(address, nbytes)``. Slot ``k`` starts at
    ``address + k * slot_bytes`` for ``0 <= k < slot_count``; ``nbytes`` covers all slots,
    so it is at least ``slot_bytes * slot_count``. A view may select nonconsecutive slots.
    The descriptor and its backing memory remain valid while the backend may access them.
    """

    @property
    def layout_label(self) -> str:
        """An opaque layout label, equal for regions laid out alike; not an allocation identity."""
        ...

    @property
    def address(self) -> int: ...

    @property
    def nbytes(self) -> int: ...

    @property
    def slot_bytes(self) -> int:
        """Bytes of one slot."""
        ...

    @property
    def slot_count(self) -> int:
        """Number of slots, back to back from :attr:`address`."""
        ...


@runtime_checkable
class LayerGroupRows(Protocol):
    """One layer group's rows of a view, described by aligned arrays.

    A row is one transfer unit. When addresses are provided, each row occupies one slot in the
    region. These members are all a backend reads of a group, apart from those a capability it
    declares names. Row addresses need not be consecutive.
    """

    @property
    def layer_group_index(self) -> int:
        """The lender's local layer-group index, not the group's position in :attr:`View.groups`."""
        ...

    @property
    def object_keys(self) -> Optional[np.ndarray]:
        """One opaque object key per row, ``uint8`` of shape ``[rows, width]`` with the lender's
        fixed width. Keys are compared only for equality under the lender's identity rules;
        they are not payload checksums. ``None`` for rows without keys, which reach only a backend
        that declares a capability admitting them."""
        ...

    @property
    def addresses(self) -> Optional[np.ndarray]:
        """One address per row, ``int64`` of shape ``[rows]``: row ``i`` is the slot at
        ``addresses[i]`` in the :class:`SlotRegion` at index :attr:`region_index`. ``None`` exactly
        when :attr:`region_index` is."""
        ...

    @property
    def region_index(self) -> Optional[int]:
        """The index, among the lender's regions, of the :class:`SlotRegion` the rows sit in, or
        ``None``."""
        ...

    def __len__(self) -> int: ...


@runtime_checkable
class View(Protocol):
    """The rows one submission moves, all from one lease: at most one :class:`LayerGroupRows`
    per layer group. Its groups and their arrays do not change before the submission's
    :meth:`Attempt.quiesced` is true."""

    @property
    def groups(self) -> Sequence[LayerGroupRows]: ...

    def row_masks(self, fill_value: bool = False) -> Tuple[np.ndarray, ...]:
        """One writable ``bool`` mask of shape ``[len(group)]`` per group, in :attr:`groups` order.

        Args:
            fill_value: What every entry starts as.
        """
        ...


class SubmissionRejected(Exception):
    """Refused before work, a peer notification or memory access escaped; also every submission
    after :meth:`Backend.shutdown`. Only this type certifies that nothing escaped."""


class InvalidSubmission(SubmissionRejected, ValueError):
    """Bad arguments or hint, or an undeclared direction; raised before anything escaped."""


@dataclass(frozen=True)
class Capabilities:
    """What a backend implements, fixed for the backend's life. A new capability is a field whose
    default means not supported, and rows that need a capability reach only a backend that
    declares it.

    Attributes:
        fetch: Implements :meth:`Backend.fetch`.
        publish: Implements :meth:`Backend.publish`.
        cancel: Implements the advisory :meth:`Attempt.cancel`.

    Raises:
        ValueError: Neither ``fetch`` nor ``publish`` is set.
    """

    fetch: bool
    publish: bool
    cancel: bool = False

    def __post_init__(self):
        if not (self.fetch or self.publish):
            raise ValueError("a backend implements at least one of fetch and publish")


def _readonly_bool_copy(mask: np.ndarray) -> np.ndarray:
    """A read-only ``bool`` copy of ``mask``."""
    copy = np.array(mask, dtype=bool)
    copy.flags.writeable = False
    return copy


@dataclass(frozen=True, eq=False)
class Outcome:
    """How one submission ended.

    A delivered row is one the destination holds whole and readable: byte for byte as the source
    holds it, or as the destination already held it under the row's object key. A
    delivered row is final: every write to it has completed and none will follow. A row not
    delivered may have been partly written; its memory is reused only once ``quiesced()`` is true.
    A miss is nothing delivered and no fault; deliveries and a fault can coexist; a peer's cancel is
    a fault, a local cancel is not. An outcome says nothing about readiness or about access
    (``quiesced()``).

    Attributes:
        delivered_masks: One read-only ``bool`` mask of shape ``[len(group)]`` per group of the
            view, in order, copied from the masks given.
        fault: What went wrong, or ``None``.
    """

    delivered_masks: Tuple[np.ndarray, ...]
    fault: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "delivered_masks",
            tuple(_readonly_bool_copy(mask) for mask in self.delivered_masks),
        )

    @classmethod
    def for_view(
        cls,
        view: View,
        delivered_masks: Optional[Sequence[np.ndarray]] = None,
        fault: Optional[str] = None,
    ) -> Outcome:
        """Returns an outcome for ``view``.

        Args:
            view: The view submitted; gives the all-false masks when ``delivered_masks`` is
                ``None``.
            delivered_masks: One mask per group of ``view``, not checked against it (see
                :meth:`check_mask_shapes`), or ``None`` for nothing delivered.
            fault: What went wrong, or ``None``.
        """
        return cls(
            tuple(delivered_masks) if delivered_masks is not None else view.row_masks(False), fault
        )

    def check_mask_shapes(self, view: View) -> None:
        """Checks the mask count and shapes against ``view``; does not prove delivery.

        Args:
            view: The view the outcome is for.

        Raises:
            ValueError: The outcome does not hold one mask of shape ``[len(group)]`` per group of
                ``view``.
        """
        if len(self.delivered_masks) != len(view.groups):
            raise ValueError(
                f"outcome has {len(self.delivered_masks)} masks for {len(view.groups)} groups"
            )
        for mask, group in zip(self.delivered_masks, view.groups):
            if mask.shape != (len(group),):
                raise ValueError(
                    f"layer group {group.layer_group_index}: outcome mask has shape {mask.shape} "
                    f"for {len(group)} rows"
                )


@runtime_checkable
class Attempt(Protocol):
    """The handle of one submission.

    Conclusion (:meth:`poll`) and quiescence (:meth:`quiesced`) are independent and come in either
    order. No method blocks or raises; trouble is a ``fault``. Every attempt concludes, at the
    latest when :meth:`Backend.shutdown` forces it; a caller's deadline does not wait for it. The
    backend alone settles each physical operation, never an adapter or the holder, and
    :meth:`Backend.shutdown` never stops what produces the evidence that an operation ended.
    """

    def poll(self) -> Optional[Outcome]:
        """``None`` until concluded; the same outcome on every call after."""
        ...

    def quiesced(self) -> bool:
        """True once access to the lent memory has ended, with evidence; the leases may be returned
        on it alone.

        A pure read that never advances a transfer. It may turn true late, only on evidence, and
        never turns false again. It covers the whole submission: a backend may leave a row not
        proven ended, which holds every row's memory, so the holder keeps submissions to such a
        backend small.
        """
        ...

    def cancel(self) -> None:
        """Advisory: the backend tries not to start what it has not started, and a started transfer
        may still finish. Called only when ``capabilities.cancel`` is true. Never raises, and never
        turns :meth:`quiesced` true without evidence; the outcome reads as if not cancelled."""
        ...


@runtime_checkable
class Backend(Protocol):
    """Moves a view's rows between lent memory and a peer or a store.

    Each backend has one instance per process, which serves every submission of every request and
    itself registers the memory it reads or writes. One thread at a time calls its methods: the
    builder, then the executor loop, then the shutdown thread. :meth:`Attempt.poll` and
    :meth:`Attempt.quiesced` are safe against anything the backend runs concurrently.
    """

    @property
    def capabilities(self) -> Capabilities: ...

    def fetch(self, view: View, source: Hint) -> Attempt:
        """Writes the view's rows from ``source``. Returns without waiting on a peer or store.

        :class:`InvalidSubmission` and :class:`SubmissionRejected` have no side effect; any other
        raise, ``ValueError`` included, may have lent the memory. Once the lent memory may be
        accessed, the call returns the handle and reports any later failure through it.

        Args:
            view: The rows to write.
            source: The caller's hint for where the rows come from.

        Returns:
            The submission's handle.

        Raises:
            InvalidSubmission: Bad arguments or hint, or a direction the backend does not declare.
            SubmissionRejected: The backend refuses the submission.
        """
        ...

    def publish(self, view: View, destination: Hint) -> Attempt:
        """Offers the view's rows to ``destination``. Returns without waiting on a peer or store.

        :class:`InvalidSubmission` and :class:`SubmissionRejected` have no side effect; any other
        raise, ``ValueError`` included, may have lent the memory. Once the lent memory may be
        accessed, the call returns the handle and reports any later failure through it.

        Args:
            view: The rows to offer.
            destination: The caller's hint for where the rows go.

        Returns:
            The submission's handle.

        Raises:
            InvalidSubmission: Bad arguments or hint, or a direction the backend does not declare.
            SubmissionRejected: The backend refuses the submission.
        """
        ...

    def shutdown(self, timeout: float) -> FrozenSet:
        """Stops serving peers and forces an outcome onto every unconcluded attempt.

        A forced outcome's fault names the shutdown. Repeatable: each call does what is left.
        Registrations are dropped only for memory nothing can access any more.

        Args:
            timeout: Seconds the call may take.

        Returns:
            Labels of the backend's choosing for the memory that may still be accessed after this
            call: that of unquiesced submissions and that still registered. Empty exactly when
            nothing may still be accessed.
        """
        ...
