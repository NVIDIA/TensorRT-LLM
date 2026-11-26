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
"""The backend contract: what a lease holder hands a backend, and what it gets back.

The lease holder holds a lease from the KV cache lender and submits its view to a backend. The
caller asks the lease holder for a transfer and supplies its hint. The assembly builds each
backend and hands it the lender's regions, in the lender's order, and its ``WakeFn``. The
obligations stated on each type are part of the contract, checked or not. Methods beyond
:class:`Backend` come as optional protocols, which a backend declares by implementing them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

import numpy as np

from .memory import MemoryKind

__all__ = [
    "Attempt",
    "Backend",
    "CacheKind",
    "Capabilities",
    "Hint",
    "InvalidSubmission",
    "LayerGroupRows",
    "MemoryKind",
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

    A backend that cannot tell the two apart reads a state's one slot as a one-entry table and its
    checkpoint as a bound, and neither mistake announces itself.
    """

    PAGED = 0
    STATE = 1


Hint = Any
"""The caller's hint for the other end of a submission (its source or destination), handed to
the backend unread by the lease holder. A hint is an immutable snapshot that the caller never
changes, so the backend may keep a reference to it, for as long as it needs, without copying it.
It describes only the other end: which rows this end reads or writes comes from the view alone."""

WakeFn = Callable[[], None]
"""Handed to a backend by the assembly and called from any thread at least when an outcome lands or
an access ends, and on any further occasion an optional protocol names: thread-safe, reentrant,
never blocks or raises, harmless after shutdown. A backend calls it only after every change it
reports is visible: an outcome or an access end through :meth:`Attempt.poll` or
:meth:`Attempt.quiesced`, a change an optional protocol names where that protocol says."""


@runtime_checkable
class SlotRegion(Protocol):
    """A contiguous memory region divided into equal-sized slots; a row lying in the region occupies
    one of its slots.

    A backend registers only the regions whose :attr:`memory_kind` it can access, each once as
    ``(address, nbytes)``; rows in a region it has not registered are rows it does not recognize.
    Slot ``k`` starts at ``address + k * slot_bytes`` for ``0 <= k < slot_count``; ``nbytes``
    covers all slots, so it is at least ``slot_bytes * slot_count``. A view may select
    nonconsecutive slots.
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

    @property
    def memory_kind(self) -> MemoryKind:
        """The kind of memory the region lies in. ``DEVICE`` is the GPU of this rank's lender; the
        region names no device id."""
        ...


@runtime_checkable
class LayerGroupRows(Protocol):
    """One layer group's rows of a view, described by aligned arrays.

    A row is one transfer unit. When slots are provided, row ``i`` occupies slot ``slots[i]`` in
    every region :attr:`region_indices` lists. These members are all a backend reads of a group,
    apart from those a capability it declares names. Slots need not be consecutive. A backend given
    rows it does not recognize rejects the submission with :class:`InvalidSubmission`; it never
    skips them.
    """

    @property
    def layer_group_index(self) -> int:
        """The lender's local layer-group index, not the group's position in :attr:`View.groups`."""
        ...

    @property
    def page_ordinals(self) -> np.ndarray:
        """One page ordinal per row, ``int64`` of shape ``[rows]``: the request's page the row
        holds, as the lender's published page layout numbers pages. Ordinals need not be
        consecutive."""
        ...

    @property
    def object_keys(self) -> np.ndarray | None:
        """One opaque object key per row, ``uint8`` of shape ``[rows, width]`` with the lender's
        fixed width. Keys are compared only for equality under the lender's identity rules;
        they are not payload checksums. ``None`` for rows without keys, which reach only a backend
        that declares a capability admitting them."""
        ...

    @property
    def slots(self) -> np.ndarray | None:
        """One slot per row, ``int64`` of shape ``[rows]``: row ``i`` occupies slot ``slots[i]`` in
        every :class:`SlotRegion` :attr:`region_indices` lists, its bytes those slots back to back
        in the listed order. ``None`` exactly when :attr:`region_indices` is."""
        ...

    @property
    def region_indices(self) -> tuple[int, ...] | None:
        """The indices, among the lender's regions, of the :class:`SlotRegion` objects the rows lie
        in, at least one and distinct, in the order a row's bytes run; or ``None``."""
        ...

    def __len__(self) -> int: ...


@runtime_checkable
class View(Protocol):
    """The rows one submission moves, all from one lease: at most one :class:`LayerGroupRows`
    per layer group.

    Its groups and their members, the rows' metadata, are an immutable snapshot, never changed or
    reused: they stay valid for as long as the lease holder keeps the view, after the lease is
    returned too. The lease bounds only the protection of the payload, the bytes in the memory
    the rows locate.
    """

    @property
    def groups(self) -> Sequence[LayerGroupRows]: ...

    def row_masks(self, fill_value: bool = False) -> tuple[np.ndarray, ...]:
        """One new writable ``bool`` mask of shape ``[len(group)]`` per group, in :attr:`groups`
        order.

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


def _delivery_mask(mask: np.ndarray) -> np.ndarray:
    """``mask`` itself if it is a read-only ``bool`` array whose type is exactly ``numpy.ndarray``;
    otherwise a read-only copy of it.

    Raises:
        TypeError: ``mask`` is not of dtype ``bool``.
    """
    if type(mask) is np.ndarray and mask.dtype == np.bool_ and not mask.flags.writeable:
        return mask
    copy = np.array(mask)
    if copy.dtype != np.bool_:
        raise TypeError(f"a delivery mask is bool, not {copy.dtype}")
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
    a fault, a local cancel is not. An outcome says neither when the request may use the rows,
    which the lender decides, nor whether access has ended (``quiesced()``).

    Attributes:
        delivered_masks: One read-only ``bool`` mask of shape ``[len(group)]`` per group of the
            view, in order. A read-only mask of dtype ``bool`` whose type is exactly
            ``numpy.ndarray`` is kept as it is, and its giver writes neither it nor memory it
            shares afterwards; any other mask is copied.
        fault: What went wrong, or ``None``.

    Raises:
        TypeError: A mask given is not of dtype ``bool``.
    """

    delivered_masks: tuple[np.ndarray, ...]
    fault: str | None = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "delivered_masks",
            tuple(_delivery_mask(mask) for mask in self.delivered_masks),
        )

    @classmethod
    def for_view(
        cls,
        view: View,
        delivered_masks: Sequence[np.ndarray] | None = None,
        fault: str | None = None,
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
    order. Neither depends on these calls: a backend advances its own work, so a lease holder that
    calls them only after a wake misses nothing. No method blocks or raises; trouble is a
    ``fault``. Every attempt concludes, at the latest when :meth:`Backend.shutdown` forces it; the
    lease holder may act on its own deadline before then. The backend alone settles each physical
    operation, never an adapter or the lease holder, and :meth:`Backend.shutdown` never stops what
    produces the evidence that an operation ended. An attempt's methods are called in its
    backend's call sequence (see :class:`Backend`).
    """

    def poll(self) -> Outcome | None:
        """``None`` until concluded; the same outcome on every call after."""
        ...

    def quiesced(self) -> bool:
        """True once access to the lent memory has ended, with evidence; its lease may be returned
        on it alone.

        A pure read that never advances a transfer. It may turn true late, only on evidence, and
        never turns false again. It covers the whole submission: a backend may leave a row not
        proven ended, which holds every row's memory, so the leases lent for such a backend are
        kept small.
        """
        ...

    def cancel(self) -> None:
        """Advisory: the backend tries not to start what it has not started, and a started transfer
        may still finish. Called only when ``capabilities.cancel`` is true. Never raises, and never
        turns :meth:`quiesced` true without evidence; the outcome reads as if not cancelled. Calling
        it again, or after the attempt concluded, does nothing."""
        ...


@runtime_checkable
class Backend(Protocol):
    """Moves a view's rows between lent memory and a peer or a store.

    An instance serves every submission of every request in its domain, that of one lender, and
    itself registers the memory it reads or writes. A process may hold several instances, of one
    class or of several, each registering its memory and shutting down on its own; a backend that
    allows only one instance has its factory refuse to construct a second.

    The methods of a backend and those of its attempts (:meth:`Attempt.poll`,
    :meth:`Attempt.quiesced`, :meth:`Attempt.cancel`) belong to one call sequence: one thread at a
    time calls them, the builder, then the executor loop, then the shutdown thread.
    :meth:`Attempt.poll` and :meth:`Attempt.quiesced` are also safe against anything the backend
    runs concurrently. The lease holder never calls :meth:`shutdown` on one thread while it goes
    on calling an attempt's methods on another.
    """

    @property
    def capabilities(self) -> Capabilities: ...

    def fetch(self, view: View, source: Hint) -> Attempt:
        """Writes the view's rows from ``source``. Returns without waiting on a peer or store.

        The lease holder guarantees that the memory the view's rows locate is writable when it
        calls, that no two rows locate overlapping memory, and that nothing but this submission
        writes it until the submission's :meth:`Attempt.quiesced` is true.

        :class:`InvalidSubmission` and :class:`SubmissionRejected` have no side effect; any other
        raise, a plain ``ValueError`` included, may have let access to the memory escape, and the
        lease holder then keeps the lease until :meth:`shutdown` returns an empty set. Once the
        lent memory may be accessed, the call returns the handle and reports any later failure
        through it.

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

        The lease holder guarantees that the payload the view's rows locate is completely written
        and readable when it calls, and stays unchanged until the submission's
        :meth:`Attempt.quiesced` is true.

        :class:`InvalidSubmission` and :class:`SubmissionRejected` have no side effect; any other
        raise, a plain ``ValueError`` included, may have let access to the memory escape, and the
        lease holder then keeps the lease until :meth:`shutdown` returns an empty set. Once the
        lent memory may be accessed, the call returns the handle and reports any later failure
        through it.

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

    def shutdown(self, timeout: float) -> frozenset:
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
