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
"""What a cache backend is asked to do, and what it still owes the caller word about."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Protocol, Union, runtime_checkable

import numpy as np

# TODO:
# 1. Something outside the engine cannot implement this until the file leaves this package.
# 2. Wrapping request ids and endpoints belongs to a later repo-wide typing pass.
# 3. Retrying from a second source needs an (id, epoch) key here and in ``native`` together.

__all__ = [
    "Attempt",
    "CacheExtent",
    "CacheKind",
    "Cancelled",
    "Chunk",
    "Delivered",
    "Failed",
    "Fetches",
    "Outcome",
    "Publishes",
    "TokenRange",
]


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
    """How one layer group's region ids are read, and with them its share of a token range.

    PAGED: several block ids covering a span, whose end is an upper bound -- any prefix of it is
           valid cache (attention).
    STATE: one slot id standing for the request as a whole, whose end is an exact checkpoint -- a
           slot rolled to another position is not a prefix of this one (recurrent state).

    A backend that cannot tell the two apart reads a slot id as a block id and a checkpoint as a
    bound, and neither mistake announces itself.
    """

    PAGED = 0
    STATE = 1


@dataclass
class Chunk:
    """One piece of a transfer. A monolithic transfer is the sole piece.

    ``token_range`` is the span this piece moves, not a claim that every layer group covers it: a
    recurrent-state group carries one slot rather than a span, and a windowed group's list is short.
    """

    block_ids_per_layer_groups: List[np.ndarray]

    kind_per_layer_group: List[CacheKind]
    """How to read each list above, in the same order.

    Stated rather than inferred: both kinds are arrays of integers, so nothing in the ids themselves
    separates a page from a slot.
    """

    token_range: TokenRange  # (0, prompt_len) when not cut along tokens
    is_last: bool
    """Whether this piece ends the series. Getting it wrong ends the transfer early and in silence.

    Cannot be derived from ``token_range``: a windowed or recurrent-state group never fills the
    span, so comparing against the full length lies."""

    def __post_init__(self):
        if len(self.kind_per_layer_group) != len(self.block_ids_per_layer_groups):
            raise ValueError(
                f"{len(self.block_ids_per_layer_groups)} layer groups but "
                f"{len(self.kind_per_layer_group)} kinds"
            )


@dataclass
class CacheExtent:
    """One ask: which content, and where this rank's blocks for it are.

    One ``name`` covers the request, while ``local``'s blocks are per layer group and a windowed
    group's list is shorter.

    Handing one to ``fetch`` or ``publish`` lends it out: a backend may keep it and read it on
    another thread. Neither it, its ``Chunk``, nor the arrays inside may be changed while it is
    lent -- the transfer would quietly move a different shape, and since a backend is not required
    to copy it, nothing would report the change.

    TODO: An ``Attempt`` concluding does not end the loan. It reports a logical outcome, and one
    can be reached while writers are still going -- a failure settles the piece while its siblings
    write on -- so a backend may still be reading afterwards. Nothing on this surface says when a
    caller may write to the extent again; native answers it by holding the only reference until
    the session retires. How long this description stays readable and how long the pages it names
    stay held are two questions, and neither is answered here.
    """

    name: int

    local: Chunk
    """This rank's blocks: the destination when receiving, the source when sending. Same shape,
    opposite direction."""

    # TODO: Which layer groups a delivery leaves whole is the answering side's to say, not the
    # asking side's, and nothing carries it today.


@dataclass(frozen=True)
class Delivered:
    """The ask was met.

    TODO: The evidence is reports, not a fence; the reports travel on a different channel from the
    data and the two are not ordered.
    """

    token_end: int
    """How far this delivery reaches, as an absolute offset rather than a count.

    One value for every layer group, read by each group's kind: a bound for ``PAGED``, a checkpoint
    for ``STATE``.
    """

    reports_pending: bool = False
    """Always false: every report this transfer was owed has arrived. Setting it raises; success
    still owed a report is what ``Failed`` and ``Cancelled`` are for.
    """

    def __post_init__(self):
        if self.reports_pending:
            raise ValueError("a delivery cannot still be owed a report")


@dataclass(frozen=True)
class Failed:
    """The transfer could not be completed."""

    reason: str
    reports_pending: bool
    """Whether a report this transfer was owed has yet to arrive. Says nothing about hardware."""


@dataclass(frozen=True)
class Cancelled:
    """Someone asked for this to stop, and it ended without delivering.

    Says nothing about whether the peers have stopped -- nothing here does. A cancelled piece is
    normally still owed reports, which is a different question again.

    Cancellation is an act on the whole transfer, not on one piece: there is no way to stop a single
    chunk while its siblings continue.
    """

    by_peer: bool = False
    """A local cancel is an ordinary end; a peer's is a transfer error, and only the backend knows
    which of the two happened."""

    reports_pending: bool = True


Outcome = Union[Delivered, Failed, Cancelled]
"""How one delivery ended. ``None`` rather than a member means it has not ended yet.

Latch which member it is, not the object: ``reports_pending`` turns from true to false over time, so
a stored outcome carries a stale one.

``reports_pending`` asks one question and only one: is a report this transfer was owed still to
arrive. Receiving waits on the writers' reports, sending on word about its own writes.

**It is not a release condition, and answering it is not permission to hand memory back.** Reports
and data travel on different channels and are not ordered, so no value of it means the peers have
stopped touching the destination. Whatever eventually answers that is a separate question this
contract does not ask.

TODO: A backend that has torn down whatever a report would arrive on answers false -- which is how
this converges today, rather than by the reports arriving. An ``Attempt`` cannot yet settle on its
own evidence.

TODO: Content addressing needs a member for "the source does not hold this", which is worth asking
somewhere else for while a failure is not.
"""


@runtime_checkable
class Attempt(Protocol):
    """One delivery of one extent."""

    def poll(self) -> Optional[Outcome]:
        """Non-blocking. ``None`` while the delivery has not reached a conclusion.

        A failure is reported as soon as it is known, which may be before every report is in --
        that second question is carried by the outcome itself.
        """
        ...


@runtime_checkable
class Fetches(Protocol):
    """Somewhere cache can be pulled from. An instance is bound to one request, though it may be a
    throwaway wrapper around longer-lived state; the request is not on the wire of this protocol, so
    a backend serving an aggregated engine can implement it too."""

    def fetch(self, extent: CacheExtent, *, src: Optional[str] = None) -> Attempt:
        """Start pulling ``extent``, optionally naming which peer to pull it from.

        An implementation must either honour ``src`` or refuse it. Quietly reading from somewhere
        else -- the request's own endpoint, say -- is a disagreement nothing reports.

        A submission that fails part way raises rather than answering with a handle, so the caller
        stops the request the way it always has.

        TODO: Peers are told one at a time, so a partial failure leaves some already holding the
        destination, and raising says nothing about them. Answering with a handle needs a caller
        that polls one.
        """
        ...


@runtime_checkable
class Publishes(Protocol):
    """Somewhere cache can be offered from: the side that generated it, bound to one request.

    The other direction of ``Fetches``, and a separate protocol because a backend may serve one and
    not the other -- a store that only answers reads implements ``Fetches`` alone.
    """

    def publish(self, extent: CacheExtent) -> Attempt:
        """Offer ``extent`` to whoever this request's cache is owed to.

        No counterpart to ``Fetches``'s ``src``: a fetch chooses among sources, while publishing
        answers the peers that already asked. Naming a destination is a routing question that
        arrives with content addressing, not before.

        A submission that fails part way raises rather than answering with a handle, the same way
        the other direction does.

        Whether the extent is visible to a reader only once all of it has landed is the
        implementation's to guarantee, not this protocol's to express: a store that reveals a
        half-written extent hands out cache that was never generated.
        """
        ...
