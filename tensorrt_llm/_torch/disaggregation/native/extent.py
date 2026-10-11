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
"""What native's peer-to-peer path is asked to move for one request, and how one piece ended.

``Outcome`` and ``Attempt`` share their names with the backend contract's in ``base``; the module
tells them apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Union, runtime_checkable

import numpy as np

from ..base.backend import CacheKind, TokenRange

# TODO:
# 1. Native moves cache through these per-request types instead of the backend contract.
# 2. Request ids and endpoints are bare values rather than wrapped types.
# 3. Retrying from a second source needs an (id, epoch) key here and in the sessions together.

__all__ = [
    "Attempt",
    "CacheExtent",
    "Cancelled",
    "Chunk",
    "Delivered",
    "Failed",
    "Outcome",
]


@dataclass
class Chunk:
    """One piece of a transfer. A monolithic transfer is the sole piece.

    ``pool_slot_ids_per_layer_group[g]`` of a PAGED group is positional: entry ``i`` is the local pool
    slot holding block ordinal ``i`` (tokens ``[i * tpb, (i + 1) * tpb)``), or ``-1`` when this side
    has nothing to offer/accept there -- SWA-evicted, not yet allocated, already cached on the
    receiver, or outside this piece's range. Paged groups cover the offered request prefix and
    beam 0 only; a STATE group carries a single slot. The sender pairs ordinals that are
    ``>= 0`` on both sides, so neither side needs to know the other's eviction or reuse state.

    ``token_range`` is the span this piece moves, not a claim that every layer group covers it: a
    recurrent-state group carries one slot rather than a span, and a windowed group's table is
    mostly holes.
    """

    pool_slot_ids_per_layer_group: List[np.ndarray]

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

    excluded_pool_views: set[tuple[int, int]] = field(default_factory=set)
    """Pool views omitted from transfer, as (layer-group index, pool-view index) pairs."""

    def __post_init__(self):
        if len(self.kind_per_layer_group) != len(self.pool_slot_ids_per_layer_group):
            raise ValueError(
                f"{len(self.pool_slot_ids_per_layer_group)} layer groups but "
                f"{len(self.kind_per_layer_group)} kinds"
            )


@dataclass
class CacheExtent:
    """One ask: which content, and where this rank's blocks for it are.

    One ``request_id`` identifies the local request, while ``local``'s blocks are per layer group,
    positional with ``-1`` holes (see ``Chunk``).

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

    request_id: int
    """The request identity the local adapter checks against its bound request."""

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

The logical member and its cause are committed at the task/session transition;
polling only observes that decision. ``reports_pending`` can still change, so a
stored outcome carries stale report progress, not a different logical result.

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
        The logical result is stable and does not depend on when the first poll occurs.
        """
        ...
