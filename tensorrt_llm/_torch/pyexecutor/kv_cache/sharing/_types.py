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
"""The lender's public types and protocols. Numpy and the standard library only, so any side can
import them without a live KV cache manager."""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from ...llm_request import LlmRequest

# A block entry's fixed-length opaque object key.
_OBJECT_KEY_BYTES = 54


def _freeze(array: np.ndarray) -> np.ndarray:
    """``array`` marked read-only (a view when it is not already)."""
    if array.flags.writeable:
        array = array.view()
        array.flags.writeable = False
    return array


def _integer(name: str, value: object, least: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < least:
        raise ValueError(f"{name} must be at least {least}, got {value}")
    return int(value)


@dataclass(frozen=True)
class StagingOptions:
    """Host staging capacity, planned in fetch-sized units.

    The lender computes the bytes of one fetch from the layer groups and their windows, and a range
    of at most ``tokens_per_fetch`` tokens always fits. A capacity budget, not concurrency: a lease
    takes one contiguous run of slots per pool group and leases that need slots are granted strictly
    in order, so holes that released leases leave can make a lease wait although enough slots are
    free in total; a lease with no block entries is ready at its first poll, with no place in line.

    Attributes:
        tokens_per_fetch: The token count used to size one fetch unit, rounded up to whole blocks
            and adjusted for each layer group's window and sinks. Not a lease's range limit:
            larger ranges may fit when their required slots fit.
        planned_fetches: The capacity multiplier in fetch units, before the byte cap. Not a
            limit on concurrent leases or a guarantee that this many fetches run together.
        max_bytes: A cap on the staging bytes, or ``None``; ``attach_lender`` raises
            ``ValueError`` if it is below one fetch.

    Raises:
        TypeError: A field is not an integer (``bool`` included).
        ValueError: A field is not positive.
    """

    tokens_per_fetch: int
    planned_fetches: int = 1
    max_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tokens_per_fetch", _integer("tokens_per_fetch", self.tokens_per_fetch)
        )
        object.__setattr__(
            self, "planned_fetches", _integer("planned_fetches", self.planned_fetches)
        )
        if self.max_bytes is not None:
            object.__setattr__(self, "max_bytes", _integer("max_bytes", self.max_bytes))


@dataclass(frozen=True)
class MemoryRegion:
    """One memory region containing equal-sized block entries.

    A region describes where a backend may access lent memory, not the manager's cache ownership
    or its allocation strategy. Its address and extent remain stable while a region hold is open.

    Attributes:
        layout_label: An opaque layout-derived label, equal for regions laid out alike; not an
            allocation identity.
        address: The region's base virtual address.
        nbytes: The region's extent in bytes.
        slot_bytes: The byte extent of one block entry.
        slot_count: The number of entries, back to back from ``address``.
        layer_group_indices: Manager-local layer-group indices whose entries use this region.
    """

    layout_label: str
    address: int
    nbytes: int
    slot_bytes: int
    slot_count: int
    layer_group_indices: Tuple[int, ...]


@dataclass(frozen=True, eq=False)
class LayerGroupBlocks:
    """One layer group's lent KV blocks, described by aligned read-only arrays.

    Entry ``i`` describes this rank's KV for logical token block ``block_ordinals[i]`` across the
    layers in this group, not a token tensor row or a physical page. Entries need not have
    consecutive ordinals or addresses. Placement fields are either all given or all ``None``.

    Equal object keys mean interchangeable bytes under the scope and caller obligations of
    ``attach_lender``. Keys are opaque identities, not payload checksums. Their format is not API,
    but their width, 54 bytes, is. Consumers store them and compare them only for equality.
    Changing the format or the layout makes previously stored objects miss, never hit wrongly.

    Attributes:
        layer_group_index: The manager's local index of the layer group.
        block_ordinals: ``int64 (n,)`` logical block ordinals; block ``b`` covers token positions
            ``[b * tokens_per_block, (b + 1) * tokens_per_block)``.
        object_keys: ``uint8 (n, 54)`` complete opaque object keys, distinct from the manager's
            32-byte reuse keys.
        addresses: ``int64 (n,)`` virtual addresses; entry ``i`` occupies ``slot_bytes`` bytes at
            ``addresses[i]`` in the region identified by ``region_index``.
        region_index: The position of this group's region in ``Lender.regions``, not a pool-group
            or layer-group identifier.
    """

    layer_group_index: int
    block_ordinals: np.ndarray
    object_keys: Optional[np.ndarray] = None
    addresses: Optional[np.ndarray] = None
    region_index: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "layer_group_index", _integer("layer_group_index", self.layer_group_index, 0)
        )
        block_ordinals = _freeze(np.ascontiguousarray(self.block_ordinals, dtype=np.int64))
        if block_ordinals.ndim != 1:
            raise ValueError(
                f"block_ordinals must be one-dimensional, got shape {block_ordinals.shape}"
            )
        object.__setattr__(self, "block_ordinals", block_ordinals)
        placed = (
            self.object_keys is not None,
            self.addresses is not None,
            self.region_index is not None,
        )
        if any(placed) and not all(placed):
            raise ValueError("object_keys, addresses and region_index are all given or all None")
        if not all(placed):
            return
        object_keys = np.asarray(self.object_keys)
        if object_keys.dtype != np.uint8 or object_keys.shape != (
            len(block_ordinals),
            _OBJECT_KEY_BYTES,
        ):
            raise ValueError(
                f"object_keys must be uint8 ({len(block_ordinals)}, {_OBJECT_KEY_BYTES}), "
                f"got {object_keys.dtype} {object_keys.shape}"
            )
        addresses = np.asarray(self.addresses)
        if (
            not np.issubdtype(addresses.dtype, np.integer)
            or addresses.shape != block_ordinals.shape
        ):
            raise ValueError(
                f"addresses must be integers of shape {block_ordinals.shape}, "
                f"got {addresses.dtype} {addresses.shape}"
            )
        object.__setattr__(self, "object_keys", _freeze(np.ascontiguousarray(object_keys)))
        object.__setattr__(
            self,
            "addresses",
            _freeze(np.ascontiguousarray(addresses, dtype=np.int64)),
        )
        object.__setattr__(self, "region_index", _integer("region_index", self.region_index, 0))

    def __len__(self) -> int:
        return int(self.block_ordinals.shape[0])

    def select(self, mask: np.ndarray) -> LayerGroupBlocks:
        """The block entries where ``mask`` is true, keeping the arrays aligned.

        Args:
            mask: One boolean per entry, ``True`` for each entry to keep.

        Returns:
            The same layer group and region with the kept entries, in their original order.

        Raises:
            ValueError: ``mask`` is not boolean with one value per block entry.
        """
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != self.block_ordinals.shape:
            raise ValueError(
                f"mask must be bool of shape {self.block_ordinals.shape}, got {mask.dtype} {mask.shape}"
            )
        return LayerGroupBlocks(
            self.layer_group_index,
            self.block_ordinals[mask],
            None if self.object_keys is None else self.object_keys[mask],
            None if self.addresses is None else self.addresses[mask],
            self.region_index,
        )


@dataclass(frozen=True, eq=False)
class LeaseView:
    """What a ready lease lends: at most one block collection per layer group.

    When returned by ``Lease.poll``, its memory is stable and ready for the requested access.
    All addresses belong to the lender's regions. Backends may read its immutable arrays on their
    own threads and access only the described memory, until they relinquish access and release
    the lease.

    Attributes:
        groups: Block collections, at most one per layer group. Tuple positions need not equal
            ``layer_group_index``.

    Raises:
        TypeError: A collection is not a ``LayerGroupBlocks``.
        ValueError: A layer group appears twice.
    """

    groups: Tuple[LayerGroupBlocks, ...]

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
        seen = set()
        for blocks in groups:
            if not isinstance(blocks, LayerGroupBlocks):
                raise TypeError(f"groups hold LayerGroupBlocks, got {type(blocks).__name__}")
            if blocks.layer_group_index in seen:
                raise ValueError(f"layer group {blocks.layer_group_index} appears twice")
            seen.add(blocks.layer_group_index)
        object.__setattr__(self, "groups", groups)

    @property
    def num_block_entries(self) -> int:
        """Total entries across groups; an ordinal in two groups contributes two entries."""
        return sum(len(blocks) for blocks in self.groups)

    def row_masks(self, fill_value: bool = False) -> Tuple[np.ndarray, ...]:
        """Make one writable boolean mask per group, filled with ``fill_value``.

        Mask ``i`` selects entries of ``groups[i]``; its element ``j`` refers to
        ``groups[i].block_ordinals[j]``, not to ordinal ``j``. Empty groups keep an empty mask.

        Args:
            fill_value: What every block entry's mask value starts as.

        Returns:
            One mask per collection, in ``groups`` order: the shape ``mark_arrived`` takes.
        """
        return tuple(np.full(len(blocks), bool(fill_value), dtype=bool) for blocks in self.groups)


# TODO: the caller repeats by hand the manager's own skip past a served prefix (raising the
# history, setting py_served_prefix_end and the prepopulated length), and no public manager method
# takes a position to raise a request's history to.
class ResumeBounds(NamedTuple):
    """Inclusive local bounds on token positions for resuming after a fetch.

    Resume at a position ``p`` with ``min_position <= p <= max_position``, no lower than the
    request's current context position. The bounds include only previously valid cache contents
    and fetched block entries that have been incorporated into the request's cache.

    The interval belongs to this rank and this manager. Across ranks, and across a target and
    its joint-reuse draft pool sharing the request's cursor, use the largest minimum and smallest
    maximum. Fetch into both managers before advancing their shared cursor. An empty effective
    interval requires dropping the request's cache in every manager fetched into and computing
    from 0, without another fetch into those caches.

    A minimum excludes positions whose required history is no longer available or which the
    manager cannot rewind to. Bounds never allow resuming strictly inside a bidirectional
    multimodal run. The maximum need not be a whole-block boundary; it can reach the prompt length
    after the remaining prompt tokens have been computed locally. A fetch leaves the last prompt
    token for local computation.

    Attributes:
        max_position: The last permitted resume position.
        min_position: The first permitted resume position, unless the context position is higher.
    """

    max_position: int
    min_position: int


@runtime_checkable
class Lease(Protocol):
    """One lent range: poll until the view or a final failure, mark a write once, then release.

    Methods run on the manager's thread, one call at a time. Backend threads may inspect the
    ready view and access its memory, but call no lender, lease or hold method. They report
    completion to the holder through their own channel.

    Caller obligations:
        - Poll every open lease each executor iteration, including iterations without new requests.
          Do not block waiting for a new request while a lease or backend needs progress.
        - Release every lease, failed ones too, only after every backend access has ended.
          Dropping an unreleased lease does not release its memory.
        - Combine every rank's outcome before deciding for the distributed request.
        - A failure is final. A lease failed at the lending call changed no request cache state.
        - An exception not listed by the called method is not recoverable through normal lending.
          Stop lending, polling, marking and querying bounds through that lender; release leases
          and holds once access ends, and treat parked fetches as empty resume intervals.
          A failure during manager cleanup can stop a lender permanently. A view already returned
          remains held until release; otherwise the lease reports failure.
    """

    def poll(self) -> Optional[LeaseView]:
        """Advance pending work and return a stable, ready view.

        Returns:
            The same view on every call once ready; ``None`` while pending, or permanently on failure.
            Read views contain readable bytes, and write views are ready for backend writes.

        Raises:
            RuntimeError: The lease was released.
        """
        ...

    @property
    def failure(self) -> Optional[str]:
        """Why the lease will never be ready, once that is known; for logs only."""
        ...

    def mark_arrived(self, masks: Sequence[np.ndarray]) -> None:
        """Incorporate the indicated arrived block entries into the request's cache.

        Write leases only, once, after ``poll()`` returned the view, before or after release. Each
        selected entry must have arrived whole and have no further backend writes in flight.
        Returning does not guarantee that incorporation has completed: ``resume_bounds`` reports
        when the request may use the result. Entries no longer belonging to the request are ignored;
        the holder still marks and releases after the request exits. A write released before its view
        was returned is abandoned and needs no mark.

        Args:
            masks: One boolean array per entry of ``LeaseView.groups``, in tuple order. Positions
                select entries, not logical block ordinals. ``LeaseView.row_masks()`` creates
                all-false masks.

        Raises:
            RuntimeError: A read lease, a second call, or a call before a view was returned.
            ValueError: A wrong number, shape or dtype of masks; nothing is recorded.
        """
        ...

    def release(self) -> None:
        """Confirm that the backend has stopped accessing all lent memory.

        Required for every lease, failed ones too; legal in every state and idempotent. Release is
        not cancellation of an access still in flight. A ready write still needs ``mark_arrived``,
        which may follow release.
        """
        ...


@runtime_checkable
class RegionHold(Protocol):
    """A backend's lifetime hold on all memory regions.

    Take a hold on the manager's thread before registering ``Lender.regions``. Their addresses
    and backing memory remain valid while the hold is open, including across manager shutdown.
    Deregister before shutdown, and release the hold only after deregistration is confirmed.
    A backend unable to confirm deregistration keeps its hold. Dropping a hold does not release
    it.
    """

    def release(self) -> None:
        """Confirm deregistration and that the backend cannot access the regions.

        Runs on the manager's thread, is legal in every state, and is idempotent.
        """
        ...


# TODO: object_keys exist only in ready views, so a flow cannot look up remote hits before it reserves
# pages for them, and a publish lends a windowed group's block entries only for the window at its end: a fork
# at an earlier block or a longer next turn can miss block entries and compute from 0.
@runtime_checkable
class Lender(Protocol):
    """Lend a request's KV blocks without transferring cache ownership.

    Lender, lease and hold methods run on the manager's thread, one call at a time. Backend
    threads access only a ready lease's described memory. A returned view's addresses are stable
    and ready, belong to ``regions``, and remain held until all backend access has ended.

    Caller obligations:
        - Take a region hold before registration or the first lease's backend access. Deregister
          and release the hold only after the backend can no longer access the regions.
        - Poll every open lease and query bounds for every waiting request each executor iteration,
          even without new requests.
        - Release a lease that waits too long, or free its request; progress is not guaranteed for
          a pending lease.
        - At shutdown stop the executor loop and backends, confirm deregistration, mark arrived
          entries and release leases and holds, then shut down the manager.
        - Fetch into both a target and its joint-reuse draft pool before advancing their shared
          request cursor; combine their bounds as ``ResumeBounds`` specifies.
    """

    @property
    def regions(self) -> Tuple[MemoryRegion, ...]:
        """The memory regions to which every ready view's addresses belong.

        Each address and extent remains stable while a region hold is open. A layout label is equal
        for regions laid out alike. No ordering, adjacency or single-allocation guarantee is made.
        A backend requiring one contiguous region must validate that requirement before it starts.

        Deregister before manager shutdown and release the hold after confirmation. Unreleased
        leases or holds can retain memory beyond shutdown.
        """
        ...

    def hold_regions(self) -> RegionHold:
        """Return a new lifetime hold on every region.

        The hold does not register regions or grant a lease. Take it on the manager's thread before
        registration. A hold taken after manager shutdown is inert.

        Returns:
            The hold, open until its ``release``.
        """
        ...

    def lend_read(self, request: LlmRequest, start: int, end: int) -> Lease:
        """Lend committed whole blocks covering token positions ``[start, end)`` for reads.

        Only blocks committed to the manager's prefix-reuse tree are published by name; generated
        tokens are not committed. A windowed group lends only retained blocks required at ``end``;
        a block without backing memory is omitted. Access only the returned view's entries.

        The lease may fail immediately or while waiting if the request's cache is missing, inactive
        or changed, its inputs cannot be named, or the manager stopped or reset reuse. After a reuse
        reset the lender cannot publish again: new weights require a new manager and lender under a
        scope distinguishing those weights. A previously granted publish may finish.

        Args:
            request: The request whose blocks are read.
            start: Inclusive first token, a multiple of ``tokens_per_block``.
            end: Exclusive end token, a multiple of ``tokens_per_block``, at most the committed tokens.

        Returns:
            The lease, possibly already failed.

        Raises:
            ValueError: Negative, reversed or non-whole-block bounds, an end past committed tokens,
                or a range exceeding the attached implementation's capacity.
        """
        ...

    def lend_write(self, request: LlmRequest, start: int, end: int) -> Lease:
        """Lend writable memory for fetched blocks covering token positions ``[start, end)``.

        A windowed group lends the retained blocks required at ``end``. The holder marks entries
        that arrived whole and releases after backend access ends. Consecutive settled fetches can
        contribute to one prefix; lend the next segment only after every rank reports settled bounds.
        Abandoning a lease contributes none of its entries.

        The request must not execute while a fetch is unsettled. Previously computed tokens below
        ``start`` may contribute to the prefix; retained windows may limit where it can resume.
        Contents from ``start`` onward may be overwritten. Fetch only into a cache whose history is
        backed by valid data. Use ``resume_bounds``, not the fetch extent, to decide where to continue.
        Failure after the lending call need not undo cache changes.

        The lease may fail for the reasons in ``lend_read``, resource exhaustion, an unsettled prior
        fetch, incompatible cache history or reuse, unnamed inputs, a bidirectional multimodal run
        split by ``end``, or requested per-token outputs that a fetch cannot restore.

        Args:
            request: The request fetched into.
            start: Inclusive first token, a whole-block boundary no earlier than the committed tokens
                rounded down to whole blocks.
            end: Exclusive whole-block boundary, no later than the whole blocks before the last prompt
                token and, for a windowed cache, no earlier than its retained history.

        Returns:
            The lease, possibly already failed.

        Raises:
            ValueError: Negative, reversed or non-whole-block bounds, a start within committed whole
                blocks, an end past the last eligible prompt block, an end below a windowed cache's
                retained history, or a range exceeding the attached implementation's capacity.
                Validation precedes changes.
        """
        ...

    def resume_bounds(self, request: LlmRequest) -> Optional[ResumeBounds]:
        """Advance work and return local bounds once the request's fetches have settled.

        Query only while retained history not advanced by this lender is backed by valid cache
        contents. The bounds cannot detect a history that runs ahead of its computed data.

        Only entries incorporated into the request's cache count as fetched. Marking arrival alone
        does not make data usable. Abandoned or failed entries contribute nothing; an overwritten
        partially committed block can make the interval empty. Shrinking a cache invalidates delivered
        entries beyond its new capacity permanently: growing it again does not restore their contents.

        Combine rank and target/draft intervals as ``ResumeBounds`` specifies. If the effective
        interval is empty, drop the request's cache in every manager fetched into and compute from 0.

        Args:
            request: The request fetched into.

        Returns:
            ``None`` while a fetch is unsettled, otherwise the inclusive permitted resume interval.

        Raises:
            ValueError: The request has no cache, including after manager shutdown.
        """
        ...
