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
"""Public lending types and protocols, defined with NumPy and the standard library on the
contract's memory kinds and the shared page layout; their values hold no manager."""

from __future__ import annotations

import enum
import numbers
import struct
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np

from tensorrt_llm._torch.disaggregation.base.memory import MemoryKind

from ..cache_layout.page_layout import PageLayout

if TYPE_CHECKING:
    import torch

    from ...llm_request import LlmRequest

# A named row's fixed-length opaque object key.
_OBJECT_KEY_BYTES = 54
_INT64_MAX = (1 << 63) - 1


class _AllArrived:
    """The type of ``ALL_ARRIVED``, which has that one value; it pickles to the same value."""

    __slots__ = ()
    _value: Optional["_AllArrived"] = None

    def __new__(cls) -> "_AllArrived":
        if cls._value is None:
            cls._value = super().__new__(cls)
        return cls._value

    def __repr__(self) -> str:
        return "ALL_ARRIVED"

    def __reduce__(self) -> str:
        return "ALL_ARRIVED"


# Every row of a write lease's view arrived: ``Lender.return_writes`` takes it in place of masks.
ALL_ARRIVED = _AllArrived()


def _immutable(array: np.ndarray) -> np.ndarray:
    """``array`` where its memory is immutable bytes, which no one can write or make writable; else
    its values in such bytes of their own."""
    root = array
    while isinstance(root, np.ndarray) and root.base is not None:
        root = root.base
    if isinstance(root, bytes):
        return array
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _frozen_int64(values: Sequence[int]) -> np.ndarray:
    """Build contiguous int64 metadata directly on immutable bytes."""
    return np.frombuffer(struct.pack(f"={len(values)}q", *values), dtype=np.int64)


def _integer(name: str, value: object, least: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < least:
        raise ValueError(f"{name} must be at least {least}, got {value}")
    return int(value)


def _entry_integers(name: str, values: object, shape: Tuple[int, ...]) -> np.ndarray:
    """``values`` as an immutable ``int64`` array of ``shape``; ``ValueError`` for another dtype or
    shape."""
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer) or array.shape != shape:
        raise ValueError(
            f"{name} must be integers of shape {shape}, got {array.dtype} {array.shape}"
        )
    return _immutable(np.ascontiguousarray(array, dtype=np.int64))


def _region_indices(values: object) -> Tuple[int, ...]:
    """``values`` as a tuple of at least one distinct non-negative integer."""
    try:
        items = tuple(values)
    except TypeError:
        raise TypeError(
            f"region_indices must be a sequence of integers, got {type(values).__name__}"
        ) from None
    indices = tuple(_integer(f"region_indices[{i}]", item, 0) for i, item in enumerate(items))
    if not indices:
        raise ValueError("region_indices must name at least one region")
    if len(set(indices)) != len(indices):
        raise ValueError(f"region_indices must be distinct, got {indices}")
    return indices


@dataclass(frozen=True)
class StagingOptions:
    """Host capacity in fetch units, sized by whole blocks, windows and sinks.

    Ranges up to ``guaranteed_fetch_tokens`` always fit; larger ones may fit their slot budget. Each
    lease needs a contiguous run per pool group, granted FIFO: fragmentation can delay it despite
    sufficient total free slots. Empty leases are ready at first poll without joining the queue.

    Attributes:
        guaranteed_fetch_tokens: Tokens per sizing unit, rounded to whole blocks per group.
        capacity_in_fetches: Multiplier before the byte cap; no concurrency limit or guarantee.
        max_bytes: Optional local byte cap; attach raises ``ValueError`` below a stage-local fetch.

    Raises:
        TypeError: A field is not an integer, including bool.
        ValueError: A field is not positive."""

    guaranteed_fetch_tokens: int
    capacity_in_fetches: int = 1
    max_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "guaranteed_fetch_tokens",
            _integer("guaranteed_fetch_tokens", self.guaranteed_fetch_tokens),
        )
        object.__setattr__(
            self, "capacity_in_fetches", _integer("capacity_in_fetches", self.capacity_in_fetches)
        )
        if self.max_bytes is not None:
            object.__setattr__(self, "max_bytes", _integer("max_bytes", self.max_bytes))


class RowSelection(enum.Enum):
    """Which rows to lend, independently of ``MemoryKind``.

    NAMED selects whole blocks with object keys: reads require committed blocks; writes leave the
    final prompt token for computation. ALL includes uncommitted and partial blocks, can reach
    prompt end, and has ``object_keys=None``."""

    NAMED = "named"
    ALL = "all"


@dataclass(frozen=True)
class MemoryRegion:
    """A region of equal-sized slots; one row may span several regions.

    Describes backend access, not cache ownership or allocation. Register only accessible memory
    kinds; ``RegionHold`` defines address and backing-memory lifetime.

    Attributes:
        layout_label: Opaque label equal for equal layouts, independent of allocation identity.
        address: Base virtual address, positive; ``address + nbytes`` fits in ``int64``.
        nbytes: Byte extent, positive.
        slot_bytes: Bytes per slot, positive.
        slot_count: Back-to-back slots starting at ``address``, positive, within ``nbytes``.
        layer_group_indices: Manager-local groups using this region, a tuple of non-negative ints.
        memory_kind: Memory kind.

    Raises:
        TypeError: ``memory_kind`` is not a ``MemoryKind``, an integer field holds another type or
            a bool, or ``layer_group_indices`` is not a tuple.
        ValueError: An integer field is out of its range, or the slots exceed ``nbytes``."""

    layout_label: str
    address: int
    nbytes: int
    slot_bytes: int
    slot_count: int
    layer_group_indices: Tuple[int, ...]
    memory_kind: MemoryKind

    def __post_init__(self) -> None:
        if not isinstance(self.memory_kind, MemoryKind):
            raise TypeError(
                f"memory_kind must be a MemoryKind, got {type(self.memory_kind).__name__}"
            )
        for name in ("address", "nbytes", "slot_bytes", "slot_count"):
            object.__setattr__(self, name, _integer(name, getattr(self, name)))
        if self.slot_bytes * self.slot_count > self.nbytes:
            raise ValueError(
                f"{self.slot_count} slots of {self.slot_bytes} bytes exceed nbytes {self.nbytes}"
            )
        if self.address + self.nbytes > _INT64_MAX:
            raise ValueError(f"address {self.address} plus nbytes {self.nbytes} exceeds int64")
        indices = self.layer_group_indices
        if type(indices) is not tuple:
            raise TypeError(f"layer_group_indices must be a tuple, got {type(indices).__name__}")
        indices = tuple(_integer("a layer group index", index, 0) for index in indices)
        object.__setattr__(self, "layer_group_indices", indices)


def _check_disjoint(regions: Sequence[MemoryRegion]) -> None:
    """``ValueError`` where two regions of one memory kind overlap; each kind sorted by address."""
    spans_by_kind: dict = {}
    for position, region in enumerate(regions):
        spans = spans_by_kind.setdefault(region.memory_kind, [])
        spans.append((region.address, region.address + region.nbytes, position))
    for kind, spans in spans_by_kind.items():
        spans.sort()
        for (_, end, first), (start, _, second) in zip(spans, spans[1:]):
            if start < end:
                raise ValueError(f"regions {first} and {second} overlap in {kind.name} memory")


@dataclass(frozen=True, eq=False)
class LentLayerGroupRows:
    """One layer group's rows in aligned immutable arrays, copied only when not already immutable.

    Each row describes this rank's logical page across the group's layers. Recurrent state has
    at most one row, ordinal 0, at the range's end. Ordinals and slots need not be consecutive.
    ``slots`` and ``region_indices`` are given together or not at all, and object keys are
    independent of them. In rows a lender lends, every slot is below the ``slot_count`` of each
    ``region_indices`` region: the lender fails a lease rather than lend a slot that is not.

    Under ``attach_lender``'s scope and caller duties, equal keys mean interchangeable bytes.
    Keys are opaque equality-only identities, not checksums: their width (54 bytes) is API, their
    format is not. Format/layout changes make stored objects miss rather than hit incorrectly.

    Attributes:
        layer_group_index: Manager-local group index.
        page_ordinals: Logical page ordinals, converted to ``int64 (n,)``. Without Helix, page
            ``b`` covers ``[b * t, (b + 1) * t)`` for ``t=PageLayout.tokens_per_page``; under Helix,
            this rank's in-place page of block ``b`` has ordinal ``b * cp_size + cp_rank``.
        object_keys: ``uint8 (n, 54)`` opaque keys, distinct from 32-byte reuse keys, or ``None``.
        slots: ``int64 (n,)`` non-negative slots. Row ``i`` concatenates ``slot_bytes`` bytes
            from ``address + slots[i] * slot_bytes`` in each ``region_indices`` region.
        region_indices: Nonempty, distinct positions in ``Lender.regions``, in row-byte order; not
            pool or layer-group IDs.

    Raises:
        TypeError: A noninteger index (including bool), or nonsequence ``region_indices``.
        ValueError: Wrong array shape, invalid key or slot dtype, negative index or slot,
            ``slots`` without ``region_indices`` or the reverse, or empty or duplicate regions.

    Performance (non-normative):
        Construction fixes each array's dtype and shape, makes it immutable, and checks that
        ``layer_group_index``, ``region_indices`` and ``slots`` hold non-negative integers and that
        ``region_indices`` is non-empty and distinct, so holders need not check these again;
        ``page_ordinals`` gets no value check. A lender gives all rows of a pool group that lie in
        the same regions one ``region_indices`` tuple object. Read the arrays with whole-array
        operations and keep references instead of looping over rows or copying defensively; the
        lender neither changes nor reuses them."""

    layer_group_index: int
    page_ordinals: np.ndarray
    object_keys: Optional[np.ndarray] = None
    slots: Optional[np.ndarray] = None
    region_indices: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "layer_group_index", _integer("layer_group_index", self.layer_group_index, 0)
        )
        page_ordinals = _immutable(np.ascontiguousarray(self.page_ordinals, dtype=np.int64))
        if page_ordinals.ndim != 1:
            raise ValueError(
                f"page_ordinals must be one-dimensional, got shape {page_ordinals.shape}"
            )
        object.__setattr__(self, "page_ordinals", page_ordinals)
        if (self.slots is None) != (self.region_indices is None):
            raise ValueError("slots and region_indices are given together or not at all")
        if self.object_keys is not None:
            object_keys = np.asarray(self.object_keys)
            if object_keys.dtype != np.uint8 or object_keys.shape != (
                len(page_ordinals),
                _OBJECT_KEY_BYTES,
            ):
                raise ValueError(
                    f"object_keys must be uint8 ({len(page_ordinals)}, {_OBJECT_KEY_BYTES}), "
                    f"got {object_keys.dtype} {object_keys.shape}"
                )
            object.__setattr__(self, "object_keys", _immutable(np.ascontiguousarray(object_keys)))
        if self.slots is not None:
            slots = _entry_integers("slots", self.slots, page_ordinals.shape)
            if np.any(slots < 0):
                raise ValueError(f"slots must be non-negative, got {slots[slots < 0][:8]}")
            object.__setattr__(self, "slots", slots)
            object.__setattr__(self, "region_indices", _region_indices(self.region_indices))

    def __len__(self) -> int:
        return int(self.page_ordinals.shape[0])

    def select(self, mask: np.ndarray) -> LentLayerGroupRows:
        """Select rows with one boolean per row, preserving order and array alignment.

        Returns the same group and regions in immutable arrays of its own; raises ``ValueError``
        for a nonboolean or wrong-shape mask."""
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != self.page_ordinals.shape:
            raise ValueError(
                f"mask must be bool of shape {self.page_ordinals.shape}, got {mask.dtype} {mask.shape}"
            )
        return LentLayerGroupRows(
            self.layer_group_index,
            self.page_ordinals[mask],
            None if self.object_keys is None else self.object_keys[mask],
            None if self.slots is None else self.slots[mask],
            self.region_indices,
        )

    @classmethod
    def _unchecked(
        cls,
        layer_group_index: int,
        page_ordinals: np.ndarray,
        object_keys: Optional[np.ndarray] = None,
        slots: Optional[np.ndarray] = None,
        region_indices: Optional[Tuple[int, ...]] = None,
    ) -> LentLayerGroupRows:
        """The rows the constructor makes of these values, taken as they are, for values that pass
        its checks unchanged: an ``int`` index, contiguous ``int64 (n,)`` ordinals and slots and
        ``uint8 (n, 54)`` keys on immutable bytes, slots and region indices given together or not
        at all, non-negative slots, and a tuple of distinct non-negative ``int`` region indices."""
        rows = object.__new__(cls)
        object.__setattr__(
            rows,
            "__dict__",
            {
                "layer_group_index": layer_group_index,
                "page_ordinals": page_ordinals,
                "object_keys": object_keys,
                "slots": slots,
                "region_indices": region_indices,
            },
        )
        return rows


@dataclass(frozen=True, eq=False)
class LeaseView:
    """Ready rows of a lease of token range ``[start, end)``, at most one collection per layer
    group.

    ``Lease.poll`` grants stable access only to the described memory, until the lease is returned.
    Of each row's page only the range's tokens are lent: a read keeps them unchanged, a write
    delivers them, and the page's other bytes carry no promise. A recurrent-state row holds the
    state at ``end``; a lend whose state stands short of its end narrows ``end`` to it. A write
    view's rows never overlap. The row collections and immutable arrays remain valid afterward: the
    lender never changes or reuses them. Holders must not reassign array headers (shape, dtype or
    strides), which would change every holder's view. All rows use the requested memory kind in
    ``Lender.regions``.

    Attributes:
        groups: Collections whose tuple positions need not equal ``layer_group_index``.
        start: The lend's ``start``.
        end: The lend's ``end``.

    Raises:
        TypeError: A collection is not ``LentLayerGroupRows``, or a bound is not an integer.
        ValueError: A layer group repeats, or not ``0 <= start <= end``."""

    groups: Tuple[LentLayerGroupRows, ...]
    start: int
    end: int

    def __post_init__(self) -> None:
        groups = tuple(self.groups)
        seen = set()
        for rows in groups:
            if not isinstance(rows, LentLayerGroupRows):
                raise TypeError(f"groups hold LentLayerGroupRows, got {type(rows).__name__}")
            if rows.layer_group_index in seen:
                raise ValueError(f"layer group {rows.layer_group_index} appears twice")
            seen.add(rows.layer_group_index)
        object.__setattr__(self, "groups", groups)
        start = _integer("start", self.start, 0)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", _integer("end", self.end, start))

    @property
    def num_rows(self) -> int:
        """Total rows across groups; the same ordinal in two groups counts twice."""
        return sum(len(rows) for rows in self.groups)

    def row_masks(self, fill_value: bool = False) -> Tuple[np.ndarray, ...]:
        """Return writable boolean masks filled with ``fill_value``, in ``groups`` order.

        Mask ``i[j]`` selects row ``groups[i].page_ordinals[j]``, not ordinal ``j``. Empty
        groups retain empty masks. These are the masks ``Lender.return_writes`` accepts."""
        return tuple(np.full(len(rows), bool(fill_value), dtype=bool) for rows in self.groups)

    @classmethod
    def _unchecked(cls, groups: Tuple[LentLayerGroupRows, ...], start: int, end: int) -> LeaseView:
        """The view the constructor makes of these values, taken as they are, for a tuple of
        ``LentLayerGroupRows`` of distinct layer groups and ``int`` bounds ``0 <= start <= end``."""
        view = object.__new__(cls)
        object.__setattr__(view, "__dict__", {"groups": groups, "start": start, "end": end})
        return view


# TODO: V2 has no public method to resume a request at a specified position.
class ResumeBounds(NamedTuple):
    """Inclusive local resume interval, counting valid original and incorporated fetched data.

    Choose ``min_position <= p <= max_position``, no lower than the current context cursor.
    Bounds exclude unavailable history, unsupported rewinds and positions strictly inside
    bidirectional multimodal runs. The maximum need not be block-aligned; local computation or
    an ALL fetch can reach prompt end, while a NAMED fetch leaves its last token for computation.

    Intersect bounds across the request's cache-holding ranks, including pipeline stages and its
    own attention data-parallel rank: take the largest minimum and smallest maximum. Intersect
    participating joint-reuse target/draft pools and fetch both before advancing their shared
    cursor; P/D receives that lend no draft pool exclude its bounds. An empty intersection requires
    dropping every fetched cache and computing from zero, without refetching into those caches.

    Attributes:
        max_position: Last permitted position.
        min_position: First permitted position, unless the current cursor is higher."""

    max_position: int
    min_position: int


class HoldReason(enum.Enum):
    """Why lending holds a page, as ``HeldPages.by_reason`` counts it; a page has one reason.

    Attributes:
        LOCKED: A lease ``lend_read`` or ``lend_write`` lent locks the page in a cache its request
            still holds.
        DEFERRED_CLOSE: The page belongs to a cache freed while a lease locked some of it, a cache
            the request left for a new one included; it stays allocated until that cache's last
            lease is returned."""

    LOCKED = "locked"
    DEFERRED_CLOSE = "deferred_close"


@dataclass(frozen=True)
class HeldPages:
    """Pages lending holds on one rank and pipeline stage, per local pool group (keys of
    ``PageLayout.pool_slot_bytes``; zero groups omitted).

    A page counts once per cache however many of that cache's leases lock it; a page several caches
    share counts once per cache, so the counts are upper bounds.

    Attributes:
        deduct: Pages ``DEFERRED_CLOSE`` holds, including pages the manager may also count against
            live requests. They stay with the cache they belong to, already outside its free pages,
            so admission does not subtract them again.
        unreclaimable: Lease rows ``LOCKED`` holds and pages ``DEFERRED_CLOSE`` holds together.
            This omits unleased pages of live locked caches: a scheduler that cannot preempt a
            locked cache must reserve its entire allocation, not just these rows.
        by_reason: Per ``HoldReason``, the pages it holds; reasons holding none are omitted.
        version: Increases whenever a count changes."""

    deduct: Mapping[int, int]
    unreclaimable: Mapping[int, int]
    by_reason: Mapping[HoldReason, Mapping[int, int]]
    version: int


# What ``Lender.return_writes`` takes of a write lease's rows: masks, or ``ALL_ARRIVED``.
Arrived = Union[Sequence[np.ndarray], _AllArrived]


@runtime_checkable
class Lease(Protocol):
    """One lent range: poll until it gives its view or fails, then return it.

    Follow ``Lender``'s thread, progress and shutdown duties. Backend threads only inspect ready
    views and access the memory they describe, reporting completion through their own channel;
    they call no API methods.

    Caller obligations:
        - Poll a lease until it returns its view or fails, and no further: the view, and access to
          the memory it describes, stay valid without another call until the lease is returned.
          Combine ranks' outcomes for distributed requests; do not wait for arrivals while
          backends need progress.
        - Return every lease once backend access ends, failed ones too: a write lease whose view
          was returned through ``Lender.return_writes``, which counts its arrived rows into the
          cache it was lent from, any other through ``release``. A lease of a cache its request
          no longer holds is returned the same way and counts nothing. Dropping a lease does not
          return it.
        - Failures are final and request-level: compute locally or refuse. Local state, cache,
          layers, progress, capacity and reuse reset can fail a lease at the call, without mutation.
        - ``poll`` after the lease was returned raises ``RuntimeError`` for caller misuse.
          ``RuntimeError`` from ``lend_*`` means the lender stopped or shut down.
        - After that error or an unlisted exception, follow ``attach_lender``'s failure duties;
          unlisted errors may stop lending permanently, and a stopped lender keeps returned views
          held and fails unseen views.

    Performance (non-normative):
        Call on the manager thread. ``poll`` does not wait: a DEVICE lease asks its ready event and
        any pending connector load, a HOST lease reads what ``Lender.progress`` found and asks its
        copy's event at most once per round of it. A returned view costs nothing to keep, and
        ``return_resumes_at_end`` is an attribute read."""

    def poll(self) -> Optional[LeaseView]:
        """Return the view once ready, the same view at every later call, or ``None`` while
        pending or failed.

        Read views hold readable bytes; write views permit backend writes. Raises ``RuntimeError``
        once the lease was returned."""
        ...

    @property
    def failure(self) -> Optional[str]:
        """Why the lease will never be ready, once that is known; for logs only."""
        ...

    @property
    def writable(self) -> bool:
        """Whether ``lend_write`` lent the lease, so a backend may write its view; fixed for the
        lease's life."""
        ...

    def release(self) -> None:
        """Return the lease once all backend access ended; idempotent and legal in every state.

        Required even on failure; it does not cancel in-flight access. It ends the protection of the
        view's memory, not the view: its arrays stay as they were. A write lease released after its
        view was returned delivers none of its rows, as if returned through ``Lender.return_writes``
        with no row arrived; released before, it abandons its write. Its unlock retries a commit its
        locks held back, and a freed request's cache closes with its last lock; an error in that
        work raises from ``release``."""
        ...

    @property
    def return_resumes_at_end(self) -> bool:
        """Whether returning this lease now through ``Lender.return_writes``, every row arrived,
        gives bounds whose interval holds its view's ``end``. Only a write lease lent in place whose
        view ``poll`` returned can be ``True``; it turns ``False`` for good once its cache shrinks or
        is freed, the manager resets reuse or shuts down, or the lender stops."""
        ...


@runtime_checkable
class RegionHold(Protocol):
    """Keep all regions' addresses and backing memory valid, including across shutdown.

    Take on the manager thread, at construction the construction thread, before registering
    accessible regions. Deregister before shutdown; release only after confirmation, retaining
    unconfirmed holds. Dropping a hold does not release it."""

    def release(self) -> None:
        """Confirm deregistration and no possible backend access, on the manager thread.

        Idempotent and legal in every state."""
        ...


# TODO: keys require reserved pages, and publication includes only the end's retained window;
# earlier forks or longer turns can miss rows and require full recomputation.
@runtime_checkable
class Lender(Protocol):
    """Lend KV blocks while the manager retains cache ownership.

    Lender, lease and hold calls run serially on the manager thread. Backend threads access only
    ready views and their described memory, held until access ends. See ``Lease`` for failure
    handling, ``RegionHold`` for registration lifetime and ``ResumeBounds`` for paired managers.

    Caller obligations:
        - Hold regions before registration or first backend access; deregister and release only
          after backend access is impossible.
        - Call ``progress`` once each executor iteration while a lease is open or a fetch is
          unsettled, even without new requests: only it advances leases waiting for their view and
          fetches waiting to settle. Pending leases have no progress guarantee: return those
          waiting too long or free their requests.
        - Pass each lend the event after which no work in flight touches the lent pages, or
          ``None`` where none does; the lender records no event of its own.
        - Return each write lease whose view was returned through ``return_writes``, saying
          whether its arrived rows may enter reuse. Where an entry for the request's current
          cache gives no bounds, query ``poll_resume_bounds`` once each iteration, after
          ``progress``, until it gives them, and no more then.
        - Shutdown: stop executor/backends, confirm deregistration, return leases and release holds,
          then shut down the manager.
        - Fetch both participating joint-reuse pools before moving their shared cursor; a P/D receive
          that lends no draft pool does not intersect that pool's bounds.

    Performance (non-normative):
        Call on the manager thread; DEVICE lends pages without copying KV bytes, and HOST queues
        its copies on the execution stream without waiting when staging is page-locked. Only
        ``progress`` runs a round of lender progress, which has work only while HOST leases or
        their fetches are open. A lend's cost grows with the layer groups, and its view's arrays
        with the rows lent; ``return_writes`` of a DEVICE lease every row of which arrived, lent
        without a rule-specific path, grows little with the rows. ``held_page_counts`` counts again
        only the caches whose locks changed, or that were freed while locked, since its previous
        call, and costs O(1) when there are none. Constructing a ``torch.cuda.Event`` lets go of
        the GIL: give lends events made once and reused, not one made each iteration, and record a
        reused event again only after its previous record completed or on the same stream, as a
        waiting lease reads only the latest record."""

    @property
    def regions(self) -> Tuple[MemoryRegion, ...]:
        """Regions containing every ready view, each with one memory kind, fixed for the lender's
        life: a lending method a later optional protocol adds gives its own.

        Register only accessible kinds; leases refer only to their requested kind. Registering a
        region exposes all of it: any party holding what a backend registered can access the whole
        region, as leases and holds bind this process alone. Holds preserve region positions,
        addresses and extents. Unsupported kinds are not listed and requests for them raise
        ``ValueError``. Equal layout labels mean equal layouts. Regions are well formed and regions
        of one memory kind do not overlap. Address ordering, adjacency, a slot's lying within one
        allocation and import into another process are not guaranteed: a backend needing one
        validates it. See ``RegionHold`` for deregistration before shutdown; open leases/holds can
        retain memory beyond it."""
        ...

    @property
    def page_layout(self) -> PageLayout:
        """This rank's page layout, fixed for the lender's life: each local layer group's kind,
        windows and global layer ids, each buffer's place in a page, the rank's topology, element
        type and quantization, and the position in ``regions`` of each pool's region."""
        ...

    def held_page_counts(self) -> HeldPages:
        """Pages lending holds on this rank and pipeline stage, per local pool group (``HeldPages``).
        Compare them across ranks and stages to find holds that differ. These diagnostics alone do
        not size recovery room: live locked caches may retain unleased pages too. The value stays
        the same, ``version`` included, while no count changes, also after the manager's shutdown."""
        ...

    def hold_regions(self) -> RegionHold:
        """Return a new hold on all regions, taken before registration on the manager thread.

        It neither registers regions nor grants a lease; it remains open until release and is
        inert if taken after manager shutdown."""
        ...

    def lend_read(
        self,
        request: LlmRequest,
        start: int,
        end: int,
        *,
        memory_kind: MemoryKind,
        rows: RowSelection,
        ready_event: Optional[torch.cuda.Event],
    ) -> Lease:
        """Lend token range ``[start, end)`` for reads in the requested ``memory_kind``.

        ``rows=NAMED`` publishes the whole blocks the manager has committed, generated or not. It
        lends a layer group holding a PRIVATE layer (CSA2's) nothing, its collection empty, and an
        OPTIONAL group the prefix claim did not select (CSA2's) none of the claimed blocks. ALL
        lends retained blocks, including the uncommitted partial tail, without names. Several beams
        lend beam 0's pages, which the manager expands at its first generation step. Windowed groups
        include only backed blocks required at ``end``. Access only returned rows. Until this
        worker's context is computed, an in-place read of the cache's own pages leaves out each
        layer group whose window is shorter than that context (no rows, no lock), so later context
        steps move its window on; other groups are lent, and a read once the context is computed
        takes them all. This worker's context ends at the request's
        ``context_current_position + context_remaining_length``, at most the prompt; under Helix, at
        the prompt.

        The lease fails at the call where this rank's cache does not hold the range: NAMED past its
        committed tokens; ALL past what it holds or a finished context step computed, or, until this
        worker's context is computed, inside a block; with recurrent state, anywhere but the
        prompt's end between the context and the first generation step; or needing more than this
        rank's capacity. A read in place also fails at the call over rows of an unsettled fetch into
        the request, and over rows below the history that a fetch lent or grew the cache over
        without delivering them. Missing or inactive caches, unnameable inputs and, by name, a reuse
        reset of this rank's manager fail it at the call too; a freed, suspended or replaced cache,
        the manager's shutdown and the lender's stop can fail it while it waits. After reset no new
        named publication is possible; new weights need a new manager, lender and distinguishing
        scope. Granted publishes may finish.

        Args:
            request: Request to read; the lease binds to the cache it holds now.
            start: Nonnegative whole-block boundary.
            end: Exclusive end, at least ``start``; a whole-block boundary for NAMED.
            memory_kind: Required ``MemoryKind``.
            rows: Required ``RowSelection``.
            ready_event: Required: an event with ``query()``, and ``wait()`` where rows are not the
                cache's own pages, such as a ``torch.cuda.Event``, recorded after every piece of
                work in flight that writes the range's pages, or ``None`` where none is in flight.
                ``poll`` gives no view before it completes. Such work includes the context steps
                that computed the range and any later move or write of the pages, such as a commit,
                a resume or a fetch into them; where these run in order, an event recorded after the
                last covers them all. Until the manager adopts the request's cache into its
                execution order, a lease in place also waits for the manager's allocation fence, so
                the event need only cover work in that execution order.

        Returns:
            A lease, possibly already failed.

        Raises:
            TypeError: Invalid ``memory_kind``, ``rows`` or ``ready_event`` type.
            ValueError: Unsupported kind/selection or invalid bounds above, alike on every rank.
            RuntimeError: The manager shut down, the lender stopped after an error, or a manager
                or CUDA operation failed. Stop lending through it after this error."""
        ...

    def lend_write(
        self,
        request: LlmRequest,
        start: int,
        end: int,
        *,
        memory_kind: MemoryKind,
        rows: RowSelection,
        ready_event: Optional[torch.cuda.Event],
    ) -> Lease:
        """Lend writable ``memory_kind`` for fetched token range ``[start, end)``.

        ALL includes the unnamed partial tail; windows lend retained rows required at ``end``. Each
        group skips its committed whole blocks. Once committed tokens pass the claimed reuse prefix,
        every group counts all of them, previously incomplete groups included; a group committed
        past ``end`` lends no rows. Before that, unselected OPTIONAL and PRIVATE groups hold none of
        the claimed prefix, so fetching ALL from 0 fills their gaps. NAMED always omits groups
        holding PRIVATE layers. Non-REQUIRED attention groups need windows. Multiple beams lend
        beam 0; the manager expands the rest at the first generation step.
        Window margins beyond ``LayerGroupDesc.read_window`` may stay unwritten without lowering
        ``poll_resume_bounds``; the lender then keeps the cache out of reuse. Follow
        ``return_writes`` for other permitted omissions. Extend a prefix only after every rank's
        prior fetch settles.

        Do not schedule an unsettled fetch. Original history must be valid; computed tokens below
        ``start`` can contribute, but retained windows constrain resumption and bytes from ``start``
        may be overwritten. Resume from settled bounds, not the requested extent. Abandoned rows
        contribute nothing; failure after this call need not undo cache changes.

        Failures include ``lend_read``'s reasons, exhaustion, unsettled prior fetches, incompatible
        history/reuse, unnamed inputs, split bidirectional runs and unrestorable per-token outputs.
        Rank-local failures also include an ALL end inside a block unless at prompt end or declared
        history (a CSA2 remote tail's start), a start inside an externally served prefix, a
        recurrent end short of the prompt, or a windowed end below history. These failures precede
        mutation.

        Args:
            request: Request fetched into; the lease binds to the cache it holds now.
            start: Nonnegative whole-block boundary.
            end: Exclusive end, at least ``start``. NAMED ends at a whole block before the final
                prompt token; ALL ends no later than the prompt's end.
            memory_kind: Required ``MemoryKind``.
            rows: Required ``RowSelection``.
            ready_event: Required: an event with ``query()``, and ``wait()`` where rows are not the
                cache's own pages, such as a ``torch.cuda.Event``, recorded after every piece of
                work in flight that still uses or resets the range's pages, such as the request's
                admission and a reset of its recurrent state, or ``None`` where none is in
                flight. A NAMED grow also requires ``wait()`` if the event is not yet complete, so
                the grow can follow that work in the cache's execution order. ``poll`` gives no
                view of the range's pages before it completes; a view of other memory may come
                first, and the rows counted from it reach the pages after it.
                Until the manager adopts the request's cache into its execution order, a lease in
                place also waits for the manager's allocation fence, so the event need only cover
                work in that execution order.

        Returns:
            A lease, possibly already failed.

        Raises:
            TypeError: Invalid ``memory_kind``, ``rows`` or ``ready_event`` type.
            ValueError: Unsupported kind/selection or invalid bounds above, alike on every rank.
            RuntimeError: As ``lend_read``; stop lending through it after this error."""
        ...

    def return_writes(
        self, entries: Sequence[Tuple[Lease, Arrived, bool]]
    ) -> List[Optional[ResumeBounds]]:
        """Return write leases whose views were returned, counting the rows that arrived, and give
        each lease's resume bounds.

        Each entry is ``(lease, arrived, reusable)``: a write lease this lender lent whose view
        ``poll`` returned; ``ALL_ARRIVED``, or one boolean mask per ``LeaseView.groups``
        collection, in tuple order, selecting rows rather than ordinals (``LeaseView.row_masks``
        gives masks to fill); and whether the arrived rows may enter reuse. A read-only mask of
        dtype ``bool`` whose type is exactly ``numpy.ndarray`` is kept as it is, and its giver
        writes neither it nor memory it shares afterwards; any other mask is copied.

        Call only after the transfer's outcome is settled on every participating rank and all
        backend access to the returned views has ended. An arrival notification alone establishes
        neither condition. Returning a lease may let the cache consume its returned memory
        immediately; ``reusable`` does not replace either completion requirement.

        An entry counts its arrived rows into the cache the lease was lent from, then returns the
        lease as ``Lease.release`` does. Arrived rows must have no writes in flight. For ALL, every
        ``transferred=True`` buffer's bytes the request reads must have arrived; buffers may stay
        unwritten if not transferred, read only by one-model draft layers absent at the peer, or
        overwritten before the request reads them, such as CSA2 remote-tail decoder rows, and such
        an entry passes ``reusable=False``. NAMED requires the whole object. ``reusable=False``
        keeps the cache from committing anything more for reuse until it closes; so does a lent
        window-margin row that did not arrive, and a later ``True`` undoes neither. Rows of a cache
        the request no longer holds count nothing; still return the lease, also after the request
        exits. Only counted rows make data usable.

        Every entry is checked before any is returned. The validation errors listed below change
        nothing: correct the entries and retry; lending, ``progress`` and ``poll_resume_bounds``
        remain permitted. Any error once returning entries begins raises after all entries'
        returns have been attempted. It permits cleanup only: stop lending, progress, lease
        polling and bounds queries, even for other entries returned successfully. Follow
        ``attach_lender``'s failure duties: finish access, release remaining leases and holds,
        drop affected fetched caches and compute from zero. Exception
        type alone does not distinguish validation from operation failure; if unsure, use cleanup
        only. Returning entries need not undo cache changes made before an error.

        Args:
            entries: The ``(lease, arrived, reusable)`` entries, each lease at most once.

        Returns:
            Per entry, in order, the bounds of the lease's cache as ``poll_resume_bounds`` gives
            them once the entries are returned; ``None`` where they are not settled yet, as when
            another fetch into the request is pending or the entry's rows are not counted yet (then
            query ``poll_resume_bounds``), or where that cache is no longer the request's or the
            request has none (no bounds apply).

        Raises:
            TypeError: Entry validation: an entry is not a ``(lease, arrived, reusable)`` triple,
                or ``reusable`` is not a ``bool``.
            ValueError: Entry validation: a lease this lender did not lend, or masks of the wrong
                count, shape or dtype.
            RuntimeError: Entry validation: a read lease, a lease returned before, in this call or
                earlier, or one whose view ``poll`` did not return (caller misuse)."""
        ...

    def progress(self) -> None:
        """Advance the lender's pending work: leases waiting for their view, fetches waiting to
        settle, and what returned leases leave to free.

        Call it once each executor iteration while a lease is open or a fetch is unsettled;
        ``Lease.poll`` and ``poll_resume_bounds`` report what it found."""
        ...

    def poll_resume_bounds(self, request: LlmRequest) -> Optional[ResumeBounds]:
        """Return local bounds of the request's current cache once all its fetches settle, as
        ``progress`` last found.

        History not advanced by this lender must already contain valid data: bounds cannot detect
        fabricated history. Only counted rows count; a fetch whose rows were not counted yet is
        unsettled. Failed or abandoned rows, and rows of the request's earlier caches, count
        nothing, and overwriting a partially committed block may empty the interval. Shrinking
        permanently invalidates deliveries beyond new capacity; regrowing does not restore them.
        Apply ``ResumeBounds``'s intersection/drop-and-recompute rule.

        Returns:
            ``None`` while unsettled, otherwise inclusive bounds for ``request``; an empty
            interval, its maximum below its minimum, once the lender stopped for good.

        Raises:
            ValueError: No request cache, including after manager shutdown."""
        ...
