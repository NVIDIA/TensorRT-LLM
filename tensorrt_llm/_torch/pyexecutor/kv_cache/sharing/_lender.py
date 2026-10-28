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
"""HOST/NAMED staging and DEVICE page leases; public contracts live in ``attach_lender`` and
``_types``. Calls run on the manager thread and order copies and events on its stream; waiting reads
recheck pages, writes count only unchanged pages, unsafe copies quarantine slots."""

from __future__ import annotations

import collections
import dataclasses
import numbers
import struct
import traceback
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Deque, Mapping, Sequence

import numpy as np
import torch
from cuda.bindings import driver as drv
from cuda.bindings import runtime as cudart

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger

from ..page_layout import PageLayout
from . import _manager, _ranges
from ._fetch_ledger import _FetchLedger
from ._identity import Identity
from ._layout import ManagerLayout, derive_layout, layout_id
from ._page_locks import CheckedBlocks, LockMode, PageLock, PageLocks
from ._slots import SlotQueue, SlotRuns, slot_counts
from ._types import (
    ALL_ARRIVED,
    HeldPages,
    LeaseView,
    LentLayerGroupRows,
    MemoryKind,
    MemoryRegion,
    ResumeBounds,
    RowSelection,
    StagingOptions,
    _check_disjoint,
    _frozen_int64,
    _immutable,
)

if TYPE_CHECKING:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import _KVCache

    from ...llm_request import LlmRequest
    from ..kv_cache_manager_v2 import KVCacheManagerV2
    from ._fetch_ledger import _FetchRecord

# Staging memory and page-index buffers kept until exit, by identity, oldest first; only a clean
# shutdown removes one. GIL-atomic dict changes let lenders on any thread skip a lock.
_retained_until_exit: dict[int, object] = {}

_REASON_SHUT_DOWN = "the KV cache manager shut down"
_REASON_REUSE_RESET = (
    "the KV cache manager reset its reuse state, after which nothing is lent by name"
)
_REASON_STOPPED = (
    "the lender stopped after an error it logged while the KV cache manager freed a request"
)
_REASON_STOPPED_AT_SHUTDOWN = (
    "the lender stopped after an error it logged while the KV cache manager shut down"
)
_REASON_SUSPENDED = "the request's cache is suspended"
_REASON_RESERVED = (
    "a KV cache connector holds a prefix reservation for the request, a load not yet started into "
    "pages its cache already counts"
)
_REASON_PLACEHOLDERS = (
    "the request carries multimodal data without digests, which its object_keys do not cover"
)
_REASON_ENCODER = "the request has encoder input, which its object_keys do not cover"
_REASON_SCRATCH = (
    "the request's cache has SWA scratch reuse on: each capacity change keeps its history within "
    "the scratch rewind of the old capacity, which a fetch's grow breaks"
)
_REASON_CONTEXT_OUTPUTS = (
    "the request returns context logits or asks for additional model outputs, which the executor "
    "gives only for computed positions and keeps across a rewind and a recompute pause"
)


def _retained() -> tuple[object, ...]:
    """What is kept until exit, oldest first; for tests."""
    return tuple(_retained_until_exit.values())


def _retain_until_exit(owner: object) -> None:
    """Keep ``owner`` until the process exits, or until ``_stop_retaining``."""
    _retained_until_exit[id(owner)] = owner


def _stop_retaining(owner: object) -> None:
    """Drop ``owner`` from the keep list, compared by identity."""
    _retained_until_exit.pop(id(owner), None)


class _HostMemory:
    """Host memory of exactly ``nbytes``, page-locked where pinning pays off. Pinned memory goes
    only through ``free``, once; pageable memory goes with the object, which the keep list holds
    until exit, so memory kept until exit stays mapped either way."""

    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes
        self._pageable: np.ndarray | None = None
        if prefer_pinned():
            # The driver pins the size asked; torch's pinned allocator rounds it to a power of two.
            error, address = cudart.cudaHostAlloc(nbytes, cudart.cudaHostAllocDefault)
            if error != cudart.cudaError_t.cudaSuccess:
                raise MemoryError(f"pinning {nbytes} bytes of staging memory failed: {error}")
            self.address = int(address)
        else:
            self._pageable = np.empty(nbytes, dtype=np.uint8)
            self.address = int(self._pageable.ctypes.data)

    def free(self) -> None:
        """Release the memory; later calls do nothing."""
        address, self.address = self.address, 0
        if not address or self._pageable is not None:
            self._pageable = None
            return
        (error,) = cudart.cudaFreeHost(address)
        if error != cudart.cudaError_t.cudaSuccess:
            logger.warning(f"KV cache lender: freeing the staging memory failed: {error}")


def _allocate_retained(nbytes: int) -> _HostMemory:
    """One host allocation of ``nbytes`` for the staging regions, put on the keep list."""
    memory = _HostMemory(max(int(nbytes), 1))
    _retain_until_exit(memory)
    return memory


def _checked_layout(manager: KVCacheManagerV2) -> ManagerLayout:
    """The attach's checks: a V2 manager without a lender, sparse buffers or non-Helix context
    parallelism, and under Helix no recurrent state or window. A stage lends its own layers."""
    _manager.require_v2(manager)
    mapping = _manager.mapping(manager)
    # TODO: under context parallelism other than Helix the lender cannot place a rank's pages.
    cp_size = int(mapping.cp_size)
    if cp_size > 1 and not mapping.has_cp_helix():
        raise ValueError(
            f"context parallelism (cp_size={cp_size}) is supported only as Helix: other ranks "
            "would give different pages the same object_keys"
        )
    if _manager.attached_lender(manager) is not None:
        raise ValueError("a lender is already attached to this KV cache manager")
    layout = derive_layout(manager)
    unbounded = [
        lg
        for lg in range(layout.num_layer_groups)
        if not layout.recurrent[lg]
        and layout.windows[lg] is None
        and not (layout.prefix_required[lg] if layout.prefix_required else True)
    ]
    if unbounded:
        raise ValueError(f"non-REQUIRED layer groups {unbounded} must have a retained window")
    if layout.cp[1] > 1:
        recurrent = [lg for lg, state in enumerate(layout.recurrent) if state]
        windowed = [
            lg
            for lg in range(layout.num_layer_groups)
            if layout.windows[lg] is not None or layout.sink_blocks[lg]
        ]
        if (
            recurrent
            or windowed
            or _manager.model_has_recurrent_state(manager)
            or _manager.model_has_attention_window(manager)
        ):
            raise ValueError(
                f"under Helix the lender lends no recurrent state (layer groups {recurrent}) and "
                f"no window (layer groups {windowed}): native's transfer under Helix is not shown "
                "to carry them end to end either"
            )
    # TODO: a sparse layer group's page index can name a host slot, and its locked pages move
    # between GPU and host while a lease holds their indices (decode offloads, prefill promotes).
    sparse = [lg for lg, flag in enumerate(layout.sparse) if flag]
    if sparse or _manager.model_has_sparse_buffers(manager):
        raise ValueError(
            f"layer groups {sparse} hold sparse buffers, whose read-only pages a cache can lock "
            "in host memory"
        )
    return layout


def _no_names(manager: KVCacheManagerV2, layout: ManagerLayout) -> str | None:
    """Why nothing is lent by name, or ``None``: a name covers a committed block's tokens alone, so
    its bytes must match across ranks, hold no state, read no later token and stay as computed."""
    if layout.cp[1] > 1:
        return "under Helix a block's tokens lie on all its ranks"
    recurrent = [lg for lg, state in enumerate(layout.recurrent) if state]
    # TODO: the manager keeps a request's recurrent state at a commit's end in a snapshot of the
    # reuse tree, whose page no request's cache lends, so no state is lent by name.
    if recurrent:
        return f"layer groups {recurrent} hold recurrent state, which no name covers"
    # A pipeline stage without recurrent layers names nothing either, as every stage of the model.
    if _manager.model_has_recurrent_state(manager):
        return "the model holds recurrent state, which no name covers, on another pipeline stage"
    if not _manager.commits_blocks(manager):
        return (
            "the manager commits no blocks (block reuse off, or a draft manager without joint "
            "reuse)"
        )
    if _manager.has_lossy_cold_page_codec(manager):
        return (
            "the manager's cold-page codec is lossy (quantization_for_cold_page): a committed "
            "block brought back from a colder tier holds other bytes than its tokens computed"
        )
    # A read-ahead upstream has not established (one-model DraftTarget's) counts as none. A target
    # whose draft lives elsewhere names nothing: a named fetch could never fill that draft's pool.
    lookahead = _manager.prompt_lookahead(manager)
    if lookahead > 0:
        return (
            f"the manager is built for a one-model draft that reads {lookahead} prompt tokens past "
            "a block's end (Eagle or MTP), so requests sharing the block's tokens could hold "
            "different bytes under one name"
        )
    return None


def _attach_lender(
    manager: KVCacheManagerV2, *, scope: bytes, staging: StagingOptions, cls: type | None = None
) -> _ManagerLender:
    """``attach_lender`` with a lender class for tests to replace one rule at a time."""
    layout = _checked_layout(manager)
    if any(window is not None for window in layout.windows) and (
        _ranges._stale_ranges_follow_formula(manager, layout)
    ):
        layout = dataclasses.replace(layout, stale_ranges_by_formula=True)
    no_names = _no_names(manager, layout)
    if not isinstance(scope, bytes):
        raise TypeError(f"scope must be bytes, got {type(scope).__name__}")
    if not isinstance(staging, StagingOptions):
        raise TypeError(f"staging must be StagingOptions, got {type(staging).__name__}")
    identity = Identity(scope, layout_id(layout.page_layout), layout.layers, layout.shards)
    device = _device_regions(manager, layout, identity)
    # The staging path relays named blocks alone, so a lender lending nothing by name stages nothing.
    no_staging = no_names
    if no_staging is None and _manager.connector_loads(manager):
        no_staging = (
            "the manager's KV cache connector writes the pages it loads off the manager's stream, "
            "unordered with staging copies"
        )
    regions: list[MemoryRegion] = []
    slots: dict[int, int] = {}
    memory = None
    offset = 0
    if no_staging is None:
        counts = slot_counts(layout, staging)
        slots = {g: int(counts.get(g, 0)) for g in layout.pool_groups}
        sizes = {g: slots[g] * int(layout.page_bytes[g]) for g in layout.pool_groups}
        # TODO: an exception after the allocation keeps the staging memory, and the page-index
        # buffer once kept, until the process exits.
        memory = _allocate_retained(sum(sizes.values()))
        base = memory.address
        for g in layout.pool_groups:
            groups = tuple(lg for lg, pg in enumerate(layout.pool_group_of) if pg == g)
            layout_label = identity.region_label(groups)
            regions.append(
                MemoryRegion(
                    layout_label,
                    base + offset,
                    sizes[g],
                    int(layout.page_bytes[g]),
                    slots[g],
                    groups,
                    memory_kind=MemoryKind.HOST,
                )
            )
            offset += sizes[g]
    all_regions = tuple(regions + device)
    _check_disjoint(all_regions)
    lender = (cls or _ManagerLender)(
        weakref.ref(manager),
        layout,
        identity,
        all_regions,
        SlotQueue(slots),
        _no_memory if memory is None else weakref.ref(memory),
        no_names=no_names,
        no_staging=no_staging,
    )
    lender._retain_index_buffer(manager)
    logger.info(
        f"KV cache lender: namespace {identity.namespace.hex()}, staging {offset >> 20} MiB "
        f"in {len(regions)} regions, {len(device)} device regions"
        f"{'' if no_staging is None else '; nothing staged: ' + no_staging}"
    )
    # Installed last, so a failure above leaves nothing attached.
    _manager.install_lender(manager, lender)
    return lender


def _no_memory() -> None:
    """The staging memory of a lender that stages nothing: none."""
    return None


def _device_regions(
    manager: KVCacheManagerV2, layout: ManagerLayout, identity: Identity
) -> list[MemoryRegion]:
    """One region per device pool, in pool-group and then pool order, which leases in place lend
    from. Its pools must not be rebalanced while the lender lives."""
    # TODO: a slot may straddle two physical allocations of its pool.
    # TODO: no pool exposes its handle type, so a region cannot say if other processes can import it.
    regions = []
    for g in layout.pool_groups:
        groups = tuple(lg for lg, pg in enumerate(layout.pool_group_of) if pg == g)
        for pool in layout.device_pools[g]:
            regions.append(
                MemoryRegion(
                    identity.device_region_label(groups, pool.index),
                    int(pool.base),
                    int(pool.num_slots) * int(pool.slot_bytes),
                    int(pool.slot_bytes),
                    int(pool.num_slots),
                    groups,
                    memory_kind=MemoryKind.DEVICE,
                )
            )
    return regions


class _QueuedCopy:
    """Copies queued together, complete once their event says so and never without one; ``done``
    asks the event at most once per progress round, and not after it reported completion."""

    def __init__(self, event: torch.cuda.Event | None) -> None:
        self._event = event
        self._done = False
        self._asked_in_round: int | None = None

    def done(self, round_: int | None = None) -> bool:
        """Whether the copies have completed, without waiting; ``round_=None`` always asks the
        event, if there is one."""
        if self._event is None:
            return False
        if not self._done and (round_ is None or round_ != self._asked_in_round):
            self._asked_in_round = round_
            self._done = bool(self._event.query())
        return self._done

    def wait(self) -> None:
        """Block the host until the copies have completed; without an event, return at once."""
        if not self._done and self._event is not None:
            self._event.synchronize()
            self._done = True


# Copies that may be queued with no record of their completion: a read's from the grant of its
# slots until its copy's record, and a quarantined lease's.
_COPY_WITHOUT_RECORD = _QueuedCopy(None)


def _no_cache_reason(request_id: int) -> str:
    return f"request {request_id} has no KV cache"


_CUDA_EVENT = torch.cuda.Event


def _check_ready_event(ready_event: object, staged: bool = False) -> None:
    """``TypeError`` unless ``ready_event`` is ``None`` or an event its holder asks with
    ``query()``; for a staging lease, whose copies a stream waits for, one with ``wait()`` too."""
    if ready_event is None:
        return
    if not callable(getattr(ready_event, "query", None)) or (
        staged and not callable(getattr(ready_event, "wait", None))
    ):
        needs = "query() and wait()" if staged else "query()"
        raise TypeError(
            f"ready_event must be an event with {needs} or None, got {type(ready_event).__name__}"
        )


class _BothDone:
    """Two readiness conditions, done once both are."""

    __slots__ = ("_ready_event", "_fence")

    def __init__(self, ready_event: torch.cuda.Event | _BothDone, fence: torch.cuda.Event) -> None:
        self._ready_event = ready_event
        self._fence = fence

    def query(self) -> bool:
        return self._fence.query() and self._ready_event.query()


def _fenced(
    ready_event: torch.cuda.Event | _BothDone | None, fence: torch.cuda.Event
) -> torch.cuda.Event | _BothDone:
    """Combine a stream fence with a lease's earlier readiness condition, where there is one."""
    return fence if ready_event is None else _BothDone(ready_event, fence)


def _frozen_array(values: np.ndarray) -> np.ndarray:
    """``values`` as contiguous ``int64`` on immutable bytes of their own."""
    return np.frombuffer(np.ascontiguousarray(values, dtype=np.int64).tobytes(), dtype=np.int64)


def _unshared(values: np.ndarray) -> np.ndarray:
    """``values`` on immutable bytes as an array object of its own, for one lease's view: no other
    view, record or lock holds it."""
    return _immutable(values).view()


# No rows: the ordinals or pages of a layer group that lends none; a view holds a view of it.
_NO_ROWS = _frozen_int64(())
# The ordinal of the one row of a layer group of recurrent state; a view holds a view of it.
_STATE_ROW = _frozen_int64((0,))

# Up to this many blocks, a cache's pages and a fast write's range of its row are read as lists,
# cheaper than NumPy's casts and reductions, whose first call after other work takes microseconds.
_LISTED_BLOCKS = 128
# Fast lends of blocks below this take their ordinals as views of these bytes, the entries and the
# locks the same view; a lease's view holds a view of its own.
_ORDINAL_BLOCKS = 1 << 14
_ORDINALS = _frozen_int64(range(_ORDINAL_BLOCKS))
# Views of ``_ORDINALS`` by range, each made once; emptied when it holds ``_ORDINAL_VIEWS_KEPT``.
_ordinal_views: dict[tuple[int, int], np.ndarray] = {}
_ORDINAL_VIEWS_KEPT = 4096
# What ``_ManagerLender._return_whole`` gives where it returns nothing, for the rules to return.
_NOT_WHOLE = object()
# The unchecked constructor of views, bound once for fast lends.
_new_view = LeaseView._unchecked
_new_object = object.__new__
_set_attribute = object.__setattr__


def _lent_rows(
    lg: int, ordinals: np.ndarray, slots: np.ndarray, regions: tuple[int, ...]
) -> LentLayerGroupRows:
    """``LentLayerGroupRows._unchecked(lg, ordinals, slots=slots, region_indices=regions)`` of
    positional arguments, as a fast lend builds each layer group's rows."""
    rows = _new_object(LentLayerGroupRows)
    _set_attribute(
        rows,
        "__dict__",
        {
            "layer_group_index": lg,
            "page_ordinals": ordinals,
            "object_keys": None,
            "slots": slots,
            "region_indices": regions,
        },
    )
    return rows


# Packers of 0 to ``_LISTED_BLOCKS`` unsigned 64-bit integers, by count.
_PAGE_PACKERS = tuple(struct.Struct(f"={count}Q") for count in range(_LISTED_BLOCKS + 1))
# The dtypes of fast lends' pages as dtype objects, which NumPy takes without converting them.
_from_buffer = np.frombuffer
_INT64 = np.dtype(np.int64)
_INT32 = np.dtype(np.int32)
_UINT64 = np.dtype(np.uint64)
_UINT32 = np.dtype(np.uint32)


def _frozen_pages(pages: Sequence[int]) -> np.ndarray:
    """At most ``_LISTED_BLOCKS`` non-negative ``pages`` as ``int64`` on immutable bytes of their
    own; ``struct.error`` for a negative one."""
    return np.frombuffer(_PAGE_PACKERS[len(pages)].pack(*pages), _INT64)


class _ParkedRow:
    """Lender-owned ``int32`` page-index rows, one per layer group, that a cache whose index slot
    was released early writes its beam-0 page indices into until it closes."""

    __slots__ = ("pages", "bytes", "buffers", "source", "kv")

    def __init__(self, groups: int, blocks: int) -> None:
        self.pages = np.full((groups, blocks), -1, dtype=np.int32)
        self.bytes = memoryview(self.pages).cast("B")
        # The buffer the manager binds each layer group's page indices to.
        self.buffers = tuple(memoryview(self.pages[lg]) for lg in range(groups))
        # As a lend reads a row: its words, where it starts, the words of a group, its blocks.
        self.source = (memoryview(self.pages).cast("B").cast("i"), 0, blocks, blocks)
        # The cache bound to the row, which writes into it until it closes.
        self.kv: _KVCache | None = None


@dataclass(eq=False, slots=True)
class _LentBlockEntries:
    """Per-group aligned block ordinals, device pages and staging slots (none until a grant); a fast
    lend of every row adds ``spans``, each group's run ``(first, last)`` (empty if
    ``last <= first``, ``None`` for state), and ``block_end``, where its block runs end."""

    layer_group_indices: list[int]
    block_ordinals: list[np.ndarray]
    device_pages: list[np.ndarray]
    staging_slots: list[np.ndarray] | tuple[()] = ()
    spans: list[tuple[int, int] | None] | None = None
    block_end: int = 0

    @property
    def num_rows(self) -> int:
        return sum(len(o) for o in self.block_ordinals)


class _ManagerLender:
    """Lender over one weakly held manager: refusals return failed leases, permanent stops raise
    ``RuntimeError``; progress recycles and grants staging slots, which DEVICE leases bypass."""

    # The lender's own state is in slots; the instance dict allows instrumented lenders to
    # override methods without changing its storage layout.
    __slots__ = (
        "_manager_ref",
        "_layout",
        "_tokens_per_block",
        "_identity",
        "_regions",
        "_slot_queue",
        "_memory_ref",
        "_no_names",
        "_no_staging",
        "_recurrent_groups",
        "_region_index_of_pool_group",
        "_device_regions_of_pool_group",
        "_staging_region_indices",
        "_device_slot_limit",
        "_any_window",
        "_ledger",
        "_waiting_for_slots",
        "_leases_holding_slots",
        "_open_leases",
        "_open_holds",
        "_quarantined",
        "_shut_down",
        "_reuse_was_reset",
        "_stop_reason",
        "_index_buffer",
        "_progress_round",
        "_locks",
        "_named_groups",
        "_page_layout",
        "_ordinal_ranges",
        "_group_regions",
        "_fast_groups",
        "_fast_layout",
        "_state_rows",
        "_shrink_epochs",
        "_state_group",
        "_state_record",
        "_state_unverified",
        "_index_rows",
        "_index_words",
        "_index_strides",
        "_parked",
        "_parked_free",
        "_parked_closing",
        "_parked_made",
        "_bad_row",
        "_writes_at_end",
        "_margin_groups",
        "__dict__",
        "__weakref__",
    )

    def __init__(
        self,
        manager: weakref.ref,
        layout: ManagerLayout,
        identity: Identity,
        regions: tuple[MemoryRegion, ...],
        slots: SlotQueue,
        memory: weakref.ref,
        *,
        no_names: str | None = None,
        no_staging: str | None = None,
    ) -> None:
        self._manager_ref = manager
        self._layout = layout
        self._tokens_per_block = int(layout.tokens_per_block)
        self._identity = identity
        self._regions = tuple(regions)
        self._slot_queue = slots
        # Only the keep list holds the staging memory strongly; this finds it there at shutdown,
        # or ``None`` for a lender that stages nothing.
        self._memory_ref = memory
        self._no_names = no_names
        # Why the lender stages nothing, or ``None`` where it stages.
        self._no_staging = no_names if no_staging is None else no_staging
        recurrent = tuple(getattr(layout, "recurrent", ()))
        # The layer groups of recurrent state: each lends one row, its state page, as block 0.
        self._recurrent_groups = frozenset(lg for lg, flag in enumerate(recurrent) if flag)
        self._region_index_of_pool_group = {g: i for i, g in enumerate(layout.pool_groups)}
        # Per pool group, the positions of its device regions in ``regions``, in pool order.
        self._device_regions_of_pool_group: dict[int, tuple[int, ...]] = {}
        for index, region in enumerate(self._regions):
            if region.memory_kind is MemoryKind.DEVICE:
                g = int(layout.pool_group_of[region.layer_group_indices[0]])
                held = self._device_regions_of_pool_group.get(g, ())
                self._device_regions_of_pool_group[g] = (*held, index)
        # Per pool group, the one-region tuple its staged rows name, shared by every lend.
        self._staging_region_indices = {
            g: (index,)
            for g, index in self._region_index_of_pool_group.items()
            if index < len(self._regions) and self._regions[index].memory_kind is MemoryKind.HOST
        }
        # Per layer group, the least ``slot_count`` of the device regions its rows in place span,
        # which no slot such a row holds may reach.
        self._device_slot_limit: dict[int, int] = {}
        for region in self._regions:
            if region.memory_kind is MemoryKind.DEVICE:
                for lg in region.layer_group_indices:
                    held = self._device_slot_limit.get(lg, region.slot_count)
                    self._device_slot_limit[lg] = min(held, region.slot_count)
        self._any_window = any(w is not None for w in layout.windows)
        # Fetch records and the bounds they give; the ledger asks this lender about copies through a
        # weak reference, so no cycle keeps the lender or a cache its records hold.
        lender = weakref.ref(self)
        self._ledger = _FetchLedger(layout, lambda copy: lender()._copy_done(copy))
        self._waiting_for_slots: Deque[_StagingLease] = collections.deque()
        self._leases_holding_slots: list[_StagingLease] = []  # granted, slots not yet returned
        # Open leases, held strongly: an open lease keeps the staging memory at shutdown even
        # once its holder has dropped it.
        self._open_leases: set[_StagingLease] = set()
        # Open region holds, held strongly too: a dropped hold still keeps the memory.
        self._open_holds: set[_RegionHold] = set()
        self._quarantined: list[SlotRuns] = []  # slots a failed copy may still touch; never reused
        self._shut_down = False
        self._reuse_was_reset = False
        self._stop_reason: str | None = None  # why lending stopped for good; the first reason stays
        self._index_buffer: object | None = None  # kept until the runtime's shutdown has returned
        # The kept page-index buffer fast lends read pages from: as an array, as one flat run of
        # words, and the words a layer group, a row and a half row span (a row's blocks).
        self._index_rows: np.ndarray | None = None
        self._index_words: memoryview | None = None
        self._index_strides = (0, 0, 0)
        # Parked rows by request; rows of freed requests whose cache may be open; rows ready to
        # bind; every row made, kept with the page-index buffer since closing caches write them.
        self._parked: dict[int, _ParkedRow] = {}
        self._parked_closing: list[_ParkedRow] = []
        self._parked_free: list[_ParkedRow] = []
        self._parked_made: list[_ParkedRow] = []
        # A row's bytes with the bad index in every word, to clear a row handed out again.
        self._bad_row = b""
        self._progress_round = 0  # rounds of progress, so each asks a copy's event at most once
        # The locks on blocks of the manager's caches, and the locked caches it closes later.
        self._locks = PageLocks(manager, recurrent, getattr(layout, "pool_group_of", None))
        self._locks.caps_history = self._any_window
        # Per layer group, whether it lends by name: one holding a layer the manager keeps out of
        # prefix reuse does not.
        self._named_groups = tuple(layout.named_groups)
        self._page_layout: PageLayout | None = None  # built at its first read
        self._ordinal_ranges: dict[int, dict[tuple[int, int], tuple[int, np.ndarray]]] = {}
        # Per layer group, the positions of the device regions its rows are placed in.
        self._group_regions = tuple(
            self._device_regions_of_pool_group.get(int(g), ())
            for g in getattr(layout, "pool_group_of", ())
        )
        # Per layer group: its index, the positions of its device regions, its device slot limit,
        # whether it holds recurrent state, and its window, as fast lends read them.
        windows = tuple(getattr(layout, "windows", ()))
        self._fast_groups = tuple(
            (
                lg,
                regions,
                self._device_slot_limit.get(lg, 0),
                lg in self._recurrent_groups,
                windows[lg] if lg < len(windows) else None,
            )
            for lg, regions in enumerate(self._group_regions)
        )
        # Whether a lend of every row in place may take a fast lend: no Helix, a device region for
        # every layer group, and windows without sinks whose stale ranges follow the window formula.
        # TODO: under Helix every in-place lend, mark and resume-bound query takes the rules.
        by_formula = getattr(layout, "stale_ranges_by_formula", False)
        sinks = tuple(getattr(layout, "sink_blocks", ()))
        self._fast_layout = (
            getattr(layout, "cp", (0, 1))[1] == 1
            and bool(self._group_regions)
            and all(self._group_regions)
            and all(
                window is None or (by_formula and lg < len(sinks) and not sinks[lg])
                for lg, window in enumerate(windows)
            )
        )
        # One frozen row per state slot a fast lend gave, by slot, for the entries of fast lends; a
        # lease's view holds a view of its own.
        self._state_rows: dict[int, np.ndarray] = {}
        # Per request, in-place shrinks the manager reported since the last free: a fast write's
        # marks trust the pages it lent only while this count stays as at the lend.
        self._shrink_epochs: dict[int, int] = {}
        # The manager's record of each request's recurrent-state slot and its group, read instead of
        # the cache; requests resumed since it last matched, which the cache answers for meanwhile.
        prepared = _manager.prepared_state_slots(manager()) if manager() is not None else None
        self._state_group, self._state_record = prepared if prepared is not None else (None, None)
        self._state_unverified: set[int] = set()
        # Open fast write leases whose return would resume at their end, by ``id`` of their cache,
        # at most one per cache; the hooks that can move a cache's history take them out.
        self._writes_at_end: dict[int, _InPlaceLease] = {}
        # Layer groups that keep a window wider than the one they read, whose margin rows a fetch
        # need not deliver.
        self._margin_groups = frozenset(
            lg for lg in range(len(windows)) if _ranges._read_slack(layout, lg) > 0
        )

    @property
    def regions(self) -> tuple[MemoryRegion, ...]:
        """See ``Lender.regions``."""
        return self._regions

    @property
    def page_layout(self) -> PageLayout:
        """See ``Lender.page_layout``."""
        if self._page_layout is None:
            # Device regions follow pool-group, then pool order; a staging region serves a group.
            device = {}
            for g, positions in self._device_regions_of_pool_group.items():
                for pool, position in zip(self._layout.device_pools[g], positions):
                    device[(g, int(pool.index))] = position
            staging = {
                int(self._layout.pool_group_of[region.layer_group_indices[0]]): position
                for position, region in enumerate(self._regions)
                if region.memory_kind is MemoryKind.HOST
            }
            self._page_layout = dataclasses.replace(
                self._layout.page_layout, device_region_index=device, staging_region_index=staging
            )
        return self._page_layout

    def held_page_counts(self) -> HeldPages:
        """See ``Lender.held_page_counts``."""
        return self._locks.held_page_counts()

    def hold_regions(self) -> _RegionHold:
        """See ``Lender.hold_regions``."""
        if self._serving_manager() is None:
            # The memory was freed or kept at the shutdown; a later hold changes neither.
            return _RegionHold(None)
        hold = _RegionHold(self)
        self._open_holds.add(hold)
        return hold

    def lend_read(
        self,
        request: LlmRequest,
        start: int,
        end: int,
        *,
        memory_kind: MemoryKind,
        rows: RowSelection,
        ready_event: torch.cuda.Event | None,
    ) -> _StagingLease | _InPlaceLease:
        """See ``Lender.lend_read``."""
        # A DEVICE lend of every row passes every check of the kind and rows.
        fast = memory_kind is MemoryKind.DEVICE and rows is RowSelection.ALL and self._lends_fast()
        if (
            fast
            and (ready_event is None or type(ready_event) is _CUDA_EVENT)
            and type(start) is int
            and type(end) is int
            and 0 <= start <= end
            and not start % self._tokens_per_block
            and not self._shut_down
            and self._stop_reason is None
        ):
            # Every check below passes: the fast lend, or the rules alone where it lends nothing.
            manager = self._manager_ref()
            if manager is not None:
                lease = self._read_every_row_fast(
                    manager, request, int(request.py_request_id), start, end, ready_event
                )
                if lease is not None:
                    return lease
                fast = False
        if not fast:
            self._check_memory_and_rows(memory_kind, rows)
        if ready_event is not None and type(ready_event) is not _CUDA_EVENT:
            _check_ready_event(ready_event, memory_kind is MemoryKind.HOST)
        named = rows is RowSelection.NAMED
        start, end = self._whole_blocks(start, end, named)
        request_id = int(request.py_request_id)
        manager = self._lending_manager()
        self._check_not_stopped()
        if fast:
            lease = self._read_every_row_fast(manager, request, request_id, start, end, ready_event)
            if lease is not None:
                return lease
        if named and self._reuse_was_reset:
            return self._failed_at_call(memory_kind, "read", request_id, _REASON_REUSE_RESET)
        kv = _manager.kv_of(manager, request_id)
        if kv is None:
            return self._failed_at_call(
                memory_kind, "read", request_id, _no_cache_reason(request_id)
            )
        unnamed = self._unnamed_input_refusal(request) if named else None
        if unnamed is not None:
            return self._failed_at_call(memory_kind, "read", request_id, unnamed)
        state = _manager.cache_state(kv)
        refusal = self._read_end_refusal(manager, request, state, end, named)
        if refusal is not None:
            return self._failed_at_call(memory_kind, "read", request_id, refusal)
        if not state.active:
            return self._failed_at_call(memory_kind, "read", request_id, _REASON_SUSPENDED)
        # TODO: a publish leaves out blocks the request's window passed though the prefix tree may
        # still hold their pages, so a fetch whose window keeps such a block finds it missing.
        entries = self._lendable_block_entries(
            manager,
            state.history,
            self._block_entries_of_range(manager, kv, start, end, request_id=request_id),
        )
        if named:
            entries = self._committed_entries(kv, entries)
        if memory_kind is MemoryKind.DEVICE:
            entries = self._without_short_windows_mid_context(manager, request, state, entries)
        else:
            counts = self._required_slots_per_pool_group(
                [len(block_ordinals) for block_ordinals in entries.block_ordinals]
            )
            refusal = self._fits_refusal(counts)
            if refusal is not None:
                return self._failed_at_call(memory_kind, "read", request_id, refusal)
        if _manager.connector_reserved(manager, request):
            return self._failed_at_call(memory_kind, "read", request_id, _REASON_RESERVED)
        reuse_keys = (
            self._reuse_keys_for(manager, request, kv, entries.block_ordinals) if named else None
        )
        if memory_kind is MemoryKind.DEVICE:
            refusal = self._ledger._unsettled_fetch_refusal(request_id, kv, entries)
            if refusal is None:
                refusal = self._unfetched_rows_refusal(request_id, kv, state, entries, end)
            if refusal is not None:
                return self._failed_at_call(memory_kind, "read", request_id, refusal)
            return self._lend_in_place(
                manager,
                request,
                kv,
                "read",
                entries,
                reuse_keys,
                start=start,
                end=end,
                ready_event=ready_event,
            )
        lease = _StagingLease(self, "read", request_id, kv, entries, reuse_keys)
        lease._ready_event = ready_event
        lease._range = (start, end)
        self._admit(lease, counts)
        return lease

    def lend_write(
        self,
        request: LlmRequest,
        start: int,
        end: int,
        *,
        memory_kind: MemoryKind,
        rows: RowSelection,
        ready_event: torch.cuda.Event | None,
    ) -> _StagingLease | _InPlaceLease:
        """See ``Lender.lend_write``."""
        fast = memory_kind is MemoryKind.DEVICE and rows is RowSelection.ALL and self._lends_fast()
        if (
            fast
            and (ready_event is None or type(ready_event) is _CUDA_EVENT)
            and type(start) is int
            and type(end) is int
            and 0 <= start <= end
            and not start % self._tokens_per_block
            and not self._shut_down
            and self._stop_reason is None
        ):
            # No Helix on a layout fast lends cover: the prompt is the request's own.
            prompt = _manager.prompt_length(request)
            manager = self._manager_ref() if end <= prompt else None
            if manager is not None:
                lease = self._write_every_row_fast(
                    manager, request, int(request.py_request_id), start, end, prompt, ready_event
                )
                if lease is not None:
                    return lease
                fast = False
        if not fast:
            self._check_memory_and_rows(memory_kind, rows)
        if ready_event is not None and type(ready_event) is not _CUDA_EVENT:
            _check_ready_event(ready_event, memory_kind is MemoryKind.HOST)
        named = rows is RowSelection.NAMED
        start, end = self._whole_blocks(start, end, named)
        request_id = int(request.py_request_id)
        layout = self._layout
        tpb = int(layout.tokens_per_block)
        prompt = _manager.prompt_length(request, cp_size=getattr(layout, "cp", (0, 1))[1])
        # The request computes its last prompt token itself, for its logits, so a named fetch ends
        # at the whole blocks before it; a fetch of every row may reach the prompt's end.
        if named and end > (prompt - 1) // tpb * tpb:
            raise ValueError(
                f"the range ends at {end}, past the whole blocks before the request's last prompt "
                f"token ({prompt} prompt tokens)"
            )
        if not named and end > prompt:
            raise ValueError(f"the range ends at {end}, past the request's {prompt} prompt tokens")
        manager = self._lending_manager()
        self._check_not_stopped()
        if fast:
            lease = self._write_every_row_fast(
                manager, request, request_id, start, end, prompt, ready_event
            )
            if lease is not None:
                return lease
        if named and self._reuse_was_reset:
            return self._failed_at_call(memory_kind, "write", request_id, _REASON_REUSE_RESET)
        kv = _manager.kv_of(manager, request_id)
        if kv is None:
            return self._failed_at_call(
                memory_kind, "write", request_id, _no_cache_reason(request_id)
            )
        unnamed = self._unnamed_input_refusal(request) if named else None
        if unnamed is not None:
            return self._failed_at_call(memory_kind, "write", request_id, unnamed)
        state = _manager.cache_state(kv)
        # Count future rows before allocating ordinals or growing the cache.
        committed = _manager.committed_by_group(kv, layout.num_layer_groups)
        starts = [
            -(-end // tpb) * tpb
            if lg not in self._recurrent_groups and tokens > end
            else max(start, tokens // tpb * tpb)
            for lg, tokens in enumerate(committed)
        ]
        runs = [
            _ranges._needed_ranges(manager, layout, lg, starts[lg] // tpb, -(-end // tpb), end)
            if not named or self._named_groups[lg]
            else []
            for lg in range(layout.num_layer_groups)
        ]
        refusal = None
        if memory_kind is MemoryKind.HOST:
            counts = self._required_slots_per_pool_group(
                [sum(e - b for b, e in lg_runs) for lg_runs in runs]
            )
            refusal = self._fits_refusal(counts)
        if refusal is None:
            refusal = self._write_range_refusal(manager, request, state, start, end, prompt)
        if refusal is not None:
            return self._failed_at_call(memory_kind, "write", request_id, refusal)
        block_ordinals = [
            self._range_ordinals(request_id, lg, lg_runs) for lg, lg_runs in enumerate(runs)
        ]
        reuse_keys = self._reuse_keys_for(manager, request, kv, block_ordinals) if named else None
        split = self._bidirectional_split_refusal(request, end)
        if split is not None:
            return self._failed_at_call(memory_kind, "write", request_id, split)
        if not state.active:
            return self._failed_at_call(memory_kind, "write", request_id, _REASON_SUSPENDED)
        if self._scratch_reuse_on(kv):
            return self._failed_at_call(memory_kind, "write", request_id, _REASON_SCRATCH)
        if named and self._returns_context_outputs(request):
            return self._failed_at_call(memory_kind, "write", request_id, _REASON_CONTEXT_OUTPUTS)
        # After every ValueError: whether an earlier fetch settled is this rank's own timing.
        unsettled = self._ledger._unsettled_fetch_refusal(request_id, kv)
        if unsettled is not None:
            return self._failed_at_call(memory_kind, "write", request_id, unsettled)
        # A named fetch's grow moves a windowed history to ``end``, so windows need pages only for
        # what it reads; a fetch of every row writes pages held since admission, history kept.
        position = end if named and self._any_window else state.history
        if position > state.history and _manager.connector_load_pending(manager, request):
            return self._failed_at_call(
                memory_kind,
                "write",
                request_id,
                "a window cannot advance history while a connector load is pending",
            )
        refusal = self._history_refusal(manager, state.history, position, end)
        if refusal is None:
            refusal = self._unwritten_pages_refusal(manager, request_id, kv, position)
        cap = self._history_cap(kv)
        if refusal is None and cap is not None and position > cap:
            refusal = f"a window would leave a locked block behind at a history of {position}"
        if refusal is not None:
            return self._failed_at_call(memory_kind, "write", request_id, refusal)
        # HOST copies and DEVICE writers share the same exclusion domain before any grow.
        planned = (
            None
            if named
            else self._block_entries_of_range(
                manager, kv, starts, end, request_id=request_id, ordinals=block_ordinals
            )
        )
        refusal = None if named else self._pageless(planned)
        ordinals = block_ordinals if named else planned.block_ordinals
        checked = self._locks.check(
            kv, dict(enumerate(ordinals)), LockMode.WRITE, ordered=True, committed=committed
        )
        if refusal is None and isinstance(checked, str):
            refusal = f"the blocks cannot be locked: {checked}"
        if refusal is not None:
            return self._failed_at_call(memory_kind, "write", request_id, refusal)
        # Read before the grow, which moves a windowed cache's history to ``end``.
        kept = self._ledger._prefix_kept_by_fetch(request_id, kv, start)
        # TODO: the V2 scheduler cannot reclaim the pages a parked fetch grew.
        # TODO: route capture stops reading a request's prepopulated length once it holds the routes
        # below it, until the request finishes, so a later resume leaves positions without routes.
        # TODO: with extra KV tokens the grow locks a page past the history it leaves, which nothing
        # writes, so a later all-reusable windowed lease that leaves that block behind fails.
        if named:
            stream = _manager.cache_stream(manager, request_id)
            if ready_event is not None and not ready_event.query():
                wait = getattr(ready_event, "wait", None)
                if not callable(wait) or stream is None:
                    return self._failed_at_call(
                        memory_kind,
                        "write",
                        request_id,
                        "a named grow needs a completed or stream-waitable ready_event",
                    )
                wait(stream)
            fence = (
                torch.cuda.Event()
                if memory_kind is MemoryKind.DEVICE and stream is not None
                else None
            )
            try:
                grown = _manager.grow(manager, request, kv, position, end)
            except RuntimeError:
                # A diagnostic page fill can fail after resize moved a window's history. Keep
                # that unfetched interval in the ledger, so it cannot count as computed tokens.
                if _manager.cache_state(kv).history > state.history:
                    self._ledger._record_fetch(
                        request_id,
                        kv,
                        start,
                        kept,
                        position,
                        committed,
                        state_end=end if self._recurrent_groups else None,
                        end=end,
                        prompt=prompt,
                        runs=_manager.bidirectional_runs(request),
                    )
                    _manager.drop_held_commit(manager, kv, start)
                raise
            if not grown:
                return self._failed_at_call(
                    memory_kind,
                    "write",
                    request_id,
                    f"no free pages to grow the cache to {end} tokens",
                )
        # From here poll_resume_bounds accounts for the fetch, whatever this lease's outcome.
        fetch = self._ledger._record_fetch(
            request_id,
            kv,
            start,
            kept,
            position,
            committed,
            state_end=end if self._recurrent_groups else None,
            end=end,
            prompt=prompt,
            runs=_manager.bidirectional_runs(request),
        )
        # A commit the cache's locks hold back would commit computed rows this fetch may overwrite.
        _manager.drop_held_commit(manager, kv, start)
        if named and fence is not None:
            # A grow can queue a wait for pages evicted to the host tier. Keep the view behind
            # that work, and retain the fetch's prefix accounting if recording its fence fails.
            try:
                fence.record(stream)
            except RuntimeError:
                stream.synchronize()
                if self._ledger._latest_fetches.get(request_id) is fetch:
                    del self._ledger._latest_fetches[request_id]
                raise
            ready_event = _fenced(ready_event, fence)
        entries = (
            self._block_entries_of_range(
                manager,
                kv,
                starts,
                end,
                named=named,
                request_id=request_id,
                ordinals=block_ordinals,
            )
            if planned is None
            else planned
        )
        fetch.entries = entries
        doomed = self._pageless(entries) if planned is None else None
        if memory_kind is MemoryKind.DEVICE:
            # An ALL write locks the blocks it checked; a NAMED write checks its blocks again after
            # its grow.
            return self._lend_in_place(
                manager,
                request,
                kv,
                "write",
                entries,
                reuse_keys,
                fetch,
                doomed,
                start=start,
                end=end,
                checked=None if named else checked,
                ready_event=ready_event,
            )
        lease = _StagingLease(self, "write", request_id, kv, entries, reuse_keys, fetch)
        lease._ready_event = ready_event
        lease._range = (start, end)
        if doomed is not None:
            # It fails at its first poll, which abandons the fetch.
            lease._defer_failure(doomed)
            self._open_leases.add(lease)
            return lease
        self._admit(lease, counts)
        return lease

    def poll_resume_bounds(self, request: LlmRequest) -> ResumeBounds | None:
        """See ``Lender.poll_resume_bounds``."""
        request_id = int(request.py_request_id)
        manager = self._serving_manager()
        if manager is None:
            raise ValueError(f"{_no_cache_reason(request_id)}: {_REASON_SHUT_DOWN}")
        kv = _manager.kv_of(manager, request_id)
        if kv is None:
            raise ValueError(_no_cache_reason(request_id))
        state = _manager.cache_state(kv)
        if self._stop_reason is not None:
            # Empty: no fetch through a stopped lender counts.
            return ResumeBounds(max_position=0, min_position=max(state.history, 1))
        return self._ledger._resume_bounds(
            _manager.prompt_length(request, manager),
            _manager.bidirectional_runs(request),
            request_id,
            manager,
            kv,
            state,
        )

    def return_writes(
        self,
        entries: Sequence[tuple[_StagingLease | _InPlaceLease, object, bool]],
    ) -> list[ResumeBounds | None]:
        """See ``Lender.return_writes``."""
        if type(entries) is list and len(entries) == 1:
            # One entry: checked, then returned, raising what its return raises; an in-place write
            # returned with every row passes ``_accepted_return``'s checks here.
            entry = entries[0]
            if type(entry) is tuple and len(entry) == 3:
                lease, arrived, reusable = entry
                if (
                    arrived is ALL_ARRIVED
                    and type(reusable) is bool
                    and type(lease) is _InPlaceLease
                    and lease._view_returned
                    and not lease._marked
                    and not lease._released
                    and lease._kind == "write"
                    and lease._lender() is self
                ):
                    return [self._return_write(lease, None, reusable)]
            lease, masks, reusable = self._accepted_return(entry, set())
            return [self._return_write(lease, masks, reusable)]
        seen: set[int] = set()
        accepted = [self._accepted_return(entry, seen) for entry in entries]
        results: list[ResumeBounds | None] = []
        error = None
        for lease, masks, reusable in accepted:
            try:
                results.append(self._return_write(lease, masks, reusable))
            except BaseException as raised:  # noqa: BLE001 - raised once every lease is returned
                error = raised if error is None else error
                results.append(None)
        if error is not None:
            raise error
        return results

    def progress(self) -> None:
        """See ``Lender.progress``."""
        if self._leases_holding_slots or self._waiting_for_slots:
            self._progress()
        else:
            # Nothing holds or waits for staging slots: the round only counts.
            self._progress_round += 1

    def _accepted_return(
        self, entry: object, seen: set[int]
    ) -> tuple[_StagingLease | _InPlaceLease, list[np.ndarray] | None, bool]:
        """``entry`` checked, with its masks as kept (``None``: every row) and its lease added to
        ``seen``; raises what ``return_writes`` raises for it, changing nothing else."""
        try:
            lease, arrived, reusable = entry
        except (TypeError, ValueError):
            raise TypeError(
                f"an entry is (lease, arrived, reusable), got {type(entry).__name__}"
            ) from None
        if type(reusable) is not bool:
            raise TypeError(f"reusable must be a bool, got {type(reusable).__name__}")
        if type(lease) not in (_InPlaceLease, _StagingLease) or lease._lender() is not self:
            raise ValueError(f"the lease was not lent by this lender: {lease!r}")
        if lease._kind != "write":
            raise RuntimeError("return_writes is for write leases")
        if lease._marked or lease._released or id(lease) in seen:
            raise RuntimeError(f"the lease was returned already: {lease!r}")
        if not lease._view_returned:
            raise RuntimeError("return_writes before poll() returned the lease's view")
        masks = None if arrived is ALL_ARRIVED else lease._accepted_masks(arrived)
        seen.add(id(lease))
        return lease, masks, reusable

    def _return_write(
        self,
        lease: _StagingLease | _InPlaceLease,
        masks: list[np.ndarray] | None,
        reusable: bool,
    ) -> ResumeBounds | None:
        """Count a checked write's arrived rows (``None``: all) into the cache it was lent from,
        keeping an unreusable cache out of reuse before the unlock's retried commit; return the
        lease even if counting raised; give the bounds, ``None`` while unsettled or where the cache
        is no longer the request's."""
        if not reusable or (masks is not None and self._margin_missing(lease, masks)):
            self._keep_out_of_reuse(lease)
        manager = self._serving_manager()
        if manager is not None and lease._fetch is not None:
            _manager.drop_held_commit(manager, lease._kv, lease._fetch.start)
        if masks is None and type(lease) is _InPlaceLease:
            bounds = self._return_whole(lease)
            if bounds is not _NOT_WHOLE:
                return bounds
        lease._marked = True
        request_id = lease._request_id
        kv = lease._kv
        try:
            whole = self._count_arrivals(lease, masks)
        finally:
            if type(lease) is _InPlaceLease and not lease._released:
                # ``release`` of a write whose view was returned and that is returned with its
                # arrivals: the lease's lock ends.
                lease._finish_returned_write()
                self._writes_at_end.pop(id(kv), None)
                if lease._lock is not None:
                    self._locks.unlock(lease._lock)
            else:
                lease.release()
        if whole and self._stop_reason is None:
            manager = self._serving_manager()
            if manager is None or _manager.kv_of(manager, request_id) is not kv:
                return None
            state, blocks = _manager.cache_extent(kv)
            ledger = self._ledger
            fetch = lease._fetch
            spans = lease._block_entries.spans
            bounds = ledger._whole_delivery_bounds(
                fetch.prompt, fetch.runs, request_id, manager, kv, state, spans, blocks
            )
            if bounds is None:
                bounds = ledger._resume_bounds(
                    fetch.prompt, fetch.runs, request_id, manager, kv, state
                )
            return bounds
        return self._bounds_after_return(lease)

    def _keep_out_of_reuse(self, lease: _StagingLease | _InPlaceLease) -> None:
        """Keep the lease's cache from committing anything more for reuse until it closes, if it is
        still open: the request's cache, or one a lock keeps open."""
        kv = lease._kv
        manager = self._serving_manager()
        if manager is None or kv is None:
            return
        if kv is _manager.kv_of(manager, lease._request_id) or id(kv) in self._locks.locked_caches:
            self._locks.keep_out_of_reuse(kv)

    def _margin_missing(
        self, lease: _StagingLease | _InPlaceLease, masks: list[np.ndarray]
    ) -> bool:
        """Whether a lent window-margin row did not arrive: a row of a group whose kept window is
        wider than its read window, which the read window passed at the fetch's end."""
        if not self._margin_groups:
            return False
        fetch = lease._fetch
        manager = self._manager_ref()
        if fetch is None or fetch.end is None or manager is None:
            return False
        entries = lease._block_entries
        for lg, ordinals, mask in zip(entries.layer_group_indices, entries.block_ordinals, masks):
            if lg not in self._margin_groups or not len(ordinals):
                continue
            beg, end = _ranges._read_stale_range(manager, self._layout, lg, fetch.end)
            if np.any((ordinals >= beg) & (ordinals < end) & ~mask):
                return True
        return False

    def _return_whole(self, lease: _InPlaceLease) -> ResumeBounds | None | object:
        """``_return_write`` of a checked in-place write returned whole as its cache's first
        delivery, equal to the rules while the fast lend holds every page it lent
        (``_lent_pages_held``), its fetch is the latest and nothing was delivered before; re-reads
        the cache only after a retried commit; else ``_NOT_WHOLE``, unchanged."""
        spans = lease._block_entries.spans
        if spans is None or self._shut_down or self._stop_reason is not None:
            return _NOT_WHOLE
        manager = self._manager_ref()
        request_id = lease._request_id
        kv = lease._kv
        if manager is None or not self._lends_fast():
            return _NOT_WHOLE
        ledger = self._ledger
        try:
            if _manager.kv_of(manager, request_id) is not kv:
                return _NOT_WHOLE
            state, blocks = extent = _manager.cache_extent(kv)
            if not self._lent_pages_held(lease, kv, extent) or not ledger._defer_first_delivery(
                request_id, lease._fetch, kv, spans
            ):
                return _NOT_WHOLE
        except Exception:  # noqa: BLE001 - the rules then return the lease and raise the error
            return _NOT_WHOLE
        lease._finish_returned_write()
        writes_at_end = self._writes_at_end
        if writes_at_end:
            writes_at_end.pop(id(kv), None)
        lock = lease._lock
        retried = lock is not None and self._locks.unlock(lock)
        if self._stop_reason is not None:
            return self._bounds_after_return(lease)
        if retried:
            manager = self._serving_manager()
            if manager is None or _manager.kv_of(manager, request_id) is not kv:
                return None
            state, blocks = _manager.cache_extent(kv)
        fetch = lease._fetch
        bounds = ledger._whole_delivery_bounds(
            fetch.prompt, fetch.runs, request_id, manager, kv, state, spans, blocks
        )
        if bounds is None:
            bounds = ledger._resume_bounds(fetch.prompt, fetch.runs, request_id, manager, kv, state)
        return bounds

    def _count_arrivals(
        self, lease: _StagingLease | _InPlaceLease, masks: list[np.ndarray] | None
    ) -> bool:
        """Count a returning write lease's arrived rows, every row for ``masks`` ``None``; whether
        they are the cache's first delivery, every row arriving into pages the lease still holds."""
        manager = None if self._shut_down else self._manager_ref()
        if manager is None or self._stop_reason is not None:
            return False
        if type(lease) is _StagingLease:
            self._apply_marks(lease, self._every_row_mask(lease) if masks is None else masks)
            return False
        request_id = lease._request_id
        kv = _manager.kv_of(manager, request_id)
        spans = lease._block_entries.spans
        held = spans is not None and self._lends_fast() and self._lent_pages_held(lease, kv)
        if (
            held
            and (masks is None or all(mask.all() for mask in masks))
            and self._ledger._defer_first_delivery(request_id, lease._fetch, kv, spans)
        ):
            return True
        self._apply_in_place_marks(
            lease, self._every_row_mask(lease) if masks is None else masks, lock_held=held
        )
        return False

    def _lent_pages_held(
        self,
        lease: _InPlaceLease,
        kv: _KVCache | None,
        extent: tuple[_manager.CacheState, int] | None = None,
    ) -> bool:
        """Whether every row a fast write lent still holds its page, kept by the lock a returned
        view holds: same active cache, no shrink reported since the lend, its blocks kept and a
        window's history at most the fetch's end. A locked page changes only at a shrink or free, so
        ``_locked_block_masks`` is then true for every row. ``extent``: ``kv``'s, if read."""
        if kv is not lease._kv:
            return False
        epochs = self._shrink_epochs
        if (epochs.get(lease._request_id, 0) if epochs else 0) != lease._shrinks:
            return False
        state, blocks = _manager.cache_extent(kv) if extent is None else extent
        if not state.active or blocks < lease._block_end:
            return False
        return not self._any_window or state.history <= lease._fetch.end

    @staticmethod
    def _every_row_mask(lease: _StagingLease | _InPlaceLease) -> list[np.ndarray]:
        """A true mask for every row of the lease's view, in group order."""
        return [np.ones(len(o), dtype=bool) for o in lease._block_entries.block_ordinals]

    @staticmethod
    def _no_row_mask(lease: _StagingLease | _InPlaceLease) -> list[np.ndarray]:
        """A false mask for every row of the lease's view, in group order."""
        return [np.zeros(len(o), dtype=bool) for o in lease._block_entries.block_ordinals]

    def _bounds_after_return(self, lease: _StagingLease | _InPlaceLease) -> ResumeBounds | None:
        """``poll_resume_bounds`` of the write lease's cache once it returned, ``None`` where that
        cache is no longer the request's, the request has none or the manager shut down."""
        request_id = lease._request_id
        manager = self._serving_manager()
        kv = None if manager is None else _manager.kv_of(manager, request_id)
        if kv is None or kv is not lease._kv:
            return None
        state = _manager.cache_state(kv)
        if self._stop_reason is not None:
            return ResumeBounds(max_position=0, min_position=max(state.history, 1))
        fetch = lease._fetch
        return self._ledger._resume_bounds(fetch.prompt, fetch.runs, request_id, manager, kv, state)

    def _on_free(self, request_id: int, kv_cache: _KVCache) -> None:
        """Manager hook, after the request's cache left the map: its waiting leases fail and its
        fetch records go. An error in that work, which it logs, stops the lender for good."""
        try:
            if self._shut_down:
                return
            request_id = int(request_id)
            if self._writes_at_end:
                self._lose_end_resume(kv_cache)
            # Granted leases go on: a read's copy is queued already, and a write's marks copy
            # nothing into pages other than the ones lent.
            if self._waiting_for_slots:
                self._fail_waiting(request_id, "the request was freed")
            self._ledger._drop_records(request_id)
            self._ordinal_ranges.pop(request_id, None)
            self._shrink_epochs.pop(request_id, None)
            self._state_unverified.discard(request_id)
            parked = self._parked.pop(request_id, None)
            if parked is not None:
                # The cache writes into its row until it closes, after ``_defer_close`` or at its
                # last unlock; rows of closed caches go back to use here, off a slot release's path.
                if self._parked_closing:
                    self._reuse_closed_rows()
                self._parked_closing.append(parked)
        except Exception:
            # Logged, not raised into the free, which then closes the cache; records may disagree
            # with slots unseen, so the lender stops, setting its reason first, which cannot raise.
            if self._stop_reason is None:
                self._stop_reason = _REASON_STOPPED
            if self._writes_at_end:
                self._lose_end_resumes()
            logger.error(f"KV cache lender: freeing request {request_id}: {traceback.format_exc()}")

    def _on_resume(self, request_id: int) -> None:
        """Manager hook after a resume, which may move recurrent state the manager records only at
        the next preparation: the cache answers for the slot until the record matches it."""
        record = self._state_record
        if record is not None and int(request_id) in record:
            self._state_unverified.add(int(request_id))

    def _parking_row(self, request_id: int, kv_cache: _KVCache) -> tuple[memoryview, ...] | None:
        """Manager hook at an early index-slot release: a lender-owned row's buffers, one per layer
        group, for the cache's beam-0 page indices from then on and for lends' reads; None, keeping
        nothing, without a page-index buffer."""
        if self._index_words is None or self._shut_down:
            return None
        request_id = int(request_id)
        previous = self._parked.get(request_id)
        if previous is not None:
            # A release of a slot released already: the cache leaves the earlier row at this bind.
            self._parked_closing.append(previous)
        row = self._unbound_row()
        row.kv = kv_cache
        self._parked[request_id] = row
        return row.buffers

    def _parked_row(self, request_id: int) -> tuple[memoryview, ...] | None:
        """Manager hook, right after a cache resumed whose index slot was released early: the
        buffers ``_parking_row`` gave for the request, to bind again, or None."""
        row = self._parked.get(int(request_id))
        return None if row is None else row.buffers

    def _unbound_row(self) -> _ParkedRow:
        """A row no open cache writes into, every page index in it bad: a row whose cache has
        closed, or a new one."""
        if not self._parked_free:
            self._reuse_closed_rows()
        if self._parked_free:
            row = self._parked_free.pop()
            # A row a failed bind leaves unbound then gives no page.
            row.bytes[:] = self._bad_row
            return row
        row = _ParkedRow(len(self._index_rows), self._index_strides[2])
        self._bad_row = b"\xff" * row.pages.nbytes
        self._parked_made.append(row)
        return row

    def _reuse_closed_rows(self) -> None:
        """Put the rows of freed requests whose caches have closed back to use."""
        still_open = []
        for row in self._parked_closing:
            if _manager.is_closed(row.kv):
                row.kv = None
                self._parked_free.append(row)
            else:
                still_open.append(row)
        self._parked_closing = still_open

    def _page_row(
        self, manager: KVCacheManagerV2, request_id: int, kv: _KVCache
    ) -> tuple[memoryview, int, int, int] | None:
        """Where active ``kv`` keeps its beam-0 page indices for lends to read: the request's row of
        the page-index buffer while it holds one, else the row parked for ``kv`` at its early
        release; None without either."""
        words = self._index_words
        if words is None:
            return None
        row = _manager.index_row(manager, request_id)
        if row is not None:
            # The key half of the slot's row.
            group_words, row_words, row_blocks = self._index_strides
            return words, row * row_words, group_words, row_blocks
        parked = self._parked.get(request_id)
        return None if parked is None or parked.kv is not kv else parked.source

    def _state_slot(self, kv: _KVCache, request_id: int | None, lg: int) -> int:
        """The request's slot of recurrent group ``lg``: the manager's record, which the cache
        confirms once after each resume; the cache's own answer where the manager records none for
        the request or the group."""
        record = self._state_record
        if record is None or lg != self._state_group or request_id is None:
            return _manager.state_slot(kv, lg)
        if request_id not in self._state_unverified:
            slot = record.get(request_id)
            if slot is not None:
                return slot
            return _manager.state_slot(kv, lg)
        slot = _manager.state_slot(kv, lg)
        if record.get(request_id) == slot:
            self._state_unverified.discard(request_id)
        return slot

    def _on_shrink(self, request_id: int, kv_cache: _KVCache) -> None:
        """Manager hook right after a possible in-place shrink: delivered entries past its blocks
        stop counting, as the shrink freed their pages, and the request's fast writes stop trusting
        the pages they lent. Logs its own errors."""
        try:
            if self._writes_at_end:
                self._lose_end_resume(kv_cache)
            epochs = self._shrink_epochs
            epochs[int(request_id)] = epochs.get(int(request_id), 0) + 1
            delivered = self._ledger._deliveries_of(int(request_id))
            if (
                self._serving_manager() is not None
                and delivered is not None
                and delivered.kv is kv_cache
            ):
                self._ledger._invalidate_delivered_blocks_past_capacity(delivered, kv_cache)
        except Exception:
            # Logged, not raised into the manager's resize. All the request's delivered entries go:
            # the cache may grow back before poll_resume_bounds finds the freed ones past its blocks.
            self._ledger._drop_deliveries(int(request_id))
            logger.error(
                f"KV cache lender: shrink of request {request_id}: {traceback.format_exc()}"
            )

    def _before_reset(self) -> None:
        """Manager hook before a reuse reset, which the runtime refuses with a cache open:
        ``RuntimeError``, changing nothing, while a lock keeps a freed request's cache open, until
        its last unlock closes it."""
        if self._locks.holds_open_caches():
            raise RuntimeError(
                "KV cache lender: a lock keeps a freed request's KV cache open until its lease is "
                "released, and the reuse state cannot be reset before"
            )

    def _on_reset(self) -> None:
        """Stop NAMED lending, as a fixed scope cannot name updated weights; caches are closed and
        frees removed waiting leases and fetches; granted publishes finish. Never raises."""
        self._reuse_was_reset = True
        self._ordinal_ranges.clear()
        if self._writes_at_end:
            self._lose_end_resumes()

    def _lose_end_resume(self, kv_cache: _KVCache) -> None:
        """The open fast write lease of ``kv_cache``, if any, no longer resumes at its end."""
        lease = self._writes_at_end.pop(id(kv_cache), None)
        if lease is not None:
            lease._at_end = False

    def _lose_end_resumes(self) -> None:
        """No open fast write lease resumes at its end any more."""
        for lease in self._writes_at_end.values():
            lease._at_end = False
        self._writes_at_end.clear()

    def _on_shutdown(self) -> None:
        """Once: wait for recorded copies, fail waiters and stop lending, logging errors; free
        staging unless open leases or holds, lost slots or a stop keep it. Device-region holds keep
        the runtime and its pools until exit."""
        if self._shut_down:
            return
        self._ordinal_ranges.clear()
        if self._writes_at_end:
            self._lose_end_resumes()
        if self._device_regions_of_pool_group and self._open_holds:
            _retain_until_exit(_manager.runtime(self._manager_ref()))
        try:
            # With page-locked staging and the fresh-page fill off, the lender's only host
            # wait: no more work comes on the stream.
            for lease in self._leases_holding_slots:
                if lease._copy is not None:
                    lease._copy.wait()
            self._progress_round += 1
            if self._stop_reason is None:  # a stopped lender's records may disagree with its slots
                self._recycle()
            for lease in list(self._waiting_for_slots):
                self._fail(lease, _REASON_SHUT_DOWN)
            for lease in [
                lease for lease in self._open_leases if lease._first_poll_failure is not None
            ]:
                self._fail(lease, lease._first_poll_failure)
            self._shut_down = True
            memory = self._memory_ref()  # None for a lender that stages nothing
            if memory is not None and not self._must_retain_memory():
                # Every copy on it has completed above, and no lease or hold reaches it any more.
                _stop_retaining(memory)
                memory.free()
            elif memory is not None:
                logger.warning(
                    f"KV cache lender: keeping {self._staging_bytes() >> 20} MiB of staging memory "
                    f"until exit: {len(self._open_leases)} leases open, "
                    f"{len(self._open_holds)} region holds open, "
                    f"{len(self._quarantined)} slot runs lost to failed copies"
                    f"{'' if self._stop_reason is None else ', and ' + self._stop_reason}"
                )
        except Exception:
            # Logged, not raised into the shutdown; staging stays until exit. As in ``_on_free``
            # records may disagree with slots: the lender stops first, keeping an earlier reason.
            self._shut_down = True
            if self._stop_reason is None:
                self._stop_reason = _REASON_STOPPED_AT_SHUTDOWN
            if self._writes_at_end:
                self._lose_end_resumes()
            logger.error(f"KV cache lender: shutting down: {traceback.format_exc()}")

    def _on_caches_closed(self) -> None:
        """Manager hook once the runtime's shutdown returned and no cache writes the page-index
        buffer: let it go. A raising shutdown skips this, keeping it until a retry or exit."""
        if self._index_buffer is None:
            return
        self._stop_retaining_index_buffer()

    def _page_lock_state(self) -> PageLocks:
        """Live lock registry, whose ``holding`` counts the unreleased locks holding a block."""
        return self._locks

    def _is_locked(self, kv_cache: _KVCache | None, begin: int = 0, end: int | None = None) -> bool:
        """Manager hook before a suspension or commit: whether a lock holds a block of ``kv_cache``
        in ``[begin, end)``, or for ``None`` of any cache, freed ones kept open included."""
        if kv_cache is None:
            return self._locks.holds_any()
        return self._locks.is_locked(kv_cache, begin, end)

    def _defer_close(self, kv_cache: _KVCache) -> bool:
        """Manager hook at a free or shutdown, in place of a close: whether locks keep ``kv_cache``
        open until its last unlock. One closed at once frees the row parked for it at its free, as
        no release takes a row before the close."""
        deferred = self._locks.defer_close(kv_cache)
        closing = self._parked_closing
        if not deferred and closing and closing[-1].kv is kv_cache:
            row = closing.pop()
            row.kv = None
            self._parked_free.append(row)
        return deferred

    def _keeps_pools(self) -> bool:
        """Manager hook, at its shutdown: whether the runtime, whose shutdown frees the pools, must
        keep them: a cache ``_defer_close`` kept is still open, or a hold covers device regions."""
        return self._locks.holds_open_caches() or bool(
            self._device_regions_of_pool_group and self._open_holds
        )

    def _history_cap(self, kv_cache: _KVCache) -> int | None:
        """Manager hook before a history move: the highest history leaving no locked block behind a
        window, or ``None``; the manager stops there until the unlock, keeping the lock's page."""
        if not self._any_window:
            return None
        manager = self._manager_ref()
        if manager is None:
            return None
        cap = None
        layout = self._layout
        for lg, lowest in self._locks.lowest_locked_blocks(kv_cache, layout.sink_blocks).items():
            if layout.windows[lg] is not None:
                last = _ranges._last_history_keeping(manager, layout, lg, lowest)
                cap = last if cap is None else min(cap, last)
        return cap

    def _staging_bytes(self) -> int:
        """The bytes of the staging regions."""
        return sum(p.nbytes for p in self._regions if p.memory_kind is MemoryKind.HOST)

    # One rule per method, so a test subclass that breaks exactly one rule overrides one method.

    def _progress(self) -> None:
        """Return the slots of recyclable leases, then grant waiting leases in order; a stopped
        lender does neither and fails the waiting leases."""
        if self._serving_manager() is None:
            return
        if self._stop_reason is not None:
            for lease in list(self._waiting_for_slots):
                self._fail(lease, self._stop_reason)
            return
        self._progress_round += 1
        if self._leases_holding_slots or self._waiting_for_slots:
            self._recycle()
            self._grant_waiting()

    def _copy_done(self, copy: _QueuedCopy | None) -> bool:
        """``copy`` has completed (or there is none), its event asked at most once this round."""
        return copy is None or copy.done(self._progress_round)

    def _read_copy_done(self, lease: _StagingLease) -> bool:
        """A read lease's copy into its slots has completed."""
        return self._copy_done(lease._copy)

    def _recyclable(self, lease: _StagingLease) -> bool:
        """The lease's slots may return: no backend access possible and no copy on them pending.
        The copy is asked last, so a lease still lent costs no event query."""
        # A failed lease was never ready, so no backend has seen its slots.
        if lease._failure is None and (not lease._released or lease._marks_due()):
            return False
        return self._copy_done(lease._copy)

    def _locked_block_masks(self, kv: _KVCache | None, lease: _LeaseBase) -> list[np.ndarray]:
        """Per layer group, entries still in the same active cache's window locking the GPU page
        lent (a held page may sit on another tier), and state rows still its state page."""
        entries = lease._block_entries
        manager = self._manager_ref()
        state = _manager.cache_state(kv) if kv is not None and kv is lease._kv else None
        masks = []
        for lg, block_ordinals, slots in zip(
            entries.layer_group_indices, entries.block_ordinals, entries.device_pages
        ):
            if state is None or not state.active:
                same = np.zeros(len(block_ordinals), dtype=bool)
            elif lg in self._recurrent_groups:
                same = slots == self._state_slot(kv, lease._request_id, lg)
            else:
                pages = _manager.locked_pages(kv, lg)
                try:
                    same = pages[block_ordinals] == slots
                except IndexError:
                    # Blocks past the cache's end lock no page.
                    same = np.zeros(len(block_ordinals), dtype=bool)
                    inside = block_ordinals < len(pages)
                    same[inside] = pages[block_ordinals[inside]] == slots[inside]
                if self._layout.windows[lg] is not None:
                    stale_beg, stale_end = _ranges._stale_range(
                        manager, self._layout, lg, state.history
                    )
                    same &= (block_ordinals < stale_beg) | (block_ordinals >= stale_end)
            masks.append(same)
        return masks

    def _unnamed_input_refusal(self, request: LlmRequest) -> str | None:
        """Why the request's KV depends on more than its object_keys cover, if it does: multimodal
        data the manager keys by placeholder tokens alone, or encoder input."""
        if _manager.keyed_by_placeholders(request):
            return _REASON_PLACEHOLDERS
        if _manager.has_encoder_input(request):
            return _REASON_ENCODER
        return None

    # TODO: the lender cannot see the model's sliding window, so it also keeps fetch ends and
    # resumes out of runs that window would cover whole.
    def _bidirectional_split_refusal(self, request: LlmRequest, end: int) -> str | None:
        """Why the fetch ends strictly inside a run of multimodal tokens the scheduler keeps within
        one context chunk, if it does, whatever the run's length."""
        for b, e in _manager.bidirectional_runs(request):
            if b < end < e:
                return (
                    f"the range ends at {end}, inside the multimodal tokens [{b}, {e}), a run the "
                    "scheduler keeps within one context chunk"
                )
        return None

    def _scratch_reuse_on(self, kv: _KVCache) -> bool:
        """Whether ``kv`` has SWA scratch reuse on (``_manager.has_scratch_reuse``), which a fetch's
        grow breaks."""
        return _manager.has_scratch_reuse(kv)

    def _committed_entries(self, kv: _KVCache, entries: _LentBlockEntries) -> _LentBlockEntries:
        """``entries`` without the blocks past each layer group's committed whole blocks: a group
        the prefix claim left out holds none of the claimed blocks a name covers."""
        tpb = int(self._layout.tokens_per_block)
        committed = _manager.committed_by_group(kv, self._layout.num_layer_groups)
        out = _LentBlockEntries([], [], [])
        for lg, ordinals, slots in zip(
            entries.layer_group_indices, entries.block_ordinals, entries.device_pages
        ):
            keep = (ordinals < committed[lg] // tpb) & self._named_groups[lg]
            out.layer_group_indices.append(lg)
            whole = ordinals.ndim == 1 and slots.shape == ordinals.shape and bool(keep.all())
            out.block_ordinals.append(ordinals if whole else _immutable(ordinals[keep]))
            out.device_pages.append(slots if whole else _immutable(slots[keep]))
        return out

    def _without_short_windows_mid_context(
        self,
        manager: KVCacheManagerV2,
        request: LlmRequest,
        state: _manager.CacheState,
        entries: _LentBlockEntries,
    ) -> _LentBlockEntries:
        """Until this worker's context is computed, omit groups whose window is shorter than it.
        Their locks would block later chunks from advancing history; lend them after context."""
        context_end = self._mid_context_end(manager, request, state)
        if context_end is None:
            return entries
        out = _LentBlockEntries([], [], [])
        for lg, block_ordinals, slots in zip(
            entries.layer_group_indices, entries.block_ordinals, entries.device_pages
        ):
            window = self._layout.windows[lg]
            keep = window is None or int(window) >= context_end
            out.layer_group_indices.append(lg)
            out.block_ordinals.append(block_ordinals if keep else block_ordinals[:0])
            out.device_pages.append(slots if keep else slots[:0])
        return out

    def _mid_context_end(
        self, manager: KVCacheManagerV2, request: LlmRequest, state: _manager.CacheState
    ) -> int | None:
        """The pending context end while a windowed cache is still computing it, if any."""
        if not self._any_window:
            return None
        context_end = _manager.context_end(request, manager)
        if state.history >= context_end or _manager.context_computed(request) >= context_end:
            return None
        return context_end

    # TODO: context outputs cover only computed positions, prompt logprobs pair logits from the
    # prompt's second token, and rewinds or recompute pauses keep them: such requests take no fetch.
    def _returns_context_outputs(self, request: LlmRequest) -> bool:
        """Whether the request returns context logits or asks for additional model outputs, which a
        fetch leaves without the fetched positions, or after a context step gapped or repeated."""
        return _manager.returns_context_outputs(request)

    def _history_refusal(
        self, manager: KVCacheManagerV2, history: int, position: int, end: int
    ) -> str | None:
        """Why the history ``position`` the fetch leaves admits no resume the bounds would count, or
        ``None``: without a window and with the floor following it, a history past ``end``."""
        if self._any_window:
            return None  # the fetch moves the history to ``end``
        if not _ranges._floor_follows_history(manager):
            return None  # the floor is a delivered fetch's lowest start, or the history if lower
        if position > end:
            return f"the cache's history of {history} tokens stands past the fetch's end, {end}"
        return None

    def _unwritten_pages_refusal(
        self, manager: KVCacheManagerV2, request_id: int, kv: _KVCache, position: int
    ) -> str | None:
        """Refuse a history move that would commit unwritten window pages (partial prefixes, early
        growth, incomplete fetches), else ``None``; a delivered row supplies the bytes."""
        if not _manager.keeps_stale_pages(manager):
            return None
        computed = self._ledger._computed_tokens(request_id, kv)
        tpb = int(self._layout.tokens_per_block)
        delivered = self._ledger._deliveries_of(request_id)
        fetched = (
            delivered.delivered_block_masks
            if delivered is not None and delivered.kv is kv
            else None
        )
        for lg in range(self._layout.num_layer_groups):
            stale_beg, stale_end = _ranges._stale_range(manager, self._layout, lg, position)
            beg = max(computed[lg] // tpb, stale_beg)  # from the first block past them
            if beg >= stale_end:
                continue
            unwritten = _manager.locked_pages(kv, lg)[beg:stale_end] >= 0
            if fetched is not None:
                got = fetched[lg][beg : beg + len(unwritten)]
                unwritten[: len(got)] &= ~got
            blocks = (beg + np.nonzero(unwritten)[0]).tolist()
            if blocks:
                # TODO: no runtime call drops one block's page in one layer group, so the fetch
                # fails where the commit could store no page for those blocks instead.
                return (
                    f"layer group {lg} leaves blocks {blocks[:8]} behind its window at {position} "
                    f"tokens; past the {computed[lg]} tokens the request computed, their pages hold "
                    "bytes it never wrote and no fetch delivered, which the commit would store"
                )
        return None

    def _fail_waiting(self, request_id: int, reason: str) -> None:
        """Fail the request's leases still waiting for slots."""
        for lease in [
            lease for lease in self._waiting_for_slots if lease._request_id == request_id
        ]:
            self._fail(lease, reason)

    def _object_keys(self, layer_group_index: int, reuse_keys: np.ndarray) -> np.ndarray:
        """The entries' object_keys: ``uint8 (n, 54)`` for ``reuse_keys`` ``uint8 (n, 32)``."""
        return self._identity.object_keys(layer_group_index, reuse_keys)

    def _must_retain_memory(self) -> bool:
        """A lease or a hold is unreleased, a slot is lost to a failed copy, or the lender stopped:
        the memory stays."""
        return (
            bool(self._open_leases)
            or bool(self._open_holds)
            or self._slots_quarantined()
            or self._stop_reason is not None
        )

    def _slots_quarantined(self) -> bool:
        """Whether slots are out of use for good: runs ``_quarantine`` took, or those of a holding
        lease whose copy has no record of its completion, which ``_recycle`` never returns."""
        return bool(self._quarantined) or any(
            lease._copy is _COPY_WITHOUT_RECORD for lease in self._leases_holding_slots
        )

    def _retain_index_buffer(self, manager: KVCacheManagerV2) -> None:
        """Keep the manager's page-index buffer until the runtime's shutdown returned: a cache a
        lease or record holds writes it as it closes, even once the manager is gone."""
        self._index_buffer = _manager.index_buffer(manager)
        _retain_until_exit(self._index_buffer)
        _retain_until_exit(self._parked_made)
        self._read_index_rows(np.asarray(self._index_buffer))

    def _stop_retaining_index_buffer(self) -> None:
        """Once the runtime's shutdown has returned: no cache writes into the page-index buffer any
        more."""
        self._read_index_rows(None)
        _stop_retaining(self._index_buffer)
        self._index_buffer = None
        # Nor into a parked row.
        _stop_retaining(self._parked_made)
        self._parked.clear()
        self._parked_closing.clear()
        self._parked_free.clear()
        self._parked_made = []

    def _read_index_rows(self, rows: np.ndarray | None) -> None:
        """Fast lends read pages from ``rows``, a contiguous ``int32`` ``[group, row, K/V, block]``
        buffer kept as is, and parked rows; from the caches for ``None`` or other arrays."""
        if rows is None or rows.dtype != np.int32 or rows.ndim != 4 or not rows.flags.c_contiguous:
            self._index_rows = self._index_words = None
            return
        _, count, halves, blocks = rows.shape
        self._index_rows = rows
        self._index_words = memoryview(rows).cast("B").cast("i")
        self._index_strides = (count * halves * blocks, halves * blocks, blocks)

    def _on_hold_released(self, hold: _RegionHold) -> None:
        """The hold's holder let go; ``KeyError`` for a hold not open, which its guard prevents."""
        self._open_holds.remove(hold)

    def _fits_refusal(self, counts: Mapping[int, int]) -> str | None:
        """Why a lease needing ``counts`` entries per pool group can never be granted here, or
        ``None``: this stage's staging holds fewer slots."""
        try:
            self._slot_queue.check_fits_capacity(counts)
        except ValueError as error:
            return (
                f"the range needs more staging slots than there are ({error}); ranges of at most "
                "guaranteed_fetch_tokens tokens always fit"
            )
        return None

    def _read_end_refusal(
        self,
        manager: KVCacheManagerV2,
        request: LlmRequest,
        state: _manager.CacheState,
        end: int,
        named: bool,
    ) -> str | None:
        """Why this stage cannot read through ``end`` now, or ``None``."""
        if named:
            if end > state.committed:
                return f"the range ends at {end}, past the {state.committed} committed tokens"
            return None
        # A window held for a lock keeps the history below what finished context steps computed.
        computed = _manager.context_computed(request)
        if end > state.history and computed < end:
            return f"the range ends at {end}, past the {state.history} tokens the cache holds"
        # A later context step writes the rest of a partial block; generation writes past the
        # history alone.
        if end % int(self._layout.tokens_per_block):
            context_end = _manager.context_end(request, manager)
            if context_end > state.history and computed < context_end:
                return (
                    f"the range ends at {end}: until this worker computes its context to "
                    f"{context_end} tokens, a read of every row ends at a whole block"
                )
        if not self._recurrent_groups:
            return None
        # The state stands at the cache's history only from the context's last step until the
        # request's first generation step, whose speculation leaves it at a checkpoint.
        prompt = _manager.prompt_length(request, manager)
        if not end == state.history == prompt:
            return (
                f"the range ends at {end}: with recurrent state a lease ends where the state "
                f"stands, at the prompt's {prompt} tokens once its context is computed and "
                f"before a generation step, the cache holding {state.history}"
            )
        return None

    def _unfetched_rows_refusal(
        self,
        request_id: int,
        kv: _KVCache,
        state: _manager.CacheState,
        entries: _LentBlockEntries,
        end: int,
    ) -> str | None:
        """Why a read in place would lend rows below the history nothing wrote, or ``None``: rows a
        fetch lent or grew over without delivering, which the request had not computed."""
        found = self._ledger._unheld_block(
            request_id, kv, entries, min(end, state.history), state.history
        )
        if found is None:
            return None
        lg, block = found
        if lg in self._recurrent_groups:
            return f"recurrent state has no complete fetched snapshot at {end} tokens"
        return (
            f"layer group {lg}: block {block} holds rows below the history of {state.history} "
            "tokens that a fetch lent or grew the cache over and did not deliver"
        )

    def _write_range_refusal(
        self,
        manager: KVCacheManagerV2,
        request: LlmRequest,
        state: _manager.CacheState,
        start: int,
        end: int,
        prompt: int,
    ) -> str | None:
        """Why this stage cannot fetch ``[start, end)`` now, or ``None``; changes nothing."""
        tpb = int(self._layout.tokens_per_block)
        if end % tpb and end not in (prompt, state.history):
            return (
                f"the range ends at {end}: a fetch of every row ends at a whole block, at the "
                f"request's {prompt} prompt tokens or at the {state.history} tokens of history "
                "its cache declares"
            )
        # A KV cache connector loads past the prefix served to the request, never into it.
        served = _manager.served_prefix_end(request) if _manager.connector_loads(manager) else 0
        if start < served:
            return (
                f"the range starts at {start}, inside the {served} tokens served to the request "
                "from outside the manager, where a KV cache connector loads"
            )
        # TODO: lendable state is kept only at the latest position: no fetch resumes mid-prompt.
        if self._recurrent_groups and end != prompt:
            return (
                f"the range ends at {end}: with recurrent state a fetch ends where the state "
                f"stands, at the request's {prompt} prompt tokens"
            )
        if self._any_window and end < state.history:
            return (
                f"the range ends at {end}, below the history of {state.history} tokens its "
                "windows keep"
            )
        return None

    def _free_slots(self, group: int) -> int:
        """Free slots of a pool group; for tests."""
        return self._slot_queue.free_slots(group)

    def _open_lease_count(self) -> int:
        """Leases not yet released; for tests."""
        return len(self._open_leases)

    # Lease records, slots and grants.

    def _serving_manager(self) -> KVCacheManagerV2 | None:
        """The manager while the lender serves it; ``None`` once it shut down or is gone."""
        if self._shut_down:
            return None
        return self._manager_ref()

    def _lending_manager(self) -> KVCacheManagerV2:
        """The manager a lend serves; ``RuntimeError`` once it shut down or is gone, when the
        lender lends no more."""
        manager = self._serving_manager()
        if manager is None:
            raise RuntimeError(f"the KV cache lender lends no more: {_REASON_SHUT_DOWN}")
        return manager

    def _check_not_stopped(self) -> None:
        """``RuntimeError`` once the lender stopped for good: it lends no more."""
        if self._stop_reason is not None:
            raise RuntimeError(f"the KV cache lender lends no more: {self._stop_reason}")

    def _check_memory_and_rows(self, memory_kind: MemoryKind, rows: RowSelection) -> None:
        """``TypeError`` for arguments of other types; ``ValueError`` for named rows where nothing
        is named, host rows other than named ones, or host memory where nothing is staged."""
        if not isinstance(memory_kind, MemoryKind):
            raise TypeError(f"memory_kind must be a MemoryKind, got {type(memory_kind).__name__}")
        if not isinstance(rows, RowSelection):
            raise TypeError(f"rows must be a RowSelection, got {type(rows).__name__}")
        if rows is RowSelection.NAMED and self._no_names is not None:
            raise ValueError(
                f"{rows} is not lent: this lender lends nothing by name, as {self._no_names}"
            )
        # TODO: staging relays named whole blocks only, so host memory lends no other rows.
        if memory_kind is MemoryKind.HOST and rows is not RowSelection.NAMED:
            raise ValueError(
                f"{rows} is not lent in {memory_kind}: staging relays named whole blocks"
            )
        # TODO: the manager's own host-tier pages cannot be lent; host lending copies into staging.
        if memory_kind is MemoryKind.HOST and self._no_staging is not None:
            raise ValueError(
                f"{memory_kind} is not lent: this lender stages nothing, as {self._no_staging}"
            )

    def _whole_blocks(self, start: int, end: int, whole_end: bool = True) -> tuple[int, int]:
        """Validate integer ``[start, end)``; require an ordered, non-negative range starting,
        and with ``whole_end`` also ending, at a whole block."""
        for name, value in (("start", start), ("end", end)):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
        start, end = int(start), int(end)
        tpb = self._tokens_per_block
        if start < 0 or end < 0 or start > end:
            raise ValueError(f"bad token range [{start}, {end})")
        if start % tpb or (whole_end and end % tpb):
            raise ValueError(
                f"a lease of named rows covers whole blocks of {tpb} tokens, got [{start}, {end})"
                if whole_end
                else f"a lease starts at a whole block of {tpb} tokens, got [{start}, {end})"
            )
        return start, end

    def _required_slots_per_pool_group(self, entries: Sequence[int]) -> dict[int, int]:
        """Required slots per pool group, summed from each layer group's block entry count."""
        counts: dict[int, int] = {}
        for lg, num_entries in enumerate(entries):
            g = int(self._layout.pool_group_of[lg])
            counts[g] = counts.get(g, 0) + int(num_entries)
        return counts

    def _admit(self, lease: _StagingLease, counts: Mapping[int, int]) -> None:
        """Record a new lease and grant it now when no lease waits and its slots are free, or at
        once, out of line, when it has no entries."""
        self._open_leases.add(lease)
        try:
            if lease._block_entries.num_rows == 0:
                # Nothing to stage: ready at the first poll, without slots or a place in line.
                self._grant(lease, None)
                return
            lease._ticket = self._slot_queue.enqueue(counts)
            self._waiting_for_slots.append(lease)
            self._grant_waiting(just_admitted=lease)
        except BaseException:
            # The caller never received this lease. Copies without completion records still keep
            # their slots quarantined, independently of the lease's open state.
            if lease._ticket is not None:
                self._slot_queue.cancel(lease._ticket)
                lease._ticket = None
                if lease in self._waiting_for_slots:
                    self._waiting_for_slots.remove(lease)
            self._open_leases.discard(lease)
            if lease._kind == "write":
                self._ledger._abandon_fetch(lease)
            raise

    def _grant_waiting(self, just_admitted: _StagingLease | None = None) -> None:
        """Grant the leases in line, strictly in order; ``just_admitted`` was looked up in this
        call."""
        while self._waiting_for_slots:
            head = self._waiting_for_slots[0]
            # TODO: an exception while slots are taken can leave them taken for no lease, or the
            # head in line ticketless, so each progress round raises "not waiting" until it fails.
            runs = self._slot_queue.try_grant(head._ticket)
            if runs is None:
                return
            self._waiting_for_slots.popleft()
            head._ticket = None
            try:
                self._grant(head, runs, recheck=head is not just_admitted)
            except Exception as error:
                # One broken grant fails only its own lease: it neither stalls the line nor raises
                # out of another lease's call. Slots a copy may have been queued on stay unused.
                if head._kind == "read" and head._slot_runs is not None:
                    self._quarantine(head)
                else:
                    head._view = head._slot_runs = None
                    self._slot_queue.give_back(runs)
                reason = f"granting staging slots failed: {error!r}"
                if head is just_admitted and head._kind == "write":
                    # The cache grew for it, so it fails at its first poll, which abandons the
                    # fetch, as a lease missing pages does.
                    head._defer_failure(reason)
                else:
                    self._fail(head, reason)
                logger.warning(f"KV cache lender: {reason}")
            except BaseException:
                # An interrupt can leave a copy queued on the slots without its completion.
                head._slot_runs = runs
                try:
                    self._quarantine(head)
                finally:
                    self._fail(head, "granting staging slots was interrupted")
                raise

    def _grant(self, lease: _StagingLease, runs: SlotRuns | None, recheck: bool = False) -> None:
        """Give ``lease`` its slots and view; a read then queues its copy into them."""
        if recheck and lease._kind == "read":
            problem = self._changed_source_failure(lease)
            if problem is not None:
                self._slot_queue.give_back(runs)
                self._fail(lease, problem)
                return
        self._assign_staging_slots(lease._block_entries, runs)
        if lease._kind == "read" and runs is not None:
            # Lost until its copy is recorded: no raise from here on leaves it ready.
            lease._copy = _COPY_WITHOUT_RECORD
        lease._view = self._build_view(lease._block_entries, lease._reuse_keys, *lease._range)
        if runs is None:
            return
        lease._slot_runs = runs
        self._leases_holding_slots.append(lease)
        if lease._kind != "read":
            return
        copy, error = self._queue_copy(
            self._copy_segments(lease._block_entries), to_staging=True, after=lease._ready_event
        )
        if copy is None:
            self._quarantine(lease)
        if error is not None:
            self._fail(lease, f"the copy into staging failed: {error}")
        if copy is not None:
            lease._copy = copy

    def _changed_source_failure(self, lease: _StagingLease) -> str | None:
        """Why a read granted after waiting cannot copy the pages it looked up any more, if so."""
        manager = self._manager_ref()
        kv = _manager.kv_of(manager, lease._request_id)
        if kv is not lease._kv:
            return "the request's cache was freed while the lease waited"
        state = _manager.cache_state(kv)
        if not state.active:
            return "the request's cache was suspended while the lease waited"
        entries = lease._block_entries
        for lg, block_ordinals, slots in zip(
            entries.layer_group_indices, entries.block_ordinals, entries.device_pages
        ):
            pages = _manager.locked_pages(kv, lg)
            if np.any(block_ordinals >= len(pages)) or np.any(pages[block_ordinals] != slots):
                return f"layer group {lg}: pages changed while the lease waited"
            stale_beg, stale_end = _ranges._stale_range(manager, self._layout, lg, state.history)
            if np.any((block_ordinals >= stale_beg) & (block_ordinals < stale_end)):
                return f"layer group {lg}: blocks left the window while the lease waited"
        return None

    def _fail(self, lease: _StagingLease, reason: str) -> None:
        """Fail a lease never seen ready: it leaves the line, and a write abandons its fetch."""
        if lease.failure is not None:
            return
        lease._set_failure(reason)
        if lease._ticket is not None:
            self._slot_queue.cancel(lease._ticket)
            lease._ticket = None
            self._waiting_for_slots.remove(lease)
        if lease._kind == "write":
            self._ledger._abandon_fetch(lease)

    def _quarantine(self, lease: _StagingLease) -> None:
        """Keep the lease's slots from reuse for good: a copy on them may still be running."""
        # The copy is lost and the runs recorded before the lease leaves the holding list, so a step
        # that raises leaves its slots out of use and the staging memory kept at the shutdown.
        lease._copy = _COPY_WITHOUT_RECORD
        # TODO: quarantined slots are never reused, so a lease needing a longer run than any left
        # between lost slots waits in line for good, holding up every lease behind it.
        if lease._slot_runs is not None:
            self._quarantined.append(lease._slot_runs)
        self._leases_holding_slots = [
            held for held in self._leases_holding_slots if held is not lease
        ]
        logger.warning(
            f"KV cache lender: a failed copy took staging slots of request {lease._request_id}"
        )

    def _recycle(self) -> None:
        """Return the slots of every lease ``_recyclable`` allows."""
        # Every lease is judged before any slot returns: an event query that raises returns none
        # and leaves every lease held for the next call.
        recyclable = [self._recyclable(lease) for lease in self._leases_holding_slots]
        # TODO: an exception while slots return, a MemoryError say, can leave a lease held after
        # some or all of its slots returned, so every later round of progress raises "freed twice".
        holding = []
        for lease, done in zip(self._leases_holding_slots, recyclable):
            if done:
                self._slot_queue.give_back(lease._slot_runs)
            else:
                holding.append(lease)
        self._leases_holding_slots = holding

    def _on_lease_released(self, lease: _StagingLease) -> None:
        """The holder let go: a write released after its view but before its return delivers none of
        its rows; its slots return at a later progress round."""
        self._open_leases.discard(lease)
        if self._serving_manager() is None:
            return
        if lease._ticket is not None:
            self._fail(lease, "released while waiting for staging slots")
        elif lease._kind == "write" and lease._failure is None and not lease._view_returned:
            # Released before anyone saw it ready: no backend wrote, and no marks are due.
            lease._first_poll_failure = None
            self._ledger._abandon_fetch(lease)
        elif lease._kind == "write" and lease._view_returned and not lease._marked:
            lease._marked = True
            if self._stop_reason is None:
                self._apply_marks(lease, self._no_row_mask(lease))

    def _apply_marks(self, lease: _StagingLease, masks: list[np.ndarray]) -> None:
        """Copy marked entries whose page is still the one lent into the request's pages; the fetch
        stays abandoned until that copy queues without error, then joins the deliveries."""
        fetch = lease._fetch
        current = self._ledger._take_out_fetch(lease, fetch)
        kv = _manager.kv_of(self._manager_ref(), lease._request_id)
        lent = self._locked_block_masks(kv, lease)
        copy = [mask & still for mask, still in zip(masks, lent)]
        error = None
        if any(c.any() for c in copy):
            segments = self._copy_segments(lease._block_entries, copy)
            self._ledger._invalidate_delivered_blocks(lease, copy)
            tail = self._covers_committed_tail(kv, lease._block_entries, copy)
            queued = None
            try:
                queued, error = self._queue_copy(
                    segments, to_staging=False, after=lease._ready_event
                )
                lease._copy = queued
            finally:
                # A copy may be queued without its completion on the lease, also when the call
                # raised or was interrupted before the lease took it.
                if queued is None or lease._copy is not queued:
                    lease._copy = _COPY_WITHOUT_RECORD
                    self._quarantine(lease)
                if tail and (queued is None or error is not None):
                    self._ledger._record_mixed_tail(lease._request_id, kv)
        if current and error is None:
            self._ledger._put_back_fetch(lease, fetch, copy)

    def _covers_committed_tail(
        self,
        kv: _KVCache,
        entries: _LentBlockEntries,
        copied: list[np.ndarray],
        *,
        complement: bool = False,
    ) -> bool:
        """Whether the copied rows (with ``complement``, the others) include a group's partial
        committed tail block, counted by the bounds undelivered; state rows are not blocks."""
        tpb = int(self._layout.tokens_per_block)
        committed = _manager.committed_by_group(kv, self._layout.num_layer_groups)
        return any(
            committed[lg] % tpb and bool(np.any(o[~m if complement else m] == committed[lg] // tpb))
            for lg, o, m in zip(entries.layer_group_indices, entries.block_ordinals, copied)
            if lg not in self._recurrent_groups
        )

    def _range_ordinals(
        self, request_id: int | None, lg: int, runs: Sequence[tuple[int, int]]
    ) -> np.ndarray:
        """Reuse immutable ordinal ranges for successive chunks; keep at most two spans per group.
        Each span reserves one further chunk, and replacement leaves earlier views immutable."""
        if request_id is None:
            if not runs:
                return _NO_ROWS
            return _frozen_array(np.concatenate([np.arange(b, e, dtype=np.int64) for b, e in runs]))
        cached = self._ordinal_ranges.setdefault(request_id, {})
        parts = []
        for i, (begin, end) in enumerate(runs):
            previous = cached.get((lg, i))
            if previous is None or begin < previous[0] or end > previous[0] + len(previous[1]):
                previous = (begin, _frozen_array(np.arange(begin, 2 * end - begin, dtype=np.int64)))
                cached[(lg, i)] = previous
            base, values = previous
            parts.append(values[begin - base : end - base])
        if len(parts) == 1:
            return parts[0]
        if not parts:
            return _NO_ROWS
        return _frozen_array(np.concatenate(parts))

    def _block_entries_of_range(
        self,
        manager: KVCacheManagerV2,
        kv: _KVCache,
        start: int | Sequence[int],
        end: int,
        *,
        named: bool = False,
        request_id: int | None = None,
        ordinals: Sequence[np.ndarray] | None = None,
    ) -> _LentBlockEntries:
        """Retained rows of ``[start, end)`` at history ``end`` (``start`` one or per group), device
        pages -1 if none: state row 0 of an active cache, partial tails, under Helix this rank's
        pages that intersect the range."""
        layout = self._layout
        tpb = int(layout.tokens_per_block)
        rank, size = getattr(layout, "cp", (0, 1))
        starts = [start] * layout.num_layer_groups if isinstance(start, int) else list(start)
        entries = _LentBlockEntries([], [], [])
        for lg in range(layout.num_layer_groups):
            if lg in self._recurrent_groups:
                entries.layer_group_indices.append(lg)
                entries.block_ordinals.append(_STATE_ROW)
                entries.device_pages.append(_frozen_int64((self._state_slot(kv, request_id, lg),)))
                continue
            block_ordinals = (
                self._range_ordinals(
                    request_id,
                    lg,
                    _ranges._needed_ranges(
                        manager, layout, lg, starts[lg] // tpb, -(-end // tpb), end
                    ),
                )
                if ordinals is None
                else ordinals[lg]
            )
            if named and not self._named_groups[lg]:
                block_ordinals = block_ordinals[:0]
            if size > 1:
                # This rank's page of block ``b`` holds tokens from ``b * tpb + rank * page``.
                page = tpb // size
                block_ordinals = _frozen_array(
                    block_ordinals[block_ordinals * tpb + rank * page < end]
                )
            if not len(block_ordinals):
                slots = _NO_ROWS
            else:
                pages = _manager.pages(kv, lg)
                try:
                    gathered = pages[block_ordinals]
                except IndexError:
                    # Blocks past the cache's end have no page.
                    gathered = np.full(len(block_ordinals), -1, dtype=np.int64)
                    inside = block_ordinals < len(pages)
                    gathered[inside] = pages[block_ordinals[inside]]
                slots = _frozen_array(gathered)
            entries.layer_group_indices.append(lg)
            entries.block_ordinals.append(block_ordinals)
            entries.device_pages.append(slots)
        return entries

    @staticmethod
    def _pageless(entries: _LentBlockEntries) -> str | None:
        """The first layer group's blocks of ``entries`` that have no page, as a failure, or
        ``None``."""
        for lg, ordinals, slots in zip(
            entries.layer_group_indices, entries.block_ordinals, entries.device_pages
        ):
            if len(slots) and slots.min() < 0:
                return f"layer group {lg}: blocks {ordinals[slots < 0].tolist()[:8]} have no page"
        return None

    def _lendable_block_entries(
        self, manager: KVCacheManagerV2, history: int, entries: _LentBlockEntries
    ) -> _LentBlockEntries:
        """``entries`` without blocks that have no page or that the request's own window has passed
        (their page may hold something else)."""
        out = _LentBlockEntries([], [], [])
        for lg, block_ordinals, slots in zip(
            entries.layer_group_indices, entries.block_ordinals, entries.device_pages
        ):
            stale_beg, stale_end = _ranges._stale_range(manager, self._layout, lg, history)
            if stale_end <= stale_beg:
                # Nothing behind a window: only blocks without a page drop out.
                whole = not len(slots) or slots.min() >= 0
                ok = None if whole else slots >= 0
            else:
                ok = (slots >= 0) & ~((block_ordinals >= stale_beg) & (block_ordinals < stale_end))
                whole = bool(ok.all())
            out.layer_group_indices.append(lg)
            out.block_ordinals.append(
                block_ordinals if whole else _frozen_array(block_ordinals[ok])
            )
            out.device_pages.append(slots if whole else _frozen_array(slots[ok]))
        return out

    def _reuse_keys_for(
        self,
        manager: KVCacheManagerV2,
        request: LlmRequest,
        kv: _KVCache,
        block_ordinals: Sequence[np.ndarray],
    ) -> list[np.ndarray]:
        """Per layer group, its block entries' reuse keys, ``uint8 (n, 32)``."""
        # TODO: every lease hashes the request's whole prefix again from block 0, so a lease late in
        # a long prompt costs time in proportion to the prompt.
        top = max((int(o.max()) + 1 for o in block_ordinals if len(o)), default=0)
        reuse_keys = _manager.block_keys(manager, request, kv, top)
        columns = [
            b"".join(reuse_keys[int(o)] for o in lg_ordinals) for lg_ordinals in block_ordinals
        ]
        return [np.frombuffer(column, dtype=np.uint8).reshape(-1, 32) for column in columns]

    def _assign_staging_slots(self, entries: _LentBlockEntries, runs: SlotRuns | None) -> None:
        """Each pool group's run goes to its layer groups in order, so the entries of one pool group
        occupy consecutive slots."""
        cursor: dict[int, int] = {}
        entries.staging_slots = []
        for lg, block_ordinals in zip(entries.layer_group_indices, entries.block_ordinals):
            g = int(self._layout.pool_group_of[lg])
            start = runs.by_pool_group.get(g, (0, 0))[0] if runs is not None else 0
            offset = cursor.get(g, 0)
            entries.staging_slots.append(
                np.arange(start + offset, start + offset + len(block_ordinals), dtype=np.int64)
            )
            cursor[g] = offset + len(block_ordinals)

    def _build_view(
        self, entries: _LentBlockEntries, reuse_keys: list[np.ndarray], start: int, end: int
    ) -> LeaseView:
        """Each entry as its staging slot, with its object key; every array a snapshot and an
        object of the view's own; the view of range ``[start, end)``."""
        groups = []
        for lg, block_ordinals, lg_keys, slots in zip(
            entries.layer_group_indices, entries.block_ordinals, reuse_keys, entries.staging_slots
        ):
            groups.append(
                LentLayerGroupRows._unchecked(
                    int(lg),
                    _unshared(block_ordinals),
                    _immutable(self._object_keys(lg, lg_keys)),
                    slots=_frozen_array(slots),
                    region_indices=self._staging_region_indices[
                        int(self._layout.pool_group_of[lg])
                    ],
                )
            )
        return LeaseView._unchecked(tuple(groups), start, end)

    def _failed_at_call(
        self, memory_kind: MemoryKind, kind: str, request_id: int, reason: str
    ) -> _StagingLease | _InPlaceLease:
        """A lease failed at the call, of the class leases of ``memory_kind`` have."""
        if memory_kind is MemoryKind.DEVICE:
            return _InPlaceLease._failed_at_call(self, kind, request_id, reason)
        return _StagingLease._failed_at_call(self, kind, request_id, reason)

    def _slot_limit_refusal(self, entries: _LentBlockEntries) -> str | None:
        """The first group's blocks whose page reaches its device slot limit (0 without device
        regions), as a failure, or ``None``; compares each group's largest page once."""
        for lg, ordinals, pages in zip(
            entries.layer_group_indices, entries.block_ordinals, entries.device_pages
        ):
            limit = self._device_slot_limit.get(lg, 0)
            if len(pages) and pages.max() >= limit:
                return (
                    f"layer group {lg}: blocks {ordinals[pages >= limit].tolist()[:8]} have pages "
                    f"past the {limit} slots of its device regions"
                )
        return None

    def _lend_in_place(
        self,
        manager: KVCacheManagerV2,
        request: LlmRequest,
        kv: _KVCache,
        kind: str,
        entries: _LentBlockEntries,
        reuse_keys: list[np.ndarray] | None,
        fetch: _FetchRecord | None = None,
        doomed: str | None = None,
        *,
        start: int,
        end: int,
        checked: CheckedBlocks | None = None,
        ready_event: torch.cuda.Event | None = None,
    ) -> _InPlaceLease:
        """Lock current pages; the lease is ready once ``ready_event`` and, while the cache is on
        the allocation side stream, the manager's fence completed, a read once pending connector
        loads finished. Refused locks, missing pages or pages past the slot limit fail the lease and
        abandon a write's fetch. ``checked``: this lend's check of exactly these blocks."""
        request_id = int(request.py_request_id)
        lease = _InPlaceLease(self, kind, request_id, kv, entries, fetch)
        if doomed is None:
            doomed = self._slot_limit_refusal(entries)
        if doomed is not None and fetch is not None:
            lease._defer_failure(doomed)
            return lease
        if doomed is None:
            lease._view = self._in_place_view(entries, reuse_keys, start, end)
            if kind == "read":
                mode = LockMode.READ
                if _manager.connector_load_pending(manager, request):
                    lease._loading = request
            else:
                mode = LockMode.WRITE
            blocks = (
                dict(zip(entries.layer_group_indices, entries.block_ordinals))
                if checked is None
                else checked.blocks
            )
            fence = _manager.allocation_fence(manager, request_id)
            ready = ready_event if fence is None else _fenced(ready_event, fence)
            try:
                lease._lock = self._locks.lock(
                    kv, blocks, mode, ready=ready, ordered=True, checked=checked
                )
            except ValueError as error:
                doomed = f"the blocks cannot be locked: {error}"
        if doomed is not None:
            if fetch is not None:
                lease._defer_failure(doomed)
            else:
                self._fail_in_place(lease, doomed)
        return lease

    def _in_place_view(
        self, entries: _LentBlockEntries, reuse_keys: list[np.ndarray] | None, start: int, end: int
    ) -> LeaseView:
        """The view of device-page entries as snapshots of its own, with optional object keys; under
        Helix, ordinal ``b * size + rank`` names this rank's page."""
        rank, size = getattr(self._layout, "cp", (0, 1))
        groups = []
        for i, (lg, block_ordinals, pages) in enumerate(
            zip(entries.layer_group_indices, entries.block_ordinals, entries.device_pages)
        ):
            group = int(self._layout.pool_group_of[lg])
            groups.append(
                LentLayerGroupRows._unchecked(
                    int(lg),
                    _frozen_array(block_ordinals * size + rank)
                    if size > 1
                    else _unshared(block_ordinals),
                    None
                    if reuse_keys is None
                    else _immutable(self._object_keys(lg, reuse_keys[i])),
                    slots=_unshared(pages),
                    region_indices=self._device_regions_of_pool_group[group],
                )
            )
        return LeaseView._unchecked(tuple(groups), start, end)

    # Fast lends of every row in place: what the rules give where they lend every block and refuse
    # nothing, or ``None``, the rules then lending as they would have.

    def _lends_fast(self) -> bool:
        """Whether this lender and ledger have a layout covered by the fast path."""
        return (
            self._fast_layout
            and type(self) is _ManagerLender
            and type(self._ledger) is _FetchLedger
        )

    def _read_every_row_fast(
        self,
        manager: KVCacheManagerV2,
        request: LlmRequest,
        request_id: int,
        start: int,
        end: int,
        ready_event: torch.cuda.Event | None,
    ) -> _InPlaceLease | None:
        """``lend_read``'s lease of every row in place for an active cache that no fetch record
        names, holds the range and a page for each of its blocks, and with a window has its context
        computed; ``None`` elsewhere, having changed nothing the rules read."""
        kv = _manager.kv_of(manager, request_id)
        if kv is None or self._ledger._has_records(request_id):
            return None
        state = _manager.cache_state(kv)
        history = state.history
        if (
            not state.active
            or self._read_end_refusal(manager, request, state, end, False) is not None
        ):
            return None
        # A computed range past history can be valid under the general rules, but has no pages in
        # this cache that the fast path can lend.
        if end > history or self._mid_context_end(manager, request, state) is not None:
            return None
        reserved, loading = _manager.connector_state(manager, request)
        if reserved:
            return None
        # A window keeps at the cache's history, at or past the range's end, what a read lends.
        rows = self._every_row(
            kv, request_id, start, end, kept_at=history, row=self._page_row(manager, request_id, kv)
        )
        if rows is None:
            return None
        entries, view, blocks = rows
        lease = _InPlaceLease(self, "read", request_id, kv, entries)
        lease._view = view
        if loading:
            lease._loading = request
        locks = self._locks
        fence = _manager.allocation_fence(manager, request_id)
        ready = ready_event if fence is None else _fenced(ready_event, fence)
        lock = locks.lock_fresh(kv, blocks, LockMode.READ, ready)
        if lock is None:
            try:
                lock = locks.lock(kv, blocks, LockMode.READ, ready=ready, ordered=True)
            except ValueError as error:
                self._fail_in_place(lease, f"the blocks cannot be locked: {error}")
                return lease
        lease._lock = lock
        return lease

    def _write_every_row_fast(
        self,
        manager: KVCacheManagerV2,
        request: LlmRequest,
        request_id: int,
        start: int,
        end: int,
        prompt: int,
        ready_event: torch.cuda.Event | None,
    ) -> _InPlaceLease | None:
        """``lend_write``'s lease of every row in place for an active cache without a fetch to
        settle or SWA scratch reuse, a range its prompt and history allow, and a page for each block
        it lends; ``None`` elsewhere, having changed nothing the rules read."""
        kv = _manager.kv_of(manager, request_id)
        if kv is None:
            return None
        state = _manager.cache_state(kv)
        if (
            not state.active
            or self._write_range_refusal(manager, request, state, start, end, prompt) is not None
        ):
            return None
        if self._history_refusal(manager, state.history, state.history, end) is not None:
            return None
        if self._any_window and (
            self._unwritten_pages_refusal(manager, request_id, kv, state.history) is not None
        ):
            return None
        if self._bidirectional_split_refusal(request, end) is not None:
            return None
        runs = _manager.bidirectional_runs(request)
        if self._scratch_reuse_on(kv):
            return None
        count = len(self._group_regions)
        committed = (0,) * count if not state.committed else _manager.committed_by_group(kv, count)
        # The pages from the cache's row, which it keeps current itself.
        row = self._page_row(manager, request_id, kv)
        rows = self._every_row(kv, request_id, start, end, committed, kept_at=end, row=row)
        if rows is None:
            return None
        entries, view, blocks = rows
        # The rows start past each group's committed blocks, so only another lock, a hand-over at a
        # free, or with a window the other locks' history cap can refuse them.
        checked = None
        if self._locks.is_locked(kv):
            checked = self._locks.check(
                kv, blocks, LockMode.WRITE, ordered=True, committed=committed
            )
            if isinstance(checked, str):
                return None
            if self._any_window:
                cap = self._history_cap(kv)
                if cap is not None and state.history > cap:
                    return None
        # Its whole return resumes at ``end`` where no record of the request's could lower the
        # bounds and the rows below ``start`` are computed.
        at_end = start <= state.history and not self._ledger._has_records(request_id)
        # Last, as it records the fetch where no earlier fetch into the cache is unsettled.
        fetch = self._ledger._record_next_fetch(
            request_id,
            kv,
            start,
            state.history,
            committed,
            end if self._recurrent_groups else None,
            end,
        )
        if fetch is None:
            return None
        if checked is not None:
            # Only a locked cache has a commit held back, which would commit computed rows this
            # fetch may overwrite.
            _manager.drop_held_commit(manager, kv, start)
        fetch.entries = entries
        fetch.prompt = prompt
        fetch.runs = runs
        lease = _InPlaceLease(self, "write", request_id, kv, entries, fetch)
        lease._view = view
        lease._shrinks = self._shrink_epochs.get(request_id, 0)
        lease._block_end = entries.block_end
        locks = self._locks
        fence = _manager.allocation_fence(manager, request_id)
        ready = ready_event if fence is None else _fenced(ready_event, fence)
        lock = locks.lock_fresh(kv, blocks, LockMode.WRITE, ready) if checked is None else None
        if lock is None:
            try:
                lock = locks.lock(
                    kv,
                    blocks,
                    LockMode.WRITE,
                    ready=ready,
                    ordered=True,
                    checked=checked,
                    committed=committed,
                )
            except ValueError as error:
                lease._defer_failure(f"the blocks cannot be locked: {error}")
                return lease
        lease._lock = lock
        if at_end:
            lease._at_end = True
            self._writes_at_end[id(kv)] = lease
        return lease

    def _every_row(
        self,
        kv: _KVCache,
        request_id: int,
        start: int,
        end: int,
        committed: Sequence[int] | None = None,
        *,
        kept_at: int | None = None,
        row: tuple[memoryview, int, int, int] | None = None,
    ) -> tuple[_LentBlockEntries, LeaseView, dict[int, np.ndarray]] | None:
        """``(entries, view, blocks to lock)`` of ``[start, end)``, given no Helix and sinkless
        formula windows: windows from what ``kept_at`` keeps, writes from ``committed`` whole blocks
        (none past ``end``), a state row per recurrent group, pages from ``row`` or the cache, view
        arrays viewing the others'; ``None`` if a page is absent or not below its limit."""
        tpb = self._tokens_per_block
        last = -(-end // tpb)
        listed = None
        if row is not None:
            words, row_at, group_words, row_blocks = row
            if last > row_blocks:
                return None
        layer_groups, block_ordinals, device_pages, groups, spans = [], [], [], [], []
        blocks = {}
        block_end = 0
        for lg, regions, limit, recurrent, window in self._fast_groups:
            if recurrent:
                slot = self._state_slot(kv, request_id, lg)
                if not 0 <= slot < limit:
                    return None
                ordinals, slots = _STATE_ROW, self._state_rows.get(slot)
                if slots is None:
                    slots = self._state_rows[slot] = _frozen_int64((slot,))
                spans.append(None)
            else:
                first = start // tpb
                if committed is not None:
                    tokens = committed[lg]
                    first = last if tokens > end else max(first, tokens // tpb)
                if window is not None:
                    # The window formula without sinks: blocks before (history + 1 - window) // tpb
                    # are behind the window.
                    behind = min(kept_at, _manager._MAX_TOKENS - tpb) + 1 - window
                    if behind > 0:
                        first = max(first, behind // tpb)
                spans.append((first, last))
                if last <= first:
                    ordinals = slots = _NO_ROWS
                else:
                    block_end = last
                    if last > _ORDINAL_BLOCKS:
                        ordinals = self._fresh_ordinals(request_id, lg, first, last)
                    else:
                        ordinals = _ordinal_views.get((first, last))
                        if ordinals is None:
                            if len(_ordinal_views) >= _ORDINAL_VIEWS_KEPT:
                                _ordinal_views.clear()
                            ordinals = _ordinal_views[first, last] = _ORDINALS[first:last]
                    if row is not None:
                        at = row_at + lg * group_words + first
                        if last - first <= _LISTED_BLOCKS:
                            held = words[at : at + last - first].tolist()
                            if max(held) >= limit:
                                return None
                            try:
                                # ``_frozen_pages``: the pages as int64 on immutable bytes.
                                slots = _from_buffer(_PAGE_PACKERS[len(held)].pack(*held), _INT64)
                            except struct.error:
                                return None
                        else:
                            held = np.frombuffer(words, _INT32, last - first, 4 * at)
                            # As unsigned, a page of -1 is past every limit too.
                            if held.view(_UINT32).max() >= limit:
                                return None
                            slots = _frozen_array(held)
                    else:
                        if listed is None:
                            listed = _manager.num_blocks(kv) <= _LISTED_BLOCKS
                        if listed:
                            pages = _manager.page_list(kv, lg)
                            if last > len(pages):
                                return None
                            held = pages[first:last]
                            if max(held) >= limit:
                                return None
                            try:
                                slots = _frozen_pages(held)
                            except struct.error:
                                return None
                        else:
                            pages = _manager.pages(kv, lg)
                            if last > len(pages):
                                return None
                            slots = _frozen_array(pages[first:last])
                            if slots.view(_UINT64).max() >= limit:
                                return None
            layer_groups.append(lg)
            block_ordinals.append(ordinals)
            device_pages.append(slots)
            blocks[lg] = ordinals
            groups.append(_lent_rows(lg, ordinals.view(), slots.view(), regions))
        entries = _LentBlockEntries(
            layer_groups, block_ordinals, device_pages, spans=spans, block_end=block_end
        )
        return entries, _new_view(tuple(groups), start, end), blocks

    def _fresh_ordinals(self, request_id: int, lg: int, first: int, last: int) -> np.ndarray:
        """Blocks ``[first, last)`` past ``_ORDINALS`` as frozen ordinals: from the request's
        templates if any, else on immutable bytes of the range's size."""
        if request_id in self._ordinal_ranges:
            return self._range_ordinals(request_id, lg, [(first, last)])
        return _frozen_array(np.arange(first, last, dtype=np.int64))

    def _fail_in_place(self, lease: _InPlaceLease, reason: str) -> None:
        """Fail an in-place lease never seen ready: its lock ends, and a write abandons its
        fetch."""
        if lease.failure is not None:
            return
        lease._set_failure(reason)
        if lease._lock is not None:
            self._locks.unlock(lease._lock)
        if lease._kind == "write":
            self._ledger._abandon_fetch(lease)

    def _apply_in_place_marks(
        self, lease: _InPlaceLease, masks: list[np.ndarray], lock_held: bool = False
    ) -> None:
        """Invalidate all lent entries (writes may touch any row), then count arrived ones whose
        pages still match (all with ``lock_held``); a lent, undelivered committed tail taints the
        cache."""
        fetch = lease._fetch
        current = self._ledger._take_out_fetch(lease, fetch)
        kv = _manager.kv_of(self._manager_ref(), lease._request_id)
        entries = lease._block_entries
        if lock_held:
            delivered = masks
        else:
            lent = self._locked_block_masks(kv, lease)
            delivered = [mask & held for mask, held in zip(masks, lent)]
        self._ledger._invalidate_delivered_blocks(lease, None)
        if kv is lease._kv and self._covers_committed_tail(kv, entries, delivered, complement=True):
            self._ledger._record_mixed_tail(lease._request_id, kv)
        if current:
            self._ledger._put_back_fetch(lease, fetch, delivered)

    def _on_in_place_released(self, lease: _InPlaceLease) -> None:
        """The holder let go: the lock ends; a write released after its view but before its return
        delivers no rows, and one released before its view, which no backend wrote through, abandons
        its fetch."""
        if lease._at_end:
            lease._at_end = False
            self._writes_at_end.pop(id(lease._kv), None)
        try:
            if lease._kind == "write" and lease._view_returned and not lease._marked:
                lease._marked = True
                manager = self._serving_manager()
                if manager is not None and lease._fetch is not None:
                    _manager.drop_held_commit(manager, lease._kv, lease._fetch.start)
                if manager is not None and self._stop_reason is None:
                    self._apply_in_place_marks(lease, self._no_row_mask(lease))
        finally:
            if lease._lock is not None:
                self._locks.unlock(lease._lock)
        if (
            lease._kind == "write"
            and not lease._view_returned
            and self._serving_manager() is not None
        ):
            self._ledger._abandon_fetch(lease)

    def _copy_segments(
        self, entries: _LentBlockEntries, mask: list[np.ndarray] | None = None
    ) -> list[list[int]]:
        """``[staging address, device address, bytes]`` per pool of each row ``mask`` keeps
        (``None``: all), by pool group, merging segments continuous on both sides."""
        by_group: dict[int, list[int]] = {}
        for i, lg in enumerate(entries.layer_group_indices):
            by_group.setdefault(int(self._layout.pool_group_of[lg]), []).append(i)
        out: list[list[int]] = []
        for g, members in by_group.items():
            dev = np.concatenate([entries.device_pages[i] for i in members])
            stg = np.concatenate([entries.staging_slots[i] for i in members])
            if mask is not None:
                keep = np.concatenate([np.asarray(mask[i], dtype=bool) for i in members])
                dev, stg = dev[keep], stg[keep]
            region = self._regions[self._region_index_of_pool_group[g]]
            # A staging slot holds the device pools' slots back to back in pool order.
            offset = 0
            for pool in self._layout.device_pools[g]:
                width = int(pool.slot_bytes)
                for d, t in zip(dev.tolist(), stg.tolist()):
                    host = region.address + t * region.slot_bytes + offset
                    device = int(pool.base) + d * width
                    last = out[-1] if out else None
                    if last and last[0] + last[2] == host and last[1] + last[2] == device:
                        last[2] += width
                    else:
                        out.append([host, device, width])
                offset += width
        return out

    # TODO: copies run serially with forward passes on the execution stream, and one call per row
    # and pool keeps small or multi-pool entries below the host-to-device bandwidth.
    def _queue_copy(
        self,
        segments: Sequence[Sequence[int]],
        to_staging: bool,
        after: torch.cuda.Event | None = None,
    ) -> tuple[_QueuedCopy | None, str | None]:
        """Queue async copies on the manager's execution stream after ``after``, the lease's ready
        event, then record one event, the lender's record of them, covering every copy queued even
        after a failure; ``(None, error)`` without one. Pageable staging may hold the call."""
        # The lender does not rely on a release waiting for the copies; a writer off this stream is
        # not ordered after them.
        stream = _manager.stream(self._manager_ref())
        if after is not None:
            stream.wait_event(after)
        handle = drv.CUstream(stream.cuda_stream)
        error = None
        for host, device, nbytes in segments:
            dst, src = (host, device) if to_staging else (device, host)
            (result,) = drv.cuMemcpyAsync(
                drv.CUdeviceptr(dst), drv.CUdeviceptr(src), nbytes, handle
            )
            if result != drv.CUresult.CUDA_SUCCESS:
                error = f"cuMemcpyAsync of {nbytes} bytes failed: {result}"
                break
        try:
            event = torch.cuda.Event()
            event.record(stream)
        except RuntimeError as record_error:
            error = error or f"recording the copy's event failed: {record_error}"
            logger.warning(f"KV cache lender: {error}")
            return None, error
        if error is not None:
            logger.warning(f"KV cache lender: {error}")
        return _QueuedCopy(event), error


class _LeaseBase:
    """A lease's failure, whether its return resumes at its end, and the checks of the masks its
    return counts."""

    def __init__(
        self,
        lender: _ManagerLender,
        kind: str,
        request_id: int,
        kv: _KVCache | None,
        entries: _LentBlockEntries | None,
        fetch: _FetchRecord | None,
    ) -> None:
        self._lender = weakref.ref(lender)
        self._kind = kind
        self._request_id = request_id
        self._kv = kv
        self._block_entries = entries
        self._fetch = fetch
        self._view: LeaseView | None = None
        self._view_returned = False
        self._released = False
        self._marked = False
        self._failure: str | None = None
        self._first_poll_failure: str | None = None
        # Set only for a fast write in place; the lender's hooks clear it.
        self._at_end = False

    def _defer_failure(self, reason: str) -> None:
        """Fail on the first poll so a fetched write is abandoned when its caller sees failure."""
        self._first_poll_failure = reason

    def _set_failure(self, reason: str) -> None:
        """Record a terminal failure and remove any reason deferred to the first poll."""
        self._failure = reason
        self._first_poll_failure = None

    @property
    def failure(self) -> str | None:
        """See ``Lease.failure``."""
        return self._failure

    @property
    def writable(self) -> bool:
        """See ``Lease.writable``."""
        return self._kind == "write"

    @property
    def return_resumes_at_end(self) -> bool:
        """See ``Lease.return_resumes_at_end``."""
        return self._at_end and self._view_returned

    def _accepted_masks(self, masks: Sequence[np.ndarray]) -> list[np.ndarray]:
        """``masks`` of a write's arrived rows as its return keeps them, a read-only exact
        ``numpy.ndarray`` as is, others copied; ``ValueError`` unless one bool mask of shape
        ``(len(blocks),)`` per group. Changes nothing."""
        masks, groups = list(masks), self._block_entries.block_ordinals
        if len(masks) != len(groups):
            raise ValueError(f"{len(masks)} masks for {len(groups)} groups")
        out = []
        for lg, blocks, mask in zip(self._block_entries.layer_group_indices, groups, masks):
            if type(mask) is not np.ndarray:
                mask = np.array(mask)
            elif mask.flags.writeable:
                mask = mask.copy()
            if mask.dtype != np.bool_ or mask.shape != (len(blocks),):
                raise ValueError(
                    f"layer group {lg}: the mask must be bool of shape "
                    f"({len(blocks)},), got {mask.dtype} {mask.shape}"
                )
            out.append(mask)
        return out


class _StagingLease(_LeaseBase):
    """Host staging lease; see ``Lease`` for thread and access lifetimes.
    Holds its lender weakly and has no finalizer."""

    def __init__(
        self,
        lender: _ManagerLender,
        kind: str,
        request_id: int,
        kv: _KVCache | None = None,
        entries: _LentBlockEntries | None = None,
        reuse_keys: list[np.ndarray] | None = None,
        fetch: _FetchRecord | None = None,
    ) -> None:
        super().__init__(lender, kind, request_id, kv, entries, fetch)
        self._reuse_keys = reuse_keys
        # The caller's event after which the pages are free of work in flight; copies wait for it.
        self._ready_event: torch.cuda.Event | None = None
        # The lend's token range, which its view gives.
        self._range = (0, 0)
        self._ticket: int | None = None  # its place in line while it waits for slots
        # The granted slots; _recycle returns them and leaves this field set.
        self._slot_runs: SlotRuns | None = None
        self._copy: _QueuedCopy | None = None

    @classmethod
    def _failed_at_call(
        cls, lender: _ManagerLender, kind: str, request_id: int, reason: str
    ) -> _StagingLease:
        """A lease failed at the call; open until released, like any other."""
        lease = cls(lender, kind, request_id)
        lease._set_failure(reason)
        if not lender._shut_down:
            lender._open_leases.add(lease)
        return lease

    def _marks_due(self) -> bool:
        """A write whose view a poll returned and that was not returned yet: its return may still
        queue a copy reading the lease's slots."""
        return self._kind == "write" and self._view_returned and not self._marked

    def __repr__(self) -> str:
        if self._released:
            state = "released"
        elif self._failure is not None:
            state = "failed"
        elif self._first_poll_failure is not None:
            state = "fails_at_first_poll"
        elif self._ticket is not None:
            state = "waiting"
        elif self._marked:
            state = "marked"
        else:
            state = "ready" if self._view_returned else "granted"
        return f"<staging {self._kind} lease of request {self._request_id}: {state}>"

    def poll(self) -> LeaseView | None:
        """See ``Lease.poll``."""
        if self._released:
            raise RuntimeError("poll after release")
        if self._kind == "read" and self._copy is _COPY_WITHOUT_RECORD and self._failure is None:
            # A grant that raised left its copy unrecorded and the lease not failed.
            self._set_failure("the copy into staging has no record of its completion")
        lender = self._lender()
        if lender is not None and lender._stop_reason is not None and not self._view_returned:
            # A stopped lender returns only views a poll returned before, also after the shutdown.
            lender._fail(self, lender._stop_reason)
            return None
        if lender is None or lender._serving_manager() is None:
            return self._poll_not_serving()
        if self._failure is not None:
            return None
        if self._first_poll_failure is not None:
            lender._fail(self, self._first_poll_failure)
            return None
        if self._view is None:
            return None
        if self._kind == "read" and not lender._read_copy_done(self):
            return None
        self._view_returned = True
        return self._view

    def release(self) -> None:
        """See ``Lease.release``."""
        if self._released:
            return
        self._released = True
        lender = self._lender()
        if lender is not None:
            lender._on_lease_released(self)

    def _poll_not_serving(self) -> LeaseView | None:
        """``poll`` after the lender stopped serving: no grant or copy, just what already landed."""
        if self._failure is not None or self._view is None:
            return None
        if self._kind == "read" and self._copy is not None and not self._copy.done():
            return None
        self._view_returned = True
        return self._view


class _InPlaceLease(_LeaseBase):
    """Locked request pages; poll waits for the lock event and pending connector loads.
    See ``Lease`` for thread and access lifetimes. Holds the lender weakly, without a finalizer."""

    def __init__(
        self,
        lender: _ManagerLender,
        kind: str,
        request_id: int,
        kv: _KVCache | None = None,
        entries: _LentBlockEntries | None = None,
        fetch: _FetchRecord | None = None,
    ) -> None:
        super().__init__(lender, kind, request_id, kv, entries, fetch)
        self._lock: PageLock | None = None
        # The fetch ledger reads a lease's copy of its marks; marks in place copy nothing.
        self._copy = None
        # The request a KV cache connector was loading into at the lend, until it has finished.
        self._loading: LlmRequest | None = None
        # For a fast write: the request's shrink count at the lend, and the end of its lent blocks.
        self._shrinks = 0
        self._block_end = 0

    @classmethod
    def _failed_at_call(
        cls, lender: _ManagerLender, kind: str, request_id: int, reason: str
    ) -> _InPlaceLease:
        """A lease failed at the call, with no lock and no fetch."""
        lease = cls(lender, kind, request_id)
        lease._set_failure(reason)
        return lease

    def _finish_returned_write(self) -> None:
        """Close a returned write's view while its lock remains held until the lender unlocks."""
        self._marked = True
        self._released = True
        self._at_end = False

    def __repr__(self) -> str:
        state = "released" if self._released else "held" if self._failure is None else "failed"
        return f"<in-place {self._kind} lease of request {self._request_id}: {state}>"

    def poll(self) -> LeaseView | None:
        """See ``Lease.poll``."""
        if self._released:
            raise RuntimeError("poll after release")
        if self._failure is not None:
            return None
        lender = self._lender()
        if lender is not None and lender._stop_reason is not None and not self._view_returned:
            lender._fail_in_place(self, lender._stop_reason)
            return None
        if self._first_poll_failure is not None:
            if lender is not None:
                lender._fail_in_place(self, self._first_poll_failure)
            else:
                self._set_failure(self._first_poll_failure)
            return None
        if not self._view_returned and self._loading is not None:
            manager = lender._serving_manager() if lender is not None else None
            if manager is not None and _manager.connector_load_pending(manager, self._loading):
                return None
            self._loading = None
        if not self._view_returned and not self._lock.is_ready():
            return None
        self._view_returned = True
        return self._view

    def release(self) -> None:
        """See ``Lease.release``."""
        if self._released:
            return
        self._released = True
        lender = self._lender()
        if lender is not None:
            lender._on_in_place_released(self)


class _RegionHold:
    """A hold on the lender's regions: its staging memory and the runtime with its device pools.
    Holds its lender weakly and has no finalizer; without a lender it is inert."""

    def __init__(self, lender: _ManagerLender | None) -> None:
        self._lender = weakref.ref(lender) if lender is not None else None
        self._released = False

    def release(self) -> None:
        """See ``RegionHold.release``."""
        if self._released:
            return
        self._released = True
        lender = self._lender() if self._lender is not None else None
        if lender is not None:
            lender._on_hold_released(self)
