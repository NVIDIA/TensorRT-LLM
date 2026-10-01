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
"""The staging and in-place lenders, their leases, and the hooks the manager calls on free and
shutdown. Like the manager, one thread at a time uses them: the builder, the executor loop, then
the shutdown thread. There are no locks and no background threads."""

from __future__ import annotations

import collections
import traceback
import weakref
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np
import torch
from cuda.bindings import driver as drv
from cuda.bindings import runtime as cudart

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger

from . import _manager
from ._identity import Identity
from ._layout import ManagerLayout, derive_layout, layout_id
from ._slots import Runs, Slots, slot_counts
from ._types import GroupRun, Part, Readiness, RegionView, StagingOptions

if TYPE_CHECKING:
    from ...llm_request import LlmRequest

# Staging memory, caches and page-index buffers kept until the process exits; only a clean shutdown
# or a last loan's end removes an entry. One thread at a time changes it, so it takes no lock.
_KEPT: List[object] = []

_SHUT_DOWN = "the KV cache manager shut down"
_SUSPENDED = "the request's cache is suspended"
_SCRATCH = "the request's cache has SWA scratch reuse on, so its window blocks may not keep a fetch"


def _retained() -> Tuple[object, ...]:
    """What is kept until exit, oldest first; for tests."""
    return tuple(_KEPT)


def _let_go(owner: object) -> None:
    """Drop ``owner`` from the keep list, compared by identity."""
    for index in range(len(_KEPT) - 1, -1, -1):
        if _KEPT[index] is owner:
            del _KEPT[index]
            return


class _HostMemory:
    """Host memory of exactly ``nbytes``, page-locked where pinning pays off. Only ``free`` releases
    it, once; nothing frees it on collection, so memory kept until exit stays mapped."""

    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes
        self._pageable: Optional[np.ndarray] = None
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


def _allocate(nbytes: int) -> _HostMemory:
    """One host allocation of ``nbytes`` for the staging parts, appended to the keep list."""
    # TODO: relay through the KV cache manager's own host tier once its host pools keep fixed,
    # registrable addresses (resizing with mremap moves them); keep this staging for managers
    # without a host tier.
    memory = _HostMemory(max(int(nbytes), 1))
    _KEPT.append(memory)
    return memory


def _context_parallel_size(manager) -> int:
    """The manager's context-parallel size; ``TypeError`` for a manager without a mapping."""
    mapping = getattr(manager, "mapping", None)
    if mapping is None:
        raise TypeError("the KV cache lender needs a manager with a mapping")
    return int(mapping.cp_size)


def _checked_layout(manager) -> ManagerLayout:
    """The checks both attaches share: a v2 manager, no context parallelism, no lender yet, no
    recurrent state."""
    _manager.require_v2(manager)
    cp_size = _context_parallel_size(manager)
    if cp_size > 1:
        # The layout derives only tensor-parallel shards: context-parallel ranks would name their
        # blocks alike while holding different pages.
        raise ValueError(
            f"context parallelism (cp_size={cp_size}) is not supported: its ranks would give "
            "different pages the same names"
        )
    # TODO: one lender per manager for now, so staging and in-place never attach together; a later
    # internal owner could serve both views.
    if _manager.attached(manager) is not None:
        raise ValueError("a lender is already attached to this KV cache manager")
    layout = derive_layout(manager)
    recurrent = [lg for lg, state in enumerate(layout.recurrent) if state]
    if recurrent:
        raise ValueError(
            f"layer groups {recurrent} hold recurrent state, which the lender does not lend"
        )
    return layout


def _check_commits(manager) -> None:
    """``ValueError`` for a manager that commits no blocks: a publish lends only committed ones."""
    if not _manager.commits_blocks(manager):
        raise ValueError(
            "staging needs a manager that commits blocks: block reuse on, and joint reuse for a "
            "draft manager; with block reuse off nothing could ever be published"
        )


def attach_staging(manager, *, scope: bytes, staging: StagingOptions) -> Staging:
    """The public ``attach_staging``: a ``Staging`` lender installed on ``manager``."""
    return _attach_staging(manager, scope=scope, staging=staging)


def _attach_staging(
    manager, *, scope: bytes, staging: StagingOptions, cls: Optional[type] = None
) -> Staging:
    """``attach_staging`` with the lender class as a parameter (``Staging`` when ``None``), so a
    test can attach a subclass that breaks one rule."""
    layout = _checked_layout(manager)
    _check_commits(manager)
    if not isinstance(scope, bytes):
        raise TypeError(f"scope must be bytes, got {type(scope).__name__}")
    if not isinstance(staging, StagingOptions):
        raise TypeError(f"staging must be StagingOptions, got {type(staging).__name__}")
    counts = slot_counts(layout, staging)
    identity = Identity(scope, layout_id(layout.layout), layout.layers, layout.shards)
    slots = {g: int(counts.get(g, 0)) for g in layout.pool_groups}
    served: Dict[int, List[int]] = {g: [] for g in layout.pool_groups}
    for lg, g in enumerate(layout.pool_group_of):
        served[int(g)].append(lg)
    sizes = {g: slots[g] * int(layout.page_bytes[g]) for g in layout.pool_groups}
    memory = _allocate(sum(sizes.values()))
    base = memory.address
    parts = []
    offset = 0
    for g in layout.pool_groups:
        name = identity.part_name(served[g])
        parts.append(Part(name, base + offset, sizes[g], int(layout.page_bytes[g]), slots[g]))
        offset += sizes[g]
    lender = (cls or Staging)(
        weakref.ref(manager), layout, identity, tuple(parts), Slots(slots), weakref.ref(memory)
    )
    lender._keep_index_buffer(manager)
    # Installed last, so a failure above leaves nothing attached.
    _manager.install(manager, lender)
    logger.info(
        f"KV cache lender: namespace {identity.namespace.hex()}, staging {offset >> 20} MiB "
        f"in {len(parts)} parts"
    )
    return lender


def attach_in_place(manager) -> InPlace:
    """The public ``attach_in_place``: an ``InPlace`` lender installed on ``manager``."""
    layout = _checked_layout(manager)
    lender = InPlace(weakref.ref(manager), layout)
    _manager.install(manager, lender)
    return lender


class _Copy:
    """Copies queued together, complete once their event says so. ``done`` asks the event at most
    once per round of the lender's progress, and never again once it reported completion."""

    def __init__(self, event: torch.cuda.Event) -> None:
        self._event = event
        self._done = False
        self._round: Optional[int] = None

    def done(self, round_: Optional[int] = None) -> bool:
        """Whether the copies have completed, without waiting; ``round_=None`` always asks."""
        if not self._done and (round_ is None or round_ != self._round):
            self._round = round_
            self._done = bool(self._event.query())
        return self._done

    def wait(self) -> None:
        """Block the host until the copies have completed."""
        if not self._done:
            self._event.synchronize()
            self._done = True


def _key_column(keys: List[bytes]) -> np.ndarray:
    return np.frombuffer(b"".join(keys), dtype=np.uint8).reshape(len(keys), 32)


def _no_cache(request_id: int) -> str:
    return f"request {request_id} has no KV cache"


def _stale(manager, layout: ManagerLayout, lg: int, history: int) -> Tuple[int, int]:
    """Block ordinals ``[beg, end)`` behind layer group ``lg``'s window at ``history``."""
    if layout.windows[lg] is None:
        return 0, 0
    return _manager.stale_blocks(manager, lg, history)


def _needed_ordinals(
    manager, layout: ManagerLayout, lg: int, start_block: int, end_block: int, history: int
) -> np.ndarray:
    """Ordinals of ``lg`` in ``[start_block, end_block)`` that a history of ``history`` tokens
    still reads: all of them for full attention; the sinks and the window otherwise."""
    ordinals = np.arange(start_block, end_block, dtype=np.int64)
    stale_beg, stale_end = _stale(manager, layout, lg, history)
    return ordinals[(ordinals < stale_beg) | (ordinals >= stale_end)]


def _checked_masks(view: RegionView, masks: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Copies of ``masks``; ``ValueError`` unless one bool mask of shape ``(len(run),)`` per run."""
    masks = list(masks)
    if len(masks) != len(view.runs):
        raise ValueError(f"{len(masks)} masks for {len(view.runs)} runs")
    out = []
    for run, mask in zip(view.runs, masks):
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != (len(run),):
            raise ValueError(
                f"layer group {run.layer_group}: the mask must be bool of shape ({len(run)},), "
                f"got {mask.dtype} {mask.shape}"
            )
        out.append(mask.copy())
    return out


@dataclass(eq=False)
class _Rows:
    """Per layer group, in order: ordinals, device slots and their staging slots, aligned."""

    layer_groups: List[int]
    ordinals: List[np.ndarray]
    device_slots: List[np.ndarray]
    staging_slots: List[np.ndarray] = field(default_factory=list)

    @property
    def num_rows(self) -> int:
        return sum(len(o) for o in self.ordinals)


@dataclass(eq=False)
class _Fetch:
    """One write lease's fetch into one cache: delivered once its marked rows' copy is queued
    without error, settled once that copy has completed."""

    start: int
    end: int
    kv: object
    delivered: bool = False
    copy: Optional[_Copy] = None


@dataclass(eq=False)
class _Delivered:
    """What the fetches into one cache delivered: per layer group, by block ordinal, the rows whose
    copy was queued and that no shrink freed since; ``origin`` is the lowest fetch start."""

    kv: object
    origin: int
    blocks: List[np.ndarray]
    usable: Optional[Tuple[int, int]] = None  # (committed tokens, usable_until) last computed


class Staging:
    """``StagingLender`` over one manager, which it references weakly. Every call first returns the
    slots of settled leases and grants waiting ones in order, so progress needs no new traffic."""

    def __init__(
        self,
        manager: weakref.ref,
        layout: ManagerLayout,
        identity: Identity,
        parts: Tuple[Part, ...],
        slots: Slots,
        memory: weakref.ref,
    ) -> None:
        self._manager_ref = manager
        self._layout = layout
        self._identity = identity
        self._parts = tuple(parts)
        self._slots = slots
        # Only the keep list holds the staging memory strongly; this finds it there at shutdown.
        self._memory = memory
        self._part_of_group = {g: i for i, g in enumerate(layout.pool_groups)}
        self._any_window = any(w is not None for w in layout.windows)
        self._fetches: Dict[int, _Fetch] = {}  # the latest fetch into each request
        self._delivered: Dict[int, _Delivered] = {}
        # Requests whose history a fetch moved past the committed tokens: the cache it grew.
        self._advanced: Dict[int, Tuple[object, int]] = {}
        self._line: Deque[_StagingLease] = collections.deque()  # waiting for slots, in order
        self._holding: List[_StagingLease] = []  # granted, slots not yet returned
        # Open leases, held strongly: an open lease keeps the staging memory at shutdown even
        # once its holder has dropped it.
        self._unreleased: Set[_StagingLease] = set()
        # Open parts holds, held strongly too: a dropped hold still keeps the memory.
        self._holds: Set[_PartsHold] = set()
        self._quarantined: List[Runs] = []  # slots a failed copy may still touch; never reused
        self._closed = False
        self._index_buffer: Optional[object] = None  # kept from the attach until the shutdown
        self._round = 0  # rounds of progress, so each asks a copy's event at most once

    @property
    def parts(self) -> Tuple[Part, ...]:
        """See ``StagingLender.parts``."""
        return self._parts

    def hold_parts(self) -> _PartsHold:
        """See ``StagingLender.hold_parts``."""
        if self._live() is None:
            # The memory was freed or kept at the shutdown; a later hold changes neither.
            return _PartsHold(None)
        hold = _PartsHold(self)
        self._holds.add(hold)
        return hold

    def lend_read(self, request: LlmRequest, start: int, end: int) -> _StagingLease:
        """See ``StagingLender.lend_read``."""
        start, end = self._whole_blocks(start, end)
        request_id = int(request.py_request_id)
        manager = self._live()
        if manager is None:
            return _StagingLease._failed(self, "read", request_id, _SHUT_DOWN)
        self._progress()
        kv = _manager.kv_of(manager, request_id)
        if kv is None:
            return _StagingLease._failed(self, "read", request_id, _no_cache(request_id))
        state = _manager.cache_state(kv)
        if end > state.committed:
            raise ValueError(
                f"the range ends at {end}, past the {state.committed} committed tokens"
            )
        if not state.active:
            return _StagingLease._failed(self, "read", request_id, _SUSPENDED)
        rows = self._lendable(manager, state.history, self._rows(manager, kv, start, end))
        counts = self._counts(rows.ordinals)
        self._check_fits(counts)
        keys = self._keys_for(manager, request, kv, rows.ordinals)
        lease = _StagingLease(self, "read", request_id, kv, rows, keys)
        self._open(lease, counts)
        return lease

    def lend_write(self, request: LlmRequest, start: int, end: int) -> _StagingLease:
        """See ``StagingLender.lend_write``."""
        start, end = self._whole_blocks(start, end)
        request_id = int(request.py_request_id)
        manager = self._live()
        if manager is None:
            return _StagingLease._failed(self, "write", request_id, _SHUT_DOWN)
        self._progress()
        layout = self._layout
        tpb = int(layout.tokens_per_block)
        # Rows and their names come from the layout for a history of ``end``, so every
        # ValueError is raised before the cache changes.
        ordinals = [
            _needed_ordinals(manager, layout, lg, start // tpb, end // tpb, end)
            for lg in range(layout.num_layer_groups)
        ]
        counts = self._counts(ordinals)
        self._check_fits(counts)
        kv = _manager.kv_of(manager, request_id)
        if kv is None:
            return _StagingLease._failed(self, "write", request_id, _no_cache(request_id))
        state = _manager.cache_state(kv)
        if start < (state.committed // tpb) * tpb:
            raise ValueError(
                f"the range starts at {start}, inside the committed whole blocks of "
                f"{state.committed} tokens"
            )
        previous = self._fetches.get(request_id)
        if previous is not None and previous.kv is kv and not self._report_settled(previous):
            raise ValueError(f"request {request_id} already has an unsettled fetch")
        keys = self._keys_for(manager, request, kv, ordinals)
        if self._any_window and end < state.history:
            raise ValueError(
                f"the range ends at {end}, below the history of {state.history} tokens its "
                "windows keep"
            )
        if not state.active:
            return _StagingLease._failed(self, "write", request_id, _SUSPENDED)
        if self._scratch_reuse_on(kv):
            return _StagingLease._failed(self, "write", request_id, _SCRATCH)
        # With a window the history moves to ``end``, so windows need pages only for the blocks a
        # history of that length reads.
        position = end if self._any_window else state.history
        if not _manager.grow(manager, request, kv, position, end):
            return _StagingLease._failed(
                self, "write", request_id, f"no free pages to grow the cache to {end} tokens"
            )
        # The cache has grown, and readiness accounts for it whatever this lease's outcome.
        if position > state.committed:
            self._advanced[request_id] = (kv, state.committed)
        fetch = _Fetch(start, end, kv)
        self._fetches[request_id] = fetch
        slots = []
        doomed = None
        for lg, lg_ordinals in enumerate(ordinals):
            pages = _manager.pages(kv, lg)
            lg_slots = np.full(len(lg_ordinals), -1, dtype=np.int64)
            inside = lg_ordinals < len(pages)
            lg_slots[inside] = pages[lg_ordinals[inside]]
            if doomed is None and np.any(lg_slots < 0):
                missing = lg_ordinals[lg_slots < 0].tolist()
                doomed = f"layer group {lg}: blocks {missing[:8]} have no page"
            slots.append(lg_slots)
        rows = _Rows(list(range(layout.num_layer_groups)), ordinals, slots)
        lease = _StagingLease(self, "write", request_id, kv, rows, keys, fetch)
        if doomed is not None:
            # It fails at its first poll, which abandons the fetch.
            lease._doomed = doomed
            self._unreleased.add(lease)
            return lease
        self._open(lease, counts)
        return lease

    def readiness(self, request: LlmRequest) -> Optional[Readiness]:
        """See ``StagingLender.readiness``."""
        request_id = int(request.py_request_id)
        manager = self._live()
        if manager is None:
            raise ValueError(f"{_no_cache(request_id)}: {_SHUT_DOWN}")
        self._progress()
        kv = _manager.kv_of(manager, request_id)
        if kv is None:
            raise ValueError(_no_cache(request_id))
        state = _manager.cache_state(kv)
        committed, history = state.committed, state.history
        # A record of a replaced cache (a restart) is moot.
        fetch = self._fetches.get(request_id)
        if fetch is not None and fetch.kv is not kv:
            del self._fetches[request_id]
            fetch = None
        delivered = self._delivered.get(request_id)
        if delivered is not None and delivered.kv is not kv:
            del self._delivered[request_id]
            delivered = None
        advanced = self._advanced.get(request_id)
        if advanced is not None and advanced[0] is not kv:
            del self._advanced[request_id]
            advanced = None
        if fetch is not None and not self._report_settled(fetch):
            return None
        if delivered is None:
            if advanced is not None:
                # Grown for a fetch that delivered nothing: nothing past committed is computed.
                return Readiness(committed, history)
            return Readiness(max(committed, history), history)
        # A shrink the manager did not report still shows as blocks past the capacity.
        self._void_past(delivered, kv)
        if delivered.usable is None or delivered.usable[0] != committed:
            delivered.usable = (committed, self._usable_until(manager, delivered, committed))
        usable = delivered.usable[1]
        return Readiness(int(usable), int(self._floor(manager, history, delivered.origin)))

    def _on_free(self, request_id: int, kv_cache, after_close: Callable[[], None]) -> bool:
        """Manager hook, after the request's cache left the map: its waiting leases fail and its
        fetch records go. Returns ``False``: the manager closes the cache itself. Never raises."""
        try:
            if self._live() is None:
                return False
            # Granted leases go on: a read's copy is queued already, and a write's marks copy
            # nothing into pages other than the ones lent.
            self._fail_waiting(int(request_id), "the request was freed")
            self._fetches.pop(int(request_id), None)
            self._delivered.pop(int(request_id), None)
            self._advanced.pop(int(request_id), None)
            self._progress()
        except Exception:
            # A hook never raises into the manager's free: the cache is then closed as usual,
            # and granted leases keep their slots.
            logger.error(f"KV cache lender: freeing request {request_id}: {traceback.format_exc()}")
        return False

    def _on_shrink(self, request_id: int, kv_cache) -> None:
        """Manager hook, right after the request's cache may have shrunk in place: delivered rows
        past its blocks lost their pages for good. Never raises."""
        try:
            delivered = self._delivered.get(int(request_id))
            if self._live() is not None and delivered is not None and delivered.kv is kv_cache:
                self._void_past(delivered, kv_cache)
        except Exception:
            # A hook never raises into the manager's resize; readiness voids the rows it sees past
            # the capacity.
            logger.error(
                f"KV cache lender: shrink of request {request_id}: {traceback.format_exc()}"
            )

    def _on_shutdown(self, impl) -> FrozenSet[object]:
        """Manager hook, first in its shutdown, acting once: waits for its copies, fails waiting
        leases, and frees the staging memory unless a lease or a hold is open or a slot was lost to
        a failed copy. Returns ``frozenset()``; never raises."""
        if self._closed:
            return frozenset()
        try:
            # The lender's only host wait: no more work comes on the stream.
            for lease in self._holding:
                if lease._copy is not None:
                    lease._copy.wait()
            self._round += 1
            self._recycle()
            for lease in list(self._line):
                self._fail(lease, _SHUT_DOWN)
            for lease in [lease for lease in self._unreleased if lease._doomed is not None]:
                self._fail(lease, lease._doomed)
            self._closed = True
            self._let_go_index_buffer()
            if not self._memory_in_use():
                # Every copy on it has completed above, and no lease or hold reaches it any more.
                memory = self._memory()
                _let_go(memory)
                memory.free()
            else:
                logger.warning(
                    f"KV cache lender: keeping {sum(p.nbytes for p in self._parts) >> 20} MiB of "
                    f"staging memory until exit: {len(self._unreleased)} leases open, "
                    f"{len(self._holds)} parts holds open, "
                    f"{len(self._quarantined)} slot runs lost to failed copies"
                )
        except Exception:
            # A hook never raises into the manager's shutdown; the staging memory then stays in
            # the keep list until exit.
            self._closed = True
            logger.error(f"KV cache lender: shutting down: {traceback.format_exc()}")
        return frozenset()

    # One rule per method, so a test subclass that breaks exactly one rule overrides one method.

    def _progress(self) -> None:
        """Return the slots of settled leases, then grant waiting leases in order."""
        # Progress happens only here, inside the lender's calls, which the executor loop makes
        # every iteration through polls and readiness; no background thread drives it.
        if self._live() is None:
            return
        self._round += 1
        self._recycle()
        self._grant_waiting()

    def _landed(self, copy: Optional[_Copy]) -> bool:
        """``copy`` has completed (or there is none), its event asked at most once this round."""
        return copy is None or copy.done(self._round)

    def _copy_landed(self, lease: _StagingLease) -> bool:
        """A read lease's copy into its slots has completed."""
        return self._landed(lease._copy)

    def _recyclable(self, lease: _StagingLease) -> bool:
        """The lease's slots may return: no backend access possible and no copy on them pending.
        The copy is asked last, so a lease still lent costs no event query."""
        # A failed lease was never ready, so no backend has seen its slots.
        if lease.failure is None:
            if not lease._released:
                return False
            if not (lease._kind == "read" or lease._marked or not lease._seen_ready):
                return False
        # TODO: every lender call asks each released lease's pending copy again, so N calls while P
        # copies pend cost N*P event queries; copies on one stream complete in order, so asking
        # from the oldest and stopping at the first pending one would cost about one per call.
        return self._landed(lease._copy)

    def _still_lent(self, kv, lease: _StagingLease) -> List[np.ndarray]:
        """Per run, the rows whose block is still in the window of the same active cache and still
        locks the GPU page lent; a page only held may sit on another tier under the same number."""
        rows = lease._rows
        manager = self._manager_ref()
        state = _manager.cache_state(kv) if kv is not None and kv is lease._kv else None
        masks = []
        for lg, ordinals, slots in zip(rows.layer_groups, rows.ordinals, rows.device_slots):
            same = np.zeros(len(ordinals), dtype=bool)
            if state is not None and state.active:
                pages = _manager.locked_pages(kv, lg)
                inside = ordinals < len(pages)
                same[inside] = pages[ordinals[inside]] == slots[inside]
                stale_beg, stale_end = _stale(manager, self._layout, lg, state.history)
                same &= (ordinals < stale_beg) | (ordinals >= stale_end)
            masks.append(same)
        return masks

    def _scratch_reuse_on(self, kv) -> bool:
        """A windowed write target whose window blocks may sit in scratch slots, which a fetch into
        them would not survive."""
        return self._any_window and _manager.scratch_reuse(kv)

    def _report_settled(self, fetch: _Fetch) -> bool:
        """The fetch's arrived rows are marked and their copy into the request's pages is done."""
        return fetch.delivered and self._landed(fetch.copy)

    def _fail_waiting(self, request_id: int, reason: str) -> None:
        """Fail the request's leases still waiting for slots."""
        for lease in [lease for lease in self._line if lease._request_id == request_id]:
            self._fail(lease, reason)

    def _abandon(self, lease: _StagingLease) -> None:
        """Drop the write lease's fetch record: it delivered nothing, and readiness counts what
        earlier fetches into the cache delivered."""
        if lease._fetch is not None and self._fetches.get(lease._request_id) is lease._fetch:
            del self._fetches[lease._request_id]

    def _names(self, layer_group: int, keys: np.ndarray) -> np.ndarray:
        """The rows' names: ``uint8 (n, 54)`` for ``keys`` ``uint8 (n, 32)``."""
        return self._identity.names(layer_group, keys)

    def _memory_in_use(self) -> bool:
        """A lease or a hold is unreleased, or a slot is lost to a failed copy: the memory stays."""
        return bool(self._unreleased) or bool(self._holds) or bool(self._quarantined)

    def _keep_index_buffer(self, manager) -> None:
        """Keep the manager's page-index buffer until its shutdown: a cache that a lease or a record
        holds when the manager goes without one writes its page indices there as it closes."""
        self._index_buffer = _manager.index_buffer(manager)
        _KEPT.append(self._index_buffer)

    def _let_go_index_buffer(self) -> None:
        """At the manager's shutdown: it closes every cache after this hook, while it still holds the
        page-index buffer, so no cache writes there later."""
        _let_go(self._index_buffer)
        self._index_buffer = None

    def _end_hold(self, hold: _PartsHold) -> None:
        """The hold's holder let go; ``KeyError`` for a hold not open, which its guard prevents."""
        self._holds.remove(hold)

    def _check_fits(self, counts: Mapping[int, int]) -> None:
        """``ValueError`` if a lease needing ``counts`` rows per pool group can never be granted."""
        try:
            self._slots.check(counts)
        except ValueError as error:
            raise ValueError(
                f"the range needs more staging slots than there are ({error}); ranges of at most "
                "fetch_tokens tokens always fit"
            ) from None

    def _free_slots(self, group: int) -> int:
        """Free slots of a pool group; for tests."""
        return self._slots.free_slots(group)

    def _open_count(self) -> int:
        """Leases not yet released; for tests."""
        return len(self._unreleased)

    # Lease records, slots and grants.

    def _live(self):
        """The manager while the lender serves it; ``None`` once it shut down or is gone."""
        if self._closed:
            return None
        return self._manager_ref()

    def _whole_blocks(self, start: int, end: int) -> Tuple[int, int]:
        start, end = int(start), int(end)
        tpb = int(self._layout.tokens_per_block)
        if start < 0 or end < 0 or start > end:
            raise ValueError(f"bad token range [{start}, {end})")
        if start % tpb or end % tpb:
            raise ValueError(
                f"a staging lease covers whole blocks of {tpb} tokens, got [{start}, {end})"
            )
        return start, end

    def _counts(self, ordinals: Sequence[np.ndarray]) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for lg, lg_ordinals in enumerate(ordinals):
            g = int(self._layout.pool_group_of[lg])
            counts[g] = counts.get(g, 0) + len(lg_ordinals)
        return counts

    def _open(self, lease: _StagingLease, counts: Mapping[int, int]) -> None:
        """Record a new lease and grant it now when no lease waits and its slots are free."""
        self._unreleased.add(lease)
        if lease._rows.num_rows == 0:
            # Nothing to stage: ready at the first poll, without slots or a place in line.
            self._grant(lease, None)
            return
        lease._ticket = self._slots.ask(counts)
        self._line.append(lease)
        self._grant_waiting(fresh=lease)

    def _grant_waiting(self, fresh: Optional[_StagingLease] = None) -> None:
        """Grant the leases in line, strictly in order; ``fresh`` was looked up in this call."""
        while self._line:
            head = self._line[0]
            runs = self._slots.take(head._ticket)
            if runs is None:
                return
            self._line.popleft()
            head._ticket = None
            try:
                self._grant(head, runs, recheck=head is not fresh)
            except Exception as error:
                # One broken grant fails only its own lease: it neither stalls the line nor raises
                # out of another lease's call. Slots a copy may have been queued on stay unused.
                if any(lease is head for lease in self._holding):
                    self._quarantine(head)
                else:
                    self._slots.give(runs)
                self._fail(head, f"granting staging slots failed: {error!r}")
                logger.warning(f"KV cache lender: {head.failure}")

    def _grant(self, lease: _StagingLease, runs: Optional[Runs], recheck: bool = False) -> None:
        """Give ``lease`` its slots and view; a read then queues its copy into them."""
        if recheck and lease._kind == "read":
            problem = self._source_changed(lease)
            if problem is not None:
                self._slots.give(runs)
                self._fail(lease, problem)
                return
        self._assign_staging(lease._rows, runs)
        lease._view = self._view(lease._rows, lease._keys)
        lease._granted = True
        if runs is None:
            return
        lease._runs = runs
        self._holding.append(lease)
        if lease._kind != "read":
            return
        copy, error = self._memcpy(self._segments(lease._rows), to_staging=True)
        if copy is None:
            self._quarantine(lease)
        else:
            lease._copy = copy
        if error is not None:
            self._fail(lease, f"the copy into staging failed: {error}")

    def _source_changed(self, lease: _StagingLease) -> Optional[str]:
        """Why a read granted after waiting cannot copy the pages it looked up any more, if so."""
        manager = self._manager_ref()
        kv = _manager.kv_of(manager, lease._request_id)
        if kv is not lease._kv:
            return "the request's cache was freed while the lease waited"
        state = _manager.cache_state(kv)
        if not state.active:
            return "the request's cache was suspended while the lease waited"
        rows = lease._rows
        for lg, ordinals, slots in zip(rows.layer_groups, rows.ordinals, rows.device_slots):
            pages = _manager.locked_pages(kv, lg)
            if np.any(ordinals >= len(pages)) or np.any(pages[ordinals] != slots):
                return f"layer group {lg}: pages changed while the lease waited"
            stale_beg, stale_end = _stale(manager, self._layout, lg, state.history)
            if np.any((ordinals >= stale_beg) & (ordinals < stale_end)):
                return f"layer group {lg}: blocks left the window while the lease waited"
        return None

    def _fail(self, lease: _StagingLease, reason: str) -> None:
        """Fail a lease never seen ready: it leaves the line, and a write abandons its fetch."""
        if lease.failure is not None:
            return
        lease._set_failure(reason)
        if lease._ticket is not None:
            self._slots.cancel(lease._ticket)
            lease._ticket = None
            self._line.remove(lease)
        if lease._kind == "write":
            self._abandon(lease)

    def _quarantine(self, lease: _StagingLease) -> None:
        """Keep the lease's slots from reuse for good: a copy on them may still be running."""
        self._holding = [held for held in self._holding if held is not lease]
        # TODO: reclaim quarantined slots, or alert when they erode staging capacity.
        if lease._runs is not None:
            self._quarantined.append(lease._runs)
        logger.warning(
            f"KV cache lender: a failed copy took staging slots of request {lease._request_id}"
        )

    def _recycle(self) -> None:
        """Return the slots of every lease ``_recyclable`` allows."""
        holding = []
        for lease in self._holding:
            if self._recyclable(lease):
                self._slots.give(lease._runs)
            else:
                holding.append(lease)
        self._holding = holding

    def _on_release(self, lease: _StagingLease) -> None:
        """The lease's holder let go. The record changes first and progress runs last, so an
        unexpected error leaves the fetch abandoned rather than unsettled."""
        self._unreleased.discard(lease)
        if self._live() is None:
            return
        if lease._ticket is not None:
            self._fail(lease, "released while waiting for staging slots")
        elif lease._kind == "write" and not lease._seen_ready and lease.failure is None:
            # Released before anyone saw it ready: no backend wrote, and no marks are due.
            lease._doomed = None
            self._abandon(lease)
        self._progress()

    def _apply_marks(self, lease: _StagingLease, masks: List[np.ndarray]) -> None:
        """Copy the marked rows whose page is still the one lent into the request's pages. The
        fetch stays abandoned until that copy is queued without error; then its rows add to the
        cache's deliveries."""
        fetch = lease._fetch
        current = fetch is not None and self._fetches.get(lease._request_id) is fetch
        if current:
            del self._fetches[lease._request_id]
        kv = _manager.kv_of(self._manager_ref(), lease._request_id)
        lent = self._still_lent(kv, lease)
        copy = [mask & still for mask, still in zip(masks, lent)]
        error = None
        if any(c.any() for c in copy):
            queued, error = self._memcpy(self._segments(lease._rows, copy), to_staging=False)
            if queued is None:
                self._quarantine(lease)
            else:
                lease._copy = queued
        if current and error is None:
            fetch.copy = lease._copy
            fetch.delivered = True
            self._fetches[lease._request_id] = fetch
            self._deliver(lease._request_id, fetch, lease._rows, copy)
        self._progress()

    def _deliver(
        self, request_id: int, fetch: _Fetch, rows: _Rows, copied: List[np.ndarray]
    ) -> None:
        """Add a fetch's copied rows to what earlier fetches into the same cache delivered, so a
        fetch split into consecutive leases counts as one."""
        delivered = self._delivered.get(request_id)
        if delivered is None or delivered.kv is not fetch.kv:
            empty = [np.zeros(0, dtype=bool) for _ in range(self._layout.num_layer_groups)]
            delivered = _Delivered(fetch.kv, fetch.start, empty)
            self._delivered[request_id] = delivered
        delivered.origin = min(delivered.origin, fetch.start)
        for lg, ordinals, mask in zip(rows.layer_groups, rows.ordinals, copied):
            got = ordinals[mask]
            if not len(got):
                continue
            blocks = delivered.blocks[lg]
            if int(got.max()) >= len(blocks):
                blocks = np.concatenate([blocks, np.zeros(int(got.max()) + 1 - len(blocks), bool)])
            blocks[got] = True
            delivered.blocks[lg] = blocks
        delivered.usable = None

    def _void_past(self, delivered: _Delivered, kv) -> None:
        """Forget delivered rows at or past the cache's block count: a shrink freed their pages,
        and a regrow brings pages without their contents."""
        kept = _manager.num_blocks(kv)
        for blocks in delivered.blocks:
            if blocks[kept:].any():
                blocks[kept:] = False
                delivered.usable = None

    def _rows(self, manager, kv, start: int, end: int) -> _Rows:
        """The blocks of ``[start, end)`` a history of ``end`` reads, per layer group, with their
        device slots (-1 where a block has no page)."""
        layout = self._layout
        tpb = int(layout.tokens_per_block)
        rows = _Rows([], [], [])
        for lg in range(layout.num_layer_groups):
            ordinals = _needed_ordinals(manager, layout, lg, start // tpb, end // tpb, end)
            pages = _manager.pages(kv, lg)
            slots = np.full(len(ordinals), -1, dtype=np.int64)
            inside = ordinals < len(pages)
            slots[inside] = pages[ordinals[inside]]
            rows.layer_groups.append(lg)
            rows.ordinals.append(ordinals)
            rows.device_slots.append(slots)
        return rows

    def _lendable(self, manager, history: int, rows: _Rows) -> _Rows:
        """``rows`` without blocks that have no page or that the request's own window has passed
        (their page may hold something else)."""
        out = _Rows([], [], [])
        for lg, ordinals, slots in zip(rows.layer_groups, rows.ordinals, rows.device_slots):
            stale_beg, stale_end = _stale(manager, self._layout, lg, history)
            ok = (slots >= 0) & ~((ordinals >= stale_beg) & (ordinals < stale_end))
            out.layer_groups.append(lg)
            out.ordinals.append(ordinals[ok])
            out.device_slots.append(slots[ok])
        return out

    def _keys_for(
        self, manager, request: LlmRequest, kv, ordinals: Sequence[np.ndarray]
    ) -> List[np.ndarray]:
        """Per layer group, the reuse keys of its rows' blocks, ``uint8 (n, 32)``."""
        # TODO: every lease hashes the request's whole prefix again from block 0, so a lease late in
        # a long prompt costs time in proportion to the prompt; cache the key chain per request.
        top = max((int(o.max()) + 1 for o in ordinals if len(o)), default=0)
        keys = _manager.block_keys(manager, request, kv, top)
        return [_key_column([keys[int(o)] for o in lg_ordinals]) for lg_ordinals in ordinals]

    def _assign_staging(self, rows: _Rows, runs: Optional[Runs]) -> None:
        """Each pool group's run goes to its layer groups in order, so the rows of one pool group
        occupy consecutive slots."""
        cursor: Dict[int, int] = {}
        rows.staging_slots = []
        for lg, ordinals in zip(rows.layer_groups, rows.ordinals):
            g = int(self._layout.pool_group_of[lg])
            start = runs.runs.get(g, (0, 0))[0] if runs is not None else 0
            offset = cursor.get(g, 0)
            rows.staging_slots.append(
                np.arange(start + offset, start + offset + len(ordinals), dtype=np.int64)
            )
            cursor[g] = offset + len(ordinals)

    def _view(self, rows: _Rows, keys: List[np.ndarray]) -> RegionView:
        runs = []
        for lg, ordinals, lg_keys, slots in zip(
            rows.layer_groups, rows.ordinals, keys, rows.staging_slots
        ):
            index = self._part_of_group[int(self._layout.pool_group_of[lg])]
            part = self._parts[index]
            runs.append(
                GroupRun(
                    lg,
                    ordinals,
                    names=self._names(lg, lg_keys),
                    addresses=part.address + slots * part.slot_bytes,
                    part=index,
                )
            )
        return RegionView(tuple(runs))

    def _group_rows(self, rows: _Rows, mask: Optional[List[np.ndarray]] = None):
        """Rows by device pool group: ``(g, device slots, staging slots)``, filtered by ``mask``
        (one boolean array per layer group; ``None``: every row)."""
        by_group: Dict[int, List[int]] = {}
        for i, lg in enumerate(rows.layer_groups):
            by_group.setdefault(int(self._layout.pool_group_of[lg]), []).append(i)
        for g, members in by_group.items():
            dev = np.concatenate([rows.device_slots[i] for i in members])
            stg = np.concatenate([rows.staging_slots[i] for i in members])
            if mask is not None:
                keep = np.concatenate([np.asarray(mask[i], dtype=bool) for i in members])
                dev, stg = dev[keep], stg[keep]
            if len(dev):
                yield g, dev, stg

    def _segments(self, rows: _Rows, mask: Optional[List[np.ndarray]] = None) -> List[List[int]]:
        """``[staging address, device address, bytes]`` for every pool of every row: a staging slot
        holds the device pools' slots back to back in pool order. Segments that continue each other
        on both sides are merged."""
        out: List[List[int]] = []
        for g, dev, stg in self._group_rows(rows, mask):
            part = self._parts[self._part_of_group[g]]
            offset = 0
            for pool in self._layout.device_pools[g]:
                width = int(pool.slot_bytes)
                for d, t in zip(dev.tolist(), stg.tolist()):
                    host = part.address + t * part.slot_bytes + offset
                    device = int(pool.base) + d * width
                    last = out[-1] if out else None
                    if last and last[0] + last[2] == host and last[1] + last[2] == device:
                        last[2] += width
                    else:
                        out.append([host, device, width])
                offset += width
        return out

    # TODO: move the copies to a side stream ordered by events once every path that releases a page
    # waits for them, and issue fewer copy calls, batched or merged.
    def _memcpy(
        self, segments: Sequence[Sequence[int]], to_staging: bool
    ) -> Tuple[Optional[_Copy], Optional[str]]:
        """Queue async copies on the manager's execution stream, then record an event covering
        every copy queued, even after one failed; ``(None, error)`` if it could record none."""
        # The execution stream orders a copy after the forward passes that wrote its pages and
        # before any later owner of those pages writes them: serial with the forward on the GPU.
        # With page-locked staging the CPU does not wait for it; pageable staging may hold the call.
        stream = _manager.stream(self._manager_ref())
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
        return _Copy(event), error

    def _floor(self, manager, history: int, origin: int) -> int:
        """The lowest start that needs no restart: the history when a window has released blocks
        at it (they have no pages), else the smaller of the first fetch start and the history."""
        for lg in range(self._layout.num_layer_groups):
            stale_beg, stale_end = _stale(manager, self._layout, lg, history)
            if stale_end > stale_beg:
                return history
        return min(origin, history)

    def _usable_until(self, manager, delivered: _Delivered, committed: int) -> int:
        """The largest start ``P >= committed`` where every layer group has what it reads among the
        committed blocks and delivered rows: full attention every block below ``P``, a window its
        sinks and in-window blocks. Non-monotonic in ``P`` under windows, so each is checked."""
        tpb = int(self._layout.tokens_per_block)
        if delivered.origin > committed:
            return committed  # the first fetch left a gap after the committed tokens
        first = committed // tpb  # the blocks below are committed
        last = max((len(blocks) for blocks in delivered.blocks), default=0)
        if last <= first:
            return committed
        ok = np.ones(last - first, dtype=bool)  # ok[i]: start at (first + 1 + i) * tpb
        for lg, blocks in enumerate(delivered.blocks):
            have = np.zeros(last - first, dtype=bool)
            mine = blocks[first:last]
            have[: len(mine)] = mine
            missing = np.concatenate([[0], np.cumsum(~have)])  # missing in [first, first + j)

            def gap(a: int, b: int) -> bool:
                a, b = max(a, first), min(b, last)
                return b > a and missing[b - first] - missing[a - first] > 0

            for i, end_block in enumerate(range(first + 1, last + 1)):
                if not ok[i]:
                    continue
                stale_beg, stale_end = _stale(manager, self._layout, lg, end_block * tpb)
                if stale_end > stale_beg:
                    bad = gap(first, min(stale_beg, end_block)) or gap(stale_end, end_block)
                else:
                    bad = gap(first, end_block)
                ok[i] = not bad
        good = np.nonzero(ok)[0]
        return committed if not len(good) else (first + 1 + int(good[-1])) * tpb


# TODO: an in-place lease gets none of the staging guarantees: nothing guards its pages against
# suspend, shrink, window advance or a pool rebalance while a loan is open, it is ready before the
# manager's stream finishes with them, and mark_arrived feeds no readiness; the caller stands in.
class InPlace:
    """``InPlaceLender`` over one manager, which it references weakly. A lease holds a loan on the
    request's cache; the request's free keeps a lent cache open until its last loan ends."""

    def __init__(self, manager: weakref.ref, layout: ManagerLayout) -> None:
        self._manager_ref = manager
        self._layout = layout
        # Open loans per cache, holding the cache strongly so that dropping a lease never lets a
        # collector close it on another thread.
        self._loans: Dict[object, int] = {}
        self._freed: Dict[object, Callable[[], None]] = {}  # lent caches the manager freed
        self._kept: Optional[FrozenSet[object]] = None  # set by the manager's shutdown
        # Kept while a loan is open, and until exit once the manager is gone.
        self._index_buffer: Optional[object] = None

    def lend_read(self, request: LlmRequest, start: int, end: int) -> _InPlaceLease:
        """See ``InPlaceLender.lend_read``."""
        return self._lend(request, start, end, "read")

    def lend_write(self, request: LlmRequest, start: int, end: int) -> _InPlaceLease:
        """See ``InPlaceLender.lend_write``."""
        return self._lend(request, start, end, "write")

    def _on_free(self, request_id: int, kv_cache, after_close: Callable[[], None]) -> bool:
        """Manager hook: ``True`` if ``kv_cache`` is on loan, which the release ending its last loan
        then closes, running ``after_close`` in that call. Never raises."""
        try:
            if kv_cache not in self._loans:
                return False
            self._freed[kv_cache] = after_close
            return True
        except Exception:
            # A hook never raises into the manager's free. It keeps the cache open while any loan
            # is, since closing a lent cache would hand its pages to another request.
            logger.error(f"KV cache lender: freeing request {request_id}: {traceback.format_exc()}")
            return bool(self._loans)

    def _on_shrink(self, request_id: int, kv_cache) -> None:
        """Manager hook after an in-place shrink: nothing to do, since the caller keeps a lent
        cache from shrinking."""

    def _on_shutdown(self, impl) -> FrozenSet[object]:
        """Manager hook: the caches still on loan, kept with ``impl`` until the process exits; the
        same set on every later call."""
        if self._kept is not None:
            return self._kept
        self._kept = frozenset(self._loans)
        if self._kept:
            try:
                # A device pool cannot be freed in part, so the pools stay with the lent caches.
                _KEPT.extend([impl, *self._kept])
                logger.warning(
                    f"KV cache lender: keeping {len(self._kept)} lent caches and their pools "
                    "until exit"
                )
            except Exception:
                # A hook never raises into the manager's shutdown, which still leaves the lent
                # caches open and the pools unfreed.
                logger.error(f"KV cache lender: shutting down: {traceback.format_exc()}")
        return self._kept

    def _end_loan(self, kv_cache) -> None:
        """End one loan on ``kv_cache``; the last loan on a freed cache closes it in this call."""
        if self._kept is not None:
            return  # every cache on loan at the manager's shutdown is kept until exit
        left = self._loans.get(kv_cache, 0) - 1
        if left > 0:
            self._loans[kv_cache] = left
            return
        self._loans.pop(kv_cache, None)
        after_close = self._freed.pop(kv_cache, None)
        if after_close is not None:
            _manager.close_cache(kv_cache)
            after_close()
        if not self._loans:
            self._let_go_index_buffer()

    def _keep_index_buffer(self, manager) -> None:
        """Keep the manager's page-index buffer while a loan is open: a lent cache the manager does
        not detach writes its page indices there as it closes, even after the manager is gone."""
        self._index_buffer = _manager.index_buffer(manager)
        _KEPT.append(self._index_buffer)

    def _let_go_index_buffer(self) -> None:
        """The last loan ended: let the buffer go to its manager. With the manager gone, a cache
        this lender held may still close after this call, so the buffer stays until exit."""
        if self._manager_ref() is None:
            return
        _let_go(self._index_buffer)
        self._index_buffer = None

    def _lend(self, request: LlmRequest, start: int, end: int, kind: str) -> _InPlaceLease:
        start, end = int(start), int(end)
        if start < 0 or end < 0 or start > end:
            raise ValueError(f"bad token range [{start}, {end})")
        request_id = int(request.py_request_id)
        manager = self._manager_ref()
        if self._kept is not None or manager is None:
            return _InPlaceLease._failed(self, kind, _SHUT_DOWN)
        kv = _manager.kv_of(manager, request_id)
        if kv is None:
            return _InPlaceLease._failed(self, kind, _no_cache(request_id))
        state = _manager.cache_state(kv)
        if not state.active:
            return _InPlaceLease._failed(self, kind, _SUSPENDED)
        runs = []
        for lg in range(self._layout.num_layer_groups):
            ordinals, slots = self._device_slots(manager, kv, lg, start, end)
            paged = slots >= 0
            if kind == "write" and not paged.all():
                missing = ordinals[~paged].tolist()
                return _InPlaceLease._failed(
                    self, kind, f"layer group {lg}: blocks {missing[:8]} have no page"
                )
            # A read leaves blocks without a page out.
            runs.append(GroupRun(lg, ordinals[paged]))
        # The loan opens here: from now on the request's free keeps the cache open.
        if not self._loans:
            self._keep_index_buffer(manager)
        self._loans[kv] = self._loans.get(kv, 0) + 1
        return _InPlaceLease(self, kind, RegionView(tuple(runs)), kv)

    def _device_slots(
        self, manager, kv, lg: int, start: int, end: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The blocks of ``lg`` that ``[start, end)`` touches, a partial last one included, that a
        history of ``end`` reads, and their device slots (-1 where a block has no locked page)."""
        layout = self._layout
        tpb = int(layout.tokens_per_block)
        first, last = start // tpb, -(-end // tpb)
        windowed = layout.windows[lg] is not None
        # Only pages the cache locks: a window block behind its history keeps at most a held page,
        # which a lower cache tier may take at any time.
        pages = _manager.locked_pages(kv, lg)
        if windowed:
            ordinals = _needed_ordinals(manager, layout, lg, first, last, end)
            slots = np.full(len(ordinals), -1, dtype=np.int64)
            inside = ordinals < len(pages)
            slots[inside] = pages[ordinals[inside]]
        else:
            ordinals = np.arange(first, last, dtype=np.int64)
            slots = np.full(len(ordinals), -1, dtype=np.int64)
            paged = max(0, min(last, len(pages)) - first)
            slots[:paged] = pages[first : first + paged]
        return ordinals, slots


class _StagingLease:
    """A staging lease; holds its lender weakly and has no finalizer. Backends may read its view's
    arrays on their own threads until release, and never call it."""

    def __init__(
        self,
        lender: Staging,
        kind: str,
        request_id: int,
        kv=None,
        rows: Optional[_Rows] = None,
        keys: Optional[List[np.ndarray]] = None,
        fetch: Optional[_Fetch] = None,
    ) -> None:
        self._lender = weakref.ref(lender)
        self._kind = kind
        self._request_id = request_id
        self._kv = kv
        self._rows = rows
        self._keys = keys
        self._fetch = fetch
        self._doomed: Optional[str] = None  # why it fails at its first poll
        self._ticket: Optional[int] = None  # its place in line while it waits for slots
        self._runs: Optional[Runs] = None
        self._view: Optional[RegionView] = None
        self._copy: Optional[_Copy] = None
        self._granted = False
        self._seen_ready = False
        self._released = False
        self._marked = False
        self._failure: Optional[str] = None

    @classmethod
    def _failed(cls, lender: Staging, kind: str, request_id: int, reason: str) -> _StagingLease:
        """A lease failed at the call; open until released, like any other."""
        lease = cls(lender, kind, request_id)
        lease._set_failure(reason)
        if not lender._closed:
            lender._unreleased.add(lease)
        return lease

    def _set_failure(self, reason: str) -> None:
        self._failure = reason
        self._doomed = None

    def poll(self) -> Optional[RegionView]:
        """See ``Lease.poll``."""
        if self._released:
            raise RuntimeError("poll after release")
        lender = self._lender()
        if lender is None or lender._live() is None:
            return self._ended_poll()
        lender._progress()
        if self._failure is not None:
            return None
        if self._doomed is not None:
            lender._fail(self, self._doomed)
            return None
        if not self._granted:
            return None
        if self._kind == "read" and not lender._copy_landed(self):
            return None
        self._seen_ready = True
        return self._view

    @property
    def failure(self) -> Optional[str]:
        """See ``Lease.failure``."""
        return self._failure

    def mark_arrived(self, masks: Sequence[np.ndarray]) -> None:
        """See ``Lease.mark_arrived``."""
        if self._kind != "write":
            raise RuntimeError("mark_arrived is for write leases")
        if not self._seen_ready:
            raise RuntimeError("mark_arrived before poll() returned the view")
        if self._marked:
            raise RuntimeError("mark_arrived called twice")
        masks = _checked_masks(self._view, masks)
        self._marked = True
        lender = self._lender()
        if lender is None or lender._live() is None:
            return
        lender._apply_marks(self, masks)

    def release(self) -> None:
        """See ``Lease.release``."""
        if self._released:
            return
        self._released = True
        lender = self._lender()
        if lender is not None:
            lender._on_release(self)

    def _ended_poll(self) -> Optional[RegionView]:
        """``poll`` after the lender stopped serving: no grant or copy, just what already landed."""
        if self._failure is not None or not self._granted:
            return None
        if self._kind == "read" and self._copy is not None and not self._copy.done():
            return None
        self._seen_ready = True
        return self._view


class _InPlaceLease:
    """An in-place lease: the loan on the request's cache, held until release. Holds its lender
    weakly and has no finalizer."""

    def __init__(
        self,
        lender: InPlace,
        kind: str,
        view: Optional[RegionView],
        kv_cache=None,
        failure: Optional[str] = None,
    ) -> None:
        self._lender = weakref.ref(lender)
        self._kind = kind
        self._view = view
        self._cache = kv_cache  # the loan, until release
        self._failure = failure
        self._seen_ready = False
        self._released = False
        self._marked = False

    @classmethod
    def _failed(cls, lender: InPlace, kind: str, reason: str) -> _InPlaceLease:
        """A lease failed at the call; it holds no loan."""
        return cls(lender, kind, None, failure=reason)

    def poll(self) -> Optional[RegionView]:
        """See ``Lease.poll``."""
        if self._released:
            raise RuntimeError("poll after release")
        if self._failure is not None:
            return None
        # The request's own pages: ready at once; the caller has let the stream's work on them end.
        self._seen_ready = True
        return self._view

    @property
    def failure(self) -> Optional[str]:
        """See ``Lease.failure``."""
        return self._failure

    def mark_arrived(self, masks: Sequence[np.ndarray]) -> None:
        """See ``Lease.mark_arrived``."""
        if self._kind != "write":
            raise RuntimeError("mark_arrived is for write leases")
        if not self._seen_ready:
            raise RuntimeError("mark_arrived before poll() returned the view")
        if self._marked:
            raise RuntimeError("mark_arrived called twice")
        # The rows are in the request's pages already: only the shapes are checked.
        _checked_masks(self._view, masks)
        self._marked = True

    def release(self) -> None:
        """See ``Lease.release``."""
        if self._released:
            return
        self._released = True
        cache, self._cache = self._cache, None
        lender = self._lender()
        if cache is not None and lender is not None:
            lender._end_loan(cache)


class _PartsHold:
    """A hold on the staging memory; holds its lender weakly and has no finalizer. Without a lender
    it is inert."""

    def __init__(self, lender: Optional[Staging]) -> None:
        self._lender = weakref.ref(lender) if lender is not None else None
        self._released = False

    def release(self) -> None:
        """See ``PartsHold.release``."""
        if self._released:
            return
        self._released = True
        lender = self._lender() if self._lender is not None else None
        if lender is not None:
            lender._end_hold(self)
