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
"""The staging lender, its leases, and the hooks the manager calls on free, shrink, reuse reset and
shutdown."""

from __future__ import annotations

import collections
import enum
import traceback
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Deque, Mapping, Sequence

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
    from tensorrt_llm.mapping import Mapping as TrtllmMapping
    from tensorrt_llm.runtime.kv_cache_manager_v2 import _KVCache

    from ...llm_request import LlmRequest
    from ..kv_cache_manager_v2 import KVCacheManagerV2

# Staging memory and page-index buffers kept until the process exits, by identity, oldest first;
# only a clean shutdown removes an entry. Each change is one dict operation, atomic under the GIL,
# so lenders of managers on different threads need no lock.
_kept: dict[int, object] = {}

_SHUT_DOWN = "the KV cache manager shut down"
_RESET = "the KV cache manager reset its reuse state, after which nothing is lent by name"
_SUSPENDED = "the request's cache is suspended"
_PLACEHOLDERS = "the request carries multimodal data without digests, which its names do not cover"
_ENCODER = "the request has encoder input, which its names do not cover"
_SCRATCH = (
    "the request's cache has SWA scratch reuse on: each capacity change keeps its history within "
    "the scratch rewind of the old capacity, which a fetch's grow breaks"
)
_CONTEXT_OUTPUTS = (
    "the request returns context logits or asks for additional model outputs, which the executor "
    "gives only for computed positions and keeps across a rewind and a recompute pause"
)


def _retained() -> tuple[object, ...]:
    """What is kept until exit, oldest first; for tests."""
    return tuple(_kept.values())


def _keep(owner: object) -> None:
    """Keep ``owner`` until the process exits, or until ``_let_go``."""
    _kept[id(owner)] = owner


def _let_go(owner: object) -> None:
    """Drop ``owner`` from the keep list, compared by identity."""
    _kept.pop(id(owner), None)


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


def _allocate(nbytes: int) -> _HostMemory:
    """One host allocation of ``nbytes`` for the staging parts, put on the keep list."""
    # TODO: staging duplicates the manager's host tier, which backends cannot use in its place: its
    # host pools may move when they resize (mremap), the manager lends none of their pages, and its
    # cold-page codec may hold a row in a form other than the row's device bytes.
    memory = _HostMemory(max(int(nbytes), 1))
    _keep(memory)
    return memory


def _mapping(manager: KVCacheManagerV2) -> TrtllmMapping:
    """The manager's mapping; ``TypeError`` for a manager without one."""
    mapping = getattr(manager, "mapping", None)
    if mapping is None:
        raise TypeError("the KV cache lender needs a manager with a mapping")
    return mapping


def _checked_layout(manager: KVCacheManagerV2) -> ManagerLayout:
    """The attach's checks of the manager: a v2 manager, no context or pipeline parallelism, no
    lender yet, no recurrent state, no sparse buffer."""
    _manager.require_v2(manager)
    mapping = _mapping(manager)
    cp_size = int(mapping.cp_size)
    if cp_size > 1:
        raise ValueError(
            f"context parallelism (cp_size={cp_size}) is not supported: its ranks would give "
            "different pages the same names"
        )
    # TODO: each pipeline stage's staging slots and windows are its own, so one call can raise on
    # one stage and lend on another.
    pp_size = int(mapping.pp_size)
    if pp_size > 1:
        raise ValueError(
            f"pipeline parallelism (pp_size={pp_size}) is not supported by staging: each stage's "
            "slots and windows are its own, so one call could raise on one stage and lend on "
            "another"
        )
    if _manager.attached(manager) is not None:
        raise ValueError("a lender is already attached to this KV cache manager")
    layout = derive_layout(manager)
    recurrent = [lg for lg, state in enumerate(layout.recurrent) if state]
    if recurrent:
        raise ValueError(
            f"layer groups {recurrent} hold recurrent state, which the lender does not lend"
        )
    # TODO: lending a sparse layer group needs each page's memory tier: a cache can lock such a
    # group's read-only pages in host memory.
    sparse = [lg for lg, flag in enumerate(layout.sparse) if flag]
    if sparse:
        raise ValueError(
            f"layer groups {sparse} hold sparse buffers, whose read-only pages a cache can lock "
            "in host memory"
        )
    return layout


def _check_commits(manager: KVCacheManagerV2) -> None:
    """``ValueError`` for a manager that commits no blocks: a publish lends only committed ones."""
    if not _manager.commits_blocks(manager):
        raise ValueError(
            "staging needs a manager that commits blocks: block reuse on, and joint reuse for a "
            "draft manager; with block reuse off nothing could ever be published"
        )


def _check_lookahead(manager: KVCacheManagerV2) -> None:
    """``ValueError`` for a manager whose blocks may depend on tokens past their end: a name covers
    the tokens up to the block's end, so two requests sharing them could hold different bytes. A
    read-ahead upstream has not established, as one-model DraftTarget's, counts as none."""
    # TODO: staging serves no manager built for a draft that reads ahead (Eagle, MTP).
    lookahead = _manager.prompt_lookahead(manager)
    if lookahead > 0:
        # The target manager of a draft whose layers live elsewhere is refused too: that draft's own
        # pool is, so a fetch could fill the target's blocks but never the draft's.
        raise ValueError(
            "staging needs blocks that depend on no token past their end; this manager is built "
            f"for a one-model draft that reads {lookahead} prompt tokens ahead (Eagle or MTP), so "
            "requests sharing a block's tokens could hold different bytes under one name"
        )


def _check_connector(manager: KVCacheManagerV2) -> None:
    """``ValueError`` for a manager with a KV cache connector: it serves a request's prefix at the
    first context chunk, measured from the committed tokens, so it cannot lower a history a windowed
    fetch moved, and its loads run off the manager's stream."""
    if _manager.kv_connector(manager) is not None:
        raise ValueError(
            "staging does not serve a manager with a KV cache connector: the connector serves a "
            "request's prefix at its first context chunk, measured from the committed tokens, and "
            "cannot lower a history a fetch moved"
        )


def _attach_staging(
    manager: KVCacheManagerV2, *, scope: bytes, staging: StagingOptions, cls: type | None = None
) -> Staging:
    """``attach_staging`` with the lender class as a parameter (``Staging`` when ``None``), so a
    test can attach a subclass that breaks one rule."""
    layout = _checked_layout(manager)
    _check_commits(manager)
    _check_lookahead(manager)
    _check_connector(manager)
    if not isinstance(scope, bytes):
        raise TypeError(f"scope must be bytes, got {type(scope).__name__}")
    if not isinstance(staging, StagingOptions):
        raise TypeError(f"staging must be StagingOptions, got {type(staging).__name__}")
    counts = slot_counts(layout, staging)
    identity = Identity(scope, layout_id(layout.layout), layout.layers, layout.shards)
    slots = {g: int(counts.get(g, 0)) for g in layout.pool_groups}
    sizes = {g: slots[g] * int(layout.page_bytes[g]) for g in layout.pool_groups}
    # TODO: an exception after the allocation keeps the staging memory, and the page-index buffer
    # once kept, until the process exits.
    memory = _allocate(sum(sizes.values()))
    base = memory.address
    parts = []
    offset = 0
    for g in layout.pool_groups:
        name = identity.part_name(lg for lg, pg in enumerate(layout.pool_group_of) if pg == g)
        parts.append(Part(name, base + offset, sizes[g], int(layout.page_bytes[g]), slots[g]))
        offset += sizes[g]
    lender = (cls or Staging)(
        weakref.ref(manager), layout, identity, tuple(parts), Slots(slots), weakref.ref(memory)
    )
    lender._keep_index_buffer(manager)
    logger.info(
        f"KV cache lender: namespace {identity.namespace.hex()}, staging {offset >> 20} MiB "
        f"in {len(parts)} parts"
    )
    # Installed last, so a failure above leaves nothing attached.
    _manager.install(manager, lender)
    return lender


class _Copy:
    """Copies queued together, complete once their event says so, and never without one. ``done``
    asks the event at most once per round of the lender's progress, and never again once it
    reported completion."""

    def __init__(self, event: torch.cuda.Event | None) -> None:
        self._event = event
        self._done = False
        self._round: int | None = None

    def done(self, round_: int | None = None) -> bool:
        """Whether the copies have completed, without waiting; ``round_=None`` always asks the
        event, if there is one."""
        if self._event is None:
            return False
        if not self._done and (round_ is None or round_ != self._round):
            self._round = round_
            self._done = bool(self._event.query())
        return self._done

    def wait(self) -> None:
        """Block the host until the copies have completed; without an event, return at once."""
        if not self._done and self._event is not None:
            self._event.synchronize()
            self._done = True


# Copies that may be queued with no record of their completion: a read's from the grant of its
# slots until its copy's record, and a quarantined lease's.
_LOST = _Copy(None)


def _no_cache(request_id: int) -> str:
    return f"request {request_id} has no KV cache"


def _stale(
    manager: KVCacheManagerV2, layout: ManagerLayout, lg: int, history: int
) -> tuple[int, int]:
    """Block ordinals ``[beg, end)`` behind layer group ``lg``'s window at ``history``."""
    if layout.windows[lg] is None:
        return 0, 0
    return _manager.stale_blocks(manager, lg, history)


def _needed_runs(
    manager: KVCacheManagerV2,
    layout: ManagerLayout,
    lg: int,
    start_block: int,
    end_block: int,
    history: int,
) -> list[tuple[int, int]]:
    """Ordinal ranges ``[beg, end)`` of ``lg`` in ``[start_block, end_block)`` that a history of
    ``history`` tokens still reads, in order and none empty: the whole range for full attention;
    the sinks and the window otherwise. Arithmetic only, so a range of any length costs nothing."""
    stale_beg, stale_end = _stale(manager, layout, lg, history)
    runs = ((start_block, min(end_block, stale_beg)), (max(start_block, stale_end), end_block))
    return [(beg, end) for beg, end in runs if end > beg]


def _ordinals(runs: Sequence[tuple[int, int]]) -> np.ndarray:
    """``int64`` ordinals of ``runs``, in order."""
    if not runs:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([np.arange(beg, end, dtype=np.int64) for beg, end in runs])


def _needed_ordinals(
    manager: KVCacheManagerV2,
    layout: ManagerLayout,
    lg: int,
    start_block: int,
    end_block: int,
    history: int,
) -> np.ndarray:
    """Ordinals of ``lg`` in ``[start_block, end_block)`` that a history of ``history`` tokens
    still reads: all of them for full attention; the sinks and the window otherwise."""
    return _ordinals(_needed_runs(manager, layout, lg, start_block, end_block, history))


@dataclass(eq=False)
class _Rows:
    """Per layer group, in order: ordinals, device slots and their staging slots, aligned."""

    layer_groups: list[int]
    ordinals: list[np.ndarray]
    device_slots: list[np.ndarray]
    staging_slots: list[np.ndarray] = field(default_factory=list)

    @property
    def num_rows(self) -> int:
        return sum(len(o) for o in self.ordinals)


@dataclass(eq=False)
class _Fetch:
    """One write lease's fetch into one cache: delivered once its marked rows' copy is queued
    without error, settled once that copy has completed. ``Staging._fetch_state`` derives its
    ``_FetchState`` from ``delivered``, ``copy`` and whether ``Staging._fetches`` holds it."""

    start: int
    kv: object
    delivered: bool = False
    copy: _Copy | None = None


class _FetchState(enum.Enum):
    """A fetch's state, which ``Staging._fetch_state`` derives. ``Staging._fetches`` holds each
    request's latest fetch until it is abandoned or the request freed. ``_apply_marks`` takes it
    out before queuing the copy and puts it back delivered once the copy is queued without error."""

    # In _fetches, undelivered: readiness is None and a new fetch into the cache fails at the call.
    # _apply_marks delivers it, or abandons it where the copy fails or the call raises; _abandon
    # abandons it too.
    OPEN = "open"
    # In _fetches, its marked rows' copy queued without error and pending: as OPEN for readiness
    # and a new fetch. The copy's completion settles it.
    DELIVERED = "delivered"
    # In _fetches, its copy complete or none queued: readiness answers, and a new fetch into the
    # cache may start. The request's next fetch replaces it.
    SETTLED = "settled"
    # Out of _fetches undelivered: none of its rows count.
    ABANDONED = "abandoned"
    # Out of _fetches after its delivery: the request's next fetch replaced it, or the request's
    # free dropped it. Replacing it drops none of its rows from the cache's deliveries.
    REPLACED = "replaced"


@dataclass(eq=False)
class _Delivered:
    """What the fetches into one cache delivered: per layer group, by block ordinal, the rows whose
    last copy from staging was queued without error and that no shrink freed since; ``origin`` is
    the lowest fetch start."""

    kv: object
    origin: int
    blocks: list[np.ndarray]
    # (tokens computed before, usable_until, the end of the last run below it), last computed
    usable: tuple[int, int, int] | None = None


# TODO: a draft pool also gives up a resume below its history whose chunk would reach it, as
# every unchunked prefill's does.
def _floor_follows_history(manager: KVCacheManagerV2) -> bool:
    """Whether a request may resume only at or past its history: a chunk resumed below it may end
    below it, where the manager's context update raises, or leave a draft pool's capacity below it,
    where the pool's context resize raises."""
    return _manager.context_moves_history(manager) or _manager.resize_ends_at_chunk(manager)


class Staging:
    """``StagingLender`` over one manager, referenced weakly. Until the manager shuts down or is
    gone, a lend past its range check, a poll and a readiness call return settled leases' slots and
    grant waiting leases in order before their own work, a mark and a release after it."""

    def __init__(
        self,
        manager: weakref.ref,
        layout: ManagerLayout,
        identity: Identity,
        parts: tuple[Part, ...],
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
        # Per-request fetch records, each with the cache it describes: the request's free drops all
        # four, and readiness drops one of a cache the request no longer holds.
        self._fetches: dict[int, _Fetch] = {}  # the latest fetch into each request
        self._delivered: dict[int, _Delivered] = {}
        # Requests whose history was past the committed tokens after a fetch's grow: the cache it
        # grew, what the latest such fetch kept of the prefix computed before it, and the history
        # its grow left.
        self._advanced: dict[int, tuple[object, int, int]] = {}
        # Requests whose committed tail block a failed copy may have left with the fetch's bytes in
        # some pools and the bytes it held before in others: the cache. Readiness is empty for it.
        self._tainted: dict[int, object] = {}
        self._line: Deque[_StagingLease] = collections.deque()  # waiting for slots, in order
        self._holding: list[_StagingLease] = []  # granted, slots not yet returned
        # Open leases, held strongly: an open lease keeps the staging memory at shutdown even
        # once its holder has dropped it.
        self._unreleased: set[_StagingLease] = set()
        # Open parts holds, held strongly too: a dropped hold still keeps the memory.
        self._holds: set[_PartsHold] = set()
        self._quarantined: list[Runs] = []  # slots a failed copy may still touch; never reused
        self._closed = False
        self._reset = False  # the manager reset its reuse state: nothing is lent by name again
        self._index_buffer: object | None = None  # kept until the runtime's shutdown has returned
        self._round = 0  # rounds of progress, so each asks a copy's event at most once

    @property
    def parts(self) -> tuple[Part, ...]:
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
        if self._reset:
            return _StagingLease._failed(self, "read", request_id, _RESET)
        kv = _manager.kv_of(manager, request_id)
        if kv is None:
            return _StagingLease._failed(self, "read", request_id, _no_cache(request_id))
        unnamed = self._unnamed(request)
        if unnamed is not None:
            return _StagingLease._failed(self, "read", request_id, unnamed)
        state = _manager.cache_state(kv)
        if end > state.committed:
            raise ValueError(
                f"the range ends at {end}, past the {state.committed} committed tokens"
            )
        if not state.active:
            return _StagingLease._failed(self, "read", request_id, _SUSPENDED)
        # TODO: a publish leaves out window blocks the request's own window has passed, although
        # the prefix tree may still hold their committed pages, so a fetch whose window still keeps
        # such a block finds it missing.
        rows = self._lendable(manager, state.history, self._rows(manager, kv, start, end))
        counts = self._counts([len(ordinals) for ordinals in rows.ordinals])
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
        # The request computes its last prompt token itself, for its logits, so a fetch ends at
        # the whole blocks before it.
        prompt = _manager.prompt_length(request)
        if end > (prompt - 1) // tpb * tpb:
            raise ValueError(
                f"the range ends at {end}, past the whole blocks before the request's last prompt "
                f"token ({prompt} prompt tokens)"
            )
        # Rows and their names come from the layout for a history of ``end``, so every
        # ValueError is raised before the cache changes. The rows are counted from their ranges
        # and checked to fit before any array is built.
        runs = [
            _needed_runs(manager, layout, lg, start // tpb, end // tpb, end)
            for lg in range(layout.num_layer_groups)
        ]
        counts = self._counts([sum(e - b for b, e in lg_runs) for lg_runs in runs])
        self._check_fits(counts)
        ordinals = [_ordinals(lg_runs) for lg_runs in runs]
        # The checks above read only the call, the request's prompt and the layout, so on a
        # manager not shut down a wrong call raises whatever this rank holds for the request.
        if self._reset:
            return _StagingLease._failed(self, "write", request_id, _RESET)
        kv = _manager.kv_of(manager, request_id)
        if kv is None:
            return _StagingLease._failed(self, "write", request_id, _no_cache(request_id))
        unnamed = self._unnamed(request)
        if unnamed is not None:
            return _StagingLease._failed(self, "write", request_id, unnamed)
        state = _manager.cache_state(kv)
        if start < (state.committed // tpb) * tpb:
            raise ValueError(
                f"the range starts at {start}, inside the committed whole blocks of "
                f"{state.committed} tokens"
            )
        keys = self._keys_for(manager, request, kv, ordinals)
        if self._any_window and end < state.history:
            raise ValueError(
                f"the range ends at {end}, below the history of {state.history} tokens its "
                "windows keep"
            )
        split = self._splits_bidirectional_run(request, end)
        if split is not None:
            return _StagingLease._failed(self, "write", request_id, split)
        if not state.active:
            return _StagingLease._failed(self, "write", request_id, _SUSPENDED)
        if self._scratch_reuse_on(kv):
            return _StagingLease._failed(self, "write", request_id, _SCRATCH)
        if self._returns_context_outputs(request):
            return _StagingLease._failed(self, "write", request_id, _CONTEXT_OUTPUTS)
        # After every ValueError: whether an earlier fetch settled is this rank's own timing.
        unsettled = self._unsettled(request_id, kv)
        if unsettled is not None:
            return _StagingLease._failed(self, "write", request_id, unsettled)
        # With a window the history moves to ``end``, so windows need pages only for the blocks a
        # history of that length reads.
        position = end if self._any_window else state.history
        refusal = self._history_refusal(manager, state.history, position, end)
        if refusal is None:
            refusal = self._unwritten_pages_refusal(manager, request_id, kv, position)
        if refusal is not None:
            return _StagingLease._failed(self, "write", request_id, refusal)
        # Read before the grow, which moves a windowed cache's history to ``end``.
        kept = self._kept_by(request_id, kv, start)
        # TODO: the V2 scheduler cannot reclaim the pages a parked fetch grew.
        # TODO: route capture stops reading a request's prepopulated length once it holds the routes
        # below it, until the request finishes, so a later resume leaves positions without routes.
        # TODO: with extra KV tokens the grow locks a page past the history it leaves, which nothing
        # writes, so a later all-reusable windowed lease that leaves that block behind fails.
        # TODO: an exception once the grow resized the cache, in its fill or a step below, can leave
        # no record of a windowed cache's moved history, so readiness counts tokens never fetched as
        # computed, or a fetch record nothing settles, so readiness stays None.
        if not _manager.grow(manager, request, kv, position, end):
            return _StagingLease._failed(
                self, "write", request_id, f"no free pages to grow the cache to {end} tokens"
            )
        # The cache has grown, and readiness accounts for it whatever this lease's outcome.
        if position > state.committed:
            self._advanced[request_id] = (kv, kept, position)
        fetch = _Fetch(start, kv)
        self._fetches[request_id] = fetch
        rows = self._rows(manager, kv, start, end)
        doomed = None
        for lg, lg_ordinals, lg_slots in zip(rows.layer_groups, rows.ordinals, rows.device_slots):
            if doomed is None and np.any(lg_slots < 0):
                missing = lg_ordinals[lg_slots < 0].tolist()
                doomed = f"layer group {lg}: blocks {missing[:8]} have no page"
        lease = _StagingLease(self, "write", request_id, kv, rows, keys, fetch)
        if doomed is not None:
            # It fails at its first poll, which abandons the fetch.
            lease._doomed = doomed
            self._unreleased.add(lease)
            return lease
        self._open(lease, counts)
        return lease

    def readiness(self, request: LlmRequest) -> Readiness | None:
        """See ``StagingLender.readiness``."""
        request_id = int(request.py_request_id)
        manager = self._live()
        if manager is None:
            raise ValueError(f"{_no_cache(request_id)}: {_SHUT_DOWN}")
        self._progress()
        kv = _manager.kv_of(manager, request_id)
        if kv is None:
            raise ValueError(_no_cache(request_id))
        history = _manager.cache_state(kv).history
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
        tainted = self._tainted.get(request_id)
        if tainted is not None and tainted is not kv:
            del self._tainted[request_id]
            tainted = None
        if fetch is not None and not self._report_settled(fetch):
            return None
        if tainted is not None:
            # Empty: the request drops its cache and computes from 0.
            return Readiness(0, max(history, 1))
        known = self._computed_before(request_id, kv)
        if delivered is None:
            # Nothing delivered: what was computed and kept counts, resumed no lower than the
            # history, which a windowed fetch moved to its end.
            usable, floor = self._outside_runs(request, known)
            return Readiness(usable, max(floor, history))
        # A shrink the manager did not report still shows as blocks past the capacity.
        self._void_past(delivered, kv)
        if delivered.usable is None or delivered.usable[0] != known:
            usable = self._usable_until(manager, delivered, known)
            delivered.usable = (known, *self._outside_runs(request, usable))
        _, usable, floor = delivered.usable
        return Readiness(usable, max(floor, int(self._floor(manager, history, delivered.origin))))

    def _on_free(self, request_id: int, kv_cache: _KVCache) -> None:
        """Manager hook, after the request's cache left the map: its waiting leases fail and its
        fetch records go. Logs its own errors."""
        try:
            if self._live() is None:
                return
            # Granted leases go on: a read's copy is queued already, and a write's marks copy
            # nothing into pages other than the ones lent.
            self._fail_waiting(int(request_id), "the request was freed")
            for records in (self._fetches, self._delivered, self._advanced, self._tainted):
                records.pop(int(request_id), None)
            self._progress()
        except Exception:
            # Logged, not raised into the manager's free: the cache is then closed as usual, and
            # granted leases keep their slots.
            logger.error(f"KV cache lender: freeing request {request_id}: {traceback.format_exc()}")

    def _on_shrink(self, request_id: int, kv_cache: _KVCache) -> None:
        """Manager hook, right after the request's cache may have shrunk in place: delivered rows
        past its blocks lost their pages for good. Logs its own errors."""
        try:
            delivered = self._delivered.get(int(request_id))
            if self._live() is not None and delivered is not None and delivered.kv is kv_cache:
                self._void_past(delivered, kv_cache)
        except Exception:
            # Logged, not raised into the manager's resize. All the request's delivered rows go:
            # the cache may grow back before readiness finds the freed ones past its blocks.
            self._delivered.pop(int(request_id), None)
            logger.error(
                f"KV cache lender: shrink of request {request_id}: {traceback.format_exc()}"
            )

    def _on_reset(self) -> None:
        """Manager hook after a reuse reset, with every cache closed as an in-place weight update
        ends: names stop meaning the bytes computed, so none is lent by name again; waiting leases
        and fetches went at their requests' free; granted publishes finish. Never raises."""
        # TODO: lending by name stops for good after the manager resets its reuse state, since
        # nothing names the weights it computes with from then on.
        self._reset = True

    def _on_shutdown(self) -> None:
        """Manager hook, first in its shutdown, acting once: waits for the copies whose completion
        it recorded, fails waiting leases, and frees the staging memory unless a lease or a hold is
        open or a slot was lost to a failed copy. Logs its own errors."""
        if self._closed:
            return
        try:
            # With page-locked staging and the fresh-page fill off, the lender's only host
            # wait: no more work comes on the stream.
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
            # Logged, not raised into the manager's shutdown; the staging memory then stays in
            # the keep list until exit.
            self._closed = True
            logger.error(f"KV cache lender: shutting down: {traceback.format_exc()}")

    def _on_caches_closed(self) -> None:
        """Manager hook, once the runtime's shutdown has returned, after which no cache writes into
        the page-index buffer, so the lender lets it go. A shutdown that raises skips this call, and
        the buffer stays until a retried one returns, or until exit."""
        if self._index_buffer is None:
            return
        self._let_go_index_buffer()

    # One rule per method, so a test subclass that breaks exactly one rule overrides one method.

    def _progress(self) -> None:
        """Return the slots of settled leases, then grant waiting leases in order."""
        if self._live() is None:
            return
        self._round += 1
        self._recycle()
        self._grant_waiting()

    def _landed(self, copy: _Copy | None) -> bool:
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
            if not lease._released or lease._written_unmarked():
                return False
        # TODO: every lender call asks each released lease's pending copy again, so N calls while P
        # copies pend cost N*P event queries.
        return self._landed(lease._copy)

    def _still_lent(self, kv: _KVCache | None, lease: _StagingLease) -> list[np.ndarray]:
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

    def _unnamed(self, request: LlmRequest) -> str | None:
        """Why the request's KV depends on more than its names cover, if it does: multimodal data
        the manager keys by placeholder tokens alone, or encoder input."""
        if _manager.keyed_by_placeholders(request):
            return _PLACEHOLDERS
        if _manager.has_encoder_input(request):
            return _ENCODER
        return None

    # TODO: the lender cannot see the model's sliding window, so it also keeps fetch ends and
    # resumes out of runs that window would cover whole.
    def _splits_bidirectional_run(self, request: LlmRequest, end: int) -> str | None:
        """Why the fetch ends strictly inside a run of multimodal tokens the scheduler keeps within
        one context chunk, if it does, whatever the run's length."""
        for b, e in _manager.bidirectional_runs(request):
            if b < end < e:
                return (
                    f"the range ends at {end}, inside the multimodal tokens [{b}, {e}), a run the "
                    "scheduler keeps within one context chunk"
                )
        return None

    def _outside_runs(self, request: LlmRequest, usable: int) -> tuple[int, int]:
        """The end lowered to the start of a run of multimodal tokens it falls strictly inside, and
        the lowest floor that then leaves no position strictly inside a run below that end: the end
        of the last run at or below it. Runs are those the scheduler keeps within one chunk."""
        runs = _manager.bidirectional_runs(request)
        for b, e in runs:
            if b < usable < e:
                usable = b
        return int(usable), max((e for _, e in runs if e <= usable), default=0)

    def _scratch_reuse_on(self, kv: _KVCache) -> bool:
        """A write target with SWA scratch reuse on, windowed or not: each capacity change keeps its
        history within the scratch rewind of the old capacity, which a fetch's grow breaks, and a
        window's next chunk overwrites scratch slots."""
        return _manager.scratch_reuse(kv)

    # TODO: the executor gives context outputs only for the positions a request computes, prompt
    # logprobs pair the logits with the prompt from its second token wherever they start, and a
    # rewind or a recompute pause keeps those held, so a request returning them takes no fetch.
    def _returns_context_outputs(self, request: LlmRequest) -> bool:
        """Whether the request returns context logits or asks for additional model outputs, which a
        fetch leaves without the fetched positions, or after a context step gapped or repeated."""
        return _manager.returns_context_outputs(request)

    def _history_refusal(
        self, manager: KVCacheManagerV2, history: int, position: int, end: int
    ) -> str | None:
        """Why the history the fetch leaves, ``position``, admits no resume at the positions
        readiness would count, or ``None``: in a cache without a window whose floor follows the
        history, the history stands past ``end``."""
        if self._any_window:
            return None  # the fetch moves the history to ``end``
        if not _floor_follows_history(manager):
            return None  # the floor is a delivered fetch's lowest start, or the history if lower
        if position > end:
            return f"the cache's history of {history} tokens stands past the fetch's end, {end}"
        return None

    def _unwritten_pages_refusal(
        self, manager: KVCacheManagerV2, request_id: int, kv: _KVCache, position: int
    ) -> str | None:
        """Why moving the history to ``position`` would have the commit store bytes the request
        never wrote, or ``None``: a window leaving behind a block whose page holds tokens past those
        the request computed keeps that page for the commit (a partial match's copy, a page grown
        early, a row an earlier lease missed), unless a fetch into the cache delivered its row."""
        if not _manager.keeps_passed_pages(manager):
            return None
        computed = self._computed_before(request_id, kv)
        first = computed // int(self._layout.tokens_per_block)  # the first block past them
        delivered = self._delivered.get(request_id)
        fetched = delivered.blocks if delivered is not None and delivered.kv is kv else None
        for lg in range(self._layout.num_layer_groups):
            stale_beg, stale_end = _stale(manager, self._layout, lg, position)
            beg = max(first, stale_beg)
            if beg >= stale_end:
                continue
            unwritten = _manager.locked_pages(kv, lg)[beg:stale_end] >= 0
            if fetched is not None:
                # A delivered row's page holds the bytes the fetch copied in.
                got = fetched[lg][beg : beg + len(unwritten)]
                unwritten[: len(got)] &= ~got
            blocks = (beg + np.nonzero(unwritten)[0]).tolist()
            if blocks:
                # TODO: no runtime call drops one block's page in one layer group, so the fetch
                # fails where the commit could store no page for those blocks instead.
                return (
                    f"layer group {lg} leaves blocks {blocks[:8]} behind its window at {position} "
                    f"tokens; past the {computed} tokens the request computed, their pages hold "
                    "bytes it never wrote and no fetch delivered, which the commit would store"
                )
        return None

    def _unsettled(self, request_id: int, kv: _KVCache) -> str | None:
        """Why the request's earlier fetch into ``kv`` keeps a new one from starting, or ``None``:
        it has not settled on this rank, where the copy its marks queued may still be pending."""
        previous = self._fetches.get(request_id)
        if previous is not None and previous.kv is kv and not self._report_settled(previous):
            return f"request {request_id} already has an unsettled fetch"
        return None

    def _report_settled(self, fetch: _Fetch) -> bool:
        """The fetch's arrived rows are marked and their copy into the request's pages is done."""
        return fetch.delivered and self._landed(fetch.copy)

    def _fetch_state(self, request_id: int, fetch: _Fetch) -> _FetchState:
        """The state of ``fetch``, a fetch into request ``request_id``, from ``_fetches`` and
        ``_report_settled``; see ``_FetchState``."""
        if self._fetches.get(request_id) is not fetch:
            return _FetchState.REPLACED if fetch.delivered else _FetchState.ABANDONED
        if self._report_settled(fetch):
            return _FetchState.SETTLED
        return _FetchState.DELIVERED if fetch.delivered else _FetchState.OPEN

    def _fail_waiting(self, request_id: int, reason: str) -> None:
        """Fail the request's leases still waiting for slots."""
        for lease in [lease for lease in self._line if lease._request_id == request_id]:
            self._fail(lease, reason)

    def _abandon(self, lease: _StagingLease) -> None:
        """Drop the write lease's fetch record: it delivered nothing, and readiness counts what
        earlier fetches into the cache delivered."""
        if lease._fetch is not None and self._fetches.get(lease._request_id) is lease._fetch:
            del self._fetches[lease._request_id]

    def _computed_before(self, request_id: int, kv: _KVCache) -> int:
        """What readiness counts as computed besides delivered rows: the tokens up to the history,
        but only the committed ones and what the latest fetch kept while the history stays where
        that fetch's grow left it past them (a windowed fetch moves it to its end)."""
        state = _manager.cache_state(kv)
        advanced = self._advanced.get(request_id)
        # Past where that fetch left it, the history moved as the request resumed and computed on.
        if advanced is None or advanced[0] is not kv or state.history > advanced[2]:
            return max(state.committed, state.history)
        return max(state.committed, advanced[1])

    def _kept_by(self, request_id: int, kv: _KVCache, start: int) -> int:
        """What a fetch from ``start`` keeps of what was computed before it, since it may overwrite
        every block from ``start`` on. It never passes the history, which no shrink goes below;
        delivered rows past it keep their own record."""
        return min(self._computed_before(request_id, kv), start)

    def _names(self, layer_group: int, keys: np.ndarray) -> np.ndarray:
        """The rows' names: ``uint8 (n, 54)`` for ``keys`` ``uint8 (n, 32)``."""
        return self._identity.names(layer_group, keys)

    def _memory_in_use(self) -> bool:
        """A lease or a hold is unreleased, or a slot is lost to a failed copy: the memory stays."""
        return (
            bool(self._unreleased)
            or bool(self._holds)
            or bool(self._quarantined)
            or any(lease._copy is _LOST for lease in self._holding)
        )

    def _keep_index_buffer(self, manager: KVCacheManagerV2) -> None:
        """Keep the manager's page-index buffer until the runtime's shutdown has returned: a cache a
        lease or a record holds writes its page indices there as it closes, also once the manager is
        gone."""
        self._index_buffer = _manager.index_buffer(manager)
        _keep(self._index_buffer)

    def _let_go_index_buffer(self) -> None:
        """Once the runtime's shutdown has returned: no cache writes into the page-index buffer any
        more."""
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

    def _live(self) -> KVCacheManagerV2 | None:
        """The manager while the lender serves it; ``None`` once it shut down or is gone."""
        if self._closed:
            return None
        return self._manager_ref()

    def _whole_blocks(self, start: int, end: int) -> tuple[int, int]:
        start, end = int(start), int(end)
        tpb = int(self._layout.tokens_per_block)
        if start < 0 or end < 0 or start > end:
            raise ValueError(f"bad token range [{start}, {end})")
        if start % tpb or end % tpb:
            raise ValueError(
                f"a staging lease covers whole blocks of {tpb} tokens, got [{start}, {end})"
            )
        return start, end

    def _counts(self, rows: Sequence[int]) -> dict[int, int]:
        """Rows per pool group, from the rows of each layer group."""
        counts: dict[int, int] = {}
        for lg, lg_rows in enumerate(rows):
            g = int(self._layout.pool_group_of[lg])
            counts[g] = counts.get(g, 0) + int(lg_rows)
        return counts

    def _open(self, lease: _StagingLease, counts: Mapping[int, int]) -> None:
        """Record a new lease and grant it now when no lease waits and its slots are free, or at
        once, out of line, when it has no rows."""
        self._unreleased.add(lease)
        if lease._rows.num_rows == 0:
            # Nothing to stage: ready at the first poll, without slots or a place in line.
            self._grant(lease, None)
            return
        lease._ticket = self._slots.ask(counts)
        self._line.append(lease)
        self._grant_waiting(fresh=lease)

    def _grant_waiting(self, fresh: _StagingLease | None = None) -> None:
        """Grant the leases in line, strictly in order; ``fresh`` was looked up in this call."""
        while self._line:
            head = self._line[0]
            # TODO: an exception while slots are taken, a MemoryError say, can leave slots taken for
            # no lease, or drop the head's ticket with the head still in line, after which every
            # round of progress raises "not waiting" until the head fails.
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
                if head._kind == "read" and head._runs is not None:
                    self._quarantine(head)
                else:
                    head._view = head._runs = None
                    self._slots.give(runs)
                reason = f"granting staging slots failed: {error!r}"
                if head is fresh and head._kind == "write":
                    # The cache grew for it, so it fails at its first poll, which abandons the
                    # fetch, as a lease missing pages does.
                    head._doomed = reason
                else:
                    self._fail(head, reason)
                logger.warning(f"KV cache lender: {reason}")
            except BaseException:
                # An interrupt can leave a copy queued on the slots without its completion.
                head._runs = runs
                self._quarantine(head)
                raise

    def _grant(self, lease: _StagingLease, runs: Runs | None, recheck: bool = False) -> None:
        """Give ``lease`` its slots and view; a read then queues its copy into them."""
        if recheck and lease._kind == "read":
            problem = self._source_changed(lease)
            if problem is not None:
                self._slots.give(runs)
                self._fail(lease, problem)
                return
        self._assign_staging(lease._rows, runs)
        if lease._kind == "read" and runs is not None:
            # Lost until its copy is recorded: no raise from here on leaves it ready.
            lease._copy = _LOST
        lease._view = self._view(lease._rows, lease._keys)
        if runs is None:
            return
        lease._runs = runs
        self._holding.append(lease)
        if lease._kind != "read":
            return
        copy, error = self._memcpy(self._segments(lease._rows), to_staging=True)
        if copy is None:
            self._quarantine(lease)
        if error is not None:
            self._fail(lease, f"the copy into staging failed: {error}")
        if copy is not None:
            lease._copy = copy

    def _source_changed(self, lease: _StagingLease) -> str | None:
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
        # The copy is lost and the runs recorded before the lease leaves the holding list, so a step
        # that raises leaves its slots out of use and the staging memory kept at the shutdown.
        lease._copy = _LOST
        # TODO: quarantined slots are never reused, so failed copies erode staging capacity: a lease
        # needing a longer run than any left between lost slots waits in line for good, holding up
        # every lease behind it.
        if lease._runs is not None:
            self._quarantined.append(lease._runs)
        self._holding = [held for held in self._holding if held is not lease]
        logger.warning(
            f"KV cache lender: a failed copy took staging slots of request {lease._request_id}"
        )

    # TODO: every lender call checks every lease holding slots, even with no copy pending, so
    # polling H such leases once each costs about H*H checks.
    def _recycle(self) -> None:
        """Return the slots of every lease ``_recyclable`` allows."""
        # Every lease is judged before any slot returns: an event query that raises returns none
        # and leaves every lease held for the next call.
        recyclable = [self._recyclable(lease) for lease in self._holding]
        # TODO: an exception while slots return, a MemoryError say, can leave a lease held after
        # some or all of its slots returned, so every later round of progress raises "freed twice".
        holding = []
        for lease, done in zip(self._holding, recyclable):
            if done:
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

    def _apply_marks(self, lease: _StagingLease, masks: list[np.ndarray]) -> None:
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
            segments = self._segments(lease._rows, copy)
            self._overwrite(lease, copy)
            tail = self._covers_committed_tail(kv, lease._rows, copy)
            queued = None
            try:
                queued, error = self._memcpy(segments, to_staging=False)
                lease._copy = queued
            finally:
                # A copy may be queued without its completion on the lease, also when the call
                # raised or was interrupted before the lease took it.
                if queued is None or lease._copy is not queued:
                    lease._copy = _LOST
                    self._quarantine(lease)
                if tail and (queued is None or error is not None):
                    self._tainted[lease._request_id] = kv
        if current and error is None:
            fetch.copy = lease._copy
            fetch.delivered = True
            self._fetches[lease._request_id] = fetch
            self._deliver(lease._request_id, fetch, lease._rows, copy)
        self._progress()

    def _covers_committed_tail(self, kv: _KVCache, rows: _Rows, copied: list[np.ndarray]) -> bool:
        """The copy writes the block the committed tokens end inside, whose committed tokens
        readiness counts without a delivery."""
        tpb = int(self._layout.tokens_per_block)
        committed = _manager.cache_state(kv).committed
        if committed % tpb == 0:
            return False
        tail = committed // tpb
        return any(bool(np.any(o[m] == tail)) for o, m in zip(rows.ordinals, copied))

    def _overwrite(self, lease: _StagingLease, copied: list[np.ndarray]) -> None:
        """Stop counting the delivered rows ``copied`` selects, before their copy is queued: one
        that fails partway can leave a row holding each fetch's bytes in different pools.
        ``_deliver`` counts them again once the copy is queued without error."""
        delivered = self._delivered.get(lease._request_id)
        if delivered is None or delivered.kv is not lease._kv:
            return
        for lg, ordinals, mask in zip(lease._rows.layer_groups, lease._rows.ordinals, copied):
            blocks = delivered.blocks[lg]
            rows = ordinals[mask]
            rows = rows[rows < len(blocks)]
            if blocks[rows].any():
                blocks[rows] = False
                delivered.usable = None

    def _deliver(
        self, request_id: int, fetch: _Fetch, rows: _Rows, copied: list[np.ndarray]
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

    def _void_past(self, delivered: _Delivered, kv: _KVCache) -> None:
        """Forget delivered rows at or past the cache's block count: a shrink freed their pages,
        and a regrow brings pages without their contents."""
        kept = _manager.num_blocks(kv)
        for blocks in delivered.blocks:
            if blocks[kept:].any():
                blocks[kept:] = False
                delivered.usable = None

    def _rows(self, manager: KVCacheManagerV2, kv: _KVCache, start: int, end: int) -> _Rows:
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

    def _lendable(self, manager: KVCacheManagerV2, history: int, rows: _Rows) -> _Rows:
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
        self,
        manager: KVCacheManagerV2,
        request: LlmRequest,
        kv: _KVCache,
        ordinals: Sequence[np.ndarray],
    ) -> list[np.ndarray]:
        """Per layer group, the reuse keys of its rows' blocks, ``uint8 (n, 32)``."""
        # TODO: every lease hashes the request's whole prefix again from block 0.
        top = max((int(o.max()) + 1 for o in ordinals if len(o)), default=0)
        keys = _manager.block_keys(manager, request, kv, top)
        columns = [b"".join(keys[int(o)] for o in lg_ordinals) for lg_ordinals in ordinals]
        return [np.frombuffer(column, dtype=np.uint8).reshape(-1, 32) for column in columns]

    def _assign_staging(self, rows: _Rows, runs: Runs | None) -> None:
        """Each pool group's run goes to its layer groups in order, so the rows of one pool group
        occupy consecutive slots."""
        cursor: dict[int, int] = {}
        rows.staging_slots = []
        for lg, ordinals in zip(rows.layer_groups, rows.ordinals):
            g = int(self._layout.pool_group_of[lg])
            start = runs.runs.get(g, (0, 0))[0] if runs is not None else 0
            offset = cursor.get(g, 0)
            rows.staging_slots.append(
                np.arange(start + offset, start + offset + len(ordinals), dtype=np.int64)
            )
            cursor[g] = offset + len(ordinals)

    def _view(self, rows: _Rows, keys: list[np.ndarray]) -> RegionView:
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

    def _segments(self, rows: _Rows, mask: list[np.ndarray] | None = None) -> list[list[int]]:
        """``[staging address, device address, bytes]`` for every pool of every row ``mask`` keeps
        (one boolean array per layer group; ``None``: every row), by device pool group. Segments
        that continue each other on both sides are merged."""
        by_group: dict[int, list[int]] = {}
        for i, lg in enumerate(rows.layer_groups):
            by_group.setdefault(int(self._layout.pool_group_of[lg]), []).append(i)
        out: list[list[int]] = []
        for g, members in by_group.items():
            dev = np.concatenate([rows.device_slots[i] for i in members])
            stg = np.concatenate([rows.staging_slots[i] for i in members])
            if mask is not None:
                keep = np.concatenate([np.asarray(mask[i], dtype=bool) for i in members])
                dev, stg = dev[keep], stg[keep]
            part = self._parts[self._part_of_group[g]]
            # A staging slot holds the device pools' slots back to back in pool order.
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

    # TODO: copies run serially with the forward passes on the execution stream, and a copy call
    # per row and pool, which rows share only where they continue each other in a pool group of
    # one pool, keeps small rows below the host-to-device bandwidth.
    def _memcpy(
        self, segments: Sequence[Sequence[int]], to_staging: bool
    ) -> tuple[_Copy | None, str | None]:
        """Queue async copies on the manager's execution stream, then record an event covering
        every copy queued, even after one failed; ``(None, error)`` if it could record none. With
        page-locked staging the CPU does not wait for them; pageable staging may hold the call."""
        # The execution stream orders a copy after the forward passes that wrote its pages and
        # before later work on it, which a page's new owner waits for; no path that releases a page
        # waits for the copies, and a writer off that stream is not ordered after them.
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

    def _floor(self, manager: KVCacheManagerV2, history: int, origin: int) -> int:
        """The lowest start that needs no restart: the history where a chunk resumed below it may
        end below it and raise, or a window has released blocks at it (they have no pages), else
        the smaller of the lowest start of a delivered fetch and the history."""
        if _floor_follows_history(manager):
            return history
        for lg in range(self._layout.num_layer_groups):
            stale_beg, stale_end = _stale(manager, self._layout, lg, history)
            if stale_end > stale_beg:
                return history
        return min(origin, history)

    def _usable_until(self, manager: KVCacheManagerV2, delivered: _Delivered, known: int) -> int:
        """The largest start ``P >= known`` where every layer group has what it reads among the
        blocks below ``known``, computed before the fetches, and delivered rows: full attention
        every block below ``P``, a window its sinks and in-window blocks."""
        # TODO: a fetch none of whose rows a window reads delivers none, so the usable end stays at
        # the blocks computed before the fetches, and the interval can come out empty although the
        # fetch covered all that a resume at its end reads.
        tpb = int(self._layout.tokens_per_block)
        first = known // tpb  # the blocks below were computed
        last = max((len(blocks) for blocks in delivered.blocks), default=0)
        if last <= first:
            return known
        # Non-monotonic in P under windows, so each is checked. Blocks below known behind a window's
        # history have no pages, but no start at or above the floor reads them.
        top = last  # no start past a full-attention group's first missing block passes that group
        missing = []  # per layer group: missing[j], the undelivered blocks of [first, first + j)
        for lg, blocks in enumerate(delivered.blocks):
            have = np.zeros(last - first, dtype=bool)
            mine = blocks[first:last]
            have[: len(mine)] = mine
            missing.append(np.concatenate([[0], np.cumsum(~have)]))
            if self._layout.windows[lg] is None and not have.all():
                top = min(top, first + int(np.argmin(have)))

        def gap(lg: int, a: int, b: int) -> bool:
            a, b = max(a, first), min(b, last)
            return b > a and missing[lg][b - first] - missing[lg][a - first] > 0

        # The largest start first: a fetch whose leases all landed is usable at its end at once.
        for end_block in range(top, first, -1):
            for lg in range(len(delivered.blocks)):
                stale_beg, stale_end = _stale(manager, self._layout, lg, end_block * tpb)
                if stale_end > stale_beg:
                    bad = gap(lg, first, min(stale_beg, end_block)) or gap(lg, stale_end, end_block)
                else:
                    bad = gap(lg, first, end_block)
                if bad:
                    break
            else:
                return end_block * tpb
        return known


class _LeaseBase:
    """A lease's failure, and the checks of a write's one mark."""

    @property
    def failure(self) -> str | None:
        """See ``Lease.failure``."""
        return self._failure

    def _take_marks(self, masks: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Copies of ``masks`` for the one mark of a write whose view ``poll()`` returned;
        ``ValueError`` unless one bool mask of shape ``(len(run),)`` per run."""
        if self._kind != "write":
            raise RuntimeError("mark_arrived is for write leases")
        if not self._seen_ready:
            raise RuntimeError("mark_arrived before poll() returned the view")
        if self._marked:
            raise RuntimeError("mark_arrived called twice")
        masks, runs = list(masks), self._view.runs
        if len(masks) != len(runs):
            raise ValueError(f"{len(masks)} masks for {len(runs)} runs")
        out = []
        for run, mask in zip(runs, masks):
            mask = np.asarray(mask)
            if mask.dtype != np.bool_ or mask.shape != (len(run),):
                raise ValueError(
                    f"layer group {run.layer_group}: the mask must be bool of shape "
                    f"({len(run)},), got {mask.dtype} {mask.shape}"
                )
            out.append(mask.copy())
        self._marked = True
        return out


class _LeaseState(enum.Enum):
    """A staging lease's state, which ``_StagingLease._state`` derives from its fields. ``release``
    moves any state to ``RELEASED``. ``Staging._line`` holds the waiting leases in order, and
    ``Staging._holding`` those with slots until ``_recycle`` or ``_quarantine`` takes them out."""

    # In line for slots: _grant_waiting grants it in turn, or dooms a fresh write whose grant
    # raised; _fail fails it.
    WAITING = "waiting"
    # Without slots or a view: its first poll fails it, as does the shutdown.
    DOOMED = "doomed"
    # Its view and any slots: poll makes it ready, a read once its copy into them has landed, and
    # fails a read whose copy is lost.
    GRANTED = "granted"
    # Poll returned its view: a backend may use its slots until release; mark_arrived marks a write.
    READY = "ready"
    # A write mark_arrived marked: its marked rows still lent go to the request's pages.
    MARKED = "marked"
    # Failed at the call, in line, at its grant, at a poll or at the shutdown.
    FAILED = "failed"
    # Let go: _recycle returns its slots once neither a backend nor a copy can reach them.
    RELEASED = "released"


class _StagingLease(_LeaseBase):
    """A staging lease; holds its lender weakly and has no finalizer. Backends may read its view's
    arrays on their own threads until release, and never call it."""

    def __init__(
        self,
        lender: Staging,
        kind: str,
        request_id: int,
        kv: _KVCache | None = None,
        rows: _Rows | None = None,
        keys: list[np.ndarray] | None = None,
        fetch: _Fetch | None = None,
    ) -> None:
        self._lender = weakref.ref(lender)
        self._kind = kind
        self._request_id = request_id
        self._kv = kv
        self._rows = rows
        self._keys = keys
        self._fetch = fetch
        self._doomed: str | None = None  # why it fails at its first poll
        self._ticket: int | None = None  # its place in line while it waits for slots
        self._runs: Runs | None = None
        self._view: RegionView | None = None
        self._copy: _Copy | None = None
        self._seen_ready = False
        self._released = False
        self._marked = False
        self._failure: str | None = None

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

    def _state(self) -> _LeaseState:
        """Its state, from the first of these fields that is set; see ``_LeaseState``."""
        if self._released:
            return _LeaseState.RELEASED
        if self._failure is not None:
            return _LeaseState.FAILED
        if self._doomed is not None:
            return _LeaseState.DOOMED
        if self._ticket is not None:
            return _LeaseState.WAITING
        if self._marked:
            return _LeaseState.MARKED
        return _LeaseState.READY if self._seen_ready else _LeaseState.GRANTED

    def _written_unmarked(self) -> bool:
        """A write whose view a poll returned and that has no mark: its backend may still write
        into its slots."""
        return self._kind == "write" and self._seen_ready and not self._marked

    def __repr__(self) -> str:
        return f"<staging {self._kind} lease of request {self._request_id}: {self._state().value}>"

    def poll(self) -> RegionView | None:
        """See ``Lease.poll``."""
        if self._released:
            raise RuntimeError("poll after release")
        if self._kind == "read" and self._copy is _LOST and self._failure is None:
            # A grant that raised left its copy unrecorded and the lease not failed.
            self._set_failure("the copy into staging has no record of its completion")
        lender = self._lender()
        if lender is None or lender._live() is None:
            return self._ended_poll()
        lender._progress()
        if self._failure is not None:
            return None
        if self._doomed is not None:
            lender._fail(self, self._doomed)
            return None
        if self._view is None:
            return None
        if self._kind == "read" and not lender._copy_landed(self):
            return None
        self._seen_ready = True
        return self._view

    def mark_arrived(self, masks: Sequence[np.ndarray]) -> None:
        """See ``Lease.mark_arrived``."""
        masks = self._take_marks(masks)
        lender = self._lender()
        if lender is not None and lender._live() is not None:
            lender._apply_marks(self, masks)

    def release(self) -> None:
        """See ``Lease.release``."""
        if self._released:
            return
        self._released = True
        lender = self._lender()
        if lender is not None:
            lender._on_release(self)

    def _ended_poll(self) -> RegionView | None:
        """``poll`` after the lender stopped serving: no grant or copy, just what already landed."""
        if self._failure is not None or self._view is None:
            return None
        if self._kind == "read" and self._copy is not None and not self._copy.done():
            return None
        self._seen_ready = True
        return self._view


class _PartsHold:
    """A hold on the staging memory; holds its lender weakly and has no finalizer. Without a lender
    it is inert."""

    def __init__(self, lender: Staging | None) -> None:
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
