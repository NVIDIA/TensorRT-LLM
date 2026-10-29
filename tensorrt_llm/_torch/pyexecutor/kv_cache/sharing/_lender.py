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
"""The host-staging implementation behind the mechanism-independent lending API.

Relays whole blocks between a request's device pages and host staging slots.

Leases needing slots are granted first come, first served; a lease with no block entries is
ready at its first poll, with no place in line. Lender and lease methods run only on the
manager's thread. Backends never lend or call a lease; their own threads read a view, access the
memory it points to, call no lease or lender method, and tell the holder through their own
channel.

Copies queue on the manager's execution stream, serial with the forward on the GPU: a step
queuing copies takes about their time longer. With page-locked staging and the fresh-page fill
off, no lend, poll, mark or resume_bounds call waits for them on the CPU; the manager's shutdown
does, and so does a process's first cache close, which waits for the whole device. Under
confidential computing staging is pageable, and a copy can hold the call queuing it until the
stream reaches it.

The executor's pool rebalance suspends the caches of its active requests, parked ones among
them, and moves their pages, lent ones included: queued copies complete first, a read still
waiting for slots fails at its grant where its pages moved, and a write marked after the
rebalance copies only the block entries whose pages stayed, all that ``resume_bounds`` counts.

Caller obligations:
    - A staging backend, on the manager's thread, takes a hold with ``hold_regions()`` and
      registers ``regions`` after the attach and before it touches a lease; it deregisters
      before the manager shuts down, and releases the hold once deregistration is confirmed.
    - Each iteration of the executor loop the holder polls every open lease and the waiter asks
      ``resume_bounds`` for every waiting request, also when no new request arrives.
    - At shutdown the executor loop stops; staging backends stop, deregister and release their
      holds; the holder marks and releases what they give back; the manager shuts down last.
    - A request whose one-model draft has a joint-reuse pool of its own shares its context
      cursor with that pool: the caller fetches the same range into both managers, each
      through its own lender, resumes within both intervals and publishes both. A fetch into
      one alone leaves the other without the prefix while the shared cursor moves past it.
    - While requests are parked in a state other than a disaggregated transfer state, the caller
      keeps room for the running requests in each pool group: the pages parked requests' caches
      lock, the other pages the scheduler cannot reclaim, and, for the generation request that
      needs the most, the pages of its cache within its windows, which a resume locks all at
      once, and the pages its next step adds stay within ``max_util_for_resume`` of the group's
      GPU pages. The pages a fetch grows are outside the V2 scheduler's reach: it neither
      evicts, pauses nor preempts a parked request. Its deadlock check counts no pass while any
      request is in a disaggregated transfer state, as one parked in the transfer-in-progress
      state is, and otherwise raises "V2 scheduler deadlock" after 1000 passes in a row that
      schedule and reclaim nothing while a context or generation request waits, so without that
      room a request that cannot be admitted, resume or grow can end in that deadlock. A share
      of each pool group for the parked fetches alone does not ensure it, with or without a
      cache tier below the GPU. The manager's ``get_page_indices_by_layer_group`` lists a
      request's pages per layer group, one beam's (under beam search each further beam also
      locks a page of its own for every block not wholly inside the prompt), and the runtime's
      ``impl.pool_group_descs`` give each pool group's number of GPU pages and its layer groups.
    - Under attention data parallelism without a cache transceiver, the caller keeps a rank from
      parking all its active requests, or accepts the stall: the executor counts parked requests
      as schedulable, so a rank whose active requests are all parked, at its cap of active
      requests or without pages for a padding dummy, schedules nothing, and its empty batch
      holds every rank's forward until a fetch settles.
    - An integrator adding a writer of recycled pages off the manager's stream makes it wait on
      that stream first, as the disaggregated receive does: such a writer is not ordered after
      staging copies, which precede only later work on that stream, the work a page's new owner
      waits for. With a KV cache connector refused, the manager's only such writer is the
      fresh-page fill (``TRTLLM_KV_FRESH_PAGE_FILL``, a diagnostic off by default), which
      waits for the manager stream before filling and synchronizes the fill stream afterward:
      it overwrites no page a queued copy still reads, and a ``lend_write`` that grows the cache
      then waits on the CPU for the queued staging copies.
    - The caller releases, or frees the request of, a lease that waits too long: a slot lost to
      a failed copy is never reused, and a slot is lost when a copy into or out of it may have
      been queued without its completion recorded, because recording that completion failed or a
      read's grant failed while queuing the copy. Such failures erode staging capacity: a lease
      needing, in a pool group, a longer run of slots than any there without a lost slot waits
      in line for good without failing, holding up every lease behind it until it is released or
      its request is freed. A lost slot also keeps the staging memory until process exit.

"""

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

from . import _manager, _ranges
from ._fetch_ledger import _FetchLedger
from ._identity import Identity
from ._layout import ManagerLayout, derive_layout, layout_id
from ._slots import SlotQueue, SlotRuns, slot_counts
from ._types import LayerGroupBlocks, LeaseView, MemoryRegion, ResumeBounds, StagingOptions

if TYPE_CHECKING:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import _KVCache

    from ...llm_request import LlmRequest
    from ..kv_cache_manager_v2 import KVCacheManagerV2
    from ._fetch_ledger import _FetchRecord

# Staging memory and page-index buffers kept until the process exits, by identity, oldest first;
# only a clean shutdown removes an entry. Each change is one dict operation, atomic under the GIL,
# so lenders of managers on different threads need no lock.
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
    # TODO: staging duplicates the manager's host tier, which backends cannot use in its place: its
    # host pools may move when they resize (mremap), the manager lends none of their pages, and its
    # cold-page codec may hold a row in a form other than the row's device bytes.
    memory = _HostMemory(max(int(nbytes), 1))
    _retain_until_exit(memory)
    return memory


def _checked_layout(manager: KVCacheManagerV2) -> ManagerLayout:
    """The attach's checks of the manager: a v2 manager, no context or pipeline parallelism, no
    lender yet, no recurrent state, no sparse buffer."""
    _manager.require_v2(manager)
    mapping = _manager.mapping(manager)
    # TODO: the layout derives only tensor-parallel shards, so context-parallel ranks, Helix's
    # included, would give different pages the same object_keys.
    cp_size = int(mapping.cp_size)
    if cp_size > 1:
        raise ValueError(
            f"context parallelism (cp_size={cp_size}) is not supported: its ranks would give "
            "different pages the same object_keys"
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
    if _manager.attached_lender(manager) is not None:
        raise ValueError("a lender is already attached to this KV cache manager")
    if _manager.has_lossy_cold_page_codec(manager):
        raise ValueError(
            "the KV cache lender cannot publish original KV bytes by name with a lossy "
            "cold-page codec (quantization_for_cold_page)"
        )
    layout = derive_layout(manager)
    recurrent = [lg for lg, state in enumerate(layout.recurrent) if state]
    if recurrent:
        raise ValueError(
            f"layer groups {recurrent} hold recurrent state, which the lender does not lend"
        )
    # TODO: a sparse layer group's page index can name a host slot, and its locked pages move
    # between GPU and host while a lease holds their indices: a decoding cache offloads its
    # history, and a prefill lock promotes a host page back to the GPU.
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
    # TODO: staging serves no manager built for a draft that reads ahead (Eagle, MTP): a block's
    # name covers only the tokens up to its end, so requests sharing them could hold different
    # draft bytes under one name.
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
    """``ValueError`` for a manager with a KV cache connector."""
    # TODO: a KV cache connector serves a prefix at the first context chunk, from the committed
    # tokens, so it cannot lower a history a windowed fetch moved, and it loads off the manager's
    # stream, unordered with staging copies.
    if _manager.kv_connector(manager) is not None:
        raise ValueError(
            "staging does not serve a manager with a KV cache connector: the connector serves a "
            "request's prefix at its first context chunk, measured from the committed tokens, and "
            "cannot lower a history a fetch moved"
        )


def _attach_staging(
    manager: KVCacheManagerV2, *, scope: bytes, staging: StagingOptions, cls: type | None = None
) -> Staging:
    """``attach_lender`` with the lender class as a parameter (``Staging`` when ``None``), so a
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
    memory = _allocate_retained(sum(sizes.values()))
    base = memory.address
    regions = []
    offset = 0
    for g in layout.pool_groups:
        groups = tuple(lg for lg, pg in enumerate(layout.pool_group_of) if pg == g)
        layout_label = identity.region_label(groups)
        regions.append(
            MemoryRegion(
                layout_label, base + offset, sizes[g], int(layout.page_bytes[g]), slots[g], groups
            )
        )
        offset += sizes[g]
    lender = (cls or Staging)(
        weakref.ref(manager),
        layout,
        identity,
        tuple(regions),
        SlotQueue(slots),
        weakref.ref(memory),
    )
    lender._retain_index_buffer(manager)
    logger.info(
        f"KV cache lender: namespace {identity.namespace.hex()}, staging {offset >> 20} MiB "
        f"in {len(regions)} regions"
    )
    # Installed last, so a failure above leaves nothing attached.
    _manager.install_lender(manager, lender)
    return lender


class _QueuedCopy:
    """Copies queued together, complete once their event says so, and never without one. ``done``
    asks the event at most once per round of the lender's progress, and never again once it
    reported completion."""

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


@dataclass(eq=False)
class _LentBlockEntries:
    """Block entries grouped by local layer group, with aligned ordinals and slot arrays.

    Outer position selects ``layer_group_indices``; each inner position selects one block's
    ordinal, device page index and, once assigned, staging slot index.
    """

    layer_group_indices: list[int]
    block_ordinals: list[np.ndarray]
    device_pages: list[np.ndarray]
    staging_slots: list[np.ndarray] = field(default_factory=list)

    @property
    def num_block_entries(self) -> int:
        return sum(len(o) for o in self.block_ordinals)


class Staging:
    """``Lender`` over one manager, referenced weakly. What its calls do depends on its mode.

    - Serving the manager: a lend past its range check, a poll and a resume_bounds call return
      recyclable leases' slots and grant waiting leases in order before their own work, a mark and a
      release after it.
    - After the manager's reuse reset: as serving, but a lend that raises no ``ValueError`` fails
      at the call.
    - Stopped for good by an error its free or shutdown hook logs: a lend that raises no
      ``ValueError`` fails at the call, a poll fails a lease whose view no poll returned, a mark
      copies nothing, and the shutdown keeps the staging memory; while the manager is served,
      progress fails the waiting leases and returns no slot, and resume_bounds gives a request with
      a cache an empty interval.
    - The manager shut down or gone: a lend that raises no ``ValueError`` fails at the call, a poll
      runs no progress and returns the view only once ready, a mark copies nothing and resume_bounds
      raises ``ValueError``; a stopped lender's poll still fails a lease whose view no poll
      returned.
    """

    def __init__(
        self,
        manager: weakref.ref,
        layout: ManagerLayout,
        identity: Identity,
        regions: tuple[MemoryRegion, ...],
        slots: SlotQueue,
        memory: weakref.ref,
    ) -> None:
        self._manager_ref = manager
        self._layout = layout
        self._identity = identity
        self._regions = tuple(regions)
        self._slot_queue = slots
        # Only the keep list holds the staging memory strongly; this finds it there at shutdown.
        self._memory_ref = memory
        self._region_index_of_pool_group = {g: i for i, g in enumerate(layout.pool_groups)}
        self._any_window = any(w is not None for w in layout.windows)
        # The fetch records and the resume_bounds they give. The ledger asks this lender whether a
        # copy completed through a weak reference, so no reference cycle keeps the lender, or a
        # cache its records hold, past the lender's last reference.
        lender = weakref.ref(self)
        self._ledger = _FetchLedger(layout, lambda copy: lender()._copy_done(copy))
        # Leases waiting for slots, in order.
        self._waiting_for_slots: Deque[_StagingLease] = collections.deque()
        self._leases_holding_slots: list[_StagingLease] = []  # granted, slots not yet returned
        # Open leases, held strongly: an open lease keeps the staging memory at shutdown even
        # once its holder has dropped it.
        self._open_leases: set[_StagingLease] = set()
        # Open staging holds, held strongly too: a dropped hold still keeps the memory.
        self._open_holds: set[_StagingHold] = set()
        self._quarantined: list[SlotRuns] = []  # slots a failed copy may still touch; never reused
        self._shut_down = False
        # Whether the manager reset its reuse state; once it has, nothing is lent by name again.
        self._reuse_was_reset = False
        self._stop_reason: str | None = None  # why lending stopped for good; the first reason stays
        self._index_buffer: object | None = None  # kept until the runtime's shutdown has returned
        self._progress_round = 0  # rounds of progress, so each asks a copy's event at most once

    @property
    def regions(self) -> tuple[MemoryRegion, ...]:
        """See ``Lender.regions``."""
        return self._regions

    def hold_regions(self) -> _StagingHold:
        """See ``Lender.hold_regions``."""
        if self._serving_manager() is None:
            # The memory was freed or kept at the shutdown; a later hold changes neither.
            return _StagingHold(None)
        hold = _StagingHold(self)
        self._open_holds.add(hold)
        return hold

    def lend_read(self, request: LlmRequest, start: int, end: int) -> _StagingLease:
        """See ``Lender.lend_read``."""
        start, end = self._whole_blocks(start, end)
        request_id = int(request.py_request_id)
        manager = self._serving_manager()
        if manager is None:
            return _StagingLease._failed_at_call(self, "read", request_id, _REASON_SHUT_DOWN)
        self._progress()
        if self._stop_reason is not None:
            return _StagingLease._failed_at_call(self, "read", request_id, self._stop_reason)
        if self._reuse_was_reset:
            return _StagingLease._failed_at_call(self, "read", request_id, _REASON_REUSE_RESET)
        kv = _manager.kv_of(manager, request_id)
        if kv is None:
            return _StagingLease._failed_at_call(
                self, "read", request_id, _no_cache_reason(request_id)
            )
        unnamed = self._unnamed_input_refusal(request)
        if unnamed is not None:
            return _StagingLease._failed_at_call(self, "read", request_id, unnamed)
        state = _manager.cache_state(kv)
        if end > state.committed:
            raise ValueError(
                f"the range ends at {end}, past the {state.committed} committed tokens"
            )
        if not state.active:
            return _StagingLease._failed_at_call(self, "read", request_id, _REASON_SUSPENDED)
        # TODO: a publish leaves out window blocks the request's own window has passed, although
        # the prefix tree may still hold their committed pages, so a fetch whose window still keeps
        # such a block finds it missing.
        entries = self._lendable_block_entries(
            manager, state.history, self._block_entries_of_range(manager, kv, start, end)
        )
        counts = self._required_slots_per_pool_group(
            [len(block_ordinals) for block_ordinals in entries.block_ordinals]
        )
        self._check_fits(counts)
        reuse_keys = self._reuse_keys_for(manager, request, kv, entries.block_ordinals)
        lease = _StagingLease(self, "read", request_id, kv, entries, reuse_keys)
        self._admit(lease, counts)
        return lease

    def lend_write(self, request: LlmRequest, start: int, end: int) -> _StagingLease:
        """See ``Lender.lend_write``."""
        start, end = self._whole_blocks(start, end)
        request_id = int(request.py_request_id)
        manager = self._serving_manager()
        if manager is None:
            return _StagingLease._failed_at_call(self, "write", request_id, _REASON_SHUT_DOWN)
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
        # Block entries and their keys come from the layout for a history of ``end``, so every
        # ValueError is raised before the cache changes. The entries are counted from their ranges
        # and checked to fit before any array is built.
        runs = [
            _ranges._needed_ranges(manager, layout, lg, start // tpb, end // tpb, end)
            for lg in range(layout.num_layer_groups)
        ]
        counts = self._required_slots_per_pool_group(
            [sum(e - b for b, e in lg_runs) for lg_runs in runs]
        )
        self._check_fits(counts)
        block_ordinals = [_ranges._ordinals(lg_runs) for lg_runs in runs]
        # The checks above read only the call, the request's prompt and the layout, so on a
        # manager not shut down a wrong call raises whatever this rank holds for the request.
        if self._stop_reason is not None:
            return _StagingLease._failed_at_call(self, "write", request_id, self._stop_reason)
        if self._reuse_was_reset:
            return _StagingLease._failed_at_call(self, "write", request_id, _REASON_REUSE_RESET)
        kv = _manager.kv_of(manager, request_id)
        if kv is None:
            return _StagingLease._failed_at_call(
                self, "write", request_id, _no_cache_reason(request_id)
            )
        unnamed = self._unnamed_input_refusal(request)
        if unnamed is not None:
            return _StagingLease._failed_at_call(self, "write", request_id, unnamed)
        state = _manager.cache_state(kv)
        if start < (state.committed // tpb) * tpb:
            raise ValueError(
                f"the range starts at {start}, inside the committed whole blocks of "
                f"{state.committed} tokens"
            )
        reuse_keys = self._reuse_keys_for(manager, request, kv, block_ordinals)
        if self._any_window and end < state.history:
            raise ValueError(
                f"the range ends at {end}, below the history of {state.history} tokens its "
                "windows keep"
            )
        split = self._bidirectional_split_refusal(request, end)
        if split is not None:
            return _StagingLease._failed_at_call(self, "write", request_id, split)
        if not state.active:
            return _StagingLease._failed_at_call(self, "write", request_id, _REASON_SUSPENDED)
        if self._scratch_reuse_on(kv):
            return _StagingLease._failed_at_call(self, "write", request_id, _REASON_SCRATCH)
        if self._returns_context_outputs(request):
            return _StagingLease._failed_at_call(self, "write", request_id, _REASON_CONTEXT_OUTPUTS)
        # After every ValueError: whether an earlier fetch settled is this rank's own timing.
        unsettled = self._ledger._unsettled_fetch_refusal(request_id, kv)
        if unsettled is not None:
            return _StagingLease._failed_at_call(self, "write", request_id, unsettled)
        # With a window the history moves to ``end``, so windows need pages only for the blocks a
        # history of that length reads.
        position = end if self._any_window else state.history
        refusal = self._history_refusal(manager, state.history, position, end)
        if refusal is None:
            refusal = self._unwritten_pages_refusal(manager, request_id, kv, position)
        if refusal is not None:
            return _StagingLease._failed_at_call(self, "write", request_id, refusal)
        # Read before the grow, which moves a windowed cache's history to ``end``.
        kept = self._ledger._prefix_kept_by_fetch(request_id, kv, start)
        # TODO: the V2 scheduler cannot reclaim the pages a parked fetch grew.
        # TODO: route capture stops reading a request's prepopulated length once it holds the routes
        # below it, until the request finishes, so a later resume leaves positions without routes.
        # TODO: with extra KV tokens the grow locks a page past the history it leaves, which nothing
        # writes, so a later all-reusable windowed lease that leaves that block behind fails.
        # TODO: an exception once the grow resized the cache, in its fill or a step below, can leave
        # no record of a windowed cache's moved history, so resume_bounds counts tokens never fetched as
        # computed, or a fetch record nothing settles, so resume_bounds stays None.
        if not _manager.grow(manager, request, kv, position, end):
            return _StagingLease._failed_at_call(
                self, "write", request_id, f"no free pages to grow the cache to {end} tokens"
            )
        # The cache has grown, and resume_bounds accounts for it whatever this lease's outcome.
        fetch = self._ledger._record_fetch(request_id, kv, start, kept, position, state.committed)
        entries = self._block_entries_of_range(manager, kv, start, end)
        doomed = None
        for lg, lg_ordinals, lg_slots in zip(
            entries.layer_group_indices, entries.block_ordinals, entries.device_pages
        ):
            if doomed is None and np.any(lg_slots < 0):
                missing = lg_ordinals[lg_slots < 0].tolist()
                doomed = f"layer group {lg}: blocks {missing[:8]} have no page"
        lease = _StagingLease(self, "write", request_id, kv, entries, reuse_keys, fetch)
        if doomed is not None:
            # It fails at its first poll, which abandons the fetch.
            lease._first_poll_failure = doomed
            self._open_leases.add(lease)
            return lease
        self._admit(lease, counts)
        return lease

    def resume_bounds(self, request: LlmRequest) -> ResumeBounds | None:
        """See ``Lender.resume_bounds``."""
        request_id = int(request.py_request_id)
        manager = self._serving_manager()
        if manager is None:
            raise ValueError(f"{_no_cache_reason(request_id)}: {_REASON_SHUT_DOWN}")
        self._progress()
        kv = _manager.kv_of(manager, request_id)
        if kv is None:
            raise ValueError(_no_cache_reason(request_id))
        history = _manager.cache_state(kv).history
        if self._stop_reason is not None:
            # Empty: no fetch through a stopped lender counts.
            return ResumeBounds(max_position=0, min_position=max(history, 1))
        return self._ledger._resume_bounds(request, request_id, manager, kv, history)

    def _on_free(self, request_id: int, kv_cache: _KVCache) -> None:
        """Manager hook, after the request's cache left the map: its waiting leases fail and its
        fetch records go. An error in that work, which it logs, stops the lender for good."""
        try:
            if self._serving_manager() is None:
                return
            # Granted leases go on: a read's copy is queued already, and a write's marks copy
            # nothing into pages other than the ones lent.
            self._fail_waiting(int(request_id), "the request was freed")
            self._ledger._drop_records(request_id)
            self._progress()
        except Exception:
            # Logged, not raised into the manager's free, which then closes the cache as usual. The
            # error can leave the records disagreeing with the slots where no caller sees it, so the
            # lender stops; the reason is set first, by an assignment that cannot raise.
            if self._stop_reason is None:
                self._stop_reason = _REASON_STOPPED
            logger.error(f"KV cache lender: freeing request {request_id}: {traceback.format_exc()}")

    def _on_shrink(self, request_id: int, kv_cache: _KVCache) -> None:
        """Manager hook, right after the request's cache may have shrunk in place: delivered entries
        past its blocks stop counting, since the shrink freed their pages. Logs its own errors."""
        try:
            delivered = self._ledger._deliveries_of(int(request_id))
            if (
                self._serving_manager() is not None
                and delivered is not None
                and delivered.kv is kv_cache
            ):
                self._ledger._invalidate_delivered_blocks_past_capacity(delivered, kv_cache)
        except Exception:
            # Logged, not raised into the manager's resize. All the request's delivered entries go:
            # the cache may grow back before resume_bounds finds the freed ones past its blocks.
            self._ledger._drop_deliveries(int(request_id))
            logger.error(
                f"KV cache lender: shrink of request {request_id}: {traceback.format_exc()}"
            )

    def _on_reset(self) -> None:
        """Manager hook after a reuse reset, with every cache closed as an in-place weight update
        ends: object_keys stop meaning the bytes computed, so none is lent by name again; waiting
            leases
        and fetches went at their requests' free; granted publishes finish. Never raises."""
        # TODO: after the manager resets its reuse state the lender lends nothing by name, since
        # ``scope`` is fixed at the attach and nothing identifies the weights it computes with from
        # then on.
        self._reuse_was_reset = True

    def _on_shutdown(self) -> None:
        """Manager hook, first in its shutdown, acting once: waits for the copies whose completion
        it recorded, fails waiting leases, and frees the staging memory unless a lease or a hold is
        open, a slot was lost to a failed copy or the lender stopped. An error in that work, which
        it logs, stops the lender for good."""
        if self._shut_down:
            return
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
            if not self._must_retain_memory():
                # Every copy on it has completed above, and no lease or hold reaches it any more.
                memory = self._memory_ref()
                _stop_retaining(memory)
                memory.free()
            else:
                logger.warning(
                    f"KV cache lender: keeping {sum(p.nbytes for p in self._regions) >> 20} MiB of "
                    f"staging memory until exit: {len(self._open_leases)} leases open, "
                    f"{len(self._open_holds)} staging holds open, "
                    f"{len(self._quarantined)} slot runs lost to failed copies"
                    f"{'' if self._stop_reason is None else ', and ' + self._stop_reason}"
                )
        except Exception:
            # Logged, not raised into the manager's shutdown; the staging memory then stays in
            # the keep list until exit. As in the free hook, the records may disagree with the
            # slots, so the lender stops before the log line, keeping any earlier stop's reason.
            self._shut_down = True
            if self._stop_reason is None:
                self._stop_reason = _REASON_STOPPED_AT_SHUTDOWN
            logger.error(f"KV cache lender: shutting down: {traceback.format_exc()}")

    def _on_caches_closed(self) -> None:
        """Manager hook, once the runtime's shutdown has returned, after which no cache writes into
        the page-index buffer, so the lender lets it go. A shutdown that raises skips this call, and
        the buffer stays until a retried one returns, or until exit."""
        if self._index_buffer is None:
            return
        self._stop_retaining_index_buffer()

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
        if lease._holder_state() is not _HolderState.FAILED:
            if not lease._released or lease._marks_due():
                return False
        # TODO: every lender call asks each released lease's pending copy again, so N calls while P
        # copies pend cost N*P event queries.
        return self._copy_done(lease._copy)

    def _locked_block_masks(self, kv: _KVCache | None, lease: _StagingLease) -> list[np.ndarray]:
        """Per layer group, entries still in the same active cache's window and still locking the
        GPU page lent; a page only held may sit on another tier under the same number."""
        entries = lease._block_entries
        manager = self._manager_ref()
        state = _manager.cache_state(kv) if kv is not None and kv is lease._kv else None
        masks = []
        for lg, block_ordinals, slots in zip(
            entries.layer_group_indices, entries.block_ordinals, entries.device_pages
        ):
            same = np.zeros(len(block_ordinals), dtype=bool)
            if state is not None and state.active:
                pages = _manager.locked_pages(kv, lg)
                inside = block_ordinals < len(pages)
                same[inside] = pages[block_ordinals[inside]] == slots[inside]
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
        """A write target with SWA scratch reuse on, windowed or not: each capacity change keeps its
        history within the scratch rewind of the old capacity, which a fetch's grow breaks, and a
        window's next chunk overwrites scratch slots."""
        return _manager.has_scratch_reuse(kv)

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
        resume_bounds would count, or ``None``: in a cache without a window whose floor follows the
        history, the history stands past ``end``."""
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
        """Why moving the history to ``position`` would have the commit store bytes the request
        never wrote, or ``None``: a window leaving behind a block whose page holds tokens past those
        the request computed keeps that page for the commit (a partial match's copy, a page grown
        early, a row an earlier lease missed), unless a fetch into the cache delivered its row."""
        if not _manager.keeps_stale_pages(manager):
            return None
        computed = self._ledger._computed_tokens(request_id, kv)
        first = computed // int(self._layout.tokens_per_block)  # the first block past them
        delivered = self._ledger._deliveries_of(request_id)
        fetched = (
            delivered.delivered_block_masks
            if delivered is not None and delivered.kv is kv
            else None
        )
        for lg in range(self._layout.num_layer_groups):
            stale_beg, stale_end = _ranges._stale_range(manager, self._layout, lg, position)
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
            lease._copy_state() is _CopyState.WITHOUT_RECORD for lease in self._leases_holding_slots
        )

    def _retain_index_buffer(self, manager: KVCacheManagerV2) -> None:
        """Keep the manager's page-index buffer until the runtime's shutdown has returned: a cache a
        lease or a record holds writes its page indices there as it closes, also once the manager is
        gone."""
        self._index_buffer = _manager.index_buffer(manager)
        _retain_until_exit(self._index_buffer)

    def _stop_retaining_index_buffer(self) -> None:
        """Once the runtime's shutdown has returned: no cache writes into the page-index buffer any
        more."""
        _stop_retaining(self._index_buffer)
        self._index_buffer = None

    def _on_hold_released(self, hold: _StagingHold) -> None:
        """The hold's holder let go; ``KeyError`` for a hold not open, which its guard prevents."""
        self._open_holds.remove(hold)

    def _check_fits(self, counts: Mapping[int, int]) -> None:
        """``ValueError`` if a lease needing ``counts`` entries per pool group can never be
        granted."""
        try:
            self._slot_queue.check_fits_capacity(counts)
        except ValueError as error:
            raise ValueError(
                f"the range needs more staging slots than there are ({error}); ranges of at most "
                "tokens_per_fetch tokens always fit"
            ) from None

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
        if lease._block_entries.num_block_entries == 0:
            # Nothing to stage: ready at the first poll, without slots or a place in line.
            self._grant(lease, None)
            return
        lease._ticket = self._slot_queue.enqueue(counts)
        self._waiting_for_slots.append(lease)
        self._grant_waiting(just_admitted=lease)

    def _grant_waiting(self, just_admitted: _StagingLease | None = None) -> None:
        """Grant the leases in line, strictly in order; ``just_admitted`` was looked up in this
        call."""
        while self._waiting_for_slots:
            head = self._waiting_for_slots[0]
            # TODO: an exception while slots are taken, a MemoryError say, can leave slots taken for
            # no lease, or drop the head's ticket with the head still in line, after which every
            # round of progress raises "not waiting" until the head fails.
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
                    head._first_poll_failure = reason
                else:
                    self._fail(head, reason)
                logger.warning(f"KV cache lender: {reason}")
            except BaseException:
                # An interrupt can leave a copy queued on the slots without its completion.
                head._slot_runs = runs
                self._quarantine(head)
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
        lease._view = self._build_view(lease._block_entries, lease._reuse_keys)
        if runs is None:
            return
        lease._slot_runs = runs
        self._leases_holding_slots.append(lease)
        if lease._kind != "read":
            return
        copy, error = self._queue_copy(self._copy_segments(lease._block_entries), to_staging=True)
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
        # TODO: quarantined slots are never reused, so failed copies erode staging capacity: a lease
        # needing a longer run than any left between lost slots waits in line for good, holding up
        # every lease behind it.
        if lease._slot_runs is not None:
            self._quarantined.append(lease._slot_runs)
        self._leases_holding_slots = [
            held for held in self._leases_holding_slots if held is not lease
        ]
        logger.warning(
            f"KV cache lender: a failed copy took staging slots of request {lease._request_id}"
        )

    # TODO: every lender call checks every lease holding slots, even with no copy pending, so
    # polling H such leases once each costs about H*H checks.
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
        """The lease's holder let go. The record changes first and progress runs last, so an
        unexpected error leaves the fetch abandoned rather than unsettled."""
        self._open_leases.discard(lease)
        if self._serving_manager() is None:
            return
        if lease._slot_state() is _SlotState.WAITING:
            self._fail(lease, "released while waiting for staging slots")
        elif lease._kind == "write" and lease._holder_state() is _HolderState.VIEW_PENDING:
            # Released before anyone saw it ready: no backend wrote, and no marks are due.
            lease._first_poll_failure = None
            self._ledger._abandon_fetch(lease)
        self._progress()

    def _apply_marks(self, lease: _StagingLease, masks: list[np.ndarray]) -> None:
        """Copy the marked entries whose page is still the one lent into the request's pages. The
        fetch stays abandoned until that copy is queued without error; then its entries add to the
        cache's deliveries."""
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
                queued, error = self._queue_copy(segments, to_staging=False)
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
        self._progress()

    def _covers_committed_tail(
        self, kv: _KVCache, entries: _LentBlockEntries, copied: list[np.ndarray]
    ) -> bool:
        """The copy writes the block the committed tokens end inside, whose committed tokens
        resume_bounds counts without a delivery."""
        tpb = int(self._layout.tokens_per_block)
        committed = _manager.cache_state(kv).committed
        if committed % tpb == 0:
            return False
        tail = committed // tpb
        return any(bool(np.any(o[m] == tail)) for o, m in zip(entries.block_ordinals, copied))

    def _block_entries_of_range(
        self, manager: KVCacheManagerV2, kv: _KVCache, start: int, end: int
    ) -> _LentBlockEntries:
        """The blocks of ``[start, end)`` a history of ``end`` reads, per layer group, with their
        device slots (-1 where a block has no page)."""
        layout = self._layout
        tpb = int(layout.tokens_per_block)
        entries = _LentBlockEntries([], [], [])
        for lg in range(layout.num_layer_groups):
            block_ordinals = _ranges._needed_ordinals(
                manager, layout, lg, start // tpb, end // tpb, end
            )
            pages = _manager.pages(kv, lg)
            slots = np.full(len(block_ordinals), -1, dtype=np.int64)
            inside = block_ordinals < len(pages)
            slots[inside] = pages[block_ordinals[inside]]
            entries.layer_group_indices.append(lg)
            entries.block_ordinals.append(block_ordinals)
            entries.device_pages.append(slots)
        return entries

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
            ok = (slots >= 0) & ~((block_ordinals >= stale_beg) & (block_ordinals < stale_end))
            out.layer_group_indices.append(lg)
            out.block_ordinals.append(block_ordinals[ok])
            out.device_pages.append(slots[ok])
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

    def _build_view(self, entries: _LentBlockEntries, reuse_keys: list[np.ndarray]) -> LeaseView:
        groups = []
        for lg, block_ordinals, lg_keys, slots in zip(
            entries.layer_group_indices, entries.block_ordinals, reuse_keys, entries.staging_slots
        ):
            index = self._region_index_of_pool_group[int(self._layout.pool_group_of[lg])]
            region = self._regions[index]
            groups.append(
                LayerGroupBlocks(
                    lg,
                    block_ordinals,
                    object_keys=self._object_keys(lg, lg_keys),
                    addresses=region.address + slots * region.slot_bytes,
                    region_index=index,
                )
            )
        return LeaseView(tuple(groups))

    def _copy_segments(
        self, entries: _LentBlockEntries, mask: list[np.ndarray] | None = None
    ) -> list[list[int]]:
        """``[staging address, device address, bytes]`` for every pool of every row ``mask`` keeps
        (one boolean array per layer group; ``None``: every row), by device pool group. Segments
        that continue each other on both sides are merged."""
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

    # TODO: copies run serially with the forward passes on the execution stream, and a copy call per
    # row and pool, which entries share only where they continue each other in a pool group of one
    # pool, keeps small entries and entries spanning several pools below the host-to-device bandwidth.
    def _queue_copy(
        self, segments: Sequence[Sequence[int]], to_staging: bool
    ) -> tuple[_QueuedCopy | None, str | None]:
        """Queue async copies on the manager's execution stream, then record an event covering
        every copy queued, even after one failed; ``(None, error)`` if it could record none. With
        page-locked staging the CPU does not wait for them; pageable staging may hold the call."""
        # The execution stream orders a copy after the forward passes that wrote its pages and
        # before later work on it, which a page's new owner waits for; the lender does not rely on a
        # release waiting for the copies, and a writer off that stream is not ordered after them.
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
        return _QueuedCopy(event), error


class _LeaseBase:
    """A lease's failure, and the checks of a write's one mark."""

    # Set by each lease class's ``__init__``; the methods below use them.
    _failure: str | None
    _kind: str
    _view_returned: bool
    _marked: bool
    _view: LeaseView | None

    @property
    def failure(self) -> str | None:
        """See ``Lease.failure``."""
        return self._failure

    def _accept_marks(self, masks: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Copies of ``masks`` for the one mark of a write whose view ``poll()`` returned;
        ``ValueError`` unless one bool mask of shape ``(len(blocks),)`` per group collection."""
        if self._kind != "write":
            raise RuntimeError("mark_arrived is for write leases")
        if not self._view_returned:
            raise RuntimeError("mark_arrived before poll() returned the view")
        if self._marked:
            raise RuntimeError("mark_arrived called twice")
        masks, groups = list(masks), self._view.groups
        if len(masks) != len(groups):
            raise ValueError(f"{len(masks)} masks for {len(groups)} groups")
        out = []
        for blocks, mask in zip(groups, masks):
            mask = np.asarray(mask)
            if mask.dtype != np.bool_ or mask.shape != (len(blocks),):
                raise ValueError(
                    f"layer group {blocks.layer_group_index}: the mask must be bool of shape "
                    f"({len(blocks)},), got {mask.dtype} {mask.shape}"
                )
            out.append(mask.copy())
        self._marked = True
        return out


class _LeaseState(enum.Enum):
    """A staging lease's state, which ``_StagingLease._state`` derives from its fields. ``release``
    moves any state to ``RELEASED``. Its slots and its copy vary apart from the state. The slots:
    none, waiting in order in ``Staging._waiting_for_slots``, or held in
    ``Staging._leases_holding_slots`` until ``_recycle`` returns them or ``_quarantine`` keeps them
    from reuse. The copy (``_copy``): none, ``_COPY_WITHOUT_RECORD``, or a ``_QueuedCopy``, pending
    or done. Three axes each derive from their own fields alone, released or not: ``_HolderState``,
    whether the lease failed and, if not, whether a poll returned its view; ``_SlotState``, whether
    it waits in line for slots; ``_CopyState``, whether its copy is ``_COPY_WITHOUT_RECORD``.
    ``Staging._recyclable`` reads the first, ``Staging._on_lease_released`` the first two and
    ``Staging._slots_quarantined`` the third."""

    # In line for slots: _grant_waiting grants it in turn, or dooms a fresh write whose grant
    # raised; _fail fails it.
    WAITING = "waiting"
    # Without slots or a view: its first poll fails it, as does the shutdown.
    FAILS_AT_FIRST_POLL = "fails_at_first_poll"
    # Its view and any slots: poll makes it ready, a read once its copy into them has landed, and
    # fails a read whose copy is lost.
    GRANTED = "granted"
    # Poll returned its view: a backend may use its slots until release; mark_arrived marks a write.
    READY = "ready"
    # A write mark_arrived marked: its marked entries still lent go to the request's pages.
    MARKED = "marked"
    # Failed at the call, in line, at its grant, at a poll or at the shutdown.
    FAILED = "failed"
    # Let go: _recycle returns its slots once neither a backend nor a copy can reach them.
    RELEASED = "released"


class _HolderState(enum.Enum):
    """Whether a staging lease failed and, if not, whether a poll returned its view, which
    ``_StagingLease._holder_state`` derives from ``_failure`` and ``_view_returned`` alone, released
    or not."""

    # Its failure is set.
    FAILED = "failed"
    # No failure, and no poll has returned its view.
    VIEW_PENDING = "view_pending"
    # No failure, and a poll returned its view.
    VIEW_RETURNED = "view_returned"


class _SlotState(enum.Enum):
    """Whether a staging lease waits in line for slots, which ``_StagingLease._slot_state`` derives
    from ``_ticket`` alone."""

    # A ticket: in line for slots.
    WAITING = "waiting"
    # No ticket.
    NOT_WAITING = "not_waiting"


class _CopyState(enum.Enum):
    """Whether a staging lease's copy is ``_COPY_WITHOUT_RECORD``, which
    ``_StagingLease._copy_state`` derives from ``_copy`` alone."""

    # _COPY_WITHOUT_RECORD: a copy that may be queued with no record of its completion.
    WITHOUT_RECORD = "without_record"
    # No copy, or one queued with the record of its completion.
    NONE_OR_RECORDED = "none_or_recorded"


class _StagingLease(_LeaseBase):
    """A staging lease; holds its lender weakly and has no finalizer. Backends may read its view's
    arrays on their own threads until release, and never call it."""

    def __init__(
        self,
        lender: Staging,
        kind: str,
        request_id: int,
        kv: _KVCache | None = None,
        entries: _LentBlockEntries | None = None,
        reuse_keys: list[np.ndarray] | None = None,
        fetch: _FetchRecord | None = None,
    ) -> None:
        self._lender = weakref.ref(lender)
        self._kind = kind
        self._request_id = request_id
        self._kv = kv
        self._block_entries = entries
        self._reuse_keys = reuse_keys
        self._fetch = fetch
        self._first_poll_failure: str | None = None  # why it fails at its first poll
        self._ticket: int | None = None  # its place in line while it waits for slots
        # The granted slots; _recycle returns them and leaves this field set.
        self._slot_runs: SlotRuns | None = None
        self._view: LeaseView | None = None
        self._copy: _QueuedCopy | None = None
        self._view_returned = False
        self._released = False
        self._marked = False
        self._failure: str | None = None

    @classmethod
    def _failed_at_call(
        cls, lender: Staging, kind: str, request_id: int, reason: str
    ) -> _StagingLease:
        """A lease failed at the call; open until released, like any other."""
        lease = cls(lender, kind, request_id)
        lease._set_failure(reason)
        if not lender._shut_down:
            lender._open_leases.add(lease)
        return lease

    def _set_failure(self, reason: str) -> None:
        self._failure = reason
        self._first_poll_failure = None

    def _state(self) -> _LeaseState:
        """Its state, from the first of these fields that is set; see ``_LeaseState``."""
        if self._released:
            return _LeaseState.RELEASED
        if self._failure is not None:
            return _LeaseState.FAILED
        if self._first_poll_failure is not None:
            return _LeaseState.FAILS_AT_FIRST_POLL
        if self._ticket is not None:
            return _LeaseState.WAITING
        if self._marked:
            return _LeaseState.MARKED
        return _LeaseState.READY if self._view_returned else _LeaseState.GRANTED

    def _holder_state(self) -> _HolderState:
        """See ``_HolderState``."""
        if self._failure is not None:
            return _HolderState.FAILED
        return _HolderState.VIEW_RETURNED if self._view_returned else _HolderState.VIEW_PENDING

    def _slot_state(self) -> _SlotState:
        """See ``_SlotState``."""
        return _SlotState.WAITING if self._ticket is not None else _SlotState.NOT_WAITING

    def _copy_state(self) -> _CopyState:
        """See ``_CopyState``."""
        if self._copy is _COPY_WITHOUT_RECORD:
            return _CopyState.WITHOUT_RECORD
        return _CopyState.NONE_OR_RECORDED

    def _marks_due(self) -> bool:
        """A write whose view a poll returned and that has no mark: ``mark_arrived`` may still come
        after release, and the copy it queues reads the lease's slots."""
        return self._kind == "write" and self._view_returned and not self._marked

    def __repr__(self) -> str:
        return f"<staging {self._kind} lease of request {self._request_id}: {self._state().value}>"

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
        lender._progress()
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

    def mark_arrived(self, masks: Sequence[np.ndarray]) -> None:
        """See ``Lease.mark_arrived``."""
        masks = self._accept_marks(masks)
        lender = self._lender()
        if (
            lender is not None
            and lender._serving_manager() is not None
            and lender._stop_reason is None
        ):
            lender._apply_marks(self, masks)

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


class _StagingHold:
    """A hold on the staging memory; holds its lender weakly and has no finalizer. Without a lender
    it is inert."""

    def __init__(self, lender: Staging | None) -> None:
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
