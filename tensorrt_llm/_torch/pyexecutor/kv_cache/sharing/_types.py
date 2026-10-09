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
"""The public types and protocols of the lender. Numpy and the standard library only, so any
side can import them without a live KV cache manager."""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from ...llm_request import LlmRequest

# A row's name: a fixed-length opaque key.
_NAME_BYTES = 54


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
    """The staging size, in whole fetches.

    The lender computes the bytes of one fetch from the layer groups and their windows, and a range
    of at most ``fetch_tokens`` tokens always fits. A capacity budget, not concurrency: a lease
    takes one contiguous run of slots per pool group and leases that need slots are granted strictly
    in order, so holes that released leases leave can make a lease wait although enough slots are
    free in total; a lease with no rows is ready at its first poll, with no place in line.

    Attributes:
        fetch_tokens: The tokens of one fetch.
        max_fetches: How many fetches staging holds.
        max_bytes: A cap on the staging bytes, or ``None``; ``attach_staging`` raises
            ``ValueError`` if it is below one fetch.

    Raises:
        TypeError: A field is not an integer (``bool`` included).
        ValueError: A field is not positive.
    """

    fetch_tokens: int
    max_fetches: int = 1
    max_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fetch_tokens", _integer("fetch_tokens", self.fetch_tokens))
        object.__setattr__(self, "max_fetches", _integer("max_fetches", self.max_fetches))
        if self.max_bytes is not None:
            object.__setattr__(self, "max_bytes", _integer("max_bytes", self.max_bytes))


@dataclass(frozen=True)
class Part:
    """One device pool group's staging host region, to register as ``(address, nbytes)``.

    ``StagingLender.parts`` states what a backend may rely on about the regions and when it
    registers and deregisters them.

    Attributes:
        name: Equal on instances laid out alike.
        address: The region's fixed host address.
        nbytes: The region's size in bytes.
        slot_bytes: The bytes of one slot, which holds one row.
        slots: The slots, back to back from ``address``.
    """

    name: str
    address: int
    nbytes: int
    slot_bytes: int
    slots: int


@dataclass(frozen=True, eq=False)
class GroupRun:
    """One layer group's rows, with read-only arrays: row ``i`` is block ``ordinals[i]``.

    Equal names mean interchangeable bytes within the limits ``attach_staging`` states, so a name is
    usable as a store key. Their format is not API, but their width, 54 bytes, is. Consumers store
    names and compare them for equality, and nothing else. A change to the format or to the layout
    behind it makes objects stored under the old one miss; they never hit wrongly.

    Attributes:
        layer_group: The manager's local index of the layer group.
        ordinals: ``int64 (n,)``, each row's block index; block ``b`` covers tokens
            ``[b * tokens_per_block, (b + 1) * tokens_per_block)``.
        names: ``uint8 (n, 54)``, each row's opaque name (``names[i].tobytes()``).
        addresses: ``int64 (n,)``; row ``i``'s slot is the part's ``slot_bytes`` bytes at
            ``addresses[i]``.
        part: The index of the row's part in ``StagingLender.parts``.
    """

    layer_group: int
    ordinals: np.ndarray
    names: Optional[np.ndarray] = None
    addresses: Optional[np.ndarray] = None
    part: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_group", _integer("layer_group", self.layer_group, 0))
        ordinals = _freeze(np.ascontiguousarray(self.ordinals, dtype=np.int64))
        if ordinals.ndim != 1:
            raise ValueError(f"ordinals must be one-dimensional, got shape {ordinals.shape}")
        object.__setattr__(self, "ordinals", ordinals)
        placed = (self.names is not None, self.addresses is not None, self.part is not None)
        if any(placed) and not all(placed):
            raise ValueError("names, addresses and part are all given or all None")
        if not all(placed):
            return
        names = np.asarray(self.names)
        if names.dtype != np.uint8 or names.shape != (len(ordinals), _NAME_BYTES):
            raise ValueError(
                f"names must be uint8 ({len(ordinals)}, {_NAME_BYTES}), "
                f"got {names.dtype} {names.shape}"
            )
        addresses = np.asarray(self.addresses)
        if not np.issubdtype(addresses.dtype, np.integer) or addresses.shape != ordinals.shape:
            raise ValueError(
                f"addresses must be integers of shape {ordinals.shape}, "
                f"got {addresses.dtype} {addresses.shape}"
            )
        object.__setattr__(self, "names", _freeze(np.ascontiguousarray(names)))
        object.__setattr__(
            self, "addresses", _freeze(np.ascontiguousarray(addresses, dtype=np.int64))
        )
        object.__setattr__(self, "part", _integer("part", self.part, 0))

    def __len__(self) -> int:
        return int(self.ordinals.shape[0])

    def select(self, mask: np.ndarray) -> GroupRun:
        """The rows where the boolean ``mask`` (one entry per row) is true, in order.

        Args:
            mask: One boolean per row, ``True`` for each row to keep.

        Returns:
            A run of the same layer group and part that holds the kept rows, in order.

        Raises:
            ValueError: ``mask`` is not boolean with one entry per row.
        """
        mask = np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != self.ordinals.shape:
            raise ValueError(
                f"mask must be bool of shape {self.ordinals.shape}, got {mask.dtype} {mask.shape}"
            )
        return GroupRun(
            self.layer_group,
            self.ordinals[mask],
            None if self.names is None else self.names[mask],
            None if self.addresses is None else self.addresses[mask],
            self.part,
        )


@dataclass(frozen=True, eq=False)
class RegionView:
    """What a ready lease lends: at most one run per layer group.

    Backends may read its arrays on their own threads until the lease is released, and touch the
    memory they point to only until then.

    Attributes:
        runs: The runs, at most one per layer group.

    Raises:
        TypeError: A run is not a ``GroupRun``.
        ValueError: A layer group appears in two runs.
    """

    runs: Tuple[GroupRun, ...]

    def __post_init__(self) -> None:
        runs = tuple(self.runs)
        seen = set()
        for run in runs:
            if not isinstance(run, GroupRun):
                raise TypeError(f"runs hold GroupRun, got {type(run).__name__}")
            if run.layer_group in seen:
                raise ValueError(f"layer group {run.layer_group} appears twice")
            seen.add(run.layer_group)
        object.__setattr__(self, "runs", runs)

    @property
    def num_rows(self) -> int:
        """Rows over all runs."""
        return sum(len(run) for run in self.runs)

    def row_masks(self, value: bool = False) -> Tuple[np.ndarray, ...]:
        """Make one writable boolean mask per run, filled with ``value``.

        Args:
            value: What every row's entry starts as.

        Returns:
            One mask per run, in run order: the shape ``mark_arrived`` takes.
        """
        return tuple(np.full(len(run), bool(value), dtype=bool) for run in self.runs)


class Readiness(NamedTuple):
    """Where a request may resume after a fetch.

    The request may resume at a ``p`` with ``restart_floor <= p <= usable_until``, with ``p`` no
    lower than its context position. The interval is this rank's own and covers only this manager's
    blocks. With several ranks the waiter takes the largest floor and the smallest end, and likewise
    over a request's target manager and its joint-reuse draft pool, which shares the request's
    cursor: the caller fetches through the draft pool's own lender too and resumes where both
    intervals allow (the smaller ``usable_until``, the larger ``restart_floor``), so never below the
    draft pool's history. An empty interval means drop the request's cache in every manager it
    fetched into, a target and its joint-reuse draft pool alike, compute from 0 and do not fetch
    again. Below the floor a resume may read blocks a sliding window has released, so it cannot
    continue there. Under a block reuse policy other than all-reusable the floor is also at least
    the request's history, since the manager's context update never moves a history back. The same
    holds in a joint-reuse draft pool, whose context resize sets the capacity from the chunk it runs
    and raises where that capacity is below the request's history. Where the request's multimodal
    data sets ``mm_bidirectional_blocks``, no position in the interval lies strictly inside a run of
    multimodal tokens, which the scheduler keeps within one context chunk: ``usable_until`` stops at
    the start of a run it would fall inside and ``restart_floor`` rises to the end of the last run
    below it, so the interval can be empty, and where the context position lies strictly inside a
    run it can end below that position, which the caller treats as empty too.

    ``usable_until`` need not be a multiple of ``tokens_per_block``. It reaches the request's
    prompt length only where the request computed its whole prompt itself: a fetch ends before the
    last prompt token, which the request computes for its logits.

    From ``lend_write`` until the request resumes, the caller keeps it among the executor's active
    requests, as the disaggregated transfer-in-progress state does, keeps it from being scheduled,
    and it resumes at ``p`` with ``py_connector_served_position`` set to ``p``, the cache's history
    raised to ``p`` where below it and, once it is back in its context state, its prepopulated
    length and context position moved to ``p`` together by ``set_prepopulated_prompt_len(p,
    tokens_per_block)`` with the manager's ``tokens_per_block``, its context chunk first set to span
    to the prompt's end: the steps a KV cache connector takes to skip a request past the prefix it
    served, here on any context chunk. The request resumes only at a ``p`` in the interval no lower
    than its context position: like a connector's skip, the step moves it only forward, and below
    that position it can leave a chunk that the next scheduling pass makes negative, which raises
    out of the scheduler. Its next chunk is then a first context chunk, which the manager settles at
    ``p``, and a scheduling pass that tries and fails to admit it drops the request's caches in
    every manager and rewinds it to 0, as for any first context chunk.
    ``set_prepopulated_prompt_len(0, tokens_per_block)`` leaves the context position where it is: a
    resume at 0 takes no step where the context position is 0, and elsewhere means dropping the
    request's cache in every manager it fetched into and computing from 0. Without
    ``set_prepopulated_prompt_len`` a request past its first context chunk stays at its own context
    position, and a context position moved by itself leaves the prepopulated length behind, which
    the executor reads for context logits and a pipelined cache transfer's first chunk. With
    ``enable_return_routed_experts``, the caller fetches into a request that asks for routed experts
    only before its first context step runs, so never after a recompute pause: route capture stops
    reading a request's prepopulated length once it holds the routes below it, at the first context
    step already when that length is 0, and keeps that state until the request finishes, also across
    a recompute pause, so a later resume can leave the skipped positions without routes, and the
    request's completion then raises out of the executor loop. The executor's pool rebalance
    suspends the active requests' caches and its own CUDA-graph padding dummies, no others, so a
    request taken out of the active requests while its cache is active stops the executor loop. Each
    response reports as ``cached_tokens`` where the request's first context step started, a count
    kept until a recompute pause, so a fetch the request resumes from after that step adds nothing
    to it.

    Attributes:
        usable_until: The last position the request may resume at.
        restart_floor: The first position the request may resume at, unless its context
            position is higher.
    """

    usable_until: int
    restart_floor: int


@runtime_checkable
class Lease(Protocol):
    """One lent range: poll until the view or ``failure``, mark a write once, release.

    Its methods run only on the manager's thread; a backend's own threads read the view, access the
    memory it points to, call no lease or lender method and tell the holder through their own
    channel when they are done. Without that signal the lease stays open. After the manager's
    shutdown ``poll``, ``mark_arrived`` and ``release`` only end records.

    Caller obligations:
        - Poll every open lease each iteration of the executor loop, also when no new request
          arrives: all progress happens inside lender calls. Once no request is live or waiting, the
          executor waits for a new request, with no timeout under MPI and for up to 1200 s under
          Ray, so the caller keeps that wait from blocking while a lease is open or a backend has
          work for the holder, as the executor does itself while a KV cache connector's transfers
          pend.
        - Release every lease, failed ones too, once the backend has let go of the memory. An
          unreleased staging lease keeps the staging memory past the manager's shutdown until the
          process exits.
        - A lease's outcome is this rank's own: slots, copies, free pages and a pipeline stage's
          windows are per rank, so ranks lending alike can end differently. The caller combines
          every rank's outcome and decides for all ranks; the lender runs no collective.
        - Failure is final. A lease already failed when ``lend_read`` or ``lend_write`` returns
          changed nothing in the request's cache, so the caller computes locally or tries later.
        - An exception a lender, lease or hold call raises for a reason its docstring does not
          list, a ``MemoryError`` say, is not recovered from: it can leave the lender's records,
          and a staging lender's of other requests too, disagreeing with the caches and the
          staging slots. The caller then lends, polls, marks and asks ``readiness`` through that
          lender no more, only releases its leases and holds, and treats each request parked for
          a fetch through it as it does an empty interval. The lender may keep caches, and with
          them the manager's device pools, and the staging memory until the process exits.
    """

    def poll(self) -> Optional[RegionView]:
        """Does pending work and returns the view once the lease is ready.

        A staging read is ready once its copy into the slots completed, and a staging write at the
        first poll after its slots were granted.

        Returns:
            The view once ready, the same object every time; ``None`` while pending, and for good
            once failed.

        Raises:
            RuntimeError: The lease was released.
        """
        ...

    @property
    def failure(self) -> Optional[str]:
        """Why the lease will never be ready, once that is known; for logs only."""
        ...

    def mark_arrived(self, masks: Sequence[np.ndarray]) -> None:
        """Marks the rows of a write lease that arrived whole.

        Write leases only, once, after ``poll()`` returned the view, before or after release.
        Required for staging writes: only marked rows reach the request's pages, copied in one
        batch that every forward pass queued after it waits for, about the fetch's bytes over the
        host-to-device bandwidth, so callers split long fetches into several leases, within the
        limit ``StagingLender`` states for a sliding window with speculative decoding's extra KV
        tokens. Marked rows are copied only where the request's active cache still locks the lent
        GPU page inside its window; after the request exits, the holder still marks and releases,
        and nothing is copied into freed pages. A staging write's slots return only after release,
        ``mark_arrived`` and the copy's completion. A write released before its view was returned
        abandons its fetch and needs no mark.

        Args:
            masks: One boolean array per run of the view, ``True`` where the row arrived whole;
                ``RegionView.row_masks()`` makes all-``False`` ones.

        Raises:
            RuntimeError: A read lease, a call before ``poll()`` returned the view, or a second
                call.
            ValueError: A wrong number, shape or dtype of masks; nothing is recorded.
        """
        ...

    def release(self) -> None:
        """The backend has stopped touching the lent memory.

        Required for every lease, failed ones too; legal in every state, and later calls do
        nothing.
        """
        ...


@runtime_checkable
class PartsHold(Protocol):
    """A staging backend's hold on the staging memory.

    A hold still open at the manager's shutdown keeps the staging memory until the process exits.
    Take one with ``StagingLender.hold_parts()`` on the manager's thread before registering
    ``StagingLender.parts``; deregister the parts before the manager shuts down, and release it on
    the manager's thread once deregistration is confirmed. A backend that fails after registering
    and cannot confirm its deregistration keeps its hold, so the memory stays until exit. The lender
    holds it, so dropping it unreleased keeps the memory.
    """

    def release(self) -> None:
        """The backend has deregistered the parts and cannot reach them.

        Runs only on the manager's thread, like every lender and lease method; legal in every
        state, and later calls do nothing.
        """
        ...


# TODO: names exist only in ready views, so a flow that looks up remote hits before it reserves
# pages for them cannot name blocks without lending them.
@runtime_checkable
class StagingLender(Protocol):
    """Relays whole blocks between a request's device pages and host staging slots.

    Leases that need slots are granted first come, first served; a lease with no rows is ready at
    its first poll, with no place in line. Lender and lease methods run only on the manager's
    thread; a backend's own threads read a view, access the memory it points to, call no lease or
    lender method, and tell the holder through their own channel. Backends never lend and never call
    a lease.

    Copies queue on the manager's execution stream: on the GPU they run serially with the forward
    passes, so a step that queues copies takes about their time longer. With page-locked staging and
    the fresh-page fill off, no lend, poll, mark or readiness call waits for them on the CPU; only
    the manager's shutdown does. Under confidential computing staging is pageable, and a copy can
    hold the call that queues it until the stream reaches it. A call asks each copy's event at most
    once, but every call asks again the pending copy of each released lease, so N calls while P such
    copies pend cost about N * P queries. Every call also checks each lease holding slots, even with
    no copy pending, so polling H such leases once each costs about H * H checks.

    A pool rebalance by the executor suspends its active requests' caches, which a parked request
    stays among, and moves their pages, lent ones included: copies already queued complete first, a
    read still waiting for slots fails at its grant where its pages moved, and a write marked after
    the rebalance copies only the rows whose pages stayed, which are all ``readiness`` counts.

    Caller obligations:
        - A staging backend, on the manager's thread, takes a hold with ``hold_parts()`` and
          registers ``parts`` after the attach and before it touches a lease; it deregisters before
          the manager shuts down, and releases the hold once deregistration is confirmed.
        - Each iteration of the executor loop the holder polls every open lease and the waiter asks
          ``readiness`` for every waiting request, also when no new request arrives.
        - At shutdown the executor loop stops; the staging backends stop, deregister and release
          their holds; the holder marks and releases what they give back; the manager shuts down
          last.
        - A request whose one-model draft has a joint-reuse pool of its own shares its context
          cursor with that pool: the caller fetches the same range into both managers, each
          through its own lender, resumes within both intervals and publishes both. A fetch into
          one alone leaves the other without the prefix while the shared cursor moves past it.

    First-version limits:
        - Staging needs block reuse, one lender per manager, no context or pipeline parallelism,
          recurrent state, sparse buffers or KV cache connector, and no one-model draft that reads
          prompt tokens past a position, since a block's name covers only the tokens up to the
          block's end; one-model DraftTarget, whose read-ahead upstream has not established,
          counts as reading none (``attach_staging``).
        - A publish leaves out the window blocks the request's own window has passed, although the
          manager's prefix tree may still hold their committed pages, so a fetch whose window still
          keeps such a block finds its row missing (``lend_read``).
        - DeepSeek-V4 keeps every window the draft length wider under any speculative decoding, and
          a fetch asks for the rows of that margin too: at 128 tokens per block and a draft length
          of 2 or more, a publish from a request whose history stands at least the draft length less
          one token past the fetch's end leaves out the margin's row, so the fetch finds it missing.
        - DeepSeek-V4's model defaults turn SWA scratch reuse on, with which a fetch fails at the
          call (``lend_write``), so a fetch there needs
          ``kv_cache_config.enable_swa_scratch_reuse=False``.
        - Under the all-reusable block reuse policy a fetch fails at the call where a window leaves
          behind, at the fetch's end, a block whose page holds tokens past the cache's history: the
          block keeps its page until the commit stores it whole, the request never wrote those
          tokens (the rest of a block its local match copied from another request's page, or a page
          grown before the fetch), and the manager has no way to drop one block's page in one layer
          group. With partial reuse on, local matches often end inside a block, so on a model with a
          sliding window such a request computes from its local match rather than fetch past the
          window (``lend_write``).
        - With speculative decoding's extra KV tokens (``num_extra_kv_tokens``, the draft length
          less one under one-model speculative decoding), each lease's grow gives the block holding
          them a page past the history the lease leaves, and nothing writes it. So under the
          all-reusable policy, where the manager keeps a sliding window of ``W`` tokens, a later
          consecutive lease spanning at least ``W + tokens_per_block - 1`` tokens leaves that block
          behind and fails at the call: a caller splitting such a fetch keeps later leases shorter,
          or the request computes the rest (``lend_write``).
        - The pages a fetch grows are outside the V2 scheduler's reach: it neither evicts, pauses
          nor preempts a parked request. Its deadlock check counts no pass while any request is in
          a disaggregated transfer state, as one parked in the transfer-in-progress state is, and
          otherwise raises "V2 scheduler deadlock" after 1000 passes in a row that schedule and
          reclaim nothing while a context or generation request waits. So while requests are
          parked in another state, a request that cannot be admitted, resume or grow can end in
          that deadlock, unless the caller keeps room for the running requests in each pool group:
          the pages parked requests' caches lock, the other pages the scheduler cannot reclaim,
          and, for the generation request that needs the most, the pages of its cache within its
          windows, which a resume locks all at once, and the pages its next step adds stay within
          ``max_util_for_resume`` of the group's GPU pages. A share of each pool group for the
          parked fetches alone does not ensure it, with or without a cache tier below the GPU. The
          manager's ``get_page_indices_by_layer_group`` lists a request's pages per layer group, one
          beam's (under beam search each further beam also locks a page of its own for every block
          not wholly inside the prompt), and the runtime's ``impl.pool_group_descs`` give each pool
          group's number of GPU pages and its layer groups.
        - Under attention data parallelism without a cache transceiver, the executor counts parked
          requests as schedulable, so a rank whose active requests are all parked, at its cap of
          active requests or without pages for a padding dummy, schedules nothing, and its empty
          batch holds every rank's forward until a fetch settles. The caller keeps a rank from
          parking all its active requests, or accepts the stall.
        - With ``enable_return_routed_experts``, the caller fetches into a request that asks for
          routed experts only before its first context step runs, so never after a recompute pause.
          Route capture stops reading a request's prepopulated length once it holds the routes below
          it, at the first context step already when that length is 0, and keeps that state until
          the request finishes, also across a recompute pause. So a later resume can leave the
          skipped positions without routes, and the request's completion then raises out of the
          executor loop.
        - Names exist only in ready views: nothing names blocks without lending them, so a caller
          cannot ask a source how far it holds a prefix before a fetch grows the cache. A publish
          lends a sliding-window layer group's rows only for the window at its ``end``
          (``lend_read``), and a fetch into a cache with a sliding window needs the rows of the
          window at the history it leaves (``lend_write``). So a request forking from a published
          prompt at an earlier block, once the window at the publisher's history has released
          blocks, or one whose fetch ends past the published end, as a longer next turn's does, once
          the window at the fetch's end has released blocks, finds rows missing and computes from 0,
          unless the publishes of its prefix together lend the window at the fetch's end: a publish
          ending there does if its publisher's own window had passed none of that window's blocks
          (the window-gap limit above). While no window has released a block at a publisher's
          history, its publish lends its whole range, so a fork finds every row; a fetch past the
          published end, where no window has released a block at its own end either, then misses
          only the rows past the published end and resumes at the first of them where the request
          may resume below its history, which is under the all-reusable policy outside a joint-reuse
          draft pool, and elsewhere computes from 0.
        - Lending stops for good once the manager resets its reuse state, after which its names no
          longer say which bytes it computes (``attach_staging``).
        - Copies issue at most one copy call per row and pool. Only in a pool group of one pool do
          rows whose device pages and staging slots both continue merge into one call, so small rows
          on scattered pages and rows spread over several pools run below the host-to-device
          bandwidth.
        - A writer of recycled pages off the manager's stream is not ordered after staging copies,
          which come only before later work on that stream, the work a page's new owner waits for:
          such a writer orders itself after them. An integrator adding one makes it wait on the
          manager's stream first, as the disaggregated receive does. With a KV cache connector
          refused, the manager's only such writer is the fresh-page fill
          (``TRTLLM_KV_FRESH_PAGE_FILL``, a diagnostic off by default), which synchronizes the
          device before it fills pages and again after: it overwrites no page a queued copy still
          reads, and a ``lend_write`` that grows the cache then waits on the CPU for the queued
          staging copies.
        - Every lease hashes the request's whole prefix again from block 0, so a lease late in a
          long prompt costs time in proportion to the prompt.
        - A slot lost to a failed copy is never reused: a slot is lost when a copy into or out of it
          may have been queued without its completion recorded, because that completion could not be
          recorded or a read's grant failed while queuing the copy. Such failures erode staging
          capacity: a lease that needs, in some pool group, a longer run of slots than the longest
          one without a lost slot there waits in line for good without failing, and holds up every
          lease behind it until it is released or its request is freed. They also keep the staging
          memory until the process exits.
    """

    @property
    def parts(self) -> Tuple[Part, ...]:
        """The staging host regions, one per device pool group.

        Each part stays at a fixed address for the lender's life and can be registered once,
        after the attach and before the first lease is used; part names are equal on instances
        laid out alike. Nothing more is promised: not one allocation, not an order in memory, not
        that parts are back to back. A backend that can register only one region sorts the parts by
        address, checks that each part ends where the next begins and registers their span; if they
        are not back to back, it raises at construction and does not start.

        A backend deregisters them before the manager shuts down and releases its hold once
        deregistration is confirmed (``PartsHold``). The manager's shutdown frees them, once, unless
        an unreleased lease (failed ones included), an unreleased hold (dropped ones included) or a
        slot lost to a failed copy keeps them until the process exits.
        """
        ...

    def hold_parts(self) -> PartsHold:
        """Returns a new hold on the staging memory.

        Take it on the manager's thread before registering ``parts``. After the manager's
        shutdown the hold is inert: the memory was freed or kept then.

        Returns:
            The hold, open until its ``release``.
        """
        ...

    def lend_read(self, request: LlmRequest, start: int, end: int) -> Lease:
        """Publishes the committed blocks ``[start, end)``: a copy of them into staging slots.

        Committed tokens are the request's prompt tokens in the manager's prefix-reuse tree: the
        ones it reused, then the ones the forward computed, committed after each context chunk
        (only after the last one under a reuse policy other than all-reusable) and never past the
        end of the context. Generated tokens never commit.

        When its slots are granted, at once or after waiting in line, the lease queues a copy of
        the request's pages into them; from then on the request may exit, and a copy already
        queued completes. A sliding-window layer group lends only the blocks a history of ``end``
        reads that its window still keeps, and a block without a page is left out. Once ``poll()``
        returns the view, the backend reads those slots and no others, until release.

        Its outcome is this rank's own. It fails at the call on no cache, a suspended cache, a
        request with multimodal data without digests or with encoder input, which its names do not
        cover, a manager shut down or one that reset its reuse state, or a copy that could not be
        queued; a read that waited for slots fails if its cache was freed, suspended or changed
        meanwhile. The caller combines every rank's outcome.

        Args:
            request: The request whose blocks are read.
            start: The first token, a multiple of ``tokens_per_block``.
            end: The end token, a multiple of ``tokens_per_block``, at most the committed tokens.

        Returns:
            The lease, possibly failed already.

        Raises:
            ValueError: Bounds that are negative, reversed or not whole blocks, an end past the
                committed tokens, or a range needing more slots than a part has.
        """
        ...

    def lend_write(self, request: LlmRequest, start: int, end: int) -> Lease:
        """Fetches into blocks ``[start, end)``: the cache grown to ``end``, and empty slots.

        The lease takes slots as ``lend_read`` does; the backend fills them, and the holder marks
        the rows that arrived whole and releases. A sliding-window layer group lends only the blocks
        a history of ``end`` reads. Fetches into one cache add up until the request commits past
        them: consecutive leases count as one fetch of their whole range, and an abandoned lease
        drops only its own rows, unless its failed copy covered the block the committed tokens end
        inside, which empties the interval (``readiness``). Whether a fetch settled is this rank's
        own, so a caller splitting a fetch lends the next segment only once ``readiness`` is not
        None on every rank, after combining their outcomes. With a sliding window each lease moves
        the history to its end at the call: where no window has released blocks at that end, the
        block reuse policy is all-reusable and the manager is no joint-reuse draft pool, the request
        may still resume below it, up to where the rows delivered reach, and otherwise
        ``restart_floor`` rises to that end, where the request resumes if the rows delivered reach
        it; else, as after a later lease that is abandoned or misses rows, the interval is empty and
        the request computes from 0. Without a sliding window the history stays where it was:
        ``restart_floor`` is that history under a block reuse policy other than all-reusable and in
        a joint-reuse draft pool, and otherwise the lower of that history and the lowest start of a
        fetch into the cache that was not abandoned. The committed tokens still count, and so do the
        other tokens the request had computed below ``start``; past ``start`` the fetch may
        overwrite them. Where a copy ``mark_arrived`` queued fails over the block the committed
        tokens end inside, the interval is empty from then on (``readiness``). The lender takes the
        tokens below the cache's history as computed: the caller fetches only into a request whose
        pages hold what its history covers, not one whose history runs ahead of its data, such as a
        disaggregated generation request before its transfer lands. A block a window leaves behind
        at ``end`` is neither fetched nor computed, and under the all-reusable policy it keeps any
        page it had until the commit stores that page whole; so where such a block has a page
        holding tokens past the cache's history, which the request never wrote, the lease fails at
        the call. That covers the block the committed tokens end inside, whose page past them holds
        what the local match copied from another request's page, pages grown before the fetch and,
        with speculative decoding's extra KV tokens, the block holding them, which an earlier
        consecutive lease's grow gave a page past the history it left: a later lease spanning at
        least ``W + tokens_per_block - 1`` tokens leaves it behind where the manager keeps a window
        of ``W`` tokens. With partial reuse on, local matches often end inside a block, and such a
        request then computes from its local match. Under the all-reusable policy a later lease also
        fails at the call where a window leaves behind at ``end`` a block whose page holds tokens
        past those the request computed and whose row no earlier lease into the cache delivered, as
        after an earlier lease missed that row. Where the interval is empty or the caller drops the
        cache, it drops the request's cache in every manager it fetched into, a target and its
        joint-reuse draft pool alike. Pages a fetch grew are freed with the request.

        Fails as ``lend_read``, also at the call on no free pages, while an earlier fetch into the
        request has not settled on this rank, on a target cache with SWA scratch reuse on
        (``enable_swa_scratch_reuse``), which must be off since each capacity change then keeps the
        history within the scratch rewind of the old capacity, which a fetch's grow breaks, and a
        window's next chunk overwrites scratch slots, where the request's history already stands
        past the positions ``readiness`` would count, as after the request computed past ``end``
        under a block reuse policy other than all-reusable, and under the all-reusable policy where
        a window leaves behind at ``end`` a block whose page holds tokens past the cache's history,
        or past those the request computed where no earlier lease into the cache delivered its row.
        It also fails at the call where the request's multimodal data sets
        ``mm_bidirectional_blocks`` and ``end`` falls strictly inside a run of multimodal tokens,
        whatever the run's length: the scheduler keeps such a run within one context chunk, and a
        chunk resumed inside it sees the run's earlier tokens only within the model's sliding
        window, which the lender cannot see. The caller ends such a fetch at a whole block at or
        below the run's start or at or past its end, as the scheduler ends a chunk, and
        ``readiness`` keeps resumes out of the runs, raising ``restart_floor`` and lowering
        ``usable_until`` where a run requires it. It also fails at the call on a request that
        returns context logits, as one asking for prompt logprobs does, or asks for additional model
        outputs, which a model may give per context token, whether or not it holds any yet: the
        executor gives these only for the positions the request computes, prompt logprobs pair the
        context logits' rows with the prompt's tokens from the second on, wherever the rows start,
        and neither a rewind nor a recompute pause clears the rows held. So after a fetch, per-token
        outputs would miss the fetched positions and prompt logprobs would pair rows with the wrong
        tokens; after a context step a fetch could also leave a gap or repeated rows, and context
        logits could overflow their storage, sized to the prompt, which fails every active request.
        A block left without a page fails it at its first poll, as does an error in the grant of its
        slots within the call. A write that fails after the call leaves the cache grown, and
        ``readiness`` accounts for it.

        Args:
            request: The request fetched into.
            start: The first token, a multiple of ``tokens_per_block`` and at least the committed
                tokens rounded down to whole blocks.
            end: The end token, a multiple of ``tokens_per_block``, at most the whole blocks
                before the request's last prompt token, which the request computes itself for its
                logits, and not below the history a windowed cache keeps.

        Returns:
            The lease, possibly failed already.

        Raises:
            ValueError: Bounds that are negative, reversed or not whole blocks, a start inside the
                committed whole blocks, an end past the whole blocks before the request's last
                prompt token or below the history a windowed cache keeps, or a range needing more
                slots than a part has; raised before the cache changes.
        """
        ...

    def readiness(self, request: LlmRequest) -> Optional[Readiness]:
        """Where the request may resume once a fetch into it settled.

        No fetched token counts as computed until this returns a ``Readiness``. That happens when
        the copy ``mark_arrived`` queued is done or the fetch is abandoned. A fetch keeps the
        committed tokens and the other tokens the request had computed below its start: from its
        start on it may overwrite them. The rows it delivers add to what it keeps. If every fetch
        into the cache was abandoned, the request may resume only at its history, and only if what
        was kept reaches it; a fetch into a cache with a sliding window moves the history to the
        fetch's end. A copy ``mark_arrived`` queued that fails counts none of the rows it was to
        copy, and where it covered the block the committed tokens end inside, the interval is empty
        from then on: that block may hold the fetch's bytes in some pools and the bytes it held
        before in others. Under a block reuse policy other than all-reusable the request never
        resumes below its history, which the manager's context update cannot move back. Neither does
        a request in a joint-reuse draft pool, whose context resize sets the capacity from the chunk
        it runs and raises where that capacity is below the request's history. Like ``lend_write``,
        it takes the tokens below the cache's history as computed, apart from those a fetch into a
        cache with a sliding window moved the history over: the caller asks only while the request's
        pages hold the rest, so not while its history runs ahead of its data. The caller combines
        every rank's interval as ``Readiness`` states. A shrink of the cache in place, such as the
        manager's context rollback of the growth a fetch made, voids every delivered row past the
        new capacity for good: growing again brings pages, not their contents. Fetched rows end
        before the request's last prompt token, so the usable end reaches the prompt length only
        where the request computed its whole prompt itself. Copies queue on the manager's stream,
        serial with the forward on the GPU; with page-locked staging and the fresh-page fill off, no
        lender call waits for them on the CPU. ``Readiness`` states how a request with a joint-reuse
        draft pool combines both pools' intervals, and that an empty one drops the request's cache
        in every manager it fetched into.

        Args:
            request: The request a fetch went into.

        Returns:
            ``None`` while a fetch into the request is unsettled, else where it may resume.

        Raises:
            ValueError: The request has no KV cache, which includes after the manager's shutdown.
        """
        ...
