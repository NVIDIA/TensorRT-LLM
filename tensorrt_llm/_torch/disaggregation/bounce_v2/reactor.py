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
"""The bounce_v2 transport reactor: one thread, both protocol roles.

Python port of the C++ ``BounceTransport`` IO loop + ``BounceSender`` +
``BounceReceiver`` (cpp/.../nixl_utils/bounce/BounceTransport.cpp), per the
hybrid design (PYTHON_BOUNCE_DESIGN.en.md Sections 2, 5.1-5.5):

  - all cudaEventQuery / TransferStatus polling lives in the C++
    ``CompletionPoller``; this thread only handles COMPLETED events, drained
    as one numpy batch per tick;
  - the C++ scatter worker threads are eliminated: scatters launch async via
    the bound ``BatchedCopyPool`` and their completion events route back
    through the poller; the ACK is sent from this thread (batched per peer
    per tick — protocol v3 batched ACK);
  - eager gather stays on the SUBMIT CALLER's thread (perf-critical: it
    overlaps the WANT->GRANT round-trip with the gather kernel).

THREADING CONTRACT (every cross-thread interaction):
  - ``submit()`` runs on any caller thread: plan build (numpy, no shared
    state), request creation under ``_req_mu`` (BEFORE the WANT goes out, so
    a fast GRANT always finds the request), WANT send under the channel
    lock, then the eager pump. ``CreditScheduler`` is internally locked;
    ``BatchedCopyPool.launch_chunk`` is thread-safe;
    ``NixlTransferAgent.post_transfer_1to1`` (and ``launch_chunk_chained``)
    runs on the reactor thread AND on submit threads.
  - NO C++ BINDING CALL EVER RUNS WHILE ``_req_mu`` IS HELD (anti-convoy
    rule): the bound calls release the GIL and take tens to hundreds of
    microseconds, so holding the request lock across one turned every other
    submit/handler into a lock+GIL convoy (measured multi-ms submit() stalls
    under concurrent load). Every sender-side C++ transition (gather launch,
    chained gather->RDMA launch, RDMA post) follows DECIDE (mutate
    state under ``_req_mu``) -> EXECUTE (C++ call, lock dropped) -> RECORD
    (re-acquire, register routes / finish the transition). Per request the
    transitions are serialized by a SINGLE-OWNER PUMP
    (``_Request.pump_busy``): whoever pumps owns all of that request's C++
    transitions until the pump drains; concurrent pumpers set ``pump_again``
    and leave, and handlers (_on_gather_done / _on_xfer_done / _on_grant)
    only mutate state and re-trigger the pump. The chunk that is mid-EXECUTE
    is marked ``_Posted.busy_op``, so the failure path never recycles a
    staging region that the in-flight call may still hand to the kernel/NIC
    (see ``_fail_request`` for the per-busy_op contract).
  - ``_req_mu`` guards the sender request table, ``_completions`` (the
    completion-id routing map) and ``_unrouted`` (completion parking): with
    C++ calls outside the lock, a completion row can be drained BEFORE the
    launching thread re-acquires ``_req_mu`` to register its route. Such
    rows PARK in ``_unrouted`` (bounded, timestamped); route registration
    checks the parking dict first and dispatches a parked row inline (after
    releasing the lock), so no completion is ever lost to the race. Rows
    that never find a route age out after ``_UNROUTED_MAX_AGE_S`` with a
    warning — they can only belong to already-failed/cancelled requests
    whose cleanup consumed their region through another path.
  - ALL receiver-role state (``_scattering``, ``_rx_flows``,
    ``_scatter_backlog``, ``_ack_batch``) is reactor-thread-only — the
    single-owner invariant of design Section 5.2. Cross-thread requests that
    must touch it (``forget_peer``, engine-side teardown ordering) go through
    the command queue ``_cmds`` and execute on the reactor thread.
  - The ROUTER socket is reactor-thread-only. Per-peer DEALER sockets live in
    ``_dealers`` and every touch (create / send / close) holds ``_ch_mu``, so
    any thread may send (mirrors the C++ mutex-guarded ``sendTo``). Sends are
    non-blocking; a full queue DROPS the message (the affected request then
    degrades to a request-timeout failure, never a wedged reactor).
  - The WATCHDOG: ``heartbeat_age_s()`` / ``alive()`` are read by waiters
    (engine adapter) from any thread; a crashed reactor fails every pending
    future with ``FAIL_REACTOR_DEAD`` in its exception boundary, and waiters
    additionally poll ``alive()`` so even a hard thread death cannot hang a
    ``wait()`` (design risk #2).

BLOCKING POINT: exactly one — ``zmq.Poller.poll`` over the ROUTER with a
fixed 1 ms timeout, when a tick found no work (GIL released). Completions
are drained non-blockingly via ``CompletionPoller.drain(0)`` each tick; no
other sleeps exist in the loop. There is DELIBERATELY no event-driven
completion wakeup (no fd signalled from the C++ poll thread): waking
the reactor per completion adds a GIL acquisition per wake that contends
with the inline-executing submit threads, and measurements showed it
regresses the median transfer latency versus simply batching all pending
completions on the next 1 ms tick.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np
import zmq

from tensorrt_llm.logger import logger

from .codec import (
    AckEntry,
    BounceMsgHeader,
    BounceMsgType,
    CreditEntry,
    decode_ack,
    decode_credits,
    decode_header,
    decode_scatter,
    decode_want,
    encode_ack,
    encode_cancel,
    encode_data,
    encode_grant,
    encode_want,
    is_cancel_want,
)
from .config import BounceV2Config
from .plan import Plan, build_plan
from .scheduler import CreditScheduler, Grant

__all__ = [
    "FAIL_GATHER",
    "FAIL_NO_PROGRESS",
    "FAIL_PEER_DROPPED",
    "FAIL_PLAN_REJECTED",
    "FAIL_PROTOCOL",
    "FAIL_REACTOR_DEAD",
    "FAIL_REACTOR_STALLED",
    "FAIL_SHUTDOWN",
    "FAIL_WRITE",
    "BounceReactor",
    "BounceResult",
]

#: Separates peer name from request id in a flow key (same as the C++ kSep).
_FLOW_SEP = "\x1f"
#: Reactor idle tick (the design's fixed 1 ms poll timeout). Deliberately a
#: constant, not a knob: 1 ms bounds the completion-batching latency; a
#: 2-3 ms tick would trade a little tail latency for fewer reactor GIL
#: acquisitions if that trade is ever wanted.
_POLL_MS = 1
#: Max ROUTER messages handled per tick (batching bound; keeps one giant
#: burst from starving completion handling within a tick).
_MAX_MSGS_PER_TICK = 512
#: Sender no-progress sweep cadence (the scan itself is cheap; C++ swept
#: every tick, we throttle mildly).
_SENDER_SWEEP_S = 0.1
#: Per-peer outbound queue cap, mirrors the C++ kSendHwm.
_SEND_HWM = 1 << 16
#: CompletionPoller drain-row kinds (mirror the binding's KIND_* constants;
#: hardcoded so the pure-Python reactor stays importable without it).
_KIND_EVENT = 0
_KIND_XFER = 1
#: BatchedCopyPool.submit_scatter_runs validation-failure code (mirrors the
#: binding's SCATTER_REJECTED; hardcoded for the same importability reason).
_SCATTER_REJECTED = -2
#: Unrouted-completion parking (see the module THREADING CONTRACT): rows
#: older than this can only belong to already-failed requests -> dropped
#: with a warning. Generous — the registration race it covers is closed in
#: microseconds.
_UNROUTED_MAX_AGE_S = 60.0
#: Hard bound on parked rows (defense against a routing bug flooding the
#: dict); beyond it the oldest row is dropped with a warning.
_UNROUTED_MAX = 4096
#: Pump execute-phase results (see _pump): CONTINUE keeps sweeping; STOP ends
#: the run after real progress (a failure counts as progress); STOP_IDLE ends
#: it with NO progress (copy pool BUSY — the decided launch was undone, and
#: reporting work would busy-spin the reactor loop while the pool is full).
_PUMP_CONTINUE = 0
_PUMP_STOP = 1
_PUMP_STOP_IDLE = 2
#: Runtime stats log cadence (one INFO line; only when counters changed).
_STATS_LOG_S = 30.0

# Failure-reason strings mirror the C++ BounceFailReason::toString set.
FAIL_PLAN_REJECTED = "bounce: plan rejected (request did not fit a transfer plan)"
FAIL_NO_PROGRESS = "bounce: no GRANT/ACK progress within request_timeout_ms"
FAIL_PEER_DROPPED = "bounce: peer dropped (forget_peer/invalidate_remote_agent)"
FAIL_GATHER = "bounce: gather kernel failed (CUDA error)"
FAIL_WRITE = "bounce: RDMA write failed"
FAIL_PROTOCOL = "bounce: protocol error (GRANT mispair/plan overflow)"
FAIL_SHUTDOWN = "bounce: transport shut down while pending"
FAIL_REACTOR_DEAD = "bounce: reactor thread died"
FAIL_REACTOR_STALLED = "bounce: reactor stalled (no tick within the stall limit)"


@dataclass(frozen=True)
class BounceResult:
    """Terminal outcome of one submitted bounce request."""

    ok: bool
    reason: str = ""


class _PostState(Enum):
    GATHERING = "GATHERING"  # gather copy launched; waiting for its event
    GATHERED = "GATHERED"  # gather done; waiting for the credit (eager path)
    WRITING = "WRITING"  # RDMA write posted; waiting for its completion
    SENT = "SENT"  # DATA emitted; waiting for the ACK


@dataclass
class _Posted:
    """One in-flight chunk of a sender request.

    A CHAINED chunk (credited launch through ``launch_chunk_chained``) goes
    straight to ``WRITING`` at its launch record: the C++ poll thread posts
    the RDMA write when the gather completes, and the chunk's ONE completion
    row publishes under the reserved ``xfer_id`` (the gather's own completion
    is consumed in C++) — ``(xfer_id, KIND_XFER, ok)`` after the write,
    ``(xfer_id, KIND_XFER, 0)`` on a failed post/shutdown, or
    ``(xfer_id, KIND_EVENT, 0)`` when the gather itself failed.

    INTERMEDIATE (mid-C++-call) states, marked by ``busy_op`` (only the
    request's single pump owner sets/clears it; every mutation under
    ``_req_mu``):
      - "launch":         ``state == GATHERING`` and ``copy_id == -1`` — an
        UNCREDITED gather submit is in flight on the pump thread; no
        completion id exists yet, and no RDMA write can come out of it.
      - "launch_chained": like "launch" but CREDITED — the in-flight
        launch_chunk_chained may already have armed the auto-post, so the
        failure path must treat it as a deferred write (see _fail_request).
      - "post":           ``state == GATHERED`` and ``xfer_id == -1`` — the
        classic POSTING intermediate: post_transfer_1to1 in flight; the NIC
        may receive the staging region at any moment, so the failure path
        must NOT recycle it (it defers to the pump's record phase instead).
    """

    chunk_idx: int
    local_offset: int
    write_bytes: int
    state: _PostState = _PostState.GATHERING
    has_credit: bool = False
    remote_handle: int = 0
    remote_addr: int = 0
    remote_dev: int = 0
    copy_id: int = -1
    xfer_id: int = -1
    #: The C++ call currently in flight for this chunk (None when idle); see
    #: the class docstring. Guards the failure path against recycling a
    #: region the in-flight call may still hand to the kernel/NIC.
    busy_op: Optional[str] = None


@dataclass
class _Request:
    """Sender-side per-request state (mirrors the C++ ``Request``)."""

    peer: str
    plan: Plan
    num_chunks: int
    future: "Future[BounceResult]"
    pending_credits: deque[CreditEntry] = field(default_factory=deque)
    posted: list[_Posted] = field(default_factory=list)
    next_post: int = 0
    next_credit: int = 0
    acked: int = 0
    last_progress: float = 0.0
    abandon_reason: str = ""
    #: C++ per-request plan handle (BatchedCopyPool.register_plan); latched
    #: to -1 once released. Released exactly once on the request's
    #: terminal paths (complete / fail / fail_all); the handle holds NO arena
    #: addresses (offsets are region-relative), so its release is independent
    #: of region recycling — a launch racing the release fails
    #: deterministically in C++ (unknown handle) and takes the existing
    #: launch-error path.
    plan_handle: int = -1
    #: Single-owner pump (see the module THREADING CONTRACT): True while some
    #: thread is inside ``_pump`` for this request; ``pump_again`` asks the
    #: owner for one more decide sweep before it exits.
    pump_busy: bool = False
    pump_again: bool = False


@dataclass
class _ScatterJob:
    """Receiver-side scatter of one DATA chunk.

    Carries the RAW wire runs: the compiled pool's ``submit_scatter_runs``
    does validation + run expansion + plan fill in ONE C++ call.
    """

    key: str
    peer: str
    rid: int
    chunk_idx: int
    offset: int  # arena region offset (the DATA region handle)
    runs: np.ndarray  # [m] SCATTER_RUN_DTYPE raw wire runs
    region_base: int  # absolute device address of the granted region
    region_bytes: int  # granted (buddy-block) region length


def _require_binding(obj, name: str):
    """Fetch a REQUIRED compiled-binding method; raise at construction time
    (not at first use) when it is missing. The bindings and this Python code
    ship in one wheel, so absence can only mean a stale/partial build."""
    fn = getattr(obj, name, None)
    if fn is None:
        raise RuntimeError(
            f"bounce_v2 requires the compiled BatchedCopyPool plan-handle bindings "
            f"({type(obj).__name__}.{name} is missing); rebuild the wheel"
        )
    return fn


def _resolve(future: "Future[BounceResult]", result: BounceResult) -> None:
    """Resolve a future exactly once (a benign double-resolve is ignored,
    mirroring the C++ promise set_value try/catch)."""
    if future.done():
        return
    try:
        future.set_result(result)
    except InvalidStateError as e:  # racing double-resolve
        logger.debug(f"bounce_v2: duplicate future resolve ignored: {e}")


class BounceReactor:
    """Single-threaded bounce transport reactor (sender + receiver roles)."""

    def __init__(
        self,
        self_name: str,
        config: BounceV2Config,
        device_id: int,
        raw_agent,
        arena_base: int,
        arena_bytes: int,
        scheduler: CreditScheduler,
        copy_pool,
        poller,
        bind_ip: str,
        set_device_fn: Optional[Callable[[], None]] = None,
    ) -> None:
        """``raw_agent``/``copy_pool``/``poller`` are the compiled-binding
        mechanism objects; ``scheduler`` is the shared pure-logic credit
        scheduler (also used by submit threads for eager staging).
        ``set_device_fn`` pins the reactor thread's CUDA device (injected so
        this module does not import torch)."""
        self._self_name = self_name
        self._cfg = config
        self._device_id = device_id
        self._agent = raw_agent
        self._arena_base = arena_base
        self._arena_bytes = arena_bytes
        self._sched = scheduler
        self._pool = copy_pool
        self._pool_busy = int(copy_pool.BUSY)
        self._max_plan_entries = int(copy_pool.max_plan_entries)
        self._poller = poller
        self._set_device = set_device_fn
        self._max_chunk = config.max_chunk_size_bytes
        # REQUIRED compiled-binding surface (validated HERE, at construction,
        # so a stale build fails the engine/reactor bring-up loudly instead of
        # the first transfer). The bindings and this Python code ship in ONE
        # wheel, so a missing method can only mean a stale/partial build.
        #
        # Receiver-side C++ scatter sink: one bound call validates the raw
        # wire runs, expands them and launches the scatter — replacing the
        # per-chunk numpy expansion that dominated the Python _on_data cost.
        self._submit_scatter_fn = _require_binding(copy_pool, "submit_scatter_runs")
        self._pool_rejected = int(getattr(copy_pool, "SCATTER_REJECTED", _SCATTER_REJECTED))
        self._max_descs_per_chunk = min(
            self._max_plan_entries, max(1024, config.max_chunk_size_bytes // 256)
        )
        # Sender-side per-request C++ plan handle: submit() marshals the whole
        # request's gather plan into C++-owned memory ONCE (register_plan)
        # and every chunk launch becomes one scalar-args, GIL-released C++
        # call — launch_chunk for eager (uncredited) chunks, and the agent's
        # launch_chunk_chained for credited chunks (gather + auto-RDMA-post
        # when the gather completes; no Python hop between them, and Python
        # sees ONE completion per chained chunk).
        self._register_plan_fn = _require_binding(copy_pool, "register_plan")
        self._release_plan_fn = _require_binding(copy_pool, "release_plan")
        self._launch_chunk_fn = _require_binding(copy_pool, "launch_chunk")
        self._launch_chained_fn = _require_binding(raw_agent, "launch_chunk_chained")

        # --- control channel ---
        self._zmq = zmq.Context(io_threads=1)
        self._router = self._zmq.socket(zmq.ROUTER)
        self._router.setsockopt_string(zmq.ROUTING_ID, self_name)
        self._router.setsockopt(zmq.LINGER, 0)
        self._router.setsockopt(zmq.ROUTER_HANDOVER, 1)  # reconnecting peers reuse their id
        if "[" in bind_ip:
            self._router.setsockopt(zmq.IPV6, 1)
        self._router.bind(f"tcp://{bind_ip}:*")
        self._endpoint: str = self._router.getsockopt_string(zmq.LAST_ENDPOINT)
        self._zpoller = zmq.Poller()
        self._zpoller.register(self._router, zmq.POLLIN)

        self._ch_mu = threading.Lock()
        self._dealers: dict[str, zmq.Socket] = {}

        # --- sender state (under _req_mu) ---
        self._req_mu = threading.Lock()
        self._requests: dict[int, _Request] = {}
        self._next_rid = 0
        # completion id -> routing tuple; kinds:
        #   ("gather", rid)                    sender gather event
        #   ("xfer", rid)                      sender RDMA write
        #   ("scatter", _ScatterJob)           receiver scatter event
        #   ("orphan_gather", local_offset)    failed request, gather still running
        #   ("orphan_xfer", local_offset, rid) failed request, write still in flight
        self._completions: dict[int, tuple] = {}
        # Unrouted-completion parking: cid -> (monotonic time, kind, ok) for
        # rows drained before their route was registered (see the module
        # THREADING CONTRACT). Guarded by _req_mu like _completions.
        self._unrouted: dict[int, tuple[float, int, int]] = {}
        # rid -> in-flight orphan write count; a deferred cancel for the rid
        # is sent once this reaches zero (mirrors mPendingCancel).
        self._orphan_writes: dict[int, int] = {}
        self._pending_cancels: dict[int, str] = {}
        self._next_sender_sweep = 0.0

        # --- receiver state (reactor-thread-only) ---
        self._scattering: dict[int, bool] = {}  # region offset -> orphaned?
        self._rx_flows: dict[str, int] = {}  # flow key -> chunks not yet scattered
        self._scatter_backlog: deque[_ScatterJob] = deque()
        self._ack_batch: dict[str, dict[int, list[AckEntry]]] = {}
        self._next_lease_sweep = 0.0

        # --- runtime stats (A/B attribution: which mechanisms actually
        # fired, per role). Increment via _bump(); logged one line per
        # _STATS_LOG_S when changed, plus a final snapshot at shutdown. ---
        self._stats_mu = threading.Lock()
        self._stats: dict[str, int] = {}
        self._last_stats_snapshot: dict[str, int] = {}
        self._next_stats_log = time.monotonic() + _STATS_LOG_S

        # --- lifecycle / watchdog ---
        self._cmds: deque[tuple] = deque()  # cross-thread commands, drained per tick
        self._cmd_mu = threading.Lock()
        self._stop = threading.Event()
        self._dead = False
        self._heartbeat = time.monotonic()
        self._thread = threading.Thread(
            target=self._run, name=f"bounce_v2-reactor-{self_name}", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------ #
    # public API (any thread)
    # ------------------------------------------------------------------ #

    @property
    def endpoint(self) -> str:
        """The ROUTER endpoint peers connect their DEALERs to."""
        return self._endpoint

    def alive(self) -> bool:
        """Watchdog: False once the reactor thread has exited or crashed."""
        return self._thread.is_alive() and not self._dead

    def heartbeat_age_s(self) -> float:
        """Seconds since the reactor last began a tick (the heartbeat is
        written at the top of the loop; equivalent for wedge detection)."""
        return time.monotonic() - self._heartbeat

    def _bump(self, key: str, n: int = 1) -> None:
        with self._stats_mu:
            self._stats[key] = self._stats.get(key, 0) + n

    def stats(self) -> dict[str, int]:
        """Snapshot of the runtime counters (tests / diagnostics)."""
        with self._stats_mu:
            return dict(self._stats)

    def _maybe_log_stats(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now < self._next_stats_log:
            return
        self._next_stats_log = now + _STATS_LOG_S
        snap = self.stats()
        # Idle silence: an idle reactor (no counter moved since the last log)
        # must not emit a line every _STATS_LOG_S.
        if not snap or snap == self._last_stats_snapshot:
            return
        self._last_stats_snapshot = snap
        line = " ".join(f"{k}={v}" for k, v in sorted(snap.items()))
        logger.info(f"bounce_v2({self._self_name}): stats {line}")

    def add_peer(self, peer: str, endpoint: str) -> bool:
        """Create (idempotently) the DEALER route to ``peer``. Thread-safe."""
        if not endpoint:
            logger.warning(f"bounce_v2({self._self_name}): add_peer({peer}) empty endpoint")
            return False
        with self._ch_mu:
            if peer in self._dealers:
                return True
            try:
                dealer = self._zmq.socket(zmq.DEALER)
                dealer.setsockopt_string(zmq.ROUTING_ID, self._self_name)
                dealer.setsockopt(zmq.LINGER, 0)
                dealer.setsockopt(zmq.SNDHWM, _SEND_HWM)
                dealer.setsockopt(zmq.IPV6, 1)
                dealer.connect(endpoint)
            except zmq.ZMQError as e:
                logger.warning(
                    f"bounce_v2({self._self_name}): add_peer({peer}) invalid endpoint "
                    f"{endpoint!r}: {e}"
                )
                return False
            self._dealers[peer] = dealer
            return True

    def remove_peer(self, peer: str) -> None:
        """Close the DEALER route to ``peer`` (idempotent). Thread-safe."""
        with self._ch_mu:
            dealer = self._dealers.pop(peer, None)
            if dealer is not None:
                dealer.close(linger=0)

    def forget_peer(self, peer: str) -> None:
        """Fail this peer's in-flight requests and reclaim its receiver-side
        flows. Thread-safe.

        Mirrors the C++ ``BounceTransport::forgetPeer`` split: the DEALER
        route is dropped SYNCHRONOUSLY here on the caller's thread
        (``remove_peer`` is thread-safe under ``_ch_mu``), which gives a
        deterministic happens-before for any ``add_peer`` the caller issues
        after ``forget_peer`` returns — the route is already gone, so that
        ``add_peer`` rebuilds it instead of racing an async removal that
        could erase the freshly re-added dealer. The state reclaim
        (sender request table + receiver flows) still runs asynchronously ON
        the reactor thread via the command queue (that state is
        single-owner); in-flight futures resolve within one tick. A cancel
        the reclaim emits toward the now-removed peer is dropped with a
        warning (accepted: the peer is being invalidated anyway).

        The SENDER victims are SNAPSHOTTED HERE at call time (the same
        sync-decide/async-execute split as the route removal): only requests
        already in flight toward the peer when forget_peer was called are
        failed by the queued reclaim. A request submitted after forget_peer
        returns (e.g. right after a compatible re-registration replaced the
        route) is untouched — failing by peer NAME at execution time would
        kill it."""
        self.remove_peer(peer)
        with self._req_mu:
            victim_rids = [rid for rid, req in self._requests.items() if req.peer == peer]
        with self._cmd_mu:
            self._cmds.append(("forget_peer", peer, victim_rids))

    def submit(
        self,
        src_ptrs: np.ndarray,
        dst_ptrs: np.ndarray,
        sizes: np.ndarray,
        dst_device_id: int,
        peer: str,
    ) -> "Future[BounceResult]":
        """Submit one WRITE of scattered VRAM descriptors to ``peer``.

        Runs on the CALLER's thread: plan build, WANT send, and (with eager
        gather enabled) the first chunks' gather launches — overlapping the
        WANT->GRANT round-trip. The returned future resolves on every
        terminal path (R5); combine with :meth:`alive` when waiting.
        """
        future: "Future[BounceResult]" = Future()
        if self._stop.is_set() or not self.alive():
            reason = FAIL_SHUTDOWN if self._stop.is_set() else FAIL_REACTOR_DEAD
            _resolve(future, BounceResult(False, reason))
            return future
        try:
            plan = build_plan(
                src_ptrs,
                dst_ptrs,
                sizes,
                max_chunk_bytes=self._max_chunk,
                max_descs_per_chunk=self._max_descs_per_chunk,
                dst_devs=dst_device_id,
            )
        except ValueError as e:
            # Should-not-happen defense (the engine's should_use screens the
            # plan preconditions): resolve FAILURE instead of raising. Bounce
            # admission is final — no silent re-route to the NIXL path.
            logger.warning(f"bounce_v2({self._self_name}): plan for peer {peer} rejected: {e}")
            _resolve(future, BounceResult(False, FAIL_PLAN_REJECTED))
            return future
        if plan.num_chunks == 0:
            _resolve(future, BounceResult(True))
            return future
        self._bump("tx_submits")
        self._bump("tx_chunks", plan.num_chunks)

        chunk_bytes = [c.packed_bytes for c in plan.chunks]
        # ONE marshalling call per request: the whole gather plan moves into
        # C++-owned memory here, so every later chunk launch is a scalar-args,
        # GIL-released call (see _exec_launch).
        g_srcs, g_offsets, g_sizes, g_starts = plan.flat_gather()
        plan_handle = int(self._register_plan_fn(g_srcs, g_offsets, g_sizes, g_starts))
        with self._req_mu:
            rid = self._next_rid
            self._next_rid += 1
            req = _Request(
                peer=peer,
                plan=plan,
                num_chunks=plan.num_chunks,
                future=future,
                last_progress=time.monotonic(),
                plan_handle=plan_handle,
            )
            if self._cfg.enable_eager_gather:
                # PRE-OWN the pump in the same critical section that
                # registers the request: the eager gathers are GUARANTEED to
                # launch on THIS (the caller's) thread — perf-critical, they
                # overlap the WANT->GRANT round-trip with the gather kernel.
                # Without this, the reactor tick (or a fast GRANT) could win
                # the pump between the registration and the caller's pump
                # call and launch them on the reactor instead.
                req.pump_busy = True
            self._requests[rid] = req
        # WANT carries our endpoint so the receiver self-bootstraps the
        # reverse route (sent outside _req_mu, like the C++). ORDERING: the
        # request is already in _requests (above), so the earliest possible
        # GRANT always finds it (and hands its credits to our pre-owned pump
        # via pump_again).
        self._send_to(peer, encode_want(rid, chunk_bytes, self._endpoint))
        if self._cfg.enable_eager_gather:
            self._pump_loop(rid, req)
        return future

    def shutdown(self) -> None:
        """Stop the reactor, join it, and resolve every pending future
        (``failAll`` semantics). Idempotent. The caller must still drain the
        GPU before tearing down the copy pool / arena (engine does)."""
        if self._stop.is_set():
            self._thread.join(timeout=5)
            return
        self._stop.set()  # the reactor observes this within one 1 ms tick
        self._thread.join(timeout=5)
        self._maybe_log_stats(force=True)
        if self._thread.is_alive():
            # Wedged reactor: libzmq sockets are NOT thread-safe, and the
            # stuck thread may still be inside a ROUTER poll/recv. Closing
            # the sockets (or term'ing the context) under it can segfault —
            # LEAK them instead; still fail every pending future.
            logger.warning(
                f"bounce_v2({self._self_name}): reactor did not join within 5 s; "
                f"leaking its ZMQ sockets/context (closing under a live thread "
                f"is unsafe)"
            )
            self._fail_all(FAIL_SHUTDOWN)
            return
        self._fail_all(FAIL_SHUTDOWN)
        with self._ch_mu:
            for dealer in self._dealers.values():
                dealer.close(linger=0)
            self._dealers.clear()
        self._router.close(linger=0)
        self._zmq.term()

    # ------------------------------------------------------------------ #
    # reactor loop
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        try:
            if self._set_device is not None:
                self._set_device()
            while not self._stop.is_set():
                self._heartbeat = time.monotonic()
                did_work = self._drain_commands()
                did_work |= self._drain_router()
                did_work |= self._drain_completions()
                did_work |= self._drain_pending_posts()
                did_work |= self._retry_scatter_backlog()
                self._reap_unrouted()
                self._check_sender_timeouts()
                self._check_receiver_lease()
                self._flush_acks()
                self._maybe_log_stats()
                if not did_work:
                    # The ONE blocking point (GIL released): the ROUTER, on
                    # the fixed 1 ms tick. Completions ride the timeout via
                    # the non-blocking drain above (deliberate — see the
                    # module docstring BLOCKING POINT).
                    self._zpoller.poll(_POLL_MS)
        except Exception as e:  # thread exception boundary (mirrors ioLoop's)
            logger.error(f"bounce_v2({self._self_name}): reactor crashed: {e}", exc_info=True)
            self._dead = True
            self._fail_all(FAIL_REACTOR_DEAD)

    def _drain_commands(self) -> bool:
        with self._cmd_mu:
            if not self._cmds:
                return False
            cmds = list(self._cmds)
            self._cmds.clear()
        for cmd in cmds:
            if cmd[0] == "forget_peer":
                self._do_forget_peer(cmd[1], cmd[2])
        return True

    def _drain_router(self) -> bool:
        did_work = False
        for _ in range(_MAX_MSGS_PER_TICK):
            try:
                parts = self._router.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                break
            except zmq.ZMQError as e:
                logger.warning(f"bounce_v2({self._self_name}): ROUTER recv failed: {e}")
                break
            did_work = True
            if len(parts) < 2:
                continue  # malformed frame set
            peer = parts[0].decode("utf-8", errors="replace")
            blob = bytes(parts[1])
            header = decode_header(blob)
            if header is None:
                logger.warning(
                    f"bounce_v2({self._self_name}): dropped unparsable control message "
                    f"from {peer} ({len(blob)} B)"
                )
                continue
            if header.msg_type == BounceMsgType.WANT:
                self._on_want(peer, header, blob)
            elif header.msg_type == BounceMsgType.GRANT:
                self._on_grant(peer, header, blob)
            elif header.msg_type == BounceMsgType.DATA:
                self._on_data(peer, header, blob)
            elif header.msg_type == BounceMsgType.ACK:
                self._on_ack(peer, header, blob)
            else:
                logger.warning(
                    f"bounce_v2({self._self_name}): unknown msg type {header.msg_type} from {peer}"
                )
        return did_work

    def _drain_completions(self) -> bool:
        rows = self._poller.drain(0)
        if rows.shape[0] == 0:
            return False
        for cid, kind, ok in rows.tolist():
            cid, kind, ok = int(cid), int(kind), int(ok)
            with self._req_mu:
                route = self._completions.pop(cid, None)
                if route is None:
                    # No route YET (a launching thread has not re-acquired
                    # _req_mu to register it — the anti-convoy restructure
                    # allows this ordering): PARK the row; registration
                    # checks the parking dict and dispatches it inline. Rows
                    # of already-failed requests age out with a warning.
                    self._park_unrouted_locked(cid, kind, ok)
                    continue
            self._dispatch_row(route, cid, kind, bool(ok))
        return True

    def _dispatch_row(self, route: tuple, cid: int, kind: int, ok: bool) -> None:
        """Deliver one drained/parked completion row to its routed handler.
        Never called with ``_req_mu`` held (handlers take it)."""
        tag = route[0]
        if tag == "gather":
            self._on_gather_done(route[1], cid, ok)
        elif tag == "xfer":
            self._on_xfer_done(route[1], cid, ok, kind)
        elif tag == "scatter":
            # Receiver state is reactor-thread-only; scatter routes are only
            # ever registered from the reactor thread, so this dispatch (and
            # any parked-row inline dispatch at that registration) runs there.
            self._finish_scatter(route[1], ok)
        elif tag == "orphan_gather":
            self._send_grants(self._sched.release_local(route[1]))
        elif tag == "orphan_xfer":
            self._on_orphan_xfer_done(route[1], route[2])

    def _repark_row(self, cid: int, kind: int, ok: int) -> None:
        """A dispatched row found its request already gone (it raced a
        concurrent failure between the drain's route pop and the handler):
        give it back — the failure cleanup may have re-registered an orphan
        route for it (dispatch now), or will look in the parking dict."""
        with self._req_mu:
            route = self._completions.pop(cid, None)
            if route is None:
                self._park_unrouted_locked(cid, kind, ok)
                return
        self._dispatch_row(route, cid, kind, bool(ok))

    def _park_unrouted_locked(self, cid: int, kind: int, ok: int) -> None:
        """Park an unrouted row (bounded). _req_mu held.

        Pathological-only caveat: with the dict full, the evicted "oldest"
        could in principle be a legitimate mid-registration row; that chunk
        then waits for the request timeout — with request_timeout_ms=0 that
        wait() never resolves. Reaching 4096 simultaneously parked rows
        requires a wedged registration path, so the loud warning is the
        actionable signal.
        """
        if len(self._unrouted) >= _UNROUTED_MAX:
            oldest = min(self._unrouted, key=lambda c: self._unrouted[c][0])
            del self._unrouted[oldest]
            logger.warning(
                f"bounce_v2({self._self_name}): unrouted-completion parking full "
                f"({_UNROUTED_MAX}); dropped oldest row id={oldest}"
            )
        self._unrouted[cid] = (time.monotonic(), kind, ok)

    def _reap_unrouted(self) -> None:
        """Age out parked rows nothing ever claimed. They can only belong to
        already-failed/cancelled requests whose cleanup recycled the region
        through another path (e.g. a row consumed mid-dispatch was treated as
        terminal), so dropping them leaks nothing."""
        if not self._unrouted:
            return
        now = time.monotonic()
        with self._req_mu:
            expired = [
                cid for cid, (t, _k, _ok) in self._unrouted.items() if now - t > _UNROUTED_MAX_AGE_S
            ]
            for cid in expired:
                del self._unrouted[cid]
        if expired:
            logger.warning(
                f"bounce_v2({self._self_name}): dropped {len(expired)} unrouted completion "
                f"row(s) older than {_UNROUTED_MAX_AGE_S:.0f}s (failed/cancelled requests)"
            )

    # ------------------------------------------------------------------ #
    # sender role
    # ------------------------------------------------------------------ #

    def _attach_credits_locked(self, rid: int, req: _Request) -> None:
        """Pair parked credits with already-posted (eager) chunks, strictly
        FIFO. PURE STATE (no C++ calls — anti-convoy rule): the resulting
        post actions are picked up by the pump's decide sweep.
        Validates the mispair guard: a credit smaller than its chunk would
        make the RDMA write overflow into an adjacent flow's region on the
        peer -> the abandon latch is set and the pump FAILS the request
        immediately (the timeout sweep alone would never resolve it when
        request_timeout_ms <= 0 — the R5 guarantee). _req_mu held."""
        while req.pending_credits and req.next_credit < req.next_post:
            credit = req.pending_credits[0]
            target = next((p for p in req.posted if p.chunk_idx == req.next_credit), None)
            if target is None or target.has_credit:
                logger.warning(
                    f"bounce_v2({self._self_name}): rid={rid} chunk={req.next_credit} "
                    f"unexpected credit (dup GRANT?); dropping"
                )
                req.pending_credits.popleft()
                req.next_credit += 1
                continue
            chunk = req.plan.chunks[target.chunk_idx]
            if chunk.packed_bytes > credit.length:
                logger.warning(
                    f"bounce_v2({self._self_name}): rid={rid} chunk={target.chunk_idx} "
                    f"packed_bytes={chunk.packed_bytes} > granted region len={credit.length} "
                    f"(GRANT mispair/reorder); abandoning flow"
                )
                req.abandon_reason = FAIL_PROTOCOL
                req.pending_credits.clear()
                return
            target.remote_handle = credit.region_handle
            target.remote_addr = credit.addr
            target.remote_dev = credit.dev_id
            target.has_credit = True
            # Now credit-backed: stop counting against the eager budget.
            self._sched.promote_local(target.local_offset)
            req.pending_credits.popleft()
            req.next_credit += 1
            req.last_progress = time.monotonic()
        if req.next_credit >= req.num_chunks and req.pending_credits:
            logger.warning(
                f"bounce_v2({self._self_name}): rid={rid} over-grant, dropping "
                f"{len(req.pending_credits)} extra credit(s)"
            )
            req.pending_credits.clear()

    # ------------------------------------------------------------------ #
    # sender pump: DECIDE (locked) -> EXECUTE (C++, unlocked) -> RECORD
    # (locked). Single owner per request; see the module THREADING CONTRACT.
    # ------------------------------------------------------------------ #

    def _pump(self, rid: int) -> bool:
        """Advance a request: attach credits, then run every ready C++
        transition (plain/chained gather launches, classic posts) with
        ``_req_mu`` DROPPED around each bound call. Any thread; returns True
        if anything advanced. Concurrent callers hand the work to the current
        owner via ``pump_again``."""
        with self._req_mu:
            req = self._requests.get(rid)
            if req is None:
                return False
            if req.pump_busy:
                req.pump_again = True
                return False
            req.pump_busy = True
        return self._pump_loop(rid, req)

    def _pump_loop(self, rid: int, req: _Request) -> bool:
        """The pump body; the caller has already taken ownership
        (``req.pump_busy`` True) — either via :meth:`_pump` or pre-owned at
        request registration (``submit``'s eager-gather guarantee)."""
        did_work = False
        # Tracks whether THIS thread still holds pump ownership: the idle
        # exit below releases it atomically with the pump_again re-check, and
        # from that instant another thread may legitimately own the pump —
        # the finally must not stomp pump_busy it no longer owns (a second
        # concurrent owner could re-select a mid-post chunk and duplicate the
        # RDMA write into a region the ACK chain then recycles under it).
        owned = True
        try:
            while True:
                action = None
                fail_reason = None
                with self._req_mu:
                    if self._requests.get(rid) is not req:
                        break  # failed/completed while we were executing
                    self._attach_credits_locked(rid, req)
                    if not req.abandon_reason:
                        action = self._next_action_locked(rid, req)
                    if req.abandon_reason:
                        fail_reason = req.abandon_reason
                        action = None
                if fail_reason is not None:
                    self._fail_request(rid, req, fail_reason)
                    did_work = True
                    break
                if action is None:
                    with self._req_mu:
                        if self._requests.get(rid) is req and req.pump_again:
                            req.pump_again = False
                            continue
                        # Release ownership ATOMICALLY with the pump_again
                        # check: a handler that hands over work after this
                        # point sees pump_busy False and becomes the owner
                        # itself — clearing in a separate critical section
                        # (the finally alone) would drop that wakeup, and a
                        # GATHERED+credited chunk awaiting its post has no
                        # other retry path (_drain_pending_posts only rescans
                        # requests with parked credits / unlaunched chunks).
                        req.pump_busy = False
                        owned = False
                        break
                try:
                    rc = self._execute_action(rid, req, action)
                except BaseException:
                    # Defense in depth: an exception escaping an executor
                    # (each already catches the expected launch errors) must
                    # not leave the chunk marked busy forever — the failure
                    # path SKIPS busy chunks, expecting a record phase that
                    # would now never run, silently wedging the region.
                    with self._req_mu:
                        action[1].busy_op = None
                    raise
                if rc != _PUMP_STOP_IDLE:
                    # A BUSY copy pool undoes the decided launch with no net
                    # progress: reporting it as work would make the reactor
                    # loop skip its poll sleep and spin while the pool is
                    # saturated (its retry is capacity-driven, one tick away).
                    did_work = True
                if rc != _PUMP_CONTINUE:
                    break  # copy pool BUSY / request failed
        finally:
            # Exception/early-exit backstop: release ownership ONLY if this
            # thread still holds it — after the atomic idle release another
            # thread may already own the pump, and clearing its flag would
            # admit a concurrent second owner. pump_again is deliberately NOT
            # consumed here — with pump_busy False the next _pump (handler or
            # the per-tick _drain_pending_posts) owns the sweep and clears
            # it. KNOWN LIMIT (defense-in-depth path only): if an exception
            # escapes _execute_action, a fully-launched request whose only
            # remaining work is a GATHERED+credited post is not rescanned by
            # _drain_pending_posts and waits for the request timeout.
            if owned:
                with self._req_mu:
                    req.pump_busy = False
        return did_work

    def _next_action_locked(self, rid: int, req: _Request) -> Optional[tuple]:
        """Pick (and mark, via ``busy_op``) the next C++ transition. _req_mu
        held; PURE STATE. Chunk transitions first (FIFO over ``posted``),
        then a new gather launch. Only the pump owner calls this; the
        ``busy_op`` skip is belt-and-braces against any future ownership bug
        re-selecting a chunk that is mid-EXECUTE on another thread."""
        for p in req.posted:
            if p.busy_op is not None:
                continue
            if p.state == _PostState.GATHERED and p.has_credit and p.xfer_id < 0:
                p.busy_op = "post"
                return ("post", p)
        if req.next_post >= req.num_chunks:
            return None
        chunk_idx = req.next_post
        chunk = req.plan.chunks[chunk_idx]
        have_credit = bool(req.pending_credits)
        if not have_credit:
            if not self._cfg.enable_eager_gather:
                return None  # classic path: gather starts only once granted
            if len(req.posted) >= self._cfg.max_inflight_chunks_per_request:
                return None  # eager gathers capped by the in-flight window
        if have_credit and chunk.packed_bytes > req.pending_credits[0].length:
            logger.warning(
                f"bounce_v2({self._self_name}): rid={rid} chunk={chunk_idx} "
                f"packed_bytes={chunk.packed_bytes} > granted region "
                f"len={req.pending_credits[0].length} (GRANT mispair/reorder); abandoning flow"
            )
            req.abandon_reason = FAIL_PROTOCOL
            req.pending_credits.clear()
            return None
        # Non-blocking staging: no region right now -> park; the reactor's
        # drain_pending_posts retries once an ACK frees space. Never blocks,
        # so oversubscription degrades to backpressure, not deadlock.
        local_off = self._sched.acquire_local(chunk.packed_bytes, eager=not have_credit)
        if local_off is None:
            return None
        # A credited launch CHAINS through the plan handle (the C++ poll
        # thread auto-posts the RDMA write): its distinct busy_op tells the
        # failure path this in-flight call may already be a write.
        chained = have_credit
        posted = _Posted(
            chunk_idx=chunk_idx,
            local_offset=local_off,
            write_bytes=chunk.packed_bytes,
            busy_op="launch_chained" if chained else "launch",
        )
        credit = None
        if have_credit:
            credit = req.pending_credits.popleft()
            req.next_credit += 1
            posted.has_credit = True
            posted.remote_handle = credit.region_handle
            posted.remote_addr = credit.addr
            posted.remote_dev = credit.dev_id
        req.posted.append(posted)
        req.next_post += 1
        return ("launch", posted, credit)

    def _execute_action(self, rid: int, req: _Request, action: tuple) -> int:
        """Run one decided transition (C++ outside ``_req_mu``) and record
        it. Returns a ``_PUMP_*`` code (STOP_IDLE only for a BUSY pool)."""
        kind = action[0]
        if kind == "launch":
            return self._exec_launch(rid, req, action[1], action[2])
        return self._exec_post(rid, req, action[1])

    def _register_route_locked(self, cid: int, route: tuple) -> list[tuple]:
        """Register a completion route, honoring the parking dict: a row that
        was drained before this registration is returned as a dispatch tuple
        ``(cid, route, kind, ok)`` — the caller MUST dispatch it after
        releasing ``_req_mu``. _req_mu held."""
        parked = self._unrouted.pop(cid, None)
        if parked is None:
            self._completions[cid] = route
            return []
        _, kind, ok = parked
        return [(cid, route, kind, ok)]

    def _orphan_route_locked(self, cid: int, route: tuple) -> tuple[bool, list[tuple]]:
        """Point an id at an orphan route during failure cleanup. Returns
        ``(row_pending, dispatches)``: row_pending False means the id's row
        was ALREADY consumed (drained and dispatched, or mid-dispatch) — the
        target is terminal and the caller must recycle the region itself.
        _req_mu held."""
        if cid in self._completions:
            self._completions[cid] = route
            return True, []
        parked = self._unrouted.pop(cid, None)
        if parked is not None:
            _, kind, ok = parked
            return True, [(cid, route, kind, ok)]
        return False, []

    def _remove_launch_locked(self, req: _Request, posted: _Posted, credit) -> None:
        """Undo a decided launch whose submit did not stick (pool BUSY or a
        launch error): the chunk is the newest posted entry (only the pump
        appends), so next_post rolls back cleanly; a credit taken for it goes
        back to the head of the queue. _req_mu held."""
        req.posted.remove(posted)
        req.next_post -= 1
        if credit is not None:
            req.pending_credits.appendleft(credit)
            req.next_credit -= 1

    def _exec_launch(self, rid: int, req: _Request, posted: _Posted, credit) -> int:
        """EXECUTE+RECORD of a gather launch: ONE scalar-args, GIL-released
        C++ call over the pre-marshalled plan — ``launch_chunk`` for an
        uncredited (eager) chunk, ``launch_chunk_chained`` for a credited one
        (gather + C++ auto-RDMA-post; the chunk goes straight to WRITING and
        its ONE completion row publishes under the reserved ``xfer_id`` — see
        the _Posted docstring for the row contract). The C++ call runs
        WITHOUT _req_mu."""
        region_base = self._arena_base + posted.local_offset
        chained = posted.busy_op == "launch_chained"
        copy_id = self._pool_busy
        reserved = -1
        err: Optional[BaseException] = None
        # SNAPSHOT the handle exactly once: _fail_request latches
        # req.plan_handle to -1 under _req_mu at any moment, and this phase
        # runs unlocked. Reading it twice could tear; a single snapshot of an
        # already-released handle instead takes the deterministic error path
        # below (ValueError from the binding's unknown-handle validation, or
        # TypeError from -1 reaching a nanobind unsigned parameter).
        plan_handle = req.plan_handle
        try:
            if chained:
                copy_id, reserved = self._launch_chained_fn(
                    self._pool,
                    plan_handle,
                    posted.chunk_idx,
                    region_base,
                    posted.remote_addr,
                    posted.write_bytes,
                    self._device_id,
                    posted.remote_dev,
                    req.peer,
                    self._poller,
                )
                copy_id, reserved = int(copy_id), int(reserved)
            else:
                copy_id = int(self._launch_chunk_fn(plan_handle, posted.chunk_idx, region_base))
        except (RuntimeError, ValueError, TypeError) as e:
            # RuntimeError: CUDA error / plan overflow from the pool.
            # ValueError: the binding's argument validation (nanobind maps
            # std::invalid_argument) — includes a launch racing the plan
            # handle's release on the failure path (unknown handle); it must
            # take the same deterministic failure path, not escape mid-launch.
            # TypeError: belt-and-braces for a negative/incompatible argument
            # reaching a nanobind unsigned parameter — same failure path.
            err = e
        grants: list[Grant] = []
        dispatches: list[tuple] = []
        fail_reason = None
        settle = False
        bumps: list[str] = []
        rc = _PUMP_CONTINUE
        with self._req_mu:
            alive = self._requests.get(rid) is req and not req.abandon_reason
            posted.busy_op = None
            if err is not None:
                # Gather launch failed (CUDA error / plan overflow / released
                # plan handle). No event was registered, so the region is safe
                # to recycle now; fail deterministically like the C++
                # GatherFailed path.
                logger.warning(
                    f"bounce_v2({self._self_name}): rid={rid} chunk={posted.chunk_idx} "
                    f"gather launch failed: {err}"
                )
                self._remove_launch_locked(req, posted, credit)
                grants.extend(self._sched.release_local(posted.local_offset))
                if alive:
                    fail_reason = FAIL_GATHER
                elif chained:
                    # The fail path counted this busy chained launch as a
                    # deferred write; nothing was launched — settle it now.
                    settle = True
                rc = _PUMP_STOP
            elif copy_id == self._pool_busy:
                # Every copy stream busy: undo and retry next tick (STOP_IDLE:
                # no net progress — see _pump's did_work handling).
                self._remove_launch_locked(req, posted, credit)
                grants.extend(self._sched.release_local(posted.local_offset))
                if not alive and chained:
                    settle = True  # deferred-write count taken by the fail path
                rc = _PUMP_STOP_IDLE
            elif alive:
                posted.copy_id = copy_id
                bumps.append("tx_gather_credit" if posted.has_credit else "tx_gather_eager")
                if reserved >= 0:
                    # Chained: the chunk's ONE completion id is the reserved
                    # xfer id from here on (the gather row is consumed in
                    # C++); the write posts on the C++ poll thread.
                    posted.xfer_id = reserved
                    posted.state = _PostState.WRITING
                    bumps.append("tx_chained_launches")
                    dispatches.extend(self._register_route_locked(reserved, ("xfer", rid)))
                else:
                    # Plain launch — or a chained call that could not reserve
                    # (poller already shut down): the copy_id resolves
                    # classically.
                    dispatches.extend(self._register_route_locked(copy_id, ("gather", rid)))
                req.last_progress = time.monotonic()
            else:
                # The request FAILED while the submit was in flight; the fail
                # path skipped a plain "launch" (no ids yet) and counted a
                # "launch_chained" as a deferred write. The fresh ids were
                # never routed, so their rows can only be pending or parked —
                # _register_route_locked covers both.
                if reserved >= 0:
                    # Chained: exactly one terminal row publishes under the
                    # reserved id (after the auto-posted write, or the gather
                    # failure) — route it as the orphan write; its dispatch
                    # releases the region and settles the deferred cancel.
                    dispatches.extend(
                        self._register_route_locked(
                            reserved, ("orphan_xfer", posted.local_offset, rid)
                        )
                    )
                elif chained:
                    # Chained launch that could NOT reserve (poller shut
                    # down): the gather row is pending under copy_id and no
                    # write was ever posted — route it as an orphan WRITE so
                    # its dispatch also settles the deferred cancel (kind is
                    # ignored by _on_orphan_xfer_done).
                    dispatches.extend(
                        self._register_route_locked(
                            copy_id, ("orphan_xfer", posted.local_offset, rid)
                        )
                    )
                else:
                    dispatches.extend(
                        self._register_route_locked(copy_id, ("orphan_gather", posted.local_offset))
                    )
        for key in bumps:
            self._bump(key)
        self._send_grants(grants)
        for cid, route, kind, ok in dispatches:
            self._dispatch_row(route, cid, kind, bool(ok))
        if settle:
            self._settle_orphan_write(rid)
        if fail_reason is not None:
            self._fail_request(rid, req, fail_reason)
        return rc

    def _exec_post(self, rid: int, req: _Request, posted: _Posted) -> int:
        """Classic RDMA post for a gathered + credited chunk (the POSTING
        intermediate: state GATHERED, busy_op "post"). The C++ call runs
        WITHOUT _req_mu; a synchronous post failure fails the request with
        the existing FAIL_WRITE semantics."""
        xid = int(
            self._agent.post_transfer_1to1(
                self._arena_base + posted.local_offset,
                posted.remote_addr,
                posted.write_bytes,
                self._device_id,
                posted.remote_dev,
                req.peer,
                self._poller,
            )
        )
        grants: list[Grant] = []
        dispatches: list[tuple] = []
        settle = False
        fail_reason = None
        posted_ok = False
        with self._req_mu:
            alive = self._requests.get(rid) is req
            posted.busy_op = None
            if alive:
                if xid < 0:
                    # Chunk stays GATHERED (terminal for its region: the NIC
                    # never saw it) — _fail_request recycles it.
                    fail_reason = FAIL_WRITE
                else:
                    posted_ok = True
                    posted.xfer_id = xid
                    posted.state = _PostState.WRITING
                    dispatches.extend(self._register_route_locked(xid, ("xfer", rid)))
                    req.last_progress = time.monotonic()
            else:
                # Request failed mid-post; the fail path counted this chunk
                # as a deferred write.
                if xid >= 0:
                    dispatches.extend(
                        self._register_route_locked(xid, ("orphan_xfer", posted.local_offset, rid))
                    )
                else:
                    grants.extend(self._sched.release_local(posted.local_offset))
                    settle = True
        if posted_ok:
            self._bump("tx_post_classic")
        self._send_grants(grants)
        for cid, route, kind, ok in dispatches:
            self._dispatch_row(route, cid, kind, bool(ok))
        if settle:
            self._settle_orphan_write(rid)
        if fail_reason is not None:
            self._fail_request(rid, req, fail_reason)
            return _PUMP_STOP
        return _PUMP_CONTINUE

    def _drain_pending_posts(self) -> bool:
        did_work = False
        with self._req_mu:
            rids = [
                rid
                for rid, req in self._requests.items()
                if req.pending_credits
                or (self._cfg.enable_eager_gather and req.next_post < req.num_chunks)
            ]
        for rid in rids:
            did_work |= self._pump(rid)
        return did_work

    def _on_gather_done(self, rid: int, copy_id: int, ok: bool) -> None:
        repark = False
        fail_req: Optional[_Request] = None
        advanced = False
        with self._req_mu:
            req = self._requests.get(rid)
            if req is None:
                # The request died between the drain's route pop and this
                # dispatch: RE-PARK the row so the failure cleanup's orphan
                # registration (which checks the parking dict) still finds
                # it — never silently consume a row another path relies on.
                repark = True
            else:
                # Route by the completion id: gathers of one request may
                # finish OUT OF ORDER across copy streams, so "oldest
                # gathering" would be wrong (it could post a write for an
                # unfinished gather).
                target = next((p for p in req.posted if p.copy_id == copy_id), None)
                if not ok:
                    # The failed gather's completion id was already consumed
                    # by the drain and the kernel is DONE with the staging
                    # region: advance the chunk to a terminal state BEFORE
                    # failing, so _fail_request releases its region now
                    # instead of re-registering the consumed id as an
                    # orphan_gather that would never fire (poller ids report
                    # exactly once).
                    if target is not None and target.state == _PostState.GATHERING:
                        target.state = _PostState.GATHERED
                    fail_req = req
                elif target is not None and target.state == _PostState.GATHERING:
                    # STATE ONLY (anti-convoy rule): a credited chunk's post
                    # is issued by the pump below, never under this lock.
                    target.state = _PostState.GATHERED
                    advanced = True
        if repark:
            # Not counted as a handled gather event: the row goes back for
            # its real consumer (the failure cleanup's orphan route).
            self._repark_row(copy_id, _KIND_EVENT, 1 if ok else 0)
            return
        self._bump("tx_gather_events")
        if fail_req is not None:
            self._fail_request(rid, fail_req, FAIL_GATHER)
            return
        if advanced:
            self._pump(rid)

    def _on_xfer_done(self, rid: int, xfer_id: int, ok: bool, kind: int = _KIND_XFER) -> None:
        data_msg: Optional[tuple[str, bytes]] = None
        repark = False
        fail_req: Optional[_Request] = None
        with self._req_mu:
            req = self._requests.get(rid)
            if req is None:
                repark = True  # see _on_gather_done: never lose a raced row
            else:
                target = next((p for p in req.posted if p.xfer_id == xfer_id), None)
                if target is None:
                    return
                if not ok:
                    # The chunk's completion id was already consumed by the
                    # drain, so the chunk must go terminal HERE:
                    # _fail_request would otherwise re-register the consumed
                    # id as an orphan write that can never fire (wedging
                    # _orphan_writes and the deferred cancel, and leaking the
                    # region). Every failed row that lands here is terminal
                    # for the staging region:
                    #   (id, KIND_XFER, 0)  classic write or chained
                    #       auto-post failed / shutdown -> the NIC/agent is
                    #       DONE with the region;
                    #   (id, KIND_EVENT, 0) chained gather failed -> the
                    #       kernel is done and the write was never posted
                    #       (NIC never saw the region) — reported as
                    #       FAIL_GATHER to keep the reason accurate.
                    # Guarded like the gather path: a duplicate delivery must
                    # not relabel a chunk that is no longer WRITING.
                    if target.state == _PostState.WRITING:
                        target.state = _PostState.SENT
                    fail_req = req
                else:
                    chunk = req.plan.chunks[target.chunk_idx]
                    data_msg = (
                        req.peer,
                        encode_data(
                            rid,
                            target.chunk_idx,
                            req.num_chunks,
                            target.remote_handle,
                            chunk.scatter_runs,
                        ),
                    )
                    target.state = _PostState.SENT
                    req.last_progress = time.monotonic()
        if repark:
            self._repark_row(xfer_id, kind, 1 if ok else 0)
            return
        if fail_req is not None:
            self._fail_request(rid, fail_req, FAIL_GATHER if kind == _KIND_EVENT else FAIL_WRITE)
            return
        if data_msg is not None:
            self._bump("tx_xfer_events")
            self._bump("tx_data_sent")
            self._send_to(*data_msg)

    def _on_ack(self, peer: str, header: BounceMsgHeader, blob: bytes) -> None:
        entries = decode_ack(blob, header)
        if entries is None:
            return
        done_future: Optional["Future[BounceResult]"] = None
        grants: list[Grant] = []
        with self._req_mu:
            req = self._requests.get(header.request_id)
            if req is None:
                return
            if peer != req.peer:
                logger.warning(
                    f"bounce_v2({self._self_name}): dropping wrong-peer ACK peer={peer} "
                    f"expected={req.peer} rid={header.request_id}"
                )
                return
            for entry in entries:
                target_idx = next(
                    (
                        i
                        for i, p in enumerate(req.posted)
                        if p.chunk_idx == entry.chunk_idx
                        and p.remote_handle == entry.region_handle
                        and p.state == _PostState.SENT
                    ),
                    None,
                )
                if target_idx is None:
                    # Duplicate/unknown ACK (reconnect, retransmit): never
                    # count it — an over-count could resolve SUCCESS early.
                    logger.warning(
                        f"bounce_v2({self._self_name}): dropping stale/invalid ACK peer={peer} "
                        f"rid={header.request_id} chunk={entry.chunk_idx} "
                        f"region={entry.region_handle}"
                    )
                    continue
                grants.extend(self._sched.release_local(req.posted[target_idx].local_offset))
                del req.posted[target_idx]
                req.acked += 1
                self._bump("tx_acked_chunks")
                req.last_progress = time.monotonic()
            plan_handle = -1
            if req.acked >= req.num_chunks:
                done_future = req.future
                # All chunks acked -> no launch can be in flight (a busy
                # chunk could never be SENT): the handle is idle, release it
                # outside the lock.
                plan_handle, req.plan_handle = req.plan_handle, -1
                del self._requests[header.request_id]
        self._release_plan(plan_handle)
        self._send_grants(grants)
        if done_future is not None:
            _resolve(done_future, BounceResult(True))

    def _on_grant(self, peer: str, header: BounceMsgHeader, blob: bytes) -> None:
        credits = decode_credits(blob, header)
        if credits is None:
            return
        with self._req_mu:
            req = self._requests.get(header.request_id)
            if req is None:
                return  # late grant for a finished/cancelled request
            # A credit carries a peer-owned address: never let an unrelated
            # peer redirect this request's RDMA write.
            if peer != req.peer:
                logger.warning(
                    f"bounce_v2({self._self_name}): dropping wrong-peer GRANT peer={peer} "
                    f"expected={req.peer} rid={header.request_id}"
                )
                return
            req.pending_credits.extend(credits)
        self._pump(header.request_id)

    def _check_sender_timeouts(self) -> None:
        if self._cfg.request_timeout_ms <= 0:
            return
        now = time.monotonic()
        if now < self._next_sender_sweep:
            return
        self._next_sender_sweep = now + _SENDER_SWEEP_S
        limit = self._cfg.request_timeout_ms / 1000.0
        with self._req_mu:
            stuck = [
                (rid, req) for rid, req in self._requests.items() if now - req.last_progress > limit
            ]
        for rid, req in stuck:
            self._fail_request(rid, req, FAIL_NO_PROGRESS)

    def _fail_request(self, rid: int, req: _Request, reason: str) -> None:
        """Terminal failure path (mirrors the C++ failRequest). Call WITHOUT
        ``_req_mu`` held (any thread); idempotent — only the caller whose
        request object is still registered performs the cleanup.

        Regions are recycled only once nothing can still touch their memory:
          - WRITING: defers until the RDMA completion (the NIC may still read
            it) — the ("xfer", rid) route becomes an orphan_xfer route. A
            CHAINED chunk sits in WRITING from its launch record even while
            its gather still runs; its ONE terminal row (under the reserved
            xfer_id) settles both the region and the deferred cancel;
          - GATHERING: defers until the copy event fires (the kernel may
            still write it) — the copy route becomes an orphan_gather route;
          - GATHERED / SENT: nothing touches the region anymore — recycle.
          - BUSY chunks (a pump's C++ call is in flight; at most one, the
            pump owner's current chunk): "launch" is skipped entirely — no
            completion id exists yet, no write can come out of it, and the
            pump's record phase releases or orphans the region itself.
            "launch_chained"/"post" may still turn into (or already be) an
            RDMA write, so they count as deferred writes here (holding the
            cancel-WANT back); the pump's record phase then registers the
            orphan route (finding a parked row inline) or releases + settles
            when no row can come.

        The chunk whose FAILED completion triggered this call must be
        advanced to a terminal state (GATHERED/SENT) by its handler first:
        its completion id was already consumed by the drain, so registering
        it as an orphan here would never fire and leak the region (the
        _orphan_route_locked existence check backstops exactly that). The
        cancel-WANT to the receiver is deferred while any write is (or may
        be) in flight — it is landing on the receiver's region; an early
        cancel would let the receiver re-grant that region under the write."""
        grants: list[Grant] = []
        dispatches: list[tuple] = []
        with self._req_mu:
            if self._requests.get(rid) is not req:
                return  # already failed/completed by another path
            if req.abandon_reason:
                reason = req.abandon_reason
            # Latch the terminal reason on the (now-dead) request object: the
            # pump owner (and any handler) re-checks the registry/latch and
            # stops advancing. Without it a second failure would KeyError on
            # `del self._requests[rid]`, a second success would RDMA-write
            # from an already-released staging region, and every further
            # iteration would leak an arena region.
            req.abandon_reason = reason
            logger.warning(
                f"bounce_v2({self._self_name}): request FAILED rid={rid} peer={req.peer} "
                f'reason="{reason}" acked={req.acked} posted={req.next_post}/{req.num_chunks}'
            )
            deferred = 0
            for p in req.posted:
                if p.busy_op == "launch":
                    continue  # no ids yet; the pump's record phase cleans up
                if p.busy_op is not None:  # launch_chained / post in flight
                    deferred += 1
                    continue
                if p.state == _PostState.WRITING:
                    pending, out = self._orphan_route_locked(
                        p.xfer_id, ("orphan_xfer", p.local_offset, rid)
                    )
                    dispatches.extend(out)
                    if pending or out:
                        deferred += 1
                    else:  # row already consumed mid-race: xfer terminal
                        grants.extend(self._sched.release_local(p.local_offset))
                elif p.state == _PostState.GATHERING:
                    pending, out = self._orphan_route_locked(
                        p.copy_id, ("orphan_gather", p.local_offset)
                    )
                    dispatches.extend(out)
                    if not pending and not out:
                        grants.extend(self._sched.release_local(p.local_offset))
                else:  # GATHERED / SENT: nothing touches the region anymore
                    grants.extend(self._sched.release_local(p.local_offset))
            if deferred:
                self._orphan_writes[rid] = self._orphan_writes.get(rid, 0) + deferred
                self._pending_cancels[rid] = req.peer
                cancel_now = False
            else:
                cancel_now = True
            future = req.future
            # The plan handle is released OUTSIDE the lock (C++ call); latch
            # it here so the release happens exactly once (a raced in-flight
            # launch fails deterministically in C++ afterwards).
            plan_handle, req.plan_handle = req.plan_handle, -1
            del self._requests[rid]
        # --- outside _req_mu: sends, dispatches, plan release, resolve ---
        self._release_plan(plan_handle)
        if cancel_now:
            self._send_to(req.peer, encode_cancel(rid, self._endpoint))
        self._send_grants(grants)
        for cid, route, kind, ok in dispatches:
            self._dispatch_row(route, cid, kind, bool(ok))
        _resolve(future, BounceResult(False, reason))
        # This may run on a SUBMIT thread (mispair / gather-launch failure at
        # submit time): the regions released above can unblock another request
        # parked on arena capacity — the reactor's per-tick
        # _drain_pending_posts picks that up within one 1 ms tick.

    def _on_orphan_xfer_done(self, local_offset: int, rid: int) -> None:
        """A failed request's in-flight write reached a terminal state: its
        staging region is recyclable, and once the rid's last deferred write
        settles, the deferred cancel may finally be sent (the receiver can
        then safely reclaim its regions)."""
        self._send_grants(self._sched.release_local(local_offset))
        self._settle_orphan_write(rid)

    def _settle_orphan_write(self, rid: int) -> None:
        """One deferred write of a failed request reached a terminal state
        (or provably never became a write): decrement the rid's count and
        send the deferred cancel-WANT once it reaches zero."""
        with self._req_mu:
            remaining = self._orphan_writes.get(rid, 0) - 1
            if remaining > 0:
                self._orphan_writes[rid] = remaining
                return
            self._orphan_writes.pop(rid, None)
            peer = self._pending_cancels.pop(rid, None)
        if peer is not None:
            self._send_to(peer, encode_cancel(rid, self._endpoint))

    def _release_plan(self, handle: int) -> None:
        """Free a C++ plan handle (no-op for the -1 already-released latch).
        Call WITHOUT ``_req_mu`` (bound call, anti-convoy rule)."""
        if handle >= 0:
            self._release_plan_fn(handle)

    def _fail_all(self, reason: str) -> None:
        with self._req_mu:
            requests = list(self._requests.values())
            self._requests.clear()
            # Cleared tables make any concurrent record-phase re-insert /
            # orphan settle a no-op (dead rows park and age out; staging
            # memory is protected by the engine's GPU drain + the poller
            # shutdown sweep, not by these tables).
            self._completions.clear()
            self._unrouted.clear()
            self._orphan_writes.clear()
            self._pending_cancels.clear()
            # Latch the handles UNDER the lock (like _fail_request) so a
            # concurrent _exec_launch snapshot sees either the live handle
            # or -1, never a half-latched batch.
            handles = [(req, req.plan_handle) for req in requests]
            for req in requests:
                req.plan_handle = -1
        for req, plan_handle in handles:
            self._release_plan(plan_handle)
            _resolve(req.future, BounceResult(False, reason))

    # ------------------------------------------------------------------ #
    # receiver role (reactor-thread-only)
    # ------------------------------------------------------------------ #

    def _on_want(self, peer: str, header: BounceMsgHeader, blob: bytes) -> None:
        decoded = decode_want(blob, header)
        if decoded is None:
            return
        chunk_sizes, endpoint = decoded
        key = self._make_key(peer, header.request_id)
        if is_cancel_want(chunk_sizes):
            # Sender-initiated cancel: it drained its in-flight writes before
            # sending this, so non-busy regions free immediately (quarantine 0).
            # Purge the flow's queued scatters BEFORE computing the busy set
            # (see _purge_scatter_backlog's ordering contract): their regions
            # then free exactly once through the flow's held set, and a stale
            # job can no longer scatter into KV addresses freed by this cancel.
            self._purge_scatter_backlog(lambda job: job.key == key)
            grants, deferred = self._sched.forget(key, busy=set(self._scattering))
            for off in deferred:
                self._scattering[off] = True
            self._rx_flows.pop(key, None)
            self._send_grants(grants)
            return
        if not self.add_peer(peer, endpoint):
            return  # no reverse route: granting would be pointless
        if key in self._rx_flows:
            # A non-empty WANT is sent exactly once per fresh rid; a repeat is
            # a replay/collision. Re-queueing would leak re-granted regions.
            logger.warning(
                f"bounce_v2({self._self_name}): dropped duplicate WANT from {peer} "
                f"rid={header.request_id}"
            )
            return
        # HARD VALIDATION (review requirement): reject the flow outright
        # unless every announced chunk satisfies
        # 0 < size <= min(max_chunk_size_bytes, arena_capacity). An oversized
        # head chunk can never be allocated; it would age into drain mode and
        # stall ALL remote granting until the flow is forgotten.
        cap = min(self._max_chunk, self._sched.arena_capacity)
        bad = next((s for s in chunk_sizes if s <= 0 or s > cap), None)
        if bad is not None:
            logger.warning(
                f"bounce_v2({self._self_name}): rejecting WANT from {peer} "
                f"rid={header.request_id}: chunk size {bad} outside (0, {cap}] "
                f"({len(chunk_sizes)} chunks announced)"
            )
            return
        self._bump("rx_wants")
        self._rx_flows[key] = len(chunk_sizes)
        grants = self._sched.on_want(key, chunk_sizes)
        self._bump("rx_credits_at_want", len(grants))
        self._send_grants(grants)

    def _on_data(self, peer: str, header: BounceMsgHeader, blob: bytes) -> None:
        runs = decode_scatter(blob, header)
        if runs is None:
            return
        key = self._make_key(peer, header.request_id)
        offset = header.region_handle
        # Drop a DATA whose region this flow no longer holds (cancelled /
        # reclaimed; possibly re-granted): scattering it would read another
        # flow's region and corrupt data.
        if not self._sched.held_by_flow(key, offset):
            return
        if offset in self._scattering:
            logger.warning(
                f"bounce_v2({self._self_name}): dropping duplicate DATA peer={peer} "
                f"rid={header.request_id} chunk={header.chunk_idx} region={offset}"
            )
            return
        region_bytes = self._sched.region_bytes(offset)
        region_base = self._arena_base + offset
        # Scatter validation check (1) — the granted region itself must lie
        # inside the registered arena. Per the C++ contract this check stays
        # in Python, which owns the arena bounds; the submit_scatter_runs
        # sink validates checks (2)/(3) (run/piece counts, per-run region
        # bounds — see BatchedCopyPool::submitScatterRuns).
        if region_bytes <= 0 or region_base + region_bytes > self._arena_base + self._arena_bytes:
            logger.warning(
                f"bounce_v2({self._self_name}): DATA region out of arena bounds peer={peer} "
                f"rid={header.request_id} region={offset}"
            )
            self._flow_chunk_done(key)
            self._send_grants(self._sched.on_scatter_done(key, offset))
            return
        # C++ scatter sink: hand the RAW wire runs to submit_scatter_runs
        # (validation checks (2)/(3) + run expansion + plan fill + launch in
        # ONE bound call). A validation failure surfaces as SCATTER_REJECTED
        # from the launch below and takes the no-ACK / release-region path
        # (via _finish_scatter(ok=False)).
        job = _ScatterJob(
            key=key,
            peer=peer,
            rid=header.request_id,
            chunk_idx=header.chunk_idx,
            offset=offset,
            runs=runs,
            region_base=region_base,
            region_bytes=region_bytes,
        )
        self._scattering[offset] = False
        if runs.shape[0] == 0:
            self._bump("rx_data")
            self._finish_scatter(job, ok=True)  # empty plan: vacuous success
            return
        if self._launch_scatter(job):
            # Launched / backlogged / launch-failed — but NOT validation-
            # rejected, which must not count as accepted DATA.
            self._bump("rx_data")

    def _submit_scatter(self, job: _ScatterJob) -> int:
        """Submit one scatter job to the copy pool's raw-runs sink. Returns
        the completion id, BUSY, or SCATTER_REJECTED."""
        # A 1-D contiguous structured array views as one flat u8 blob of
        # n*36 bytes — exactly the wire payload the binding expects.
        return int(
            self._submit_scatter_fn(job.region_base, job.region_bytes, job.runs.view(np.uint8))
        )

    def _launch_scatter(self, job: _ScatterJob) -> bool:
        """Launch (or backlog) one scatter job. Returns False only when the
        C++ sink REJECTED the runs (bounds/size validation — details logged
        in C++): the job is then terminal with the no-ACK semantics
        (_finish_scatter(ok=False) releases the region and settles the flow
        accounting without acking)."""
        try:
            copy_id = self._submit_scatter(job)
        except RuntimeError as e:
            logger.warning(
                f"bounce_v2({self._self_name}): scatter launch failed rid={job.rid} "
                f"chunk={job.chunk_idx}: {e}"
            )
            self._finish_scatter(job, ok=False)
            return True
        if copy_id == self._pool_rejected:
            logger.warning(
                f"bounce_v2({self._self_name}): scatter runs rejected by validation "
                f"rid={job.rid} chunk={job.chunk_idx} region={job.offset}; no ACK"
            )
            self._finish_scatter(job, ok=False)
            return False
        if copy_id == self._pool_busy:
            self._scatter_backlog.append(job)  # retried next tick (raw runs kept)
            return True
        with self._req_mu:
            dispatches = self._register_route_locked(copy_id, ("scatter", job))
        for cid, route, kind, ok in dispatches:
            self._dispatch_row(route, cid, kind, bool(ok))
        return True

    def _retry_scatter_backlog(self) -> bool:
        did_work = False
        while self._scatter_backlog:
            job = self._scatter_backlog[0]
            try:
                copy_id = self._submit_scatter(job)
            except RuntimeError as e:
                logger.warning(
                    f"bounce_v2({self._self_name}): scatter launch failed rid={job.rid} "
                    f"chunk={job.chunk_idx}: {e}"
                )
                self._scatter_backlog.popleft()
                self._finish_scatter(job, ok=False)
                did_work = True
                continue
            if copy_id == self._pool_rejected:
                # Deterministically impossible on a retry (validation runs
                # before the BUSY check and the inputs are unchanged), but
                # handled defensively with the same no-ACK terminal.
                self._scatter_backlog.popleft()
                self._finish_scatter(job, ok=False)
                did_work = True
                continue
            if copy_id == self._pool_busy:
                break
            self._scatter_backlog.popleft()
            with self._req_mu:
                dispatches = self._register_route_locked(copy_id, ("scatter", job))
            for cid, route, kind, ok in dispatches:
                self._dispatch_row(route, cid, kind, bool(ok))
            did_work = True
        return did_work

    def _purge_scatter_backlog(self, drop: Callable[[_ScatterJob], bool]) -> None:
        """Drop every queued (never-launched) scatter job matching ``drop``,
        un-tracking its region from ``_scattering``. A stale backlog job of a
        reclaimed flow would otherwise be submitted later and scatter into
        final KV addresses that may already be freed/reused (silent KV
        corruption). ORDERING CONTRACT: this MUST run BEFORE the caller
        snapshots ``busy=set(self._scattering)`` for the scheduler reclaim —
        the dropped regions are then reclaimed exactly once through their
        flow's held set (freed or quarantined by the reclaim) instead of
        being deferred as busy orphans whose scatter completion never comes
        (a leak) or being both orphaned and flow-freed (a double release).
        Reactor-thread-only."""
        if not self._scatter_backlog:
            return
        keep: deque[_ScatterJob] = deque()
        for job in self._scatter_backlog:
            if drop(job):
                self._scattering.pop(job.offset, None)
            else:
                keep.append(job)
        self._scatter_backlog = keep

    def _finish_scatter(self, job: _ScatterJob, ok: bool) -> None:
        """Scatter reached a terminal state: free/settle the region, batch
        the ACK. A failed scatter sends NO ACK (the sender must time out, not
        believe corrupt/absent data landed) but still releases the region."""
        orphaned = self._scattering.pop(job.offset, False)
        if orphaned:
            grants = self._sched.free_orphan_region(job.offset)
        else:
            grants = self._sched.on_scatter_done(job.key, job.offset)
            # Chunk-terminal for the flow accounting even on failure (a
            # failed scatter would otherwise leak the _rx_flows entry).
            self._flow_chunk_done(job.key)
        if ok and not orphaned:
            self._bump("rx_scatter_ok")
            self._ack_batch.setdefault(job.peer, {}).setdefault(job.rid, []).append(
                AckEntry(job.chunk_idx, job.offset)
            )
        self._send_grants(grants)

    def _flow_chunk_done(self, key: str) -> None:
        """One announced chunk of flow ``key`` reached a terminal state
        (scattered OK, rejected by validation, or failed to scatter):
        decrement the outstanding-chunk count, dropping the flow entry at
        zero. No-op for already-reclaimed flows."""
        remaining = self._rx_flows.get(key)
        if remaining is None:
            return
        if remaining <= 1:
            self._rx_flows.pop(key, None)
        else:
            self._rx_flows[key] = remaining - 1

    def _flush_acks(self) -> None:
        if not self._ack_batch:
            return
        batch, self._ack_batch = self._ack_batch, {}
        for peer, by_rid in batch.items():
            for rid, entries in by_rid.items():
                self._bump("rx_ack_entries", len(entries))
                self._send_to(peer, encode_ack(rid, entries))

    def _check_receiver_lease(self) -> None:
        """Throttled lease sweep + quarantine reap (cadence follows the
        smallest enabled timeout / 10, clamped to [50 ms, 1 s], like the C++)."""
        now = time.monotonic()
        if now < self._next_lease_sweep:
            return
        lease_ms = self._cfg.receiver_flow_timeout_ms or 0
        quarantine_ms = self._cfg.quarantine_ms or 0
        smallest = min((v for v in (lease_ms, quarantine_ms) if v > 0), default=0)
        if smallest <= 0:
            self._next_lease_sweep = now + 1.0
            return
        self._next_lease_sweep = now + min(max(smallest / 10000.0, 0.05), 1.0)
        # Open-coded composition of stale_flows + forget + reap_quarantine
        # (instead of the scheduler's check_timeouts convenience) so the stale
        # set is computed EXACTLY ONCE and reused for both the backlog purge
        # and the reclaim — a flow turning stale between two separate sweeps
        # could otherwise be reclaimed with its stale backlog job left behind.
        stale = self._sched.stale_flows(lease_ms / 1000.0) if lease_ms > 0 else []
        if stale:
            # Purge the stale flows' queued scatters BEFORE computing the busy
            # set (see _purge_scatter_backlog's ordering contract): their
            # regions are then quarantined exactly once through each flow's
            # held set below, never stranded as orphans, and never scattered
            # into KV addresses freed after the reclaim.
            stale_set = set(stale)
            self._purge_scatter_backlog(lambda job: job.key in stale_set)
        quarantine_s = quarantine_ms / 1000.0
        grants: list[Grant] = []
        for flow in stale:
            flow_grants, deferred = self._sched.forget(
                flow, busy=set(self._scattering), quarantine_s=quarantine_s
            )
            grants.extend(flow_grants)
            for off in deferred:
                self._scattering[off] = True
            self._rx_flows.pop(flow, None)
            logger.warning(
                f"bounce_v2({self._self_name}): flow lease expired (no progress within "
                f"{lease_ms} ms) flow={flow!r} -> reclaimed (regions quarantined "
                f"{quarantine_ms} ms before reuse)"
            )
        grants.extend(self._sched.reap_quarantine())
        self._send_grants(grants)

    def _do_forget_peer(self, peer: str, victim_rids: list[int]) -> None:
        """Runs ON the reactor thread; reclaims STATE ONLY (the DEALER route
        was already dropped synchronously by ``forget_peer`` — removing it
        here again could erase a route a subsequent ``add_peer`` re-created).

        Sender: fail EXACTLY ``victim_rids`` — the requests snapshotted by
        ``forget_peer`` at call time (already-resolved rids are skipped; rids
        are never reused). Requests submitted after forget_peer returned,
        e.g. right after a compatible re-registration, are untouched. Cancels
        emitted toward the removed peer are dropped by ``_send_to`` with a
        warning (same accepted behavior as the C++).

        Receiver: drop the peer's queued scatters, reclaim its flows by peer
        prefix AT EXECUTION TIME (busy regions defer as orphans; the rest
        quarantine — the gone peer's NIC may still be writing them). The
        prefix cut cannot be snapshotted like the sender side: ``_rx_flows``
        and the backlog are reactor-thread-only (single-owner invariant), so
        the caller thread must not read them. The residual window is one
        tick: a WANT from a re-registered live peer that lands between the
        forget_peer call and this command can have its infant flow reclaimed;
        that remote sender's request then fails via its own timeout (never a
        hang or corruption — its regions quarantine normally), and this is
        no worse than the C++ drainForgets, which also cuts by peer name at
        execution time."""
        with self._req_mu:
            victims = [
                (rid, self._requests[rid])
                for rid in victim_rids
                if rid in self._requests and self._requests[rid].peer == peer
            ]
        for rid, req in victims:
            self._fail_request(rid, req, FAIL_PEER_DROPPED)
        # Purge the peer's queued scatters BEFORE computing the busy set (see
        # _purge_scatter_backlog's ordering contract).
        self._purge_scatter_backlog(lambda job: job.peer == peer)
        quarantine_s = (self._cfg.quarantine_ms or 0) / 1000.0
        grants, deferred = self._sched.forget_prefix(
            peer + _FLOW_SEP, busy=set(self._scattering), quarantine_s=quarantine_s
        )
        for off in deferred:
            self._scattering[off] = True
        prefix = peer + _FLOW_SEP
        for key in [k for k in self._rx_flows if k.startswith(prefix)]:
            del self._rx_flows[key]
        self._send_grants(grants)

    # ------------------------------------------------------------------ #
    # channel helpers (any thread; _ch_mu serializes all DEALER access)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_key(peer: str, rid: int) -> str:
        return f"{peer}{_FLOW_SEP}{rid}"

    @staticmethod
    def _split_key(key: str) -> tuple[str, int]:
        peer, _, rid = key.rpartition(_FLOW_SEP)
        return peer, int(rid)

    def _send_to(self, peer: str, blob: bytes) -> None:
        with self._ch_mu:
            dealer = self._dealers.get(peer)
            if dealer is None:
                logger.warning(
                    f"bounce_v2({self._self_name}): send to unknown peer {peer} "
                    f"(add_peer first); dropped"
                )
                return
            try:
                dealer.send(blob, zmq.NOBLOCK)
            except zmq.Again:
                # Full queue: DROP rather than block the reactor/submitter;
                # the affected request degrades to a request-timeout failure.
                logger.warning(
                    f"bounce_v2({self._self_name}): send to {peer} dropped "
                    f"(queue full / peer stalled)"
                )
            except zmq.ZMQError as e:
                logger.warning(f"bounce_v2({self._self_name}): send to {peer} failed: {e}")

    def _send_grants(self, grants: list[Grant]) -> None:
        if not grants:
            return
        self._bump("rx_credits_sent", len(grants))
        by_flow: dict[str, list[CreditEntry]] = {}
        for g in grants:
            by_flow.setdefault(g.flow, []).append(
                # Carry OUR device id so the sender writes to the right GPU;
                # the region handle (arena offset) is echoed back in DATA.
                CreditEntry(g.addr, g.length, self._device_id, g.offset)
            )
        self._bump("rx_grant_msgs", len(by_flow))
        for flow, credits in by_flow.items():
            peer, rid = self._split_key(flow)
            self._send_to(peer, encode_grant(rid, credits))
