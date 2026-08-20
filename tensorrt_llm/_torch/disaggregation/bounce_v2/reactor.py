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
    state), request creation + eager pump under ``_req_mu``, WANT send under
    the channel lock. ``CreditScheduler`` is internally locked;
    ``BatchedCopyPool.submit_copy`` is thread-safe;
    ``NixlTransferAgent.post_transfer_1to1`` runs on the reactor thread AND
    on submit threads (a racing GRANT parks credits, then the submitter's
    eager pump attaches them: _pump_locked -> _post_write_locked) — every
    such call site holds ``_req_mu``, which serializes the posts.
  - ``_req_mu`` guards the sender request table AND ``_completions`` (the
    completion-id routing map): both are touched by submit threads (eager
    pump registers gather ids) and the reactor.
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

BLOCKING POINT: exactly one — ``zmq.Poller.poll(1 ms)`` over the ROUTER when
a tick found no work (GIL released). Completions ride the same <=1 ms cap via
``CompletionPoller.drain(0)`` each tick (the design's documented polling
fallback; no other sleeps exist in the loop).
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
#: Reactor idle-poll cap (the design's 1 ms tick).
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
    """One in-flight chunk of a sender request."""

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


@dataclass
class _ScatterJob:
    """Receiver-side prepared scatter (run-expanded copy arrays)."""

    key: str
    peer: str
    rid: int
    chunk_idx: int
    offset: int  # arena region offset (the DATA region handle)
    srcs: np.ndarray  # [n] uint64 absolute source addresses inside the region
    dsts: np.ndarray  # [n] uint64 final KV destination addresses
    sizes: np.ndarray  # [n] uint32


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
        self._max_descs_per_chunk = min(
            self._max_plan_entries, max(1024, config.max_chunk_size_bytes // 256)
        )
        # EXPERIMENTAL C++ chain (TRTLLM_BOUNCE_V2_EXP_CPP_CHAIN): the C++
        # poll thread posts a credited chunk's RDMA write as soon as its
        # gather event fires (see _arm_chain_locked); Python then sees ONE
        # completion per chunk instead of gather + write.
        self._chain_fn = None
        if config.enable_cpp_chain:
            self._chain_fn = getattr(raw_agent, "post_transfer_1to1_on_event", None)
            if self._chain_fn is None:
                logger.warning(
                    f"bounce_v2({self_name}): TRTLLM_BOUNCE_V2_EXP_CPP_CHAIN requested but the "
                    f"agent binding lacks post_transfer_1to1_on_event; using the classic "
                    f"gather->post path"
                )
            else:
                logger.info(f"bounce_v2({self_name}): C++ gather->RDMA chain enabled")

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
        with self._req_mu:
            rid = self._next_rid
            self._next_rid += 1
            req = _Request(
                peer=peer,
                plan=plan,
                num_chunks=plan.num_chunks,
                future=future,
                last_progress=time.monotonic(),
            )
            self._requests[rid] = req
        # WANT carries our endpoint so the receiver self-bootstraps the
        # reverse route (sent outside _req_mu, like the C++).
        self._send_to(peer, encode_want(rid, chunk_bytes, self._endpoint))
        if self._cfg.enable_eager_gather:
            with self._req_mu:
                # A racing GRANT may already have pumped (or failed) the request.
                cur = self._requests.get(rid)
                if cur is not None:
                    self._pump_locked(rid, cur)
        return future

    def shutdown(self) -> None:
        """Stop the reactor, join it, and resolve every pending future
        (``failAll`` semantics). Idempotent. The caller must still drain the
        GPU before tearing down the copy pool / arena (engine does)."""
        if self._stop.is_set():
            self._thread.join(timeout=5)
            return
        self._stop.set()
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
                self._check_sender_timeouts()
                self._check_receiver_lease()
                self._flush_acks()
                self._maybe_log_stats()
                if not did_work:
                    # The ONE blocking point: <=1 ms on the ROUTER (GIL
                    # released). Completions ride the same cap via the
                    # non-blocking drain above.
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
            with self._req_mu:
                route = self._completions.pop(cid, None)
            if route is None:
                continue  # e.g. a completion for an already-failed request
            tag = route[0]
            if tag == "gather":
                self._on_gather_done(route[1], cid, bool(ok))
            elif tag == "xfer":
                self._on_xfer_done(route[1], cid, bool(ok), int(kind))
            elif tag == "scatter":
                self._finish_scatter(route[1], bool(ok))
            elif tag == "orphan_gather":
                self._send_grants(self._sched.release_local(route[1]))
            elif tag == "orphan_xfer":
                self._on_orphan_xfer_done(route[1], route[2])
        return True

    # ------------------------------------------------------------------ #
    # sender role
    # ------------------------------------------------------------------ #

    def _attach_credits_locked(self, rid: int, req: _Request) -> None:
        """Pair parked credits with already-posted (eager) chunks, strictly
        FIFO. Validates the mispair guard: a credit smaller than its chunk
        would make the RDMA write overflow into an adjacent flow's region on
        the peer -> the request FAILS IMMEDIATELY (via _fail_request_locked;
        the timeout sweep alone would never resolve it when
        request_timeout_ms <= 0 — the R5 guarantee)."""
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
                self._fail_request_locked(rid, req, FAIL_PROTOCOL)
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
            if target.state == _PostState.GATHERING:
                # Eager chunk still gathering: with the C++ chain enabled the
                # credit lets us arm the gather->post chain right now; losing
                # the race to the gather's own completion is fine — the
                # classic path then posts from _on_gather_done as before.
                self._arm_chain_locked(rid, req, target)
            elif target.state == _PostState.GATHERED:
                # Eagerly-gathered chunk was only waiting for its credit.
                self._post_write_locked(rid, req, target)
                if req.abandon_reason:
                    return
        if req.next_credit >= req.num_chunks and req.pending_credits:
            logger.warning(
                f"bounce_v2({self._self_name}): rid={rid} over-grant, dropping "
                f"{len(req.pending_credits)} extra credit(s)"
            )
            req.pending_credits.clear()

    def _pump_locked(self, rid: int, req: _Request) -> None:
        """Advance the request: attach credits, then launch gathers for as
        many chunks as credits/eager-budget/copy-streams allow. _req_mu held."""
        self._attach_credits_locked(rid, req)
        if req.abandon_reason:
            return
        grants: list[Grant] = []
        while req.next_post < req.num_chunks:
            chunk_idx = req.next_post
            chunk = req.plan.chunks[chunk_idx]
            have_credit = bool(req.pending_credits)
            if not have_credit:
                if not self._cfg.enable_eager_gather:
                    break  # classic path: gather starts only once granted
                if len(req.posted) >= self._cfg.max_inflight_chunks_per_request:
                    break  # eager gathers capped by the in-flight window
            if have_credit and chunk.packed_bytes > req.pending_credits[0].length:
                logger.warning(
                    f"bounce_v2({self._self_name}): rid={rid} chunk={chunk_idx} "
                    f"packed_bytes={chunk.packed_bytes} > granted region "
                    f"len={req.pending_credits[0].length} (GRANT mispair/reorder); abandoning flow"
                )
                req.abandon_reason = FAIL_PROTOCOL
                req.pending_credits.clear()
                self._fail_request_locked(rid, req, FAIL_PROTOCOL)
                break
            # Non-blocking staging: no region right now -> park; the reactor's
            # drain_pending_posts retries once an ACK frees space. Never blocks,
            # so oversubscription degrades to backpressure, not deadlock.
            local_off = self._sched.acquire_local(chunk.packed_bytes, eager=not have_credit)
            if local_off is None:
                break
            region_base = self._arena_base + local_off
            try:
                copy_id = self._pool.submit_copy(
                    chunk.src_ptrs,
                    (np.uint64(region_base) + chunk.bounce_offsets).astype(np.uint64),
                    chunk.sizes,
                )
            except RuntimeError as e:
                # Gather launch failed (CUDA error / plan overflow). No event
                # was registered, so the region is safe to recycle now; fail
                # deterministically like the C++ GatherFailed path.
                logger.warning(
                    f"bounce_v2({self._self_name}): rid={rid} chunk={chunk_idx} "
                    f"gather launch failed: {e}"
                )
                grants.extend(self._sched.release_local(local_off))
                self._fail_request_locked(rid, req, FAIL_GATHER)
                break
            if copy_id == self._pool_busy:
                grants.extend(self._sched.release_local(local_off))
                break  # every copy stream busy: retry next tick
            self._bump("tx_gather_credit" if have_credit else "tx_gather_eager")
            posted = _Posted(
                chunk_idx=chunk_idx,
                local_offset=local_off,
                write_bytes=chunk.packed_bytes,
                copy_id=copy_id,
            )
            if have_credit:
                credit = req.pending_credits.popleft()
                req.next_credit += 1
                posted.has_credit = True
                posted.remote_handle = credit.region_handle
                posted.remote_addr = credit.addr
                posted.remote_dev = credit.dev_id
            self._completions[copy_id] = ("gather", rid)
            req.posted.append(posted)
            req.next_post += 1
            req.last_progress = time.monotonic()
            if posted.has_credit:
                # Credited at launch: arm the C++ chain immediately (no-op
                # when the chain is disabled/unavailable).
                self._arm_chain_locked(rid, req, posted)
        self._send_grants(grants)

    def _drain_pending_posts(self) -> bool:
        did_work = False
        with self._req_mu:
            for rid in list(self._requests):
                req = self._requests.get(rid)
                if req is None:
                    continue
                if req.pending_credits or (
                    self._cfg.enable_eager_gather and req.next_post < req.num_chunks
                ):
                    before = (req.next_post, req.next_credit)
                    self._pump_locked(rid, req)
                    cur = self._requests.get(rid)
                    did_work |= cur is None or (cur.next_post, cur.next_credit) != before
        return did_work

    def _arm_chain_locked(self, rid: int, req: _Request, posted: _Posted) -> bool:
        """Try to arm the C++ gather->RDMA chain for a credited, still-
        gathering chunk (_req_mu held). On success the chunk's ONE remaining
        completion is the reserved xfer id: the gather completion is consumed
        in C++ (never drained), the write posts on the C++ poll thread, and
        the classic copy_id route is dropped; the chunk moves to WRITING so
        the failure path treats its staging region exactly like an in-flight
        write (recycled only once the chain reaches a terminal state — the
        reserved id resolves only after the gather is terminal AND the write,
        if posted, is terminal). Returns False when the chain is disabled,
        the chunk is not eligible, or the arm lost the race to the gather's
        own completion — the classic two-hop path then proceeds unchanged."""
        if self._chain_fn is None or not posted.has_credit:
            return False
        if posted.state != _PostState.GATHERING:
            return False
        reserved = self._chain_fn(
            posted.copy_id,
            self._arena_base + posted.local_offset,
            posted.remote_addr,
            posted.write_bytes,
            self._device_id,
            posted.remote_dev,
            req.peer,
            self._poller,
        )
        if reserved < 0:
            # Lost the race to the gather's own completion (or C++ refused):
            # the classic path proceeds; count it so an A/B run can tell how
            # often the chain actually fired vs fell back.
            self._bump("tx_chain_arm_race")
            return False
        self._bump("tx_chain_armed")
        self._completions.pop(posted.copy_id, None)
        posted.xfer_id = int(reserved)
        posted.state = _PostState.WRITING
        self._completions[posted.xfer_id] = ("xfer", rid)
        req.last_progress = time.monotonic()
        return True

    def _post_write_locked(self, rid: int, req: _Request, posted: _Posted) -> None:
        """Gathered + credited -> post the 1:1 RDMA write. _req_mu held."""
        xfer_id = self._agent.post_transfer_1to1(
            self._arena_base + posted.local_offset,
            posted.remote_addr,
            posted.write_bytes,
            self._device_id,
            posted.remote_dev,
            req.peer,
            self._poller,
        )
        if xfer_id < 0:
            self._fail_request_locked(rid, req, FAIL_WRITE)
            return
        self._bump("tx_post_classic")
        posted.xfer_id = xfer_id
        posted.state = _PostState.WRITING
        self._completions[xfer_id] = ("xfer", rid)
        req.last_progress = time.monotonic()

    def _on_gather_done(self, rid: int, copy_id: int, ok: bool) -> None:
        self._bump("tx_gather_events")
        with self._req_mu:
            req = self._requests.get(rid)
            if req is None:
                return
            # Route by the completion id: gathers of one request may finish
            # OUT OF ORDER across copy streams, so "oldest gathering" would
            # be wrong (it could post a write for an unfinished gather).
            target = next((p for p in req.posted if p.copy_id == copy_id), None)
            if not ok:
                # The failed gather's completion id was already consumed by
                # the drain and the kernel is DONE with the staging region:
                # advance the chunk to a terminal state BEFORE failing, so
                # _fail_request_locked releases its region now instead of
                # re-registering the consumed id as an orphan_gather that
                # would never fire (poller ids report exactly once).
                if target is not None and target.state == _PostState.GATHERING:
                    target.state = _PostState.GATHERED
                self._fail_request_locked(rid, req, FAIL_GATHER)
                return
            if target is None or target.state != _PostState.GATHERING:
                return
            target.state = _PostState.GATHERED
            if target.has_credit:
                self._post_write_locked(rid, req, target)

    def _on_xfer_done(self, rid: int, xfer_id: int, ok: bool, kind: int = _KIND_XFER) -> None:
        data_msg: Optional[tuple[str, bytes]] = None
        with self._req_mu:
            req = self._requests.get(rid)
            if req is None:
                return
            target = next((p for p in req.posted if p.xfer_id == xfer_id), None)
            if target is None:
                return
            if not ok:
                # The chunk's completion id was already consumed by the drain,
                # so the chunk must go terminal HERE: _fail_request_locked
                # would otherwise re-register the consumed id as an orphan
                # write that can never fire (wedging _orphan_writes and the
                # deferred cancel, and leaking the region). Every failed row
                # that lands here is terminal for the staging region:
                #   (id, KIND_XFER, 0)  classic write or chained post failed /
                #       shutdown -> the NIC/agent is DONE with the region;
                #   (id, KIND_EVENT, 0) chained gather failed -> the kernel is
                #       done and the write was never posted (NIC never saw the
                #       region) — reported as FAIL_GATHER to keep the reason
                #       accurate.
                # Guarded like the gather path: a duplicate delivery must not
                # relabel a chunk that is no longer WRITING.
                if target.state == _PostState.WRITING:
                    target.state = _PostState.SENT
                self._fail_request_locked(
                    rid, req, FAIL_GATHER if kind == _KIND_EVENT else FAIL_WRITE
                )
                return
            chunk = req.plan.chunks[target.chunk_idx]
            data_msg = (
                req.peer,
                encode_data(
                    rid, target.chunk_idx, req.num_chunks, target.remote_handle, chunk.scatter_runs
                ),
            )
            target.state = _PostState.SENT
            req.last_progress = time.monotonic()
        self._bump("tx_xfer_events")
        if data_msg is not None:
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
            if req.acked >= req.num_chunks:
                done_future = req.future
                del self._requests[header.request_id]
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
            self._pump_locked(header.request_id, req)

    def _check_sender_timeouts(self) -> None:
        if self._cfg.request_timeout_ms <= 0:
            return
        now = time.monotonic()
        if now < self._next_sender_sweep:
            return
        self._next_sender_sweep = now + _SENDER_SWEEP_S
        limit = self._cfg.request_timeout_ms / 1000.0
        with self._req_mu:
            stuck = [rid for rid, req in self._requests.items() if now - req.last_progress > limit]
            for rid in stuck:
                req = self._requests.get(rid)
                if req is not None:
                    self._fail_request_locked(rid, req, FAIL_NO_PROGRESS)

    def _fail_request_locked(self, rid: int, req: _Request, reason: str) -> None:
        """Terminal failure path (mirrors the C++ failRequest). _req_mu held.

        Regions are recycled only once nothing can still touch their memory:
        a WRITING chunk's region defers until its RDMA completion (the NIC
        may still read it); a GATHERING chunk's region defers until its copy
        event fires (the kernel may still write it); GATHERED/SENT recycle
        now. The chunk whose FAILED completion triggered this call must be
        advanced to a terminal state (GATHERED/SENT) by its handler first:
        its completion id was already consumed by the drain, so registering
        it as an orphan here would never fire and leak the region. The
        cancel-WANT to the receiver is deferred while any write is in flight
        (it is landing on the receiver's region; an early cancel would let
        the receiver re-grant that region under the write)."""
        if req.abandon_reason:
            reason = req.abandon_reason
        # Latch the terminal reason on the (now-dead) request object: callers
        # still iterating over it (_attach_credits_locked/_pump_locked after a
        # failed _post_write_locked) check abandon_reason and stop advancing.
        # Without the latch they would keep pumping a DELETED request — a
        # second failure would KeyError on `del self._requests[rid]`, a second
        # success would RDMA-write from an already-released staging region,
        # and every further iteration would leak an arena region.
        req.abandon_reason = reason
        logger.warning(
            f"bounce_v2({self._self_name}): request FAILED rid={rid} peer={req.peer} "
            f'reason="{reason}" acked={req.acked} posted={req.next_post}/{req.num_chunks}'
        )
        grants: list[Grant] = []
        deferred_write = False
        for p in req.posted:
            if p.state == _PostState.WRITING:
                self._completions[p.xfer_id] = ("orphan_xfer", p.local_offset, rid)
                self._orphan_writes[rid] = self._orphan_writes.get(rid, 0) + 1
                deferred_write = True
            elif p.state == _PostState.GATHERING:
                self._completions[p.copy_id] = ("orphan_gather", p.local_offset)
            else:  # GATHERED / SENT: nothing touches the region anymore
                grants.extend(self._sched.release_local(p.local_offset))
        if deferred_write:
            self._pending_cancels[rid] = req.peer
        else:
            self._send_to(req.peer, encode_cancel(rid, self._endpoint))
        future = req.future
        del self._requests[rid]
        self._send_grants(grants)
        _resolve(future, BounceResult(False, reason))

    def _on_orphan_xfer_done(self, local_offset: int, rid: int) -> None:
        """A failed request's in-flight write reached a terminal state: its
        staging region is recyclable, and once the rid's last such write
        drains, the deferred cancel may finally be sent (the receiver can
        then safely reclaim its regions)."""
        self._send_grants(self._sched.release_local(local_offset))
        with self._req_mu:
            remaining = self._orphan_writes.get(rid, 0) - 1
            if remaining > 0:
                self._orphan_writes[rid] = remaining
                return
            self._orphan_writes.pop(rid, None)
            peer = self._pending_cancels.pop(rid, None)
        if peer is not None:
            self._send_to(peer, encode_cancel(rid, self._endpoint))

    def _fail_all(self, reason: str) -> None:
        with self._req_mu:
            requests = list(self._requests.values())
            self._requests.clear()
            self._completions.clear()
            self._orphan_writes.clear()
            self._pending_cancels.clear()
        for req in requests:
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
        job = self._validate_and_expand_runs(
            key, peer, header, runs, offset, region_base, region_bytes
        )
        if job is None:
            # Bounds/size violation: no scatter, NO ACK (a false ACK would
            # claim the data landed), but the region is released so it cannot
            # leak — the sender times out. Still chunk-terminal for the flow
            # accounting, or the _rx_flows entry would leak forever.
            self._flow_chunk_done(key)
            self._send_grants(self._sched.on_scatter_done(key, offset))
            return
        self._bump("rx_data")
        self._scattering[offset] = False
        if job.srcs.shape[0] == 0:
            self._finish_scatter(job, ok=True)  # empty plan: vacuous success
            return
        self._launch_scatter(job)

    def _validate_and_expand_runs(
        self,
        key: str,
        peer: str,
        header: BounceMsgHeader,
        runs: np.ndarray,
        offset: int,
        region_base: int,
        region_bytes: int,
    ) -> Optional[_ScatterJob]:
        """Bounds-check every scatter run against THIS flow's granted region
        (per-run: exact Python-int arithmetic — hostile u64 strides cannot
        wrap), then expand runs to per-piece (src, dst, size) arrays for the
        copy op. Returns None on any violation (logged)."""
        n_runs = int(runs.shape[0])
        if region_bytes <= 0 or region_base + region_bytes > self._arena_base + self._arena_bytes:
            logger.warning(
                f"bounce_v2({self._self_name}): DATA region out of arena bounds peer={peer} "
                f"rid={header.request_id} region={offset}"
            )
            return None
        total_pieces = int(runs["count"].astype(np.uint64).sum())
        if n_runs > self._max_plan_entries or total_pieces > self._max_plan_entries:
            logger.warning(
                f"bounce_v2({self._self_name}): rejected scatter with {n_runs} runs / "
                f"{total_pieces} pieces (max {self._max_plan_entries}) peer={peer} "
                f"rid={header.request_id} chunk={header.chunk_idx}"
            )
            return None
        srcs = np.empty(total_pieces, dtype=np.uint64)
        dsts = np.empty(total_pieces, dtype=np.uint64)
        sizes = np.empty(total_pieces, dtype=np.uint32)
        pos = 0
        for r in runs:
            count = int(r["count"])
            b_off = int(r["bounce_offset"])
            b_stride = int(r["bounce_stride"])
            d_addr = int(r["dst_addr"])
            d_stride = int(r["dst_stride"])
            piece = int(r["piece_size"])
            span = (count - 1) * b_stride + piece
            if count < 1 or b_off > region_bytes or span > region_bytes - b_off:
                logger.warning(
                    f"bounce_v2({self._self_name}): scatter run out of region bounds peer={peer} "
                    f"rid={header.request_id} chunk={header.chunk_idx} "
                    f"(off={b_off} span={span} region={region_bytes})"
                )
                return None
            idx = np.arange(count, dtype=np.uint64)
            srcs[pos : pos + count] = np.uint64(region_base + b_off) + idx * np.uint64(b_stride)
            dsts[pos : pos + count] = np.uint64(d_addr) + idx * np.uint64(d_stride)
            sizes[pos : pos + count] = piece
            pos += count
        return _ScatterJob(
            key=key,
            peer=peer,
            rid=header.request_id,
            chunk_idx=header.chunk_idx,
            offset=offset,
            srcs=srcs,
            dsts=dsts,
            sizes=sizes,
        )

    def _launch_scatter(self, job: _ScatterJob) -> None:
        try:
            copy_id = self._pool.submit_copy(job.srcs, job.dsts, job.sizes)
        except RuntimeError as e:
            logger.warning(
                f"bounce_v2({self._self_name}): scatter launch failed rid={job.rid} "
                f"chunk={job.chunk_idx}: {e}"
            )
            self._finish_scatter(job, ok=False)
            return
        if copy_id == self._pool_busy:
            self._scatter_backlog.append(job)  # retried next tick
            return
        with self._req_mu:
            self._completions[copy_id] = ("scatter", job)

    def _retry_scatter_backlog(self) -> bool:
        did_work = False
        while self._scatter_backlog:
            job = self._scatter_backlog[0]
            try:
                copy_id = self._pool.submit_copy(job.srcs, job.dsts, job.sizes)
            except RuntimeError as e:
                logger.warning(
                    f"bounce_v2({self._self_name}): scatter launch failed rid={job.rid} "
                    f"chunk={job.chunk_idx}: {e}"
                )
                self._scatter_backlog.popleft()
                self._finish_scatter(job, ok=False)
                did_work = True
                continue
            if copy_id == self._pool_busy:
                break
            self._scatter_backlog.popleft()
            with self._req_mu:
                self._completions[copy_id] = ("scatter", job)
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
            for rid in victim_rids:
                req = self._requests.get(rid)
                if req is not None and req.peer == peer:
                    self._fail_request_locked(rid, req, FAIL_PEER_DROPPED)
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
