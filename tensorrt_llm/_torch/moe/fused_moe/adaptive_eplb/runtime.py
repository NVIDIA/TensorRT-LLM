# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime of adaptive EPLB: statistics windows, the decision process and installs.

The main thread advances the runtime once per executor step and never waits; a worker thread runs
the blocking work (the all-gathers, the decision process's messages, the expert copies). Every rank
decides alike in its own decision process. An install copies new replicas into free slots, and its
route tables take effect from the call after every rank's copies finished; ranks whose decisions
differ stop the run before that. The decision process imports this module with only torch, numpy
and this package available, so import other ``tensorrt_llm`` modules inside functions.
"""

from __future__ import annotations

import atexit
import dataclasses
import hashlib
import math
import mmap
import os
import queue
import socket
import subprocess
import sys
import threading
import traceback
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING

import numpy as np
import torch

from .policy import (
    EvaluationScheduler,
    ExpertGeometry,
    TimeModel,
    active_slots,
    agreed_timings,
    build_route_tables,
    choose_layers_to_install,
    evaluate_gains,
    plan_layout,
    solve_shares,
)

if TYPE_CHECKING:
    from .balancer import AdaptiveMoeLayerLoadBalancer
    from .transport import PeerSlotTransport

    Gather = Callable[[torch.Tensor, list[torch.Tensor], threading.Event], None]
    GenerationProgress = Callable[[int, int], tuple[int, int, int]]

_DECISION_MAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decision_main.py")
ROWS_PER_EXPERT = 512
"""int: Rows a statistics window collects per expert, so that an expert's rows per forward carry a
relative standard error of about 1 / sqrt(512) = 4.4 % at any batch size or draft length: a window
lasts as many forwards as the previous window's rows per forward need to reach it. The first window
lasts one forward; with fewer rows than that it only measures the rows per forward."""
REQUESTS_PER_CALL = 64
"""int: Generation requests whose progress one call reads (about 20 us of main-thread time)."""
CLOSE_WAIT_S = 30.0
"""float: Seconds a close waits for the worker thread, again after telling it to leave."""
GATHER_SLICE_S = 1.0
"""float: Seconds of one wait in an all-gather between looks at whether to leave it."""
FIELDS = (
    "remaining",
    "requests",
    "accepted",
    "stats_copy_ms",
    "applied",
    "applied_layers",
    "applied_ms",
)
"""tuple: The values at the end of a rank's window statistics, the float32 array the ranks
all-gather, which holds ``rows [L, E] | forwards [L] | applied window [L] | FIELDS``.

``remaining``, ``requests`` and ``accepted`` sum the tokens the generation requests still have to
produce, the requests, and the tokens they accept per step. ``stats_copy_ms`` is the GPU time of
copying the statistics; ``applied``, ``applied_layers`` and ``applied_ms`` are the window, layer
count and GPU time of the last route tables applied.
"""
ROW = ("copied_call", "moved", "copy_ms", "digest")
"""tuple: The float64 values a rank all-gathers after its copies of an update ran: the call they
finished in, the layers it moved, their copy time and the update's digest."""


def window_forwards(rows_per_forward: float, experts: int) -> int:
    """Forwards a window needs to collect :data:`ROWS_PER_EXPERT` rows per expert."""
    return max(1, math.ceil(ROWS_PER_EXPERT * experts / rows_per_forward))


def window_stats_size(layers: int, experts: int) -> int:
    return layers * (experts + 2) + len(FIELDS)


@dataclass(frozen=True)
class DecisionConfig:
    """Inputs of the decision process; ``home_layouts`` is ``[L, num_slots]`` int64 and
    ``copy_ms`` the copy time per replica measured at start-up, which stays among the install
    timings the copies are priced by."""

    geometry: ExpertGeometry
    model: TimeModel
    home_layouts: np.ndarray
    ep_rank: int
    ranks_per_node: int
    stats_fd: int = -1
    copy_ms: float | None = None


@dataclass
class LayoutUpdate:
    """The decision of one window.

    It holds the new route tables of the chosen layers, the layers whose layout changes
    (``moved``, copied as one timed batch) and the expert copies this rank makes for them
    (``copy_items``). ``digest`` lets the ranks check that they all decided the same.
    """

    window: int
    copy_items: dict[int, list[tuple[int, int, int]]] = field(default_factory=dict)
    tables: dict[int, np.ndarray] = field(default_factory=dict)
    moved: list[int] = field(default_factory=list)
    digest: int = 0

    def fingerprint(self) -> int:
        """48-bit digest (exact in a float64) of the tables and moved layers every rank shares."""
        digest = hashlib.blake2b(digest_size=6)
        for layer in sorted(self.tables):
            digest.update(np.int64(layer).tobytes() + self.tables[layer].tobytes())
        digest.update(np.asarray(self.moved, dtype=np.int64).tobytes())
        return int.from_bytes(digest.digest(), "little")


def copy_items(
    old: np.ndarray, new: np.ndarray, geometry: ExpertGeometry, rank: int, ranks_per_node: int
) -> list[tuple[int, int, int]]:
    """``(dst_local_slot, src_rank, src_local_slot)`` of every expert ``rank`` copies from ``old``
    to ``new``, from its own node first. Raises ``ValueError`` when a destination is not free."""
    local = geometry.local_slots
    node = rank // ranks_per_node
    items = []
    for slot in range(rank * local, (rank + 1) * local):
        expert = int(new[slot])
        if expert < 0 or old[slot] == expert:
            continue
        if old[slot] >= 0:
            raise ValueError(f"slot {slot} holds expert {old[slot]}, not free for expert {expert}")
        holders = np.flatnonzero(old == expert)
        src = int(min(holders, key=lambda s: (s // local // ranks_per_node != node, s)))
        items.append((slot - rank * local, src // local, src % local))
    return items


class _Decider:
    """State of the decision process: the committed layouts and every timing of the run."""

    def __init__(self, config: DecisionConfig, stats: np.ndarray) -> None:
        self.config = config
        self.stats = stats
        self.layout = np.array(config.home_layouts, dtype=np.int64)
        self.shares = (self.layout >= 0).astype(np.float64)
        self.committed = np.full(len(self.layout), -1)
        self.install_timings: dict[int, tuple[int, float]] = {}
        self.stats_copies: list[float] = []
        self.evaluations = EvaluationScheduler(config.model, config.geometry)
        self.active = active_slots(config.geometry)
        # (apply call, ms per replica) of the timed copies; the start-up one, at call inf, always counts.
        self.copy_samples = [] if config.copy_ms is None else [(math.inf, config.copy_ms)]
        self.forwards = 1
        self.next_forwards: int | None = None
        self.open_call = 0
        self.awaiting_copy: dict[int, tuple] = {}
        # (window, rows per forward [L, E], the requests they stand for) of the last two windows.
        self.kept: deque[tuple[int, np.ndarray, float]] = deque(maxlen=2)
        self.close_requests = self.largest_change = 0.0

    def handle(self, message: tuple) -> tuple:
        if message[0] == "window":
            return ("update", self.decide(message[1]))
        return self.schedule(*message[1:])

    def schedule(self, window: int, rows: np.ndarray) -> tuple:
        """The next window's open call once every rank's copies of ``window`` ran."""
        col = {name: rows[:, i] for i, name in enumerate(ROW)}
        apply_call = int(col["copied_call"].max()) + 1
        per_rank = [[(window, int(n), float(ms))] for n, ms in zip(col["moved"], col["copy_ms"])]
        # got: the replicas the update's busiest receiving GPU copied
        close, growth, steps_left, got = self.awaiting_copy.pop(window)
        self.copy_samples += [(apply_call, ms / got) for _, n, ms in agreed_timings(per_rank) if n]
        self.forwards = self.next_forwards or self.forwards
        plan = self.evaluations.plan(
            apply_call,
            close,
            self.forwards,
            self.stats_copies,
            [(group, *timing) for group, timing in sorted(self.install_timings.items())],
            growth,
            steps_left,
        )
        self.open_call = apply_call + plan.open_after
        return ("schedule", window + 1, self.open_call, apply_call, plan, self.forwards)

    def decide(self, window: int) -> LayoutUpdate:
        cfg = self.config
        stats = self.stats
        geometry = cfg.geometry
        layers, slots = self.layout.shape
        experts = geometry.num_experts
        close = self.open_call + self.forwards
        tail = {name: stats[:, layers * (experts + 2) + i] for i, name in enumerate(FIELDS)}
        applied = [
            [(int(g), int(n), float(ms))] if g >= 0 else []
            for g, n, ms in zip(tail["applied"], tail["applied_layers"], tail["applied_ms"])
        ]
        self.install_timings.update((g, (n, ms)) for g, n, ms in agreed_timings(applied))
        if (tail["stats_copy_ms"] > 0).all():
            self.stats_copies.append(float(tail["stats_copy_ms"].max()))
        update = LayoutUpdate(window)
        self.awaiting_copy[window] = (close, None, None, 0)  # until the window gives a decision
        forwards = stats[:, layers * experts : layers * (experts + 1)].astype(np.float64)
        if forwards.min() < 1:  # a rank counted no forward of some layer: this window gives no rows
            return update
        # Rows per forward of every expert: every rank's rows over its own forwards, summed.
        routed = stats[:, : layers * experts].astype(np.float64).reshape(len(stats), layers, -1)
        rows = (routed / forwards[:, :, None]).sum(0)
        per_forward = float(rows.sum(1).mean())
        counted = float(routed.sum((0, 2)).mean())
        sizing = self.next_forwards is None and counted < ROWS_PER_EXPERT * experts
        if per_forward > 0.0:
            self.next_forwards = window_forwards(per_forward, experts)
        if sizing:
            return update
        total = {name: float(values.astype(np.float64).sum()) for name, values in tail.items()}
        steps_left = None
        if min(total["accepted"], total["requests"]) > 0:
            per_step = total["accepted"] / total["requests"]  # tokens a request accepts per step
            steps_left = total["remaining"] / per_step / total["requests"]
        # A batch that changed between two closes more than it ever did (a steady batch never does, a
        # draining one keeps doing so) is priced where the new layouts serve it: the requests at this
        # close carried a window ahead at that rate, over their mean across the window.
        requests, before = total["requests"], self.close_requests
        change = abs(requests / before - 1.0) if min(requests, before) > 0.0 else 0.0
        self.close_requests, ahead = requests, None
        if change > self.largest_change:
            ahead = requests * requests / before
            rows = rows * (ahead / ((before + requests) / 2.0))
        self.largest_change = max(self.largest_change, change)
        requests = ahead or ((before + requests) / 2.0 if before > 0.0 else requests)
        applied_window = stats[:, layers * (experts + 1) : layers * (experts + 2)].min(0)
        planned = np.flatnonzero(self.committed <= applied_window)
        # The busiest rank's rows per forward size the CUDA graph bucket the steps pad to.
        model = cfg.model.at(float((routed.sum(2) / forwards).mean(1).max()))
        self.evaluations.model = model
        new_layout = np.empty((len(planned), slots), dtype=np.int64)
        new_shares, resolved = np.empty((2, len(planned), slots))
        home, installed = cfg.home_layouts[planned], self.layout[planned]
        # This window's rows, replaced below by an earlier window's rows where one is kept.
        previous_load = rows[planned]
        for i, layer in enumerate(planned.tolist()):
            plan = plan_layout(rows[layer], geometry, home[i], installed[i], self.active)
            new_layout[i] = plan.layout
            new_shares[i] = solve_shares(rows[layer], new_layout[i], geometry, model)
            resolved[i] = solve_shares(rows[layer], installed[i], geometry, model)
            # The latest kept window that neither the installed layout (planned on the window the
            # layer was committed in) nor the new one (planned on this window) was planned on, at
            # this window's batch; a layer without one keeps only this window's gains.
            seen = next((x for x in reversed(self.kept) if x[0] != self.committed[layer]), None)
            if seen is not None:
                previous_load[i] = seen[1][layer] * (ahead / seen[2] if ahead and seen[2] else 1.0)
        # Shares solved again on the installed layouts replace the old ones where they gain on both
        # windows: the layers that do not install get them without a copy, the candidates gain on them.
        refresh = np.full(len(planned), True)
        for load in (rows[planned], previous_load):
            old = (load, installed, self.shares[planned])
            refresh &= evaluate_gains(*old, installed, resolved, model, geometry) > 0.0
        current = (rows[planned], installed, np.where(refresh[:, None], resolved, old[2]))
        gains = evaluate_gains(*current, new_layout, new_shares, model, geometry)
        # A layer's copy: its busiest receiving GPU's replicas at the lower median of the copy times per
        # replica within the steps left (one slow copy prices nothing out; the start-up one counts).
        fresh = (new_layout >= 0) & (new_layout != installed)
        copied = fresh.reshape(-1, geometry.ep_size, geometry.local_slots).sum(2)
        recent = [ms for call, ms in self.copy_samples if call >= close - (steps_left or 0.0)]
        per_replica = sorted(recent)[(len(recent) - 1) // 2] if recent else None
        # The home layout is a candidate too: retiring the replicas copies nothing, so the plan stays
        # only where it gains more than the home layout, net of a layer's copy over the steps left.
        home_shares = (home >= 0).astype(np.float64)
        home_gains = evaluate_gains(*current, home, home_shares, model, geometry)
        left, copy = steps_left or 0.0, (per_replica or 0.0) * copied.max(1, initial=0)
        to_home = (home_gains > 0.0) & (home_gains * left >= gains * left - copy)
        # Both candidates must also gain on that window, and the plan beat the home layout there.
        earlier = (previous_load, *current[1:])
        plan_gains = evaluate_gains(*earlier, new_layout, new_shares, model, geometry)
        back_gains = evaluate_gains(*earlier, home, home_shares, model, geometry)
        to_home |= (home_gains > 0.0) & (back_gains * left >= plan_gains * left - copy)
        gains, home_gains = np.minimum(gains, plan_gains), np.minimum(home_gains, back_gains)
        new_layout[to_home], new_shares[to_home] = home[to_home], home_shares[to_home]
        gains[to_home] = home_gains[to_home]
        gain = dict(zip(planned.tolist(), gains.tolist()))
        retire = [layer for layer in planned[to_home].tolist() if gain.pop(layer) * left > 0.0]
        chosen = choose_layers_to_install(gain, steps_left, dict(zip(planned.tolist(), copy)))
        if per_replica is None:
            # No copy has been timed yet: install the best layer alone to time one.
            chosen = chosen[:1]
        chosen += retire  # a retirement copies nothing: it pays as soon as it gains
        growth = self.evaluations.read_cost(
            range(layers), rows, self.layout, self.shares, self.open_call
        )
        self.kept.append((window, rows, requests))
        moved, received = [], np.zeros(geometry.ep_size)
        for i, layer in enumerate(planned.tolist()):
            if layer not in chosen:
                if not refresh[i]:
                    continue
                new_layout[i], new_shares[i] = installed[i], current[2][i]
            tables = build_route_tables(new_layout[i], new_shares[i], experts, slots).packed()
            update.tables[layer] = np.array(tables, dtype=np.int32)
            fresh = (new_layout[i] >= 0) & (new_layout[i] != self.layout[layer])
            if fresh.any():
                moved.append(layer)
                received += copied[i]
                update.copy_items[layer] = copy_items(
                    self.layout[layer], new_layout[i], geometry, cfg.ep_rank, cfg.ranks_per_node
                )
            self.layout[layer] = new_layout[i]
            self.shares[layer] = new_shares[i]
            self.committed[layer] = window
        update.moved = sorted(moved)
        self.awaiting_copy[window] = (close, growth, steps_left, int(received.max()))
        update.digest = update.fingerprint()
        return update


def serve(conn: Connection) -> None:
    """Answer every message on ``conn`` until it closes; a failure is sent back and ends it."""
    config = conn.recv()
    stats = np.frombuffer(mmap.mmap(config.stats_fd, 0), np.float32)
    decider = _Decider(config, stats.reshape(config.geometry.ep_size, -1))
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        try:
            conn.send(decider.handle(message))
        except Exception:  # the main thread raises it
            conn.send(("error", traceback.format_exc()))
            return


class DecisionProcess:
    """The decision process of one rank: a fresh interpreter without MPI or a GPU.

    It shares ``stats`` (``[ep_size, window_stats_size]`` float32, which the all-gather fills) with
    this process and runs ``decision_main.py``, which imports numpy, torch and this module but not
    ``tensorrt_llm``, and loads the torch op libraries this process loaded.
    """

    def __init__(self, config: DecisionConfig) -> None:
        size = window_stats_size(len(config.home_layouts), config.geometry.num_experts)
        nbytes = config.geometry.ep_size * size * 4
        fd = os.memfd_create("adaptive-eplb-windows")
        os.ftruncate(fd, nbytes)
        self._map = mmap.mmap(fd, nbytes)
        self.stats = torch.frombuffer(self._map, dtype=torch.float32).view(-1, size)
        parent, child = socket.socketpair()
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PMIX", "PMI_", "OMPI_"))}
        env.update(TLLM_DISABLE_MPI="1", TRT_LLM_NO_LIB_INIT="1", CUDA_VISIBLE_DEVICES="")
        libraries = sorted(torch.ops.loaded_libraries)
        argv = [sys.executable, _DECISION_MAIN, str(child.fileno()), *libraries]
        self.process = subprocess.Popen(argv, pass_fds=(child.fileno(), fd), env=env)
        child.close()
        os.close(fd)
        self._conn = Connection(parent.detach())
        self._conn.send(dataclasses.replace(config, stats_fd=fd))

    def ask(self, message: tuple) -> tuple:
        """Send ``message`` and wait for the answer; ``RuntimeError`` once the process failed or exited."""
        self._conn.send(message)
        try:
            message = self._conn.recv()
        except EOFError:
            raise RuntimeError(f"adaptive EPLB decision process exited ({self.process.poll()})")
        if message[0] == "error":
            raise RuntimeError(f"adaptive EPLB decision process failed:\n{message[1]}")
        return message

    def close(self) -> None:
        """Close the pipe and end the process, killing it if it does not exit in time."""
        self._conn.close()
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


class _WorkerThread(threading.Thread):
    """Runs blocking jobs in order; results go to ``inbox``, a failure is raised by the main thread.

    ``jobs.join()`` waits until every posted job ran.
    """

    def __init__(self) -> None:
        super().__init__(name="adaptive-eplb-worker", daemon=True)
        self.jobs: queue.Queue = queue.Queue()
        self.inbox: deque[tuple] = deque()
        self.failure: BaseException | None = None
        self.start()

    def post(self, job: Callable[[], tuple | None]) -> None:
        self.jobs.put(job)

    def run(self) -> None:
        for job in iter(self.jobs.get, None):
            try:
                result = job()
            except BaseException as exc:  # raised on the main thread
                self.failure = exc
                return
            if result is not None:
                self.inbox.append(result)
            self.jobs.task_done()


class AdaptiveRuntime:
    """Statistics windows, decisions and installs of all balanced layers of one rank.

    ``gather(send, rows, leave)`` all-gathers ``send`` (the window statistics of this rank, or a
    float64 :data:`ROW`) into ``rows``, one per EP rank, until ``leave`` is set; ``tables`` are the
    layers' ``[L, 2 * (experts + slots)]`` route tables and ``statistics`` their
    :func:`~.policy.statistics_buffers`.

    One update is in flight at a time: its route tables apply before the next window opens, and the
    worker waits for the statistics copy of that window's close before it gathers it. The pinned
    host tables, the applied windows and the copies into free slots rely on that order.

    A failure (the worker thread's, the decision process's, or ranks that decided a window
    differently) stops the runtime instead of the server: it is logged once, no further window
    opens and no further update applies, and the route tables in use stay. They stay valid, as an
    update copies experts only into slots those tables do not route to.
    """

    def __init__(
        self,
        layers: Sequence[AdaptiveMoeLayerLoadBalancer],
        transport: PeerSlotTransport,
        copy_stream: torch.cuda.Stream,
        decision: DecisionProcess,
        gather: Gather,
        device: torch.device,
        tables: torch.Tensor,
        statistics: dict[str, torch.Tensor],
    ) -> None:
        # Imported here: the decision process imports this module without the rest of the package.
        from tensorrt_llm._utils import prefer_pinned

        self.layers = layers
        self.transport = transport
        self.copy_stream = copy_stream
        self.decision = decision
        self.gather = gather
        self.leave = threading.Event()
        self.gathers_posted = self._gathers_run = 0  # all-gathers posted to / run by the worker
        self.worker = _WorkerThread()
        # Closed before the interpreter finalizes even without a shutdown: a daemon worker that
        # returns from a collective while the interpreter finalizes ends inside the C++ call, and
        # that aborts the process.
        self._closed = False
        atexit.register(self.close)
        self.call = -1
        self.window = 0
        self.open_at: int | None = 0
        self.close_at: int | None = None
        self.window_opened_call = -1
        self.window_forwards = 1
        self.applied_window = np.full(len(layers), -1.0)
        self.request_totals = np.zeros(3)  # the FIELDS remaining, requests and accepted
        self.wants_request_progress = False
        self._requests_read = 0
        self.last_applied = (-1.0, 0.0, 0.0)
        self.pending_updates: deque[tuple[int, LayoutUpdate]] = deque()
        self._apply_events: deque = deque()
        self.closed_windows = 0
        self.installs = 0
        self.installed_layers = 0
        self.failure: BaseException | None = None
        # Timing event pairs of the publishes: one comes back here once the worker read its time
        # (creating two CUDA events costs the main thread more than the rest of a publish).
        self._apply_event_pairs: deque = deque()
        self._storage = statistics["storage"]
        pin = torch.device(device).type == "cuda" and prefer_pinned()
        self._own_stats = torch.zeros(decision.stats.shape[1], pin_memory=pin)
        # The window's statistics as copied, read on the worker thread through the same views.
        self._own_raw = torch.zeros_like(self._storage, device="cpu", pin_memory=pin)
        rows, forwards = statistics["rows"], statistics["forwards"]
        rows_bytes = rows.numel() * rows.element_size()
        self._raw_rows = self._own_raw[:rows_bytes].view(rows.dtype).numpy()
        at = forwards.storage_offset() * forwards.element_size() - self._storage.storage_offset()
        self._raw_forwards = self._own_raw[at:].view(forwards.dtype).numpy()
        self._gathered = list(decision.stats.unbind(0))
        self.tables = tables
        self._host_tables = torch.empty_like(tables, device="cpu", pin_memory=pin)
        self._host_tables.copy_(tables)
        self._host_tables_np = self._host_tables.numpy()  # the host mirror, written row by row
        self._stats_copy_begin = torch.cuda.Event(enable_timing=True)
        self._stats_copy_end = torch.cuda.Event(enable_timing=True)
        self._pending_stats_copy: torch.cuda.Event | None = None
        self._copy_begin = torch.cuda.Event(enable_timing=True, blocking=True)
        self._copy_end = torch.cuda.Event(enable_timing=True, blocking=True)

    def close(self, gathers: int = 0) -> None:
        """End the worker thread after its jobs, then the decision process; once, from the balancer's
        shutdown (``gathers``: the most all-gathers any EP rank posted) or at interpreter exit."""
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        self.worker.jobs.put(None)
        self.worker.join(timeout=CLOSE_WAIT_S)
        # Zeros for the all-gathers its worker missed: a window's statistics, then an install's ROW.
        for index in range(self._gathers_run, 0 if self.worker.is_alive() else gathers):
            row = torch.zeros(len(ROW), dtype=torch.float64)
            send = row if index % 2 else torch.zeros_like(self._own_stats)
            self.gather(send, [torch.empty_like(send) for _ in self._gathered], self.leave)
        if self.worker.is_alive():
            self.leave.set()
            self.worker.join(timeout=CLOSE_WAIT_S)
        if self.worker.is_alive():
            from tensorrt_llm.logger import logger

            logger.warning("adaptive EPLB: the worker thread did not end at its close")
        self.decision.close()

    def counters(self) -> dict[str, int]:
        """The windows closed, the installs applied (updates with new route tables) and the layers
        they installed, since the runtime started."""
        return dict(
            windows=self.closed_windows,
            installs=self.installs,
            installedLayers=self.installed_layers,
        )

    def add_request_progress(self, num_requests: int, progress: GenerationProgress) -> None:
        """Add the progress of up to :data:`REQUESTS_PER_CALL` more of the ``num_requests``
        generation requests to the window's request totals: ``progress(start, stop)`` gives the
        tokens the requests ``[start, stop)`` still have to produce, how many of them are requests
        and the tokens they accept per step."""
        start = self._requests_read
        stop = start + REQUESTS_PER_CALL
        self.request_totals += progress(start, stop)
        self._requests_read = stop
        self.wants_request_progress = stop < num_requests

    def boundary(self) -> None:
        """Advance windows, decisions, copies and route tables by one call; never waits. After a
        failure it only drops the worker's messages (:attr:`failure`)."""
        if self.failure is not None:
            self.worker.inbox.clear()
            return
        try:
            self._boundary()
        except Exception as error:  # noqa: BLE001 - an optional optimization never stops serving
            self._stop(error)

    def _stop(self, error: BaseException) -> None:
        """Stop after a failure: keep the route tables in use, apply nothing further, log once."""
        from tensorrt_llm.logger import logger

        self.failure = error
        self.pending_updates.clear()
        self.worker.inbox.clear()
        self.open_at = self.close_at = None
        self._pending_stats_copy = None
        self.wants_request_progress = False
        logger.error(
            f"adaptive EPLB stopped after {error!r}; the expert layouts in use stay as they are"
        )

    def _boundary(self) -> None:
        self.call += 1
        call = self.call
        worker = self.worker
        if worker.failure is not None:
            raise RuntimeError("adaptive EPLB worker thread failed") from worker.failure
        # The messages first: a window scheduled to open at this call opens at it.
        while worker.inbox:
            message = worker.inbox.popleft()
            if message[0] == "schedule":
                _, self.window, self.open_at, apply_call, plan, self.window_forwards = message
                self._event(
                    "scheduled",
                    window=self.window,
                    apply_call=apply_call,
                    next_open=self.open_at,
                    plan=plan,
                    forwards=self.window_forwards,
                )
            elif message[0] == "copied":
                self._copies_done(*message[1:])
            else:
                update = message[1]
                self._event(
                    "decision",
                    window=update.window,
                    call=call,
                    moved=update.moved,
                    tables=update.tables,
                    copy_items=update.copy_items,
                )
                worker.post(lambda update=update: self._install(update))
                self.gathers_posted += 1
        if self.pending_updates and self.pending_updates[0][0] <= call:
            self._apply_tables(call, self.pending_updates.popleft()[1])
        if self.close_at is None and self.open_at is not None and call >= self.open_at:
            assert not self.pending_updates, "an update must apply before the next window opens"
            self._open_window()
            self.open_at = None
            self.close_at = call + self.window_forwards
            self.window_opened_call = call
        elif call == self.close_at:
            self._close_window()
        if self._pending_stats_copy is not None and not self.wants_request_progress:
            copied, self._pending_stats_copy = self._pending_stats_copy, None
            totals = self.request_totals.copy()
            window = self.window
            worker.post(lambda: self._gather_window(copied, totals, window))
            self.gathers_posted += 1

    def _event(self, event: str, **data: object) -> None:
        """Log an event of the windows, decisions and installs at debug level: its scalars and
        layer lists (tables, copies and statistics are not printed)."""
        from tensorrt_llm.logger import logger

        shown = {k: v for k, v in data.items() if not isinstance(v, (dict, np.ndarray))}
        logger.debug(f"adaptive EPLB {event}: {', '.join(f'{k}={v}' for k, v in shown.items())}")

    def _open_window(self) -> None:
        """Zero the statistics."""
        self._storage.zero_()

    def _close_window(self) -> None:
        """Copy the statistics into this rank's window statistics, timing the copies."""
        assert not self.pending_updates, "an update must apply before its next window closes"
        self.closed_windows += 1
        self._stats_copy_begin.record()
        self._own_raw.copy_(self._storage, non_blocking=True)
        self._stats_copy_end.record()
        self._pending_stats_copy = self._stats_copy_end
        self.close_at = None
        self.wants_request_progress = True
        self._requests_read = 0
        self.request_totals[:] = 0

    def _gather_window(self, copied: torch.cuda.Event, totals: np.ndarray, window: int) -> tuple:
        """Worker thread: complete this rank's window statistics, all-gather them and get the
        decision."""
        copied.synchronize()
        while self._apply_events and self._apply_events[0][3].query():
            applied, layers, begin, end = self._apply_events.popleft()
            self.last_applied = (float(applied), float(layers), begin.elapsed_time(end))
            self._apply_event_pairs.append((begin, end))
        n = len(self.layers)
        experts = self.layers[0].expert_count
        own = self._own_stats.numpy()
        own[: n * experts] = self._raw_rows
        own[n * experts : n * (experts + 1)] = self._raw_forwards
        own[n * (experts + 1) : n * (experts + 2)] = self.applied_window
        stats_copy_ms = self._stats_copy_begin.elapsed_time(self._stats_copy_end)
        own[n * (experts + 2) :] = (*totals, stats_copy_ms, *self.last_applied)
        self._gathers_run += 1
        self.gather(self._own_stats, self._gathered, self.leave)
        self._event(
            "window",
            window=window,
            open_call=self.window_opened_call,
            stats=self.decision.stats.numpy(),
        )
        return self.decision.ask(("window", window))

    def _copies_done(self, update: LayoutUpdate, rows: np.ndarray) -> None:
        """Every rank's copies of ``update`` ran: check that all ranks decided alike."""
        digests = rows[:, ROW.index("digest")]
        if (digests != update.digest).any():
            raise RuntimeError(
                f"adaptive EPLB ranks decided window {update.window} differently: layout digests "
                f"{[hex(int(d)) for d in digests]}"
            )
        apply_call = int(rows[:, ROW.index("copied_call")].max()) + 1
        assert not self.pending_updates, "one update is in flight at a time"
        self.pending_updates.append((apply_call, update))
        self._event("copied", window=update.window, apply_call=apply_call, rows=rows)
        self.worker.post(lambda: self.decision.ask(("copied", update.window, rows)))

    def _install(self, update: LayoutUpdate) -> tuple:
        """Worker thread: copy the update's experts as one batch, then all-gather the completion."""
        ms = 0.0
        if update.moved:
            experts = [
                (layer, *item) for layer in update.moved for item in update.copy_items[layer]
            ]
            self.transport.copy(experts, self.copy_stream, self._copy_begin, self._copy_end)
            self._copy_end.synchronize()
            ms = self._copy_begin.elapsed_time(self._copy_end)
        row = (self.call, len(update.moved), ms, update.digest)
        rows = torch.zeros(len(self._gathered), len(ROW), dtype=torch.float64)
        self._gathers_run += 1
        self.gather(torch.tensor(row, dtype=torch.float64), list(rows), self.leave)
        return ("copied", update, rows.numpy())

    def _apply_tables(self, call: int, update: LayoutUpdate) -> None:
        """Install the update's route tables with one timed copy of the span of layers it changes
        (the other layers in it hold the same tables already; none for an update without tables);
        later forwards route by them."""
        # The previous copy from the pinned host tables finished before the window after it closed.
        assert not self._apply_events, "the previous route tables' copy is still in flight"
        host_tables = self._host_tables_np
        for layer, tables in update.tables.items():
            host_tables[layer] = tables
            self.applied_window[layer] = update.window
        if self._apply_event_pairs:
            begin, end = self._apply_event_pairs.popleft()
        else:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        if update.tables:
            first, stop = min(update.tables), max(update.tables) + 1
            self.tables[first:stop].copy_(self._host_tables[first:stop], non_blocking=True)
            self.installs += 1
            self.installed_layers += len(update.tables)
        end.record()
        self._apply_events.append((update.window, len(update.tables), begin, end))
        self._event("applied", window=update.window, call=call, layers=sorted(update.tables))
