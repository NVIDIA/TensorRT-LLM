# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pipeline-parallel micro-batch ring and its depth (synchronous relay).

The multi-rank tests run the real ``PyExecutor._executor_loop_pp`` on ``pp_size``
threads, one executor per rank, with an MPI stand-in whose ``isend_object`` completes
only when the peer receives (rendezvous, the strict case for deadlocks) and a GPU
stand-in per rank: a worker thread that runs each launched micro-batch after the
previous stage handed it over, for a per-micro-batch cost, and hands it to the next
stage either through a bounded buffer (a buffered transport) or by waiting until the
next stage starts it (``PPCommNCCL``'s send-blocks-the-next-receive). Scheduling,
request state and the executed-batch handling are minimal stand-ins; the ring, the
schedule propagation, the follower's adoption of the first rank's chunks, the
sample-state relay, the executed-batch count broadcast and the slot bookkeeping are
the executor's own code.
"""

import contextlib
import pickle
import queue
import random
import sys
import threading
import time
import types
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import py_executor as py_executor_module
from tensorrt_llm._torch.pyexecutor.pp_ring import (
    DISAGG_RING_DEPTH,
    executed_microbatch_offset,
    last_rank_due_relay_slot,
    relay_lag,
    resolve_num_micro_batches,
)
from tensorrt_llm._torch.pyexecutor.pp_stage_agree import PPStageAgreement, probe_lag
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

pytestmark = pytest.mark.cpu_only

MIN_ASYNC = PyExecutor.MIN_ASYNC_MICRO_BATCH_NUM
TIMEOUT_S = 60.0
_POLL_S = 0.05

# --------------------------------------------------------------------------- resolution


def ring_slots(pp_size: int, ring_depth: Optional[int]) -> int:
    """Slots of a ring ``ring_depth`` deep (``None``: the default ring of ``pp_size`` slots).

    The loop runs any depth of at least ``pp_size``; a disaggregated server uses
    ``max(pp_size, DISAGG_RING_DEPTH)`` (``resolve_num_micro_batches``), the harness
    below also runs the depths in between.
    """
    depth = pp_size if ring_depth is None else int(ring_depth)
    assert depth >= pp_size
    return depth


@pytest.mark.parametrize("pp_size", [1, 2, 4, 8])
def test_aggregated_executor_keeps_the_default_ring(pp_size):
    assert resolve_num_micro_batches(pp_size, False, MIN_ASYNC) == pp_size
    assert resolve_num_micro_batches(pp_size, True, MIN_ASYNC) == max(pp_size, MIN_ASYNC)
    assert resolve_num_micro_batches(pp_size, False, MIN_ASYNC, disaggregated=False) == pp_size


@pytest.mark.parametrize("pp_size", [2, 3, 4, 8])
def test_default_offsets_equal_the_stock_expressions(pp_size):
    """Off a disaggregated server every offset is the one the loop computed before."""
    for async_relay in (False, True):
        n = resolve_num_micro_batches(pp_size, async_relay, MIN_ASYNC)
        lag = relay_lag(pp_size, n, async_relay)
        assert executed_microbatch_offset(True, lag) == -1
        assert executed_microbatch_offset(False, lag) == 1 - pp_size
    n = resolve_num_micro_batches(pp_size, False, MIN_ASYNC)
    for microbatch_id in range(n):
        assert (
            last_rank_due_relay_slot(microbatch_id, relay_lag(pp_size, n, False), n)
            == (microbatch_id + 1 - pp_size) % n
        )


def test_disaggregated_ring_is_used_by_the_synchronous_relay_only():
    assert DISAGG_RING_DEPTH == 8
    assert resolve_num_micro_batches(4, False, MIN_ASYNC, disaggregated=True) == 8
    assert relay_lag(4, 8, False) == 8
    assert executed_microbatch_offset(False, 8) == -7
    # The asynchronous relay keeps its window and its pp_size lag.
    assert resolve_num_micro_batches(4, True, MIN_ASYNC, disaggregated=True) == MIN_ASYNC
    assert relay_lag(4, MIN_ASYNC, True) == 4
    # A non-PP executor keeps its one slot.
    assert resolve_num_micro_batches(1, False, MIN_ASYNC, disaggregated=True) == 1
    # A pipeline deeper than the disaggregated depth keeps one slot per stage.
    assert resolve_num_micro_batches(16, False, MIN_ASYNC, disaggregated=True) == 16


# --------------------------------------------------------------------------- MPI stand-in


class _Handle:
    def __init__(self, consumed: threading.Event, eager: bool):
        self._consumed = consumed
        self._eager = eager

    def wait(self):
        if self._eager:
            return
        if not self._consumed.wait(TIMEOUT_S):
            raise TimeoutError("isend never matched: the PP loop deadlocked")


class _World:
    """Point-to-point object messages between rank threads (MPI stand-in).

    ``eager=False`` (the tests): an ``isend`` completes only when the peer received it,
    MPI's rendezvous protocol and the strict case for deadlocks. ``eager=True``: it
    completes at once, MPI's eager protocol for small messages (a context-only stage's
    schedule and sample state are a few hundred bytes).
    """

    def __init__(self, world_size: int, eager: bool = False):
        self.world_size = world_size
        self.eager = eager
        self._lock = threading.Lock()
        self._queues: Dict[tuple, queue.Queue] = {}

    def _queue(self, src, dst, tag) -> queue.Queue:
        with self._lock:
            return self._queues.setdefault((src, dst, int(tag)), queue.Queue())

    def isend(self, obj, src, dst, tag) -> _Handle:
        consumed = threading.Event()
        # Pickle as MPI does: ranks never share objects.
        self._queue(src, dst, tag).put((pickle.dumps(obj), consumed))
        return _Handle(consumed, self.eager)

    def recv(self, src, dst, tag):
        try:
            payload, consumed = self._queue(src, dst, tag).get(timeout=TIMEOUT_S)
        except queue.Empty:
            raise TimeoutError(f"rank {dst} never received tag {tag} from {src}") from None
        consumed.set()
        return pickle.loads(payload)


class _Dist:
    def __init__(self, world: _World, rank: int):
        pp = world.world_size
        self._world = world
        self.rank = rank
        self.pp_rank = rank
        self.pp_size = pp
        self.tp_size = 1
        self.cp_size = 1
        self.cp_config = {}
        self.is_first_pp_rank = rank == 0
        self.is_last_pp_rank = rank == pp - 1
        self.is_second_last_pp_rank = rank == pp - 2
        self.prev_pp_rank = (rank - 1) % pp
        self.next_pp_rank = (rank + 1) % pp

    def recv_object(self, src, tag):
        return self._world.recv(src, self.rank, tag)

    def isend_object(self, obj, dest, tag):
        return self._world.isend(obj, self.rank, dest, tag)


# --------------------------------------------------------------------------- GPU stand-in


class _Link:
    """Stage boundary ``s -> s + 1`` as the two transports behave on the device.

    Buffered (the staging ring): a send only queues the message for the copy stream,
    which blocks the sender's compute only when ``capacity`` copies are pending (the
    backlog bound); the copy stream delivers into ``capacity`` receiver slots and waits
    while they are all full. Blocking (``PPCommNCCL``): sends run in order on a side
    stream and each completes only when the receiver starts its receive, and the
    sender's next receive waits behind its last send (one communicator).
    """

    def __init__(self, buffered: bool, capacity: int):
        self.buffered = buffered
        self._backlog: queue.Queue = queue.Queue(maxsize=capacity if buffered else 0)
        self._slots: queue.Queue = queue.Queue(maxsize=capacity if buffered else 1)
        self.last_send_started: Optional[threading.Event] = None
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._deliver, daemon=True)
        self._thread.start()

    def send(self, microbatch: int):
        started = threading.Event()
        self.last_send_started = started
        self._backlog.put((microbatch, started), timeout=TIMEOUT_S)

    def _deliver(self):
        # Every wait polls ``_stopping`` so ``stop`` ends the thread even when it is
        # blocked on a full slot queue or an unmatched blocking send.
        while not self._stopping.is_set():
            try:
                item = self._backlog.get(timeout=_POLL_S)
            except queue.Empty:
                continue
            while not self._stopping.is_set():
                try:
                    self._slots.put(item, timeout=_POLL_S)
                    break
                except queue.Full:
                    continue
            if not self.buffered:
                while not self._stopping.is_set() and not item[1].wait(_POLL_S):
                    pass

    def recv(self) -> int:
        microbatch, started = self._slots.get(timeout=TIMEOUT_S)
        started.set()
        return microbatch

    def stop(self):
        self._stopping.set()
        self._thread.join(TIMEOUT_S)
        assert not self._thread.is_alive(), "the link's delivery thread did not stop"


class _Gpu:
    """One rank's device: runs launched micro-batches in order on a worker thread."""

    def __init__(
        self,
        stage: int,
        pp_size: int,
        inbound: Optional[_Link],
        outbound: Optional[_Link],
        cost_s,
        log: "_Log",
    ):
        self.stage = stage
        self.pp_size = pp_size
        self._inbound = inbound
        self._outbound = outbound
        self._cost_s = cost_s
        self._log = log
        self._jobs: queue.Queue = queue.Queue()
        self.error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def launch(self, microbatch: int) -> threading.Event:
        done = threading.Event()
        self._jobs.put((microbatch, done))
        return done

    def stop(self):
        self._jobs.put(None)
        self._thread.join(TIMEOUT_S)
        if self._outbound is not None:
            self._outbound.stop()

    def _run(self):
        try:
            while True:
                job = self._jobs.get()
                if job is None:
                    return
                microbatch, done = job
                if self._inbound is not None:
                    outbound = self._outbound
                    if (
                        outbound is not None
                        and not outbound.buffered
                        and (outbound.last_send_started is not None)
                    ):
                        # One communicator: this receive is ordered behind the last send.
                        if not outbound.last_send_started.wait(TIMEOUT_S):
                            raise TimeoutError("blocking send never matched")
                    received = self._inbound.recv()
                    assert received == microbatch, (
                        f"stage {self.stage} received micro-batch {received}, ran {microbatch}"
                    )
                self._log.busy(self.stage, +1)
                time.sleep(self._cost_s(microbatch))
                self._log.busy(self.stage, -1)
                if self._outbound is not None:
                    self._outbound.send(microbatch)
                done.set()
        except BaseException as exc:  # noqa: BLE001 - surfaced by the test
            self.error = exc


class _Log:
    def __init__(self, pp_size: int):
        self._lock = threading.Lock()
        self.busy_now = 0
        self.max_busy = 0
        self.launched: Dict[int, List[int]] = {r: [] for r in range(pp_size)}
        self.handled: Dict[int, List[int]] = {r: [] for r in range(pp_size)}
        self.relayed: Dict[int, List[int]] = {r: [] for r in range(pp_size)}
        self.chunks: Dict[int, List[tuple]] = {r: [] for r in range(pp_size)}
        self.max_lead = 0

    def busy(self, stage, delta):
        with self._lock:
            self.busy_now += delta
            self.max_busy = max(self.max_busy, self.busy_now)


class _Noop:
    """A cheap do-nothing stand-in (``MagicMock`` records every call, which on the
    loop's hot path is slow enough to dominate the timing)."""

    def __init__(self, **attributes):
        self.__dict__.update(attributes)

    def __getattr__(self, name):
        return _noop

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __bool__(self):
        return False


def _noop(*args, **kwargs):
    return _NOOP


_NOOP = _Noop()

# --------------------------------------------------------------------------- requests


class _PyResult:
    def __init__(self):
        self._diff = []

    def get_diff(self):
        return list(self._diff)

    def reset_diff(self):
        self._diff = []

    def apply_diff(self, diff):
        self._diff.extend(diff)


class _Request:
    """The request fields the PP loop and the follower adoption read."""

    def __init__(self, request_id: int, prompt_len: int):
        self.request_id = request_id
        self.py_request_id = request_id
        self.prompt_len = prompt_len
        self.context_current_position = 0
        self.context_chunk_size = 0
        self.py_batch_idx = None
        self.py_result = _PyResult()
        self.is_attention_dp_dummy = False
        self.context_done = False

    @property
    def is_last_context_chunk(self) -> bool:
        return self.context_current_position + self.context_chunk_size >= self.prompt_len

    @property
    def is_first_context_chunk(self) -> bool:
        return self.context_current_position == 0

    @property
    def is_context_init_state(self) -> bool:
        return not self.context_done


class _SampleEvent:
    def __init__(self, done: threading.Event):
        self._done = done

    def synchronize(self):
        if not self._done.wait(TIMEOUT_S):
            raise TimeoutError("the micro-batch never finished on the device")


# --------------------------------------------------------------------------- executor


def _fcfs_chunks(active_requests, token_budget, inflight):
    """FCFS context chunks into one token budget; sizes the chosen requests."""
    budget = token_budget
    chosen = []
    for req in active_requests:
        if budget == 0:
            break
        if req.context_done or req.request_id in inflight:
            continue
        req.context_chunk_size = min(budget, req.prompt_len - req.context_current_position)
        budget -= req.context_chunk_size
        chosen.append(req)
    return chosen


class _Executor(PyExecutor):
    """``PyExecutor`` whose model, scheduler and response handling are stand-ins."""

    def _schedule(self):
        """FCFS context chunks into one token budget (the first rank only)."""
        batch = ScheduledRequests()
        chosen = _fcfs_chunks(self.active_requests, self._token_budget, self._inflight)
        batch.reset_context_requests(chosen)
        return batch, [], len(chosen)

    def _fetch_and_activate_new_requests(self):
        return []

    def _handle_control_request(self):
        pass

    def _init_iter_stats_if_sampled(self, num_new_requests):
        return None

    def _pad_attention_dp_dummy_request(self):
        pass

    def _pp_retry_until_can_schedule(self, scheduled_batch):
        pass

    def _prepare_disagg_gen_transmission_complete(self, scheduled_batch):
        pass

    def _handle_dynamic_draft_len(self, scheduled_batch):
        pass

    def _revert_gen_alloc(self, scheduled_batch):
        pass

    def _maybe_record_hang_diagnostic_phase(self, *args, **kwargs):
        pass

    def _take_pending_new_active_requests(self, iter_stats):
        return None

    @contextlib.contextmanager
    def _step_scope(self, scheduled_batch):
        yield

    @contextlib.contextmanager
    def _profiler(self):
        yield lambda: None

    def _add_inflight_ids(self, scheduled_requests):
        added = [r.request_id for r in scheduled_requests.context_requests_last_chunk]
        self._inflight.update(added)
        scheduled_requests.added_inflight_req_ids = added

    def _remove_inflight_ids(self, scheduled_requests):
        self._inflight.difference_update(scheduled_requests.added_inflight_req_ids)
        scheduled_requests.added_inflight_req_ids = []

    def _update_request_states(self, scheduled_requests):
        for req in scheduled_requests.context_requests:
            self._log.chunks[self.dist.rank].append(
                (req.request_id, req.context_current_position, req.context_chunk_size)
            )
            req.context_current_position += req.context_chunk_size
            if req.context_current_position >= req.prompt_len:
                req.context_done = True

    def _launch(self, scheduled_batch) -> threading.Event:
        microbatch = self._launched
        self._launched += 1
        self._log.launched[self.dist.rank].append(
            sorted(r.request_id for r in scheduled_batch.context_requests)
        )
        if self.dist.rank == 0:
            with self._log._lock:
                lead = self._launched - len(self._log.handled[0])
                self._log.max_lead = max(self._log.max_lead, lead)
        return self._gpu.launch(microbatch)

    def _sample_state(self, scheduled_batch, done):
        return types.SimpleNamespace(
            requests=scheduled_batch.context_requests_last_chunk
            + scheduled_batch.generation_requests,
            sampler_event=_SampleEvent(done),
            host=("tokens", self._launched - 1),
        )

    def _forward_step_inter_pp(self, scheduled_batch, gpu_forward_start=None, gpu_forward_end=None):
        done = self._launch(scheduled_batch)
        self._update_request_states(scheduled_batch)
        return self._sample_state(scheduled_batch, done)

    def _forward_step(self, scheduled_batch, *args, **kwargs):
        self._pending_done = self._launch(scheduled_batch)
        return {"logits": None}

    def _sample_async(self, scheduled_batch, batch_outputs):
        return self._sample_state(scheduled_batch, self._pending_done)

    def _handle_guided_decoder_errors(self, scheduled_batch, failed):
        pass

    def _ring_broadcast_sample_state(self, executed_batch):
        if executed_batch is not None:
            self._log.relayed[self.dist.rank].append(executed_batch.sample_state.host[1])
        super()._ring_broadcast_sample_state(executed_batch)

    def _handle_executed_batch(self, executed_batch):
        if executed_batch is None:
            return
        self._log.handled[self.dist.rank].append(executed_batch.sample_state.host[1])
        scheduled = executed_batch.scheduled_requests
        self._remove_inflight_ids(scheduled)
        for req in scheduled.context_requests_last_chunk:
            self.active_requests.remove(req)


def _make_executor(world, rank, num_micro_batches, prompts, token_budget, gpu, log, adopt=True):
    """One PP rank. ``adopt``: the leader-batch adoption, with a follower pass
    that schedules nothing, so only adoption sizes the follower's chunks.
    Otherwise there is no adoption and the follower's own pass sizes them, as
    the first rank's pass does (identical pools)."""
    exe = object.__new__(_Executor)
    exe.dist = _Dist(world, rank)
    exe.device_id = 0
    exe._log = log
    exe._gpu = gpu
    exe._launched = 0
    exe._inflight = set()
    exe._token_budget = token_budget
    exe.inflight_req_ids = _NOOP
    exe.active_requests = [_Request(i, p) for i, p in enumerate(prompts)]
    exe.waiting_queue = []
    exe.is_shutdown = True
    exe.kv_connector_manager = None
    exe.hang_detector = _NOOP
    exe.enable_iter_perf_stats = False
    exe.enable_attention_dp = False
    exe.iter_counter = 0
    exe._mm_encoder_item_scheduling_enabled = False
    exe._is_kv_manager_v2 = False
    exe._pp_adopt_leader_chunks = adopt
    exe._pp_rebalance_drain_iters = None
    exe._pending_recompute_pause_ids = set()
    exe._profile_enabled = False
    exe._disagg_coordinator = _Noop(admit=lambda fitting: (fitting, False))

    def _local_pass(active, inflight):
        chosen = [] if adopt else _fcfs_chunks(active, token_budget, exe._inflight)
        return types.SimpleNamespace(
            context_requests=chosen, generation_requests=[], fitting_disagg_gen_init_requests=[]
        )

    def _adopt(req, size):
        assert adopt, "adoption ran without the leader-batch adoption"
        req.context_chunk_size = size
        return True

    exe.scheduler = _Noop(schedule_request=_local_pass, adopt_context_chunk=_adopt)
    exe.kv_cache_manager = object()
    exe.resource_manager = _NOOP
    exe.kv_cache_transceiver = None
    exe.guided_decoder = None
    exe.disable_overlap_scheduler = True
    exe.pp_multi_stream_sample = False
    exe.perf_manager = _NOOP
    exe.pp_async_broadcast_sample_state = False
    exe.num_micro_batches = num_micro_batches
    exe.micro_batches = [None] * num_micro_batches
    exe.send_handles = [None] * num_micro_batches
    exe.send_schedule_handles = [None] * num_micro_batches
    exe.send_expected_batch_num_handles = [None] * num_micro_batches
    exe.unhandled_batch_counter = 0
    exe.previous_batch = None
    exe.executed_batch_response_queue = queue.Queue()
    return exe


def _run_pipeline(
    pp_size,
    ring_depth,
    prompts,
    token_budget,
    buffered,
    capacity,
    cost_s,
    monkeypatch,
    eager_mpi=False,
    adopt=True,
):
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(py_executor_module, "CUASSERT", lambda result: result)
    monkeypatch.setattr(py_executor_module, "cudart", MagicMock())
    num_micro_batches = ring_slots(pp_size, ring_depth)
    world = _World(pp_size, eager_mpi)
    log = _Log(pp_size)
    links = [_Link(buffered, capacity) for _ in range(pp_size - 1)]
    gpus = [
        _Gpu(
            s,
            pp_size,
            links[s - 1] if s > 0 else None,
            links[s] if s < pp_size - 1 else None,
            cost_s,
            log,
        )
        for s in range(pp_size)
    ]
    executors = [
        _make_executor(world, r, num_micro_batches, prompts, token_budget, gpus[r], log, adopt)
        for r in range(pp_size)
    ]
    errors: List[BaseException] = []

    def _rank(exe):
        try:
            exe._executor_loop_pp()
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=_rank, args=(exe,), daemon=True) for exe in executors]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(TIMEOUT_S * 2)
    wall = time.perf_counter() - start
    for gpu in gpus:
        gpu.stop()
    assert not any(t.is_alive() for t in threads), "the PP loop deadlocked"
    assert not errors, errors
    assert not [g.error for g in gpus if g.error is not None]
    return executors, log, wall, num_micro_batches


def _expected_chunks(prompts, token_budget):
    """FCFS chunk sequence: what every stage must run, micro-batch by micro-batch."""
    remaining = list(prompts)
    position = [0] * len(prompts)
    batches = []
    while any(remaining):
        budget = token_budget
        batch = []
        for i, left in enumerate(remaining):
            if budget == 0:
                break
            if left == 0:
                continue
            size = min(budget, left)
            batch.append((i, position[i], size))
            position[i] += size
            remaining[i] -= size
            budget -= size
        batches.append(batch)
    return batches


@pytest.mark.parametrize("ring_depth", [None, 4, 5, 8])
@pytest.mark.parametrize("buffered", [False, True])
@pytest.mark.parametrize("eager_mpi", [False, True])
@pytest.mark.parametrize("adopt", [True, False], ids=["adopt", "own_pass"])
def test_real_loop_runs_every_chunk_on_every_stage_in_order(
    ring_depth, buffered, eager_mpi, adopt, monkeypatch
):
    pp_size = 4
    prompts = [3 * 64 + 10, 64, 5 * 64, 2 * 64 + 1, 7 * 64]
    token_budget = 64
    rng = random.Random(ring_depth or 0)
    costs = {}

    def cost(microbatch):
        if microbatch not in costs:
            costs[microbatch] = 0.002 * rng.uniform(0.5, 1.5)
        return costs[microbatch]

    executors, log, _, n = _run_pipeline(
        pp_size,
        ring_depth,
        prompts,
        token_budget,
        buffered,
        2,
        cost,
        monkeypatch,
        eager_mpi=eager_mpi,
        adopt=adopt,
    )
    expected = _expected_chunks(prompts, token_budget)
    for rank in range(pp_size):
        # Every stage ran exactly the first rank's chunks, micro-batch by micro-batch
        # (sized by the follower adoption, or by the follower's own pass without
        # it), and each request's chunks in order.
        ran = iter(log.chunks[rank])
        assert [sorted(next(ran) for _ in batch) for batch in expected] == [
            sorted(batch) for batch in expected
        ], rank
        assert next(ran, None) is None
        assert log.launched[rank] == [sorted(c[0] for c in batch) for batch in expected]
        # Each micro-batch's sample state is relayed and handled once, in launch order.
        assert log.handled[rank] == list(range(len(expected))), rank
        assert sorted(log.relayed[rank]) == log.relayed[rank] == list(range(len(expected)))
        assert executors[rank].active_requests == []
        assert executors[rank].unhandled_batch_counter == 0
        assert all(mb is None for mb in executors[rank].micro_batches)
    # The first rank runs up to the ring depth ahead of what it has handled.
    assert n == (ring_depth or pp_size)
    assert log.max_lead == n
    # All four stages computed at the same time at least once.
    assert log.max_busy == pp_size


def test_non_final_chunks_of_different_requests_overlap_across_stages(monkeypatch):
    """With a small token budget the stages hold chunks of different requests at once."""
    pp_size = 4
    prompts = [96, 96, 96, 96]
    executors, log, _, _ = _run_pipeline(
        pp_size, 8, prompts, 48, True, 2, lambda mb: 0.003, monkeypatch
    )
    expected = _expected_chunks(prompts, 48)
    # FCFS gives each micro-batch one 48-token chunk; requests follow one another in
    # chunk order, and the pipeline still holds four micro-batches at once.
    assert [len(batch) for batch in expected] == [1] * 8
    assert log.max_busy == pp_size


# --------------------------------------------------------------------------- further checks


@pytest.mark.parametrize(
    "pp_size,ring_depth", [(2, None), (2, 3), (2, 5), (3, None), (3, 4), (3, 7)]
)
@pytest.mark.parametrize("buffered", [False, True])
@pytest.mark.parametrize("eager_mpi", [False, True])
@pytest.mark.parametrize("adopt", [True, False], ids=["adopt", "own_pass"])
def test_real_loop_other_pipeline_depths(
    pp_size, ring_depth, buffered, eager_mpi, adopt, monkeypatch
):
    """pp 2 and 3 (the second-last rank is the first rank at pp 2) with deeper rings."""
    prompts = [3 * 64 + 10, 64, 5 * 64, 2 * 64 + 1]
    token_budget = 64
    rng = random.Random(pp_size * 10 + (ring_depth or 0))
    costs = {}

    def cost(microbatch):
        if microbatch not in costs:
            costs[microbatch] = 0.002 * rng.uniform(0.5, 1.5)
        return costs[microbatch]

    executors, log, _, n = _run_pipeline(
        pp_size,
        ring_depth,
        prompts,
        token_budget,
        buffered,
        2,
        cost,
        monkeypatch,
        eager_mpi=eager_mpi,
        adopt=adopt,
    )
    expected = _expected_chunks(prompts, token_budget)
    for rank in range(pp_size):
        assert log.launched[rank] == [sorted(c[0] for c in batch) for batch in expected], rank
        assert log.handled[rank] == list(range(len(expected))), rank
        assert executors[rank].active_requests == []
        assert executors[rank].unhandled_batch_counter == 0
        assert all(mb is None for mb in executors[rank].micro_batches)
    assert n == (ring_depth or pp_size)
    assert log.max_lead == n


class _GatedExecutor(_Executor):
    """First-rank FCFS scheduling that admits a request only from a given iteration on,
    so the ring runs partially empty (idle iterations) between bursts."""

    def _schedule(self):
        batch = ScheduledRequests()
        budget = self._token_budget
        chosen = []
        for req in self.active_requests:
            if budget == 0:
                break
            if self.iter_counter < self._release.get(req.request_id, 0):
                continue
            if req.context_done or req.request_id in self._inflight:
                continue
            req.context_chunk_size = min(budget, req.prompt_len - req.context_current_position)
            budget -= req.context_chunk_size
            chosen.append(req)
        batch.reset_context_requests(chosen)
        return batch, [], len(chosen)


@pytest.mark.parametrize("ring_depth", [None, 8])
@pytest.mark.parametrize("eager_mpi", [False, True])
@pytest.mark.parametrize("adopt", [True, False], ids=["adopt", "own_pass"])
def test_real_loop_with_idle_gaps_between_bursts(ring_depth, eager_mpi, adopt, monkeypatch):
    """Requests released at later iterations: idle slots interleave with queued ones."""
    pp_size = 4
    prompts = [128, 64, 192, 64, 128, 64]
    release = {0: 0, 1: 0, 2: 12, 3: 13, 4: 30, 5: 31}
    original = _make_executor

    def make_gated(*args, **kwargs):
        exe = original(*args, **kwargs)
        exe.__class__ = _GatedExecutor
        exe._release = release
        return exe

    monkeypatch.setattr(sys.modules[__name__], "_make_executor", make_gated)
    executors, log, _, _ = _run_pipeline(
        pp_size,
        ring_depth,
        prompts,
        64,
        True,
        2,
        lambda mb: 0.002,
        monkeypatch,
        eager_mpi=eager_mpi,
        adopt=adopt,
    )
    first = log.chunks[0]
    for rank in range(pp_size):
        # Every stage ran the first rank's chunk sequence, micro-batch by micro-batch.
        assert log.chunks[rank] == first, rank
        assert log.handled[rank] == list(range(len(log.launched[0]))), rank
        assert executors[rank].active_requests == []
        assert executors[rank].unhandled_batch_counter == 0
    # Each request's chunks tile its prompt in order.
    for request_id, prompt_len in enumerate(prompts):
        chunks = [(pos, size) for rid, pos, size in first if rid == request_id]
        assert chunks[0][0] == 0
        assert all(a[0] + a[1] == b[0] for a, b in zip(chunks, chunks[1:]))
        assert chunks[-1][0] + chunks[-1][1] == prompt_len


# --------------------------------------------------------------------------- stage agreement


class _AgreeingExecutor(_Executor):
    """First-rank FCFS scheduling that skips the first chunks the PP stage
    agreement holds until every stage reported them."""

    def _schedule(self):
        batch = ScheduledRequests()
        budget = self._token_budget
        chosen = []
        for req in self._schedulable_active_requests():
            if budget == 0:
                break
            if req.context_done or req.request_id in self._inflight:
                continue
            self._first_scheduled.setdefault(req.request_id, self.iter_counter)
            req.context_chunk_size = min(budget, req.prompt_len - req.context_current_position)
            budget -= req.context_chunk_size
            chosen.append(req)
        batch.reset_context_requests(chosen)
        return batch, [], len(chosen)


class _AgreeingPool:
    """Every stage reserves the same reusable prefix (0): the loop-level test is
    about the proposal and report exchange, not the depths
    (``test_pp_stage_agreement.py``)."""

    def __init__(self):
        self.kv_cache_map = {}
        self.probes = 0
        self.reservations = 0

    def probe_context_reuse(self, req, include_claimed=False):
        assert include_claimed
        self.probes += 1
        return 0

    def reserve(self, req, cap):
        self.reservations += 1
        self.kv_cache_map[req.py_request_id] = 0
        return 0


def _adopt_first(req, position, size):
    assert req.context_current_position == position
    req.context_chunk_size = size
    return True


@pytest.mark.parametrize("pp_size,ring_depth", [(2, None), (3, 5), (4, None), (4, 8)])
@pytest.mark.parametrize("eager_mpi", [False, True])
def test_real_loop_with_the_stage_agreement_exchange(pp_size, ring_depth, eager_mpi, monkeypatch):
    """The reuse reports ride beside the loop without a deadlock, every report is
    matched, and the first rank admits each first chunk only after the reports."""
    prompts = [3 * 64 + 10, 64, 5 * 64, 2 * 64 + 1]
    token_budget = 64
    num_micro_batches = ring_slots(pp_size, ring_depth)
    lag = probe_lag(relay_lag(pp_size, num_micro_batches, False))
    original = _make_executor

    def make_agreeing(world, rank, *args, **kwargs):
        exe = original(world, rank, *args, **kwargs)
        exe.__class__ = _AgreeingExecutor
        exe._first_scheduled = {}
        exe._pp_stage_agree = PPStageAgreement(rank, pp_size, lag, reuse_enabled=True)
        exe.kv_cache_manager = _AgreeingPool()
        exe.enable_joint_kv_cache_reuse = False
        exe.scheduler.adopt_first_context_chunk = _adopt_first
        exe.scheduler.release_context_admission = lambda req: None
        exe.scheduler.reserve_first_context_chunk = exe.kv_cache_manager.reserve
        # Followers never decline a reservation for lack of room here.
        exe.scheduler.reservation_leaves_chunk_room = lambda req: True
        return exe

    monkeypatch.setattr(sys.modules[__name__], "_make_executor", make_agreeing)
    executors, log, _, n = _run_pipeline(
        pp_size,
        ring_depth,
        prompts,
        token_budget,
        True,
        2,
        lambda mb: 0.002,
        monkeypatch,
        eager_mpi=eager_mpi,
    )
    expected = _expected_chunks(prompts, token_budget)
    leader = executors[0]
    for rank in range(pp_size):
        assert log.launched[rank] == [sorted(c[0] for c in batch) for batch in expected], rank
        assert log.handled[rank] == list(range(len(expected))), rank
        assert executors[rank].active_requests == []
        agreement = executors[rank]._pp_stage_agree
        assert agreement.iteration == leader._pp_stage_agree.iteration
        assert not agreement.pending_sends
        # Every stage reserved every prompt once; the followers also probed it.
        assert executors[rank].kv_cache_manager.reservations == len(prompts)
        if rank:
            assert executors[rank].kv_cache_manager.probes == len(prompts)
    # No first chunk before every stage's answer to its proposal reached the
    # first rank; every prompt was admitted at the agreed depth.
    assert min(leader._first_scheduled.values()) == lag
    assert leader._pp_stage_agree.num_admitted == len(prompts)
    assert leader._pp_stage_agree.num_undone == 0
    assert n == (ring_depth or pp_size)


class _PressurePool:
    """A stage's KV pool in units: a reservation pins one (its held prefix); a
    running request (first chunk admitted, prompt not finished here) two."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.kv_cache_map = {}
        self.reservations = 0
        self.released = 0
        self.probes = 0

    def probe_context_reuse(self, req, include_claimed=False):
        self.probes += 1
        return 0

    def used(self) -> int:
        return sum(1 if state == "reserved" else 2 for state in self.kv_cache_map.values())

    def reserve(self, req, cap):
        self.release(req)
        self.reservations += 1
        self.kv_cache_map[req.py_request_id] = "reserved"
        return 0

    def release(self, req):
        if self.kv_cache_map.get(req.py_request_id) == "reserved":
            del self.kv_cache_map[req.py_request_id]
            self.released += 1


class _PressureExecutor(_AgreeingExecutor):
    """First-rank FCFS pass over a ``_PressurePool``: a confirmed first chunk is
    admitted only if converting its reservation into a running request fits;
    otherwise the pool refuses it pages, as the V2 scheduler would."""

    def _schedule(self):
        pool = self.kv_cache_manager
        self._refused = False
        batch = ScheduledRequests()
        budget = self._token_budget
        chosen = []
        for req in self._schedulable_active_requests():
            if budget == 0:
                break
            if req.context_done or req.request_id in self._inflight:
                continue
            if req.context_current_position == 0:
                if pool.used() + 1 > pool.capacity:
                    req.py_pp_page_refused = True
                    self._refused = True
                    continue
                pool.kv_cache_map[req.request_id] = "running"
            self._first_scheduled.setdefault(req.request_id, self.iter_counter)
            req.context_chunk_size = min(budget, req.prompt_len - req.context_current_position)
            budget -= req.context_chunk_size
            chosen.append(req)
        batch.reset_context_requests(chosen)
        return batch, [], len(chosen)

    def _pp_stage_agree_exchange(self):
        super()._pp_stage_agree_exchange()
        agreement = self._pp_stage_agree
        if self.dist.rank == 0:
            # What the first rank holds during this iteration's pass.
            self._pressure_log[agreement.iteration - 1] = set(agreement._proposed) | set(
                agreement._confirmed
            )

    def _handle_executed_batch(self, executed_batch):
        super()._handle_executed_batch(executed_batch)
        if executed_batch is not None:
            for req in executed_batch.scheduled_requests.context_requests_last_chunk:
                self.kv_cache_manager.kv_cache_map.pop(req.request_id, None)


@pytest.mark.parametrize("pp_size,ring_depth", [(2, None), (4, None), (4, 8)])
@pytest.mark.parametrize("eager_mpi", [False, True])
def test_real_loop_gives_reservations_back_under_page_pressure(
    pp_size, ring_depth, eager_mpi, monkeypatch
):
    """Page pressure in the real PP loop: the first rank's pool fits one
    running prompt plus one held prefix. Every waiting prompt is reserved at
    once; when the running one finishes, the held prefixes leave no room for
    any first chunk and the pass stalls. The first rank gives reservations back
    and every follower drops them with the next schedule: no deadlock, every
    prompt runs, and every follower adopts each first chunk from a reservation
    it still holds while holding nothing the first rank did not hold in that
    iteration's pass."""
    prompts = [3 * 64 + 10, 64, 5 * 64, 2 * 64 + 1, 64 + 7]
    token_budget = 64
    num_micro_batches = ring_slots(pp_size, ring_depth)
    lag = probe_lag(relay_lag(pp_size, num_micro_batches, False))
    leader_log = {}
    adoptions = []
    original = _make_executor

    def make_pressured(world, rank, *args, **kwargs):
        exe = original(world, rank, *args, **kwargs)
        exe.__class__ = _PressureExecutor
        exe._first_scheduled = {}
        exe._pressure_log = leader_log
        exe._refused = False
        exe._pp_stage_agree = PPStageAgreement(rank, pp_size, lag, reuse_enabled=True)
        pool = _PressurePool(3)
        exe.kv_cache_manager = pool
        exe.enable_joint_kv_cache_reuse = False
        exe.scheduler.context_pages_refused_last_pass = lambda: exe._refused
        exe.scheduler.release_context_admission = pool.release
        exe.scheduler.reserve_first_context_chunk = pool.reserve
        # Followers never decline a reservation for lack of room here.
        exe.scheduler.reservation_leaves_chunk_room = lambda req: True

        def adopt_first(req, position, size):
            iteration = exe._pp_stage_agree.iteration
            held = {rid for rid, state in pool.kv_cache_map.items() if state == "reserved"}
            assert pool.kv_cache_map.get(req.request_id) == "reserved", "no reservation"
            assert held <= leader_log[iteration], (held, leader_log[iteration])
            adoptions.append((rank, req.request_id))
            pool.kv_cache_map[req.request_id] = "running"
            return _adopt_first(req, position, size)

        exe.scheduler.adopt_first_context_chunk = adopt_first
        return exe

    monkeypatch.setattr(sys.modules[__name__], "_make_executor", make_pressured)
    executors, log, _, n = _run_pipeline(
        pp_size,
        ring_depth,
        prompts,
        token_budget,
        True,
        2,
        lambda mb: 0.002,
        monkeypatch,
        eager_mpi=eager_mpi,
    )
    leader = executors[0]
    agreement = leader._pp_stage_agree
    first = log.chunks[0]
    for rank in range(pp_size):
        assert log.chunks[rank] == first, rank
        assert executors[rank].active_requests == []
        assert not executors[rank].kv_cache_manager.kv_cache_map
        assert executors[rank]._pp_stage_agree.iteration == agreement.iteration
        assert not executors[rank]._pp_stage_agree.pending_sends
    for request_id, prompt_len in enumerate(prompts):
        chunks = [(pos, size) for rid, pos, size in first if rid == request_id]
        assert chunks[0][0] == 0 and sum(size for _, size in chunks) == prompt_len
    assert sorted(adoptions) == sorted(
        (rank, rid) for rank in range(1, pp_size) for rid in range(len(prompts))
    )
    assert agreement.num_admitted == len(prompts) and agreement.num_undone == 0
    assert agreement.num_given_back >= 1
    # A follower dropped every reservation the first rank gave back.
    for rank in range(1, pp_size):
        pool = executors[rank].kv_cache_manager
        assert pool.reservations - pool.released == len(prompts)
    assert n == (ring_depth or pp_size)


@pytest.mark.parametrize("ring_depth", [None, 8])
def test_real_loop_takes_its_offsets_from_the_ring_helpers(ring_depth, monkeypatch):
    """The PP loop places executed micro-batches and the relay through the
    ``pp_ring`` helpers: every non-last stage's executed slot is the one
    ``executed_microbatch_offset`` gives for ``relay_lag`` of its ring, and the
    last stage's due relay slot is ``last_rank_due_relay_slot``. On the default
    ring these are the stock ``1 - pp_size`` offset and slot."""
    pp_size = 4
    lags, offsets, due = [], [], []

    def spy_lag(*args):
        result = relay_lag(*args)
        lags.append((args, result))
        return result

    def spy_offset(*args):
        result = executed_microbatch_offset(*args)
        offsets.append((args, result))
        return result

    def spy_due(*args):
        result = last_rank_due_relay_slot(*args)
        due.append((args, result))
        return result

    monkeypatch.setattr(py_executor_module, "relay_lag", spy_lag)
    monkeypatch.setattr(py_executor_module, "executed_microbatch_offset", spy_offset)
    monkeypatch.setattr(py_executor_module, "last_rank_due_relay_slot", spy_due)
    _, _, _, n = _run_pipeline(
        pp_size, ring_depth, [3 * 64, 2 * 64, 64], 64, False, 2, lambda mb: 0.001, monkeypatch
    )
    assert n == (ring_depth or pp_size)
    assert lags and offsets
    assert {args for args, _ in lags} == {(pp_size, n, False)}
    assert {result for _, result in lags} == {n}
    # Non-last stages relay the micro-batch launched n - 1 iterations earlier,
    # the last stage the one it ran in the previous iteration.
    assert {entry for entry in offsets} == {((False, n), 1 - n), ((True, n), -1)}
    for (microbatch_id, lag, slots), slot in due:
        assert (lag, slots) == (n, n)
        assert slot == (microbatch_id + 1 - n) % n
