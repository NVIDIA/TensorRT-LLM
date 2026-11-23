# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A PP context worker that sends KV, with and without the lagged send agreement.

The real ``PyExecutor._executor_loop_pp`` and ``_handle_executed_batch`` run on
``pp_size`` threads, with the real disagg coordinator, the real
``AsyncTransferManager``, the real PP termination handlers, and the real
``KvCacheTransceiverV2`` status, consensus and lagged-mode code. Stand-ins:
the transfer itself (a session that completes, fails, is cancelled by the peer
or hangs after a delay), the device (one worker thread per stage, buffered
stage-to-stage hand-over, a fixed cost per micro-batch), MPI (point-to-point
objects with rendezvous isends, and world collectives matched by call order),
and scheduling (FCFS chunks) and response handling.

Stock, every PP rank enters PP-group collectives several times per iteration,
which caps the pipeline at about two micro-batches in flight. With the lagged
agreement (``pp_lagged_consensus.py``, on wherever it applies) no rank enters
any collective, the first rank runs the ring depth ahead, and a failing,
cancelled or timed-out send is settled the same way on every stage.
"""

import contextlib
import pickle
import queue
import threading
import time
import types
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.disaggregation.base.transfer import (
    SessionStatus,
    WaitResult,
    get_unique_rid,
)
from tensorrt_llm._torch.disaggregation.orchestration.pp_lagged_consensus import (
    PP_DISAGG_DECISION_TAG,
    PP_DISAGG_REPORT_TAG,
    PPDisaggLaggedConsensus,
)
from tensorrt_llm._torch.disaggregation.orchestration.pp_termination import (
    DisaggPPTerminationHandler,
    LaggedDisaggPPTerminationHandler,
)
from tensorrt_llm._torch.disaggregation.orchestration.transfer_manager import AsyncTransferManager
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor import py_executor as py_executor_module
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle

pytestmark = pytest.mark.cpu_only

MIN_ASYNC = PyExecutor.MIN_ASYNC_MICRO_BATCH_NUM


def ring_slots(pp_size: int, ring_depth: Optional[int]) -> int:
    """Slots of a ring ``ring_depth`` deep (``None``: the default ring of ``pp_size`` slots)."""
    depth = pp_size if ring_depth is None else int(ring_depth)
    assert depth >= pp_size
    return depth


TIMEOUT_S = 20.0
STOP_CHECK_EVERY = 16

# --------------------------------------------------------------------------- MPI stand-in


class _Handle:
    def __init__(self, consumed: threading.Event):
        self._consumed = consumed

    def wait(self):
        if not self._consumed.wait(TIMEOUT_S):
            raise TimeoutError("isend never matched: the PP loop deadlocked")


class _World:
    """Point-to-point messages (an isend completes when the peer receives it) and
    world collectives, matched by each rank's call order and checked for kind."""

    def __init__(self, size: int):
        self.size = size
        self._lock = threading.Lock()
        self._queues: Dict[tuple, queue.Queue] = {}
        self._cv = threading.Condition()
        self._slots: Dict[int, Dict[int, tuple]] = {}
        self._next = [0] * size
        self.collectives: Dict[int, List[str]] = {r: [] for r in range(size)}
        # Test-only rendezvous used by the stop check; not counted.
        self._stop_cv = threading.Condition()
        self._stop_slots: Dict[int, Dict[int, bool]] = {}

    def _queue(self, src, dst, tag) -> queue.Queue:
        with self._lock:
            return self._queues.setdefault((src, dst, int(tag)), queue.Queue())

    def isend(self, obj, src, dst, tag) -> _Handle:
        consumed = threading.Event()
        self._queue(src, dst, tag).put((pickle.dumps(obj), consumed))
        return _Handle(consumed)

    def recv(self, src, dst, tag):
        try:
            payload, consumed = self._queue(src, dst, tag).get(timeout=TIMEOUT_S)
        except queue.Empty:
            raise TimeoutError(f"rank {dst} never received tag {tag} from {src}") from None
        consumed.set()
        return pickle.loads(payload)

    def gather(self, rank: int, kind: str, obj):
        with self._cv:
            seq = self._next[rank]
            self._next[rank] += 1
            slot = self._slots.setdefault(seq, {})
            slot[rank] = (kind, pickle.dumps(obj))
            self.collectives[rank].append(kind)
            self._cv.notify_all()
            if not self._cv.wait_for(lambda: len(slot) == self.size, timeout=TIMEOUT_S):
                raise TimeoutError(f"rank {rank} waited in collective {seq} ({kind}) forever")
        kinds = {k for k, _ in slot.values()}
        if len(kinds) != 1:
            raise AssertionError(f"ranks entered different collectives at step {seq}: {slot}")
        return [pickle.loads(slot[r][1]) for r in range(self.size)]

    def all_done(self, rank: int, check: int, done: bool) -> bool:
        with self._stop_cv:
            slot = self._stop_slots.setdefault(check, {})
            slot[rank] = done
            self._stop_cv.notify_all()
            if not self._stop_cv.wait_for(lambda: len(slot) == self.size, timeout=TIMEOUT_S):
                raise TimeoutError("stop check never completed")
            return all(slot.values())


class _Dist:
    def __init__(self, world: _World, rank: int):
        pp = world.size
        self._world = world
        self.rank = rank
        self.pp_rank = rank
        self.pp_size = pp
        self.world_size = pp
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

    def allgather(self, obj):
        return self._world.gather(self.rank, "allgather", obj)

    def pp_allgather(self, obj):
        return self._world.gather(self.rank, "pp_allgather", obj)

    def tp_allgather(self, obj):
        raise AssertionError("tp_size is 1: no TP collective expected")


# --------------------------------------------------------------------------- device stand-in


class _Link:
    """Buffered stage boundary: sends queue into ``capacity`` receiver slots."""

    def __init__(self, capacity: int = 2):
        self._slots: queue.Queue = queue.Queue(maxsize=capacity)

    def send(self, microbatch: int):
        self._slots.put(microbatch, timeout=TIMEOUT_S)

    def recv(self) -> int:
        return self._slots.get(timeout=TIMEOUT_S)


class _Gpu:
    def __init__(self, stage, inbound, outbound, cost_s, log):
        self.stage = stage
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

    def _run(self):
        try:
            while True:
                job = self._jobs.get()
                if job is None:
                    return
                microbatch, done = job
                if self._inbound is not None:
                    received = self._inbound.recv()
                    assert received == microbatch, (self.stage, received, microbatch)
                self._log.busy(self.stage, +1)
                time.sleep(self._cost_s)
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
        self.launch_times: List[float] = []
        self.handled: Dict[int, int] = {r: 0 for r in range(pp_size)}
        self.max_lead = 0
        # rank -> [(iteration, request id)]
        self.terminated: Dict[int, List[tuple]] = {r: [] for r in range(pp_size)}
        # rank -> {request id: (iteration, state name)} when its transfer claim ended
        self.released: Dict[int, Dict[int, tuple]] = {r: {} for r in range(pp_size)}

    def busy(self, stage, delta):
        with self._lock:
            self.busy_now += delta
            self.max_busy = max(self.max_busy, self.busy_now)


class _Noop:
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

# --------------------------------------------------------------------------- transfer stand-in


class _Fabric:
    """How each stage's send of each request ends, and after how long.

    ``plan[(rank, request_id)]`` is one of ``ok`` (completes), ``fail`` (fails
    after its writes drained), ``peer_cancel`` (the generation side cancels it)
    or ``hang`` (never finishes until cancelled locally); default ``ok``.
    """

    def __init__(self, delay_s: float, plan: Optional[Dict[tuple, str]] = None):
        self.delay_s = delay_s
        self.plan = dict(plan or {})


class _TxSession:
    def __init__(self, rid: int, behaviour: str, delay_s: float):
        self.disagg_request_id = rid
        self._behaviour = behaviour
        self._due = time.monotonic() + delay_s
        self._cancelled = False
        self.closed = False

    @property
    def status(self) -> SessionStatus:
        if self._cancelled:
            return SessionStatus.CANCELLED
        if self._behaviour == "hang" or time.monotonic() < self._due:
            return SessionStatus.TRANSFERRING
        return {
            "ok": SessionStatus.TRANSFERRED,
            "fail": SessionStatus.ERROR,
            "peer_cancel": SessionStatus.CANCELLED,
        }[self._behaviour]

    def is_completed(self) -> bool:
        return self.status == SessionStatus.TRANSFERRED

    def has_failed(self) -> bool:
        return self.status in (SessionStatus.ERROR, SessionStatus.CANCELLED)

    def wait_complete(self, blocking: bool = True) -> Optional[WaitResult]:
        status = self.status
        if status == SessionStatus.TRANSFERRED:
            return WaitResult.COMPLETED
        if status in (SessionStatus.ERROR, SessionStatus.CANCELLED):
            return WaitResult.FAILED
        return WaitResult.TIMEOUT if blocking else None

    def has_transferring_tasks(self) -> bool:
        return False

    def cancel(self, by_peer: bool = False) -> bool:
        self._cancelled = True
        return True

    def close(self) -> bool:
        self.closed = True
        return True


class _TransferWorker:
    def __init__(self, rank: int, fabric: _Fabric):
        self._rank = rank
        self._fabric = fabric

    def create_tx_session(self, req):
        rid = get_unique_rid(req)
        return _TxSession(rid, self._fabric.plan.get((self._rank, rid), "ok"), self._fabric.delay_s)

    def sweep_stale_req_infos(self):
        pass

    def has_all_peer_req_infos_for_send(self, rid):
        return True


class _Transceiver(KvCacheTransceiverV2):
    """``KvCacheTransceiverV2`` whose KV publication is a stand-in session; the
    status, consensus, cancel and lagged-mode code is the real one."""

    def respond_and_send_async(self, req) -> None:
        self._ever_had_send_session = True
        rid = get_unique_rid(req)
        if self._get_or_create_send_session(req) is None:
            return
        self._send_reqs[rid] = req
        req.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS


def _make_transceiver(dist: _Dist, fabric: _Fabric, timeout_ms: int) -> _Transceiver:
    t = object.__new__(_Transceiver)
    t._dist = dist
    t._mapping = types.SimpleNamespace(
        pp_size=dist.pp_size,
        tp_size=1,
        cp_size=1,
        world_size=dist.pp_size,
        rank=dist.rank,
        enable_attention_dp=False,
    )
    t._send_sessions = {}
    t._recv_sessions = {}
    t._send_reqs = {}
    t._recv_reqs = {}
    t._wait_reqs = {}
    t._transfer_worker = _TransferWorker(dist.rank, fabric)
    t._ever_had_send_session = False
    t._ever_had_recv_session = False
    t._enable_pipelined_transfer = False
    t._fp4_mla_bridge_enabled = False
    t.kv_transfer_timeout_ms = timeout_ms
    t.kv_transfer_poll_interval_ms = 0
    t._sender_future_timeout_ms = 0
    t._init_sync_policy()
    return t


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
    """The fields the PP loop, the coordinator and the transceiver read."""

    def __init__(self, request_id: int, prompt_len: int):
        self.request_id = request_id
        self.py_request_id = request_id
        self.prompt_len = prompt_len
        self.context_current_position = 0
        self.context_chunk_size = 0
        self.py_batch_idx = None
        self.is_attention_dp_dummy = False
        self.is_dummy_request = False
        self.py_result = _PyResult()
        self.context_done = False
        self.state = LlmRequestState.CONTEXT_INIT
        self.is_context_only_request = True
        self.is_finished_due_to_cancellation = False
        self.is_finished_due_to_length = False
        self.is_child = False
        self.parent_request_id = None
        self.py_disaggregated_params = None
        self.py_kv_send_session_retired = False
        self.py_kv_transfer_start_time = None
        self.py_kv_transfer_timed_out = False
        self.is_disagg_generation_transmission_in_progress = False
        self.is_disagg_generation_init_state = False
        self.cached_tokens = 0
        self.last_chunk_handled = False

    @property
    def is_last_context_chunk(self) -> bool:
        return self.context_current_position + self.context_chunk_size >= self.prompt_len

    @property
    def is_first_context_chunk(self) -> bool:
        return self.context_current_position == 0

    @property
    def is_context_init_state(self) -> bool:
        return not self.context_done

    @property
    def is_context_finished(self) -> bool:
        return self.context_done

    def set_kv_cache_transfer_start(self, _time):
        pass

    def create_response(self, *args, **kwargs):
        return None


class _SampleEvent:
    def __init__(self, done: threading.Event):
        self._done = done

    def synchronize(self):
        if not self._done.wait(TIMEOUT_S):
            raise TimeoutError("the micro-batch never finished on the device")


class _AsyncTransferManager(AsyncTransferManager):
    """Records when each request's transfer claim ends, and in which state."""

    def end_transfer(self, request) -> bool:
        released = super().end_transfer(request)
        if released:
            log = self._log
            log.released[self._rank][request.py_request_id] = (
                self._executor.iter_counter,
                request.state,
            )
        return released


# --------------------------------------------------------------------------- executor


class _Executor(PyExecutor):
    """``PyExecutor`` with the real PP loop and executed-batch handling; the model,
    scheduler and response creation are stand-ins."""

    def _schedule(self):
        batch = ScheduledRequests()
        budget = self._token_budget
        chosen = []
        for req in self.active_requests:
            if budget == 0:
                break
            if req.context_done or req.request_id in self._inflight:
                continue
            req.context_chunk_size = min(budget, req.prompt_len - req.context_current_position)
            budget -= req.context_chunk_size
            chosen.append(req)
        batch.reset_context_requests(chosen)
        return batch, [], len(chosen)

    @property
    def should_stop_processing(self):
        # Lockstep: every rank checks at the same iterations (a test-only
        # rendezvous, not counted as a collective).
        if self.active_requests or self.iter_counter % STOP_CHECK_EVERY:
            return False
        if self._stop_checked == self.iter_counter:
            return self._stop_answer
        self._stop_checked = self.iter_counter
        self._stop_answer = self._world.all_done(
            self.dist.rank, self.iter_counter, self._locally_done()
        )
        return self._stop_answer

    def _locally_done(self) -> bool:
        if self.async_transfer_manager.has_any_inflight_requests():
            return False
        if self.kv_cache_transceiver is not None and self.kv_cache_transceiver._send_sessions:
            return False
        handler = self._disagg_pp_termination_handler
        if self._require_terminated and handler is not None and handler._pending_termination:
            return False
        return True

    def _fetch_and_activate_new_requests(self):
        if self.iter_counter > self._max_iterations:
            raise RuntimeError("the run did not settle within the iteration budget")
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

    def _handle_first_token_response(self, scheduled_batch):
        pass

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
            req.context_current_position += req.context_chunk_size
            if req.context_current_position >= req.prompt_len:
                req.context_done = True

    def _launch(self, scheduled_batch) -> threading.Event:
        microbatch = self._launched
        self._launched += 1
        if self.dist.rank == 0:
            with self._log._lock:
                self._log.launch_times.append(time.perf_counter())
                lead = self._launched - self._log.handled[0]
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

    # Executed-batch handling: the real ``_handle_executed_batch`` runs; these are
    # the pieces it calls that need the model or the C++ request.
    def _update_requests(self, sample_state, *args, **kwargs):
        with self._log._lock:
            self._log.handled[self.dist.rank] += 1
        for req in sample_state.requests:
            req.last_chunk_handled = True

    def _save_kv_to_connector_async(self, scheduled_requests):
        pass

    def _handle_responses(self, emit_first_iter: bool = True):
        # A context-only request leaves the active list once its send started
        # (it is "finished"); its KV stays until the transfer releases it.
        keep = []
        for req in self.active_requests:
            if self.kv_cache_transceiver is None and req.last_chunk_handled:
                # No transceiver (the pipeline baseline): done once handled.
                continue
            if req.state in (
                LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS,
                LlmRequestState.DISAGG_CONTEXT_COMPLETE,
                LlmRequestState.DISAGG_TRANS_ERROR,
            ):
                continue
            keep.append(req)
        self.active_requests[:] = keep
        return []

    def _commit_kv_cache_stats(self, scheduled_requests):
        pass

    def _flush_iter_stats_synced(self):
        pass

    def _do_terminate_request(self, request):
        self._log.terminated[self.dist.rank].append((self.iter_counter, request.py_request_id))


def _make_executor(world, rank, num_micro_batches, prompts, token_budget, gpu, log, cfg):
    exe = object.__new__(_Executor)
    exe.dist = _Dist(world, rank)
    exe._world = world
    exe.device_id = 0
    exe._log = log
    exe._gpu = gpu
    exe._launched = 0
    exe._inflight = set()
    exe._token_budget = token_budget
    exe._stop_checked = -1
    exe._stop_answer = False
    exe._require_terminated = cfg["require_terminated"]
    exe._max_iterations = cfg["max_iterations"]
    exe.inflight_req_ids = _NOOP
    exe.active_requests = [_Request(i, p) for i, p in enumerate(prompts)]
    exe.waiting_queue = []
    exe.canceled_req_ids = []
    exe.is_shutdown = True
    exe.kv_connector_manager = None
    exe.hang_detector = _NOOP
    exe.enable_iter_perf_stats = False
    exe.enable_attention_dp = False
    exe.enable_kv_cache_reuse = cfg["block_reuse"]
    exe.iter_counter = 0
    exe.speculation_gate = None
    exe._mm_encoder_item_scheduling_enabled = False
    exe._is_kv_manager_v2 = False
    # The leader-batch adoption: the follower pass below schedules
    # nothing, so the followers run the first rank's chunks by adoption.
    exe._pp_adopt_leader_chunks = True
    exe._pp_rebalance_drain_iters = None
    exe._pending_recompute_pause_ids = set()
    exe._prefetched_request_ids = set()
    exe._pending_transfer_responses = []
    exe._pending_response_terminations = []
    exe._profile_enabled = False
    local_pass = types.SimpleNamespace(
        context_requests=[], generation_requests=[], fitting_disagg_gen_init_requests=[]
    )

    def _adopt(req, size):
        req.context_chunk_size = size
        return True

    exe.scheduler = _Noop(
        schedule_request=lambda active, inflight: local_pass, adopt_context_chunk=_adopt
    )
    exe.kv_cache_manager = object()
    exe.model_engine = types.SimpleNamespace()
    exe.resource_manager = _NOOP
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
    # The real disagg plumbing.
    exe.kv_cache_transceiver = (
        _make_transceiver(exe.dist, cfg["fabric"], cfg["timeout_ms"])
        if cfg["transceiver"]
        else None
    )
    transfers = _AsyncTransferManager(
        types.SimpleNamespace(resource_managers={}), should_store_blocks=False
    )
    transfers._log = log
    transfers._rank = rank
    transfers._executor = exe
    exe.async_transfer_manager = transfers
    if exe.kv_cache_transceiver is not None and not cfg.get("lagged", True):
        # The stock arm: a transceiver the lagged agreement does not cover (as
        # with pipelined transfer) keeps the PP-group collectives.
        exe.kv_cache_transceiver.pp_lagged_consensus_refusal = lambda: ["the stock collectives"]
    exe._init_disagg_pp_termination()
    return exe


def _run(
    monkeypatch,
    *,
    lagged: bool,
    pp_size: int = 4,
    ring_depth: Optional[int] = None,
    prompts=(128,) * 12,
    token_budget: int = 64,
    cost_s: float = 0.002,
    fabric: Optional[_Fabric] = None,
    timeout_ms: int = 600_000,
    block_reuse: bool = True,
    require_terminated: Optional[bool] = None,
    max_iterations: int = 4000,
    transceiver: bool = True,
):
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(py_executor_module, "CUASSERT", lambda result: result)
    monkeypatch.setattr(py_executor_module, "cudart", MagicMock())
    n = ring_slots(pp_size, ring_depth)
    world = _World(pp_size)
    log = _Log(pp_size)
    links = [_Link() for _ in range(pp_size - 1)]
    gpus = [
        _Gpu(
            s,
            links[s - 1] if s > 0 else None,
            links[s] if s < pp_size - 1 else None,
            cost_s,
            log,
        )
        for s in range(pp_size)
    ]
    cfg = {
        "fabric": fabric or _Fabric(delay_s=0.003),
        "timeout_ms": timeout_ms,
        "block_reuse": block_reuse,
        # Stock frees a queued request only while executed batches are handled;
        # an idle tail leaves the last ones queued, so stock is not held to it.
        "require_terminated": lagged if require_terminated is None else require_terminated,
        "max_iterations": max_iterations,
        "transceiver": transceiver,
        "lagged": lagged,
    }
    executors = [
        _make_executor(world, r, n, list(prompts), token_budget, gpus[r], log, cfg)
        for r in range(pp_size)
    ]
    errors: List[BaseException] = []

    def _rank(exe):
        try:
            exe._executor_loop_pp()
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=_rank, args=(exe,), daemon=True) for exe in executors]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(TIMEOUT_S * 2)
    for gpu in gpus:
        gpu.stop()
    assert not any(t.is_alive() for t in threads), "the PP loop deadlocked"
    assert not errors, errors
    assert not [g.error for g in gpus if g.error is not None]
    return executors, world, log, n


def _period(log) -> float:
    """Mean launch interval of the first rank over the middle half of the run."""
    times = log.launch_times
    first, last = len(times) // 4, (3 * len(times)) // 4
    return (times[last] - times[first]) / (last - first)


def _num_microbatches(prompts, token_budget):
    return sum(-(-p // token_budget) for p in prompts)


# --------------------------------------------------------------------------- (a) / (b)


def test_wiring_stock_where_it_does_not_apply_and_lagged_where_it_does(monkeypatch):
    """A transceiver the agreement does not cover: the stock handler and no
    agreement; otherwise the lagged pair."""
    world = _World(4)
    log = _Log(4)
    cfg = {
        "fabric": _Fabric(0.0),
        "timeout_ms": 1000,
        "block_reuse": True,
        "require_terminated": False,
        "max_iterations": 10,
        "transceiver": True,
        "lagged": False,
    }
    exe = _make_executor(world, 1, 4, [64], 64, None, log, cfg)
    assert type(exe._disagg_pp_termination_handler) is DisaggPPTerminationHandler
    assert exe._pp_disagg_lag is None
    assert not exe.kv_cache_transceiver._pp_lagged_consensus
    cfg["lagged"] = True
    exe = _make_executor(world, 1, 8, [64], 64, None, log, cfg)
    assert type(exe._disagg_pp_termination_handler) is LaggedDisaggPPTerminationHandler
    assert isinstance(exe._pp_disagg_lag, PPDisaggLaggedConsensus)
    assert exe._pp_disagg_lag.lag == 7  # probe_lag(relay_lag(4, 8, sync)) = 8 - 1
    assert exe.kv_cache_transceiver._pp_lagged_consensus
    cfg["block_reuse"] = False
    exe = _make_executor(world, 2, 4, [64], 64, None, log, cfg)
    assert exe._disagg_pp_termination_handler is None
    assert exe._pp_disagg_lag.lag == 3


def test_stock_ctx_enters_pp_collectives_every_iteration_and_halves_the_pipeline(monkeypatch):
    prompts, budget, cost = (128,) * 16, 64, 0.008
    executors, world, log, n = _run(
        monkeypatch, lagged=False, prompts=prompts, token_budget=budget, cost_s=cost
    )
    iterations = executors[0].iter_counter
    handled = log.handled[0]
    assert handled == _num_microbatches(prompts, budget)
    calls = world.collectives
    # Every rank entered the same collectives (the fake checks the order).
    assert all(calls[r] == calls[0] for r in calls)
    num_allgather = calls[0].count("allgather")
    num_pp_allgather = calls[0].count("pp_allgather")
    # Two receive-side world allgathers per iteration, sends or not.
    assert num_allgather == 2 * iterations
    # Two pp_allgathers per status pass: one per iteration once a send exists,
    # plus two passes per handled batch after that.
    assert num_pp_allgather >= 2 * handled
    period = _period(log)
    print(
        f"\n[stock] iterations={iterations} handled={handled} "
        f"allgather={num_allgather} pp_allgather={num_pp_allgather} "
        f"period={period * 1e3:.2f} ms (stage {cost * 1e3:.1f} ms) "
        f"max_busy={log.max_busy} max_lead={log.max_lead}"
    )
    # At most two micro-batches are ever on the devices at once (4 stages).
    assert log.max_busy <= 2
    assert period >= 1.6 * cost
    # Every send settled the same way on every stage.
    for r in range(1, 4):
        assert set(log.released[r]) == set(log.released[0]) == set(range(len(prompts)))


@pytest.mark.parametrize("ring_depth", [None, 8])
def test_lagged_ctx_enters_no_collective_and_fills_the_ring(monkeypatch, ring_depth):
    prompts, budget, cost = (128,) * 16, 64, 0.008
    executors, world, log, n = _run(
        monkeypatch,
        lagged=True,
        ring_depth=ring_depth,
        prompts=prompts,
        token_budget=budget,
        cost_s=cost,
    )
    iterations = executors[0].iter_counter
    assert log.handled[0] == _num_microbatches(prompts, budget)
    assert all(not calls for calls in world.collectives.values()), world.collectives
    period = _period(log)
    print(
        f"\n[lagged ring={n}] iterations={iterations} handled={log.handled[0]} "
        f"collectives=0 period={period * 1e3:.2f} ms (stage {cost * 1e3:.1f} ms) "
        f"max_busy={log.max_busy} max_lead={log.max_lead}"
    )
    assert log.max_busy == 4
    assert log.max_lead == n
    # Loose: the stand-in's host work is a visible share of an 8 ms stage; the
    # bound relative to a pipeline without a transceiver is in the test below.
    assert period <= 1.45 * cost
    # Every send completed and every request was freed once, in the same
    # iteration on every stage.
    for r in range(4):
        assert set(log.released[r]) == set(range(len(prompts)))
        assert all(
            state == LlmRequestState.DISAGG_CONTEXT_COMPLETE
            for _, state in log.released[r].values()
        )
        assert log.terminated[r] == log.terminated[0]
        assert sorted(rid for _, rid in log.terminated[r]) == list(range(len(prompts)))
    for r in range(1, 4):
        assert log.released[r] == log.released[0]
    agreement = executors[0]._pp_disagg_lag
    assert agreement.num_completed == len(prompts)
    assert agreement.num_terminated == len(prompts)


@pytest.mark.parametrize("ring_depth", [None, 8])
def test_lagged_exchange_matches_every_message_and_bounds_the_free(monkeypatch, ring_depth):
    """Every report and decision is received by the end of the loop (``drain``
    included), and with an instant transfer a request is freed on every stage
    no later than two agreement rounds after its send starts.

    The bound is ``2 * (lag + 1)`` iterations: one round settles the send
    (release), the release queues the termination, and a second round frees it.
    Stock (vote ring) frees about one iteration after the send starts.
    """
    starts = {}
    start_transfer = _AsyncTransferManager.start_transfer

    def _record_start(self, request, *args, **kwargs):
        starts[(self._rank, request.py_request_id)] = self._executor.iter_counter
        return start_transfer(self, request, *args, **kwargs)

    monkeypatch.setattr(_AsyncTransferManager, "start_transfer", _record_start)
    prompts = (128,) * 12
    executors, world, log, n = _run(
        monkeypatch,
        lagged=True,
        ring_depth=ring_depth,
        prompts=prompts,
        fabric=_Fabric(0.0),
    )
    lag = executors[0]._pp_disagg_lag.lag
    tags = (PP_DISAGG_REPORT_TAG, PP_DISAGG_DECISION_TAG)
    unmatched = {
        key: q.qsize() for key, q in world._queues.items() if key[2] in tags and not q.empty()
    }
    assert not unmatched, f"lagged-agreement messages never received: {unmatched}"
    iterations = executors[0].iter_counter
    assert all(exe._pp_disagg_lag.iteration == iterations for exe in executors)
    for r in range(4):
        freed = {rid: it for it, rid in log.terminated[r]}
        assert sorted(freed) == list(range(len(prompts)))
        for rid, it in freed.items():
            assert it - starts[(r, rid)] <= 2 * (lag + 1), (r, rid, it, starts[(r, rid)], lag)


def test_lagged_without_block_reuse_frees_at_once(monkeypatch):
    prompts = (128,) * 6
    executors, world, log, n = _run(monkeypatch, lagged=True, prompts=prompts, block_reuse=False)
    assert all(not calls for calls in world.collectives.values())
    for r in range(4):
        assert sorted(rid for _, rid in log.terminated[r]) == list(range(len(prompts)))
        # Freed in the iteration the agreement released it, on every stage.
        assert [(it, rid) for it, rid in log.terminated[r]] == sorted(
            (it, rid) for rid, (it, _) in log.released[r].items()
        )
    for r in range(1, 4):
        assert log.terminated[r] == log.terminated[0]


def test_cpu_period_against_a_ctx_without_transceiver(monkeypatch):
    """Stage period on the stand-in with a KV-sending transceiver, stock and
    lagged, against the same pipeline with no transceiver at all.

    The bound is looser than on a GPU: the stand-in's own per-iteration host
    work (Python threads sharing the GIL) is a larger share of a stage here,
    hence 1.15x.
    """
    prompts, budget, cost = (128,) * 16, 64, 0.016
    periods = {}
    for name, kwargs in (
        ("no transceiver", {"lagged": False, "transceiver": False}),
        ("stock", {"lagged": False}),
        ("lagged", {"lagged": True}),
    ):
        _, world, log, _ = _run(
            monkeypatch, prompts=prompts, token_budget=budget, cost_s=cost, **kwargs
        )
        periods[name] = _period(log)
        if name == "no transceiver":
            assert all(not calls for calls in world.collectives.values())
    base = periods["no transceiver"]
    print(
        "\n[period vs no transceiver] "
        + ", ".join(f"{k} {v * 1e3:.2f} ms ({v / base:.2f}x)" for k, v in periods.items())
        + f"; stage {cost * 1e3:.1f} ms"
    )
    assert periods["lagged"] <= 1.15 * base
    assert periods["stock"] >= 1.6 * base


# --------------------------------------------------------------------------- (c) failures


# The PP loop flags transfer timeouts only while it handles executed batches
# (_handle_executed_batch -> check_transfer_timeouts), so these runs keep
# work flowing well past the timeout: 40 requests x 2 chunks x 2 ms per stage.
FAILURE_RUN_REQUESTS = 40
FAILURE_TIMEOUT_MS = 60


def _outcomes(log, rank):
    return {rid: state for rid, (_, state) in log.released[rank].items()}


@pytest.mark.parametrize("lagged", [False, True])
def test_send_failure_on_one_stage_fails_the_request_on_every_stage(monkeypatch, lagged):
    fabric = _Fabric(0.003, {(2, 3): "fail"})
    prompts = (128,) * 6
    executors, world, log, n = _run(monkeypatch, lagged=lagged, prompts=prompts, fabric=fabric)
    for r in range(4):
        outcomes = _outcomes(log, r)
        assert set(outcomes) == set(range(len(prompts)))
        assert outcomes[3] == LlmRequestState.DISAGG_TRANS_ERROR
        assert all(
            state == LlmRequestState.DISAGG_CONTEXT_COMPLETE
            for rid, state in outcomes.items()
            if rid != 3
        )
        assert executors[r].kv_cache_transceiver._send_sessions == {}
    for r in range(1, 4):
        assert _outcomes(log, r) == _outcomes(log, 0)
    if lagged:
        # Same iteration everywhere; a failed send is not freed by the
        # release path (stock behaviour, both modes).
        for r in range(1, 4):
            assert log.released[r] == log.released[0]
            assert log.terminated[r] == log.terminated[0]
        assert 3 not in {rid for _, rid in log.terminated[0]}
        assert all(not calls for calls in world.collectives.values())


@pytest.mark.parametrize("lagged", [False, True])
def test_peer_cancel_on_one_stage_retires_the_send_on_every_stage(monkeypatch, lagged):
    """The GEN cancels one stage's send. Stock (no in-flight cancel): the sessions
    are retired as cancelled everywhere and each stage then releases the request
    when its own transfer timeout fires."""
    fabric = _Fabric(0.003, {(1, 2): "peer_cancel"})
    prompts = (128,) * FAILURE_RUN_REQUESTS
    executors, world, log, n = _run(
        monkeypatch, lagged=lagged, prompts=prompts, fabric=fabric, timeout_ms=FAILURE_TIMEOUT_MS
    )
    for r in range(4):
        outcomes = _outcomes(log, r)
        assert set(outcomes) == set(range(len(prompts)))
        assert executors[r].kv_cache_transceiver._send_sessions == {}
    for r in range(1, 4):
        assert _outcomes(log, r) == _outcomes(log, 0)
    if lagged:
        for r in range(4):
            assert sorted(rid for _, rid in log.terminated[r]) == list(range(len(prompts)))
        for r in range(1, 4):
            assert log.terminated[r] == log.terminated[0]
        assert executors[0]._pp_disagg_lag.num_cancelled == 1


@pytest.mark.parametrize("lagged", [False, True])
def test_send_timeout_on_one_stage_resolves_on_every_stage(monkeypatch, lagged):
    """One stage's send never finishes. Each stage's own transfer timeout cancels
    and releases the request (rank-local, as stock); the agreement then frees it
    in one iteration on every stage."""
    fabric = _Fabric(0.003, {(3, 1): "hang"})
    prompts = (128,) * FAILURE_RUN_REQUESTS
    executors, world, log, n = _run(
        monkeypatch, lagged=lagged, prompts=prompts, fabric=fabric, timeout_ms=FAILURE_TIMEOUT_MS
    )
    for r in range(4):
        outcomes = _outcomes(log, r)
        assert set(outcomes) == set(range(len(prompts)))
        assert executors[r].kv_cache_transceiver._send_sessions == {}
    if lagged:
        for r in range(4):
            assert sorted(rid for _, rid in log.terminated[r]) == list(range(len(prompts)))
        for r in range(1, 4):
            assert log.terminated[r] == log.terminated[0]
        assert all(not calls for calls in world.collectives.values())


# --------------------------------------------------------------------------- (d) gen-first


@pytest.mark.parametrize("lagged", [False, True])
def test_generation_first_requests_dispatch_only_where_peer_info_arrived(monkeypatch, lagged):
    """Generation-first requests wait until the GEN's peer info is on every stage.

    Peer info reaches the stages at different times (stage 2 last). No stage may
    start a request's KV send before that stage has the peer info, every stage
    promotes the same requests, and with the lagged agreement no collective is entered.
    """
    gen_first = {1, 4, 7}
    arrival_s = {0: 0.0, 1: 0.03, 2: 0.15, 3: 0.08}
    t0 = time.monotonic()
    violations = []
    dispatched = {r: set() for r in range(4)}

    def _has_peer_info(self, rid):
        return rid not in gen_first or time.monotonic() - t0 >= arrival_s[self._rank]

    monkeypatch.setattr(_TransferWorker, "has_all_peer_req_infos_for_send", _has_peer_info)
    respond = _Transceiver.respond_and_send_async

    def _respond(self, req):
        rid = get_unique_rid(req)
        if not self._transfer_worker.has_all_peer_req_infos_for_send(rid):
            violations.append((self._dist.rank, rid))
        dispatched[self._dist.rank].add(rid)
        return respond(self, req)

    monkeypatch.setattr(_Transceiver, "respond_and_send_async", _respond)
    fetch = _Executor._fetch_and_activate_new_requests

    def _fetch(self):
        fetch(self)
        if getattr(self, "_gen_first_fed", False):
            return []
        self._gen_first_fed = True
        new = [req for req in self.active_requests if req.py_request_id in gen_first]
        for req in new:
            req.py_disaggregated_params = types.SimpleNamespace(
                schedule_style=DisaggScheduleStyle.GENERATION_FIRST, disagg_request_id=None
            )
        return new

    monkeypatch.setattr(_Executor, "_fetch_and_activate_new_requests", _fetch)
    schedule = _Executor._schedule

    def _schedule(self):
        held = self.active_requests
        self.active_requests = [
            req for req in held if req.state != LlmRequestState.DISAGG_CONTEXT_WAIT_SCHEDULER
        ]
        try:
            return schedule(self)
        finally:
            self.active_requests = held

    monkeypatch.setattr(_Executor, "_schedule", _schedule)
    prompts = (128,) * 10
    executors, world, log, n = _run(monkeypatch, lagged=lagged, prompts=prompts)
    assert not violations, f"KV send started before that stage had peer info: {violations}"
    for r in range(4):
        assert dispatched[r] == set(range(len(prompts)))
        assert set(log.released[r]) == set(range(len(prompts)))
        assert executors[r].kv_cache_transceiver._wait_reqs == {}
    if lagged:
        assert all(not calls for calls in world.collectives.values())
        assert all(exe._pp_disagg_lag.num_promoted == len(gen_first) for exe in executors)
