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
"""Tests for RankFailureRecovery, which takes part in every membership change of one rank."""

import contextlib
import os
import pickle
import re
import threading
import time
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
from torch.distributed import DistStoreError

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager import (
    DeepseekV4CacheManager,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import DSACacheManagerV2
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.distributed.host_exchange import MnnvlDist
from tensorrt_llm._torch.distributed.membership import (
    RENDEZVOUS_ENV,
    SHARED_SECRET_ENV,
    MembershipRecord,
    MembershipState,
    Reply,
    ResumeState,
    StoreGroup,
    UnsignedValueError,
    next_epoch,
)
from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMappingError, MnnvlPeerHandleError
from tensorrt_llm._torch.distributed.startup_record import StartupRecord
from tensorrt_llm._torch.moe.fused_moe.communication.execution_state import MoeExecutionState
from tensorrt_llm._torch.moe.fused_moe.ep_group_health import EPGroupHealth
from tensorrt_llm._torch.moe.fused_moe.eplb_expert_restore import (
    EplbExpertRestore,
    ExpertRestorePlanMismatchError,
)
from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import MoeLoadBalancer
from tensorrt_llm._torch.pyexecutor import rank_failure_memory as memory_module
from tensorrt_llm._torch.pyexecutor import rank_failure_recovery as recovery_module
from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    ExecutorRequestQueue,
    RequestQueueItem,
)
from tensorrt_llm._torch.pyexecutor.hang_detector import HangDetector
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.mamba_cache_manager_v2 import (
    MambaHybridCacheManagerV2,
)
from tensorrt_llm._torch.pyexecutor.llm_request import PyResult
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor, WaitingState
from tensorrt_llm._torch.pyexecutor.rank_failure_memory import StartupMemory
from tensorrt_llm._torch.pyexecutor.rank_failure_recovery import RankFailureRecovery
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import FCFSWaitingQueue
from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import RankState
from tensorrt_llm.executor.rank_failure_requests import RankFailureError
from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY
from tensorrt_llm.runtime.kv_cache_manager_v2 import OutOfPagesError

pytestmark = pytest.mark.cpu_only

WORLD = 4
FAILED = 2
SURVIVORS = 0b1011
ALL_RANKS = (1 << WORLD) - 1
JOINED = (0, 0, 1, 0)  # The first replacement of rank 2 stands for it.
MIB = 1 << 20
# A rank's KV cache pool and free device memory once the deployment serves.
STARTUP_MEMORY = StartupMemory(kv_cache_quota=64 * MIB)
FREE_BYTES = 1 << 30
# Where the active ranks resume once the replacement of rank 2 joins them.
JOIN_RESUME = ResumeState(iteration=20, epoch=40)
# The host exchange epoch at which rank 0 admits the replacement of rank 2.
ADMISSION_EPOCH = 41
TIMEOUT_S = 5.0


@pytest.fixture(autouse=True)
def _no_cuda_synchronize(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)


@pytest.fixture(autouse=True)
def pause_words(monkeypatch):
    """Give each rank a pause word in this process's memory instead of /dev/shm, keyed by path."""
    words = {}
    monkeypatch.setattr(recovery_module, "pause_word_path", lambda rank: f"rank {rank}")
    monkeypatch.setattr(
        recovery_module,
        "open_pause_word",
        lambda path: words.setdefault(path, memoryview(bytearray(4)).cast("i")),
    )
    return words


@pytest.fixture
def shared_secret(monkeypatch):
    monkeypatch.setenv(SHARED_SECRET_ENV, "ab" * 32)
    monkeypatch.setenv(RENDEZVOUS_ENV, "head-node:29600")


@pytest.fixture
def fresh_startup_record(monkeypatch):
    """Give this process a startup record with nothing recorded yet."""
    monkeypatch.setattr(Distributed, "startup_record", StartupRecord())


class FakeExchange:
    """Host exchange of one rank, in which rank 0 sends ``rank0_object``."""

    def __init__(self):
        self.interrupted = None
        self.on_interrupt = None
        self.waiting_for = 0
        self.waited_epochs = ()
        self.stopped_epoch = 6
        self.epoch = 40
        self.installs = []
        self.admits = []
        self.wakes = 0
        self.rank0_object = None
        self.exchanges = []

    def set_interrupt_handler(self, interrupted, on_interrupt=None):
        self.interrupted, self.on_interrupt = interrupted, on_interrupt

    def wake(self):
        self.wakes += 1

    def exchange(self, payload, readers, recipients):
        """Record this rank's object, readers and recipients, and return rank 0's payload if read."""
        self.exchanges.append((pickle.loads(payload), readers, recipients))
        rank0_payload = pickle.dumps(self.rank0_object)
        return [rank0_payload if r == 0 and r in readers else None for r in range(WORLD)]

    def install(self, comm, active_mask, epoch):
        self.installs.append((comm.allgather(b"row-mine"), active_mask, epoch))

    def admit(self, active_mask, epoch):
        self.admits.append((active_mask, epoch))


class FakeStore:
    """In-memory rendezvous store for one rank."""

    def __init__(self):
        self.values = {}
        self._changed = threading.Condition()

    def set(self, key, value):
        with self._changed:
            self.values[key] = value
            self._changed.notify_all()

    def wait(self, keys, timeout):
        with self._changed:
            if not self._changed.wait_for(lambda: self.check(keys), timeout.total_seconds()):
                raise DistStoreError(f"wait timeout after {timeout}")

    def get(self, key):
        return self.values[key]

    def add(self, key, amount):
        self.values[key] = self.values.get(key, 0) + amount
        return self.values[key]

    def check(self, keys):
        return all(key in self.values for key in keys)

    def multi_get(self, keys):
        return [self.values[key] for key in keys]

    def delete_key(self, key):
        return self.values.pop(key, None) is not None


class FakeClient:
    """Membership client whose ``answers`` map a (reply, generation) report to the next records."""

    def __init__(self, incarnation=0):
        self.incarnation = incarnation
        self.record = None
        self.reports = []
        self.answers = {}
        self.callbacks = None

    def bind(self, on_record, on_tick=None, on_error=None):
        self.callbacks = SimpleNamespace(on_record=on_record, on_error=on_error)
        if self.record is not None:
            on_record(self.record)

    def deliver(self, record):
        self.record = record
        self.callbacks.on_record(record)

    def fail(self, error):
        self.callbacks.on_error(error)

    def report(self, **changes):
        self.reports.append(changes)
        for record in self.answers.get((changes.get("reply"), changes.get("reply_generation")), ()):
            self.deliver(record)

    def stop(self):
        pass


def _executor(rank=0):
    """Return an overlap executor with the steps that rank failure recovery calls."""
    executor = SimpleNamespace(
        dist=MagicMock(spec=MnnvlDist, rank=rank, world_size=WORLD, tp_size=WORLD),
        iter_counter=5,
        hang_detector=Mock(spec=HangDetector, paused=Mock(return_value=False), timeout=300),
        executor_request_queue=Mock(wraps=ExecutorRequestQueue(MagicMock(), 8, False, 0)),
        waiting_queue=[],
        canceled_req_ids=[],
        is_shutdown=False,
        request_accumulated=[],
        active_requests=[],
        previous_batch=None,
        drafter=None,
        use_spec_decode=False,
        guided_decoder=None,
        capturable_guided_decoder=None,
        perf_manager=MagicMock(),
        model_engine=SimpleNamespace(moe_load_balancer=None, model=torch.nn.Module()),
        kv_cache_manager=SimpleNamespace(gpu_quota=STARTUP_MEMORY.kv_cache_quota),
        fail_requests_by_id=Mock(),
        recompute_requests=Mock(),
        worker_thread=None,
        rank_failure_recovery=None,
        _wait_for_model_engine_input_copy=MagicMock(),
        resource_manager=SimpleNamespace(resource_managers={}),
    )
    for step in (
        "_accept_or_discard_previous_batch",
        "_discard_previous_batch",
        "resume_at_iteration",
        "known_request_ids",
        "export_waiting_state",
        "import_waiting_state",
    ):
        setattr(executor, step, partial(getattr(PyExecutor, step), executor))

    def retire():  # Accept or drop the step in flight, as the executor does.
        executor._accept_or_discard_previous_batch()
        executor.previous_batch = None

    executor.retire_inflight_batch = Mock(side_effect=retire)
    return executor


def _bind_moe_state(monkeypatch, r):
    """Bind ``r.moe`` to the MoE communication of the rank's model, as prepare_model does."""
    comm = SimpleNamespace(execution_state=r.moe)
    model = r.executor.model_engine.model
    monkeypatch.setattr(recovery_module, "moe_comms", lambda m: [comm] if m is model else [])


def _moe_state():
    moe = MagicMock(spec=MoeExecutionState)
    moe.ep_group_health = MagicMock(spec=EPGroupHealth)
    moe.aborted.return_value = False
    moe.waited_flags.return_value = {}
    moe.workspace_bytes.return_value = 0
    moe.workspace_rows_needing_fresh_pages.return_value = frozenset(range(WORLD))
    return moe


@dataclass
class _Rank:
    """The recovery of one rank and the collaborators the test gives it."""

    recovery: Optional[RankFailureRecovery]
    executor: SimpleNamespace
    exchange: FakeExchange
    moe: MagicMock
    store: FakeStore
    client: FakeClient


def _unstarted_rank(rank, incarnation=0):
    return _Rank(
        None, _executor(rank), FakeExchange(), _moe_state(), FakeStore(), FakeClient(incarnation)
    )


def _rank(rank=0):
    """Return an original rank whose recovery took over its membership client."""
    r = _unstarted_rank(rank)
    r.recovery = RankFailureRecovery(r.executor, r.exchange, r.moe, r.store, STARTUP_MEMORY)
    with patch.object(recovery_module, "membership_client", return_value=r.client):
        r.recovery.start()
    r.executor.rank_failure_recovery = r.recovery
    return r


def _replacement(monkeypatch):
    """Return the first replacement of rank 2 before RankFailureRecovery.create takes it over."""
    monkeypatch.setattr(StartupRecord, "replacement", True)
    monkeypatch.setenv(RENDEZVOUS_ENV, "host:1")
    monkeypatch.setattr(Distributed, "startup_record", StartupRecord())
    r = _unstarted_rank(FAILED, incarnation=1)
    r.executor.dist.host_exchange = r.exchange
    r.executor.iter_counter = 0
    monkeypatch.setattr(recovery_module, "membership_client", lambda: r.client)
    monkeypatch.setattr(recovery_module, "rendezvous_store", lambda: r.store)
    monkeypatch.setattr(recovery_module, "authority_process", lambda: None)
    monkeypatch.setattr(recovery_module, "publish_tuned_tactics", lambda store, rank: None)
    _bind_moe_state(monkeypatch, r)
    return r


def _record(state, generation=1, active_mask=SURVIVORS, resume=ResumeState()):
    return MembershipRecord(
        generation=generation,
        state=state,
        active_mask=active_mask,
        resume=resume,
        incarnations=(0,) * WORLD,
    )


def _joining_record(state, joining=1 << FAILED, generation=2):
    return MembershipRecord(
        generation=generation,
        state=state,
        active_mask=ALL_RANKS,
        resume=JOIN_RESUME,
        incarnations=JOINED,
        joining_mask=joining,
    )


def _replies(r):
    return [report["reply"] for report in r.client.reports if "reply" in report]


def _exit_raises(monkeypatch):
    def exit_(code):
        raise SystemExit(code)

    monkeypatch.setattr(recovery_module.os, "_exit", exit_)


def _wait_until(condition, timeout_s=TIMEOUT_S):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def _publish_rows(store, record, rank):
    """Publish the host exchange rows the other active ranks send when ``record`` installs."""
    prefix = f"host_exchange_rows/{record.generation}/0"
    for peer in range(WORLD):
        if peer != rank and record.is_active(peer):
            store.set(f"{prefix}/{peer}", pickle.dumps(f"rows-{peer}".encode()))
            store.add(StoreGroup._counter(prefix, peer), 1 << peer % 63)


def _install(r, record):
    """Deliver ``record``, an INSTALL, and take part in the change the host exchange stops for."""
    _publish_rows(r.store, record, r.recovery.rank)
    r.client.deliver(record)
    r.exchange.on_interrupt()


def _item(req_id, target=None, relax=False):
    item = MagicMock()
    item.id = req_id
    item.request.client_id = 100 + req_id
    item.request.py_scheduling_params = (
        None if target is None else Mock(attention_dp_rank=target, attention_dp_relax=relax)
    )
    return item


# Membership changes on the executor thread.


@pytest.mark.parametrize(
    "active_mask, joining_mask, aborts",
    [
        (0b1111, 0b0100, False),  # A replacement for failed rank 2 joins and no process leaves.
        (0b1011, 0b0010, True),  # A replacement takes over active rank 1, whose old process leaves.
        (0b1101, 0b0100, True),  # Rank 1 fails while the replacement of rank 2 joins.
    ],
)
def test_only_removal_aborts_steps_in_flight(active_mask, joining_mask, aborts):
    """Check that only a change that drops a process aborts the all-to-all."""
    r = _rank()
    r.client.deliver(_record(MembershipState.COMMITTED))  # Rank 2 failed earlier.
    proposal = _record(MembershipState.PROPOSE, generation=2, active_mask=active_mask).model_copy(
        update=dict(joining_mask=joining_mask)
    )
    r.client.deliver(proposal)
    assert r.moe.request_abort.called == aborts
    r.executor.executor_request_queue.wake.assert_called_once()
    # The pending change wakes the host exchange wait, and the commit before it does not.
    assert r.exchange.wakes == 1
    # An admission stops the host exchange only after a forward vote.
    assert r.exchange.interrupted() == aborts


def _wait_for_release_worker(r):
    _wait_until(lambda: not r.recovery._memory.release_pending())


def _vote(r, batch_size=4, peers=((3, 0, 0), (5, 0, 0), (2, 0, 0)), wait_for_release=True):
    """Return the (batch size, held, release) row ``r`` sent in a forward vote and the outcome.

    With ``wait_for_release``, the vote follows the end of the spare slot release worker.
    """
    if wait_for_release:
        _wait_for_release_worker(r)
    sent = []

    def gather(row, active_only):
        assert active_only
        sent.append(row)
        return np.array([row, *peers])

    r.executor.dist.tp_allgather_int64.side_effect = gather
    outcome = r.recovery.forward_vote(batch_size)
    return sent[0], outcome


def test_admission_stops_exchange_after_held_vote():
    """Check that an admission stops the host exchange once a vote had a rank hold for it."""
    r = _rank()
    r.client.deliver(_record(MembershipState.COMMITTED))  # Rank 2 failed earlier.
    wake = r.executor.executor_request_queue.wake
    # A peer with no batch keeps every rank from the forward.
    assert _vote(r, peers=((0, 1, 0), (5, 0, 0), (2, 0, 0))) == ([4, 0, 0], (False, True))
    wake.assert_not_called()
    assert not r.exchange.interrupted()  # It has not seen the change yet.
    r.client.deliver(_joining_record(MembershipState.PROPOSE))
    assert r.exchange.interrupted()
    wake.reset_mock()
    _vote(r)
    wake.assert_called_once()
    r.client.deliver(_record(MembershipState.COMMITTED, generation=3))
    _vote(r)
    r.client.deliver(_record(MembershipState.PROPOSE, generation=4, active_mask=0b0011))
    assert r.exchange.interrupted()


def test_phase_transition_outside_the_table_raises():
    """Check that a rank refuses a phase transition that the transition table does not allow."""
    r = _rank()
    with pytest.raises(RuntimeError, match="from membership phase SERVING to RESUMING"):
        r.recovery._set_phase(recovery_module._Phase.RESUMING, "a test")


def test_change_returns_after_exchange_install():
    r = _rank(rank=1)
    _publish_rows(r.store, _record(MembershipState.PROPOSE), rank=1)
    r.client.deliver(_record(MembershipState.PROPOSE))
    done = threading.Thread(target=r.exchange.on_interrupt, daemon=True)
    done.start()
    _wait_until(lambda: r.client.reports)
    assert r.client.reports[-1] == {
        "reply": Reply.READY,
        "reply_generation": 1,
        "paused_at": ResumeState(iteration=5, epoch=6),
    }
    r.client.deliver(_record(MembershipState.INSTALL, resume=ResumeState(iteration=9, epoch=8)))
    done.join(TIMEOUT_S)
    assert not done.is_alive()
    assert r.client.reports[-2:] == [{"joining_ready": False}, {"install_steps": 1}]
    assert r.exchange.installs == [([b"rows-0", b"row-mine", None, b"rows-3"], SURVIVORS, 8)]
    # INSTALLED follows only past the survivors' resume iteration.
    assert r.recovery.pauses_forward()
    r.executor.iter_counter = 9
    r.recovery.before_iteration()
    assert _replies(r) == [Reply.READY]
    # The installed change no longer stops the host exchange.
    assert not r.exchange.interrupted()
    r.client.deliver(_record(MembershipState.PROPOSE, generation=2, active_mask=0b0011))
    assert r.exchange.interrupted()


@pytest.mark.parametrize(
    "record, status",
    [
        (_record(MembershipState.ABORTED), 1),
        (_record(MembershipState.PROPOSE, active_mask=0b0101), 1),  # Rank 1 is excluded.
        (_record(MembershipState.SHUTDOWN), 0),
    ],
)
def test_abort_exclusion_or_shutdown_ends_process(monkeypatch, record, status):
    _exit_raises(monkeypatch)
    r = _rank(rank=1)
    r.client.deliver(record)
    with pytest.raises(SystemExit) as exited:
        r.exchange.on_interrupt()
    assert exited.value.code == status


def _loop_cleanup(recovery, fatal_error=None, completed=True):
    """Run the executor's cleanup after its loop ended and return the executor."""
    executor = SimpleNamespace(
        response_cv=threading.Condition(),
        shutdown_event=threading.Event(),
        rank_failure_recovery=recovery,
        num_micro_batches=0,
        _event_loop_completed=completed,
        _fatal_error=fatal_error,
        _executor_events=Mock(),
        _flush_staged_responses=Mock(),
        _intake_window=threading.Event(),
    )
    PyExecutor._executor_loop_cleanup(executor)
    return executor


@pytest.mark.parametrize("rank, generation", [(0, 0), (1, 0), (1, 1)])
def test_completed_stop_returns_to_cleanup(monkeypatch, rank, generation):
    """Check that a completed run leaves the membership, and rank 0 asks for SHUTDOWN."""
    _exit_raises(monkeypatch)
    hooks = []
    monkeypatch.setattr(recovery_module.atexit, "register", lambda *hook: hooks.append(hook))
    r = _rank(rank=rank)
    r.client.deliver(_record(MembershipState.SHUTDOWN, generation=generation))
    _loop_cleanup(r.recovery)._executor_events.at_shutdown.assert_called_once()
    assert ({"shutdown": True} in r.client.reports) == (rank == 0)
    # After failures the process ends before MPI_Finalize, which waits for every rank.
    assert len(hooks) == generation
    if hooks:
        ((end_process, *args),) = hooks
        with pytest.raises(SystemExit) as exited:
            end_process(*args)
        assert exited.value.code == 0


def test_stop_without_shutdown_record_logs_it(monkeypatch):
    """Check that a completed run logs a missing SHUTDOWN record and still leaves the membership."""
    monkeypatch.setattr(recovery_module, "_AUTHORITY_TIMEOUT_S", 0.01)
    error = Mock()
    monkeypatch.setattr(recovery_module.logger, "error", error)
    r = _rank(rank=1)
    r.client.stop = Mock()
    r.recovery.stop(True)
    r.client.stop.assert_called_once()
    ((message,), _) = error.call_args
    assert "received no SHUTDOWN record" in message
    assert message.endswith("phase SERVING")


def test_completed_stop_ends_on_aborted_deployment(monkeypatch):
    """Check that a completed run exits with status 1 when the deployment aborts before SHUTDOWN."""
    monkeypatch.setattr(recovery_module, "_AUTHORITY_TIMEOUT_S", 0.01)
    _exit_raises(monkeypatch)
    r = _rank(rank=1)
    r.client.deliver(_record(MembershipState.ABORTED))
    with pytest.raises(SystemExit) as exited:
        r.recovery.stop(True)
    assert exited.value.code == 1


@pytest.mark.parametrize("fatal_error", [None, RuntimeError("Fatal error: a test error")])
def test_failed_run_stop_exits_without_shutdown(monkeypatch, fatal_error):
    """Check that a failed loop, or a completed one after a fatal error, exits with no SHUTDOWN."""
    _exit_raises(monkeypatch)
    r = _rank(rank=0)
    with pytest.raises(SystemExit) as exited:
        _loop_cleanup(r.recovery, fatal_error, completed=fatal_error is not None)
    assert exited.value.code == 1
    assert {"shutdown": True} not in r.client.reports


@pytest.mark.parametrize("rank, shutdown", [(0, False), (1, False), (1, True)])
def test_authority_loss_ends_the_process(monkeypatch, rank, shutdown):
    """Check that losing the authority ends the process unless the deployment shuts down."""
    _exit_raises(monkeypatch)
    r = _rank(rank=rank)
    if shutdown:
        r.client.deliver(_record(MembershipState.SHUTDOWN))
    with pytest.raises(SystemExit) if not shutdown else contextlib.nullcontext() as exited:
        r.client.fail(ConnectionError("the membership authority closed the connection"))
    assert shutdown or exited.value.code == 1


@pytest.mark.parametrize("paused_before", [False, True])
def test_hang_detector_paused_during_change(paused_before):
    """Check that the hang detector is paused while the rank takes part in a membership change."""
    detector = HangDetector(timeout=600)
    detector.start()
    try:
        if not paused_before:
            detector.checkpoint()
        r = _rank(rank=1)
        r.executor.hang_detector = detector
        r.executor.dist.allgather.side_effect = lambda local: [local] * WORLD
        seen = []
        install = r.exchange.install
        r.exchange.install = lambda *args: (seen.append(detector.paused()), install(*args))
        retire = r.executor.retire_inflight_batch
        retire.side_effect = lambda: seen.append(detector.paused())
        install_record = _record(MembershipState.INSTALL, resume=ResumeState(iteration=9))
        committed = install_record.model_copy(update=dict(state=MembershipState.COMMITTED))
        r.client.answers[(Reply.INSTALLED, 1)] = [committed]
        _install(r, install_record)
        assert detector.paused() == paused_before
        r.executor.iter_counter = 10
        r.recovery.before_iteration()
        assert detector.paused() == paused_before
        assert seen == [True, True]
    finally:
        detector.stop()


def test_start_shares_pause_word(pause_words):
    """Share the rank's pause word, which holds the PID while a pause runs, with the hang detector.

    The pausing call writes it, so the node supervisor sees the pause before the paused call runs.
    """
    detector = HangDetector(timeout=600)
    detector.start()
    try:
        detector.checkpoint()
        r = _unstarted_rank(1)
        r.executor.hang_detector = detector
        r.recovery = RankFailureRecovery(r.executor, r.exchange, r.moe, r.store, STARTUP_MEMORY)
        with patch.object(recovery_module, "membership_client", return_value=r.client):
            r.recovery.start()
        word = pause_words["rank 1"]
        assert word[0] == 0
        with detector.pause():
            assert word[0] == os.getpid()
        assert word[0] == 0
        assert r.client.reports == []
    finally:
        detector.stop()


# Dropping steps and continuing requests.


def _request(request_id, dummy=False):
    return SimpleNamespace(
        py_request_id=request_id, is_dummy=dummy, is_finished=False, py_seq_slot=request_id
    )


def _batch(requests, outputs_marks=()):
    return SimpleNamespace(
        scheduled_requests=SimpleNamespace(all_requests=lambda: list(requests)),
        sample_state=SimpleNamespace(sampler_event=MagicMock()),
        gpu_forward_events_from_perf_pool=False,
        guided_batch_id=None,
        outputs_marks=list(outputs_marks),
        context_commits=[],
    )


def _recomputed(executor):
    """Return the ids of the requests that the executor paused for recompute, in order."""
    return [
        request.py_request_id
        for call in executor.recompute_requests.call_args_list
        for request in call.args[0]
    ]


@pytest.mark.parametrize("aborted", [False, True])
def test_forward_failed_by_abort_is_discarded(aborted):
    """Check that a forward that failed for the abort discards the step before it too."""
    r = _rank()
    r.client.deliver(_record(MembershipState.PROPOSE))
    r.moe.aborted.return_value = aborted
    r.executor.previous_batch = _batch([_request(10)])
    scheduled = SimpleNamespace(all_requests=lambda: [_request(11)])
    assert r.recovery.discard_failed_forward(scheduled) is aborted
    # The executor's forward error path does this after a discard.
    r.executor._accept_or_discard_previous_batch()
    assert (r.executor.previous_batch is None) == aborted


@pytest.mark.parametrize("failed_forward", [False, True])
def test_abort_without_change_ends_serving(monkeypatch, failed_forward):
    """Check that an all-to-all aborted with no membership change within the bound raises."""
    bound_s = 0.1
    monkeypatch.setattr(recovery_module, "AUTHORITY_SILENCE_TIMEOUT_S", bound_s)
    r = _rank()
    r.moe.aborted.return_value = True
    started_s = time.monotonic()
    with pytest.raises(RuntimeError, match="no membership change arrived"):
        if failed_forward:
            r.recovery.discard_failed_forward(SimpleNamespace(all_requests=lambda: []))
        else:
            r.recovery.accept_step(_batch([_request(10)]))
    assert time.monotonic() - started_s >= bound_s


def test_peer_abort_waits_for_change_record():
    """Check that a step a peer's abort cut short is discarded once the change's record arrives."""
    r = _rank()
    r.moe.aborted.return_value = True
    delivery = threading.Timer(0.05, r.client.deliver, [_record(MembershipState.PROPOSE)])
    delivery.start()
    try:
        assert not r.recovery.accept_step(_batch([_request(10)]))
    finally:
        delivery.join()


def test_abort_before_shutdown_discards_step():
    """Check that a step aborted for a change that SHUTDOWN replaced is discarded quietly."""
    r = _rank()
    r.client.deliver(_record(MembershipState.PROPOSE))
    r.moe.aborted.return_value = True
    r.client.deliver(_record(MembershipState.SHUTDOWN, generation=2))
    assert not r.recovery.accept_step(_batch([_request(10)]))


@pytest.mark.parametrize(
    "chunked, chunk_size",
    [(False, 8), (True, 1), (True, 8)],
    ids=["unchunked", "chunk per step", "pending fragments"],
)
def test_discarded_steps_drop_generation_outputs(chunked, chunk_size):
    """Check that the two discarded steps in flight leave only the accepted steps' outputs."""
    r = _rank()
    result = PyResult(
        prompt_len=4,
        max_new_tokens=8,
        use_device_memory=False,
        return_generation_logits=True,
        use_chunked_generation_logits=chunked,
        chunk_size=chunk_size,
        additional_outputs=["hidden"],
    )
    request = SimpleNamespace(
        **vars(_request(10)),
        py_result=result,
        py_return_context_logits=False,
        py_return_generation_logits=True,
        py_additional_outputs=["hidden"],
    )

    def mark():
        """Mark a step before its forward."""
        return r.recovery.mark_outputs(SimpleNamespace(all_requests=lambda: [request]))

    def sample():
        """Store a step's outputs as its sampling does."""
        result.append_generation_logits(torch.zeros(1, 1, 16))
        result.append_additional_generation_outputs("hidden", torch.zeros(1, 1, 4))

    first_marks = mark()
    sample()
    assert r.recovery.accept_step(_batch([request], first_marks))
    # The overlap scheduler marks the next step before it accepts the step in flight.
    discarded_marks = mark()
    sample()
    next_marks = mark()
    r.client.deliver(_record(MembershipState.PROPOSE))
    r.moe.aborted.return_value = True
    assert not r.recovery.accept_step(_batch([request], discarded_marks))
    sample()
    assert not r.recovery.accept_step(_batch([request], next_marks))
    assert result.additional_generation_outputs["hidden"].shape[0] == 1

    r.moe.aborted.return_value = False
    accepted_marks = mark()
    sample()
    assert r.recovery.accept_step(_batch([request], accepted_marks))
    result.transfer_remaining_device_logits()
    assert result.generation_logits.shape[1] == 2
    assert result.additional_generation_outputs["hidden"].shape[0] == 2


# Rank 0 restores lost expert 3 into its spare slot. Failed rank 2's slots are left out.
_PLACEMENT = ((4, ((0, -1), (1, -1), (3, -1), (2, -1))),)
_INSTALL = _record(MembershipState.INSTALL, resume=ResumeState(iteration=9, epoch=8))
_REJOIN_INSTALL = _INSTALL.model_copy(update=dict(active_mask=ALL_RANKS, joining_mask=1 << FAILED))


def _own_slots(placement, rank):
    """Return the slots of ``rank`` in each MoE layer of ``placement``."""
    return [list(ranks[rank]) for _, ranks in placement]


def _resuming(known_by_peers=None, record=_INSTALL):
    """Return rank 0 past the resume of ``record``, whose peers gather ``known_by_peers``."""
    known_by_peers = known_by_peers or {}
    r = _rank()
    peers = {
        peer: (known_by_peers.get(peer, set()), _own_slots(_PLACEMENT, peer))
        for peer in range(1, WORLD)
        if record.is_active(peer)
    }
    r.executor.dist.allgather.side_effect = lambda local: [local] + [
        peers.get(peer) for peer in range(1, WORLD)
    ]
    _install(r, record)
    r.moe.aborted.return_value = True
    r.executor.iter_counter = record.resume.iteration + 1
    committed = record.model_copy(update=dict(state=MembershipState.COMMITTED, joining_mask=0))
    r.client.answers[(Reply.INSTALLED, record.generation)] = [committed]
    return r


def test_resume_recomputes_interrupted_fails_lost(monkeypatch):
    r = _resuming({1: {20}})
    executor = r.executor
    in_dropped_step, in_pending_step, untouched = _request(10), _request(14), _request(15)
    executor.active_requests = [in_dropped_step, in_pending_step, untouched, _request(-1, True)]
    assert r.recovery.discard_failed_forward(
        SimpleNamespace(all_requests=lambda: [in_dropped_step])
    )
    # Its forward ran before the change committed.
    executor.previous_batch = _batch([in_pending_step])
    executor.waiting_queue = [SimpleNamespace(id=12)]
    executor.executor_request_queue.request_queue.put(RequestQueueItem(13))
    # A cancel of lost request 11 does not hold it.
    executor.executor_request_queue.request_queue.put(
        RequestQueueItem(11, is_canceled_request=True)
    )
    # The worker answers request 16 during the install.
    outstanding = iter([(10, 11, 12, 13, 16, 20), (10, 11, 12, 13, 20)])
    r.recovery.set_outstanding_requests(lambda: {r: 100 + r for r in next(outstanding)})

    executor.iter_counter = 9
    r.recovery.before_iteration()  # Not yet past the last iteration a survivor reached.
    r.moe.rearm.assert_not_called()
    assert r.recovery.pauses_forward()

    executor.iter_counter = 10
    executor.adp_ctx_waiting_iters_count = executor.adp_ctx_batching_wait_iters_count = 3
    info = Mock()
    monkeypatch.setattr(recovery_module.logger, "info", info)
    r.recovery.before_iteration()
    logged = [c.args[0] for c in info.call_args_list]
    (resumed,) = [message for message in logged if "resumed serving" in message]
    assert resumed.endswith("2 interrupted requests continue by recompute")
    assert "Membership generation 1 ended 1 requests that no active rank holds" in logged
    assert executor.previous_batch is None
    # A replacement starts its attention-DP balance counters at 0, so every active rank does too.
    assert executor.adp_ctx_waiting_iters_count == 0
    assert executor.adp_ctx_batching_wait_iters_count == 0
    # Only the requests of the discarded steps run again. The rest of the prefix cache stays.
    assert _recomputed(executor) == [10, 14]
    r.moe.ep_group_health.mark_failed.assert_called_once_with(FAILED)
    # The workspaces move to fresh pages among the survivors before the all-to-all is rearmed.
    steps = ("remap_workspaces", "rearm", "install_mask")
    assert [name for name, _, _ in r.moe.method_calls if name in steps] == list(steps)
    assert _replies(r) == [Reply.INSTALLED]
    # A change that admits no rank runs no waiting-queue handover.
    assert r.exchange.exchanges == []
    assert not r.recovery.pauses_forward()
    # No survivor holds request 11.
    ((lost, error), _) = executor.fail_requests_by_id.call_args
    assert lost == {11: 111}
    assert isinstance(error, RankFailureError)


def test_rearm_synchronizes_before_record_lock(monkeypatch):
    """Check that the rank waits for the device before it takes the membership record's lock."""
    r = _resuming()
    lock = r.recovery._record_changed
    steps = []

    def lock_free():
        taken = []

        def take():
            if lock.acquire(blocking=False):
                lock.release()
                taken.append(True)

        thread = threading.Thread(target=take)
        thread.start()
        thread.join()
        return bool(taken)

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: steps.append(("sync", lock_free())))
    r.moe.remap_workspaces.side_effect = lambda *args: steps.append("remap")
    r.moe.rearm.side_effect = lambda: steps.append("rearm")
    r.recovery.before_iteration()
    remapped = steps.index("remap")
    assert steps[remapped + 1 : remapped + 3] == [("sync", True), "rearm"]


@pytest.mark.parametrize("failure", ["own mapping", "lost expert", "other plan"])
def test_failed_install_holds_the_steps(monkeypatch, failure):
    """Check that a rank replies FAILED to an install error that no rank removal fixes."""
    r = _resuming()
    # The authority answers FAILED with the next change.
    next_change = _record(MembershipState.PROPOSE, generation=2, active_mask=0b0011)
    r.client.answers[(Reply.FAILED, 1)] = [next_change]
    if failure == "own mapping":
        r.moe.remap_workspaces.side_effect = MnnvlMappingError([0], "cuMemCreate failed")
    elif failure == "lost expert":
        _eplb(monkeypatch, r)
        r.executor.model_engine.moe_load_balancer.expert_restore.plan_expert_restore = Mock(
            side_effect=ValueError("No node with an active rank keeps host weights for expert 3")
        )
    else:
        _eplb(monkeypatch, r)
        r.executor.model_engine.moe_load_balancer.expert_restore.restore_lost_experts = Mock(
            side_effect=ExpertRestorePlanMismatchError(
                "EP rank 2 planned expert placement 5f1d0c8a9e2b4d67 with workspace rows of "
                "1048576 bytes, but the lowest active rank planned 0a3c9e7b1f2d4865 with rows of "
                "1048576 bytes"
            )
        )
    r.recovery.before_iteration()
    assert _replies(r) == [Reply.FAILED]
    r.moe.rearm.assert_not_called()
    assert r.recovery.pauses_forward()


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("CUDA error: an illegal memory access was encountered"),
        OutOfPagesError("The GPU KV cache pool cannot be resized to 0 bytes"),
    ],
)
def test_unexpected_install_error_ends_the_process(error):
    """Check that an unexpected install error propagates instead of a FAILED reply."""
    r = _resuming()
    next_change = _record(MembershipState.PROPOSE, generation=2, active_mask=0b0011)
    r.client.answers[(Reply.FAILED, 1)] = [next_change]
    r.moe.remap_workspaces.side_effect = error
    with pytest.raises(type(error)):
        r.recovery.before_iteration()
    assert _replies(r) == []


def test_unimportable_peer_is_waited_on():
    """Check that a rank that cannot import a peer's MNNVL handles waits on it for a new record."""
    r = _resuming()
    r.moe.remap_workspaces.side_effect = MnnvlPeerHandleError([3], "import failed")

    def waiting_for():
        r.recovery._on_tick()
        return [report["waiting_for"] for report in r.client.reports if "waiting_for" in report][-1]

    resuming = threading.Thread(target=r.recovery.before_iteration, daemon=True)
    resuming.start()
    _wait_until(lambda: waiting_for() == 1 << 3)
    assert resuming.is_alive()
    r.client.deliver(_record(MembershipState.PROPOSE, generation=2, active_mask=0b0011))
    resuming.join(TIMEOUT_S)
    assert not resuming.is_alive()
    assert waiting_for() == 0
    assert _replies(r) == []
    r.moe.rearm.assert_not_called()
    assert r.recovery.pauses_forward()


def test_newer_change_installed_before_resume_iteration():
    """Check that a newer change installed before the resume iteration replaces the older one."""
    r = _rank(rank=1)
    r.executor.dist.allgather.side_effect = lambda local: [(set(), None), local, None, None]
    _install(r, _record(MembershipState.INSTALL, resume=ResumeState(iteration=9, epoch=8)))
    newer = _record(MembershipState.INSTALL, 2, 0b0011, ResumeState(iteration=12, epoch=10))
    _install(r, newer)
    assert r.recovery._phase is recovery_module._Phase.INSTALLING
    assert r.exchange.installs[-1] == ([b"rows-0", b"row-mine", None, None], 0b0011, 10)
    committed = newer.model_copy(update=dict(state=MembershipState.COMMITTED))
    r.client.answers[(Reply.INSTALLED, 2)] = [committed]
    r.executor.iter_counter = 12  # Past the replaced change's resume iteration only.
    r.recovery.before_iteration()
    assert _replies(r) == []
    r.executor.iter_counter = 13
    r.recovery.before_iteration()
    (installed,) = [report for report in r.client.reports if "reply" in report]
    assert (installed["reply"], installed["reply_generation"]) == (Reply.INSTALLED, 2)
    assert not r.recovery.pauses_forward()


def test_resume_deletes_change_group_keys():
    """Delete this rank's store keys of the change's groups once serving resumes.

    Only the values of the simulated active peers 1 and 3 remain, as each member deletes its own.
    """
    peer_values = {"host_exchange_rows/1/0/1", "host_exchange_rows/1/0/3"}
    r = _resuming()

    def keys():
        return {key for key in r.store.values if key.startswith("host_exchange_rows/")}

    assert keys() > peer_values
    r.recovery.before_iteration()
    assert keys() == peer_values


@pytest.mark.parametrize("together", [False, True])
def test_commit_with_next_change_resumes(together):
    """Check that a rank resumes at the commit and holds at the next vote with the next change."""
    r = _resuming()
    committed = _INSTALL.model_copy(update=dict(state=MembershipState.COMMITTED))
    admission = _joining_record(MembershipState.PROPOSE)
    r.client.answers[(Reply.INSTALLED, 1)] = [committed, admission] if together else [committed]
    r.recovery.before_iteration()
    if not together:
        r.client.deliver(admission)
    assert _replies(r) == [Reply.INSTALLED]
    assert not r.exchange.interrupted()
    assert _vote(r)[0] == [0, 1, 0]
    assert r.exchange.interrupted()


# Device memory: the restore of lost experts, the KV cache pool and the release of its memory.


class _Device:
    """Device memory of a rank, which also acts as its KV cache manager."""

    def __init__(self, extra_free=0, gpu_quota=STARTUP_MEMORY.kv_cache_quota, backed=0, unit=MIB):
        self.calls = []
        self.free = FREE_BYTES + extra_free
        self.cached = 0
        self.gpu_quota = gpu_quota
        self.backed = backed
        self.unit = unit
        self.used = 0
        self.floor = 8 * MIB
        self.growable = True

    def measure(self, monkeypatch):
        """Let the recovery read the free memory and empty the caching allocator."""
        monkeypatch.setattr(memory_module.torch.cuda, "mem_get_info", lambda: (self.free, 1 << 40))
        monkeypatch.setattr(memory_module.torch.cuda, "memory_reserved", lambda: self.cached)
        monkeypatch.setattr(
            memory_module.torch.cuda, "empty_cache", lambda: self.cache(-self.cached)
        )

    def take(self, num_bytes):
        self.free -= num_bytes

    def cache(self, num_bytes):
        """Keep ``num_bytes`` more of freed blocks in the caching allocator, as serving does."""
        self.take(num_bytes)
        self.cached += num_bytes

    def resize_gpu_quota(self, quota):
        self.calls.append(("resize", quota))
        quota = max(0, quota) // self.unit * self.unit
        grows = quota > self.gpu_quota
        if (
            quota < max(self.floor, self.used)
            or quota - self.gpu_quota > self.free
            or (grows and not self.growable)
        ):
            raise OutOfPagesError(f"The GPU KV cache pool cannot be resized to {quota} bytes")
        self.take(quota - self.gpu_quota)
        self.gpu_quota = quota
        return quota

    @contextlib.contextmanager
    def suspended(self):
        """Stand for the executor's KV cache pools while every request is suspended."""
        self.calls.append("suspend")
        try:
            yield self
        finally:
            self.calls.append("resume")


def _resizes(device):
    return [call[1] for call in device.calls if call[0] == "resize"]


def _eplb(monkeypatch, r, restored=6 * MIB, workspace=4 * MIB, device=None, released=0):
    """Set up the EPLB of rank 0, which restores ``restored`` and releases ``released`` bytes."""
    device = _Device() if device is None else device
    device.measure(monkeypatch)
    calls = device.calls
    balancer = Mock(spec=MoeLoadBalancer)
    balancer.ep_rank = 0
    balancer.abandon_iterations = lambda: calls.append("abandon")
    # Rank 0 takes lost expert 3 into its spare slot.
    planned = (np.array([[0, 3], [1, -1], [-1, -1], [2, -1]]),)
    plan = SimpleNamespace(device_bytes=restored, placements=planned)

    def back(num_bytes):
        device.take(num_bytes)
        device.backed += num_bytes
        balancer.expert_restore.backed_spare_slot_bytes += num_bytes

    def restore_lost_experts(restore, staging_workspace, allgather, times, progress):
        calls.append(("restore", restore))
        assert staging_workspace is r.moe.staging_workspace.return_value
        back(restored)
        progress()
        times.update({"back": 1.0, "load": 2.0, "install": 3.0})
        return restored

    def release_spare_slots(layer_slots):
        """Unmap the spare slots on the release worker, which leaves the executor's count alone."""
        assert layer_slots == [[1]]
        calls.append("release")
        freed = min(released, device.backed)  # A slot already unmapped frees nothing.
        device.take(-freed)
        device.backed -= freed
        return freed

    def plan_expert_restore(active, current, rejoining):
        calls.append(("plan", active, current, rejoining))
        return plan

    balancer.expert_restore = SimpleNamespace(
        current_placement=lambda: _PLACEMENT,
        plan_expert_restore=plan_expert_restore,
        restore_lost_experts=restore_lost_experts,
        # A setup that releases memory stands for a rejoin, whose placement empties the slot.
        empty_spare_slots=lambda: [[1] if released else []],
        release_spare_slots=release_spare_slots,
        backed_spare_slot_bytes=device.backed,
    )
    r.executor.model_engine.moe_load_balancer = balancer
    r.executor.kv_cache_manager = device
    r.executor.kv_pools_suspended = device.suspended
    r.moe.workspace_bytes.return_value = workspace
    r.moe.remap_workspaces.side_effect = lambda gather, fresh_rows: calls.append("remap")
    return calls, plan, device


@pytest.mark.parametrize("extra_free", [0, 7 * MIB, 20 * MIB, -3 * MIB])
def test_install_restores_lost_experts(monkeypatch, extra_free):
    r = _resuming()
    calls, plan, device = _eplb(
        monkeypatch, r, workspace=4 * MIB - 4096, device=_Device(extra_free=extra_free)
    )
    r.recovery.before_iteration()
    names = [call if isinstance(call, str) else call[0] for call in calls]
    # Every active rank abandons its EPLB iterations at the resume point, before it plans.
    assert names[:2] == ["abandon", "plan"]
    (_, active, _, rejoining) = calls[1]
    assert active == [True, True, False, True]
    assert rejoining == [False] * WORLD
    assert ("restore", plan) in calls
    order = [names.index(name) for name in ("suspend", "resize", "remap", "restore", "resume")]
    assert order == sorted(order)
    # The pool also gives up the workspace size for the fresh workspace pages.
    assert _resizes(device) == [54 * MIB + 4096]
    # The authority sees every completed install step, the restore's own progress included, so a
    # long install is not taken for a hang.
    steps = [
        report.get("install_steps", report.get("reply"))
        for report in r.client.reports
        if {"install_steps", "reply"} & report.keys()
    ]
    assert steps == [1, 2, 3, 4, 5, 6, 7, Reply.INSTALLED]
    _vote(r)
    r.recovery.before_iteration()
    # Once serving resumed, the pool takes back the workspace memory once per change.
    assert _resizes(device) == [54 * MIB + 4096, 58 * MIB]
    # The restored expert fills the only spare slot, so no spare slot is released.
    assert "release" not in calls
    assert device.backed == 6 * MIB
    # Serving keeps the memory it had free.
    assert device.free == FREE_BYTES + extra_free


def test_newer_change_stops_expert_restore(monkeypatch):
    """Check that a restore stops loading experts once a newer change arrives."""
    r = _resuming()
    calls, _, _ = _eplb(monkeypatch, r)
    expert_restore = r.executor.model_engine.moe_load_balancer.expert_restore

    def restore_lost_experts(plan, staging_workspace, allgather, times, progress):
        r.client.deliver(_record(MembershipState.PROPOSE, generation=2, active_mask=0b0011))
        progress()
        calls.append("load after the newer change")

    expert_restore.restore_lost_experts = restore_lost_experts
    r.recovery.before_iteration()
    assert "load after the newer change" not in calls
    r.moe.rearm.assert_not_called()


def test_full_kv_pool_preempts_newest_requests(monkeypatch):
    """Check that a restore into a full KV cache pool preempts the newest requests until it fits."""
    r = _resuming()
    calls, plan, device = _eplb(monkeypatch, r)
    executor = r.executor
    executor.active_requests = [_request(i) for i in (20, 21, 22, 23)]
    held = {20: 46 * MIB, 21: 6 * MIB, 22: 6 * MIB, 23: 6 * MIB}
    device.used = sum(held.values())

    def recompute(requests):  # Each request gives its pages and its sequence slot back.
        for request in requests:
            device.used -= held[request.py_request_id]
            request.py_seq_slot = None

    executor.recompute_requests.side_effect = recompute
    r.recovery.before_iteration()
    assert _replies(r) == [Reply.INSTALLED]
    assert _recomputed(executor) == [23, 22]
    assert _resizes(device) == [54 * MIB] * 3
    assert calls[-4:] == [("resize", 54 * MIB), "remap", ("restore", plan), "resume"]


def test_kv_pool_without_recomputable_requests_replies_failed(monkeypatch):
    """Check that a KV cache pool that no preemption can shrink fails the install with a reply.

    The pool keeps at least 60 MiB, above the 64 MiB startup quota less the 10 MiB restore, and
    holds only a dummy request, which is not recomputed.
    """
    r = _resuming()
    r.client.answers[(Reply.FAILED, 1)] = [
        _record(MembershipState.PROPOSE, generation=2, active_mask=0b0011)
    ]
    calls, _, device = _eplb(monkeypatch, r)
    device.floor = 60 * MIB
    executor = r.executor
    executor.active_requests = [_request(-1, dummy=True)]
    r.recovery.before_iteration()
    assert _replies(r) == [Reply.FAILED]
    assert _recomputed(executor) == []
    assert device.gpu_quota == STARTUP_MEMORY.kv_cache_quota
    assert _resizes(device) == [54 * MIB]
    assert calls.count("suspend") == calls.count("resume") == 1
    assert "remap" not in calls
    r.moe.rearm.assert_not_called()


@pytest.mark.parametrize("backed, fits", [(46 * MIB, True), (56 * MIB, False)])
def test_restore_beyond_kv_pool_refused_before_preempting(monkeypatch, backed, fits):
    """Check that a restore that the KV cache pool cannot release replies FAILED at once.

    The spare slots of earlier failures stay backed until a rejoin, so a second failure's restore
    of 10 MiB needs more than the 64 MiB startup quota less 56 MiB can give.
    """
    r = _resuming()
    r.client.answers[(Reply.FAILED, 1)] = [
        _record(MembershipState.PROPOSE, generation=2, active_mask=0b0011)
    ]
    calls, plan, device = _eplb(monkeypatch, r, device=_Device(backed=backed))
    executor = r.executor
    executor.active_requests = [_request(i) for i in (20, 21)]

    def recompute(requests):
        for request in requests:
            request.py_seq_slot = None

    executor.recompute_requests.side_effect = recompute
    r.recovery.before_iteration()
    assert _replies(r) == [Reply.INSTALLED if fits else Reply.FAILED]
    assert _recomputed(executor) == []
    assert _resizes(device) == ([8 * MIB] if fits else [])
    assert (("restore", plan) in calls) is fits


@pytest.mark.parametrize(
    "allocate, quota", [("cache", 58 * MIB), ("take", 54 * MIB), ("no growth", 54 * MIB)]
)
def test_release_never_charges_serving_memory(monkeypatch, allocate, quota):
    """Check that the release into a full pool neither shrinks it nor charges serving's memory."""
    r = _resuming()
    _, _, device = _eplb(monkeypatch, r)
    r.recovery.before_iteration()  # The install and the commit.
    assert device.gpu_quota == 54 * MIB
    if allocate == "no growth":
        device.growable = False
    else:
        device.used = 54 * MIB
        getattr(device, allocate)(10 * MIB)
    _vote(r)
    r.recovery.before_iteration()  # The release.
    assert device.gpu_quota == quota
    assert not r.recovery.pauses_forward()


def _short_release(monkeypatch, fresh_rows=frozenset(range(WORLD))):
    """Return rank 0 after a release that the remap of ``fresh_rows`` left 4 MiB short."""
    clock = SimpleNamespace(now_s=0.0)
    monkeypatch.setattr(
        memory_module,
        "time",
        SimpleNamespace(monotonic=lambda: clock.now_s, perf_counter=time.perf_counter),
    )
    r = _resuming()
    _, _, device = _eplb(monkeypatch, r)
    r.moe.workspace_rows_needing_fresh_pages.return_value = fresh_rows
    r.moe.remap_workspaces.side_effect = lambda gather, fresh_rows: device.take(4 * MIB)
    r.recovery.before_iteration()  # The install and the commit.
    _vote(r)
    r.recovery.before_iteration()  # The release.
    r.executor.retire_inflight_batch.reset_mock()
    return r, device, clock


def _retry_vote(r, device, clock, freed=4 * MIB):
    """Free ``freed`` bytes, let a second pass and return the release flag of the next vote."""
    device.take(-freed)
    clock.now_s += 1.0
    return _vote(r)[0][2]


def test_retried_release_regains_freed_memory(monkeypatch):
    """Check that the retry retires the step in flight and takes back what the ended rank freed."""
    r, device, clock = _short_release(monkeypatch)
    retire = r.executor.retire_inflight_batch
    assert device.gpu_quota == 54 * MIB
    device.take(-4 * MIB)  # The excluded process ended.
    clock.now_s += 1.0
    assert _vote(r)[0] == [4, 0, 1]
    r.recovery.before_iteration()
    retire.assert_called_once()
    assert device.gpu_quota == 58 * MIB
    assert device.free == FREE_BYTES
    # The pool has its quota again, so no more retries.
    assert _retry_vote(r, device, clock) == 0
    r.recovery.before_iteration()
    retire.assert_called_once()


@pytest.mark.parametrize(
    "taken_outside, quota",
    [
        (0, 58 * MIB),
        (MIB, 57 * MIB),  # 1 MiB went outside PyTorch meanwhile.
    ],
)
def test_retry_waits_for_owed_row(monkeypatch, taken_outside, quota):
    """Check that the retry waits for the owed row from the device memory alone, not the store."""
    r, device, clock = _short_release(monkeypatch)
    for read in ("get", "check", "multi_get"):
        monkeypatch.setattr(r.store, read, Mock(side_effect=AssertionError("read the store")))
    device.take(taken_outside)
    assert _retry_vote(r, device, clock, freed=0) == 0
    assert _retry_vote(r, device, clock, freed=MIB) == 0
    assert _retry_vote(r, device, clock, freed=3 * MIB) == 1
    r.recovery.before_iteration()
    assert device.gpu_quota == quota


def test_unowed_shortfall_is_not_retried(monkeypatch):
    """Check that a pool short of memory that serving took is not retried when no row is owed."""
    r, device, clock = _short_release(monkeypatch, fresh_rows=frozenset({FAILED}))
    assert _retry_vote(r, device, clock) == 0


def test_next_change_release_replaces_pending_retry(monkeypatch):
    """Check that the release of the next change replaces the retry of the last one."""
    r, device, clock = _short_release(monkeypatch)
    second = _record(MembershipState.INSTALL, 2, 0b1001, ResumeState(iteration=12, epoch=10))
    _install(r, second)
    r.executor.iter_counter = 13
    committed = second.model_copy(update=dict(state=MembershipState.COMMITTED))
    r.client.answers[(Reply.INSTALLED, 2)] = [committed]
    r.recovery.before_iteration()  # The install and the commit.
    _vote(r)
    r.recovery.before_iteration()  # The release.
    assert device.gpu_quota == 48 * MIB
    assert _retry_vote(r, device, clock) == 1  # Rank 2 ended.
    r.recovery.before_iteration()
    assert device.gpu_quota == 52 * MIB
    assert _retry_vote(r, device, clock) == 0  # Rank 1 ended.
    assert device.free == FREE_BYTES


def test_healthy_step_checks_no_device_memory(monkeypatch):
    """Check that a healthy vote asks for no retry, reads no device memory and empties no cache."""
    r = _rank()
    mem_get_info, empty_cache = Mock(), Mock()
    monkeypatch.setattr(memory_module.torch.cuda, "mem_get_info", mem_get_info)
    monkeypatch.setattr(memory_module.torch.cuda, "empty_cache", empty_cache)
    assert _vote(r)[0] == [4, 0, 0]
    r.recovery.before_iteration()
    mem_get_info.assert_not_called()
    empty_cache.assert_not_called()
    r.executor.retire_inflight_batch.assert_not_called()


def _rejoining(monkeypatch):
    """Return rank 0 resuming from a rejoin, which empties the backed spare slot of 6 MiB."""
    r = _resuming(record=_REJOIN_INSTALL)
    device = _Device(gpu_quota=58 * MIB, backed=6 * MIB)
    calls, _, _ = _eplb(monkeypatch, r, restored=0, released=6 * MIB, device=device)
    return r, calls, device


def test_rejoin_releases_memory_after_resume(monkeypatch):
    """Check that the emptied spare slots return to the KV cache pool after a vote with no hold."""
    r, calls, device = _rejoining(monkeypatch)
    info = Mock()
    monkeypatch.setattr(memory_module.logger, "info", info)
    r.recovery.before_iteration()
    (_, active, _, rejoining) = calls[1]
    assert active == [True] * WORLD
    assert rejoining == [False, False, True, False]
    assert _replies(r) == [Reply.INSTALLED]
    assert not r.recovery.pauses_forward()
    r.moe.ep_group_health.mark_active.assert_any_call(FAILED)
    # Serving resumed with the pool short of the workspace and of the spare slot, which a worker
    # releases meanwhile. The executor counts it as backed until the pool grows into it.
    expert_restore = r.executor.model_engine.moe_load_balancer.expert_restore
    _wait_for_release_worker(r)
    assert device.backed == 0
    assert expert_restore.backed_spare_slot_bytes == 6 * MIB
    assert device.gpu_quota == 54 * MIB
    retire = r.executor.retire_inflight_batch
    retire.reset_mock()
    # Rank 1 holds its steps for the next change, so no rank releases memory at this step boundary.
    assert _vote(r, peers=((3, 1, 0), (5, 0, 0), (2, 0, 0)))[0] == [4, 0, 1]
    r.recovery.before_iteration()
    retire.assert_not_called()
    assert device.gpu_quota == 54 * MIB
    assert _vote(r)[0] == [4, 0, 1]
    r.recovery.before_iteration()
    retire.assert_called_once()
    assert expert_restore.backed_spare_slot_bytes == 0
    assert device.gpu_quota == STARTUP_MEMORY.kv_cache_quota
    # A rank with nothing to release still retires its step in flight at the same step boundary.
    other = _rank(rank=1)
    _vote(other, peers=((4, 0, 1), (5, 0, 0), (2, 0, 0)))
    other.recovery.before_iteration()
    other.executor.retire_inflight_batch.assert_called_once()
    # One line times the release on the worker and the pool's growth on the executor apart.
    (released,) = [c.args[0] for c in info.call_args_list if "returned memory" in c.args[0]]
    assert re.fullmatch(
        "Rank 0 returned memory to the KV cache pool after membership generation 1: released "
        rf"{6 * MIB} bytes of empty spare slots in \d+ ms on a worker thread, grew the KV cache "
        rf"quota by {10 * MIB} to {STARTUP_MEMORY.kv_cache_quota} bytes in 1 steps and \d+ ms, "
        "0 bytes short until device memory is freed",
        released,
    )


def test_release_waits_for_newer_change(monkeypatch):
    """Check that a change after the release vote keeps the pool from growing until it commits."""
    r, _, device = _rejoining(monkeypatch)
    r.recovery.before_iteration()
    assert _vote(r)[0] == [4, 0, 1]
    newer = _record(MembershipState.PROPOSE, generation=2, active_mask=0b0011)
    retire = r.executor.retire_inflight_batch
    retire.reset_mock()
    retire.side_effect = lambda: r.client.deliver(newer)
    r.recovery.before_iteration()
    retire.assert_called_once()
    assert device.gpu_quota == 54 * MIB
    assert r.recovery.pauses_forward()
    # The newer change's restore would wait for the release worker; the test ends before it.
    r.recovery._memory.wait_for_release()


def test_release_runs_off_the_executor_thread(monkeypatch):
    """Check that a worker releases the empty spare slots, and that the vote waits for it."""
    r, _, device = _rejoining(monkeypatch)
    expert_restore = r.executor.model_engine.moe_load_balancer.expert_restore
    release = expert_restore.release_spare_slots
    may_finish = threading.Event()
    threads = []

    def blocked_release(layer_slots):
        threads.append(threading.current_thread())
        assert may_finish.wait(TIMEOUT_S)
        return release(layer_slots)

    expert_restore.release_spare_slots = blocked_release
    r.recovery.before_iteration()  # The install and the commit return while the worker runs.
    assert _replies(r) == [Reply.INSTALLED]
    _wait_until(lambda: threads)
    assert threads[0] is not threading.current_thread()
    # The rank asks for no release while its worker runs.
    assert _vote(r, wait_for_release=False)[0] == [4, 0, 0]
    r.recovery.before_iteration()
    assert device.gpu_quota == 54 * MIB
    may_finish.set()
    assert _vote(r)[0] == [4, 0, 1]
    r.recovery.before_iteration()
    assert device.gpu_quota == STARTUP_MEMORY.kv_cache_quota
    assert expert_restore.backed_spare_slot_bytes == 0


def _commit_second_change(r):
    """Install and commit a change that rank 1 fails in, past the rejoin that ``r`` resumed from."""
    second = _record(MembershipState.INSTALL, 2, 0b1001, ResumeState(iteration=12, epoch=10))
    _install(r, second)
    r.executor.iter_counter = 13
    committed = second.model_copy(update=dict(state=MembershipState.COMMITTED))
    r.client.answers[(Reply.INSTALLED, 2)] = [committed]
    r.recovery.before_iteration()
    assert _replies(r) == [Reply.INSTALLED, Reply.INSTALLED]


def test_change_during_release_waits_for_it(monkeypatch):
    """Check that a change during the release plans its restore once the release ended."""
    r, _, device = _rejoining(monkeypatch)
    memory = r.recovery._memory
    expert_restore = r.executor.model_engine.moe_load_balancer.expert_restore
    release, plan_expert_restore = (
        expert_restore.release_spare_slots,
        expert_restore.plan_expert_restore,
    )
    wait_for_release = memory.wait_for_release
    waited = threading.Event()
    order = []

    def held_release(layer_slots):  # Ends only once the executor thread waits for it.
        assert waited.wait(TIMEOUT_S)
        freed = release(layer_slots)
        order.append("released")
        return freed

    def plan(active, current, rejoining):
        order.append(("plan", device.backed, expert_restore.backed_spare_slot_bytes))
        return plan_expert_restore(active, current, rejoining)

    def wait_then_release():
        waited.set()
        wait_for_release()

    expert_restore.release_spare_slots = held_release
    expert_restore.plan_expert_restore = plan
    r.recovery.before_iteration()  # The rejoin's install and commit start the held release.
    monkeypatch.setattr(memory, "wait_for_release", wait_then_release)
    try:
        _commit_second_change(r)
    finally:
        waited.set()
        memory.wait_for_release()  # The second commit's release, or a held one on a failure.
    # The rejoin planned with the spare slot backed, the second change once it was released.
    assert order[:3] == [("plan", 6 * MIB, 6 * MIB), "released", ("plan", 0, 0)]


def test_second_commit_before_release_step_keeps_the_released_memory(monkeypatch):
    """Check that the pool grows into spare slots that the next change's restore waited for."""
    r, _, device = _rejoining(monkeypatch)
    # Rank 0 keeps its workspace row, so neither change shrinks its KV cache pool.
    r.moe.workspace_rows_needing_fresh_pages.return_value = frozenset({FAILED})
    r.recovery.before_iteration()  # The rejoin's install and commit.
    assert device.gpu_quota == 58 * MIB
    _commit_second_change(r)
    assert _vote(r)[0] == [4, 0, 1]
    r.recovery.before_iteration()
    assert device.gpu_quota == STARTUP_MEMORY.kv_cache_quota


def test_release_error_raised_on_the_executor_thread(monkeypatch):
    """Check that an error of the release worker ends the release step of the executor thread."""
    r, _, _ = _rejoining(monkeypatch)
    expert_restore = r.executor.model_engine.moe_load_balancer.expert_restore

    def failing_release(layer_slots):
        raise RuntimeError("cuMemUnmap failed")

    expert_restore.release_spare_slots = failing_release
    r.recovery.before_iteration()
    assert _vote(r)[0] == [4, 0, 1]
    with pytest.raises(RuntimeError, match="cuMemUnmap failed"):
        r.recovery.before_iteration()


def test_pool_grows_back_by_chunks_over_release_votes(monkeypatch):
    """Check that the pool grows back by at most one chunk per release vote until it is full."""
    monkeypatch.setattr(memory_module, "_KV_CACHE_GROW_CHUNK_BYTES", 3 * MIB)
    info = Mock()
    monkeypatch.setattr(memory_module.logger, "info", info)
    r, _, device = _rejoining(monkeypatch)
    r.recovery.before_iteration()
    assert device.gpu_quota == 54 * MIB
    quotas = []
    while _vote(r)[0][2]:
        r.recovery.before_iteration()
        quotas.append(device.gpu_quota)
        assert len(quotas) <= 4
    assert quotas == [57 * MIB, 60 * MIB, 63 * MIB, STARTUP_MEMORY.kv_cache_quota]
    assert device.free == FREE_BYTES
    # The release is logged once, when the pool stops growing.
    (released,) = [c.args[0] for c in info.call_args_list if "returned memory" in c.args[0]]
    assert f"quota by {10 * MIB} to {STARTUP_MEMORY.kv_cache_quota} bytes in 4 steps" in released


def test_chunk_the_device_cannot_back_leaves_the_owed_quota_short(monkeypatch):
    """Check that a chunk that the device cannot back reports all the owed quota as short."""
    monkeypatch.setattr(memory_module, "_KV_CACHE_GROW_CHUNK_BYTES", 3 * MIB)
    info = Mock()
    monkeypatch.setattr(memory_module.logger, "info", info)
    r, _, device = _rejoining(monkeypatch)
    r.recovery.before_iteration()
    device.growable = False
    assert _vote(r)[0] == [4, 0, 1]
    r.recovery.before_iteration()
    assert device.gpu_quota == 54 * MIB
    (released,) = [c.args[0] for c in info.call_args_list if "returned memory" in c.args[0]]
    assert released.endswith(f", {10 * MIB} bytes short until device memory is freed")


def test_change_between_grow_chunks_stops_them(monkeypatch):
    """Check that a change voted between two chunks of the pool's growth stops the growth."""
    monkeypatch.setattr(memory_module, "_KV_CACHE_GROW_CHUNK_BYTES", 3 * MIB)
    r, _, device = _rejoining(monkeypatch)
    r.recovery.before_iteration()
    assert _vote(r)[0] == [4, 0, 1]
    r.recovery.before_iteration()
    assert device.gpu_quota == 57 * MIB
    r.client.deliver(_record(MembershipState.PROPOSE, generation=2, active_mask=0b0011))
    # The rank holds its forward for the change and asks for no further chunk.
    assert _vote(r)[0] == [0, 1, 0]
    r.recovery.before_iteration()
    assert device.gpu_quota == 57 * MIB
    assert r.recovery.pauses_forward()


@pytest.mark.parametrize("eplb", [False, True], ids=["no EPLB", "no empty spare slot"])
def test_commit_without_empty_spare_slots_starts_no_worker(monkeypatch, eplb):
    """Check that a commit with no empty spare slot to release starts no release worker."""
    r = _resuming()
    release_spare_slots = Mock(return_value=0)
    if eplb:
        _eplb(monkeypatch, r)
        r.executor.model_engine.moe_load_balancer.expert_restore.release_spare_slots = (
            release_spare_slots
        )
    r.recovery.before_iteration()
    assert _replies(r) == [Reply.INSTALLED]
    _wait_for_release_worker(r)
    release_spare_slots.assert_not_called()


# EPLB and waiting requests across a membership change.


def test_restore_plans_from_slots_after_abandon(monkeypatch):
    """Check that the restore plans from the slots a rank holds once its EPLB iterations end."""
    r = _resuming()
    calls, _, _ = _eplb(monkeypatch, r)
    balancer = r.executor.model_engine.moe_load_balancer
    updated = ((4, ((1, -1), (1, -1), (3, -1), (2, -1))),)

    def abandon_iterations():
        calls.append("abandon")
        balancer.expert_restore.current_placement = lambda: updated

    balancer.abandon_iterations = abandon_iterations
    r.recovery.before_iteration()
    assert _replies(r) == [Reply.INSTALLED]
    (_, _, (current,), _) = next(call for call in calls if call[0] == "plan")
    assert current.tolist() == [[1, -1], [1, -1], [-1, -1], [2, -1]]


@pytest.mark.parametrize("given_up", [False, True], ids=["replacement pending", "given up"])
@pytest.mark.parametrize("rank", [0, 1])
def test_requests_pinned_to_absent_rank_fail(rank, given_up):
    """Check that requests pinned to an absent rank fail on rank 0, retryable until given up."""
    r = _rank(rank)
    record = _record(MembershipState.COMMITTED)
    r.client.deliver(record.model_copy(update={"given_up_mask": 1 << FAILED if given_up else 0}))
    r.store.check = lambda keys: True
    r.store.get = r.store.multi_get = Mock(side_effect=UnsignedValueError("forged"))
    waiting = FCFSWaitingQueue()
    waiting.add_requests([_item(1), _item(2, target=FAILED), _item(3, target=FAILED, relax=True)])
    r.recovery.fail_unservable_requests(waiting, [RankState(rank=i) for i in range(WORLD)])
    assert [item.id for item in waiting] == [1, 2, 3]  # Every rank serves.

    survivors = [RankState(rank=i) for i in range(WORLD) if i != FAILED]
    r.recovery.fail_unservable_requests(waiting, survivors)
    assert [item.id for item in waiting] == [1, 3]
    fail = r.executor.fail_requests_by_id
    if rank == 0:
        ((failed, error), _) = fail.call_args
        assert failed == {2: 102}
        assert isinstance(error, RankFailureError) != given_up
        # Resubmitted, it would fail the same way until the replacement joins.
        assert given_up or not error.resubmit
        assert "attention_dp_relax" in str(error)
    else:
        fail.assert_not_called()


# One MoE layer of four experts at startup. Rank r holds expert r and an empty spare slot.
_STARTUP = ((4, ((0, -1), (1, -1), (2, -1), (3, -1))),)


def test_agreed_placement_uses_own_slots():
    """Check that the agreed placement is the slots each active rank gathered."""
    own_slots = [
        _own_slots(((4, ((1, -1), (0, -1), (3, -1), (2, -1))),), 0),
        _own_slots(_STARTUP, 1),  # Rank 1 missed the update, so its slots still hold expert 1.
        None,  # Rank 2 is inactive.
        _own_slots(_STARTUP, 3),  # Rank 3 joins.
    ]
    record = MembershipRecord(
        generation=2, state=MembershipState.INSTALL, active_mask=0b1011, joining_mask=1 << 3
    )
    (agreed,) = recovery_module.agreed_placement(own_slots, record)
    assert agreed.tolist() == [[1, -1], [1, -1], [-1, -1], [3, -1]]
    # Without rank 0, the others' slots stay theirs.
    without_rank0 = record.model_copy(update=dict(active_mask=0b1010))
    (agreed,) = recovery_module.agreed_placement(own_slots, without_rank0)
    assert agreed.tolist() == [[-1, -1], [1, -1], [-1, -1], [3, -1]]
    # Only a newer change leaves a survivor's or a joining rank's slots out of the gather.
    for missing in (1, 3):
        gathered = [None if r == missing else slots for r, slots in enumerate(own_slots)]
        with pytest.raises(recovery_module._MembershipRestarted):
            recovery_module.agreed_placement(gathered, record)


# The end of the startup of each rank, when its recovery takes over its membership.


_FITTING_PEER = (True, 10 * MIB, 0, STARTUP_MEMORY.kv_cache_quota)


def _original_rank(monkeypatch, rank, restore_bytes=6 * MIB, peer=_FITTING_PEER):
    """Return an original rank with EPLB, its device and startup steps, with rank 1's ``peer``."""
    r = _unstarted_rank(rank)
    steps = []
    executor = r.executor
    executor.dist.host_exchange = r.exchange
    executor.dist.allgather.side_effect = lambda sizes: [sizes, peer, _FITTING_PEER, _FITTING_PEER]
    executor.llm_args = SimpleNamespace(
        kv_cache_config=SimpleNamespace(free_gpu_memory_fraction=0.9)
    )
    executor.kv_cache_manager = device = _Device()
    executor.kv_pools_suspended = device.suspended
    device.measure(monkeypatch)
    r.moe.workspace_bytes.return_value = 4 * MIB
    _bind_moe_state(monkeypatch, r)
    expert_restore = MagicMock(spec=EplbExpertRestore)
    expert_restore.largest_failure_restore_bytes.side_effect = lambda: (
        steps.append("restore fits"),
        restore_bytes,
    )[1]
    executor.model_engine.moe_load_balancer = MagicMock(spec=MoeLoadBalancer)
    executor.model_engine.moe_load_balancer.expert_restore = expert_restore
    r.client.bind = Mock(side_effect=lambda *args, **kwargs: steps.append("bind"))
    for name, value in (
        ("membership_client", lambda: r.client),
        ("authority_process", lambda: None),
        ("rendezvous_store", lambda: (steps.append("store"), r.store)[1]),
        ("publish_startup_record", lambda store: store.set("agreed_startup", b"published")),
        ("publish_tuned_tactics", lambda store, rank: steps.append(f"tactics of rank {rank}")),
        ("wait_for_every_rank", lambda: steps.append("every rank started")),
    ):
        monkeypatch.setattr(recovery_module, name, value)
    return r, device, steps


@pytest.mark.parametrize("rank", [0, 1])
def test_original_rank_binds_client_after_startup(
    fresh_startup_record, shared_secret, monkeypatch, rank
):
    """Check the steps of an original rank at the end of its startup."""
    r, device, steps = _original_rank(monkeypatch, rank)
    r.executor.resource_manager.resource_managers = {
        ResourceManagerType.KV_CACHE_MANAGER: KVCacheManagerV2.__new__(KVCacheManagerV2)
    }
    RankFailureRecovery.create(r.executor)
    # Each rank publishes its tuned tactics for its replacement before it reports that it started.
    tactics = f"tactics of rank {rank}"
    assert steps == ["restore fits", "store", tactics, "every rank started", "bind"]
    # The check shrank the KV cache pool for the largest failure's restore and took its quota back.
    assert _resizes(device) == [54 * MIB, STARTUP_MEMORY.kv_cache_quota]
    expected = {"agreed_startup": b"published"}
    assert r.store.values == (expected if rank == 0 else {})


@pytest.mark.parametrize(
    "restore_bytes, floor, peer, advice",
    [
        # The pool of 64 MiB keeps 8 MiB, so it needs 57 + 8 MiB: a fraction of 0.9 * 65 / 64.
        (
            53 * MIB,
            8 * MIB,
            _FITTING_PEER,
            "Rank 0 needs 0.06 GiB, and its pool of 0.06 GiB keeps up to 0.01 GiB for the pages in "
            "use. Raise kv_cache_config.free_gpu_memory_fraction from 0.9 to at least 0.92, or "
            "use more ranks",
        ),
        # The pool with no pages in use shrinks to 0 bytes, still 2 MiB short of 62 + 4 MiB.
        (
            62 * MIB,
            0,
            _FITTING_PEER,
            "Rank 0 needs 0.06 GiB, and its pool of 0.06 GiB keeps up to 0.00 GiB for the pages in "
            "use. Raise kv_cache_config.free_gpu_memory_fraction from 0.9 to at least 0.93",
        ),
        (
            52 * MIB,
            8 * MIB,
            (False, 3 << 30, 1 << 29, 20 << 30),
            "Rank 1 needs 3.00 GiB, and its pool of 20.00 GiB keeps up to 0.50 GiB for the pages "
            "in use. Raise kv_cache_config.free_gpu_memory_fraction from 0.9 to at least 0.16, or "
            "use more ranks",
        ),
        # No fraction gives the pool room for 30.5 GiB: 0.9 * 30.5 / 20 is above 1.
        (
            52 * MIB,
            8 * MIB,
            (False, 30 << 30, 1 << 29, 20 << 30),
            "Rank 1 needs 30.00 GiB, and its pool of 20.00 GiB keeps up to 0.50 GiB for the pages "
            "in use. Use more ranks, so that each rank restores fewer of a lost node's experts.",
        ),
    ],
)
def test_startup_rejects_unrestorable_largest_failure(
    fresh_startup_record, shared_secret, monkeypatch, restore_bytes, floor, peer, advice
):
    """Check that every original rank must be able to restore the largest failure at startup."""
    r, device, steps = _original_rank(monkeypatch, 0, restore_bytes, peer)
    device.floor = floor
    with pytest.raises(ValueError, match=re.escape(f"on ranks {[0] if peer[0] else [1]}")) as error:
        RankFailureRecovery.create(r.executor)
    assert advice in str(error.value)
    assert steps == ["restore fits"]
    assert device.calls.count("suspend") == 1
    # The pool tried the largest failure's quota and took its startup quota back.
    assert _resizes(device)[0] == 64 * MIB - restore_bytes - 4 * MIB
    assert device.gpu_quota == STARTUP_MEMORY.kv_cache_quota


@pytest.mark.parametrize(
    "manager_cls", [DSACacheManagerV2, DeepseekV4CacheManager, MambaHybridCacheManagerV2]
)
def test_startup_rejects_kv_cache_manager_with_own_reuse_commits(
    fresh_startup_record, shared_secret, monkeypatch, manager_cls
):
    """Check that startup refuses a KV cache manager that overrides its reuse commits."""
    r, device, steps = _original_rank(monkeypatch, 0)
    r.executor.resource_manager.resource_managers = {
        ResourceManagerType.KV_CACHE_MANAGER: KVCacheManagerV2.__new__(KVCacheManagerV2),
        ResourceManagerType.DRAFT_KV_CACHE_MANAGER: manager_cls.__new__(manager_cls),
    }
    with pytest.raises(ValueError, match=f"not {manager_cls.__name__}, which overrides"):
        RankFailureRecovery.create(r.executor)
    assert steps == []


@pytest.mark.parametrize("replacement", [False, True], ids=["original", "replacement"])
def test_no_rank_creates_a_process_group(fresh_startup_record, monkeypatch, replacement):
    """Check that no rank creates a torch.distributed group, which no replacement could join."""
    monkeypatch.setattr(StartupRecord, "replacement", replacement)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="process group"):
        RankFailureRecovery.create(_executor(rank=FAILED))


# Rejoin. A replacement rank starts alone and the active ranks admit it.


def test_replacement_waits_for_admission(monkeypatch):
    _exit_raises(monkeypatch)
    r = _replacement(monkeypatch)
    _publish_rows(r.store, _joining_record(MembershipState.INSTALL), FAILED)
    created = []

    def create():
        try:
            created.append(RankFailureRecovery.create(r.executor))
        except SystemExit as error:  # The patched os._exit.
            created.append(error)

    joined = threading.Thread(target=create, daemon=True)
    joined.start()
    # It reports only that it is ready to join. There is no EPLB here, so it sends no placement.
    _wait_until(lambda: r.client.reports)
    assert r.client.reports == [{"joining_ready": True}]
    r.client.deliver(_record(MembershipState.COMMITTED))  # Without rank 2, so not admitted yet.
    r.client.deliver(_joining_record(MembershipState.PROPOSE))
    _wait_until(lambda: len(r.client.reports) == 2)
    assert r.client.reports[-1]["reply"] == Reply.READY
    r.moe.request_abort.assert_not_called()  # It is not a member yet, so there is nothing to stop.
    r.client.deliver(_joining_record(MembershipState.INSTALL))
    joined.join(TIMEOUT_S)
    assert not joined.is_alive()
    (recovery,) = created
    assert isinstance(recovery, RankFailureRecovery)
    # It maps its fresh row at INSTALL and replies INSTALLED only at the resume point.
    rows = [b"rows-0", b"rows-1", b"row-mine", b"rows-3"]
    assert r.exchange.installs == [(rows, SURVIVORS, JOIN_RESUME.epoch)]
    assert Reply.INSTALLED not in _replies(r)
    assert r.executor.iter_counter == JOIN_RESUME.iteration + 1
    assert recovery.pauses_forward()


def _admitting(monkeypatch, rank):
    """Return ``rank`` after it installed the admission of rank 2, with request 7 handed over."""
    install = _joining_record(MembershipState.INSTALL)
    if rank == FAILED:
        r = _replacement(monkeypatch)
        _publish_rows(r.store, install, rank)
        r.client.record = install
        r.recovery = RankFailureRecovery.create(r.executor)
        r.store.set("host_exchange_epoch/2", str(ADMISSION_EPOCH).encode())  # Rank 0 admits it.
    else:
        r = _rank(rank)
        r.client.deliver(_record(MembershipState.COMMITTED))  # Rank 2 failed earlier.
        _install(r, install)
        r.executor.iter_counter = JOIN_RESUME.iteration + 1
    r.executor.dist.allgather.side_effect = lambda local: [
        local if i == rank else (set(), None) for i in range(WORLD)
    ]
    r.exchange.rank0_object = WaitingState([RequestQueueItem(7)], [8], False)
    r.executor.waiting_queue = FCFSWaitingQueue()
    committed = install.model_copy(update=dict(state=MembershipState.COMMITTED, joining_mask=0))
    r.client.answers[(Reply.INSTALLED, 2)] = [committed]
    return r


@pytest.mark.parametrize("shutdown", [False, True])
def test_rank0_admits_joining_rank(monkeypatch, shutdown):
    r = _admitting(monkeypatch, rank=0)
    executor = r.executor
    executor.is_shutdown = shutdown
    prompt = b"a user's prompt"
    executor.waiting_queue.add_requests(
        [RequestQueueItem(7, SimpleNamespace(priority=DEFAULT_REQUEST_PRIORITY, prompt=prompt))]
    )
    executor.canceled_req_ids.append(9)
    report = r.client.report
    handed = {}
    written = []
    set_value = r.store.set
    monkeypatch.setattr(
        r.store, "set", lambda key, value: (written.append(value), set_value(key, value))
    )

    def report_installed(**changes):
        if changes.get("reply") == Reply.INSTALLED:  # The joining rank has read it by then.
            handed.update(r.store.values)
        report(**changes)

    r.client.report = report_installed
    r.recovery.before_iteration()
    epoch = next_epoch(r.exchange.epoch)
    assert handed["host_exchange_epoch/2"] == str(epoch).encode()
    assert r.exchange.admits == [(ALL_RANKS, epoch)]
    assert len(r.exchange.installs) == 1  # Only at INSTALL.
    # Rank 0 hands the waiting queue, the pending cancellations and the shutdown flag to the
    # joining rank only.
    (((queued, canceled, handed_shutdown), readers, recipients),) = r.exchange.exchanges
    assert [item.id for item in queued] == [7]
    assert canceled == [9]
    assert handed_shutdown is shutdown
    assert (readers, recipients) == ([], [FAILED])
    # Every rank reads the request gather, which carries only request ids and slots.
    (gathered,) = executor.dist.allgather.call_args.args
    assert gathered == ({7}, None)
    assert not any(prompt in value for value in written)
    # Once committed, the store no longer keeps the epoch.
    assert "host_exchange_epoch/2" not in r.store.values
    executor.dist.broadcast.assert_not_called()
    r.moe.ep_group_health.mark_active.assert_any_call(FAILED)
    assert _replies(r) == [Reply.INSTALLED]
    assert not r.recovery.pauses_forward()


@pytest.mark.parametrize("shutdown", [False, True])
def test_joining_rank_takes_rank0_waiting_queue(monkeypatch, shutdown):
    r = _admitting(monkeypatch, rank=FAILED)
    r.exchange.rank0_object = WaitingState([RequestQueueItem(7)], [8], shutdown)
    executor = r.executor
    # The queue that an earlier attempt of this admission took.
    executor.waiting_queue.add_requests([RequestQueueItem(5)])
    r.recovery.before_iteration()
    # Request 5 counts as lost unless an active rank holds it, and a joining rank holds none.
    ((own_requests, _),) = executor.dist.allgather.call_args.args
    assert own_requests == set()
    assert r.exchange.admits == [(ALL_RANKS, ADMISSION_EPOCH)]
    assert len(r.exchange.installs) == 1  # Only at INSTALL.
    assert [item.id for item in executor.waiting_queue] == [7]
    assert executor.canceled_req_ids == [8]
    # A rank admitted after the shutdown request drains and stops with the others.
    assert executor.is_shutdown is shutdown
    ((_, readers, recipients),) = r.exchange.exchanges
    assert (readers, recipients) == ([0], [])
    executor.dist.broadcast.assert_not_called()
    # It ran no step and enters the host exchange only at its admission.
    executor.retire_inflight_batch.assert_not_called()
    r.moe.remap_workspaces.assert_called_once()
    r.moe.rearm.assert_called_once()
    r.moe.install_mask.assert_called_once()
    assert _replies(r) == [Reply.INSTALLED]
    assert not r.recovery.pauses_forward()


@pytest.mark.parametrize("rank", [0, 1])
def test_active_ranks_retire_step_before_resize(monkeypatch, rank):
    """Check that the active ranks retire the step in flight at the resume point before a resize."""
    r = _admitting(monkeypatch, rank=rank)
    executor = r.executor
    executor.previous_batch = _batch([_request(14)])
    device = _Device()
    steps = device.calls

    def retire():
        steps.append(("retire", executor.previous_batch is not None))
        executor.previous_batch = None

    executor.retire_inflight_batch.side_effect = retire
    r.exchange.admit = lambda active_mask, epoch: steps.append("admit")
    r.moe.workspace_bytes.return_value = 4 * MIB
    device.measure(monkeypatch)
    executor.kv_cache_manager = device
    executor.kv_pools_suspended = device.suspended
    r.recovery.before_iteration()
    _vote(r)
    r.recovery.before_iteration()
    # An admission applies the step in flight before the admit, and the release retires it again.
    order = [step for step in steps if step != "resume" and step[0] != "resize"]
    assert order == [("retire", True), "admit", "suspend", ("retire", False), "suspend"]
    # Only the joining rank reads rank 0's waiting queue.
    ((_, readers, recipients),) = r.exchange.exchanges
    assert (readers, recipients) == ([], [FAILED] if rank == 0 else [])
    assert len(executor.waiting_queue) == 0
    assert _replies(r) == [Reply.INSTALLED]


# A newer change that replaces the admission of rank 2: rank 3 fails and rank 2 still joins.
_NEWER = _joining_record(MembershipState.PROPOSE, generation=3).model_copy(
    update=dict(active_mask=0b0111)
)
_NEWER_INSTALL = _NEWER.model_copy(
    update=dict(state=MembershipState.INSTALL, resume=ResumeState(iteration=30, epoch=50))
)


@pytest.mark.parametrize(
    "installed", [False, True], ids=["proposed after the retire", "installed inside the retire"]
)
def test_newer_change_during_resume_keeps_collectives(monkeypatch, installed):
    """Check that a newer change in the resume keeps the active ranks in the same collectives."""
    r = _admitting(monkeypatch, rank=0)

    def retire():  # The retire's collectives stop for the newer change.
        if installed:
            _install(r, _NEWER_INSTALL)
        else:
            r.client.deliver(_NEWER)

    r.executor.retire_inflight_batch.side_effect = retire
    r.recovery.before_iteration()
    # A rank that installed the newer change still hands over the waiting queue of the replaced one.
    r.executor.dist.allgather.assert_called_once()
    assert len(r.exchange.exchanges) == 1
    epoch = next_epoch(r.exchange.epoch)
    assert r.exchange.admits == ([] if installed else [(ALL_RANKS, epoch)])
    assert r.store.check(["host_exchange_epoch/2"]) != installed
    assert _replies(r) == []


def test_joining_rank_left_out_exits(monkeypatch):
    """Check that a joining rank left out by a newer change exits, so another one can start."""
    _exit_raises(monkeypatch)
    r = _admitting(monkeypatch, rank=FAILED)
    without_rank_2 = _NEWER.model_copy(update=dict(active_mask=0b1011, joining_mask=0))
    r.client.answers[(Reply.INSTALLED, 2)] = [without_rank_2]
    with pytest.raises(SystemExit) as exited:
        r.recovery.before_iteration()
    assert exited.value.code == 1


def test_joining_rank_missing_commit_stops(monkeypatch):
    """Check that a joining rank that reads a newer PROPOSE, not its commit, stops like others."""
    _exit_raises(monkeypatch)
    r = _admitting(monkeypatch, rank=FAILED)
    committed = _joining_record(MembershipState.COMMITTED, joining=0)
    newer = _NEWER.model_copy(update=dict(joining_mask=0))
    r.client.answers[(Reply.INSTALLED, 2)] = [committed, newer]
    # An abort answers a READY to generation 3, so that a READY ends the test.
    r.client.answers[(Reply.READY, 3)] = [
        newer.model_copy(update=dict(state=MembershipState.ABORTED))
    ]
    r.recovery.before_iteration()
    assert _replies(r) == [Reply.INSTALLED]
    assert r.exchange.interrupted()
