# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whole-executor containment tests; these do not qualify a GPU/RDMA reuse fence."""

from __future__ import annotations

import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.disaggregation.base import Chunk, TokenRange
from tensorrt_llm._torch.disaggregation.base.transfer import SessionStatus
from tensorrt_llm._torch.disaggregation.native import transfer as transfer_mod
from tensorrt_llm._torch.disaggregation.native.retirement import (
    QuiescenceFatalEvent,
    RetirementWatchdog,
)
from tensorrt_llm._torch.disaggregation.transceiver import (
    KvCacheTransceiverV2,
    _fail_unproven_kv_transfer,
)
from tensorrt_llm._torch.pyexecutor import hang_detector
from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
from tensorrt_llm.executor import EngineDeadError
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.worker_process_monitor import _read_process_state
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def retirement_mpi_world(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Isolate MPI runtime qualification without creating or aborting a real world.

    Args:
        monkeypatch: Restores the MPI runtime probes after each case.

    Returns:
        A two-rank MPI communicator standing in for this executor's worker world.
    """
    from mpi4py import MPI

    from tensorrt_llm._torch.disaggregation import transceiver

    communicator = Mock(spec=["Is_inter", "Get_size", "Get_rank", "Abort"])
    communicator.Is_inter.return_value = False
    communicator.Get_size.return_value = 2
    communicator.Get_rank.return_value = 1
    monkeypatch.setattr(MPI, "Is_initialized", lambda: True)
    monkeypatch.setattr(MPI, "Is_finalized", lambda: False)
    monkeypatch.setattr(MPI, "Query_thread", lambda: MPI.THREAD_MULTIPLE)
    for module in (transceiver, hang_detector):
        monkeypatch.setattr(module, "ENABLE_MULTI_DEVICE", True)
        monkeypatch.setattr(module, "mpi_disabled", lambda: False)
        monkeypatch.setattr(module, "mpi_comm", lambda: communicator)
    return communicator


@pytest.mark.parametrize(
    "invalid_world",
    [
        "single_device",
        "disabled",
        "uninitialized",
        "finalized",
        "thread_level",
        "null_comm",
        "intercomm",
        "wrong_size",
        "wrong_rank",
    ],
)
def test_retirement_rejects_unqualified_executor_world(
    monkeypatch: pytest.MonkeyPatch, retirement_mpi_world: Mock, invalid_world: str
) -> None:
    """A partial-rank or non-thread-safe kill route cannot enable the deadline policy."""
    from mpi4py import MPI

    from tensorrt_llm._torch.disaggregation import transceiver

    mapping = Mapping(world_size=2, rank=1, tp_size=2)
    assert transceiver._retirement_executor_comm(mapping) is retirement_mpi_world
    warning = Mock()
    monkeypatch.setattr(transceiver.logger, "warning", warning)
    if invalid_world == "single_device":
        monkeypatch.setattr(transceiver, "ENABLE_MULTI_DEVICE", False)
    elif invalid_world == "disabled":
        monkeypatch.setattr(transceiver, "mpi_disabled", lambda: True)
    elif invalid_world == "uninitialized":
        monkeypatch.setattr(MPI, "Is_initialized", lambda: False)
    elif invalid_world == "finalized":
        monkeypatch.setattr(MPI, "Is_finalized", lambda: True)
    elif invalid_world == "thread_level":
        monkeypatch.setattr(MPI, "Query_thread", lambda: MPI.THREAD_SERIALIZED)
    elif invalid_world == "null_comm":
        monkeypatch.setattr(transceiver, "mpi_comm", lambda: MPI.COMM_NULL)
    elif invalid_world == "intercomm":
        retirement_mpi_world.Is_inter.return_value = True
    elif invalid_world == "wrong_size":
        retirement_mpi_world.Get_size.return_value = 3
    else:
        retirement_mpi_world.Get_rank.return_value = 0
    with pytest.raises(ValueError, match="MPI_THREAD_MULTIPLE.*rank/size matching Mapping"):
        transceiver._retirement_executor_comm(mapping)
    warning.assert_not_called()


@pytest.mark.parametrize("rank", [0, 1])
def test_retirement_warns_executor_root_about_job_wide_abort(
    monkeypatch: pytest.MonkeyPatch, retirement_mpi_world: Mock, rank: int
) -> None:
    """Qualification identifies participating ranks, not a separately isolated MPI job."""
    from tensorrt_llm._torch.disaggregation import transceiver

    warning = Mock()
    monkeypatch.setattr(transceiver.logger, "warning", warning)
    retirement_mpi_world.Get_rank.return_value = rank
    communicator = transceiver._retirement_executor_comm(
        Mapping(world_size=2, rank=rank, tp_size=2)
    )
    assert communicator is retirement_mpi_world
    if rank == 0:
        warning.assert_called_once()
        message = warning.call_args.args[0]
        assert "MPI job" in message
        assert "independent" in message
    else:
        warning.assert_not_called()


def test_fatal_callback_uses_captured_world_on_background_thread(
    monkeypatch: pytest.MonkeyPatch, retirement_mpi_world: Mock
) -> None:
    """The watchdog must not select a different global or thread-local communicator."""
    from tensorrt_llm import _utils
    from tensorrt_llm._torch.disaggregation import transceiver

    global_world = Mock()
    monkeypatch.setattr(_utils, "comm", global_world)
    monkeypatch.setattr(_utils.thread_local_comm, "value", retirement_mpi_world, raising=False)
    monkeypatch.setattr(transceiver, "mpi_comm", _utils.mpi_comm)
    instance = transceiver.KvCacheTransceiverV2.__new__(transceiver.KvCacheTransceiverV2)
    instance._retirement_mpi_comm = transceiver._retirement_executor_comm(
        Mapping(world_size=2, rank=1, tp_size=2)
    )
    # A new thread would normally resolve mpi_comm() without the creator's
    # thread-local override. Containment must instead use the captured object.
    lookup = Mock(wraps=_utils.mpi_comm)
    kill = Mock()
    monkeypatch.setattr(hang_detector, "mpi_comm", lookup)
    monkeypatch.setattr(hang_detector.os, "kill", kill)
    monkeypatch.setattr(transceiver, "_log_unproven_kv_transfer", Mock())
    event = QuiescenceFatalEvent(12, "receive", "cancelled", 1.0, 2.0, 2.0)
    background_world = []

    def dispatch() -> None:
        """Observe the new thread's MPI context before invoking the bound callback."""
        background_world.append(_utils.mpi_comm())
        instance._fail_unproven_transfer(event)

    thread = threading.Thread(target=dispatch)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert background_world == [global_world]
    retirement_mpi_world.Abort.assert_called_once_with(137)
    global_world.Abort.assert_not_called()
    lookup.assert_not_called()
    kill.assert_not_called()


@pytest.mark.parametrize("thread_error", [None, "start", "join"])
def test_unproven_fatal_bounds_log_join_and_always_dispatches_kill(
    monkeypatch: pytest.MonkeyPatch, thread_error: str | None
) -> None:
    """Try to finish the fatal record, but terminate even if thread setup fails."""
    from tensorrt_llm._torch.disaggregation import transceiver

    event = QuiescenceFatalEvent(12, "receive", "cancelled", 1.0, 2.0, 2.0)
    record = Mock()
    log_thread = Mock()
    log_thread.join.side_effect = lambda **_: record(event)
    factory = Mock(return_value=log_thread)
    kill = Mock()
    monkeypatch.setattr(transceiver, "_log_unproven_kv_transfer", record)
    monkeypatch.setattr(transceiver.threading, "Thread", factory)
    monkeypatch.setattr(hang_detector, "propagate_hard_kill", kill)
    if thread_error is None:
        kill.side_effect = lambda **_: record.assert_called_once_with(event)
        _fail_unproven_kv_transfer(event)
    else:
        getattr(log_thread, thread_error).side_effect = RuntimeError("test thread failure")
        with pytest.raises(RuntimeError, match="test thread failure"):
            _fail_unproven_kv_transfer(event)
        record.assert_not_called()

    factory.assert_called_once_with(
        target=record, args=(event,), name="kv-retirement-fatal-log", daemon=True
    )
    log_thread.start.assert_called_once_with()
    if thread_error == "start":
        log_thread.join.assert_not_called()
    else:
        log_thread.join.assert_called_once_with(timeout=0.5)
    kill.assert_called_once_with(diagnostics=False, communicator=None)


def test_unproven_fatal_dispatches_kill_without_waiting_forever_for_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blocked logging handler cannot block the watchdog's kill dispatch."""
    from tensorrt_llm._torch.disaggregation import transceiver

    kill = Mock()
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocked_log(*_args: object) -> None:
        """Hold the diagnostic write until containment has already dispatched."""
        entered.set()
        try:
            release.wait(timeout=5)
        finally:
            finished.set()

    monkeypatch.setattr(hang_detector, "propagate_hard_kill", kill)
    monkeypatch.setattr(transceiver.logger, "critical", blocked_log)
    event = QuiescenceFatalEvent(12, "receive", "cancelled", 1.0, 2.0, 2.0)
    try:
        _fail_unproven_kv_transfer(event)
        assert entered.wait(timeout=1)
        assert not finished.is_set()
        kill.assert_called_once_with(diagnostics=False, communicator=None)
    finally:
        release.set()
        assert finished.wait(timeout=1)


def test_deadline_hard_kill_bypasses_blocking_diagnostic_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The no-diagnostics path skips both stream flushing and log handlers."""
    monkeypatch.setattr(hang_detector, "ENABLE_MULTI_DEVICE", False)
    flush = Mock(side_effect=AssertionError("must not flush a possibly blocked stream"))
    log = Mock(side_effect=AssertionError("must not acquire a logging lock"))
    kill = Mock()
    monkeypatch.setattr(hang_detector, "_best_effort_flush_streams", flush)
    monkeypatch.setattr(hang_detector, "_best_effort_log_error", log)
    monkeypatch.setattr(hang_detector.os, "kill", kill)
    hang_detector.propagate_hard_kill(diagnostics=False)
    kill.assert_called_once_with(os.getpid(), signal.SIGKILL)
    flush.assert_not_called()
    log.assert_not_called()


@pytest.mark.parametrize("exposure", ["never_exposed", "settled_before_grace", "unresolved"])
def test_peer_loss_fails_closed_only_for_unsettled_exposure(exposure: str) -> None:
    """Peer loss is not a broadcast kill; each world evaluates its own claims.

    This tests containment decisions, not NIXL/RDMA revocation or the launcher's
    physical blast radius. The existing communicator test covers kill dispatch.
    """
    now = [10.0]
    contain = Mock()
    unaffected_contain = Mock()
    watchdog = RetirementWatchdog(contain, clock=lambda: now[0])
    unaffected = RetirementWatchdog(unaffected_contain, clock=lambda: now[0])
    owner = watchdog.create_owner(12, "receive", 1.0)
    operation = object()
    if exposure != "never_exposed":
        assert owner.expose(operation)
    owner.request_drain("peer lost")
    assert not owner.expose(object())

    now[0] = 10.5
    watchdog.progress()
    contain.assert_not_called()
    if exposure == "settled_before_grace":
        assert owner.settle(operation)
        assert owner.settle(operation)  # Repeated safe evidence is idempotent.
        assert owner.can_retire()
    elif exposure == "unresolved":
        assert not owner.can_retire()

    # The same peer-loss trigger has a different result only when this world's
    # own exposed claim is still unproven at its fixed grace deadline.
    now[0] = 11.0
    watchdog.progress()
    watchdog.progress()
    if exposure == "unresolved":
        contain.assert_called_once_with(watchdog.fatal)
        assert watchdog.fatal is not None
        assert watchdog.fatal.reason == "peer lost"
        assert not owner.settle(operation)
        assert not owner.can_retire()
        assert not owner.close()
        with pytest.raises(RuntimeError, match="admission is closed"):
            watchdog.create_owner(13, "receive", 1.0)
        with pytest.raises(RuntimeError, match="retained owners"):
            watchdog.stop()
    else:
        contain.assert_not_called()
        assert watchdog.fatal is None
        assert owner.can_retire()
        assert owner.close()
        assert owner.close()
        assert watchdog.create_owner(13, "receive", 1.0).close()
        watchdog.stop()

    # An independent world with no unresolved exposure remains usable. It could
    # fail later only if one of its own transfers acquires an unproven claim.
    unaffected.progress()
    unaffected_contain.assert_not_called()
    new_owner = unaffected.create_owner(14, "send", 1.0)
    assert new_owner.expose(operation)
    assert new_owner.settle(operation)
    assert new_owner.close()
    unaffected.stop()


def test_failed_sender_write_retains_roots_until_fatal_deadline() -> None:
    """A persistent backend error closes sender admission, not physical ownership."""
    now = [10.0]
    contain = Mock()
    watchdog = RetirementWatchdog(contain, clock=lambda: now[0])
    owner = watchdog.create_owner(12, "send", 1.0)
    task = transfer_mod.SendTaskBase(DisaggregatedParams(disagg_request_id=12))
    task.bind_logical_outcomes(transfer_mod._LogicalOutcomes(owner))
    assert task.begin_physical_operation(7)
    request = object()
    status = Mock(wait=Mock(return_value=False), is_completed=Mock(return_value=False))
    status.last_status_str.return_value = "ERROR"
    sender = object.__new__(transfer_mod.Sender)
    sender._shutdown = True  # No worker threads or transport were created by this fixture.
    sender._sessions_lock = threading.Lock()
    sender._enforce_physical_ownership = True
    sender._ownership_poisoned = None
    sender._ownership_poison_lock = watchdog.lock
    sender._agent = Mock(submit_transfer_requests=Mock(return_value=status))

    assert sender._submit_transfer(task, 7, request) == (False, "ERROR")
    operation = task._physical_operations[7]
    assert operation.state is transfer_mod._PhysicalOperationState.IN_DOUBT
    assert operation.request is request
    assert operation.status is status
    assert not task.resources_drained

    # Sender poisoning rejects a different request even before grace expires.
    later = transfer_mod.SendTaskBase(DisaggregatedParams(disagg_request_id=13))
    assert later.begin_physical_operation(8)
    with pytest.raises(transfer_mod._TransferNotSubmittedError, match="before backend submission"):
        sender._submit_transfer(later, 8, object())
    sender._agent.submit_transfer_requests.assert_called_once_with(request)
    assert later._physical_operations[8].state is transfer_mod._PhysicalOperationState.NOT_SUBMITTED
    assert later.resources_drained

    now[0] = 10.5
    assert not task.poll_in_doubt_physical_operation(7)
    watchdog.progress()
    contain.assert_not_called()
    assert not owner.can_retire()

    # Repeated ERROR observations cannot settle the write or reset its deadline.
    now[0] = 11.0
    assert not task.poll_in_doubt_physical_operation(7)
    watchdog.progress()
    watchdog.progress()
    contain.assert_called_once_with(watchdog.fatal)
    assert watchdog.fatal is not None
    assert watchdog.fatal.request_id == 12
    assert watchdog.fatal.direction == "send"
    assert watchdog.fatal.reason == "backend quiescence unproven"
    assert watchdog.fatal.started_at == 10.0
    assert watchdog.fatal.deadline == 11.0
    assert operation.state is transfer_mod._PhysicalOperationState.IN_DOUBT
    assert operation.request is request
    assert operation.status is status
    assert not task.resources_drained
    assert not owner.close()


@pytest.mark.parametrize("kill_raises", [False, True])
def test_failed_termination_cannot_reopen_admission_or_retire_roots(
    monkeypatch: pytest.MonkeyPatch, kill_raises: bool
) -> None:
    """A failed kill attempt never converts UNPROVEN_FATAL into safe retirement."""
    from tensorrt_llm._torch.disaggregation import transceiver

    logged = threading.Event()
    log_threads: list[threading.Thread] = []

    def record_log(*_args: object) -> None:
        """Capture the diagnostic thread so the kill double cannot leave it running."""
        log_threads.append(threading.current_thread())
        logged.set()

    monkeypatch.setattr(transceiver.logger, "critical", record_log)
    kill = Mock(side_effect=OSError("test kill failure") if kill_raises else None)
    monkeypatch.setattr(hang_detector, "propagate_hard_kill", kill)
    now = [1.0]
    watchdog = RetirementWatchdog(_fail_unproven_kv_transfer, clock=lambda: now[0])
    owner = watchdog.create_owner(12, "send", 1.0)
    physical_operation = object()
    assert owner.expose(physical_operation)
    owner.request_drain("cancelled")
    now[0] = 2.0
    watchdog.progress()
    assert logged.wait(timeout=1)
    log_threads[0].join(timeout=1)
    assert not log_threads[0].is_alive()
    fatal = watchdog.fatal
    assert fatal is not None
    kill.assert_called_once_with(diagnostics=False, communicator=None)

    assert not owner.settle(physical_operation)
    assert not owner.can_retire()
    assert not owner.close()
    with pytest.raises(RuntimeError, match="admission is closed"):
        watchdog.create_owner(13, "receive", 1.0)
    with pytest.raises(RuntimeError, match="retained owners"):
        watchdog.stop()
    watchdog.progress()
    assert watchdog.fatal is fatal
    kill.assert_called_once_with(diagnostics=False, communicator=None)


_FATAL_ROOTS_SCRIPT = """
import gc
import sys
import threading
import weakref
from types import SimpleNamespace
from tensorrt_llm._torch.disaggregation.native.retirement import RetirementWatchdog
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2
from tensorrt_llm._torch.pyexecutor import hang_detector

class Resource:
    pass

manager, request, agent = Resource(), Resource(), Resource()
transceiver = KvCacheTransceiverV2.__new__(KvCacheTransceiverV2)
transceiver._kv_cache_manager = manager
transceiver._send_reqs = {12: request}
transceiver._transfer_worker = SimpleNamespace(agent=agent)
transceiver._retirement_mpi_comm = object()
refs = [weakref.ref(item) for item in (transceiver, manager, request, agent)]
notified = threading.Event()
def cannot_terminate(**_kwargs):
    notified.set()
    if sys.argv[1] == "raise":
        raise OSError("controlled test cannot terminate")
hang_detector.propagate_hard_kill = cannot_terminate
now = [1.0]
watchdog = RetirementWatchdog(transceiver._fail_unproven_transfer, clock=lambda: now[0])
owner = watchdog.create_owner(12, "send", 1.0)
assert owner.expose(object())
owner.request_drain("test cancellation")
now[0] = 2.0
watchdog.start()
assert notified.wait(5), "fatal callback never ran"
# If the fatal thread exits, the old implementation becomes collectable. The
# live thread must own its callback/root graph after the caller drops everything.
watchdog._thread.join(timeout=1)
assert watchdog._thread.is_alive(), "fatal root keeper exited"
del manager, request, agent, transceiver, watchdog, owner
gc.collect()
assert all(ref() is not None for ref in refs), "physical resource roots were collected"
print("FATAL_ROOTS_RETAINED", flush=True)
"""


def _standalone_process_env() -> dict[str, str]:
    """Remove enclosing launcher identities before starting an owned test job.

    Returns:
        The current runtime environment without MPI/Slurm rank identities.
    """
    prefixes = ("SLURM_", "PMIX_", "PMI_", "OMPI_", "I_MPI_", "HYDRA_", "MPI_")
    return {key: value for key, value in os.environ.items() if not key.startswith(prefixes)}


@pytest.mark.parametrize("kill_result", ["return", "raise"])
@pytest.mark.timeout(360)
def test_fatal_watchdog_retains_complete_transceiver_roots(kill_result: str) -> None:
    """Fatal root retention survives GC without leaking fatal threads into pytest."""
    result = subprocess.run(
        [sys.executable, "-c", _FATAL_ROOTS_SCRIPT, kill_result],
        env={**_standalone_process_env(), "TLLM_DISABLE_MPI": "1"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-4000:]
    assert "FATAL_ROOTS_RETAINED" in result.stdout


def _shutdown_transceiver() -> tuple[KvCacheTransceiverV2, transfer_mod.RxSession, Mock]:
    """Connect real shutdown/session logic to isolated allocator and network boundaries.

    Returns:
        A transceiver with one generation-first receive session and its AUX allocator.
    """
    watchdog = RetirementWatchdog(Mock(), clock=lambda: 10.0)
    receiver = transfer_mod.Receiver.__new__(transfer_mod.Receiver)
    receiver._retirement_watchdog = watchdog
    receiver._enforce_physical_ownership = True
    receiver._sessions_lock = threading.Lock()
    receiver._sessions = {}
    receiver._pre_cancelled_rids = {}
    receiver._shutdown = False
    receiver._ownership_admission_lock = threading.Lock()
    receiver._ownership_poisoned = None
    receiver._bounce = Mock()
    receiver._dealers = {}
    receiver._messenger = Mock()
    receiver.send_cancel_to_senders = Mock(return_value=None)
    aux = Mock()
    aux.alloc_slot.return_value = SimpleNamespace(id=3)
    session = transfer_mod.RxSession(
        12,
        DisaggregatedParams(
            disagg_request_id=12, schedule_style=DisaggScheduleStyle.GENERATION_FIRST
        ),
        receiver,
        aux_buffer=aux,
        timeout_s=5.0,
        retirement_watchdog=watchdog,
    )
    worker = transfer_mod.TransferWorker.__new__(transfer_mod.TransferWorker)
    worker._retirement_watchdog = watchdog
    worker._receiver = receiver
    worker._bounce = receiver._bounce
    worker._registered_mem = []
    instance = KvCacheTransceiverV2.__new__(KvCacheTransceiverV2)
    instance._transfer_worker = worker
    instance._send_sessions = {}
    instance._send_reqs = {}
    instance._recv_sessions = {12: session}
    instance._recv_reqs = {12: object()}
    return instance, session, aux


@pytest.mark.parametrize("prepared,already_failed", [(False, False), (True, False), (True, True)])
def test_shutdown_retires_unpublished_receive_and_aux_once(
    prepared: bool, already_failed: bool
) -> None:
    """Real shutdown cancels unexposed admission without rewriting an earlier failure."""
    instance, session, aux = _shutdown_transceiver()
    task = session.prepare_receive(Chunk([], [], TokenRange(0, 16), True)) if prepared else None
    if already_failed:
        session.fail_admission(RuntimeError("original receive failure"))
    earlier_outcome = session._logical_outcomes.terminal
    instance.shutdown()
    outcome = session._logical_outcomes.terminal
    assert outcome is not None
    assert outcome.status is (SessionStatus.ERROR if already_failed else SessionStatus.CANCELLED)
    if earlier_outcome is not None:
        assert outcome is earlier_outcome
    if task is not None:
        assert task.logical_outcome is outcome
    assert session.resources_drained()
    assert not session._receiver._sessions
    assert not instance._send_sessions and not instance._send_reqs
    assert not instance._recv_sessions and not instance._recv_reqs
    watchdog = instance._transfer_worker._retirement_watchdog
    assert not watchdog._owners
    assert watchdog.fatal is None
    assert instance._shutdown_complete
    instance.shutdown()
    session.cancel_local(by_peer=True)
    assert session._logical_outcomes.terminal is outcome
    aux.free_slot.assert_called_once_with(3)
    session._receiver.send_cancel_to_senders.assert_not_called()


def test_shutdown_retains_receive_reserved_for_blocked_publication() -> None:
    """Cancellation cannot retire a publication that has reserved ownership and may escape."""
    instance, session, aux = _shutdown_transceiver()
    task = session.prepare_receive(Chunk([], [], TokenRange(0, 16), True))
    assert task is not None
    task.expected_transfers = 1
    entered, release = threading.Event(), threading.Event()
    published, errors = set(), []

    def publish() -> None:
        """Pause at the network boundary after publication admission wins."""
        entered.set()
        assert release.wait(5)
        published.add(7)

    def dispatch() -> None:
        """Run real publication, retaining exceptions for the test thread."""
        try:
            session.try_begin_transfer(
                0, {"sender"}, {7}, publish=publish, published_writers=published
            )
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=dispatch)
    thread.start()
    try:
        assert entered.wait(1)
        with pytest.raises(RuntimeError, match="physical resources remain active"):
            instance.shutdown()
        assert instance._recv_sessions[12] is session
        assert instance._recv_reqs[12] is not None
        assert session.status is SessionStatus.CANCELLED
        assert not session.resources_drained()
        aux.free_slot.assert_not_called()
        session._receiver._messenger.stop.assert_not_called()
    finally:
        release.set()
        thread.join(timeout=1)
    assert not thread.is_alive()
    assert not errors
    task.record_writer_result(7, True, wait_for_local_completion=False)
    session._aux_physical_owner.record_writer_result(7, True, wait_for_local_completion=False)
    instance.shutdown()
    assert session.status is SessionStatus.CANCELLED
    aux.free_slot.assert_called_once_with(3)
    session._receiver.send_cancel_to_senders.assert_not_called()


def test_existing_proxy_closes_endpoint_after_retirement_worker_death() -> None:
    """The existing supervisor fails pending requests and rejects replacement work."""
    proxy = GenerationExecutorProxy.__new__(GenerationExecutorProxy)
    proxy._engine_dead = False
    proxy._fatal_error = None
    proxy.doing_shutdown = False
    proxy.workers_started = False
    proxy._multi_frontend_ipc_dir = None
    proxy._owns_mpi_session = False
    proxy._error_queue = queue.Queue()
    result = SimpleNamespace(queue=queue.Queue())
    proxy._results = {1: result}
    proxy.pre_shutdown = Mock()

    proxy._handle_worker_death(RuntimeError("UNPROVEN_FATAL worker world terminated"))
    assert isinstance(result.queue.get_nowait(), EngineDeadError)
    assert not proxy.check_health()
    with pytest.raises(EngineDeadError):
        proxy.submit(Mock())
    proxy.pre_shutdown.assert_called_once_with()


_MPI_RETIREMENT_SCRIPT = """
import os
import sys
import threading
from mpi4py import MPI
from tensorrt_llm import _utils
from tensorrt_llm._torch.disaggregation.native.retirement import RetirementWatchdog
from tensorrt_llm._torch.disaggregation.transceiver import (
    _fail_unproven_kv_transfer, _retirement_executor_comm
)
from tensorrt_llm.executor.worker_process_monitor import capture_worker_process_identity
from tensorrt_llm.mapping import Mapping

world = MPI.COMM_WORLD
split_executors = sys.argv[1:] == ["split-executors"]
comm = world.Split(world.Get_rank() // 2) if split_executors else world
_utils.thread_local_comm.value = comm
retirement_comm = _retirement_executor_comm(
    Mapping(world_size=comm.Get_size(), rank=comm.Get_rank(), tp_size=comm.Get_size())
)
assert retirement_comm == comm
world.Barrier()
identity = capture_worker_process_identity(world.Get_rank())
print(f"RETIREMENT_RANK_READY:{identity.rank}:{identity.pid}:{identity.start_time}", flush=True)
def fail(event):
    assert event.expired_at >= event.deadline
    print("UNPROVEN_FATAL_OBSERVED", flush=True)
    _fail_unproven_kv_transfer(event, retirement_comm)
watchdog = RetirementWatchdog(fail)
if split_executors and world.Get_rank() >= 2:
    # This executor has no operation or deadline capable of causing its own abort.
    assert watchdog.fatal is None
    watchdog.start()
    print(f"RETIREMENT_OTHER_EXECUTOR_READY:{world.Get_rank()}", flush=True)
world.Barrier()
try:
    if world.Get_rank() == 0:
        owner = watchdog.create_owner(12, "receive", 0.1)
        assert owner.expose(object())
        owner.request_drain("test never-settles transfer")
        watchdog.start()
        threading.Event().wait()  # Model/backend progress is intentionally unavailable.
    else:
        comm.Barrier()  # No normal collective can notify the affected executor's peer.
        threading.Event().wait()  # The other executor cannot complete ordinary cleanup.
    print("RETIREMENT_RANK_SURVIVED", flush=True)
finally:
    print(f"RETIREMENT_ORDINARY_CLEANUP:{world.Get_rank()}", flush=True)
"""


def _run_mpi_world(
    script: str, *, world_size: int = 2, script_args: tuple[str, ...] = ()
) -> subprocess.CompletedProcess[str]:
    """Run an owned MPI job, killing its process group on test timeout.

    Args:
        script: Python code executed by each rank in the owned test world.
        world_size: Number of child ranks in this launcher job.
        script_args: Optional arguments passed to each rank's script.

    Returns:
        Completed launcher status and captured output.
    """
    command = [
        "mpirun",
        "--allow-run-as-root",
        "--oversubscribe",
        "-n",
        str(world_size),
        sys.executable,
        "-c",
        script,
        *script_args,
    ]
    with subprocess.Popen(
        command,
        env={**_standalone_process_env(), "TLLM_DISABLE_MPI": "0"},
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=600)
        except BaseException:  # cleanup the owned world on timeout or test interruption
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate(timeout=30)
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("mpirun") is None,
    reason="requires Linux process identities and mpirun",
)
@pytest.mark.timeout(900)
def test_real_mpi_retirement_kills_blocked_world_before_fresh_world_starts() -> None:
    """Exercise launcher containment and fresh-world startup, not DMA fencing.

    No tensors or transport are substituted for hardware fence evidence here:
    this is strictly the supervisor/control-plane portion of qualification.
    """
    failed = _run_mpi_world(_MPI_RETIREMENT_SCRIPT)
    output = failed.stdout + failed.stderr
    assert "RETIREMENT_RANK_READY:0:" in output, output[-4000:]
    assert "RETIREMENT_RANK_READY:1:" in output, output[-4000:]
    assert "UNPROVEN_FATAL" in output, output[-4000:]
    assert failed.returncode != 0, output[-4000:]
    assert "RETIREMENT_RANK_SURVIVED" not in output, output[-4000:]
    identities = [
        line.split(":")[1:]
        for line in output.splitlines()
        if line.startswith("RETIREMENT_RANK_READY:")
    ]
    assert len(identities) == 2, output[-4000:]
    for _rank, pid, start_time in identities:
        state = _read_process_state(int(pid))
        assert state is None or state[0] == "Z" or state[1] != int(start_time), (
            f"Old worker {pid} remains live after launcher exit: {state}"
        )

    # A fresh communicator is created only after the previous launch has ended.
    # This is NOT permission to reuse GPU pages: that requires the separately
    # qualified driver/registration teardown contract with a surviving peer.
    replacement = _run_mpi_world(
        "from mpi4py import MPI; MPI.COMM_WORLD.Barrier(); print('FRESH_WORLD_READY')"
    )
    assert replacement.returncode == 0, replacement.stderr[-4000:]
    assert replacement.stdout.count("FRESH_WORLD_READY") == 2


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("mpirun") is None,
    reason="requires Linux process identities and mpirun",
)
@pytest.mark.timeout(900)
def test_openmpi_subcommunicator_abort_also_terminates_other_executor_in_job() -> None:
    """Qualify Open MPI's shared-job blast radius, not a GPU or RDMA fence."""
    from mpi4py import MPI

    if MPI.get_vendor()[0] != "Open MPI":
        pytest.skip("job-wide subcommunicator abort behavior is qualified only for Open MPI")
    failed = _run_mpi_world(_MPI_RETIREMENT_SCRIPT, world_size=4, script_args=("split-executors",))
    output = failed.stdout + failed.stderr
    assert failed.returncode != 0, output[-4000:]
    assert output.count("UNPROVEN_FATAL_OBSERVED") == 1, output[-4000:]
    assert "RETIREMENT_OTHER_EXECUTOR_READY:2" in output, output[-4000:]
    assert "RETIREMENT_OTHER_EXECUTOR_READY:3" in output, output[-4000:]
    assert "RETIREMENT_RANK_SURVIVED" not in output, output[-4000:]
    assert "RETIREMENT_ORDINARY_CLEANUP" not in output, output[-4000:]
    identities = [
        line.split(":")[1:]
        for line in output.splitlines()
        if line.startswith("RETIREMENT_RANK_READY:")
    ]
    assert {int(rank) for rank, _pid, _start in identities} == {0, 1, 2, 3}, output[-4000:]
    assert len(identities) == 4, output[-4000:]
    for _rank, pid, start_time in identities:
        state = _read_process_state(int(pid))
        assert state is None or state[0] == "Z" or state[1] != int(start_time), (
            f"Shared-job worker {pid} remains live after launcher exit: {state}"
        )
