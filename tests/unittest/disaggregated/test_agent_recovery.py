# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.disaggregation.native import transfer
from tensorrt_llm._torch.disaggregation.nixl._agent_cpp import (
    BindingsNixlTransferAgent,
    BindingsNixlTransferStatus,
    TransferState,
)
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def prevent_unexpected_process_exit():
    with patch.object(
        transfer.hang_detector,
        "propagate_hard_kill",
        side_effect=AssertionError("unexpected hard stop"),
    ):
        yield


def make_sender(agent):
    sender = object.__new__(transfer.Sender)
    sender._agent = agent
    sender._context_first_ownership = False
    sender._agent_generation = 0
    sender._agent_failed = False
    sender._retirement_timeout_s = 10
    sender._enforce_physical_ownership = False
    sender._recover_agent = Mock()
    sender._agent_operation_gate = transfer._AgentOperationGate()
    sender._loaded_remote_agents_lock = threading.Lock()
    sender._remote_agent_descriptors = {"decode11": b"receiver-descriptor"}
    sender._unreleased_statuses = []
    sender._shutdown_requested = False
    return sender


def make_worker():
    agent = Mock()
    agent.get_local_agent_desc.return_value = b"old-descriptor"
    worker = object.__new__(transfer.TransferWorker)
    worker._agent = agent
    worker._sender = make_sender(agent)
    worker._receiver = SimpleNamespace(
        _agent=agent,
        _sessions_lock=threading.Lock(),
        _sessions={},
        _sender_ep_instance_map={"prefill": object()},
    )
    worker._config = SimpleNamespace(
        device_id=0, kv_cache_manager=SimpleNamespace(mapping=SimpleNamespace(rank=5, world_size=8))
    )
    worker._rank_info = SimpleNamespace(
        instance_name="prefill", instance_rank=5, transfer_engine_info=b"old-descriptor"
    )
    worker._registered_mem = [object(), object()]
    worker._agent_recovery_lock = threading.Lock()
    worker._agent_recovery_stop = threading.Event()
    worker._agent_recovery_thread = None
    worker._sender._agent_failed = True
    return worker


def make_task():
    return SimpleNamespace(
        lock=threading.Lock(), _native_inflight=0, _enforce_physical_ownership=None
    )


def make_request():
    return SimpleNamespace(op=transfer.TransferOp.WRITE, remote_name="decode11", dst_descs=object())


def test_failed_write_schedules_recovery_and_fences_further_submissions():
    status = Mock()
    status.wait.return_value = False
    status.last_status_str.return_value = "NIXL_ERR_REMOTE_DISCONNECT"
    status.release.return_value = True
    agent = Mock()
    agent.submit_transfer_requests.return_value = status
    sender = make_sender(agent)
    task = make_task()

    assert sender._submit_transfer(task, 11, make_request(), agent_generation=0) == (
        False,
        "NIXL_ERR_REMOTE_DISCONNECT",
    )
    assert task._native_inflight == 0
    assert sender._agent_failed
    sender._recover_agent.assert_called_once_with(0)
    assert sender._submit_transfer(task, 11, make_request(), agent_generation=0)[0] is False
    agent.submit_transfer_requests.assert_called_once()


def test_old_generation_write_is_never_submitted():
    sender = make_sender(Mock())
    sender._agent_generation = 1
    sender._get_session = Mock(return_value=SimpleNamespace(_agent_generation=0))
    sender._num_threads = 1
    sender._send_task_queues = [Mock()]
    meta = transfer.WriteMeta(make_task(), 1, "decode", 11, "tcp://peer", 42, None, None, None)
    assert meta.agent_generation is None
    sender._enqueue(meta)
    assert (
        sender._submit_transfer(
            meta.task, 11, make_request(), agent_generation=meta.agent_generation
        )[0]
        is False
    )
    sender._agent.submit_transfer_requests.assert_not_called()
    sender._recover_agent.assert_not_called()


def test_native_submission_without_a_handle_is_a_request_error():
    with patch("tensorrt_llm._torch.disaggregation.nixl._agent_cpp.CppNixlTransferAgent") as native:
        native.return_value.submit_transfer_requests.return_value = None
        agent = BindingsNixlTransferAgent("test")
        with pytest.raises(RuntimeError, match="could not create a transfer request"):
            agent.submit_transfer_requests(object())


def test_failed_release_retains_physical_owner_until_agent_shutdown():
    worker = make_worker()
    sender = worker._sender
    sender._agent_failed = False
    status = Mock()
    status.wait.return_value = False
    status.release.return_value = False
    sender._agent.submit_transfer_requests.return_value = status
    task = make_task()

    old_agent = worker._agent

    def shutdown():
        assert task._native_inflight == 1

    old_agent.shutdown.side_effect = shutdown
    replacement = Mock()
    replacement.get_local_agent_desc.return_value = b"new-descriptor"
    deadline = threading.Event()
    sender._recover_agent.side_effect = lambda _generation: deadline.set()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(sender._submit_transfer, task, 11, make_request(), agent_generation=0)
        try:
            # The mocked recovery callback runs only after the request is rooted.
            assert deadline.wait(2)
            assert not pending.done()
            assert task._native_inflight == 1
            assert sender._unreleased_statuses[0][:2] == (status, task)
            with patch.object(transfer, "_create_nixl_agent", return_value=replacement):
                assert worker._recover_transfer_agent(0)
            assert pending.result(timeout=2)[0] is False
        finally:
            for _status, _task, retired in sender._unreleased_statuses:
                retired.set()
            pending.result(timeout=2)
    assert task._native_inflight == 0
    assert sender._unreleased_statuses == []


def test_stalled_sibling_drains_for_recovery():
    worker = make_worker()
    sender = worker._sender
    sender._agent_failed = False
    entered, failure_reported = threading.Event(), threading.Event()

    def stalled_wait(timeout_ms):
        assert 0 <= timeout_ms <= 100
        entered.set()
        assert failure_reported.wait(2)
        return TransferState.IN_PROGRESS

    stalled_native, failed_native = Mock(), Mock()
    stalled_native.wait.side_effect = stalled_wait
    failed_native.wait.return_value = TransferState.FAILURE
    for native in (stalled_native, failed_native):
        native.release.return_value = True
    sender._agent.submit_transfer_requests.side_effect = [
        BindingsNixlTransferStatus(stalled_native),
        BindingsNixlTransferStatus(failed_native),
    ]

    def recover(generation):
        failure_reported.set()
        worker._recover_transfer_agent(generation)

    sender._recover_agent.side_effect = recover
    old_agent = worker._agent
    task = make_task()

    def shutdown():
        assert task._native_inflight == 1

    old_agent.shutdown.side_effect = shutdown
    replacement = Mock()
    replacement.get_local_agent_desc.return_value = b"new-descriptor"
    with patch.object(transfer, "_create_nixl_agent", return_value=replacement):
        with ThreadPoolExecutor(max_workers=1) as pool:
            stalled = pool.submit(
                sender._submit_transfer, task, 11, make_request(), agent_generation=0
            )
            try:
                assert entered.wait(2)
                assert not sender._submit_transfer(
                    make_task(), 11, make_request(), agent_generation=0
                )[0]
                assert not stalled.result(timeout=2)[0]
            finally:
                if not stalled.done():
                    sender._agent_failed = True
                failure_reported.set()
    old_agent.shutdown.assert_called_once()
    assert task._native_inflight == 0
    assert sender._agent_generation == 1


@pytest.mark.parametrize("terminal", [TransferState.SUCCESS, TransferState.FAILURE])
def test_interruptible_wait_polls_until_terminal(terminal):
    native = Mock()
    native.wait.side_effect = [TransferState.IN_PROGRESS, terminal]
    status = BindingsNixlTransferStatus(native)
    assert status.wait(stop_requested=lambda: False) == (terminal == TransferState.SUCCESS)
    assert native.wait.call_count == 2
    assert all(call.args == (100,) for call in native.wait.call_args_list)


def test_concurrent_errors_rebuild_one_agent_and_restore_registered_regions():
    worker = make_worker()
    replacement = Mock()
    replacement.get_local_agent_desc.return_value = b"new-descriptor"
    old_agent = worker._agent
    with patch.object(transfer, "_create_nixl_agent", return_value=replacement) as create:
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert all(pool.map(worker._recover_transfer_agent, [0, 0]))
    create.assert_called_once_with("prefill5", rank=5, world_size=8)
    old_agent.shutdown.assert_called_once()
    assert replacement.register_memory.call_count == 2
    replacement.load_remote_agent.assert_called_once_with("decode11", b"receiver-descriptor")
    assert worker._agent is worker._sender._agent is worker._receiver._agent is replacement
    assert worker._rank_info.transfer_engine_info == b"new-descriptor"
    assert worker._receiver._sender_ep_instance_map == {}
    assert worker._sender._agent_generation == 1
    assert not worker._sender._agent_failed


def test_recovery_waits_for_active_operations_and_receive_publications():
    worker = make_worker()
    worker._receiver._sessions[42] = object()
    with patch.object(transfer, "_create_nixl_agent") as create:
        assert not worker._recover_transfer_agent(0)
        create.assert_not_called()
        worker._agent.shutdown.assert_not_called()

    worker._receiver._sessions.clear()
    entered = threading.Event()
    replacement = Mock()
    replacement.get_local_agent_desc.return_value = b"new-descriptor"

    def create_agent(*args, **kwargs):
        entered.set()
        return replacement

    metadata_waiting = threading.Event()
    gate = worker._sender._agent_operation_gate
    wait = gate._condition.wait

    def observed_wait(*args, **kwargs):
        metadata_waiting.set()
        return wait(*args, **kwargs)

    with (
        patch.object(gate._condition, "wait", side_effect=observed_wait),
        patch.object(transfer, "_create_nixl_agent", side_effect=create_agent),
    ):
        with ThreadPoolExecutor(max_workers=1) as pool:
            with worker._sender._agent_operation_gate.transfer():
                future = pool.submit(worker._recover_transfer_agent, 0)
                assert metadata_waiting.wait(2)
                assert not entered.is_set()
            assert future.result(timeout=2)


def test_failed_agent_creation_keeps_sends_fenced_and_does_not_raise():
    worker = make_worker()
    with patch.object(transfer, "_create_nixl_agent", side_effect=RuntimeError("no UAR")):
        assert not worker._recover_transfer_agent(0)
    assert worker._sender._agent_failed
    assert worker._sender._agent_generation == 0
    assert worker._agent is None


def test_logical_task_failure_is_not_native_retirement():
    session = object.__new__(transfer.TxSession)
    task = SimpleNamespace(status=transfer.TaskStatus.ERROR, _native_inflight=1)
    session._enforce_physical_ownership = False
    session.kv_tasks = [task]
    session.aux_task = None
    assert not session.resources_drained()
    assert session._failed_wait_result() is None
    task._native_inflight = 0
    assert session.resources_drained()
    assert session._failed_wait_result() == transfer.WaitResult.FAILED


def test_late_recovery_callback_does_not_restart_a_healthy_generation():
    worker = make_worker()
    worker._sender._agent_generation = 1
    worker._sender._agent_failed = False
    worker._request_agent_recovery(0)
    assert worker._agent_recovery_thread is None


def test_grouped_submissions_retire_each_handle_before_recovery():
    first, second = Mock(), Mock()
    first.wait.return_value, first.release.return_value = True, True
    second.wait.return_value, second.release.return_value = False, True
    second.last_status_str.return_value = "NIXL_ERR_REMOTE_DISCONNECT"
    agent = Mock()

    def submit(_request):
        if first.release.called:
            return second
        return first

    agent.submit_transfer_requests.side_effect = submit
    sender = make_sender(agent)
    submitted = Mock()
    request = transfer._TransferRequestGroup((make_request(), make_request()))
    assert (
        sender._submit_transfer(
            make_task(), 11, request, on_submitted=submitted, agent_generation=0
        )[0]
        is False
    )
    first.release.assert_called_once()
    second.release.assert_called_once()
    submitted.assert_called_once()
    sender._recover_agent.assert_called_once_with(0)


def test_missing_remote_descriptors_fail_only_the_request():
    agent = Mock()
    agent.check_remote_descs.return_value = False
    sender = make_sender(agent)
    task = make_task()
    with pytest.raises(transfer._TransferNotSubmittedError):
        sender._submit_transfer(task, 11, make_request(), agent_generation=0)
    agent.submit_transfer_requests.assert_not_called()
    sender._recover_agent.assert_not_called()
    assert not sender._agent_failed
    assert task._native_inflight == 0


def test_group_reserves_native_ownership_between_members():
    first, second = Mock(), Mock()
    for status in (first, second):
        status.wait.return_value = status.release.return_value = True
    agent = Mock()
    agent.submit_transfer_requests.side_effect = [first, second]
    sender = make_sender(agent)
    task = make_task()
    task.status = transfer.TaskStatus.TRANSFERRING
    session = object.__new__(transfer.TxSession)
    session._enforce_physical_ownership = False
    session.kv_tasks, session.aux_task = [task], None
    session._closed, session.lock = False, threading.Lock()
    session._sender = session._aux_buffer = None

    class Members(tuple):
        def __iter__(self):
            yield self[0]
            if first.wait.called:
                # An auxiliary failure can mark this KV task ERROR here.
                task.status = transfer.TaskStatus.ERROR
                assert not session.resources_drained()
                assert session.close() is False
            yield self[1]

    assert sender._submit_direct_transfer(
        task, Members((make_request(), make_request())), None, 0
    ) == (True, None)
    assert task._native_inflight == 0
    sender._recover_agent.assert_not_called()


@pytest.mark.parametrize("active_receive", [False, True])
def test_unretired_handle_has_a_bounded_fail_stop_without_freeing_roots(active_receive):
    worker = make_worker()
    sender = worker._sender
    sender._agent_failed = False
    sender._retirement_timeout_s = 0
    status = Mock()
    status.wait.return_value = status.release.return_value = False
    sender._agent.submit_transfer_requests.return_value = status
    if active_receive:
        worker._receiver._sessions[42] = object()
    else:
        worker._agent.shutdown.side_effect = RuntimeError("backend teardown failed")
    sender._recover_agent.side_effect = worker._recover_transfer_agent
    task = make_task()
    with patch.object(
        transfer.hang_detector, "propagate_hard_kill", side_effect=SystemExit
    ) as kill:
        with pytest.raises(SystemExit):
            sender._submit_transfer(task, 11, make_request(), agent_generation=0)
    kill.assert_called_once_with(diagnostics=False)
    assert task._native_inflight == 1
    assert sender._unreleased_statuses[0][:2] == (status, task)
    assert not sender._unreleased_statuses[0][2].is_set()


@pytest.mark.parametrize(
    "already_stopped,last_poll_at", [(False, 0.24), (False, 0.25), (True, 0.24)]
)
def test_interruptible_wait_obeys_deadline_and_initial_stop(already_stopped, last_poll_at):
    native = Mock()
    native.wait.return_value = TransferState.IN_PROGRESS
    status = BindingsNixlTransferStatus(native)
    # Initial deadline, first poll, first completion check, final poll/check.
    with patch(
        "tensorrt_llm._torch.disaggregation.nixl._agent_cpp.time.monotonic",
        side_effect=[0.0, 0.0, 0.2, last_poll_at, 0.25],
    ):
        assert not status.wait(timeout_ms=250, stop_requested=lambda: already_stopped)
    if already_stopped:
        native.wait.assert_not_called()
    else:
        assert native.wait.call_count == 2
        assert native.wait.call_args_list[0].args == (100,)
        last_poll_ms = native.wait.call_args_list[-1].args[0]
        assert 0 <= last_poll_ms <= 10
        if last_poll_at == 0.25:
            assert last_poll_ms == 0


@pytest.mark.parametrize("retry_succeeds", [False, True])
def test_background_recovery_coalesces_requests_retries_and_stops(retry_succeeds):
    worker = make_worker()
    attempted, allow_failure = threading.Event(), threading.Event()
    replacement = Mock()
    replacement.get_local_agent_desc.return_value = b"new-descriptor"

    def create(*args, **kwargs):
        if not attempted.is_set():
            attempted.set()
            assert allow_failure.wait(2)
            raise RuntimeError("transient creation failure")
        return replacement

    with (
        patch.object(transfer.torch.cuda, "set_device"),
        patch.object(transfer.cudart, "cudaSetDevice", return_value=0),
        patch.object(transfer, "CUASSERT"),
        patch.object(transfer, "_create_nixl_agent", side_effect=create) as factory,
    ):
        worker._request_agent_recovery(0)
        thread = worker._agent_recovery_thread
        try:
            assert attempted.wait(2)
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(worker._request_agent_recovery, [0, 0]))
            assert worker._agent_recovery_thread is thread
            if not retry_succeeds:
                worker._agent_recovery_stop.set()
            allow_failure.set()
            thread.join(timeout=3)
            assert not thread.is_alive()
            assert factory.call_count == (2 if retry_succeeds else 1)
            assert worker._sender._agent_generation == int(retry_succeeds)
        finally:
            worker._agent_recovery_stop.set()
            allow_failure.set()
            thread.join(timeout=3)


def test_transceiver_shutdown_can_retire_a_failed_release_before_session_close():
    worker = make_worker()
    sender = worker._sender
    sender._agent_failed = False
    status = Mock()
    status.wait.return_value = status.release.return_value = False
    sender._agent.submit_transfer_requests.return_value = status
    task = make_task()
    task.status = transfer.TaskStatus.ERROR
    rooted, allow_shutdown = threading.Event(), threading.Event()
    sender._recover_agent.side_effect = lambda _generation: rooted.set()
    worker._agent.shutdown.side_effect = lambda: allow_shutdown.wait(2)
    worker.shutdown = Mock()
    session = object.__new__(transfer.TxSession)
    session._enforce_physical_ownership = False
    session.kv_tasks, session.aux_task = [task], None
    session._closed, session.lock = False, threading.Lock()
    session._sender = session._aux_buffer = None
    transceiver = object.__new__(KvCacheTransceiverV2)
    transceiver._transfer_worker = worker
    transceiver._send_sessions, transceiver._send_reqs = {7: session}, {}
    transceiver._recv_sessions, transceiver._recv_reqs = {}, {}
    transceiver._pending_rank_info_admissions = None
    with (
        patch.object(transfer.torch.cuda, "set_device"),
        patch.object(transfer.cudart, "cudaSetDevice", return_value=0),
        patch.object(transfer, "CUASSERT"),
        patch.object(transfer, "_create_nixl_agent") as factory,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        pending = pool.submit(sender._submit_transfer, task, 11, make_request(), agent_generation=0)
        try:
            assert rooted.wait(2)
            sender._recover_agent = worker._request_agent_recovery
            with pytest.raises(RuntimeError, match="physical resources remain active"):
                transceiver.shutdown()
            thread = worker._agent_recovery_thread
            allow_shutdown.set()
            assert pending.result(timeout=2)[0] is False
            thread.join(timeout=2)
            assert not thread.is_alive()
            transceiver.shutdown()
            assert transceiver._shutdown_complete
            factory.assert_not_called()
            worker.shutdown.assert_called_once()
            assert worker._agent is None
        finally:
            allow_shutdown.set()
            worker._agent_recovery_stop.set()
            for _status, _task, retired in sender._unreleased_statuses:
                retired.set()
            pending.result(timeout=2)
