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

pytestmark = pytest.mark.cpu_only


def make_sender(agent):
    sender = object.__new__(transfer.Sender)
    sender._agent = agent
    sender._context_first_ownership = False
    sender._agent_generation = 0
    sender._agent_failed = False
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
        # The mocked recovery callback runs only after the request is rooted.
        assert deadline.wait(2)
        assert not pending.done()
        assert task._native_inflight == 1
        assert sender._unreleased_statuses[0][:2] == (status, task)
        with patch.object(transfer, "_create_nixl_agent", return_value=replacement):
            assert worker._recover_transfer_agent(0)
        assert pending.result(timeout=2)[0] is False
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
                failure_reported.set()
    old_agent.shutdown.assert_called_once()
    assert task._native_inflight == 0
    assert sender._agent_generation == 1
    assert not sender._agent_failed


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

    with patch.object(transfer, "_create_nixl_agent", side_effect=create_agent):
        with ThreadPoolExecutor(max_workers=1) as pool:
            with worker._sender._agent_operation_gate.transfer():
                future = pool.submit(worker._recover_transfer_agent, 0)
                assert not entered.wait(0.05)
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


def test_missing_remote_descriptors_trigger_recovery_before_submission():
    agent = Mock()
    agent.check_remote_descs.return_value = False
    sender = make_sender(agent)
    assert sender._submit_transfer(make_task(), 11, make_request(), agent_generation=0)[0] is False
    agent.submit_transfer_requests.assert_not_called()
    sender._recover_agent.assert_called_once_with(0)
