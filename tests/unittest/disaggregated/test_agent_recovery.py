# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from tensorrt_llm._torch.disaggregation.native import transfer
from tensorrt_llm._torch.disaggregation.nixl._agent_cpp import (
    BindingsNixlTransferStatus,
    TransferState,
)
from tensorrt_llm._torch.disaggregation.transceiver import KvCacheTransceiverV2

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def worker(monkeypatch):
    sender = object.__new__(transfer.Sender)
    sender.__dict__.update(
        _agent=Mock(),
        shutdown=Mock(),  # No sender sockets or worker threads are started by this fixture.
        _agent_failed=True,
        _agent_generation=0,
        _context_first_ownership=False,
        _enforce_physical_ownership=False,
        _admission_lock=threading.Lock(),
        _agent_operation_gate=transfer._AgentOperationGate(),
        _recovery_fatal_callback=Mock(),
        _retirement_timeout_s=1,
        _recover_agent=Mock(),
        _loaded_remote_agents_lock=threading.Lock(),
        _loaded_remote_agents={"decode11"},
        _remote_agent_descriptors={"decode11": b"peer"},
        _deferred_remote_agents={},
        _unreleased_statuses=[],
        _shutdown_requested=False,
    )
    worker = object.__new__(transfer.TransferWorker)
    worker.__dict__.update(
        shutdown=Mock(),
        _agent=sender._agent,
        _sender=sender,
        _receiver=SimpleNamespace(
            _agent=sender._agent,
            _sessions_lock=threading.Lock(),
            _sessions={},
            _sender_ep_instance_map={"ctx": object()},
        ),
        _config=SimpleNamespace(
            device_id=0,
            agent_buffer_size_mb=17,
            agent_bounce_params=None,
            kv_cache_manager=SimpleNamespace(mapping=SimpleNamespace(rank=0, world_size=1)),
        ),
        _rank_info=SimpleNamespace(
            instance_name="ctx", instance_rank=0, transfer_engine_info=b"old"
        ),
        _registered_mem=[object(), object()],
        _agent_recovery_lock=threading.Lock(),
        _agent_recovery_stop=threading.Event(),
        _agent_recovery_thread=None,
    )
    replacement = Mock()
    replacement.get_local_agent_desc.return_value = b"new"
    monkeypatch.setattr(transfer, "_create_nixl_agent", Mock(return_value=replacement))
    monkeypatch.setattr(transfer.torch.cuda, "set_device", Mock())
    monkeypatch.setattr(transfer.cudart, "cudaSetDevice", Mock(return_value=0))
    monkeypatch.setattr(transfer, "CUASSERT", Mock())
    return worker


@pytest.fixture
def task():
    return transfer.SendTaskBase(transfer.DisaggregatedParams(disagg_request_id=73))


def submit(sender, task, generation, peer="decode11"):
    request = SimpleNamespace(op=transfer.TransferOp.WRITE, remote_name=peer, dst_descs=object())
    return sender._submit_transfer(task, 11, request, agent_generation=generation)


@pytest.mark.parametrize("released", [True, False])
def test_peer_error_recovers_only_when_native_retirement_fails(worker, task, released):
    sender, old = worker._sender, worker._agent
    sender._agent_failed = False
    status = Mock(in_progress=False)
    status.wait.return_value, status.release.return_value = False, released
    status.last_status_str.return_value = "peer disconnected"
    old.submit_transfer_requests.return_value = status
    sender._recover_agent.side_effect = worker._recover_transfer_agent

    def shutdown():
        assert task._native_inflight == 1

    old.shutdown.side_effect = shutdown
    assert not submit(sender, task, 0)[0]
    assert task._native_inflight == 0 and not sender._unreleased_statuses
    assert not sender._agent_failed
    assert sender._agent_generation == int(not released)
    assert old.shutdown.call_count == int(not released)
    if not released:
        replacement = transfer._create_nixl_agent.return_value
        transfer._create_nixl_agent.assert_called_once_with(
            "ctx0", rank=0, world_size=1, agent_buffer_size_mb=17, agent_bounce_params=None
        )
        assert replacement.register_memory.call_args_list == [
            call(desc) for desc in worker._registered_mem
        ]
        assert worker._agent is sender._agent is worker._receiver._agent is replacement
        assert worker._rank_info.transfer_engine_info == b"new"
        assert not worker._receiver._sender_ep_instance_map
        assert worker._recover_transfer_agent(0)  # A stale callback cannot rebuild again.
        assert not submit(sender, task, 0)[0]
    sender._agent.submit_transfer_requests.return_value = Mock(
        in_progress=False, wait=Mock(return_value=True), release=Mock(return_value=True)
    )
    assert submit(sender, task, sender._agent_generation) == (True, None)


@pytest.mark.parametrize("owner", ["receive", "write"])
def test_recovery_retains_owners_until_they_are_physically_retired(worker, task, owner):
    sender, old = worker._sender, worker._agent
    status, retired = Mock(in_progress=True), threading.Event()
    if owner == "receive":
        worker._receiver._sessions[73] = object()
    else:
        task._native_inflight = 1
        sender._unreleased_statuses = [(status, task, retired)]
    assert not worker._recover_transfer_agent(0)
    old.shutdown.assert_not_called()
    transfer._create_nixl_agent.assert_not_called()
    if owner == "write":
        assert task._native_inflight == 1 and not retired.is_set()
    worker._receiver._sessions.clear()
    status.in_progress = False
    assert worker._recover_transfer_agent(0)
    assert task._native_inflight == 0 and not sender._agent_failed
    if owner == "write":
        assert retired.is_set()


@pytest.mark.parametrize("stage", ["cuda", "construct", "register", "late", "receive", "write"])
def test_recovery_deadline_preserves_roots_and_keeps_admission_fenced(
    worker, task, monkeypatch, stage
):
    sender, old = worker._sender, worker._agent
    factory, now = transfer._create_nixl_agent, [0.0]
    replacement, retired = factory.return_value, threading.Event()
    monkeypatch.setattr(transfer, "time", SimpleNamespace(monotonic=lambda: now[0]))

    def expire(*_args, **_kwargs):
        now[0] = 2.0
        if stage not in ("late", "receive", "write"):
            raise RuntimeError("setup failed")
        return b"new"

    if stage in ("cuda", "receive", "write"):
        transfer.torch.cuda.set_device.side_effect = expire
    elif stage == "construct":
        factory.side_effect = expire
    elif stage == "register":
        replacement.register_memory.side_effect = expire
    else:
        replacement.get_local_agent_desc.side_effect = expire
    if stage == "receive":
        worker._receiver._sessions[73] = object()
    elif stage == "write":
        task._native_inflight = 1
        sender._unreleased_statuses = [(Mock(in_progress=True), task, retired)]
    worker._request_agent_recovery(0)
    thread = worker._agent_recovery_thread
    try:
        thread.join(timeout=2)
        assert not thread.is_alive()
        sender._recovery_fatal_callback.assert_called_once()
        assert sender._agent_failed and sender._agent_generation == 0
        assert worker._registered_mem and worker._rank_info.transfer_engine_info == b"old"
        if stage in ("cuda", "receive", "write"):
            factory.assert_not_called()
            old.shutdown.assert_not_called()
        if stage == "register":
            replacement.register_memory.assert_called_once()
        if stage in ("register", "late"):
            replacement.shutdown.assert_called_once()
        if stage == "write":
            assert task._native_inflight == 1 and not retired.is_set()
    finally:
        worker._agent_recovery_stop.set()
        thread.join(timeout=2)


def test_failed_peer_replay_retries_without_new_registration(worker, task, monkeypatch):
    sender = worker._sender
    now = [0.0]
    monkeypatch.setattr(transfer, "time", SimpleNamespace(monotonic=lambda: now[0]))
    sender._remote_agent_descriptors["dead0"] = b"stale"
    sender._loaded_remote_agents.add("dead0")
    replacement = transfer._create_nixl_agent.return_value

    def load(name, descriptor):
        if name == "dead0":
            raise RuntimeError("peer unavailable")

    replacement.load_remote_agent.side_effect = load
    assert worker._recover_transfer_agent(0)
    assert sender._loaded_remote_agents == {"decode11"}
    replacement.load_remote_agent.reset_mock()
    replacement.check_remote_descs.side_effect = (
        lambda name, _: name in sender._loaded_remote_agents
    )
    replacement.submit_transfer_requests.return_value = Mock(
        in_progress=False, wait=Mock(return_value=True), release=Mock(return_value=True)
    )
    assert submit(sender, task, 1) == (True, None)
    replacement.load_remote_agent.assert_not_called()
    for _ in range(2):  # Repeated demands obey the per-peer retry backoff.
        with pytest.raises(transfer._TransferNotSubmittedError):
            submit(sender, task, 1, "dead0")
    replacement.load_remote_agent.assert_called_once_with("dead0", b"stale")
    replacement.load_remote_agent.side_effect = None
    now[0] = 2.0
    assert submit(sender, task, 1, "dead0") == (True, None)
    assert not sender._deferred_remote_agents and not sender._agent_failed


@pytest.mark.parametrize(
    "terminal,stop",
    [(TransferState.SUCCESS, False), (TransferState.FAILURE, False), (TransferState.SUCCESS, True)],
)
def test_native_wait_is_bounded_and_does_not_retire_interrupted_handles(terminal, stop):
    native = Mock()
    native.wait.side_effect = [TransferState.IN_PROGRESS, terminal]
    status = BindingsNixlTransferStatus(native)
    assert status.wait(timeout_ms=1000, stop_requested=lambda: stop) == (
        not stop and terminal == TransferState.SUCCESS
    )
    assert native.wait.call_count == (0 if stop else 2)
    assert all(0 <= call.args[0] <= 100 for call in native.wait.call_args_list)
    assert status.in_progress is stop


def test_fenced_agent_rejects_new_receive_publications(worker):
    receiver = object.__new__(transfer.Receiver)
    receiver.shutdown = Mock()
    receiver.__dict__.update(
        _sessions_lock=threading.Lock(),
        _sessions={},
        _shutdown=False,
        _recovery_sender=weakref.ref(worker._sender),
    )
    session = SimpleNamespace(_enforce_physical_ownership=False, disagg_request_id=73)
    with pytest.raises(RuntimeError, match="receive admission is closed"):
        receiver.setup_session(session)
    assert not receiver._sessions


@pytest.mark.parametrize("side", ["gen", "ctx"])
def test_terminal_sets_are_built_once_per_poll(side):
    class CountedIntents(set):
        unions = 0

        def __or__(self, other):
            self.unions += 1
            return super().__or__(other)

    count = 1024  # Detect repeated construction without a timing assertion.
    intents = CountedIntents({count})
    node = object.__new__(KvCacheTransceiverV2)
    node.__dict__.update(
        _agent_recovery=True,
        _gen_idle_poll_skip=False,
        _gen_need_sync=False,
        _ctx_need_tp_sync=False,
        _ctx_need_pp_sync=False,
        _recv_sessions=dict.fromkeys(range(count + 2)),
        _send_sessions=dict.fromkeys(range(count + 2)),
        _dist=SimpleNamespace(tp_allgather=None),
        _gen_allgather=None,
        _failure_intents=lambda *_: intents,
        _cancel_failed_transfers=lambda *_: None,
        _retirement_blocked=lambda session: False,
        _local_agent_ready=lambda: True,
    )

    def vote(*args, **kwargs):
        assert args[6] == list(range(count + 1))
        assert intents.unions <= 1
        return (
            ([], [], [], args[6], set(), {count}, True)
            if side == "gen"
            else ([], [], [], args[6], {count}, True)
        )

    node._consensus_outcome = vote
    getattr(node, f"_{side}_consensus_outcome")([], [0], [1], list(range(2, count)), [])
