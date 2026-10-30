# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only ordering tests for request-arrival notification."""

import copy
import datetime
import queue
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.idle_notification import IdleRequestNotification
from tensorrt_llm._torch.pyexecutor.idle_notification_runtime import (
    IdleNotificationRuntime,
    _validate_progress_modes,
)
from tensorrt_llm._torch.pyexecutor.request_utils import RequestBroadcaster

pytestmark = pytest.mark.cpu_only


def test_ready_requests_keep_one_count_and_do_not_notify() -> None:
    send, receive, fetch = Mock(), Mock(), Mock()
    broadcast = Mock(return_value=1)
    requests = [object()]
    gate = IdleRequestNotification(send, receive)

    result, count = gate.count(0, requests, broadcast, fetch)

    assert result is requests
    assert count == 1
    broadcast.assert_called_once_with(1)
    send.assert_not_called()
    receive.assert_not_called()
    fetch.assert_not_called()


def test_leader_announces_idle_before_fetch_then_notifies() -> None:
    operations = []

    def broadcast(count: int) -> int:
        operations.append(("count", count))
        return count

    def fetch() -> list:
        operations.append(("fetch",))
        return [object()]

    gate = IdleRequestNotification(lambda notice: operations.append(("notice", notice)), Mock())
    requests, count = gate.count(0, [], broadcast, fetch)

    assert len(requests) == count == 1
    assert operations == [("count", -1), ("fetch",), ("notice", (0, True)), ("count", 1)]


def test_follower_receives_notice_before_resuming_collective() -> None:
    operations = []
    counts = iter([-1, 1])

    def broadcast(count: int) -> int:
        operations.append("count")
        return next(counts)

    def receive() -> tuple[int, bool]:
        operations.append("notice")
        return 0, True

    gate = IdleRequestNotification(Mock(), receive)

    assert gate.count(1, [], broadcast, None) == ([], 1)
    assert operations == ["count", "notice", "count"]


def test_failed_leader_fetch_notifies_followers_without_error_content() -> None:
    send = Mock()

    def fetch() -> list:
        raise RuntimeError("private request failure detail")

    gate = IdleRequestNotification(send, Mock())
    with pytest.raises(RuntimeError, match="private request failure detail"):
        gate.count(0, [], lambda count: count, fetch)
    send.assert_called_once_with((0, False))


@pytest.mark.parametrize("notice", [(1, True), (0, False)])
def test_stale_epoch_or_fetch_failure_does_not_enter_next_count(notice) -> None:
    broadcast = Mock(return_value=-1)
    gate = IdleRequestNotification(Mock(), lambda: notice)

    with pytest.raises((ValueError, RuntimeError)):
        gate.count(1, [], broadcast, None)
    broadcast.assert_called_once()


def test_empty_fetch_and_next_epoch_remain_ordered() -> None:
    send = Mock()
    gate = IdleRequestNotification(send, Mock())

    assert gate.count(0, [], lambda count: count, lambda: []) == ([], 0)
    assert gate.count(0, [], lambda count: count, lambda: []) == ([], 0)
    assert [call.args[0] for call in send.call_args_list] == [(0, True), (1, True)]


def test_request_broadcaster_defers_fetch_and_preserves_payload_path() -> None:
    broadcaster = RequestBroadcaster(
        SimpleNamespace(rank=0, world_size=2, has_pp=False, supports_single_round_broadcast=False),
        SimpleNamespace(pause=nullcontext),
    )
    send = Mock()
    broadcaster.idle_notification = IdleRequestNotification(send, Mock())
    broadcaster._broadcast_request_count = Mock(side_effect=lambda count, **kwargs: count)
    broadcaster._collect_py_objects = Mock(return_value=None)
    broadcaster._broadcast_requests = Mock()
    requests = [object()]

    result, objects = broadcaster.broadcast([], prefer_cpu=True, idle_fetch=lambda: requests)

    assert result is requests
    assert objects is None
    assert [call.args[0] for call in broadcaster._broadcast_request_count.call_args_list] == [-1, 1]
    broadcaster._broadcast_requests.assert_called_once_with(requests, None)
    send.assert_called_once_with((0, True))


def test_active_single_round_path_does_not_use_notification() -> None:
    broadcaster = RequestBroadcaster(
        SimpleNamespace(rank=0, world_size=2, has_pp=False, supports_single_round_broadcast=True),
        SimpleNamespace(pause=nullcontext),
    )
    broadcaster.idle_notification = Mock()
    requests = [object()]
    broadcaster._broadcast_single_round = Mock(return_value=(requests, None))

    assert broadcaster.broadcast(requests) == (requests, None)
    broadcaster.idle_notification.count.assert_not_called()


def _ordinary_executor() -> SimpleNamespace:
    return SimpleNamespace(
        dist=SimpleNamespace(world_size=2, tp_size=2, pp_size=1, cp_size=1),
        enable_attention_dp=False,
        _disable_mpi=False,
        _resource_governor_enabled=False,
        enable_kv_pool_rebalance=False,
        is_benchmark_disagg=False,
        kv_cache_transceiver=None,
        kv_connector_manager=None,
        drafter=None,
        draft_model_engine=None,
        model_engine=SimpleNamespace(
            spec_config=None, _mm_item_scheduler=None, mm_encoder_item_scheduling_enabled=False
        ),
        llm_args=SimpleNamespace(sleep_config=None),
        executor_request_queue=SimpleNamespace(batch_wait_timeout_ms=0),
        _gen_admission=SimpleNamespace(enabled=False),
        request_broadcaster=SimpleNamespace(sync_clock=False),
        _first_token_timer=None,
        _mm_encoder_item_scheduling_enabled=False,
    )


def test_supported_progress_modes_are_admitted() -> None:
    _validate_progress_modes(_ordinary_executor())


@pytest.mark.parametrize("tp_size,cp_size", [(1, 2), (2, 2), (4, 2)])
@pytest.mark.parametrize("rebalance", [False, True])
def test_non_pipelined_context_parallel_and_rebalance_modes_are_admitted(
    tp_size, cp_size, rebalance
) -> None:
    executor = _ordinary_executor()
    executor.dist.tp_size = tp_size
    executor.dist.cp_size = cp_size
    executor.dist.world_size = tp_size * cp_size
    executor.enable_kv_pool_rebalance = rebalance

    _validate_progress_modes(executor)

    assert executor.dist.cp_size == cp_size
    assert executor.enable_kv_pool_rebalance is rebalance


@pytest.mark.parametrize("tp_size,cp_size,pp_size,world_size", [(2, 2, 1, 2), (2, 1, 2, 4)])
def test_context_parallel_mapping_mismatch_and_pipeline_remain_rejected(
    tp_size, cp_size, pp_size, world_size
) -> None:
    executor = _ordinary_executor()
    executor.dist.tp_size = tp_size
    executor.dist.cp_size = cp_size
    executor.dist.pp_size = pp_size
    executor.dist.world_size = world_size

    with pytest.raises(ValueError, match="non-pipelined MPI TP/CP"):
        _validate_progress_modes(executor)


def test_context_parallel_broadcaster_uses_same_idle_count_and_payload_route() -> None:
    broadcaster = RequestBroadcaster(
        SimpleNamespace(
            rank=0,
            world_size=4,
            tp_size=2,
            cp_size=2,
            has_pp=False,
            supports_single_round_broadcast=False,
        ),
        SimpleNamespace(pause=nullcontext),
    )
    broadcaster.idle_notification = IdleRequestNotification(Mock(), Mock())
    broadcaster._broadcast_header = Mock(side_effect=lambda header, **kwargs: header)
    broadcaster._collect_py_objects = Mock(return_value=None)
    broadcaster._broadcast_requests = Mock()
    requests = [object()]

    result, objects = broadcaster.broadcast([], prefer_cpu=True, idle_fetch=lambda: requests)

    assert result is requests
    assert objects is None
    assert [call.args[0] for call in broadcaster._broadcast_header.call_args_list] == [[-1], [1]]
    broadcaster._broadcast_requests.assert_called_once_with(requests, None)


@pytest.mark.parametrize("public_pool", [False, True])
def test_builtin_ngram_is_inline_and_admitted_with_its_original_pool(public_pool) -> None:
    from tensorrt_llm._torch.speculative.ngram import NGramDrafter, NGramPoolManager
    from tensorrt_llm.llmapi import NGramDecodingConfig

    config = NGramDecodingConfig(max_draft_len=3, is_public_pool=public_pool)
    manager = NGramPoolManager(config, max_num_requests=8)
    drafter = NGramDrafter(config, manager)
    executor = _ordinary_executor()
    executor.model_engine.spec_config = config
    executor.drafter = drafter

    _validate_progress_modes(executor)
    drafter.prepare_draft_tokens(SimpleNamespace(generation_requests=[]))
    assert manager.pool == {}
    assert manager.start_index == {}

    request = SimpleNamespace(
        request_id=1,
        py_batch_idx=0,
        py_orig_prompt_len=4,
        py_max_new_tokens=4,
        get_tokens=lambda beam: [1, 2, 1, 2],
    )
    drafter.prepare_draft_tokens(SimpleNamespace(generation_requests=[request]))
    assert isinstance(request.py_draft_tokens, list)
    assert 1 in manager.start_index

    # Admit the real implementation before testing unsupported substitutions.
    executor.draft_model_engine = object()
    with pytest.raises(ValueError, match="built-in NGram"):
        _validate_progress_modes(executor)
    executor.draft_model_engine = None
    drafter.spec_config = NGramDecodingConfig(max_draft_len=3)
    with pytest.raises(ValueError, match="built-in NGram"):
        _validate_progress_modes(executor)
    drafter.spec_config = config

    class CustomPool(NGramPoolManager):
        pass

    drafter.spec_resource_manager = CustomPool(config, max_num_requests=8)
    with pytest.raises(ValueError, match="built-in NGram"):
        _validate_progress_modes(executor)
    drafter.spec_resource_manager = manager

    class CustomDrafter(NGramDrafter):
        pass

    executor.drafter = CustomDrafter(config, manager)
    with pytest.raises(ValueError, match="built-in NGram"):
        _validate_progress_modes(executor)


def _sa_progress_executor_fixture(monkeypatch, global_pool):
    """Real factory/registry objects with native allocation isolated for CPU QA.

    Zero-capacity metadata tests the admission join only. It is not evidence
    of initialized device buffers, native kernels or model correctness.
    """
    import torch

    from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager, ResourceManagerType
    from tensorrt_llm._torch.speculative import suffix_automaton
    from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
    from tensorrt_llm._torch.speculative.sa_worker import SASpecMetadata
    from tensorrt_llm._torch.speculative.utils import get_spec_worker
    from tensorrt_llm.llmapi import SADecodingConfig

    config = SADecodingConfig(max_draft_len=3, enable_global_pool=global_pool)
    selected_config = copy.deepcopy(config)  # Mirrors ModelLoader's normal meta-init path.

    def cpu_base(worker, *args, **kwargs):
        torch.nn.Module.__init__(worker)
        worker._auxiliary_state_handlers = []

    monkeypatch.setattr(SpecWorkerBase, "__init__", cpu_base)
    worker = get_spec_worker(selected_config, model_config=None, mapping=None)
    native = Mock()
    native.get_state_size.return_value = 32
    monkeypatch.setattr(suffix_automaton, "_sa_native", native)
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock())
    manager = suffix_automaton.SuffixAutomatonManager(config, max_num_requests=8, max_seq_len=32)
    metadata = SASpecMetadata(
        max_num_requests=0,
        max_draft_len=3,
        max_total_draft_tokens=3,
        spec_dec_mode=config.spec_dec_mode,
        sa_manager=manager,
    )
    executor = _ordinary_executor()
    engine = executor.model_engine
    engine.spec_config = config
    engine.model = SimpleNamespace(
        spec_config=selected_config, spec_worker=worker, draft_model=None
    )
    engine._get_spec_worker = lambda: worker
    engine.spec_metadata = metadata
    executor.resource_manager = ResourceManager(
        {ResourceManagerType.SPEC_RESOURCE_MANAGER: manager}
    )
    return executor, worker, manager, metadata, native


@pytest.mark.parametrize("global_pool", [False, True])
def test_standalone_sa_factory_and_same_manager_metadata_join(monkeypatch, global_pool) -> None:
    executor, worker, manager, metadata, native = _sa_progress_executor_fixture(
        monkeypatch, global_pool
    )
    engine = executor.model_engine
    assert engine.model.spec_config is not engine.spec_config
    assert engine.model.spec_config == engine.spec_config
    assert worker.spec_config is engine.model.spec_config
    _validate_progress_modes(executor)

    # No current request means no host build or independently advancing job.
    manager.prepare_resources(SimpleNamespace(context_requests=[], generation_requests=[]))
    native.build_automaton_host.assert_not_called()
    request = SimpleNamespace(
        request_id=1,
        is_generation_only_request=False,
        is_first_context_chunk=True,
        get_tokens=lambda beam: [1, 2, 3],
    )
    manager.prepare_resources(SimpleNamespace(context_requests=[request], generation_requests=[]))
    assert manager._pending_copies == {1}
    manager.free_resources(request)
    assert not manager._pending_copies
    assert not manager._active_slots
    assert not manager._initialized_requests
    manager.shutdown()


@pytest.mark.parametrize(
    "field",
    [
        "config",
        "worker",
        "pool",
        "metadata",
        "auxiliary",
        "drafter",
        "draft_model",
        "worker_config",
    ],
)
def test_standalone_sa_join_rejects_unreviewed_or_mismatched_state(monkeypatch, field) -> None:
    from tensorrt_llm.llmapi import SADecodingConfig

    executor, worker, manager, metadata, native = _sa_progress_executor_fixture(
        monkeypatch, global_pool=False
    )
    _validate_progress_modes(executor)
    engine = executor.model_engine
    if field == "config":
        engine.model.spec_config = SADecodingConfig(max_draft_len=4)
        worker.spec_config = engine.model.spec_config
    elif field == "worker":
        engine.model.spec_worker = object()
    elif field == "pool":
        metadata.sa_manager = object()
    elif field == "metadata":
        engine.spec_metadata = None
    elif field == "auxiliary":
        worker._auxiliary_state_handlers.append(object())
    elif field == "drafter":
        executor.drafter = object()
    elif field == "draft_model":
        engine.model.draft_model = object()
    else:
        worker.spec_config = copy.deepcopy(engine.model.spec_config)
    with pytest.raises(ValueError, match="built-in SA worker/pool"):
        _validate_progress_modes(executor)


@pytest.mark.parametrize("already_queued", [False, True])
@pytest.mark.parametrize("item_id", [1, -1, -2])
@pytest.mark.parametrize("live_requests", [0, 1])
def test_real_fetch_has_one_original_batch_deadline_and_parks_only_if_idle(
    already_queued, item_id, live_requests
) -> None:
    from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
        ExecutorRequestQueue,
        RequestQueueItem,
    )
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    events = []
    dist = SimpleNamespace(
        rank=0,
        world_size=2,
        tp_size=2,
        cp_size=1,
        pp_size=1,
        has_pp=False,
        supports_single_round_broadcast=False,
    )
    requests = ExecutorRequestQueue(
        dist, max_batch_size=8, enable_iter_perf_stats=False, batch_wait_timeout_ms=100
    )
    item = RequestQueueItem(item_id)
    underlying = Mock()
    underlying.empty.side_effect = lambda: events.append("queue-empty") or not already_queued
    underlying.get_nowait.side_effect = [item, queue.Empty]

    def get(timeout=None):
        events.append(("get", timeout))
        if timeout is None:
            return item
        raise queue.Empty

    underlying.get.side_effect = get
    requests.request_queue = underlying
    original_fetch = requests.get_from_request_queue
    requests.get_from_request_queue = Mock(wraps=original_fetch)
    broadcaster = RequestBroadcaster(dist, SimpleNamespace(pause=nullcontext))
    broadcaster.idle_notification = IdleRequestNotification(
        lambda notice: events.append(("notice", notice)), Mock()
    )
    broadcaster._broadcast_header = Mock(
        side_effect=lambda header, **kwargs: events.append(("header", header)) or header
    )
    broadcaster._collect_py_objects = Mock(return_value=None)
    broadcaster._broadcast_requests = Mock()

    class Waiting:
        def __len__(self):
            return 0

        add_requests = Mock()

    waiting = Waiting()
    executor = _ordinary_executor()
    vars(executor).update(
        kv_connector_manager=None,
        control_requests=[],
        is_shutdown=False,
        _has_pending_connector_transfers=lambda: False,
        _idle_notification_runtime=object(),
        dist=dist,
        request_accumulated=[],
        hang_detector=SimpleNamespace(pause=nullcontext),
        executor_request_queue=requests,
        request_broadcaster=broadcaster,
        _first_token_timer=None,
        _handle_special_queue_items=lambda items: items,
    )
    _validate_progress_modes(executor)
    with patch(
        "tensorrt_llm._torch.pyexecutor.executor_request_queue.time.monotonic",
        side_effect=[100.0, 100.0],
    ) as clock:
        PyExecutor._fetch_and_enqueue_requests(
            executor, waiting, total_num_live_requests=live_requests
        )

    requests.get_from_request_queue.assert_called_once_with(
        None if live_requests == 0 else datetime.timedelta(0)
    )
    assert clock.call_count == 2  # One deadline creation and one remaining-time read.
    if live_requests == 0:
        assert events[0] == ("header", [-1])
        assert events[-2:] == [("notice", (0, True)), ("header", [1])]
    else:
        assert events[-1] == ("header", [1])
        assert not any(event[0] == "notice" for event in events if isinstance(event, tuple))
    timed_gets = [
        event
        for event in events
        if isinstance(event, tuple) and event[0] == "get" and event[1] is not None
    ]
    assert len(timed_gets) == 1
    assert timed_gets[0][1] == pytest.approx(0.1)
    assert underlying.get.call_count == (2 if live_requests == 0 and not already_queued else 1)
    waiting.add_requests.assert_called_once_with([item])
    broadcaster._broadcast_requests.assert_called_once_with([item], None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("enable_attention_dp", True),
        ("_resource_governor_enabled", True),
        ("kv_connector_manager", object()),
        ("drafter", object()),
    ],
)
def test_unsupported_progress_modes_are_rejected_without_disabling_them(field, value) -> None:
    executor = _ordinary_executor()
    setattr(executor, field, value)

    with pytest.raises(ValueError):
        _validate_progress_modes(executor)
    assert getattr(executor, field) is value


@pytest.mark.parametrize("clean", [False, True])
def test_worker_owns_transport_and_never_frees_communicator_on_failure(monkeypatch, clean) -> None:
    from tensorrt_llm.executor import ipc

    queues = []

    class Queue:
        def __init__(self, **kwargs):
            self.address = ("tcp://192.0.2.1:1234", b"test-only-authentication")
            self.socket = Mock()
            self.closed = False
            queues.append(self)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(ipc, "ZeroMqQueue", Queue)
    comm = SimpleNamespace(gather=Mock(return_value=None), Free=Mock())
    executor = SimpleNamespace(
        dist=SimpleNamespace(rank=1),
        request_broadcaster=SimpleNamespace(idle_notification=None),
        _event_loop_completed=clean,
    )
    runtime = IdleNotificationRuntime(comm, "192.0.2.1")

    with runtime.worker(executor):
        assert isinstance(executor.request_broadcaster.idle_notification, IdleRequestNotification)
        assert not queues[0].closed

    assert executor.request_broadcaster.idle_notification is None
    assert queues[0].closed
    queues[0].socket.setsockopt.assert_called_once()
    assert comm.Free.call_count == int(clean)


def test_partial_worker_setup_closes_queues_without_collective_cleanup(monkeypatch) -> None:
    from tensorrt_llm.executor import ipc

    receiver = SimpleNamespace(
        address=("tcp://192.0.2.1:1234", b"test-only-authentication"), socket=Mock(), close=Mock()
    )
    monkeypatch.setattr(ipc, "ZeroMqQueue", Mock(return_value=receiver))
    comm = SimpleNamespace(gather=Mock(side_effect=RuntimeError("setup failed")), Free=Mock())
    executor = SimpleNamespace(
        dist=SimpleNamespace(rank=1),
        request_broadcaster=SimpleNamespace(idle_notification=None),
        _event_loop_completed=False,
    )

    with (
        pytest.raises(RuntimeError, match="setup failed"),
        IdleNotificationRuntime(comm, "192.0.2.1").worker(executor),
    ):
        pytest.fail("Failed setup must not enter the executor body")

    receiver.close.assert_called_once()
    assert executor.request_broadcaster.idle_notification is None
    comm.Free.assert_not_called()


@pytest.mark.parametrize("completed", [False, True])
def test_worker_context_exception_uses_actual_loop_completion(monkeypatch, completed) -> None:
    from tensorrt_llm.executor import ipc

    receiver = SimpleNamespace(
        address=("tcp://192.0.2.1:1234", b"test-only-authentication"), socket=Mock(), close=Mock()
    )
    monkeypatch.setattr(ipc, "ZeroMqQueue", Mock(return_value=receiver))
    comm = SimpleNamespace(gather=Mock(return_value=None), Free=Mock())
    executor = SimpleNamespace(
        dist=SimpleNamespace(rank=1),
        request_broadcaster=SimpleNamespace(idle_notification=None),
        _event_loop_completed=False,
    )

    with (
        pytest.raises(RuntimeError, match="context teardown"),
        IdleNotificationRuntime(comm, "192.0.2.1").worker(executor),
    ):
        executor._event_loop_completed = completed
        raise RuntimeError("context teardown")

    receiver.close.assert_called_once()
    assert executor.request_broadcaster.idle_notification is None
    assert comm.Free.call_count == int(completed)


def test_queue_cleanup_error_does_not_skip_peers_normal_collective(monkeypatch) -> None:
    import zmq

    from tensorrt_llm.executor import ipc

    first = SimpleNamespace(socket=Mock(), close=Mock(side_effect=zmq.ZMQError(zmq.EINVAL)))
    second = SimpleNamespace(socket=Mock(), close=Mock())
    monkeypatch.setattr(ipc, "ZeroMqQueue", Mock(side_effect=[first, second]))
    comm = SimpleNamespace(
        gather=Mock(return_value=[None, ("a", b"key"), ("b", b"key")]), Free=Mock()
    )
    executor = SimpleNamespace(
        dist=SimpleNamespace(rank=0),
        request_broadcaster=SimpleNamespace(idle_notification=None),
        _event_loop_completed=True,
    )

    with pytest.raises(zmq.ZMQError), IdleNotificationRuntime(comm, "192.0.2.1").worker(executor):
        pass

    first.close.assert_called_once()
    second.close.assert_called_once()
    comm.Free.assert_called_once()
    assert executor.request_broadcaster.idle_notification is None


def test_mismatched_rank_enablement_rejects_before_socket_or_communicator_dup(monkeypatch) -> None:
    from tensorrt_llm import _utils
    from tensorrt_llm._torch import distributed
    from tensorrt_llm._torch.pyexecutor.idle_notification_runtime import prepare_idle_notification

    class FakeMPIDist:
        pass

    comm = SimpleNamespace(allgather=Mock(return_value=["0", "1"]), Dup=Mock())
    monkeypatch.setattr(distributed, "MPIDist", FakeMPIDist)
    monkeypatch.setattr(_utils, "mpi_comm", lambda: comm)
    monkeypatch.setenv("TLLM_MPI_IDLE_NOTIFICATION", "0")

    with pytest.raises(ValueError, match="match on every MPI rank"):
        prepare_idle_notification(SimpleNamespace(dist=FakeMPIDist()))
    comm.Dup.assert_not_called()
