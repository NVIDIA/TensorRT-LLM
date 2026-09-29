# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation admission must finish local KV writes before publishing RDMA slots."""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.disaggregation.orchestration.coordinator import DisaggTransferCoordinator
from tensorrt_llm._torch.pyexecutor.disagg_adapter import PyExecutorEffects
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests


@pytest.fixture(params=["async", "sync"])
def transfer_mode(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    monkeypatch.setenv(
        "TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP", "1" if request.param == "sync" else "0"
    )
    return request.param


def _coordinator(manager: Mock, receive: Mock) -> DisaggTransferCoordinator:
    executor = SimpleNamespace(
        resource_manager=SimpleNamespace(
            resource_managers={ResourceManagerType.KV_CACHE_MANAGER: manager}
        ),
    )
    executor._prepare_disagg_gen_resources = MethodType(
        PyExecutor._prepare_disagg_gen_resources, executor
    )
    coordinator = object.__new__(DisaggTransferCoordinator)
    coordinator._effects = PyExecutorEffects(executor)
    coordinator._transceiver = SimpleNamespace(
        request_and_receive_async=receive,
        request_and_receive_sync=receive,
        kv_transfer_timeout_ms=None,
    )
    coordinator.reap_gen_receives = Mock()
    coordinator._check_transfer_errors = Mock()
    return coordinator


def _manager() -> Mock:
    """Run the real admission and resource hooks with mocked cache allocation."""
    manager = Mock(spec=KVCacheManagerV2)
    manager._stream = Mock()
    manager._disagg_receive_ready = {}
    manager.is_draft = False
    manager.kv_connector_manager = None
    manager.enable_block_reuse = False
    manager._has_cp_helix = False
    manager.num_extra_kv_tokens = 0
    manager.kv_cache_map = {}
    manager.prepare_context_cache.return_value = 0
    manager.prepare_disagg_gen_init = MethodType(KVCacheManagerV2.prepare_disagg_gen_init, manager)
    manager.prepare_resources = MethodType(KVCacheManagerV2.prepare_resources, manager)
    return manager


def _request(manager: Mock, request_id: int = 1) -> SimpleNamespace:
    request = SimpleNamespace(
        py_request_id=request_id,
        prompt_len=128,
        py_draft_tokens=[],
        is_first_context_chunk=True,
    )
    manager.kv_cache_map[request_id] = Mock(capacity=0)
    manager.kv_cache_map[request_id].resize.return_value = True
    return request


@pytest.mark.cpu_only
def test_disagg_receive_waits_for_admission_event(
    transfer_mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resource preparation must wait before publishing any receive destination."""
    events = []
    manager = _manager()
    request = _request(manager)
    ready = Mock()
    monkeypatch.setattr(torch.cuda, "Event", Mock(return_value=ready))
    manager.prepare_context_cache.side_effect = lambda _: events.append("admit") or 0
    manager._fill_fresh_kv_pages.side_effect = lambda _: events.append("fill")
    ready.record.side_effect = lambda _: events.append("record")
    ready.synchronize.side_effect = lambda: events.append("ready")

    def report_connector(batch: ScheduledRequests, *, finalize_prefix_reservations: bool) -> None:
        assert batch.context_requests == [request]
        assert finalize_prefix_reservations is False
        events.append("connector")

    manager.report_batch_to_connector = Mock(side_effect=report_connector)
    receive = Mock(side_effect=lambda _: events.append("publish"))

    assert manager.prepare_disagg_gen_init(request)
    _coordinator(manager, receive).receive_gen_init([request])

    assert events == ["admit", "fill", "record", "ready", "connector", "publish"]
    ready.record.assert_called_once_with(manager._stream)
    ready.synchronize.assert_called_once_with()
    assert manager._disagg_receive_ready == {}
    manager._stream.synchronize.assert_not_called()


@pytest.mark.cpu_only
def test_receive_waits_only_for_selected_requests(transfer_mode: str) -> None:
    """Deferred admissions retain their own readiness events."""
    manager = _manager()
    first, deferred = _request(manager, 1), _request(manager, 2)
    first_ready, deferred_ready = Mock(), Mock()
    manager._disagg_receive_ready = {1: first_ready, 2: deferred_ready}
    coordinator = _coordinator(manager, Mock())

    coordinator.receive_gen_init([first])

    first_ready.synchronize.assert_called_once_with()
    deferred_ready.synchronize.assert_not_called()
    assert manager._disagg_receive_ready == {2: deferred_ready}
    coordinator.receive_gen_init([deferred])
    deferred_ready.synchronize.assert_called_once_with()
    assert manager._disagg_receive_ready == {}
    manager._stream.synchronize.assert_not_called()


@pytest.mark.cpu_only
def test_empty_disagg_admission_does_not_wait_or_receive(transfer_mode: str) -> None:
    manager = _manager()
    ready = Mock()
    manager._disagg_receive_ready[1] = ready
    receive = Mock()

    _coordinator(manager, receive).receive_gen_init([])

    ready.synchronize.assert_not_called()
    manager._stream.synchronize.assert_not_called()
    receive.assert_not_called()


@pytest.mark.cpu_only
def test_prepared_request_does_not_wait_again(transfer_mode: str) -> None:
    manager = _manager()
    request = _request(manager)
    ready = Mock()
    manager._disagg_receive_ready[request.py_request_id] = ready
    coordinator = _coordinator(manager, Mock())

    coordinator.receive_gen_init([request])
    manager.prepare_resources(SimpleNamespace(context_requests=[request]))

    ready.synchronize.assert_called_once_with()
    manager._stream.synchronize.assert_not_called()


@pytest.mark.cpu_only
def test_failed_disagg_event_does_not_publish_receive(transfer_mode: str) -> None:
    manager = _manager()
    request = _request(manager)
    ready = Mock()
    ready.synchronize.side_effect = RuntimeError("cache event failed")
    manager._disagg_receive_ready[request.py_request_id] = ready
    receive = Mock()

    with pytest.raises(RuntimeError, match="cache event failed"):
        _coordinator(manager, receive).receive_gen_init([request])

    receive.assert_not_called()
    assert manager._disagg_receive_ready[request.py_request_id] is ready
    manager._stream.synchronize.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize("failure", ["prepare", "resize"])
def test_failed_admission_does_not_record_event(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    manager = _manager()
    request = _request(manager)
    create_event = Mock()
    monkeypatch.setattr(torch.cuda, "Event", create_event)
    if failure == "prepare":
        manager.prepare_context_cache.return_value = None
    else:
        manager.kv_cache_map[request.py_request_id].resize.return_value = False

    assert not manager.prepare_disagg_gen_init(request)

    create_event.assert_not_called()
    assert manager._disagg_receive_ready == {}


@pytest.mark.cpu_only
def test_readmission_refreshes_readiness_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deferred request's next admission may enqueue additional cache work."""
    manager = _manager()
    request = _request(manager)
    first_ready, latest_ready = Mock(), Mock()
    monkeypatch.setattr(torch.cuda, "Event", Mock(side_effect=[first_ready, latest_ready]))

    assert manager.prepare_disagg_gen_init(request)
    assert manager.prepare_disagg_gen_init(request)
    manager.prepare_resources(SimpleNamespace(context_requests=[request]))

    first_ready.synchronize.assert_not_called()
    latest_ready.record.assert_called_once_with(manager._stream)
    latest_ready.synchronize.assert_called_once_with()
    assert manager._disagg_receive_ready == {}


@pytest.mark.cpu_only
def test_cancelled_admission_releases_event() -> None:
    manager = _manager()
    request = _request(manager)
    ready = Mock()
    manager._disagg_receive_ready[request.py_request_id] = ready
    manager.conversation_manager = None
    manager._allocated_draft_lens = {}
    manager._request_stats_enabled_ids = set()
    manager._fresh_pages_filled = {}
    manager._early_freed_index_requests = set()
    manager.impl = Mock()
    manager.index_mapper = Mock()

    KVCacheManagerV2.free_resources(manager, request)

    assert manager._disagg_receive_ready == {}
    ready.synchronize.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_deferred_partial_copy_cannot_overwrite_received_kv(transfer_mode: str) -> None:
    """Model admission's queued copy and an independently ordered RDMA writer."""
    cache_stream = torch.cuda.Stream()
    receive_stream = torch.cuda.Stream()
    sibling = torch.full((4096,), 17, dtype=torch.int32, device="cuda")
    received = torch.full_like(sibling, 29)
    destination = torch.zeros_like(sibling)
    torch.cuda.synchronize()

    manager = _manager()
    manager._stream = cache_stream
    request = _request(manager)

    def queue_partial_copy(_: object) -> int:
        with torch.cuda.stream(cache_stream):
            torch.cuda._sleep(100_000_000)
            destination.copy_(sibling)
        return 0

    def receive(_: object) -> None:
        with torch.cuda.stream(receive_stream):
            destination.copy_(received)

    # Establish that this schedule exposes the overwrite without a host fence.
    queue_partial_copy(request)
    receive(request)
    torch.cuda.synchronize()
    assert torch.equal(destination, sibling)

    manager.prepare_context_cache.side_effect = queue_partial_copy
    coordinator = _coordinator(manager, Mock(side_effect=receive))
    assert manager.prepare_disagg_gen_init(request)
    coordinator.receive_gen_init([request])
    torch.cuda.synchronize()
    assert torch.equal(destination, received)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_receive_does_not_drain_work_after_admission(transfer_mode: str) -> None:
    """A later execution-stream tail must still be pending at receive publication."""
    manager = _manager()
    manager._stream = torch.cuda.Stream()
    request = _request(manager)
    tail = torch.cuda.Event()
    assert manager.prepare_disagg_gen_init(request)
    with torch.cuda.stream(manager._stream):
        torch.cuda._sleep(500_000_000)
        tail.record()
    completed_at_publication = []
    receive = Mock(side_effect=lambda _: completed_at_publication.append(tail.query()))

    try:
        _coordinator(manager, receive).receive_gen_init([request])
        assert completed_at_publication == [False]
    finally:
        tail.synchronize()
