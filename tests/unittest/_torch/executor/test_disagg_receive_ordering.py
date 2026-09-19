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
    manager = Mock(spec=KVCacheManagerV2)
    manager._stream = Mock()
    manager.synchronize_for_disagg_receive = MethodType(
        KVCacheManagerV2.synchronize_for_disagg_receive, manager
    )
    return manager


@pytest.mark.cpu_only
def test_disagg_receive_waits_after_admission_before_publication(transfer_mode: str) -> None:
    events = []
    manager = _manager()
    manager.prepare_resources.side_effect = lambda _: events.append("prepare")
    manager.report_batch_to_connector = Mock(side_effect=lambda _: events.append("connector"))
    manager._stream.synchronize.side_effect = lambda: events.append("ready")
    receive = Mock(side_effect=lambda _: events.append("publish"))

    _coordinator(manager, receive).receive_gen_init([Mock(), Mock()])

    assert events == ["prepare", "connector", "ready", "publish", "publish"]
    manager._stream.synchronize.assert_called_once_with()


@pytest.mark.cpu_only
def test_empty_disagg_admission_does_not_wait_or_receive(transfer_mode: str) -> None:
    manager = _manager()
    receive = Mock()

    _coordinator(manager, receive).receive_gen_init([])

    manager.prepare_resources.assert_not_called()
    manager._stream.synchronize.assert_not_called()
    receive.assert_not_called()


@pytest.mark.cpu_only
def test_failed_disagg_fence_does_not_publish_receive(transfer_mode: str) -> None:
    manager = _manager()
    manager._stream.synchronize.side_effect = RuntimeError("cache stream failed")
    receive = Mock()

    with pytest.raises(RuntimeError, match="cache stream failed"):
        _coordinator(manager, receive).receive_gen_init([Mock()])

    receive.assert_not_called()


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

    def queue_partial_copy() -> None:
        with torch.cuda.stream(cache_stream):
            torch.cuda._sleep(100_000_000)
            destination.copy_(sibling)

    def receive(_: object) -> None:
        with torch.cuda.stream(receive_stream):
            destination.copy_(received)

    # Establish that this schedule exposes the overwrite without a host fence.
    queue_partial_copy()
    receive(None)
    torch.cuda.synchronize()
    assert torch.equal(destination, sibling)

    # Admission queues the copy before the coordinator prepares receive resources.
    coordinator = _coordinator(manager, Mock(side_effect=receive))
    queue_partial_copy()
    coordinator.receive_gen_init([Mock()])
    torch.cuda.synchronize()
    assert torch.equal(destination, received)
