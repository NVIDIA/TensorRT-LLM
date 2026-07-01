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
"""Tests for the passive PyExecutor committed-membership metrics transport."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import zmq

from tensorrt_llm._torch.modules.fused_moe.ep_group_health import (
    EP_GROUP_HEALTH_EXTRA_ATTR,
    EPGroupHealth,
)
from tensorrt_llm._torch.modules.fused_moe.ep_metrics import pending_ep_health_metrics
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.executor.base_worker import BaseWorker
from tensorrt_llm.executor.proxy import GenerationExecutorProxy
from tensorrt_llm.executor.rpc import RPCCancelled, RPCClient, RPCError, RPCServer, RPCTimeout
from tensorrt_llm.executor.rpc.rpc_common import get_unique_ipc_addr
from tensorrt_llm.executor.rpc_proxy import GenerationExecutorRpcProxy
from tensorrt_llm.executor.rpc_proxy_mixin import RpcExecutorMixin
from tensorrt_llm.llmapi.llm import BaseLLM
from tensorrt_llm.serve.openai_server import OpenAIServer

EP_HEALTH_SOURCE_EPOCH = "source-a"


def _make_executor(health: EPGroupHealth) -> PyExecutor:
    executor = PyExecutor.__new__(PyExecutor)
    executor._ep_health_model_config = SimpleNamespace(
        extra_attrs={EP_GROUP_HEALTH_EXTRA_ATTR: health}
    )
    executor.model_engine = SimpleNamespace(
        model=SimpleNamespace(model_config=executor._ep_health_model_config)
    )
    return executor


def test_pyexecutor_returns_committed_membership_snapshot() -> None:
    health = EPGroupHealth(4)
    health.mark_failed(2)
    executor = _make_executor(health)

    assert executor._get_ep_health_stats() == {
        "sourceEpoch": health.source_epoch,
        "worldSize": 4,
        "activeCount": 3,
        "failedRanks": [2],
        "generation": 1,
    }
    executor.enable_iter_perf_stats = False
    assert executor.get_latest_iteration_stats() == []


def test_pyexecutor_discovers_tracker_attached_after_initialization() -> None:
    executor = PyExecutor.__new__(PyExecutor)
    executor._ep_health_model_config = SimpleNamespace(
        extra_attrs={EP_GROUP_HEALTH_EXTRA_ATTR: None}
    )
    assert executor._get_ep_health_stats() == pending_ep_health_metrics()

    health = EPGroupHealth(4)
    health.mark_failed(1)
    executor._ep_health_model_config.extra_attrs[EP_GROUP_HEALTH_EXTRA_ATTR] = health

    stats = executor._get_ep_health_stats()
    assert stats["sourceEpoch"] == health.source_epoch
    assert stats["failedRanks"] == [1]


def test_pyexecutor_rejects_unqualified_multi_group_metrics() -> None:
    executor = _make_executor(EPGroupHealth(4))
    executor._ep_health_model_config.mapping = SimpleNamespace(
        moe_tp_size=1,
        pp_size=2,
        moe_cluster_size=1,
    )

    with pytest.raises(NotImplementedError, match="multi-group MoE topologies"):
        executor._get_ep_health_stats()


def test_base_worker_exposes_engine_health_snapshot() -> None:
    health = EPGroupHealth(4)
    health.mark_failed(1)
    worker = BaseWorker.__new__(BaseWorker)
    worker.engine = _make_executor(health)

    assert worker.fetch_ep_health_stats()["failedRanks"] == [1]


def test_base_worker_reports_unsupported_engine_without_rpc_failure() -> None:
    worker = BaseWorker.__new__(BaseWorker)
    worker.engine = SimpleNamespace()

    assert worker.fetch_ep_health_stats() is None


def test_proxy_fetches_health_over_dedicated_rpc() -> None:
    expected = {
        "sourceEpoch": EP_HEALTH_SOURCE_EPOCH,
        "worldSize": 4,
        "activeCount": 3,
        "failedRanks": [1],
        "generation": 1,
    }
    proxy = GenerationExecutorProxy.__new__(GenerationExecutorProxy)
    proxy.rpc_client = MagicMock()
    proxy.rpc_client.fetch_ep_health_stats.return_value.remote.return_value = expected

    assert proxy._get_ep_health_stats() == expected
    proxy.rpc_client.fetch_ep_health_stats.return_value.remote.assert_called_once_with(timeout=1.0)


def test_proxy_propagates_transient_rpc_failure() -> None:
    proxy = GenerationExecutorProxy.__new__(GenerationExecutorProxy)
    proxy.rpc_client = MagicMock()
    proxy.rpc_client.fetch_ep_health_stats.return_value.remote.side_effect = TimeoutError

    with pytest.raises(TimeoutError):
        proxy._get_ep_health_stats()


def test_rpc_proxy_fetches_health_over_dedicated_rpc() -> None:
    expected = {
        "sourceEpoch": EP_HEALTH_SOURCE_EPOCH,
        "worldSize": 4,
        "activeCount": 4,
        "failedRanks": [],
        "generation": 0,
    }
    proxy = GenerationExecutorRpcProxy.__new__(GenerationExecutorRpcProxy)
    proxy.rpc_client = MagicMock()
    proxy.rpc_client.fetch_ep_health_stats.return_value.remote.return_value = expected

    assert proxy._get_ep_health_stats(timeout=0.25) == expected
    proxy.rpc_client.fetch_ep_health_stats.return_value.remote.assert_called_once_with(timeout=0.25)


def test_rpc_executor_mixin_forwards_health_for_ray_and_rpc_executors() -> None:
    """Every RpcExecutorMixin consumer, including RayExecutor, gets telemetry."""
    expected = {
        "sourceEpoch": EP_HEALTH_SOURCE_EPOCH,
        "worldSize": 4,
        "activeCount": 4,
        "failedRanks": [],
        "generation": 0,
    }
    executor = SimpleNamespace(rpc_client=MagicMock())
    executor.rpc_client.fetch_ep_health_stats.return_value.remote.return_value = expected

    assert RpcExecutorMixin._get_ep_health_stats(executor, timeout=0.5) == expected
    executor.rpc_client.fetch_ep_health_stats.return_value.remote.assert_called_once_with(
        timeout=0.5
    )


def test_ep_health_snapshot_crosses_real_rpc_and_llm_bridge() -> None:
    """Exercise the passive committed-membership path without a model worker."""

    class HealthRpcSurface:
        _get_ep_health_stats = BaseWorker._get_ep_health_stats
        fetch_ep_health_stats = BaseWorker.fetch_ep_health_stats

    health = EPGroupHealth(4)
    health.mark_failed(2)
    worker = HealthRpcSurface()
    worker.engine = _make_executor(health)
    address = get_unique_ipc_addr()

    with RPCServer(worker) as rpc_server:
        rpc_server.bind(address)
        rpc_server.start()
        with RPCClient(address, hmac_key=rpc_server.hmac_key) as rpc_client:
            proxy = GenerationExecutorProxy.__new__(GenerationExecutorProxy)
            proxy.rpc_client = rpc_client
            llm = BaseLLM.__new__(BaseLLM)
            llm._executor = proxy

            assert llm._get_ep_health_stats() == {
                "sourceEpoch": health.source_epoch,
                "worldSize": 4,
                "activeCount": 3,
                "failedRanks": [2],
                "generation": 1,
            }


def test_server_mount_metrics_registers_local_ep_health_collector(monkeypatch) -> None:
    """The Prometheus endpoint registry includes the coherent local snapshot."""
    import prometheus_client
    import prometheus_fastapi_instrumentator
    from prometheus_client import multiprocess
    from prometheus_client.core import GaugeMetricFamily

    class SnapshotCollector:
        def collect(self):
            metric = GaugeMetricFamily(
                "trtllm_ep_health_available",
                "Whether EP health telemetry is available.",
            )
            metric.add_metric([], 1)
            return [metric]

    metrics_collector = MagicMock()
    metrics_collector.register_ep_health_metrics.side_effect = lambda registry: registry.register(
        SnapshotCollector()
    )
    monkeypatch.setattr(multiprocess, "MultiProcessCollector", MagicMock())
    instrumentator = MagicMock()
    instrumentator.add.return_value = instrumentator
    instrumentator.instrument.return_value = instrumentator
    instrumentator.expose.return_value = instrumentator
    monkeypatch.setattr(
        prometheus_fastapi_instrumentator, "Instrumentator", MagicMock(return_value=instrumentator)
    )
    monkeypatch.setattr(prometheus_client, "make_asgi_app", MagicMock(return_value=MagicMock()))

    server = OpenAIServer.__new__(OpenAIServer)
    server.app = SimpleNamespace(routes=[])
    server.metrics_collector = metrics_collector
    server.mount_metrics()

    registry = metrics_collector.register_ep_health_metrics.call_args.args[0]
    output = prometheus_client.generate_latest(registry).decode()
    assert "trtllm_ep_health_available 1.0" in output
    assert len(server.app.routes) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transient_error",
    [
        TimeoutError("health read timed out"),
        RPCTimeout("health RPC timed out"),
        RPCCancelled("rank-0 worker is being replaced"),
        RPCError("health transport failed", cause=ConnectionError("closed")),
        zmq.ZMQError(zmq.ENOTCONN, "socket disconnected"),
        RPCError(
            "wrapped socket failure",
            cause=zmq.ZMQError(zmq.ENOTCONN, "socket disconnected"),
        ),
    ],
)
async def test_server_retries_after_transient_first_read_failure(
    transient_error: Exception,
) -> None:
    expected = {
        "sourceEpoch": EP_HEALTH_SOURCE_EPOCH,
        "worldSize": 4,
        "activeCount": 3,
        "failedRanks": [1],
        "generation": 1,
    }
    server = OpenAIServer.__new__(OpenAIServer)
    server.metrics_collector = MagicMock()
    server.metrics_collector.log_ep_health_stats.return_value = True
    server._ep_health_collector_task = None
    server._ep_health_stop_event = asyncio.Event()

    results = iter((transient_error, expected))

    def read_health():
        result = next(results)
        if isinstance(result, Exception):
            raise result
        server._ep_health_stop_event.set()
        return result

    health_reader = MagicMock(side_effect=read_health)
    server.generator = SimpleNamespace(_get_ep_health_stats=health_reader)
    server._wait_for_ep_health_stop = AsyncMock(side_effect=[False, True])

    await server._start_ep_health_collector()
    assert server._ep_health_collector_task is not None
    await server._ep_health_collector_task

    server.metrics_collector.log_ep_health_unavailable.assert_called_once_with()
    server.metrics_collector.log_ep_health_stats.assert_called_once_with(expected)
    assert health_reader.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transient_error",
    [
        TimeoutError("timed out"),
        RPCCancelled("rank-0 worker is being replaced"),
        zmq.ZMQError(zmq.ENOTCONN, "socket disconnected"),
        RPCError(
            "wrapped socket failure",
            cause=zmq.ZMQError(zmq.ENOTCONN, "socket disconnected"),
        ),
    ],
)
async def test_server_loop_retries_transient_rank0_loss_and_recovers(
    transient_error: Exception,
) -> None:
    """Transient rank-0 loss clears availability, then accepts its replacement."""
    before_restart = {
        "sourceEpoch": "source-before-restart",
        "worldSize": 4,
        "activeCount": 3,
        "failedRanks": [1],
        "generation": 2,
    }
    after_restart = {
        "sourceEpoch": "source-after-restart",
        "worldSize": 4,
        "activeCount": 4,
        "failedRanks": [],
        "generation": 0,
    }
    server = OpenAIServer.__new__(OpenAIServer)
    server.metrics_collector = MagicMock()
    server.metrics_collector.log_ep_health_stats.return_value = True
    server._ep_health_stop_event = asyncio.Event()

    results = iter((before_restart, transient_error, after_restart))

    def read_health():
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    health_reader = MagicMock(side_effect=read_health)
    server.generator = SimpleNamespace(_get_ep_health_stats=health_reader)
    server._wait_for_ep_health_stop = AsyncMock(side_effect=[False, False, True])

    await asyncio.wait_for(server._ep_health_collector_loop(), timeout=1.0)

    assert health_reader.call_count == 3
    server.metrics_collector.log_ep_health_unavailable.assert_called_once_with()
    assert [
        call.args[0] for call in server.metrics_collector.log_ep_health_stats.call_args_list
    ] == [before_restart, after_restart]
    assert server._wait_for_ep_health_stop.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "read_result",
    [
        None,
        ValueError("invalid health configuration"),
        RPCError("remote health configuration failed", cause=ValueError("invalid")),
        RPCError(
            "unsupported multi-group topology",
            cause=NotImplementedError("multi-group"),
        ),
    ],
)
async def test_server_loop_stops_after_deterministic_health_loss(read_result) -> None:
    """Unsupported and deterministic failures do not create a polling loop."""
    server = OpenAIServer.__new__(OpenAIServer)
    server.metrics_collector = MagicMock()
    if isinstance(read_result, Exception):
        health_reader = MagicMock(side_effect=read_result)
    else:
        health_reader = MagicMock(return_value=read_result)
    server.generator = SimpleNamespace(_get_ep_health_stats=health_reader)

    await asyncio.wait_for(server._ep_health_collector_loop(), timeout=0.5)

    health_reader.assert_called_once_with()
    server.metrics_collector.log_ep_health_unavailable.assert_called_once_with()
    server.metrics_collector.log_ep_health_stats.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "startup_error",
    [
        ValueError("invalid health configuration"),
        RPCError("remote health configuration failed", cause=ValueError("invalid")),
        RPCError("method unavailable"),
    ],
)
async def test_server_stops_after_deterministic_first_read_failure(startup_error) -> None:
    server = OpenAIServer.__new__(OpenAIServer)
    server.metrics_collector = MagicMock()
    server._ep_health_collector_task = None
    health_reader = MagicMock(side_effect=startup_error)
    server.generator = SimpleNamespace(_get_ep_health_stats=health_reader)

    await server._start_ep_health_collector()
    task = server._ep_health_collector_task
    assert task is not None
    await task

    health_reader.assert_called_once_with()
    assert task.done()
    server.metrics_collector.log_ep_health_unavailable.assert_called_once_with()
    server.metrics_collector.log_ep_health_stats.assert_not_called()
    await server._stop_ep_health_collector()
    assert server._ep_health_collector_task is None


@pytest.mark.asyncio
async def test_server_retries_after_collector_rejects_payload() -> None:
    snapshot = {
        "sourceEpoch": EP_HEALTH_SOURCE_EPOCH,
        "worldSize": 4,
        "activeCount": 4,
        "failedRanks": [],
        "generation": 0,
    }
    server = OpenAIServer.__new__(OpenAIServer)
    server.metrics_collector = MagicMock()
    server.metrics_collector.log_ep_health_stats.side_effect = [False, True]
    server._ep_health_stop_event = asyncio.Event()

    read_count = 0

    def read_health():
        nonlocal read_count
        read_count += 1
        if read_count == 2:
            server._ep_health_stop_event.set()
        return snapshot

    server.generator = SimpleNamespace(_get_ep_health_stats=MagicMock(side_effect=read_health))
    server._wait_for_ep_health_stop = AsyncMock(side_effect=[False, True])

    await server._ep_health_collector_loop()

    assert server.generator._get_ep_health_stats.call_count == 2
    assert [
        args.args[0] for args in server.metrics_collector.log_ep_health_stats.call_args_list
    ] == [snapshot, snapshot]


@pytest.mark.asyncio
async def test_server_retries_after_initial_payload_rejection() -> None:
    snapshot = {
        "sourceEpoch": EP_HEALTH_SOURCE_EPOCH,
        "worldSize": 4,
        "activeCount": 4,
        "failedRanks": [],
        "generation": 0,
    }
    server = OpenAIServer.__new__(OpenAIServer)
    server.metrics_collector = MagicMock()
    server.metrics_collector.log_ep_health_stats.side_effect = [False, True]
    server._ep_health_collector_task = None
    server._ep_health_stop_event = asyncio.Event()

    read_count = 0

    def read_health():
        nonlocal read_count
        read_count += 1
        if read_count == 2:
            server._ep_health_stop_event.set()
        return snapshot

    server.generator = SimpleNamespace(_get_ep_health_stats=MagicMock(side_effect=read_health))
    server._wait_for_ep_health_stop = AsyncMock(side_effect=[False, True])

    await server._start_ep_health_collector()

    assert server._ep_health_collector_task is not None
    await server._ep_health_collector_task
    assert server.generator._get_ep_health_stats.call_count == 2
    assert server.metrics_collector.log_ep_health_stats.call_count == 2


@pytest.mark.asyncio
async def test_server_does_not_poll_unsupported_backend() -> None:
    server = OpenAIServer.__new__(OpenAIServer)
    server.metrics_collector = MagicMock()
    server._ep_health_collector_task = None
    server.generator = SimpleNamespace(_get_ep_health_stats=MagicMock(return_value=None))

    await server._start_ep_health_collector()
    task = server._ep_health_collector_task
    assert task is not None
    await task

    assert task.done()
    server.generator._get_ep_health_stats.assert_called_once_with()
    server.metrics_collector.log_ep_health_stats.assert_not_called()
    server.metrics_collector.log_ep_health_unavailable.assert_called_once_with()
    await server._stop_ep_health_collector()
    assert server._ep_health_collector_task is None


@pytest.mark.asyncio
async def test_server_retries_explicitly_pending_registration() -> None:
    snapshot = {
        "sourceEpoch": "source-after-attachment",
        "worldSize": 4,
        "activeCount": 4,
        "failedRanks": [],
        "generation": 0,
    }
    server = OpenAIServer.__new__(OpenAIServer)
    server.metrics_collector = MagicMock()
    server.metrics_collector.log_ep_health_stats.return_value = True
    server._ep_health_collector_task = None
    server._ep_health_stop_event = asyncio.Event()
    server.generator = SimpleNamespace(
        _get_ep_health_stats=MagicMock(side_effect=[pending_ep_health_metrics(), snapshot])
    )
    server._wait_for_ep_health_stop = AsyncMock(side_effect=[False, True])

    await server._start_ep_health_collector()
    assert server._ep_health_collector_task is not None
    await server._ep_health_collector_task

    server.metrics_collector.log_ep_health_unavailable.assert_called_once_with()
    server.metrics_collector.log_ep_health_stats.assert_called_once_with(snapshot)


@pytest.mark.asyncio
async def test_server_lifespan_starts_and_stops_health_collector(monkeypatch) -> None:
    """The real server lifespan owns the health collector task."""
    generator = SimpleNamespace(
        args=SimpleNamespace(enable_energy_metrics=False, enable_iter_perf_stats=False),
        _get_ep_health_stats=MagicMock(),
        shutdown=MagicMock(),
    )
    monkeypatch.setattr(OpenAIServer, "_init_llm", lambda self, chat_template=None: None)
    monkeypatch.setattr(OpenAIServer, "register_routes", lambda self: None)
    server = OpenAIServer(
        generator=generator,
        model="test-model",
        tool_parser=None,
        server_role=None,
        metadata_server_cfg=None,
    )
    server.metrics_collector = MagicMock()
    server.metrics_collector.log_ep_health_stats.return_value = True
    collector_started = asyncio.Event()

    async def controlled_collector_loop() -> None:
        collector_started.set()
        await server._ep_health_stop_event.wait()

    server._ep_health_collector_loop = controlled_collector_loop

    async with server.app.router.lifespan_context(server.app):
        await asyncio.wait_for(collector_started.wait(), timeout=1.0)
        collector_task = server._ep_health_collector_task
        assert collector_task is not None
        assert not collector_task.done()

    assert collector_task.done()
    assert not collector_task.cancelled()
    generator._get_ep_health_stats.assert_not_called()
    generator.shutdown.assert_called_once_with()


@pytest.mark.asyncio
async def test_server_lifespan_body_failure_still_shuts_down_once(monkeypatch) -> None:
    """An exception thrown at the lifespan yield cannot bypass cleanup."""
    generator = SimpleNamespace(
        args=SimpleNamespace(enable_energy_metrics=False, enable_iter_perf_stats=False),
        _get_ep_health_stats=MagicMock(return_value=None),
        shutdown=MagicMock(),
    )
    monkeypatch.setattr(OpenAIServer, "_init_llm", lambda self, chat_template=None: None)
    monkeypatch.setattr(OpenAIServer, "register_routes", lambda self: None)
    server = OpenAIServer(
        generator=generator,
        model="test-model",
        tool_parser=None,
        server_role=None,
        metadata_server_cfg=None,
    )
    server.metrics_collector = MagicMock()

    with pytest.raises(RuntimeError, match="lifespan body failed"):
        async with server.app.router.lifespan_context(server.app):
            raise RuntimeError("lifespan body failed")

    assert server._ep_health_collector_task is None
    generator.shutdown.assert_called_once_with()


@pytest.mark.asyncio
async def test_server_lifespan_drains_inflight_health_poll_before_shutdown(monkeypatch) -> None:
    """Startup stays non-blocking, while shutdown drains the health RPC."""
    expected = {
        "sourceEpoch": EP_HEALTH_SOURCE_EPOCH,
        "worldSize": 4,
        "activeCount": 4,
        "failedRanks": [],
        "generation": 0,
    }
    poll_started = threading.Event()
    release_poll = threading.Event()

    def read_health():
        poll_started.set()
        release_poll.wait(timeout=5.0)
        return expected

    health_reader = MagicMock(side_effect=read_health)
    generator = SimpleNamespace(
        args=SimpleNamespace(enable_energy_metrics=False, enable_iter_perf_stats=False),
        _get_ep_health_stats=health_reader,
        shutdown=MagicMock(),
    )
    monkeypatch.setattr(OpenAIServer, "_init_llm", lambda self, chat_template=None: None)
    monkeypatch.setattr(OpenAIServer, "register_routes", lambda self: None)
    server = OpenAIServer(
        generator=generator,
        model="test-model",
        tool_parser=None,
        server_role=None,
        metadata_server_cfg=None,
    )
    server.metrics_collector = MagicMock()
    server.metrics_collector.log_ep_health_stats.return_value = True

    lifespan = server.app.router.lifespan_context(server.app)
    await asyncio.wait_for(lifespan.__aenter__(), timeout=1.0)
    exit_task = None
    try:
        assert await asyncio.wait_for(asyncio.to_thread(poll_started.wait, 1.0), timeout=2.0)
        exit_task = asyncio.create_task(lifespan.__aexit__(None, None, None))
        await asyncio.wait_for(server._ep_health_stop_event.wait(), timeout=1.0)

        generator.shutdown.assert_not_called()
        assert not exit_task.done()
    finally:
        release_poll.set()
        if exit_task is None:
            exit_task = asyncio.create_task(lifespan.__aexit__(None, None, None))
        await asyncio.wait_for(exit_task, timeout=1.0)

    health_reader.assert_called_once_with()
    generator.shutdown.assert_called_once_with()


@pytest.mark.asyncio
async def test_server_lifespan_cancellation_still_drains_health_poll(monkeypatch) -> None:
    """Cancelling lifespan cleanup cannot abandon a running health RPC thread."""
    snapshot = {
        "sourceEpoch": EP_HEALTH_SOURCE_EPOCH,
        "worldSize": 4,
        "activeCount": 4,
        "failedRanks": [],
        "generation": 0,
    }
    poll_started = threading.Event()
    release_poll = threading.Event()

    def read_health():
        poll_started.set()
        release_poll.wait(timeout=5.0)
        return snapshot

    generator = SimpleNamespace(
        args=SimpleNamespace(enable_energy_metrics=False, enable_iter_perf_stats=False),
        _get_ep_health_stats=MagicMock(side_effect=read_health),
        shutdown=MagicMock(),
    )
    monkeypatch.setattr(OpenAIServer, "_init_llm", lambda self, chat_template=None: None)
    monkeypatch.setattr(OpenAIServer, "register_routes", lambda self: None)
    server = OpenAIServer(
        generator=generator,
        model="test-model",
        tool_parser=None,
        server_role=None,
        metadata_server_cfg=None,
    )
    server.metrics_collector = MagicMock()
    server.metrics_collector.log_ep_health_stats.return_value = True

    lifespan = server.app.router.lifespan_context(server.app)
    await asyncio.wait_for(lifespan.__aenter__(), timeout=1.0)
    exit_task = None
    try:
        assert await asyncio.wait_for(asyncio.to_thread(poll_started.wait, 1.0), timeout=2.0)
        exit_task = asyncio.create_task(lifespan.__aexit__(None, None, None))
        await asyncio.wait_for(server._ep_health_stop_event.wait(), timeout=1.0)
        for _ in range(3):
            exit_task.cancel()
            await asyncio.sleep(0)
            generator.shutdown.assert_not_called()
            assert not exit_task.done()
    finally:
        release_poll.set()

    assert exit_task is not None
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(exit_task, timeout=1.0)
    generator._get_ep_health_stats.assert_called_once_with()
    generator.shutdown.assert_called_once_with()
