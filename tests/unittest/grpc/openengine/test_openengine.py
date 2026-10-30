# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the OpenEngine gRPC adapter."""

import asyncio
import gc
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

grpc = pytest.importorskip(  # noqa: E402
    "grpc", reason='gRPC runtime not installed (pip install "grpcio>=1.73.1,<2")'
)
pytest.importorskip("grpc_health", reason="gRPC health runtime not installed")
from grpc_health.v1 import health_pb2, health_pb2_grpc  # noqa: E402

from tensorrt_llm.grpc.openengine.bindings import (  # noqa: E402
    kv_pb2,
    openengine_pb2_grpc,
    server_pb2,
)
from tensorrt_llm.grpc.openengine.server import OpenEngineServer  # noqa: E402
from tensorrt_llm.inputs.media_io import _get_aiohttp_session  # noqa: E402

# grpc.aio starts a `_poll_wrapper` daemon thread on server start and tears it
# down only once its internal state is released, which happens after the event
# loop closes -- later than pytest-threadleak's teardown snapshot. The thread is
# grpc's to own, not this test's, so exempt the module the same way the SMG
# adapter tests do.
pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]


def test_format_bind_address_brackets_ipv6() -> None:
    """A bare IPv6 host is bracketed; IPv4 and already-bracketed hosts are not."""
    from tensorrt_llm.grpc.openengine.server import _format_bind_address

    assert _format_bind_address("127.0.0.1", 8000) == "127.0.0.1:8000"
    assert _format_bind_address("::1", 8000) == "[::1]:8000"
    assert _format_bind_address("[::1]", 8000) == "[::1]:8000"


@pytest.mark.parametrize("fail_stop", [False, True])
@pytest.mark.parametrize("fail_metrics", [False, True])
def test_stop_drains_both_media_pools_even_if_shutdown_fails(fail_stop, fail_metrics) -> None:
    async def exercise_stop():
        server = OpenEngineServer.__new__(OpenEngineServer)
        server._server = SimpleNamespace(
            stop=AsyncMock(side_effect=RuntimeError("stop failed") if fail_stop else None)
        )
        server._health = SimpleNamespace(enter_graceful_shutdown=AsyncMock())
        server._health_task = None
        server._metrics = SimpleNamespace(
            stop=AsyncMock(side_effect=RuntimeError("metrics failed") if fail_metrics else None)
        )
        server._input_executor = ThreadPoolExecutor(max_workers=1)
        server._media_executor = ThreadPoolExecutor(max_workers=1)
        futures = [
            pool.submit(lambda: True) for pool in (server._input_executor, server._media_executor)
        ]
        media_session = await _get_aiohttp_session()
        if fail_stop or fail_metrics:
            message = "metrics failed" if fail_metrics else "stop failed"
            with pytest.raises(RuntimeError, match=message):
                await server.stop(grace=0)
        else:
            await server.stop(grace=0)
        assert media_session.closed
        assert server._stopping
        server._metrics.stop.assert_awaited_once()
        server._health.enter_graceful_shutdown.assert_awaited_once()
        assert all(future.result() for future in futures)
        for pool in (server._input_executor, server._media_executor):
            with pytest.raises(RuntimeError, match="shutdown"):
                pool.submit(lambda: None)

    asyncio.run(exercise_stop())


def test_openengine_server_serves_the_control_contract() -> None:
    """The server binds, attaches Control, answers, and shuts down cleanly.

    Exercises TRT-LLM's lifecycle around the servicer -- port-zero resolution,
    startup, reachability, graceful stop -- and that Control is wired in and
    answering rather than falling through to the generated base.
    """
    llm = SimpleNamespace(
        args=SimpleNamespace(
            max_seq_len=2048,
            max_batch_size=8,
            max_num_tokens=2048,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            context_parallel_size=1,
            guided_decoding_backend=None,
            reasoning_parser=None,
            kv_cache_config=SimpleNamespace(tokens_per_block=32),
        ),
        llm_id="test-instance",
        tokenizer=object(),
        _check_health=lambda: True,
    )

    async def exercise_server() -> None:
        server = OpenEngineServer(host="127.0.0.1", port=0, llm=llm, model="test-model")
        # port=0 must be replaced by the kernel-assigned port, or nothing
        # downstream (including this test's channel) can reach the server.
        assert server.port != 0

        await server.start()
        with pytest.raises(RuntimeError, match="Failed to bind"):
            OpenEngineServer(host="127.0.0.1", port=server.port, llm=llm, model="test-model")
        channel = grpc.aio.insecure_channel(f"127.0.0.1:{server.port}")
        try:
            control = openengine_pb2_grpc.ControlStub(channel)
            info = await control.GetServerInfo(server_pb2.GetServerInfoRequest(), timeout=5)
            assert info.engine_name == "tensorrt_llm"
            sources = await control.GetKvEventSources(kv_pb2.GetKvEventSourcesRequest(), timeout=5)
            assert list(sources.sources) == []
            with pytest.raises(grpc.aio.AioRpcError) as error:
                async for _ in control.SubscribeKvEvents(
                    kv_pb2.SubscribeKvEventsRequest(), timeout=5
                ):
                    pass
            assert error.value.code() == grpc.StatusCode.UNIMPLEMENTED
            assert info.schema_revision == 1
            assert info.minimum_client_revision == 1
            assert info.schema_release == "768a93c7b44e40f28c692ad0b471a8f2"
        finally:
            await channel.close()
            await server.stop(grace=0)

    asyncio.run(exercise_server())


def test_health_and_control_are_reachable_on_pod_address() -> None:
    """The shared listener accepts Pod-IP probes and control calls on one port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
            route.connect(("192.0.2.1", 9))
            pod_ip = route.getsockname()[0]
    except OSError:
        pytest.skip("No non-loopback IPv4 route is available")
    if pod_ip.startswith("127."):
        pytest.skip("No non-loopback IPv4 address is available")

    llm = SimpleNamespace(
        args=SimpleNamespace(guided_decoding_backend=None),
        llm_id="test-instance",
        tokenizer=None,
        _check_health=lambda: True,
    )

    async def exercise_server() -> None:
        server = OpenEngineServer("0.0.0.0", 0, llm, "test-model")
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"{pod_ip}:{server.port}") as channel:
                health = health_pb2_grpc.HealthStub(channel)
                response = await health.Check(
                    health_pb2.HealthCheckRequest(service="inference"), timeout=5
                )
                assert response.status == health_pb2.HealthCheckResponse.SERVING
                control = openengine_pb2_grpc.ControlStub(channel)
                sources = await control.GetKvEventSources(
                    kv_pb2.GetKvEventSourcesRequest(), timeout=5
                )
                assert list(sources.sources) == []
        finally:
            await server.stop(grace=0)

    asyncio.run(exercise_server())


def test_standard_grpc_health_check_and_watch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Readiness changes reach both RPCs without changing server liveness."""
    import tensorrt_llm.grpc.openengine.server as oe_server

    monkeypatch.setattr(oe_server, "_HEALTH_CHECK_INTERVAL_SECONDS", 0.01)
    engine = {"healthy": False}
    llm = SimpleNamespace(
        args=SimpleNamespace(guided_decoding_backend=None),
        llm_id="test-instance",
        tokenizer=None,
        _check_health=lambda: engine["healthy"],
    )

    async def exercise_server() -> None:
        server = OpenEngineServer("127.0.0.1", 0, llm, "test-model")
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{server.port}") as channel:
                health = health_pb2_grpc.HealthStub(channel)

                async def check(service: str) -> int:
                    response = await health.Check(
                        health_pb2.HealthCheckRequest(service=service), timeout=5
                    )
                    return response.status

                assert await check("liveness") == health_pb2.HealthCheckResponse.SERVING
                assert await check("inference") == health_pb2.HealthCheckResponse.NOT_SERVING
                assert await check("") == health_pb2.HealthCheckResponse.NOT_SERVING
                with pytest.raises(grpc.aio.AioRpcError) as unknown:
                    await check("unknown")
                assert unknown.value.code() == grpc.StatusCode.NOT_FOUND
                unknown_stream = health.Watch(health_pb2.HealthCheckRequest(service="unknown"))
                try:
                    assert (
                        await asyncio.wait_for(unknown_stream.read(), 5)
                    ).status == health_pb2.HealthCheckResponse.SERVICE_UNKNOWN
                finally:
                    unknown_stream.cancel()

                stream = health.Watch(health_pb2.HealthCheckRequest(service="inference"))
                remaining_stream = health.Watch(health_pb2.HealthCheckRequest(service="inference"))
                try:
                    initial = await asyncio.wait_for(stream.read(), 5)
                    assert initial.status == health_pb2.HealthCheckResponse.NOT_SERVING
                    remaining_initial = await asyncio.wait_for(remaining_stream.read(), 5)
                    assert remaining_initial.status == initial.status

                    engine["healthy"] = True
                    serving = await asyncio.wait_for(stream.read(), 5)
                    assert serving.status == health_pb2.HealthCheckResponse.SERVING
                    assert (
                        await asyncio.wait_for(remaining_stream.read(), 5)
                    ).status == serving.status
                    assert await check("inference") == serving.status
                    assert await check("") == serving.status

                    stream.cancel()
                    assert await asyncio.wait_for(stream.code(), 5) == grpc.StatusCode.CANCELLED
                    engine["healthy"] = False
                    unavailable = await asyncio.wait_for(remaining_stream.read(), 5)
                    assert unavailable.status == health_pb2.HealthCheckResponse.NOT_SERVING
                    assert await check("inference") == unavailable.status
                    assert await check("") == unavailable.status
                    assert await check("liveness") == health_pb2.HealthCheckResponse.SERVING
                finally:
                    stream.cancel()
                    remaining_stream.cancel()
        finally:
            await server.stop(grace=0)

    asyncio.run(exercise_server())


def test_standard_grpc_health_watch_reports_shutdown() -> None:
    """Shutdown withdraws inference and finishes watchers without failing liveness."""
    llm = SimpleNamespace(
        args=SimpleNamespace(guided_decoding_backend=None),
        llm_id="test-instance",
        tokenizer=None,
        _check_health=lambda: True,
    )

    async def exercise_server() -> None:
        server = OpenEngineServer("127.0.0.1", 0, llm, "test-model")
        await server.start()
        stop_task = None
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{server.port}") as channel:
                health = health_pb2_grpc.HealthStub(channel)
                liveness = health.Watch(health_pb2.HealthCheckRequest(service="liveness"))
                inference = health.Watch(health_pb2.HealthCheckRequest(service="inference"))
                for stream in (liveness, inference):
                    assert (
                        await asyncio.wait_for(stream.read(), 5)
                    ).status == health_pb2.HealthCheckResponse.SERVING

                stop_task = asyncio.create_task(server.stop(grace=10))
                assert (
                    await asyncio.wait_for(inference.read(), 5)
                ).status == health_pb2.HealthCheckResponse.NOT_SERVING
                assert await asyncio.wait_for(inference.read(), 5) is grpc.aio.EOF
                assert await asyncio.wait_for(liveness.read(), 5) is grpc.aio.EOF
                await asyncio.wait_for(stop_task, 5)
        finally:
            if stop_task is None:
                await server.stop(grace=0)
            elif not stop_task.done():
                await server.stop(grace=0)
                await stop_task

    asyncio.run(exercise_server())


def test_inference_health_timeout_does_not_start_overlapping_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stalled predicate withdraws readiness without accumulating worker threads."""
    import tensorrt_llm.grpc.openengine.server as oe_server

    monkeypatch.setattr(oe_server, "_HEALTH_CHECK_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(oe_server, "_HEALTH_CHECK_TIMEOUT_SECONDS", 0.05)
    release = threading.Event()
    calls = []

    def check_health() -> bool:
        calls.append(1)
        if len(calls) > 1:
            release.wait(5)
        return True

    llm = SimpleNamespace(
        args=SimpleNamespace(guided_decoding_backend=None),
        llm_id="test-instance",
        tokenizer=None,
        _check_health=check_health,
    )

    async def exercise_server() -> None:
        server = OpenEngineServer("127.0.0.1", 0, llm, "test-model")
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{server.port}") as channel:
                health = health_pb2_grpc.HealthStub(channel)
                inference = health_pb2.HealthCheckRequest(service="inference")
                stream = health.Watch(inference)
                try:
                    assert (
                        await asyncio.wait_for(stream.read(), 5)
                    ).status == health_pb2.HealthCheckResponse.SERVING
                    assert (
                        await asyncio.wait_for(stream.read(), 5)
                    ).status == health_pb2.HealthCheckResponse.NOT_SERVING
                    assert (
                        await health.Check(inference, timeout=5)
                    ).status == health_pb2.HealthCheckResponse.NOT_SERVING
                    assert (
                        await health.Check(
                            health_pb2.HealthCheckRequest(service="liveness"), timeout=5
                        )
                    ).status == health_pb2.HealthCheckResponse.SERVING
                    await asyncio.sleep(0.1)
                    assert len(calls) == 2
                    release.set()
                    assert (
                        await asyncio.wait_for(stream.read(), 5)
                    ).status == health_pb2.HealthCheckResponse.SERVING
                finally:
                    stream.cancel()
        finally:
            release.set()
            await server.stop(grace=0)

    asyncio.run(exercise_server())


@pytest.mark.parametrize("fatal_field", ["_fatal_error", "_engine_dead"])
def test_fatal_engine_failure_fails_liveness_after_temporary_unavailability(
    monkeypatch: pytest.MonkeyPatch, fatal_field: str
) -> None:
    """Only a sticky fatal error fails liveness; temporary readiness can recover."""
    import tensorrt_llm.grpc.openengine.server as oe_server

    monkeypatch.setattr(oe_server, "_HEALTH_CHECK_INTERVAL_SECONDS", 0.01)
    engine = {"healthy": True}
    executor = SimpleNamespace(_fatal_error=None, _engine_dead=False)
    llm = SimpleNamespace(
        args=SimpleNamespace(guided_decoding_backend=None),
        llm_id="test-instance",
        tokenizer=None,
        _executor=executor,
        _check_health=lambda: engine["healthy"] and executor._fatal_error is None,
    )

    async def exercise_server() -> None:
        server = OpenEngineServer("127.0.0.1", 0, llm, "test-model")
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{server.port}") as channel:
                health = health_pb2_grpc.HealthStub(channel)
                inference = health_pb2.HealthCheckRequest(service="inference")
                liveness = health_pb2.HealthCheckRequest(service="liveness")
                stream = health.Watch(inference)
                live_stream = health.Watch(liveness)
                try:
                    assert (
                        await asyncio.wait_for(stream.read(), 5)
                    ).status == health_pb2.HealthCheckResponse.SERVING
                    assert (
                        await asyncio.wait_for(live_stream.read(), 5)
                    ).status == health_pb2.HealthCheckResponse.SERVING
                    engine["healthy"] = False
                    assert (
                        await asyncio.wait_for(stream.read(), 5)
                    ).status == health_pb2.HealthCheckResponse.NOT_SERVING
                    assert (
                        await health.Check(liveness, timeout=5)
                    ).status == health_pb2.HealthCheckResponse.SERVING
                    engine["healthy"] = True
                    assert (
                        await asyncio.wait_for(stream.read(), 5)
                    ).status == health_pb2.HealthCheckResponse.SERVING
                    setattr(
                        executor,
                        fatal_field,
                        RuntimeError("engine failed") if fatal_field == "_fatal_error" else True,
                    )
                    assert (
                        await asyncio.wait_for(stream.read(), 5)
                    ).status == health_pb2.HealthCheckResponse.NOT_SERVING
                    assert (
                        await asyncio.wait_for(live_stream.read(), 5)
                    ).status == health_pb2.HealthCheckResponse.NOT_SERVING
                    setattr(executor, fatal_field, None if fatal_field == "_fatal_error" else False)
                    await asyncio.sleep(0.05)
                    assert (
                        await health.Check(inference, timeout=5)
                    ).status == health_pb2.HealthCheckResponse.NOT_SERVING
                    assert (
                        await health.Check(liveness, timeout=5)
                    ).status == health_pb2.HealthCheckResponse.NOT_SERVING
                finally:
                    stream.cancel()
                    live_stream.cancel()
        finally:
            await server.stop(grace=0)

    asyncio.run(exercise_server())


def test_inference_health_recovers_after_monitor_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed poll clears readiness and a later successful poll restores it."""
    import tensorrt_llm.grpc.openengine.server as oe_server

    monkeypatch.setattr(oe_server, "_HEALTH_CHECK_INTERVAL_SECONDS", 0.01)
    engine = {"raise": False}

    def check_health() -> bool:
        if engine["raise"]:
            raise TypeError("health predicate failed")
        return True

    llm = SimpleNamespace(
        args=SimpleNamespace(guided_decoding_backend=None),
        llm_id="test-instance",
        tokenizer=None,
        _check_health=check_health,
    )

    async def exercise_server() -> None:
        server = OpenEngineServer("127.0.0.1", 0, llm, "test-model")
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{server.port}") as channel:
                health = health_pb2_grpc.HealthStub(channel)
                inference = health_pb2.HealthCheckRequest(service="inference")
                liveness = health_pb2.HealthCheckRequest(service="liveness")
                assert (
                    await health.Check(inference, timeout=5)
                ).status == health_pb2.HealthCheckResponse.SERVING

                stream = health.Watch(inference)
                try:
                    assert (
                        await asyncio.wait_for(stream.read(), 5)
                    ).status == health_pb2.HealthCheckResponse.SERVING
                    engine["raise"] = True
                    assert (
                        await asyncio.wait_for(stream.read(), 5)
                    ).status == health_pb2.HealthCheckResponse.NOT_SERVING
                    assert (
                        await health.Check(inference, timeout=5)
                    ).status == health_pb2.HealthCheckResponse.NOT_SERVING
                    assert (
                        await health.Check(health_pb2.HealthCheckRequest(), timeout=5)
                    ).status == health_pb2.HealthCheckResponse.NOT_SERVING
                    assert (
                        await health.Check(liveness, timeout=5)
                    ).status == health_pb2.HealthCheckResponse.SERVING
                    engine["raise"] = False
                    assert (
                        await asyncio.wait_for(stream.read(), 5)
                    ).status == health_pb2.HealthCheckResponse.SERVING
                finally:
                    stream.cancel()
        finally:
            await server.stop(grace=0)

    asyncio.run(exercise_server())


@pytest.mark.parametrize("publisher_enabled", [False, True])
def test_event_startup_requires_loaded_publisher(publisher_enabled: bool) -> None:
    """Configured events must not advertise a source absent from the loaded engine."""
    from tensorrt_llm.grpc.openengine.kv_events import KvEventsUnavailableError

    llm = SimpleNamespace(
        args=SimpleNamespace(
            guided_decoding_backend=None,
            kv_cache_config=SimpleNamespace(
                kv_events_config=SimpleNamespace(enable_kv_cache_events=True, publisher="zmq")
            ),
        ),
        _executor=SimpleNamespace(
            get_kv_cache_capacity=lambda: {"kvEventsEnabled": True} if publisher_enabled else {},
            is_shutdown=lambda: False,
        ),
    )

    async def exercise_server() -> None:
        server = OpenEngineServer(host="127.0.0.1", port=0, llm=llm, model="test-model")
        try:
            if publisher_enabled:
                await server.start()
            else:
                with pytest.raises(KvEventsUnavailableError, match="no streaming publisher"):
                    await server.start()
        finally:
            await server.stop(grace=0)

    asyncio.run(exercise_server())


def test_is_loopback_classifies_bind_hosts() -> None:
    """The listener is unauthenticated, so a non-loopback bind must be flagged."""
    from tensorrt_llm.grpc.openengine.server import _is_loopback

    assert _is_loopback("127.0.0.1")
    assert _is_loopback("::1")
    assert _is_loopback("[::1]")
    assert _is_loopback("localhost")
    assert not _is_loopback("0.0.0.0")
    assert not _is_loopback("10.0.0.7")
    # An unresolvable name is not assumed safe.
    assert not _is_loopback("some-host")


class _StopLaunch(Exception):
    """Ends launch_server once the server would start accepting requests."""


@pytest.mark.parametrize(
    "request_stats,iteration_stats,output_dir,enabled",
    [
        (False, False, None, False),
        (True, False, None, True),
        (False, True, None, False),
        (True, True, None, True),
        ("false", False, None, False),
        ("0", False, None, False),
        ("off", True, None, False),
        (False, False, "/unused/perf-metrics", True),
    ],
)
@pytest.mark.parametrize("metrics_options", [{}, {"metrics_port": 0}])
def test_metrics_listener_preserves_engine_collection_config(
    monkeypatch: pytest.MonkeyPatch,
    request_stats: bool | str,
    iteration_stats: bool,
    output_dir: str | None,
    enabled: bool,
    metrics_options: dict,
) -> None:
    """Schema coercion and file-output opt-in must match HTTP; ports never enable export."""
    created = []
    import tensorrt_llm.grpc.openengine.server as oe_server

    class Model:
        def __init__(self, **kwargs) -> None:
            assert kwargs["return_perf_metrics"] == (
                True if output_dir is not None else request_stats
            )
            assert kwargs["enable_iter_perf_stats"] is iteration_stats
            self.llm_id = "test-instance"

        def shutdown(self) -> None:
            pass

    class Metrics:
        def __init__(self, llm, model, host, port, **kwargs) -> None:
            created.append((host, port))

        async def wait_for_collection(self) -> None:
            pass

    class Server:
        def __init__(self, **kwargs) -> None:
            pass

        async def start(self) -> None:
            raise _StopLaunch

        async def stop(self) -> None:
            pass

    monkeypatch.setattr(oe_server, "PyTorchLLM", Model)
    monkeypatch.setattr(oe_server, "OpenEngineMetrics", Metrics)
    monkeypatch.setattr(oe_server, "OpenEngineServer", Server)
    with pytest.raises(_StopLaunch):
        oe_server.launch_server(
            "127.0.0.1",
            0,
            {
                "model": "test-model",
                "backend": "pytorch",
                "return_perf_metrics": request_stats,
                "enable_iter_perf_stats": iteration_stats,
                "perf_metrics_output_dir": output_dir,
            },
            **metrics_options,
        )
    assert created == (
        [("127.0.0.1", metrics_options.get("metrics_port", 8001))] if enabled else []
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--openengine-metrics-port", "8001"], "requires --grpc --grpc-protocol openengine"),
        (
            ["--grpc", "--openengine-metrics-port", "8001"],
            "requires --grpc --grpc-protocol openengine",
        ),
        (["--openengine-metrics-host", "0.0.0.0"], "requires --grpc --grpc-protocol openengine"),
        (
            [
                "--grpc",
                "--grpc-protocol",
                "openengine",
                "--port",
                "8001",
                "--openengine-metrics-port",
                "8001",
            ],
            "must differ from --port",
        ),
    ],
)
def test_metrics_cli_rejects_incompatible_server_options(
    arguments: list[str], message: str
) -> None:
    """Reject invalid metrics deployment combinations before attempting model load."""
    from click.testing import CliRunner

    from tensorrt_llm.commands.serve import serve

    result = CliRunner().invoke(serve, ["nonexistent-model", *arguments])
    assert result.exit_code == 2
    assert message in result.output


@pytest.mark.parametrize(
    ("value", "gc_enabled"),
    [("1", False), ("0", True), (None, True)],
)
def test_launch_server_disables_gc_only_when_requested(
    monkeypatch: pytest.MonkeyPatch, value: str | None, gc_enabled: bool
) -> None:
    """TRTLLM_SERVER_DISABLE_GC=1 turns cyclic GC off before serving, as trtllm-serve does."""
    import tensorrt_llm.grpc.openengine.server as oe_server

    if value is None:
        monkeypatch.delenv("TRTLLM_SERVER_DISABLE_GC", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_SERVER_DISABLE_GC", value)
    seen = {}

    class _Llm:
        def __init__(self, **kwargs) -> None:
            seen["engine_id"] = kwargs["_openengine_discovery"]["engine_id"]
            self.llm_id = "test-instance"

        def shutdown(self) -> None:
            pass

    class _Server:
        def __init__(self, **kwargs) -> None:
            seen["instance_id"] = kwargs["instance_id"]

        async def start(self) -> None:
            seen["gc_enabled"] = gc.isenabled()
            raise _StopLaunch

        async def stop(self) -> None:
            pass

    monkeypatch.setattr(oe_server, "PyTorchLLM", _Llm)
    monkeypatch.setattr(oe_server, "OpenEngineServer", _Server)
    gc_was_enabled = gc.isenabled()
    gc.enable()
    try:
        with pytest.raises(_StopLaunch):
            oe_server.launch_server("127.0.0.1", 0, {"backend": "pytorch", "model": "test-model"})
    finally:
        if gc_was_enabled:
            gc.enable()
        else:
            gc.disable()

    assert seen["gc_enabled"] is gc_enabled
    assert seen["instance_id"] == seen["engine_id"]
