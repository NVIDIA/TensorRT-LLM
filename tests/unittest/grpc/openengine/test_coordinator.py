# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise request ownership over real private gRPC sockets."""

import asyncio
from contextlib import asynccontextmanager, suppress
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

grpc = pytest.importorskip("grpc", reason="gRPC runtime not installed")
pytest.importorskip("grpc_health", reason="gRPC health runtime not installed")
from grpc_health.v1 import health_pb2, health_pb2_grpc  # noqa: E402

import tensorrt_llm.grpc.openengine.coordinator as coordination  # noqa: E402
from tensorrt_llm.grpc.openengine.bindings import (  # noqa: E402
    generation_pb2,
    lifecycle_pb2,
    openengine_pb2_grpc,
)
from tensorrt_llm.grpc.openengine.coordinator import (  # noqa: E402
    CoordinationError,
    Coordinator,
    DuplicateRequestError,
    FrontendClient,
)
from tensorrt_llm.grpc.openengine.server import OpenEngineServer  # noqa: E402

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]


class Handle:
    def __init__(self, refuses=False):
        self.aborted = False
        self.refuses = refuses

    def abort(self):
        if self.refuses:
            raise RuntimeError("engine refused abort")
        self.aborted = True


@asynccontextmanager
async def group(coordinator_type=Coordinator, client_type=FrontendClient, count=2):
    # Short paths avoid the AF_UNIX pathname limit under pytest's tmp_path.
    with TemporaryDirectory(prefix="oe-test-") as directory:
        failures = []
        coordinator = coordinator_type(
            directory, count, "shared-engine", lambda: True, failures.append
        )
        clients = [client_type(directory, i, failures.append) for i in range(count)]
        await coordinator.start()
        for client in clients:
            await client.start()
        coordinator.ready = True
        try:
            yield coordinator, clients, failures
        finally:
            for client in clients:
                await client.close()
            await coordinator.close()


@pytest.mark.asyncio
async def test_cross_frontend_ownership_load_and_abort():
    """Duplicate IDs or local-only control would misroute/corrupt engine work."""
    async with group() as (_, (first, second), failures):
        reservation = await first.reserve("request")
        handle = Handle()
        reservation.bind(handle)
        with pytest.raises(DuplicateRequestError):
            await second.reserve("request")
        status = await second.request("status")
        assert status == {
            "count": 1,
            "ready": True,
            "instance_id": "shared-engine",
            "stopping": False,
            "shutdown_pending": False,
        }
        assert await second.request("abort", request_id="request") == {
            "aborted": 1,
            "failed": 0,
        }
        assert handle.aborted
        assert (await second.request("status"))["count"] == 1
        await reservation.release()
        replacement = await second.reserve("request")
        refused = await first.reserve("refuses")
        refused.bind(Handle(refuses=True))
        assert await second.request("abort") == {"aborted": 1, "failed": 1}
        assert replacement.aborted  # Abort before engine submission is retained.
        deferred = Handle()
        replacement.bind(deferred)
        assert deferred.aborted
        assert not failures


@pytest.mark.asyncio
async def test_forwarded_shutdown_rejections_do_not_fail_group():
    """A frontend with stale identity must not turn rejection into engine death."""
    from conftest import AbortError, FakeServicerContext

    async with group() as (coordinator, (first, second), failures):
        transitions = []

        def stop():
            already = bool(transitions)
            if not already:
                transitions.append("stop")
            coordinator.ready = False
            coordinator.stopping = True
            return already

        coordinator._request_shutdown = stop
        llm = SimpleNamespace(
            args=SimpleNamespace(guided_decoding_backend=None),
            tokenizer=None,
            llm_id="old-engine",
            _check_health=lambda: True,
        )
        server = OpenEngineServer(
            "127.0.0.1", 0, llm, "test", frontend=first, instance_id="old-engine"
        )
        await server.start()
        control = server._control
        try:
            for expected, code in [
                ("", grpc.StatusCode.INVALID_ARGUMENT),
                ("old-engine", grpc.StatusCode.FAILED_PRECONDITION),
            ]:
                with pytest.raises(AbortError) as rejected:
                    await control.Shutdown(
                        lifecycle_pb2.ShutdownRequest(expected_instance_id=expected),
                        FakeServicerContext(),
                    )
                assert rejected.value.code == code
                assert (await second.request("status"))["ready"]
                assert not failures
                assert not transitions

            control._instance_id = "shared-engine"
            replies = await asyncio.gather(
                *(
                    control.Shutdown(
                        lifecycle_pb2.ShutdownRequest(expected_instance_id="shared-engine"),
                        FakeServicerContext(),
                    )
                    for _ in range(3)
                )
            )
            assert sorted(reply.status for reply in replies) == [
                lifecycle_pb2.SHUTDOWN_STATUS_ACCEPTED,
                lifecycle_pb2.SHUTDOWN_STATUS_ALREADY_STOPPING,
                lifecycle_pb2.SHUTDOWN_STATUS_ALREADY_STOPPING,
            ]
            with pytest.raises(CoordinationError):
                await second.reserve("after-stop")
            assert transitions == ["stop"]
            assert not failures
        finally:
            await server.stop()


@pytest.mark.asyncio
async def test_cancelled_shutdown_keeps_launcher_commit():
    """Losing the forwarding reply must not undo the launcher's stop event."""
    committed, release_reply = asyncio.Event(), asyncio.Event()

    class DelayedShutdown(Coordinator):
        async def _dispatch(self, message, context):
            result = await super()._dispatch(message, context)
            if message["operation"] == "shutdown":
                committed.set()
                await release_reply.wait()
            return result

    async with group(DelayedShutdown) as (coordinator, (first, second), failures):

        def stop():
            already = coordinator.stopping
            coordinator.stopping = True
            coordinator.ready = False
            return already

        coordinator._request_shutdown = stop
        request = asyncio.create_task(
            first.request("shutdown", expected_instance_id="shared-engine")
        )
        try:
            await asyncio.wait_for(committed.wait(), 2)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            assert (await second.request("status"))["stopping"]
            with pytest.raises(CoordinationError):
                await second.reserve("after-cancel")
            assert not failures
        finally:
            release_reply.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("peer_failure", [False, True])
async def test_cancelled_shutdown_finishes_group_publication(peer_failure):
    """A cancelled reply cannot strand a committed shutdown during fanout."""
    async with group() as (coordinator, (first, _), failures):
        entered, release, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original = coordinator._peers[0].request

        async def delayed(message):
            entered.set()
            await release.wait()
            if peer_failure:
                raise RuntimeError("injected peer publication failure")
            return await original(message)

        coordinator._peers[0].request = delayed

        def stop():
            stopped.set()
            return False

        coordinator._request_shutdown = stop
        task = asyncio.create_task(first.request("shutdown", expected_instance_id="shared-engine"))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            heartbeat = await first.request("heartbeat", frontend=first.frontend_id)
            assert heartbeat["stopping"] and heartbeat["shutdown_pending"]
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            release.set()
            await asyncio.wait_for(stopped.wait(), 2)
            await asyncio.gather(coordinator._shutdown_task, return_exceptions=True)
            assert coordinator.stopping
            assert bool(failures) == peer_failure
        finally:
            release.set()


@pytest.mark.asyncio
async def test_shutdown_wins_over_stale_probe_status():
    """A delayed healthy coordinator response cannot resurrect local readiness."""
    from conftest import FakeServicerContext

    from tensorrt_llm.grpc.openengine.control import OpenEngineControlServicer
    from tensorrt_llm.grpc.openengine.servicer import OpenEngineInferenceServicer

    entered, release = asyncio.Event(), asyncio.Event()
    stopping = False

    class Frontend:
        calls = 0

        async def request(self, operation):
            self.calls += 1
            if self.calls == 2:
                entered.set()
                await release.wait()
            return {"ready": True, "stopping": False}

    llm = SimpleNamespace(
        args=SimpleNamespace(guided_decoding_backend=None),
        tokenizer=None,
        _check_health=lambda: True,
    )
    inference = OpenEngineInferenceServicer(llm, "test", is_stopping=lambda: stopping)
    control = OpenEngineControlServicer(llm, "test", inference, frontend=Frontend())

    async def probe(model):
        return lifecycle_pb2.HealthCheck(state=lifecycle_pb2.HEALTH_STATE_READY)

    control._inference_probe = probe
    task = asyncio.create_task(
        control.Health(
            lifecycle_pb2.HealthRequest(include_inference_probe=True), FakeServicerContext()
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        stopping = True
        release.set()
        assert (await asyncio.wait_for(task, 2)).state == lifecycle_pb2.HEALTH_STATE_NOT_READY
    finally:
        release.set()
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


@pytest.mark.asyncio
async def test_shutdown_wins_over_pending_group_probe_admission():
    """A probe rejected by stopping group admission must report NOT_READY."""
    from conftest import FakeServicerContext

    from tensorrt_llm.grpc.openengine.control import OpenEngineControlServicer

    async with group() as (coordinator, (first, _), failures):
        entered, release = asyncio.Event(), asyncio.Event()
        original_reserve = first.reserve

        async def delayed_reserve(request_id):
            entered.set()
            await release.wait()
            return await original_reserve(request_id)

        first.reserve = delayed_reserve
        from tensorrt_llm.grpc.openengine.servicer import OpenEngineInferenceServicer

        llm = SimpleNamespace(
            args=SimpleNamespace(guided_decoding_backend=None),
            tokenizer=None,
            _check_health=lambda: True,
        )
        control = OpenEngineControlServicer(
            llm,
            "test",
            OpenEngineInferenceServicer(llm, "test", frontend=first),
            frontend=first,
            instance_id="shared-engine",
        )
        task = asyncio.create_task(
            control.Health(
                lifecycle_pb2.HealthRequest(include_inference_probe=True), FakeServicerContext()
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 2)
            coordinator.ready = False
            coordinator.stopping = True
            release.set()
            response = await asyncio.wait_for(task, 2)
            assert response.state == lifecycle_pb2.HEALTH_STATE_NOT_READY
            assert not failures
        finally:
            release.set()
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task


@pytest.mark.asyncio
async def test_forwarded_shutdown_withdraws_health_before_owner_cleanup():
    """An attached frontend must publish unready before acknowledging shutdown."""
    async with group() as (coordinator, (_, frontend), failures):

        def request_stop():
            already = coordinator.stopping
            coordinator.ready = False
            coordinator.stopping = True
            return already

        coordinator._request_shutdown = request_stop
        llm = SimpleNamespace(
            args=SimpleNamespace(guided_decoding_backend=None),
            llm_id="shared-engine",
            tokenizer=None,
            _check_health=lambda: True,
        )
        server = OpenEngineServer("127.0.0.1", 0, llm, "test", frontend=frontend)
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{server.port}") as channel:
                control = openengine_pb2_grpc.ControlStub(channel)
                response = await control.Shutdown(
                    lifecycle_pb2.ShutdownRequest(expected_instance_id="shared-engine"), timeout=3
                )
                assert response.status == lifecycle_pb2.SHUTDOWN_STATUS_ACCEPTED
                health = health_pb2_grpc.HealthStub(channel)
                assert (
                    await health.Check(
                        health_pb2.HealthCheckRequest(service="readiness"), timeout=3
                    )
                ).status == health_pb2.HealthCheckResponse.NOT_SERVING
                assert (
                    await health.Check(health_pb2.HealthCheckRequest(service="liveness"), timeout=3)
                ).status == health_pb2.HealthCheckResponse.SERVING
                assert not failures
        finally:
            await server.stop(grace=0)


@pytest.mark.asyncio
async def test_shutdown_publishes_readiness_on_every_frontend_before_reply():
    """An attached peer must not stay ready after another frontend acknowledges."""
    async with group(count=3) as (coordinator, clients, failures):

        def stop():
            already = coordinator.stopping
            coordinator.ready = False
            coordinator.stopping = True
            return already

        coordinator._request_shutdown = stop
        llm = SimpleNamespace(
            args=SimpleNamespace(guided_decoding_backend=None),
            llm_id="shared-engine",
            tokenizer=None,
            _check_health=lambda: True,
        )
        servers = [OpenEngineServer("127.0.0.1", 0, llm, "test", frontend=c) for c in clients]
        channels = []
        try:
            for server in servers:
                await server.start()
                channels.append(grpc.aio.insecure_channel(f"127.0.0.1:{server.port}"))
            for channel in channels:
                status = await health_pb2_grpc.HealthStub(channel).Check(
                    health_pb2.HealthCheckRequest(service="readiness"), timeout=3
                )
                assert status.status == health_pb2.HealthCheckResponse.SERVING
            reply = await openengine_pb2_grpc.ControlStub(channels[1]).Shutdown(
                lifecycle_pb2.ShutdownRequest(expected_instance_id="shared-engine"), timeout=3
            )
            assert reply.status == lifecycle_pb2.SHUTDOWN_STATUS_ACCEPTED
            for channel in channels:
                status = await health_pb2_grpc.HealthStub(channel).Check(
                    health_pb2.HealthCheckRequest(service="readiness"), timeout=3
                )
                assert status.status == health_pb2.HealthCheckResponse.NOT_SERVING
            assert not failures
        finally:
            for channel in channels:
                await channel.close()
            for server in servers:
                await server.stop(grace=0)


@pytest.mark.asyncio
async def test_cancelled_reservation_releases_after_commit_before_reply():
    """Cancellation at the IPC boundary must not permanently occupy an ID."""
    committed, reply_allowed = asyncio.Event(), asyncio.Event()

    class DelayedCoordinator(Coordinator):
        async def _dispatch(self, message, context):
            result = await super()._dispatch(message, context)
            if message["operation"] == "reserve":
                committed.set()
                await reply_allowed.wait()
            return result

    async with group(DelayedCoordinator) as (_, (first, second), failures):
        task = asyncio.create_task(first.reserve("cancelled"))
        await asyncio.wait_for(committed.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (await second.request("status"))["count"] == 1
        reply_allowed.set()

        # Cleanup is owned by the frontend, independently of the cancelled RPC.
        async def released():
            while (await second.request("status"))["count"]:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(released(), 2)
        assert (await second.request("status"))["count"] == 0
        replacement = await second.reserve("cancelled")
        await replacement.release()
        assert not failures


@pytest.mark.asyncio
@pytest.mark.parametrize("all_requests", [False, True])
async def test_delayed_abort_cannot_cancel_reused_request_id(all_requests):
    """An abort snapshot must stay fenced when the old stream finishes."""
    async with group() as (coordinator, (first, second), failures):
        old = await first.reserve("reused")
        entered, proceed = asyncio.Event(), asyncio.Event()
        original = coordinator._peers[0].request

        async def delayed(message):
            entered.set()
            await proceed.wait()
            return await original(message)

        coordinator._peers[0].request = delayed
        fields = {} if all_requests else {"request_id": "reused"}
        abort = asyncio.create_task(second.request("abort", **fields))
        await asyncio.wait_for(entered.wait(), 2)
        await old.release()
        replacement = await first.reserve("reused")
        # Replayed cleanup from the previous stream cannot release its successor.
        await first.request(
            "release",
            request_id=old.request_id,
            token=old.token,
            frontend=first.frontend_id,
        )
        handle = Handle()
        replacement.bind(handle)
        proceed.set()
        assert await abort == {"aborted": 0, "failed": 0}
        assert not handle.aborted
        assert (await second.request("status"))["count"] == 1
        assert not failures


@pytest.mark.asyncio
async def test_coordinator_loss_fails_closed():
    """An unknown reservation outcome must stop admission, not strand work."""
    async with group() as (coordinator, (first, _), failures):
        await coordinator.close()
        with pytest.raises(CoordinationError):
            await first.reserve("unknown")
        assert failures
        for reservation in list(first.reservations.values()):
            with suppress(CoordinationError):
                await reservation.release()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model, expected_code",
    [
        ("test-model", grpc.StatusCode.UNAVAILABLE),
        ("", grpc.StatusCode.INVALID_ARGUMENT),
    ],
)
async def test_generate_cleanup_failure_preserves_grpc_status(model, expected_code):
    """Lost release acknowledgments must not escape as UNKNOWN or mask an abort."""
    async with group() as (coordinator, (frontend, _), failures):

        class ResultHandle(Handle):
            async def __aiter__(self):
                yield SimpleNamespace(
                    prompt_token_ids=[1],
                    outputs=[],
                    cached_tokens=0,
                    error="engine stopped",
                    finished=True,
                )

        handle = ResultHandle()
        llm = SimpleNamespace(
            llm_id="shared-engine",
            args=SimpleNamespace(guided_decoding_backend=None),
            tokenizer=None,
            generate_async=lambda **kwargs: handle,
        )
        original_request = frontend.request
        release_attempted = False

        async def lose_coordinator_on_release(operation, **fields):
            nonlocal release_attempted
            if operation == "release":
                release_attempted = True
                await coordinator.close()
            return await original_request(operation, **fields)

        frontend.request = lose_coordinator_on_release
        server = OpenEngineServer("127.0.0.1", 0, llm, "test-model", frontend=frontend)
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{server.port}") as channel:
                stream = openengine_pb2_grpc.InferenceStub(channel).Generate(
                    generation_pb2.GenerateRequest(
                        request_id="cleanup-failure",
                        model=model,
                        token_ids=generation_pb2.TokenIds(ids=[1]),
                    ),
                    timeout=5,
                )
                with pytest.raises(grpc.aio.AioRpcError) as error:
                    async for _ in stream:
                        pass
                assert error.value.code() == expected_code

            # An already-aborted RPC can deliver its status before the owned
            # cleanup task finishes. Observe cleanup without driving it ourselves.
            async def cleaned_up():
                while frontend.reservations:
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(cleaned_up(), 2)
            assert release_attempted
            assert failures
        finally:
            await server.stop()


@pytest.mark.asyncio
async def test_standard_health_tracks_frontend_group_readiness():
    async with group() as (coordinator, (frontend, _), _):
        llm = SimpleNamespace(
            llm_id="shared-engine",
            args=SimpleNamespace(guided_decoding_backend=None),
            tokenizer=None,
            _check_health=lambda: True,
        )
        server = OpenEngineServer("127.0.0.1", 0, llm, "test-model", frontend=frontend)
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{server.port}") as channel:
                health = health_pb2_grpc.HealthStub(channel)
                control = openengine_pb2_grpc.ControlStub(channel)
                request = health_pb2.HealthCheckRequest(service="readiness")
                assert (
                    await health.Check(request, timeout=5)
                ).status == health_pb2.HealthCheckResponse.SERVING
                assert (
                    await control.Health(lifecycle_pb2.HealthRequest(), timeout=5)
                ).state == lifecycle_pb2.HEALTH_STATE_READY

                coordinator.ready = False
                await server._refresh_inference_health()
                assert (
                    await health.Check(request, timeout=5)
                ).status == health_pb2.HealthCheckResponse.NOT_SERVING
                assert (
                    await control.Health(lifecycle_pb2.HealthRequest(), timeout=5)
                ).state == lifecycle_pb2.HEALTH_STATE_NOT_READY
                assert (
                    await health.Check(health_pb2.HealthCheckRequest(service="liveness"), timeout=5)
                ).status == health_pb2.HealthCheckResponse.SERVING

                coordinator.ready = True
                await server._refresh_inference_health()
                assert (
                    await health.Check(request, timeout=5)
                ).status == health_pb2.HealthCheckResponse.SERVING
                assert (
                    await control.Health(lifecycle_pb2.HealthRequest(), timeout=5)
                ).state == lifecycle_pb2.HEALTH_STATE_READY

                coordinator.stopping = True
                await server._refresh_inference_health()
                assert (
                    await health.Check(request, timeout=5)
                ).status == health_pb2.HealthCheckResponse.NOT_SERVING
                assert (
                    await control.Health(lifecycle_pb2.HealthRequest(), timeout=5)
                ).state == lifecycle_pb2.HEALTH_STATE_NOT_READY
        finally:
            await server.stop()


@pytest.mark.asyncio
async def test_health_probe_cleanup_failure_is_unavailable():
    """Coordinator loss after a completed probe must not surface as UNKNOWN."""
    async with group() as (coordinator, (frontend, _), failures):

        class ProbeHandle(Handle):
            async def aresult(self):
                return None

        llm = SimpleNamespace(
            llm_id="shared-engine",
            args=SimpleNamespace(guided_decoding_backend=None),
            tokenizer=None,
            _check_health=lambda: True,
            generate_async=lambda *args, **kwargs: ProbeHandle(),
        )
        original_request = frontend.request

        async def lose_coordinator_on_release(operation, **fields):
            if operation == "release":
                await coordinator.close()
            return await original_request(operation, **fields)

        frontend.request = lose_coordinator_on_release
        server = OpenEngineServer("127.0.0.1", 0, llm, "test-model", frontend=frontend)
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{server.port}") as channel:
                control = openengine_pb2_grpc.ControlStub(channel)
                with pytest.raises(grpc.aio.AioRpcError) as error:
                    await control.Health(
                        lifecycle_pb2.HealthRequest(include_inference_probe=True),
                        timeout=5,
                    )
                assert error.value.code() == grpc.StatusCode.UNAVAILABLE
            assert failures
        finally:
            await server.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_id",
    ["\x01" * (12 * 1024 * 1024), "é" * 513],
    ids=["json-expansion", "utf8-size"],
)
async def test_oversized_request_id_does_not_stop_group(request_id):
    """A rejected public ID must not kill unrelated requests or strand ownership."""
    async with group() as (_, (first, second), failures):
        llm = SimpleNamespace(
            llm_id="shared-engine",
            args=SimpleNamespace(guided_decoding_backend=None),
            tokenizer=None,
        )
        server = OpenEngineServer("127.0.0.1", 0, llm, "test-model", frontend=first)
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{server.port}") as channel:
                inference = openengine_pb2_grpc.InferenceStub(channel)
                control = openengine_pb2_grpc.ControlStub(channel)
                with pytest.raises(grpc.aio.AioRpcError) as generate_error:
                    await inference.Generate(
                        generation_pb2.GenerateRequest(request_id=request_id, model="test-model"),
                        timeout=5,
                    ).read()
                assert generate_error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
                with pytest.raises(grpc.aio.AioRpcError) as abort_error:
                    await control.Abort(
                        lifecycle_pb2.AbortRequest(request_id=request_id), timeout=5
                    )
                assert abort_error.value.code() == grpc.StatusCode.INVALID_ARGUMENT

            assert not first.reservations
            assert (await second.request("status"))["count"] == 0
            # A boundary-length UTF-8 ID still supports remote abort and release.
            accepted_id = "é" * 512
            reservation = await first.reserve(accepted_id)
            handle = Handle()
            reservation.bind(handle)
            assert await second.request("abort", request_id=accepted_id) == {
                "aborted": 1,
                "failed": 0,
            }
            assert handle.aborted
            await reservation.release()
            assert (await second.request("status"))["ready"]
            assert not failures
        finally:
            await server.stop()


@pytest.mark.asyncio
async def test_abort_all_batches_preserve_outcomes(monkeypatch):
    """Large valid snapshots must fit the transport and abort every generation."""
    # Lower the real transport limit so the unbatched snapshot fails without a
    # 64 MiB fixture. Production batches still fit, even with worst-case escaping.
    monkeypatch.setattr(
        coordination,
        "_OPTIONS",
        [
            ("grpc.max_receive_message_length", 1024 * 1024),
            ("grpc.max_send_message_length", 1024 * 1024),
        ],
    )
    async with group() as (_, (first, second), failures):
        handles = []
        for i in range(257):
            reservation = await first.reserve("\x01" * 1021 + f"{i:03d}")
            handle = Handle(refuses=i == 128)
            reservation.bind(handle)
            handles.append(handle)
        assert await second.request("abort") == {"aborted": 256, "failed": 1}
        assert all(handle.aborted != handle.refuses for handle in handles)
        assert (await second.request("status"))["ready"]
        assert not failures


@pytest.mark.asyncio
async def test_abort_all_deadline_preserves_group_and_partial_outcomes(monkeypatch):
    """Exhausting the snapshot budget must cancel dispatch without killing serving."""
    monkeypatch.setattr(coordination, "_ABORT_TIMEOUT", 2.0)
    monkeypatch.setattr(coordination, "_RPC_TIMEOUT", 1.0)
    monkeypatch.setattr(coordination, "_ABORT_BATCH_SIZE", 1)
    async with group() as (coordinator, (first, second), failures):
        for i in range(3):
            reservation = await first.reserve(str(i))
            reservation.bind(Handle())
        original_request = coordinator._peers[0].request
        calls = 0
        cancelled = asyncio.Event()

        async def delayed_batch(message):
            nonlocal calls
            calls += 1
            if calls == 1:
                return await original_request(message)
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        monkeypatch.setattr(coordinator._peers[0], "request", delayed_batch)
        assert await second.request("abort") == {"aborted": 1, "failed": 2}
        assert cancelled.is_set()
        assert calls == 2
        assert (await second.request("status"))["ready"]
        assert not failures


@pytest.mark.asyncio
async def test_peer_abort_rpc_deadline_preserves_group(monkeypatch):
    """A slow peer's gRPC deadline is an incomplete abort, not a group failure."""

    class SlowFrontend(FrontendClient):
        async def _dispatch(self, message, context):
            await asyncio.sleep(0.2)
            return await super()._dispatch(message, context)

    async with group(client_type=SlowFrontend) as (_, (first, second), failures):
        reservation = await first.reserve("slow")
        reservation.bind(Handle())
        with monkeypatch.context() as patch:
            patch.setattr(coordination, "_RPC_TIMEOUT", 0.05)
            assert await second.request("abort") == {"aborted": 0, "failed": 1}
        assert (await second.request("status"))["ready"]
        assert not failures
        await reservation.release()


@pytest.mark.asyncio
async def test_unavailable_abort_peer_cancels_slow_sibling(monkeypatch):
    """A lost peer must stop admission before another peer's abort deadline."""
    async with group() as (coordinator, (first, second), failures):
        unavailable = await first.reserve("unavailable")
        slow = await second.reserve("slow")
        await first._server.stop(0)
        slow_started = asyncio.Event()
        slow_cancelled = asyncio.Event()
        unavailable_request = coordinator._peers[0].request

        async def peer_unavailable(message):
            await slow_started.wait()
            return await unavailable_request(message)

        async def slow_peer(message):
            slow_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                slow_cancelled.set()

        monkeypatch.setattr(coordinator._peers[0], "request", peer_unavailable)
        monkeypatch.setattr(coordinator._peers[1], "request", slow_peer)
        with pytest.raises(CoordinationError):
            await asyncio.wait_for(second.request("abort"), timeout=2)
        assert slow_cancelled.is_set()
        assert not coordinator.ready
        assert failures
        await unavailable.release()
        await slow.release()


@pytest.mark.asyncio
async def test_expired_heartbeat_stops_admission_and_cannot_be_revived(monkeypatch):
    """A live but unresponsive frontend must invalidate the whole group's readiness."""
    now = 100.0
    monkeypatch.setattr(coordination, "monotonic", lambda: now)
    async with group() as (_, (first, second), failures):
        reservation = await second.reserve("in-flight")
        now += 9.0
        assert (await first.request("heartbeat", frontend=0))["ready"]
        now += 2.0
        # Frontend 0 still responds, but frontend 1 has missed its lease.
        # The first observer is the late heartbeat itself; refreshing before
        # checking expiry would incorrectly keep the group ready.
        assert not (await second.request("heartbeat", frontend=1))["ready"]
        assert not (await first.request("status"))["ready"]
        assert len(failures) == 1
        with pytest.raises(CoordinationError):
            await first.reserve("new-work")
        await reservation.release()  # Existing ownership can still be cleaned up.
        assert (await first.request("status"))["count"] == 0
        assert len(failures) == 1


@pytest.mark.asyncio
async def test_frontend_withdraw_stops_group_admission():
    async with group() as (coordinator, (first, second), failures):
        reservation = await second.reserve("in-flight")
        assert await first.request("withdraw", frontend=0) == {}
        status = await second.request("status")
        assert not status["ready"]
        assert status["stopping"]
        assert not coordinator.ready
        assert failures == ["OpenEngine frontend 0 is stopping"]
        with pytest.raises(CoordinationError):
            await second.reserve("new-work")
        await reservation.release()
