# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise request ownership over real private gRPC sockets."""

import asyncio
from contextlib import asynccontextmanager, suppress
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import grpc
import pytest

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
async def group(coordinator_type=Coordinator, client_type=FrontendClient):
    # Short paths avoid the AF_UNIX pathname limit under pytest's tmp_path.
    with TemporaryDirectory(prefix="oe-test-") as directory:
        failures = []
        coordinator = coordinator_type(directory, 2, "shared-engine", lambda: True, failures.append)
        clients = [client_type(directory, i, failures.append) for i in range(2)]
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
        }
        assert await second.request("abort", request_id="request") == {"aborted": 1, "failed": 0}
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
            "release", request_id=old.request_id, token=old.token, frontend=first.frontend_id
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
    [("test-model", grpc.StatusCode.UNAVAILABLE), ("", grpc.StatusCode.INVALID_ARGUMENT)],
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
                        lifecycle_pb2.HealthRequest(include_inference_probe=True), timeout=5
                    )
                assert error.value.code() == grpc.StatusCode.UNAVAILABLE
            assert failures
        finally:
            await server.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_id", ["\x01" * (12 * 1024 * 1024), "é" * 513], ids=["json-expansion", "utf8-size"]
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
        async def _abort(self, message, context):
            await asyncio.sleep(0.2)
            return await super()._abort(message, context)

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
