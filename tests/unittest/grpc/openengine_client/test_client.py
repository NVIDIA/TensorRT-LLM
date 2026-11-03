# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connection, Control, and Abort behavior of the OpenEngine client against a scripted engine."""

import asyncio
import itertools
import socket
from typing import Any

import grpc
import pytest
from openengine_client_helpers import WAIT_SECONDS, Hang, StubEngine, script, token

from tensorrt_llm.grpc.openengine.bindings import generation_pb2, kv_pb2, lifecycle_pb2, server_pb2
from tensorrt_llm.grpc.openengine_client import (
    AbortOutcome,
    OpenEngineClient,
    OpenEngineClientError,
    OpenEngineIncompatibleServerError,
    OpenEngineProtocolError,
    OpenEngineRpcError,
)

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]


def _with_engine(engine: StubEngine, body, **connect_kwargs: Any) -> Any:
    async def main() -> Any:
        async with engine:
            connect_kwargs.setdefault("ready_timeout", WAIT_SECONDS)
            async with await OpenEngineClient.connect(engine.target, **connect_kwargs) as client:
                return await asyncio.wait_for(body(engine, client), WAIT_SECONDS * 2)

    return asyncio.run(main())


def _connect_error(engine: StubEngine, **connect_kwargs: Any) -> BaseException:
    async def main() -> BaseException:
        async with engine:
            with pytest.raises(OpenEngineClientError) as raised:
                await OpenEngineClient.connect(engine.target, **connect_kwargs)
            return raised.value

    return asyncio.run(main())


async def _status(code: grpc.StatusCode, context: grpc.aio.ServicerContext) -> None:
    await context.abort(code, "scripted")


def test_connect_waits_for_ready_then_reads_info() -> None:
    engine = StubEngine(
        health_states=(
            lifecycle_pb2.HEALTH_STATE_STARTING,
            lifecycle_pb2.HEALTH_STATE_STARTING,
            lifecycle_pb2.HEALTH_STATE_READY,
        )
    )

    async def body(engine, client):
        return client.server_info, client.model_info, [rpc for rpc, _ in engine.control_calls]

    server_info, model_info, calls = _with_engine(engine, body)
    assert server_info.engine_name == "stub"
    assert model_info.served_model_name == "stub-model"
    assert calls == [
        "GetServerInfo",
        "Health",
        "Health",
        "Health",
        "GetServerInfo",
        "GetModelInfo",
    ]


def test_incompatible_engine_is_rejected_before_it_is_ready() -> None:
    engine = StubEngine(
        health_states=(lifecycle_pb2.HEALTH_STATE_STARTING,),
        server_info={"minimum_client_revision": 2},
    )

    async def main() -> tuple[BaseException, list[str]]:
        async with engine:
            with pytest.raises(OpenEngineIncompatibleServerError) as raised:
                await OpenEngineClient.connect(engine.target, ready_timeout=WAIT_SECONDS)
            return raised.value, [rpc for rpc, _ in engine.control_calls]

    error, calls = asyncio.run(main())
    assert "requires client revision 2" in str(error)
    assert calls == ["GetServerInfo"]


def test_server_info_unavailable_before_ready_is_read_after_ready() -> None:
    answers = itertools.chain([grpc.StatusCode.UNAVAILABLE], itertools.repeat(None))

    async def server_info(request, context):
        if next(answers) is not None:
            await context.abort(grpc.StatusCode.UNAVAILABLE, "loading")
        return server_pb2.ServerInfo(schema_revision=1, minimum_client_revision=1)

    engine = StubEngine(
        health_states=(lifecycle_pb2.HEALTH_STATE_STARTING, lifecycle_pb2.HEALTH_STATE_READY),
        control={"GetServerInfo": server_info},
    )

    async def body(engine, client):
        return [rpc for rpc, _ in engine.control_calls]

    assert _with_engine(engine, body) == [
        "GetServerInfo",
        "Health",
        "GetServerInfo",
        "Health",
        "GetServerInfo",
        "GetModelInfo",
    ]


def test_server_info_is_retried_while_loading_and_checked_once_it_answers() -> None:
    answers = iter([grpc.StatusCode.UNAVAILABLE, None])

    async def server_info(request, context):
        if next(answers) is not None:
            await context.abort(grpc.StatusCode.UNAVAILABLE, "loading")
        return server_pb2.ServerInfo(schema_revision=1, minimum_client_revision=2)

    engine = StubEngine(
        health_states=(lifecycle_pb2.HEALTH_STATE_STARTING,),
        control={"GetServerInfo": server_info},
    )

    async def main() -> tuple[BaseException, list[str]]:
        async with engine:
            with pytest.raises(OpenEngineIncompatibleServerError) as raised:
                await OpenEngineClient.connect(engine.target, ready_timeout=WAIT_SECONDS)
            return raised.value, [rpc for rpc, _ in engine.control_calls]

    error, calls = asyncio.run(main())
    assert "requires client revision 2" in str(error)
    assert calls == ["GetServerInfo", "Health", "GetServerInfo"]


def test_connect_gives_up_when_the_engine_never_becomes_ready() -> None:
    engine = StubEngine(health_states=(lifecycle_pb2.HEALTH_STATE_STARTING,))
    error = _connect_error(engine, ready_timeout=0.5)
    assert isinstance(error, OpenEngineRpcError)
    assert error.code == grpc.StatusCode.UNAVAILABLE
    assert "HEALTH_STATE_STARTING" in error.details


def test_connect_to_a_port_nobody_listens_on_fails_within_the_ready_timeout() -> None:
    async def main(port: int) -> tuple[BaseException, float]:
        loop = asyncio.get_running_loop()
        start = loop.time()
        with pytest.raises(OpenEngineRpcError) as raised:
            await OpenEngineClient.connect(f"127.0.0.1:{port}", ready_timeout=0.5)
        return raised.value, loop.time() - start

    # Holding the bound, non-listening socket keeps the port refusing connections.
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        port = holder.getsockname()[1]
        error, elapsed = asyncio.run(main(port))
    assert f"127.0.0.1:{port}" in str(error)
    assert elapsed < WAIT_SECONDS


@pytest.mark.parametrize(
    ("server_info", "allow_newer", "problem"),
    [
        pytest.param({"minimum_client_revision": 2}, True, "requires client revision 2", id="min"),
        pytest.param({"schema_revision": 2}, False, "newer than the supported", id="newer"),
        pytest.param({"schema_revision": 0}, True, "schema_revision 0", id="zero"),
    ],
)
def test_incompatible_schema_is_rejected(
    server_info: dict, allow_newer: bool, problem: str
) -> None:
    engine = StubEngine(server_info=server_info)
    error = _connect_error(engine, ready_timeout=WAIT_SECONDS, allow_newer_schema=allow_newer)
    assert isinstance(error, OpenEngineIncompatibleServerError)
    assert problem in str(error)


def test_newer_schema_is_accepted_when_allowed() -> None:
    engine = StubEngine(server_info={"schema_revision": 2})

    async def body(engine, client):
        return client.server_info.schema_revision

    assert _with_engine(engine, body, allow_newer_schema=True) == 2


def test_refresh_rereads_server_info() -> None:
    calls = itertools.count(1)
    engine = StubEngine()

    async def server_info(request, context):
        return server_pb2.ServerInfo(
            instance_id=f"call-{next(calls)}", schema_revision=1, minimum_client_revision=1
        )

    engine.control["GetServerInfo"] = server_info

    async def body(engine, client):
        before = client.server_info.instance_id
        await client.refresh()
        return before, client.server_info.instance_id

    # connect() reads ServerInfo twice (before and after the engine is ready).
    assert _with_engine(engine, body) == ("call-2", "call-3")


def test_health_forwards_the_inference_probe_flag() -> None:
    engine = StubEngine()

    async def body(engine, client):
        response = await client.health(include_inference_probe=True)
        return response.state, engine.control_calls[-1][1].include_inference_probe

    assert _with_engine(engine, body) == (lifecycle_pb2.HEALTH_STATE_READY, True)


def test_get_load_returns_none_when_unimplemented() -> None:
    engine = StubEngine()

    async def body(engine, client):
        return await client.get_load()

    assert _with_engine(engine, body) is None


def test_get_load_raises_other_failures() -> None:
    engine = StubEngine(
        control={"GetLoad": lambda request, context: _status(grpc.StatusCode.UNAVAILABLE, context)}
    )

    async def body(engine, client):
        with pytest.raises(OpenEngineRpcError) as raised:
            await client.get_load()
        return raised.value

    error = _with_engine(engine, body)
    assert error.code == grpc.StatusCode.UNAVAILABLE
    assert error.rpc == "GetLoad"
    assert error.details == "scripted"


def test_control_rpcs_use_the_control_timeout() -> None:
    async def hang(request, context):
        await asyncio.Event().wait()

    engine = StubEngine(control={"GetLoad": hang})

    async def body(engine, client):
        with pytest.raises(OpenEngineRpcError) as raised:
            await client.get_load()
        return raised.value.code

    assert _with_engine(engine, body, control_timeout=1.0) == grpc.StatusCode.DEADLINE_EXCEEDED


@pytest.mark.parametrize("call", ["health", "get_load", "abort", "abort_all", "refresh"])
def test_a_per_call_timeout_replaces_the_control_timeout(call: str) -> None:
    hang_after_connect = asyncio.Event()

    async def hang(request, context):
        await asyncio.Event().wait()

    def hanging(handler):
        async def maybe_hang(request, context):
            if hang_after_connect.is_set():
                await hang(request, context)
            return await handler(request, context)

        return maybe_hang

    engine = StubEngine(control={"GetLoad": hang})
    for rpc in ("GetServerInfo", "Health", "Abort"):
        engine.control[rpc] = hanging(engine.control[rpc])

    async def body(engine, client):
        hang_after_connect.set()
        method = getattr(client, call)
        args = ("req-1",) if call == "abort" else ()
        loop = asyncio.get_running_loop()
        start = loop.time()
        with pytest.raises(OpenEngineRpcError) as raised:
            await method(*args, timeout=0.3)
        return raised.value.code, loop.time() - start

    code, elapsed = _with_engine(engine, body, control_timeout=WAIT_SECONDS)
    assert code == grpc.StatusCode.DEADLINE_EXCEEDED
    assert elapsed < WAIT_SECONDS / 2


def test_closing_the_client_fails_control_calls_in_flight() -> None:
    started = asyncio.Event()

    async def hang(request, context):
        started.set()
        await asyncio.Event().wait()

    engine = StubEngine(control={"GetLoad": hang})

    async def main() -> BaseException:
        async with engine:
            client = await OpenEngineClient.connect(engine.target, ready_timeout=WAIT_SECONDS)
            call = asyncio.create_task(client.get_load())
            await asyncio.wait_for(started.wait(), WAIT_SECONDS)
            await client.close()
            with pytest.raises(OpenEngineClientError) as raised:
                await asyncio.wait_for(call, WAIT_SECONDS)
            return raised.value

    error = asyncio.run(main())
    assert type(error) is OpenEngineClientError
    assert "closed during the GetLoad call" in str(error)


def test_cancelling_the_caller_of_a_control_call_propagates() -> None:
    started = asyncio.Event()

    async def hang(request, context):
        started.set()
        await asyncio.Event().wait()

    engine = StubEngine(control={"GetLoad": hang})

    async def body(engine, client):
        call = asyncio.create_task(client.get_load())
        await asyncio.wait_for(started.wait(), WAIT_SECONDS)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        return call.cancelled()

    assert _with_engine(engine, body) is True


@pytest.mark.parametrize(
    ("status", "outcome"),
    [
        (lifecycle_pb2.ABORT_STATUS_ABORTED, AbortOutcome.ABORTED),
        (lifecycle_pb2.ABORT_STATUS_ALREADY_FINISHED, AbortOutcome.ALREADY_FINISHED),
    ],
)
def test_abort_by_request_id_maps_the_status(status: int, outcome: AbortOutcome) -> None:
    async def abort(request, context):
        return lifecycle_pb2.AbortResponse(status=status)

    engine = StubEngine(control={"Abort": abort})

    async def body(engine, client):
        result = await client.abort("req-1")
        return result, engine.control_calls[-1][1].request_id

    assert _with_engine(engine, body) == (outcome, "req-1")


def test_abort_of_an_unknown_request_counts_as_success() -> None:
    engine = StubEngine(
        control={"Abort": lambda request, context: _status(grpc.StatusCode.NOT_FOUND, context)}
    )

    async def body(engine, client):
        return await client.abort("unknown")

    assert _with_engine(engine, body) == AbortOutcome.NOT_FOUND


def test_abort_without_a_status_is_a_protocol_error() -> None:
    async def abort(request, context):
        return lifecycle_pb2.AbortResponse()

    engine = StubEngine(control={"Abort": abort})

    async def body(engine, client):
        with pytest.raises(OpenEngineProtocolError):
            await client.abort("req-1")

    _with_engine(engine, body)


def test_abort_rejects_an_empty_request_id() -> None:
    engine = StubEngine()

    async def body(engine, client):
        with pytest.raises(ValueError):
            await client.abort("")
        return [rpc for rpc, _ in engine.control_calls].count("Abort")

    assert _with_engine(engine, body) == 0


@pytest.mark.parametrize("call", ["abort_all", "abort_kv_session"])
def test_abort_all_and_kv_session_reject_an_unknown_status(call: str) -> None:
    async def abort(request, context):
        return lifecycle_pb2.AbortResponse()

    engine = StubEngine(control={"Abort": abort})

    async def body(engine, client):
        with pytest.raises(OpenEngineProtocolError):
            if call == "abort_all":
                await client.abort_all()
            else:
                await client.abort_kv_session(kv_pb2.KvSessionRef(session_id="s-1"))

    _with_engine(engine, body)


def test_abort_all_targets_every_request() -> None:
    engine = StubEngine()

    async def body(engine, client):
        await client.abort_all()
        return engine.control_calls[-1][1].WhichOneof("target")

    assert _with_engine(engine, body) == "all_requests"


def test_close_cancels_open_streams_and_rejects_new_calls() -> None:
    engine = StubEngine(script(token(0, 1), Hang()))

    async def main() -> Any:
        async with engine:
            client = await OpenEngineClient.connect(engine.target, ready_timeout=WAIT_SECONDS)
            stream = client.generate(
                generation_pb2.GenerateRequest(
                    request_id="req-1", model="m", token_ids=generation_pb2.TokenIds(ids=[1])
                )
            )
            first = await stream.__anext__()
            await client.close()
            await client.close()
            rest = [response async for response in stream]
            await asyncio.wait_for(engine.cancel_seen.wait(), WAIT_SECONDS)
            with pytest.raises(OpenEngineClientError, match="closed"):
                await client.health()
            with pytest.raises(OpenEngineClientError, match="closed"):
                client.generate(generation_pb2.GenerateRequest())
            return first.WhichOneof("event"), rest, engine.cancelled

    assert asyncio.run(main()) == ("token", [], ["req-1"])


def test_oversized_request_fails_locally() -> None:
    engine = StubEngine()
    limit = 1024

    async def body(engine, client):
        request = generation_pb2.GenerateRequest(
            request_id="big", model="m", token_ids=generation_pb2.TokenIds(ids=[2**31] * limit)
        )
        with pytest.raises(OpenEngineRpcError) as raised:
            async with client.generate(request) as stream:
                async for _ in stream:
                    pass
        return raised.value.code, engine.requests

    result = _with_engine(engine, body, channel_options=[("grpc.max_send_message_length", limit)])
    assert result == (grpc.StatusCode.RESOURCE_EXHAUSTED, [])
