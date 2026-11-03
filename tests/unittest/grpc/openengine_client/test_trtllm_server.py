# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OpenEngine client against TensorRT-LLM's own OpenEngine server (with a fake LLM)."""

import asyncio
from typing import Any

import grpc
import pytest
from openengine_client_helpers import WAIT_SECONDS, FakeLlm, trtllm_engine, wait_until

from tensorrt_llm.grpc.openengine.bindings import generation_pb2, kv_pb2, lifecycle_pb2
from tensorrt_llm.grpc.openengine_client import AbortOutcome, OpenEngineClient, OpenEngineRpcError

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]


def _request(request_id: str = "req-1") -> generation_pb2.GenerateRequest:
    return generation_pb2.GenerateRequest(
        request_id=request_id,
        model="test-model",
        token_ids=generation_pb2.TokenIds(ids=[1, 2, 3]),
    )


def _run(llm: FakeLlm, body) -> Any:
    async def main() -> Any:
        async with trtllm_engine(llm) as server:
            async with await OpenEngineClient.connect(
                f"127.0.0.1:{server.port}", ready_timeout=WAIT_SECONDS
            ) as client:
                return await asyncio.wait_for(body(llm, client), WAIT_SECONDS * 2)

    return asyncio.run(main())


async def _running_requests(client: OpenEngineClient) -> int:
    load = await client.get_load()
    assert load is not None
    return load.running_requests


async def _start(client: OpenEngineClient, llm: FakeLlm, request_id: str = "req-1"):
    """Open a stream and read its first event, leaving the request running on the engine."""
    stream = client.generate(_request(request_id))
    first = await stream.__anext__()
    assert first.WhichOneof("event") == "token"
    return stream, llm.handles[-1]


def test_connect_reads_the_engine_identity_and_model() -> None:
    async def body(llm, client):
        health = await client.health()
        return client.server_info, client.model_info, health.state

    server_info, model_info, state = _run(FakeLlm(), body)
    assert server_info.engine_name == "tensorrt_llm"
    assert server_info.schema_revision == 1
    assert server_info.instance_id == "fake-instance"
    assert model_info.served_model_name == "test-model"
    assert model_info.supports_token_ids_input
    assert state == lifecycle_pb2.HEALTH_STATE_READY


def test_generate_streams_tokens_then_finish_with_usage() -> None:
    async def body(llm, client):
        async with client.generate(_request()) as stream:
            return [response async for response in stream]

    responses = _run(FakeLlm(steps=[[11], [12, 13]]), body)
    events = [response.WhichOneof("event") for response in responses]
    assert events == ["token", "token", "finished"]
    token_ids = [info.token_id for r in responses[:-1] for info in r.token.tokens]
    assert token_ids == [11, 12, 13]
    assert responses[-1].finished.reason == generation_pb2.FINISH_REASON_LENGTH
    assert responses[-1].usage.prompt_tokens == 3
    assert responses[-1].usage.completion_tokens == 3
    assert not any(response.HasField("usage") for response in responses[:-1])


def test_generated_request_id_is_accepted() -> None:
    async def body(llm, client):
        request = _request(request_id="")
        async with client.generate(request) as stream:
            responses = [response async for response in stream]
        return stream.request_id, responses[-1].request_id

    stream_id, echoed_id = _run(FakeLlm(), body)
    assert stream_id.startswith("trtllm-")
    assert echoed_id == stream_id


def test_duplicate_active_request_id_is_rejected_before_acceptance() -> None:
    async def body(llm, client):
        first, _ = await _start(client, llm, "dup")
        try:
            with pytest.raises(OpenEngineRpcError) as raised:
                async with client.generate(_request("dup")) as second:
                    async for _ in second:
                        pass
        finally:
            await first.aclose()
        return raised.value.code

    assert _run(FakeLlm(hang_after=1), body) == grpc.StatusCode.ALREADY_EXISTS


def test_abort_running_request_finishes_it_as_cancelled() -> None:
    async def body(llm, client):
        stream, handle = await _start(client, llm)
        outcome = await client.abort("req-1")
        async with stream:
            rest = [response async for response in stream]
        return outcome, rest, handle.aborted

    outcome, rest, aborted = _run(FakeLlm(hang_after=1), body)
    assert outcome == AbortOutcome.ABORTED
    assert aborted
    assert rest[-1].finished.reason == generation_pb2.FINISH_REASON_CANCELLED
    assert rest[-1].HasField("usage")


def test_abort_after_finish_and_of_unknown_request_succeed() -> None:
    async def body(llm, client):
        async with client.generate(_request()) as stream:
            async for _ in stream:
                pass
        return await client.abort("req-1"), await client.abort("never-sent")

    assert _run(FakeLlm(), body) == (AbortOutcome.ALREADY_FINISHED, AbortOutcome.ALREADY_FINISHED)


def test_abort_all_ends_every_running_request() -> None:
    async def body(llm, client):
        first, first_handle = await _start(client, llm, "a")
        second, second_handle = await _start(client, llm, "b")
        response = await client.abort_all()
        reasons = []
        for stream in (first, second):
            async with stream:
                reasons.append([r async for r in stream][-1].finished.reason)
        return response.status, reasons, first_handle.aborted and second_handle.aborted

    status, reasons, aborted = _run(FakeLlm(hang_after=1), body)
    assert status == lifecycle_pb2.ABORT_STATUS_ABORTED
    assert reasons == [generation_pb2.FINISH_REASON_CANCELLED] * 2
    assert aborted


def test_abort_by_kv_session_is_unimplemented() -> None:
    async def body(llm, client):
        with pytest.raises(OpenEngineRpcError) as raised:
            await client.abort_kv_session(kv_pb2.KvSessionRef(session_id="s"))
        return raised.value.code

    assert _run(FakeLlm(), body) == grpc.StatusCode.UNIMPLEMENTED


def test_leaving_the_stream_early_aborts_the_engine_request() -> None:
    async def body(llm, client):
        async with client.generate(_request()) as stream:
            async for _ in stream:
                break
        handle = llm.handles[-1]
        await wait_until(lambda: handle.aborted)
        await wait_until_idle(client)
        return handle.aborted

    assert _run(FakeLlm(hang_after=1), body)


def test_cancelling_the_reading_task_aborts_the_engine_request() -> None:
    async def body(llm, client):
        started = asyncio.Event()

        async def consume() -> None:
            async with client.generate(_request()) as stream:
                async for _ in stream:
                    started.set()

        task = asyncio.create_task(consume())
        await asyncio.wait_for(started.wait(), WAIT_SECONDS)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        handle = llm.handles[-1]
        await wait_until(lambda: handle.aborted)
        await wait_until_idle(client)
        return handle.aborted

    assert _run(FakeLlm(hang_after=1), body)


def test_cancel_from_another_task_aborts_the_engine_request_quietly() -> None:
    async def body(llm, client):
        stream, handle = await _start(client, llm)
        asyncio.get_running_loop().call_soon(stream.cancel)
        rest = [response async for response in stream]
        await wait_until(lambda: handle.aborted)
        await wait_until_idle(client)
        return rest

    assert _run(FakeLlm(hang_after=1), body) == []


def test_aclose_after_break_aborts_the_engine_request() -> None:
    async def body(llm, client):
        stream = client.generate(_request())
        async for _ in stream:
            break
        await stream.aclose()
        handle = llm.handles[-1]
        await wait_until(lambda: handle.aborted)
        await wait_until_idle(client)
        return stream.done

    assert _run(FakeLlm(hang_after=1), body)


def test_closing_the_client_aborts_running_requests() -> None:
    async def main() -> bool:
        llm = FakeLlm(hang_after=1)
        async with trtllm_engine(llm) as server:
            client = await OpenEngineClient.connect(
                f"127.0.0.1:{server.port}", ready_timeout=WAIT_SECONDS
            )
            _, handle = await _start(client, llm)
            await client.close()
            await wait_until(lambda: handle.aborted)
            return handle.aborted

    assert asyncio.run(main())


async def wait_until_idle(client: OpenEngineClient) -> None:
    """Wait until the engine reports no running requests."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + WAIT_SECONDS
    while await _running_requests(client):
        assert loop.time() < deadline, "the engine still reports running requests"
        await asyncio.sleep(0.01)
