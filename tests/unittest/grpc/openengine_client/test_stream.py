# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate stream behavior of the OpenEngine client against a scripted engine."""

import asyncio
import re
import types
from typing import Any

import grpc
import pytest
from openengine_client_helpers import (
    WAIT_SECONDS,
    Hang,
    Status,
    StubEngine,
    engine_error,
    finished,
    prefill_ready,
    prompt,
    script,
    token,
    wait_until,
)

import tensorrt_llm.grpc.openengine_client.stream as stream_module
from tensorrt_llm.grpc.openengine.bindings import error_pb2, generation_pb2
from tensorrt_llm.grpc.openengine_client import (
    OpenEngineClient,
    OpenEngineEngineError,
    OpenEngineProtocolError,
    OpenEngineRpcError,
)

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]


def _request(request_id: str = "req-1", num_sequences: int | None = None):
    request = generation_pb2.GenerateRequest(
        request_id=request_id,
        model="stub-model",
        token_ids=generation_pb2.TokenIds(ids=[1, 2, 3]),
    )
    if num_sequences is not None:
        request.sampling.num_sequences = num_sequences
    return request


def _run(engine: StubEngine, body) -> Any:
    async def main() -> Any:
        async with engine:
            async with await OpenEngineClient.connect(
                engine.target, ready_timeout=WAIT_SECONDS
            ) as client:
                return await asyncio.wait_for(body(engine, client), WAIT_SECONDS * 2)

    return asyncio.run(main())


async def _collect(stream) -> list[generation_pb2.GenerateResponse]:
    async with stream:
        return [response async for response in stream]


def test_interleaved_outputs_end_after_every_output_finishes() -> None:
    engine = StubEngine(
        script(
            token(0, 1),
            token(1, 2),
            finished(1, with_usage=False),
            token(0, 3),
            finished(0),
        )
    )

    async def body(engine, client):
        return await _collect(client.generate(_request(num_sequences=2)))

    responses = _run(engine, body)
    assert [response.WhichOneof("event") for response in responses] == [
        "token",
        "token",
        "finished",
        "token",
        "finished",
    ]
    assert responses[-1].usage.completion_tokens == 2


def test_multi_token_deltas_are_passed_through() -> None:
    engine = StubEngine(script(token(0, 5, 6, 7), finished(0)))

    async def body(engine, client):
        return await _collect(client.generate(_request()))

    responses = _run(engine, body)
    assert [info.token_id for info in responses[0].token.tokens] == [5, 6, 7]


def test_missing_request_id_is_generated_without_touching_the_caller_request() -> None:
    engine = StubEngine()

    async def body(engine, client):
        request = _request(request_id="")
        stream = client.generate(request)
        await _collect(stream)
        return request, stream.request_id, engine.requests[0].request_id

    request, stream_id, sent_id = _run(engine, body)
    assert request.request_id == ""
    assert re.fullmatch(r"trtllm-[0-9a-f]{32}", stream_id)
    assert sent_id == stream_id


def test_caller_request_id_is_kept() -> None:
    engine = StubEngine()

    async def body(engine, client):
        await _collect(client.generate(_request(request_id="mine")))
        return engine.requests[0].request_id

    assert _run(engine, body) == "mine"


def test_prefill_ready_ends_the_stream() -> None:
    engine = StubEngine(script(prefill_ready()))

    async def body(engine, client):
        return await _collect(client.generate(_request()))

    responses = _run(engine, body)
    assert [response.WhichOneof("event") for response in responses] == ["prefill_ready"]


def test_status_before_acceptance_is_raised_as_rpc_error() -> None:
    engine = StubEngine(script(Status(grpc.StatusCode.UNIMPLEMENTED, "lora is not supported")))

    async def body(engine, client):
        with pytest.raises(OpenEngineRpcError) as raised:
            await _collect(client.generate(_request()))
        return raised.value

    error = _run(engine, body)
    assert error.code == grpc.StatusCode.UNIMPLEMENTED
    assert error.details == "lora is not supported"
    assert error.rpc == "Generate"
    assert error.request_id == "req-1"


def test_engine_error_event_is_raised_with_its_fields() -> None:
    engine = StubEngine(script(token(0, 1), engine_error()))

    async def body(engine, client):
        received = []
        with pytest.raises(OpenEngineEngineError) as raised:
            async with client.generate(_request()) as stream:
                async for response in stream:
                    received.append(response)
        return received, raised.value

    received, error = _run(engine, body)
    assert [response.WhichOneof("event") for response in received] == ["token"]
    assert error.code == error_pb2.ERROR_CODE_OVERLOADED
    assert error.retryable
    assert error.usage.completion_tokens == 2
    assert error.request_id == "req-1"


def test_generate_timeout_is_raised_as_deadline_exceeded() -> None:
    engine = StubEngine(script(token(0, 1), Hang()))

    async def body(engine, client):
        with pytest.raises(OpenEngineRpcError) as raised:
            await _collect(client.generate(_request(), timeout=0.2))
        return raised.value.code

    assert _run(engine, body) == grpc.StatusCode.DEADLINE_EXCEEDED


def test_concurrent_streams_share_one_client_without_crosstalk() -> None:
    async def echo(request, context):
        marker = int(request.request_id.split("-")[1])
        yield token(0, marker)
        yield finished(0)

    engine = StubEngine(echo)

    async def body(engine, client):
        async def one(number: int) -> int:
            responses = await _collect(client.generate(_request(request_id=f"r-{number}")))
            return responses[0].token.tokens[0].token_id

        return await asyncio.gather(*(one(number) for number in range(64)))

    assert _run(engine, body) == list(range(64))


@pytest.mark.parametrize(
    ("steps", "num_sequences", "problem"),
    [
        pytest.param((token(0, 1),), None, "ended before a terminal event", id="eof"),
        pytest.param((token(None, 1), finished(0)), None, "no output_index", id="no-index"),
        pytest.param((token(1, 1), finished(0)), None, "output_index 1", id="index-range"),
        pytest.param(
            (finished(0, with_usage=False), token(0, 1), finished(1)),
            2,
            "follows output 0's finish",
            id="token-after-finish",
        ),
        pytest.param(
            (finished(0, with_usage=False), finished(0)),
            2,
            "follows output 0's finish",
            id="duplicate-finish",
        ),
        pytest.param((token(0, 1), prompt(), finished(0)), None, "prompt event", id="late-prompt"),
        pytest.param((prompt(), prompt(), finished(0)), None, "prompt event", id="two-prompts"),
        pytest.param(
            (finished(0, with_usage=False), prompt(), finished(1)),
            2,
            "prompt event",
            id="prompt-after-finish",
        ),
        pytest.param(
            (finished(0, generation_pb2.FINISH_REASON_UNSPECIFIED),),
            None,
            "without a known reason",
            id="no-reason",
        ),
        pytest.param((finished(0, 99),), None, "without a known reason", id="unknown-reason"),
        pytest.param(
            (token(0, 1, with_usage=True), finished(0)), None, "carries usage", id="early-usage"
        ),
        pytest.param(
            (generation_pb2.GenerateResponse(request_id="other", token=token(0, 1).token),),
            None,
            "request_id 'other'",
            id="foreign-request-id",
        ),
    ],
)
def test_stream_contract_violations_are_raised(
    steps: tuple, num_sequences: int | None, problem: str
) -> None:
    engine = StubEngine(
        script(*steps, Hang()) if steps[-1].HasField("finished") else script(*steps)
    )

    async def body(engine, client):
        with pytest.raises(OpenEngineProtocolError) as raised:
            await _collect(client.generate(_request(num_sequences=num_sequences)))
        return raised.value

    error = _run(engine, body)
    assert problem in str(error)
    assert error.request_id == "req-1"


def test_violation_cancels_the_call() -> None:
    engine = StubEngine(script(token(None, 1), Hang()))

    async def body(engine, client):
        with pytest.raises(OpenEngineProtocolError):
            await _collect(client.generate(_request()))
        await asyncio.wait_for(engine.cancel_seen.wait(), WAIT_SECONDS)
        return engine.cancelled

    assert _run(engine, body) == ["req-1"]


def test_events_after_the_terminal_event_are_logged_not_raised(monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(stream_module.logger, "warning", warnings.append)
    engine = StubEngine(script(engine_error(), token(0, 9)))

    async def body(engine, client):
        with pytest.raises(OpenEngineEngineError):
            await _collect(client.generate(_request()))

    _run(engine, body)
    assert len(warnings) == 1
    assert "token event after request req-1 ended" in warnings[0]


def test_status_that_arrived_before_the_next_iteration_is_logged(monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(stream_module.logger, "warning", warnings.append)
    engine = StubEngine(script(finished(0), Status(grpc.StatusCode.INTERNAL, "late failure")))

    async def body(engine, client):
        async with client.generate(_request()) as stream:
            events = [(await stream.__anext__()).WhichOneof("event")]
            await wait_until(lambda: stream._call.done())
            events += [response.WhichOneof("event") async for response in stream]
        return events

    assert _run(engine, body) == ["finished"]
    assert len(warnings) == 1
    assert "INTERNAL after its terminal event" in warnings[0]


def test_prefill_ready_never_cancels_the_call(monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(stream_module.logger, "warning", warnings.append)
    monkeypatch.setattr(stream_module, "DRAIN_TIMEOUT_SECONDS", 0.1)
    release = asyncio.Event()

    async def generate(request, context):
        yield prefill_ready()
        await release.wait()
        yield token(0, 9)

    engine = StubEngine(generate)

    async def body(engine, client):
        async with client.generate(_request()) as stream:
            events = [response.WhichOneof("event") async for response in stream]
        done_at_once = stream.done
        await asyncio.sleep(0.3)
        cancelled = list(engine.cancelled)
        release.set()
        await wait_until(lambda: len(warnings) == 1)
        return events, done_at_once, cancelled

    assert _run(engine, body) == (["prefill_ready"], True, [])
    assert "token event after request req-1 ended" in warnings[0]


def test_server_that_never_closes_after_the_terminal_event_is_cancelled(monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(stream_module.logger, "warning", warnings.append)
    monkeypatch.setattr(stream_module, "DRAIN_TIMEOUT_SECONDS", 0.2)
    engine = StubEngine(script(finished(0), Hang()))

    async def body(engine, client):
        responses = await _collect(client.generate(_request()))
        await asyncio.wait_for(engine.cancel_seen.wait(), WAIT_SECONDS)
        return responses

    assert len(_run(engine, body)) == 1
    assert "did not close request req-1" in warnings[0]


def test_leaving_the_context_cancels_the_call() -> None:
    engine = StubEngine(script(token(0, 1), Hang()))

    async def body(engine, client):
        async with client.generate(_request()) as stream:
            async for _ in stream:
                break
        await asyncio.wait_for(engine.cancel_seen.wait(), WAIT_SECONDS)
        return stream.done, engine.cancelled

    assert _run(engine, body) == (True, ["req-1"])


def test_cancelling_the_reading_task_cancels_the_call() -> None:
    engine = StubEngine(script(token(0, 1), Hang()))

    async def body(engine, client):
        first = asyncio.Event()

        async def consume() -> None:
            async with client.generate(_request()) as stream:
                async for _ in stream:
                    first.set()

        task = asyncio.create_task(consume())
        await asyncio.wait_for(first.wait(), WAIT_SECONDS)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(engine.cancel_seen.wait(), WAIT_SECONDS)
        return engine.cancelled

    assert _run(engine, body) == ["req-1"]


def test_cancel_from_another_task_ends_iteration_quietly() -> None:
    engine = StubEngine(script(token(0, 1), Hang()))

    async def body(engine, client):
        async with client.generate(_request()) as stream:
            received = []
            async for response in stream:
                received.append(response)
                asyncio.get_running_loop().call_soon(stream.cancel)
        await asyncio.wait_for(engine.cancel_seen.wait(), WAIT_SECONDS)
        return len(received), stream.done

    assert _run(engine, body) == (1, True)


def test_cancel_before_iteration_never_sends_the_request() -> None:
    engine = StubEngine()

    async def body(engine, client):
        stream = client.generate(_request())
        stream.cancel()
        stream.cancel()
        state = (stream.done, stream.cancelled, stream in client._streams)
        responses = [response async for response in stream]
        return state, responses, engine.requests

    assert _run(engine, body) == ((True, True, False), [], [])


def test_cancel_after_the_terminal_event_is_a_no_op() -> None:
    engine = StubEngine(script(finished(0)))

    async def body(engine, client):
        async with client.generate(_request()) as stream:
            async for _ in stream:
                stream.cancel()
        return engine.cancelled

    assert _run(engine, body) == []


def test_cancelled_stream_is_told_apart_from_a_completed_one() -> None:
    engine = StubEngine(script(token(0, 1), Hang()))

    async def body(engine, client):
        completed = client.generate(_request(request_id="done"))
        engine.generate = script(token(0, 1), finished(0))
        await _collect(completed)
        engine.generate = script(token(0, 1), Hang())
        cancelled = client.generate(_request(request_id="cut"))
        async with cancelled:
            async for _ in cancelled:
                cancelled.cancel()
        return completed.cancelled, cancelled.cancelled

    assert _run(engine, body) == (False, True)


def test_task_cancellation_wins_over_a_concurrent_stream_cancel() -> None:
    engine = StubEngine(script(token(0, 1), Hang()))

    async def body(engine, client):
        first = asyncio.Event()
        after_loop: list[str] = []
        stream = client.generate(_request())

        async def consume() -> None:
            async with stream:
                async for _ in stream:
                    first.set()
            after_loop.append("ran")

        task = asyncio.create_task(consume())
        await asyncio.wait_for(first.wait(), WAIT_SECONDS)
        await asyncio.sleep(0)
        stream.cancel()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return after_loop, stream.done, stream.cancelled

    assert _run(engine, body) == ([], True, True)


def test_task_cancelled_after_the_terminal_event_leaves_the_stream_completed() -> None:
    engine = StubEngine(script(finished(0), Hang()))

    async def body(engine, client):
        stream = client.generate(_request())
        events = [(await stream.__anext__()).WhichOneof("event")]
        # The next iteration waits for the server to close the call.
        draining = asyncio.create_task(stream.__anext__())
        await asyncio.sleep(0.1)
        assert not draining.done()
        draining.cancel()
        with pytest.raises(asyncio.CancelledError):
            await draining
        return events, stream.done, stream.cancelled

    assert _run(engine, body) == (["finished"], True, False)


def test_local_failure_to_start_the_call_ends_the_stream() -> None:
    engine = StubEngine()

    async def body(engine, client):
        stream = client.generate(_request())

        def broken(request, timeout=None):
            raise TypeError("broken stub")

        stream._stub = types.SimpleNamespace(Generate=broken)
        with pytest.raises(TypeError):
            await stream.__anext__()
        return stream.done, stream in client._streams, engine.requests

    assert _run(engine, body) == (True, False, [])


def test_closing_the_client_after_the_terminal_event_ends_iteration_quietly() -> None:
    engine = StubEngine(script(finished(0), Hang()))

    async def main() -> tuple[list[str], bool]:
        async with engine:
            client = await OpenEngineClient.connect(engine.target, ready_timeout=WAIT_SECONDS)
            stream = client.generate(_request())
            events = [(await stream.__anext__()).WhichOneof("event")]
            await client.close()
            async for response in stream:
                events.append(response.WhichOneof("event"))
            return events, stream.done

    assert asyncio.run(main()) == (["finished"], True)


def test_unknown_events_are_skipped(monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(stream_module.logger, "warning", warnings.append)
    # Field 15 is not part of GenerateResponse; it stands in for an event type
    # added by a newer schema revision.
    unknown = generation_pb2.GenerateResponse.FromString(b"\x7a\x00")
    engine = StubEngine(script(unknown, token(0, 1), finished(0)))

    async def body(engine, client):
        return await _collect(client.generate(_request()))

    responses = _run(engine, body)
    assert [response.WhichOneof("event") for response in responses] == ["token", "finished"]
    assert len(warnings) == 1
    assert "unknown event" in warnings[0]


def test_request_is_copied_when_the_stream_is_created() -> None:
    engine = StubEngine(script(token(0, 1), finished(0)))

    async def body(engine, client):
        request = _request()
        stream = client.generate(request)
        request.request_id = "changed"
        request.sampling.num_sequences = 2
        responses = await _collect(stream)
        sent = engine.requests[0]
        return len(responses), sent.request_id, sent.sampling.HasField("num_sequences")

    assert _run(engine, body) == (2, "req-1", False)
