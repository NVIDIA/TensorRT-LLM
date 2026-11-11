# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any, Callable, List, Optional, Tuple, Union

import anyio
import pydantic_core
from fastapi.responses import StreamingResponse
from openai.types.responses import (ResponseCompletedEvent,
                                    ResponseCreatedEvent, ResponseErrorEvent,
                                    ResponseFailedEvent,
                                    ResponseIncompleteEvent,
                                    ResponseInProgressEvent)

from tensorrt_llm.executor import EngineDeadError, RequestError
from tensorrt_llm.executor.utils import (CONTEXT_LENGTH_EXCEEDED_CODE,
                                         is_context_length_exceeded_message)
from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import (OpenAIBaseModel,
                                                ResponsesResponse,
                                                StreamingResponsesResponse)


def _stream_terminal_event(
    final_response: ResponsesResponse,
    sequence_number: int = -1,
) -> Union[ResponseCompletedEvent, ResponseIncompleteEvent,
           ResponseFailedEvent]:
    """The terminal event matching the response's status.

    "incomplete" ends in ``response.incomplete`` and "failed" in
    ``response.failed``; everything else, including the client-initiated
    "cancelled", ends in ``response.completed``.
    """
    payload = final_response.model_dump(by_alias=True)
    if final_response.status == "incomplete":
        return ResponseIncompleteEvent(
            type="response.incomplete",
            sequence_number=sequence_number,
            response=payload,
        )
    if final_response.status == "failed":
        return ResponseFailedEvent(
            type="response.failed",
            sequence_number=sequence_number,
            response=payload,
        )
    return ResponseCompletedEvent(
        type="response.completed",
        sequence_number=sequence_number,
        response=payload,
    )


def _response_failed_frame(snapshot: dict, code: str, message: str,
                           sequence_number: int) -> str:
    """A ``response.failed`` frame whose snapshot fails with ``code``.

    ``snapshot`` is a ``ResponsesResponse`` dump, which has no ``error``; the
    SDK's Response, which the event carries, does. The SDK's ``ResponseError``
    does not list ``context_length_exceeded``, the code OpenAI fails an
    over-length input with and the one clients act on. So the event is
    validated with ``server_error`` in ``error.code``; any other code is then
    written into its serialized form, with the serializer every frame uses.
    """
    event = ResponseFailedEvent(
        type="response.failed",
        sequence_number=sequence_number,
        response={
            **snapshot, "error": {
                "code": "server_error",
                "message": message
            }
        },
    )
    if code == "server_error":
        return _format_sse_event(event)
    payload = event.model_dump(mode="json", by_alias=True)
    payload["response"]["error"]["code"] = code
    return (f"event: {event.type}\n"
            f"data: {pydantic_core.to_json(payload).decode()}\n\n")


# Why a stream stopped before its terminal event, sent as the `error` code.
STREAM_TERMINATION_ENGINE_ERROR = "engine_error"

STREAM_TERMINATION_UPSTREAM_ERROR = "upstream_error"

STREAM_TERMINATION_INTERNAL_ERROR = "internal_error"

# The engine rejected an input longer than the context window. This answers
# the request rather than breaking it: the stream fails with OpenAI's code for
# it, which clients act on (the Codex CLI compacts its history), and then ends
# normally.
STREAM_TERMINATION_CONTEXT_LENGTH_EXCEEDED = CONTEXT_LENGTH_EXCEEDED_CODE

_SSE_EVENT_DELIMITER = "\n\n"

# A terminal event frame always follows another frame, and JSON payloads
# escape newlines, so these match only at a frame boundary.
_TERMINAL_FRAME_STARTS = tuple(f"{_SSE_EVENT_DELIMITER}event: response.{kind}"
                               for kind in ("completed", "incomplete",
                                            "failed"))

_TERMINAL_FRAME_STARTS_BYTES = tuple(marker.encode()
                                     for marker in _TERMINAL_FRAME_STARTS)

_SSE_TAIL_LENGTH = max(len(marker) for marker in _TERMINAL_FRAME_STARTS)


def _sse_delimiter_and_markers(sample: Any) -> Tuple[Any, tuple]:
    if isinstance(sample, bytes):
        return _SSE_EVENT_DELIMITER.encode(), _TERMINAL_FRAME_STARTS_BYTES
    return _SSE_EVENT_DELIMITER, _TERMINAL_FRAME_STARTS


def classify_stream_termination(exc: Exception) -> str:
    """Attribute a stream-ending exception to a cause."""
    # An over-length rejection crosses the executor IPC boundary as message
    # text, so it is recognized by that text.
    if isinstance(exc, RequestError) and is_context_length_exceeded_message(
            str(exc)):
        return STREAM_TERMINATION_CONTEXT_LENGTH_EXCEEDED
    if isinstance(exc, (RequestError, EngineDeadError)):
        return STREAM_TERMINATION_ENGINE_ERROR
    # By module, to keep the HTTP client library out of this layer.
    if type(exc).__module__.split(".")[0] in ("aiohttp", "httpx", "httpcore"):
        return STREAM_TERMINATION_UPSTREAM_ERROR
    return STREAM_TERMINATION_INTERNAL_ERROR


def describe_stream_termination(exc: Exception, cause: str) -> str:
    """The client-facing detail of a stream-ending exception.

    Only engine errors carry their message; transport and internal errors can
    name internal hosts. An over-length rejection is described by its message
    alone, the text a non-streamed request is rejected with.
    """
    if cause == STREAM_TERMINATION_CONTEXT_LENGTH_EXCEEDED:
        return str(exc)
    if cause == STREAM_TERMINATION_ENGINE_ERROR and str(exc):
        return f"{type(exc).__name__}: {exc}"
    return type(exc).__name__


def stream_error_event(cause: str, detail: str,
                       events_sent: int) -> List[bytes]:
    """A bare ``error`` event, for a relay that holds no response to snapshot."""
    return [
        _sse_event(
            ResponseErrorEvent(
                type="error",
                sequence_number=events_sent,
                code=cause,
                message=detail,
                param=None,
            ))
    ]


async def guard_responses_stream(
    stream: AsyncGenerator[Any, None],
    terminal_events: Callable[[str, str, int], List[Any]],
) -> AsyncGenerator[Any, None]:
    """End a stream that fails before its terminal event with events saying so.

    Frames pass through untouched, and the exception is re-raised after the
    terminal events, except for an over-length rejection: those events answer
    the request in full, so the stream ends normally. A client hangup
    (CancelledError or GeneratorExit) is not caught: there is nobody left to
    tell.
    """
    events_sent = 0
    tail = None
    in_terminal_frame = False
    completed = False
    try:
        async for chunk in stream:
            if not completed:
                if tail is None:
                    tail = chunk[:0]
                text = tail + chunk
                delimiter, markers = _sse_delimiter_and_markers(text)
                events_sent += text.count(delimiter) - tail.count(delimiter)
                if in_terminal_frame:
                    # A delimiter that ends the frame takes a byte of this chunk.
                    frame_rest = max(len(tail) - len(delimiter) + 1, 0)
                else:
                    starts = [text.find(marker) for marker in markers]
                    starts = [start for start in starts if start >= 0]
                    if starts:
                        in_terminal_frame = True
                        frame_rest = min(starts) + len(delimiter)
                # Only a whole terminal frame completes the stream; a client
                # cut off inside it has not received the response.
                if in_terminal_frame:
                    completed = text.find(delimiter, frame_rest) >= 0
                tail = text[-_SSE_TAIL_LENGTH:]
            yield chunk
    except Exception as exc:
        if completed:
            raise
        cause = classify_stream_termination(exc)
        logger.error("Responses stream terminated before completion "
                     f"({cause}): {type(exc).__name__}: {exc}")
        try:
            frames = terminal_events(cause,
                                     describe_stream_termination(exc, cause),
                                     events_sent)
        except Exception as report_error:  # noqa: BLE001 - keep the original error
            logger.error(f"Failed to build the terminal event: {report_error}")
            frames = []
        if frames and tail:
            delimiter, _ = _sse_delimiter_and_markers(tail)
            if not tail.endswith(delimiter):
                # A relay can fail mid-frame; end that frame before ours.
                yield delimiter
        for frame in frames:
            yield frame
        if frames and cause == STREAM_TERMINATION_CONTEXT_LENGTH_EXCEEDED:
            # Re-raising would abort the connection, which a client reads as
            # a transport error: one that settles on response.failed only
            # when the stream ends (the Codex CLI does) never sees its code.
            return
        raise


# Seconds a stream may stay silent before its response.in_progress event is
# sent again. Clients and proxies cut a stream that sends nothing for long
# enough, and a stream is silent until its first token (queueing and prefill)
# and for as long as the model writes a tool call, which is sent only once
# complete.
RESPONSES_STREAM_KEEPALIVE_INTERVAL_S = 15.0

_IN_PROGRESS_FRAME_START = "event: response.in_progress\n"

# The response is over once one of these has gone out.
_FINAL_FRAME_STARTS = ("event: error\n", "event: response.completed\n",
                       "event: response.incomplete\n",
                       "event: response.failed\n")

_JSON_DECODER = json.JSONDecoder()


def _top_level_member(payload: str, key: str) -> Optional[Tuple[int, int, Any]]:
    """Where the value of ``key`` starts and ends in a compact JSON object, and the value.

    Only a member of the object itself counts: each member is decoded to find
    where it ends, so the same key nested in a value, or written inside a
    string, is passed over. None when ``key`` is not a member.
    """
    if not payload.startswith("{"):
        return None
    pos = 1
    try:
        while True:
            name, pos = _JSON_DECODER.raw_decode(payload, pos)
            if payload[pos] != ":":
                return None
            start = pos + 1
            value, pos = _JSON_DECODER.raw_decode(payload, start)
            if name == key:
                return start, pos, value
            if payload[pos] != ",":
                return None
            pos += 1
    except (ValueError, IndexError):
        return None


def _stamp_sequence_number(frame: str, sequence_number: int) -> str:
    """``frame`` with ``sequence_number`` as its event's sequence number.

    Only the digits of the event's own number change: the payload is not
    serialized again, and a ``sequence_number`` inside the response or the
    model's output is left alone. A frame without a number of its own is
    returned unchanged.
    """
    head, sep, rest = frame.partition("\ndata: ")
    payload, newline, tail = rest.partition("\n")
    member = _top_level_member(payload, "sequence_number") if sep else None
    if member is None or type(member[2]) is not int:
        return frame
    start, end, _ = member
    return (f"{head}{sep}{payload[:start]}{sequence_number}{payload[end:]}"
            f"{newline}{tail}")


async def _with_ticks(
        source: AsyncGenerator[Any, None],
        timeout: Callable[[], Optional[float]]) -> AsyncGenerator[Any, None]:
    """Yield the items of ``source``, and None each time ``timeout()`` seconds pass first.

    ``timeout()`` is asked before every wait; None waits for the next item. One
    task reads ``source`` to its end, so a tick never interrupts a read and the
    source is never resumed from another task; that task is cancelled, and the
    source closed, when the consumer stops.
    """
    # One item at a time: the reader waits for the consumer, so a slow client
    # holds back the source as reading it directly would.
    items: asyncio.Queue = asyncio.Queue(maxsize=1)
    end = object()
    consumer_stopped = False

    async def read_all() -> None:
        try:
            async for item in source:
                await items.put(item)
        except asyncio.CancelledError:
            # Stopped while handing over an item, the reader leaves the source
            # suspended at its yield, so it closes the source itself.
            aclose = getattr(source, "aclose", None)
            if aclose is not None:
                await aclose()
            if not consumer_stopped:
                # Not stopped by the consumer, which still waits for the end.
                await items.put(end)
            raise
        except Exception:
            await items.put(end)
            raise
        await items.put(end)

    reader = asyncio.create_task(read_all())
    try:
        while True:
            try:
                item = await asyncio.wait_for(items.get(), timeout())
            except asyncio.TimeoutError:
                yield None
                continue
            if item is end:
                # Raises what reading the source raised.
                await reader
                return
            yield item
    finally:
        consumer_stopped = True
        reader.cancel()
        # Return only once the reader has stopped and closed the source.
        await asyncio.gather(reader, return_exceptions=True)


class ClosingStreamingResponse(StreamingResponse):
    """A streaming response that closes its body iterator however it ends.

    A client that disconnects while a frame is being sent leaves the iterator
    suspended at its yield: on ASGI 2.3 the send is cancelled, on 2.4 it raises
    and Starlette raises ``ClientDisconnect``, skipping any background task. The
    iterator, and the generation stream it reads, would then be closed only when
    garbage-collected.
    """

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self.body_iterator.aclose()


async def keep_responses_stream_alive(
        frames: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    """Send ``frames``, repeating ``response.in_progress`` while they are silent.

    ``frames`` yields one SSE event per item, numbered from 0, the first two
    being ``response.created`` and ``response.in_progress``. Until the response
    ends, the ``response.in_progress`` event is sent again whenever no event has
    gone out for ``RESPONSES_STREAM_KEEPALIVE_INTERVAL_S`` seconds, before the
    first token too. It is part of the stream format and carries the same
    response as at the start, which clients already hold.

    Events are numbered here, where they leave the server. The processor that
    makes them cannot do it: postprocess workers each format events with their
    own copy of it, and when a keepalive is due the reader may already hold the
    next numbered event. A repeated event takes the next number, and every event
    after it is renumbered to its place in the stream. Until then each event
    already carries its place and goes out as it came.
    """
    in_progress: Optional[str] = None
    ended = False
    renumber = False
    sent = 0
    loop = asyncio.get_running_loop()
    last_sent = loop.time()

    def until_keepalive() -> Optional[float]:
        if in_progress is None or ended:
            return None
        return max(
            0.0,
            RESPONSES_STREAM_KEEPALIVE_INTERVAL_S - (loop.time() - last_sent))

    items = _with_ticks(frames, until_keepalive)
    try:
        async for frame in items:
            if frame is None:
                frame = in_progress
                renumber = True
            elif in_progress is None and frame.startswith(
                    _IN_PROGRESS_FRAME_START):
                in_progress = frame
            if renumber:
                frame = _stamp_sequence_number(frame, sent)
            ended = ended or frame.startswith(_FINAL_FRAME_STARTS)
            yield frame
            sent += 1
            last_sent = loop.time()
    finally:
        # Stops the task reading ``frames`` when this stream ends early.
        await items.aclose()


_OPENING_FRAME_STARTS = ("event: response.created\n", _IN_PROGRESS_FRAME_START)


async def relay_after_opening(stream: AsyncGenerator[Union[str, bytes], None],
                              events_sent: int) -> AsyncGenerator[str, None]:
    """Relay a worker's Responses stream to a client that already has its opening.

    The client holds ``events_sent`` events, ``response.created`` and
    ``response.in_progress`` among them, and is kept alive with that
    ``response.in_progress`` by the stream this relay feeds
    (``keep_responses_stream_alive``). So every ``response.created`` and
    ``response.in_progress`` of the worker's is dropped, wherever it comes:
    the worker's opening and its keepalives, which repeat that opening, carry
    the worker's response object, not the one the client holds, and a second
    source of keepalives would only add to the relaying stream's own. Every
    other event is numbered to follow on, and is yielded whole, one per item
    as ``keep_responses_stream_alive`` takes them, wherever ``stream`` splits
    it. ``stream`` is closed when the relay ends.
    """
    delimiter = _SSE_EVENT_DELIMITER.encode()
    pending = b""
    try:
        async for chunk in stream:
            if isinstance(chunk, str):
                chunk = chunk.encode()
            *events, pending = (pending + chunk).split(delimiter)
            for event in events:
                frame = event.decode(errors="replace") + _SSE_EVENT_DELIMITER
                if not event or frame.startswith(_OPENING_FRAME_STARTS):
                    continue
                yield _stamp_sequence_number(frame, events_sent)
                events_sent += 1
        if pending:
            # The stream stopped inside an event.
            yield pending.decode(errors="replace")
    finally:
        aclose = getattr(stream, "aclose", None)
        if aclose is not None:
            await aclose()


def _format_sse_event(event: OpenAIBaseModel) -> str:
    # by_alias: fields such as text.format.schema go out under their wire name.
    return (f"event: {getattr(event, 'type', 'unknown')}\n"
            f"data: {event.model_dump_json(indent=None, by_alias=True)}\n\n")


def _sse_event(event: StreamingResponsesResponse) -> bytes:
    return _format_sse_event(event).encode("utf-8")


async def responses_done_generator(
        response: ResponsesResponse) -> AsyncGenerator[bytes, None]:
    """Stream an already-complete response as a well-formed SSE run.

    Used for a response the context worker finished; Responses has no
    ``[DONE]``.
    """
    payload = response.model_dump(by_alias=True)
    yield _sse_event(
        ResponseCreatedEvent(
            type="response.created",
            response=payload,
            sequence_number=0,
        ))
    yield _sse_event(
        ResponseInProgressEvent(
            type="response.in_progress",
            response=payload,
            sequence_number=1,
        ))
    yield _sse_event(_stream_terminal_event(response, sequence_number=2))
