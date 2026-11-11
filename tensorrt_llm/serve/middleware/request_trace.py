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
"""Record each request a trtllm-serve app receives and the exact reply it sent.

Enabled by ``TRTLLM_REQUEST_TRACE_DIR=<directory>``; unset, nothing is installed. The
trace writes full request bodies and full replies -- prompts and completions -- to disk;
only credential headers are redacted. See ``docs/source/features/request-trace.md``.

Layout, two JSONL files per UTC hour plus the writer's accounting::

    $TRTLLM_REQUEST_TRACE_DIR/
      2026-11-10T12/
        requests-<host>-<pid>-<start>.jsonl   one line per request, once its body is complete
        responses-<host>-<pid>-<start>.jsonl  one line per request, when its reply has ended
      _writer/<generation>.json               what the writer was given and where it went

:class:`RequestTraceMiddleware` is pure ASGI: it reads the request body ahead of the app
and replays it unchanged, passes every later ``receive`` message through, and observes
``send`` without altering it, so ``Request.is_disconnected()`` keeps working and the
recorded reply is byte for byte what the server sent. It is installed inside
``ServerArrivalTimeMiddleware``, whose stamp it records, and outside
``PerfMetricsMiddleware``, whose headers and SSE event it therefore records too. Every
step of the trace is guarded: a failure in its own code costs a line, never the request.
:class:`RequestTrace` turns what the middleware observed into records and hands them to
:class:`RequestTraceWriter`, which completes a response line's reply fields on its
worker thread.
"""

import array
import asyncio
import base64
import functools
import hashlib
import importlib
import json
import os
import re
import sys
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, cast

import msgspec
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from tensorrt_llm.logger import logger
from tensorrt_llm.serve._base.sharded_jsonl_writer import ShardedJsonlRecordWriter
from tensorrt_llm.serve.conversation_id import extract_conversation_id_from_headers

__all__ = [
    "PROMPT_TOKEN_DIGEST_ENCODING",
    "REQUEST_TRACE_DIR_ENV",
    "REQUEST_TRACE_IDENTITY_ENV",
    "IdentityPlugin",
    "RequestTrace",
    "RequestTraceFacts",
    "RequestTraceMiddleware",
    "RequestTraceWriter",
    "classify_ending",
    "load_identity_plugin",
    "prompt_token_digest",
]

#: Directory the trace is written to. Unset or empty: no trace, no middleware.
REQUEST_TRACE_DIR_ENV = "TRTLLM_REQUEST_TRACE_DIR"
#: ``<module>:<callable>`` that adds deployment-specific fields to each request line.
REQUEST_TRACE_IDENTITY_ENV = "TRTLLM_REQUEST_TRACE_IDENTITY"

#: ``(headers, body, route) -> fields``. ``headers`` is the request's case-insensitive
#: header multidict (unredacted), ``body`` the decoded request body (a JSON value, the text
#: of a body that did not decode, or None), ``route`` the URL path. The returned mapping is
#: merged into the request line; ``None`` adds nothing.
IdentityPlugin = Callable[[Mapping[str, str], Any, str], Optional[Mapping[str, Any]]]

# Methods whose requests carry a body; a GET (health, metrics, model list) leaves no trace.
_TRACED_METHODS = frozenset(("POST", "PUT", "PATCH"))
# The request_type values the disaggregated server stamps on the hops it fans one client
# request out to. "context_and_generation" rides on client-facing requests and stays traced.
_INTERNAL_DISAGG_REQUEST_TYPES = frozenset(("context_only", "generation_only"))
# Header names whose values are credentials, matched case-insensitively: these names, plus
# any name containing a fragment. "auth" covers authorization, the forwarded spellings
# gateways use (x-forwarded-authorization, x-original-authorization, x-auth-*) and oauth;
# the rest cover API keys, bearer and session tokens, signed assertions (x-goog-iap-jwt-
# assertion), cloud signatures (x-amz-credential, x-hub-signature-*) and forwarded client
# certificates. A header name a client sends is recorded; only the value is replaced.
_CREDENTIAL_HEADERS = frozenset(("cookie", "set-cookie"))
_CREDENTIAL_HEADER_FRAGMENTS = (
    "auth",
    "api-key",
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "jwt",
    "assertion",
    "credential",
    "signature",
    "cert",
)
_REDACTED = "[redacted]"
# Requests whose conversation id cannot be resolved; a field value, never a path.
_NO_SESSION = "_no_session"
_SESSION_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_MAX_SESSION_LEN = 128
# The request header the serving app reads to decode a msgpack body.
_MSGPACK_HEADER = "x-trtllm-msgpack"
# Generation knobs lifted from the body as sent, so a reader need not know each protocol.
_SAMPLING_KEYS = (
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "temperature",
    "top_p",
    "top_k",
    "n",
    "stream",
    "parallel_tool_calls",
    "store",
    "truncation",
    "reasoning_effort",
)
_RESPONSES_TERMINAL_EVENTS = ("response.completed", "response.incomplete", "response.failed")
_TOOL_CALL_ITEM_TYPES = ("function_call", "custom_tool_call")
# How many frames from the end of a stream are searched for the Responses terminal event.
# It is normally the last frame; the perf-metrics event can follow it, and a relay may
# append an error frame. Only frames that name a terminal event are parsed.
_TAIL_FRAME_CAP = 8
# Replies whose JSON body or stream chunks state a finish reason per choice.
_COMPLETION_OBJECTS = ("chat.completion", "text_completion")

_WRITER_QUEUE_SIZE = 1024
_WRITER_BATCH_SIZE = 32
_WRITER_SHUTDOWN_TIMEOUT_SECONDS = 5

#: The byte layout the prompt-token digest is computed over: int32, little endian, no
#: header -- the buffer a context worker relays as ``prompt_token_ids_b64``, so the digest
#: of a relayed buffer is the digest of its bytes. A reader reproduces it with
#: ``hashlib.sha256(numpy.asarray(ids, dtype="<i4").tobytes())``.
PROMPT_TOKEN_DIGEST_ENCODING = "int32le"

_msgpack_decoder = msgspec.msgpack.Decoder()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -- facts only a handler knows -------------------------------------------------------


def _int32le_bytes(ids: Iterable[int]) -> bytes:
    buffer = array.array("i", ids)
    if buffer.itemsize != 4:  # pragma: no cover - no supported platform
        buffer = array.array("l", ids)
    if sys.byteorder != "little":  # pragma: no cover - no supported platform
        buffer.byteswap()
    return buffer.tobytes()


def prompt_token_digest(
    ids: Optional[Iterable[int]] = None, *, b64: Optional[str] = None
) -> Optional[dict[str, Any]]:
    """``{"count", "sha256", "encoding"}`` of a prompt's token ids, or None.

    Takes the ids, or the base64 int32 buffer a context worker returns in their place;
    the buffer is hashed as it is, because its bytes are the encoding the digest is
    defined over. A digest, never the ids: it lets a reader verify that a rendering of
    the recorded body is the prompt the engine saw without storing tokenizer output.
    None when nothing was given, the input cannot be read, or it holds no id.
    """
    if ids is not None:
        try:
            data = _int32le_bytes(ids)
        except (TypeError, OverflowError, ValueError) as error:
            logger.warning(f"Request trace: cannot digest the prompt token ids: {error}")
            return None
    elif b64:
        try:
            data = base64.b64decode(b64, validate=True)
        except (TypeError, ValueError) as error:
            logger.warning(f"Request trace: cannot decode prompt_token_ids_b64: {error}")
            return None
        data = data[: len(data) - len(data) % 4]
    else:
        return None
    if not data:
        return None
    return {
        "count": len(data) // 4,
        "sha256": hashlib.sha256(data).hexdigest(),
        "encoding": PROMPT_TOKEN_DIGEST_ENCODING,
    }


def _engine_request_field(request_id: Any, prompt_token_ids: Any) -> dict[str, Any]:
    """One ``engine_requests`` entry of the response line: the id and the prompt digest."""
    entry: dict[str, Any] = {"request_id": request_id}
    digest = prompt_token_digest(prompt_token_ids)
    if digest is not None:
        entry["prompt_tokens"] = digest
    return entry


@dataclass
class RequestTraceFacts:
    """What only a handler knows about a traced request, read for the response line.

    The middleware puts one of these in ``request.state.request_trace`` for every request
    it records (and nowhere else: a handler finds None when the trace is off or the request
    is not recorded, and then does nothing). The handlers fill in what they learn as they
    learn it, with plain attribute writes; :meth:`as_fields` turns the object into the
    response line's fields on the writer's thread, and that is where the prompt token ids
    are digested, so the event loop's cost does not grow with the prompt. Until then the
    object holds the ids as the handler gave them: a reference to the list the engine
    request holds, which is never mutated in place, or the base64 buffer a context worker
    returned. The line carries a digest, never the ids.
    """

    #: One entry per engine request the handler submitted for this reply, as recorded:
    #: ``{"request_id", "prompt_token_ids"}``. On the line: ``request_id`` and the
    #: ``prompt_tokens`` digest.
    engine_requests: list[dict[str, Any]] = field(default_factory=list)
    #: The model name the server served the request under.
    served_model: Optional[str] = None
    #: What the disaggregated server knows: ``disagg_request_id``, ``ctx_request_id``,
    #: ``ctx_server`` and ``gen_server``. On the line, the ``prompt_tokens`` digest joins them.
    disaggregated: Optional[dict[str, Any]] = None
    #: The prompt the context worker tokenized, every report as given (ids, base64 buffer);
    #: the latest readable one is digested for the line.
    context_prompts: list[tuple[Optional[Iterable[int]], Optional[str]]] = field(
        default_factory=list
    )
    #: Why a stream stopped, as the producer understood it: ``{"cause", "detail"}``.
    termination: Optional[dict[str, str]] = None

    def add_engine_request(
        self, request_id: Any, prompt_token_ids: Optional[Iterable[int]] = None
    ) -> None:
        """Record an engine request and the prompt ids it was given, to be digested."""
        self.engine_requests.append(
            {"request_id": request_id, "prompt_token_ids": prompt_token_ids}
        )

    def note_disaggregated(self, **facts: Any) -> None:
        """Merge the given disaggregated facts; None values are ignored."""
        known = {key: value for key, value in facts.items() if value is not None}
        if known:
            self.disaggregated = {**(self.disaggregated or {}), **known}

    def set_context_prompt_tokens(
        self, ids: Optional[Iterable[int]] = None, *, b64: Optional[str] = None
    ) -> None:
        """Record the prompt the context worker tokenized; a later report replaces it.

        A disaggregated server may run the prefill again on another context worker
        when a generation reservation falls through; the recorded stream was produced
        from the later prefill. Nothing is decoded or hashed here.
        """
        if ids is not None or b64:
            self.context_prompts.append((ids, b64))

    def note_termination(self, cause: str, detail: str = "") -> None:
        """Record why a stream stopped; the first report wins."""
        if self.termination is None:
            self.termination = {"cause": cause, "detail": detail}

    def as_fields(self) -> dict[str, Any]:
        """The set facts as response-line fields; the digests are computed here."""
        fields: dict[str, Any] = {}
        if self.engine_requests:
            fields["engine_requests"] = [
                _engine_request_field(entry["request_id"], entry.get("prompt_token_ids"))
                for entry in self.engine_requests
            ]
        if self.served_model is not None:
            fields["served_model"] = self.served_model
        disaggregated = dict(self.disaggregated or {})
        for ids, b64 in reversed(self.context_prompts):
            digest = prompt_token_digest(ids, b64=b64)
            if digest is not None:
                disaggregated["prompt_tokens"] = digest
                break
        if disaggregated:
            fields["disaggregated"] = disaggregated
        if self.termination is not None:
            fields["termination"] = dict(self.termination)
        return fields


# -- request side -------------------------------------------------------------------


def _is_credential_header(name: str) -> bool:
    lowered = name.lower()
    return lowered in _CREDENTIAL_HEADERS or any(
        fragment in lowered for fragment in _CREDENTIAL_HEADER_FRAGMENTS
    )


def redact_headers(raw_headers: Iterable[tuple[bytes, bytes]]) -> list[list[str]]:
    """Headers as ordered ``[name, value]`` pairs with credential values redacted.

    Pairs rather than a dict because HTTP allows repeats (two proxies each append their
    own ``x-forwarded-for``). Every header the trace records passes through here: a
    credential header keeps its name, which shows the client sent one, and loses its value.
    """
    pairs = []
    for name, value in raw_headers:
        text_name = name.decode("latin-1")
        text_value = _REDACTED if _is_credential_header(text_name) else value.decode("latin-1")
        pairs.append([text_name, text_value])
    return pairs


def sanitize_session_key(value: Optional[str]) -> str:
    """Reduce a client-supplied conversation id to a stable, bounded key."""
    if not value:
        return _NO_SESSION
    cleaned = _SESSION_UNSAFE.sub("_", str(value).strip())[:_MAX_SESSION_LEN]
    if not cleaned or cleaned.startswith("."):
        return _NO_SESSION
    return cleaned


def resolve_session(headers: Mapping[str, str], body: Any) -> str:
    """The key a request's two lines are stamped with: its conversation id, sanitized.

    Body ``conversation_params.conversation_id`` first, then the conversation-id headers,
    in the precedence the servers resolve a conversation id with.
    """
    conversation_id = None
    if isinstance(body, dict):
        params = body.get("conversation_params")
        if isinstance(params, dict):
            conversation_id = params.get("conversation_id")
    if not conversation_id:
        conversation_id = extract_conversation_id_from_headers(headers)
    return sanitize_session_key(conversation_id)


def is_internal_disaggregated_hop(body: Any) -> bool:
    """True for a hop the disaggregated server generated rather than client traffic.

    Read off the body because nothing else separates the two on a worker. Unknown values
    read as client traffic: a request is recorded rather than dropped when in doubt.
    """
    if not isinstance(body, dict):
        return False
    params = body.get("disaggregated_params")
    if not isinstance(params, dict):
        return False
    return params.get("request_type") in _INTERNAL_DISAGG_REQUEST_TYPES


def decode_request_body(raw: bytes, headers: Mapping[str, str]) -> tuple[Any, Optional[str]]:
    """The request body as the server decodes it: JSON, or msgpack when the header says so.

    Returns ``(value, None)``, or ``(text, error)`` for a body that did not decode -- the
    text is kept because such a request is on its way to a validation rejection, and its
    body exists nowhere else. An empty body is ``(None, None)``. A body nested deeper than
    the decoder can follow (``RecursionError``) is a body that did not decode, as it is for
    the server, which answers it with a 400.
    """
    if not raw:
        return None, None
    try:
        if headers.get(_MSGPACK_HEADER) == "1":
            return _msgpack_decoder.decode(raw), None
        return json.loads(raw), None
    except (ValueError, RecursionError) as error:  # msgspec.DecodeError is a ValueError
        return raw.decode("utf-8", "replace"), f"{type(error).__name__}: {error}"


def sampling_as_sent(body: Any) -> Optional[dict[str, Any]]:
    """The generation knobs the client sent, keyed as it sent them.

    A key the client omitted is omitted here too: that a request carried no token cap is
    itself a fact a reader is after. Nested knobs (``reasoning.effort``, ``text.verbosity``,
    ``thinking``, ``tool_choice``) are flattened.
    """
    if not isinstance(body, dict):
        return None
    sampling: dict[str, Any] = {key: body[key] for key in _SAMPLING_KEYS if key in body}
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        for key in ("effort", "summary"):
            if reasoning.get(key) is not None:
                sampling[f"reasoning_{key}"] = reasoning[key]
    text = body.get("text")
    if isinstance(text, dict) and text.get("verbosity") is not None:
        sampling["verbosity"] = text["verbosity"]
    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        for key in ("type", "budget_tokens"):
            if thinking.get(key) is not None:
                sampling[f"thinking_{key}"] = thinking[key]
    tool_choice = body.get("tool_choice")
    if isinstance(tool_choice, str):
        sampling["tool_choice"] = tool_choice
    elif isinstance(tool_choice, dict) and tool_choice.get("type") is not None:
        sampling["tool_choice"] = tool_choice["type"]
    return sampling or None


def _server_arrival_time(scope: Scope) -> Optional[float]:
    """The arrival stamp ``ServerArrivalTimeMiddleware`` put in the request state.

    Taken from the steady clock the perf metrics use, so a trace line lines up with a
    perf-metrics record; not a wall clock.
    """
    value = scope.get("state", {}).get("server_arrival_time")
    return float(value) if isinstance(value, (int, float)) else None


# -- response side ------------------------------------------------------------------


def classify_ending(
    *,
    started: bool,
    completed: bool,
    http_status: Optional[int],
    disconnected: bool,
    send_failed: bool,
    cancelled: bool,
    failed: bool,
) -> str:
    """How a request ended, from what the middleware observed.

    ``completed``: the reply's final body chunk was handed to the server, with a 1xx-3xx
    status. uvicorn's HTTP protocols accept a reply for a client that already left without
    raising, so a completed reply is what the server sent, not proof the client read it;
    the response line's ``disconnected`` field says whether the client had gone;
    ``rejected_<code>``: the same, with a 4xx status (a validation rejection, a 404);
    ``error``: a 5xx reply, an exception out of the app (the framework answers 500), or a
    reply that stopped without its final chunk and without a disconnect to explain it;
    ``client_disconnected``: the client went away before the reply was complete -- the
    app saw ``http.disconnect`` and stopped (Starlette's stream listener), raised because
    the client was gone (Starlette's ``ClientDisconnect`` while reading the body), or a
    send raised;
    ``not_started``: the app ended without sending anything and without an exception, as a
    streaming response does when the client disconnects before the reply begins;
    ``cancelled``: the request's task was cancelled (server shutdown) with no disconnect.

    Args:
        started: ``http.response.start`` was sent.
        completed: an ``http.response.body`` message without ``more_body`` was sent.
        http_status: the status of ``http.response.start``, when sent.
        disconnected: an ``http.disconnect`` message was received.
        send_failed: a ``send`` raised. uvicorn's HTTP protocols return silently for a gone
            client (its WebSocket protocols raise), so under uvicorn the disconnect shows on
            the receive side instead.
        cancelled: the app raised ``asyncio.CancelledError``.
        failed: the app raised any other exception.
    """
    if completed:
        if http_status is not None and 400 <= http_status < 500:
            return f"rejected_{http_status}"
        if http_status is not None and http_status >= 500:
            return "error"
        return "completed"
    if not started:
        if failed:
            return "client_disconnected" if disconnected else "error"
        if cancelled and not disconnected:
            return "cancelled"
        return "not_started"
    if disconnected or send_failed:
        return "client_disconnected"
    if cancelled:
        return "cancelled"
    return "error"


def _response_kind(headers: Iterable[tuple[bytes, bytes]]) -> str:
    content_type = b""
    for name, value in headers:
        if name.lower() == b"content-type":
            content_type = value.lower()
            break
    if b"text/event-stream" in content_type:
        return "sse_text"
    if b"json" in content_type:
        return "json"
    return "text"


def response_payload(
    headers: Iterable[tuple[bytes, bytes]], chunks: Sequence[bytes]
) -> dict[str, Any]:
    """The reply as the server sent it: ``{"kind": ..., "body": ...}``.

    ``sse_text`` holds the stream's frames as one text, ``json`` the decoded JSON body and
    ``text`` anything else (a JSON body that does not parse included), decoded as UTF-8
    with replacement. The chunks are joined before decoding, so a multi-byte character
    split across two chunks survives.
    """
    kind = _response_kind(headers)
    data = b"".join(chunks)
    if kind == "json":
        try:
            return {"kind": kind, "body": json.loads(data)}
        except (ValueError, RecursionError):
            kind = "text"
    return {"kind": kind, "body": data.decode("utf-8", "replace")}


def _last_frames(text: str, count: int) -> list[str]:
    """The last ``count`` SSE frames of a stream, oldest first, unparsed."""
    parts = text.rstrip().rsplit("\n\n", count)
    if len(parts) > count:
        parts = parts[1:]
    return parts


def _parse_frame(frame: str) -> Optional[tuple[Optional[str], dict[str, Any]]]:
    """One SSE frame as ``(event type, data)``, or None when its data is not a JSON object."""
    event_type, data = None, None
    for line in frame.split("\n"):
        if line.startswith("event: "):
            event_type = line[len("event: ") :].strip()
        elif line.startswith("data: "):
            try:
                data = json.loads(line[len("data: ") :])
            except ValueError:
                data = None
    return (event_type, data) if isinstance(data, dict) else None


def _tail_frames(
    text: str, count: int = _TAIL_FRAME_CAP
) -> list[tuple[Optional[str], dict[str, Any]]]:
    """The last ``count`` SSE frames of a stream as ``(event type, data)``, newest last.

    Only the tail is split and parsed; parsing a long stream's every frame would cost a
    JSON parse per delta.
    """
    frames = []
    for frame in _last_frames(text, count):
        parsed = _parse_frame(frame)
        if parsed is not None:
            frames.append(parsed)
    return frames


def _terminal_responses_object(text: Optional[str], json_body: Any) -> Optional[dict[str, Any]]:
    """The Responses object a reply ended on, from a stream's tail or a JSON body.

    The stream's last ``_TAIL_FRAME_CAP`` frames are scanned newest first, and only a
    frame that names a terminal event is parsed, so trailing frames (the perf-metrics
    event, a relay's error frame) cost a substring check each and no more.
    """
    if text is not None:
        for frame in reversed(_last_frames(text, _TAIL_FRAME_CAP)):
            if not any(name in frame for name in _RESPONSES_TERMINAL_EVENTS):
                continue
            parsed = _parse_frame(frame)
            if parsed is None:
                continue
            event_type, data = parsed
            response = data.get("response")
            if (event_type or data.get("type")) in _RESPONSES_TERMINAL_EVENTS and isinstance(
                response, dict
            ):
                return response
        return None
    if isinstance(json_body, dict) and json_body.get("object") == "response":
        return json_body
    return None


def _responses_summary(response: dict[str, Any]) -> dict[str, Any]:
    """What the terminal Responses object states about how the generation ended.

    ``response_status`` (completed / incomplete / failed), ``incomplete_reason`` and
    ``error_code`` tell a truncated generation from a finished one; ``response_id``,
    ``usage``, ``max_output_tokens`` (the cap the generation ran under, server default
    included), the output item types and the tool call ids let a reader skip the body.
    """
    summary: dict[str, Any] = {}
    if isinstance(response.get("status"), str):
        summary["response_status"] = response["status"]
    details = response.get("incomplete_details")
    if isinstance(details, dict) and isinstance(details.get("reason"), str):
        summary["incomplete_reason"] = details["reason"]
    error = response.get("error")
    if isinstance(error, dict) and error.get("code") is not None:
        summary["error_code"] = error["code"]
    if isinstance(response.get("id"), str):
        summary["response_id"] = response["id"]
    if isinstance(response.get("usage"), dict):
        summary["usage"] = response["usage"]
    if response.get("max_output_tokens") is not None:
        summary["max_output_tokens"] = response["max_output_tokens"]
    output = response.get("output")
    if isinstance(output, list):
        items = [item for item in output if isinstance(item, dict)]
        summary["output_item_types"] = [item.get("type") for item in items]
        summary["output_call_ids"] = [
            item["call_id"]
            for item in items
            if item.get("type") in _TOOL_CALL_ITEM_TYPES and isinstance(item.get("call_id"), str)
        ]
    return summary


# A chunk carries a finish reason or a usage object only when the key is followed by
# something other than null. Completion chunks are serialized with every field present
# ("finish_reason":null and "usage":null on each one), so the frames that merely mention a
# key are skipped by a regular expression instead of being parsed one by one.
_FINISH_REASON_SET = re.compile(r'"finish_reason"\s*:(?!\s*null)')
_USAGE_SET = re.compile(r'"usage"\s*:(?!\s*null)')


def _chat_chunks(text: str, pattern: re.Pattern[str]) -> Iterable[dict[str, Any]]:
    """The chat completion chunks of the frames in which ``pattern`` finds a set value."""
    if pattern.search(text) is None:
        return
    for frame in text.split("\n\n"):
        if pattern.search(frame) is None:
            continue
        for line in frame.split("\n"):
            if not line.startswith("data: ") or line.startswith("data: [DONE]"):
                continue
            try:
                data = json.loads(line[len("data: ") :])
            except ValueError:
                continue
            if isinstance(data, dict) and data.get("object") == "chat.completion.chunk":
                yield data


def _chat_summary(text: Optional[str], json_body: Any) -> dict[str, Any]:
    """``finish_reasons`` per choice and the ``usage`` object of a completion reply.

    From the JSON body of a chat or text completion, or from the streamed chat completion
    chunks that carry them: a chat stream states ``usage`` only when the client asked for
    it (``stream_options.include_usage``), and the trace never changes what a client asked
    for. Other protocols' frames (Anthropic ``message_delta`` has a ``usage`` too) are
    left alone.
    """
    summary: dict[str, Any] = {}
    if text is None:
        if isinstance(json_body, dict) and json_body.get("object") in _COMPLETION_OBJECTS:
            choices = json_body.get("choices")
            if isinstance(choices, list):
                summary["finish_reasons"] = [
                    choice.get("finish_reason") for choice in choices if isinstance(choice, dict)
                ]
            if isinstance(json_body.get("usage"), dict):
                summary["usage"] = json_body["usage"]
        return summary
    reasons: dict[int, Any] = {}
    for data in _chat_chunks(text, _FINISH_REASON_SET):
        for choice in data.get("choices") or []:
            if isinstance(choice, dict) and choice.get("finish_reason") is not None:
                reasons[int(choice.get("index") or 0)] = choice["finish_reason"]
    if reasons:
        summary["finish_reasons"] = [reasons[index] for index in sorted(reasons)]
    for data in _chat_chunks(text, _USAGE_SET):
        if isinstance(data.get("usage"), dict):
            summary["usage"] = data["usage"]
    return summary


def summarize_reply(payload: Optional[dict[str, Any]], route: str) -> dict[str, Any]:
    """Fields lifted from the recorded reply so a reader can skip its body.

    Everything here is derived from ``payload``, which the response line already holds;
    a reply that is neither a Responses object nor chat-shaped adds nothing. A Responses
    route's snapshot frames are large and share no key with the chat summary, so they are
    not scanned for one.
    """
    if payload is None:
        return {}
    text = payload["body"] if payload["kind"] == "sse_text" else None
    json_body = payload["body"] if payload["kind"] == "json" else None
    response = _terminal_responses_object(text, json_body)
    if response is not None:
        return _responses_summary(response)
    if route.endswith("/responses"):
        return {}
    return _chat_summary(text, json_body)


# -- identity plugin ----------------------------------------------------------------


def load_identity_plugin(spec: str) -> IdentityPlugin:
    """Import the ``<module>:<callable>`` named by ``TRTLLM_REQUEST_TRACE_IDENTITY``.

    Raises ``ValueError`` for a spec that does not name an importable callable: the
    operator asked for the fields explicitly, so a plugin that cannot load stops the
    server at startup rather than silently writing lines without them.
    """
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name.strip() or not attribute.strip():
        raise ValueError(
            f"Invalid request trace identity plugin {spec!r}: expected <module>:<callable>"
        )
    try:
        module = importlib.import_module(module_name.strip())
        plugin: object = getattr(module, attribute.strip())
    except (ImportError, AttributeError) as error:
        raise ValueError(f"Cannot load request trace identity plugin {spec!r}: {error}") from error
    if not callable(plugin):
        raise ValueError(f"Request trace identity plugin {spec!r} is not callable")
    return cast(IdentityPlugin, plugin)


# -- writer and recorder ------------------------------------------------------------


#: Key of a callable a response line carries to the writer: it returns the fields the
#: reply yields (headers, body, summaries) and runs on the writer's thread.
_DEFERRED_FIELDS = "_complete"


def _request_digest(raw_body: bytes) -> dict[str, Any]:
    """The request line's body hash, computed on the writer thread (6 ms for 10 MB)."""
    return {"body_sha256": hashlib.sha256(raw_body).hexdigest()}


def _reply_fields(
    trace_id: str, route: str, headers: list[tuple[bytes, bytes]], chunks: Sequence[bytes]
) -> dict[str, Any]:
    """The response line's reply fields: joined, decoded and summarized on the writer thread.

    A failing summary costs the summary, never the line.
    """
    payload = response_payload(headers, chunks)
    fields: dict[str, Any] = {"headers": redact_headers(headers), "response": payload}
    try:
        fields.update(summarize_reply(payload, route))
    except Exception as error:
        logger.warning("Request trace %s: cannot summarize the reply: %s", trace_id, error)
    return fields


def _response_fields(
    trace_id: str,
    route: str,
    headers: Optional[list[tuple[bytes, bytes]]],
    chunks: Sequence[bytes],
    facts: Optional[RequestTraceFacts],
) -> dict[str, Any]:
    """The response line's deferred fields, on the writer thread: the reply and the facts.

    The reply fields need a reply that started (``headers``); the handler's facts need
    the handler to have recorded any. A failure in the facts costs them, never the line.
    """
    fields: dict[str, Any] = {}
    if headers is not None:
        fields.update(_reply_fields(trace_id, route, headers, chunks))
    if facts is not None:
        try:
            fields.update(facts.as_fields())
        except Exception as error:
            logger.warning(
                "Request trace %s: cannot record the handler's facts: %s", trace_id, error
            )
    return fields


class RequestTraceWriter(ShardedJsonlRecordWriter):
    """The trace's writer: one directory per UTC hour, one file per record kind.

    A response line reaches the writer with its reply still as the raw chunks and a
    callable that turns them into fields; the callable runs here, on the writer's thread,
    so joining, decoding and summarizing a long reply never happens on the event loop.
    """

    def __init__(self, output_dir: Optional[str], *, shard_suffix: Optional[str] = None):
        super().__init__(
            output_dir,
            subject="request trace",
            queue_size=_WRITER_QUEUE_SIZE,
            batch_size=_WRITER_BATCH_SIZE,
            shutdown_timeout=_WRITER_SHUTDOWN_TIMEOUT_SECONDS,
            shard_suffix=shard_suffix,
        )

    def _shard_key(self, record: Any) -> str:
        # The hour of the record's own timestamp, so a line is found by when it was
        # written: "2026-11-10T12:34:56.789+00:00"[:13] is "2026-11-10T12".
        event = record["event"]
        stamp = record["recorded_at" if event == "request" else "finished_at"]
        return f"{stamp[:13]}/{event}s"

    def _encode_record(self, record: Any) -> tuple[bytes, int]:
        complete = record.pop(_DEFERRED_FIELDS, None) if isinstance(record, dict) else None
        if complete is not None:
            record.update(complete())
        return super()._encode_record(record)


class RequestTrace:
    """Build the request and response lines and hand them to the writer.

    Constructed from the environment by :meth:`from_env`, started and closed from the
    app lifespan, and installed through ``register_middlewares(request_trace=...)``.
    """

    #: Keys of the request line itself; a plugin field with one of these names is refused.
    RESERVED_KEYS = frozenset(
        (
            "event",
            "trace_id",
            "session",
            "recorded_at",
            "server_arrival_time",
            "method",
            "route",
            "headers",
            "body",
            "body_parse_error",
            "body_complete",
            "body_sha256",
            "sampling",
        )
    )

    def __init__(self, writer: RequestTraceWriter, identity: Optional[IdentityPlugin] = None):
        """Keep the writer the lines go to and the optional identity plugin.

        Args:
            writer: Where the lines go.
            identity: Adds fields to each request line; see :data:`IdentityPlugin`.
        """
        self._writer = writer
        self._identity = identity
        self._identity_failures = 0
        self._refused_keys = 0
        self._failures = 0

    def note_failure(self, step: str, error: BaseException) -> None:
        """Count a failure inside the trace's own code; the first and every 1000th is logged.

        The middleware calls this instead of letting the exception reach the request: a
        trace that cannot record a line must still let the request through untouched.
        """
        self._failures += 1
        if self._failures == 1 or self._failures % 1000 == 0:
            logger.warning(
                "Request trace: the %s was not recorded (%s: %s); %d failure(s) so far",
                step,
                type(error).__name__,
                error,
                self._failures,
            )

    @classmethod
    def from_env(cls) -> Optional["RequestTrace"]:
        """The trace the environment asks for, or None when the directory is unset.

        Reads ``TRTLLM_REQUEST_TRACE_DIR`` and ``TRTLLM_REQUEST_TRACE_IDENTITY`` when
        called, not at import time, so a setting made before the server is constructed
        is seen. A plugin that cannot load raises ``ValueError``.
        """
        output_dir = os.environ.get(REQUEST_TRACE_DIR_ENV, "").strip()
        if not output_dir:
            return None
        identity_spec = os.environ.get(REQUEST_TRACE_IDENTITY_ENV, "").strip()
        identity = load_identity_plugin(identity_spec) if identity_spec else None
        return cls(RequestTraceWriter(output_dir), identity=identity)

    @property
    def writer(self) -> RequestTraceWriter:
        return self._writer

    async def start(self) -> None:
        await self._writer.start()

    async def close(self) -> None:
        await self._writer.close()

    def record_request(
        self,
        *,
        trace_id: str,
        scope: Scope,
        raw_body: bytes,
        body: Any,
        parse_error: Optional[str],
        body_complete: bool,
    ) -> str:
        """Queue the request line and return the session key its response line shares.

        The body hash is left to the writer's thread (:func:`_request_digest`); the body
        itself was decoded by the middleware, which needs it for the session key and the
        internal-hop check before the handler runs.
        """
        headers = Headers(scope=scope)
        route = str(scope.get("path", ""))
        session = resolve_session(headers, body)
        record: dict[str, Any] = {
            "event": "request",
            "trace_id": trace_id,
            "session": session,
            "recorded_at": _utc_now(),
            "server_arrival_time": _server_arrival_time(scope),
            "method": scope.get("method"),
            "route": route,
            "headers": redact_headers(scope.get("headers", [])),
            "body": body,
        }
        if parse_error is not None:
            record["body_parse_error"] = parse_error
        if not body_complete:
            record["body_complete"] = False
        sampling = sampling_as_sent(body)
        if sampling is not None:
            record["sampling"] = sampling
        self._add_identity(record, headers, body, route)
        record[_DEFERRED_FIELDS] = functools.partial(_request_digest, raw_body)
        self._writer.submit(record)
        return session

    def _add_identity(
        self, record: dict[str, Any], headers: Headers, body: Any, route: str
    ) -> None:
        """Merge the plugin's fields; a failing plugin costs its fields, never the line."""
        if self._identity is None:
            return
        try:
            fields = self._identity(headers, body, route)
            accepted = {
                str(key): value
                for key, value in (fields or {}).items()
                if str(key) not in self.RESERVED_KEYS
            }
            json.dumps(accepted, allow_nan=False)  # a field JSON cannot hold would cost the line
        except Exception as error:
            self._identity_failures += 1
            if self._identity_failures == 1 or self._identity_failures % 1000 == 0:
                logger.warning(
                    "Request trace identity plugin failed %d time(s); its fields are left off: "
                    "%s: %s",
                    self._identity_failures,
                    type(error).__name__,
                    error,
                )
            return
        refused = len(fields or {}) - len(accepted)
        if refused:
            self._refused_keys += refused
            if self._refused_keys == refused or self._refused_keys % 1000 < refused:
                logger.warning(
                    "Request trace identity plugin returned %d reserved key(s), refused",
                    refused,
                )
        record.update(accepted)

    def record_response(
        self,
        *,
        trace_id: str,
        session: str,
        route: str,
        status: str,
        http_status: Optional[int],
        headers: Optional[list[tuple[bytes, bytes]]],
        chunks: Sequence[bytes],
        error: Optional[BaseException],
        disconnected: bool,
        facts: Optional[RequestTraceFacts] = None,
    ) -> None:
        """Queue the response line: how the request ended and the reply as sent.

        The reply's chunks are not touched here; the writer's thread joins, decodes and
        summarizes them (:func:`_reply_fields`), so the event loop does no work that grows
        with the reply. ``facts`` are what the handler recorded (:class:`RequestTraceFacts`);
        the same thread digests their prompt token ids (:func:`_response_fields`).
        """
        record: dict[str, Any] = {
            "event": "response",
            "trace_id": trace_id,
            "session": session,
            "finished_at": _utc_now(),
            "status": status,
            "http_status": http_status,
        }
        if disconnected:
            record["disconnected"] = True
        if error is not None:
            record["error"] = f"{type(error).__name__}: {error}"
        if headers is not None or facts is not None:
            record[_DEFERRED_FIELDS] = functools.partial(
                _response_fields, trace_id, route, headers, chunks, facts
            )
        self._writer.submit(record)


# -- middleware ---------------------------------------------------------------------


class _TracedRequest:
    """What the middleware observes about one request, and when it writes each line.

    The request line is written as soon as the body is complete, before the handler runs,
    so a request a crash takes down is still on disk; a body cut short by a disconnect is
    written at the first ``send`` or at the end, marked incomplete. The response line is
    written when the app returns or raises. Every hook runs under a guard: a failure in
    the trace's own code is counted and logged (:meth:`RequestTrace.note_failure`) and
    costs the line it was building, never the request. ``asyncio.CancelledError`` is not
    an ``Exception`` and passes through.
    """

    def __init__(self, trace: RequestTrace, scope: Scope) -> None:
        self._trace = trace
        self._scope = scope
        self.trace_id = f"tr_{uuid.uuid4().hex}"
        self._body = bytearray()
        self._body_complete = False
        self._request_recorded = False
        self._request_failed = False
        self._skipped = False
        self._session = _NO_SESSION
        self._disconnected = False
        self._started = False
        self._completed = False
        self._http_status: Optional[int] = None
        self._response_headers: Optional[list[tuple[bytes, bytes]]] = None
        self._chunks: list[bytes] = []
        self._send_error: Optional[BaseException] = None
        self._app_error: Optional[BaseException] = None
        self._facts: Optional[RequestTraceFacts] = None
        self._finished = False

    def on_received(self, message: Message) -> None:
        try:
            kind = message["type"]
            if kind == "http.request":
                self._body += message.get("body", b"")
                if not message.get("more_body", False):
                    self._body_complete = True
                    self._record_request()
            elif kind == "http.disconnect" and not self._completed:
                # After the reply is complete the server answers every receive with a
                # disconnect message whether or not the client is still there, so only a
                # disconnect seen before completion says anything about the client.
                self._disconnected = True
        except Exception as error:
            self._trace.note_failure("request line", error)

    def on_sent(self, message: Message) -> None:
        try:
            kind = message["type"]
            if kind == "http.response.start":
                self._started = True
                self._http_status = int(message["status"])
                self._response_headers = [
                    (bytes(name), bytes(value)) for name, value in message.get("headers", [])
                ]
                self._record_request()
            elif kind == "http.response.body":
                self._chunks.append(bytes(message.get("body", b"")))
                if not message.get("more_body", False):
                    self._completed = True
        except Exception as error:
            self._trace.note_failure("response line", error)

    def on_send_failed(self, error: BaseException) -> None:
        self._send_error = error

    def on_app_error(self, error: BaseException) -> None:
        self._app_error = error

    def _record_request(self) -> None:
        """Write the request line once; ``_request_recorded`` is set only when it is queued.

        A failure latches instead, so the line is not attempted again at the next hook and
        the response line, when it comes, stands alone with the same ``trace_id``.
        """
        if self._request_recorded or self._request_failed or self._skipped:
            return
        raw_body = bytes(self._body)
        if not raw_body:
            # A body-less POST (the disaggregated router's polls, a profiler switch) has
            # nothing to record.
            self._skipped = True
            return
        try:
            body, parse_error = decode_request_body(raw_body, Headers(scope=self._scope))
            if is_internal_disaggregated_hop(body):
                self._skipped = True
                return
            self._session = self._trace.record_request(
                trace_id=self.trace_id,
                scope=self._scope,
                raw_body=raw_body,
                body=body,
                parse_error=parse_error,
                body_complete=self._body_complete,
            )
        except Exception as error:
            self._request_failed = True
            self._trace.note_failure("request line", error)
            return
        self._request_recorded = True
        self._body = bytearray()  # the queued line holds the body now
        # The handler runs after this (the body is read before the app): give it the
        # object it fills with what only it knows. Absent for a request that is not
        # recorded, so a handler that finds None does no work at all.
        self._facts = RequestTraceFacts()
        self._scope.setdefault("state", {})["request_trace"] = self._facts

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            self._record_request()
            if self._skipped:
                return
            error = self._app_error if self._app_error is not None else self._send_error
            status = classify_ending(
                started=self._started,
                completed=self._completed,
                http_status=self._http_status,
                disconnected=self._disconnected,
                send_failed=self._send_error is not None,
                cancelled=isinstance(self._app_error, asyncio.CancelledError),
                failed=self._app_error is not None
                and not isinstance(self._app_error, asyncio.CancelledError),
            )
            self._trace.record_response(
                trace_id=self.trace_id,
                session=self._session,
                route=str(self._scope.get("path", "")),
                status=status,
                http_status=self._http_status,
                headers=self._response_headers,
                chunks=tuple(self._chunks),  # the writer thread owns this copy
                error=error,
                disconnected=self._disconnected,
                facts=self._facts,
            )
        except Exception as error:
            self._trace.note_failure("response line", error)


class RequestTraceMiddleware:
    """Pure ASGI middleware that records each request and the reply the server sent.

    The request body is read before the app runs and replayed to it message by message,
    so the request line exists before the handler starts whether or not the handler reads
    the body. Once the replay is exhausted, ``receive`` is the server's own: every later
    message -- ``http.disconnect`` above all -- passes through unchanged, with nothing
    awaited besides the server's callable, so ``Request.is_disconnected()`` and
    Starlette's disconnect listener keep working, which a wrapping middleware (FastAPI's
    ``@app.middleware("http")``) breaks. ``send`` is observed the same way: every message
    goes out unchanged and is recorded once the server accepted it (uvicorn accepts a
    reply for a client that already left, so the response line also says whether a
    disconnect was seen). Registered inside ``ServerArrivalTimeMiddleware`` and outside
    ``PerfMetricsMiddleware`` (serve/middleware/register.py), so the request line carries
    the arrival stamp and the response line the reply with everything the perf-metrics
    middleware added. Nothing the trace does can change the reply: its hooks are guarded,
    and a failure inside them costs a line, never the request.
    """

    def __init__(self, app: ASGIApp, trace: RequestTrace) -> None:
        self.app = app
        self._trace = trace

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or str(scope.get("method", "")).upper() not in _TRACED_METHODS:
            await self.app(scope, receive, send)
            return
        traced = _TracedRequest(self._trace, scope)
        replay: list[Message] = []

        async def receive_traced() -> Message:
            if replay:
                return replay.pop(0)
            message = await receive()
            traced.on_received(message)
            return message

        async def send_traced(message: Message) -> None:
            try:
                await send(message)
            except Exception as error:
                traced.on_send_failed(error)
                raise
            traced.on_sent(message)

        try:
            while True:
                message = await receive()
                replay.append(message)
                traced.on_received(message)
                if message["type"] != "http.request" or not message.get("more_body", False):
                    break
            await self.app(scope, receive_traced, send_traced)
        except BaseException as error:
            traced.on_app_error(error)
            raise
        finally:
            traced.finish()
