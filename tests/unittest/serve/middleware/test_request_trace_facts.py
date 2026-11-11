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
"""The request trace records what only a handler knows, on the response line.

The middleware seeds ``request.state.request_trace`` with a ``RequestTraceFacts`` for
every request it records; the handlers fill in the engine request ids, a digest of the
prompt tokens, the served model and why a stream stopped. A handler that never touches
it leaves the response line exactly as the middleware writes it on its own.
"""

import array
import asyncio
import base64
import hashlib
import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import numpy as np
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from starlette.testclient import TestClient
from test_common.serve.app_factory import AggConfig
from test_common.serve.fakes import ScriptedOutput, ScriptedStep

from tensorrt_llm.executor import RequestError
from tensorrt_llm.serve.middleware import request_trace as request_trace_module
from tensorrt_llm.serve.middleware.register import register_middlewares
from tensorrt_llm.serve.middleware.request_trace import (
    PROMPT_TOKEN_DIGEST_ENCODING,
    RequestTrace,
    RequestTraceFacts,
    RequestTraceWriter,
    prompt_token_digest,
)
from tensorrt_llm.serve.openai_disagg_server import RawRequestResponseHooks
from tensorrt_llm.serve.openai_server import _trace_facts
from tensorrt_llm.serve.responses_utils import (
    STREAM_TERMINATION_INTERNAL_ERROR,
    STREAM_TERMINATION_UPSTREAM_ERROR,
    guard_responses_stream,
    note_stream_termination,
)

pytestmark = pytest.mark.cpu_only

CHAT = "/v1/chat/completions"
# The response-line keys the middleware writes on its own, before any handler fact.
BASE_RESPONSE_KEYS = {
    "event",
    "trace_id",
    "session",
    "finished_at",
    "status",
    "http_status",
    "headers",
    "response",
    "disconnected",
    "error",
    "finish_reasons",
    "usage",
    "response_status",
    "incomplete_reason",
    "error_code",
    "response_id",
    "max_output_tokens",
    "output_item_types",
    "output_call_ids",
}
FACT_KEYS = {"engine_requests", "served_model", "disaggregated", "termination"}


def int32le(ids):
    return array.array("i", ids).tobytes()


def make_trace(tmp_path):
    return RequestTrace(RequestTraceWriter(str(tmp_path), shard_suffix="-w"))


def lines(directory, kind):
    """The ``kind`` lines of every hour bucket, whatever the writer named its shards."""
    records = []
    for path in sorted(Path(directory).glob(f"*/{kind}-*.jsonl")):
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    return records


def traced_app(trace):
    @asynccontextmanager
    async def lifespan(app):
        await trace.start()
        yield
        await trace.close()

    app = FastAPI(lifespan=lifespan)
    register_middlewares(
        app,
        collect_perf_metrics=False,
        expose_perf_metrics=False,
        perf_metrics_writer=None,
        request_trace=trace,
    )
    return app


def sse(frames):
    async def body():
        for frame in frames:
            yield frame

    return StreamingResponse(body(), media_type="text/event-stream")


# -- the digest ---------------------------------------------------------------------


def test_ids_and_their_base64_buffer_give_the_same_digest():
    ids = [163586, 1, 2, 3, 2**31 - 1, -1]
    from_ids = prompt_token_digest(ids)
    from_b64 = prompt_token_digest(b64=base64.b64encode(int32le(ids)).decode())
    assert from_ids == from_b64
    assert from_ids["count"] == len(ids)
    assert from_ids["encoding"] == PROMPT_TOKEN_DIGEST_ENCODING == "int32le"


def test_the_digest_is_what_a_reader_recomputes_from_the_ids():
    ids = [10, 20, 30]
    assert prompt_token_digest(ids)["sha256"] == hashlib.sha256(int32le(ids)).hexdigest()
    assert (
        prompt_token_digest(np.asarray(ids, dtype=np.int64))["sha256"]
        == hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest()
    ), "the ids the engine holds may be a numpy array; the layout is int32 little endian"


@pytest.mark.parametrize(
    "call",
    [
        lambda: prompt_token_digest(),
        lambda: prompt_token_digest(None, b64=None),
        lambda: prompt_token_digest(b64="not base64!!"),
        # A valid length with a character outside the alphabet: only a validating decode
        # refuses it; a lenient one would digest six bytes nobody sent.
        lambda: prompt_token_digest(b64="AAAA AAAA"),
        lambda: prompt_token_digest(["x", "y"]),
        lambda: prompt_token_digest([2**40]),
        lambda: prompt_token_digest([]),
        lambda: prompt_token_digest(b64=base64.b64encode(b"\x01\x02\x03").decode()),
    ],
    ids=[
        "nothing",
        "none-and-none",
        "not-base64",
        "space-in-base64",
        "not-ints",
        "too-large",
        "no-ids",
        "no-whole-id",
    ],
)
def test_nothing_an_unreadable_input_or_an_empty_prompt_gives_no_digest(call):
    assert call() is None, "a digest of nothing says nothing"


def test_a_buffer_that_is_not_whole_int32s_is_cut_to_whole_ids():
    digest = prompt_token_digest(b64=base64.b64encode(int32le([5, 6]) + b"\x01").decode())
    assert digest["count"] == 2 and digest["sha256"] == prompt_token_digest([5, 6])["sha256"]


# -- the facts object ---------------------------------------------------------------


def test_each_engine_request_is_recorded_with_its_prompt_digest_and_nothing_more():
    facts = RequestTraceFacts()
    facts.add_engine_request(7, [1, 2, 3])
    facts.add_engine_request(8, [4])
    facts.add_engine_request("9", None)
    entries = facts.as_fields()["engine_requests"]
    assert entries == [
        {"request_id": 7, "prompt_tokens": prompt_token_digest([1, 2, 3])},
        {"request_id": 8, "prompt_tokens": prompt_token_digest([4])},
        {"request_id": "9"},
    ]
    assert sorted(entries[0]["prompt_tokens"]) == ["count", "encoding", "sha256"], (
        "a digest, never the ids"
    )
    assert "[1, 2, 3]" not in json.dumps(facts.as_fields())


def test_the_context_workers_prompt_digest_is_replaced_by_a_later_report():
    """A disaggregated server may run the prefill again on another context worker."""
    facts = RequestTraceFacts()
    facts.set_context_prompt_tokens([1, 2, 3])
    facts.set_context_prompt_tokens(b64=base64.b64encode(int32le([9, 9, 9])).decode())
    assert facts.as_fields()["disaggregated"]["prompt_tokens"] == prompt_token_digest([9, 9, 9])
    facts.set_context_prompt_tokens(b64="not base64!!")  # unreadable: the digest stays
    assert facts.as_fields()["disaggregated"]["prompt_tokens"] == prompt_token_digest([9, 9, 9])
    facts.set_context_prompt_tokens(None, b64=None)  # nothing reported: nothing recorded
    assert len(facts.context_prompts) == 3 and facts.context_prompts[-1][1] == "not base64!!"


def test_the_ids_are_held_as_given_and_digested_only_when_the_fields_are_built(monkeypatch):
    """The handlers run on the event loop; the digest belongs to the writer's thread."""
    calls = []

    def recording_digest(ids=None, *, b64=None):
        calls.append((ids, b64))
        return prompt_token_digest(ids, b64=b64)

    monkeypatch.setattr(request_trace_module, "prompt_token_digest", recording_digest)
    ids = [1, 2, 3]
    facts = RequestTraceFacts()
    facts.add_engine_request(1, ids)
    facts.set_context_prompt_tokens(b64=base64.b64encode(int32le([4])).decode())
    assert calls == [], "recording a fact hashes nothing"
    assert facts.engine_requests[0]["prompt_token_ids"] is ids, "a reference, not a copy"
    fields = facts.as_fields()
    assert len(calls) == 2, "the digests are computed where the fields are built"
    assert fields["engine_requests"][0]["prompt_tokens"] == prompt_token_digest(ids)
    assert fields["disaggregated"]["prompt_tokens"] == prompt_token_digest([4])


def test_disaggregated_facts_merge_and_ignore_what_is_not_known():
    facts = RequestTraceFacts()
    facts.note_disaggregated(disagg_request_id=5, ctx_request_id=None)
    facts.note_disaggregated(ctx_server="ctx-a", gen_server=None)
    facts.note_disaggregated(gen_server="gen-b")
    assert facts.disaggregated == {
        "disagg_request_id": 5,
        "ctx_server": "ctx-a",
        "gen_server": "gen-b",
    }
    assert RequestTraceFacts().disaggregated is None


def test_the_first_termination_report_wins():
    facts = RequestTraceFacts()
    facts.note_termination("engine_error", "RequestError: x")
    facts.note_termination("internal_error", "KeyError")
    assert facts.termination == {"cause": "engine_error", "detail": "RequestError: x"}


def test_as_fields_holds_only_what_was_set():
    assert RequestTraceFacts().as_fields() == {}
    facts = RequestTraceFacts()
    facts.served_model = "m"
    facts.add_engine_request(1, [1])
    assert set(facts.as_fields()) == {"served_model", "engine_requests"}
    assert set(facts.as_fields()) <= FACT_KEYS


# -- the plumbing through the middleware -------------------------------------------


def test_a_handler_fills_the_facts_and_the_response_line_carries_them(tmp_path):
    trace = make_trace(tmp_path)
    app = traced_app(trace)

    @app.post(CHAT)
    async def chat(request: Request):
        facts = request.state.request_trace
        assert isinstance(facts, RequestTraceFacts)
        facts.served_model = "served-m"
        facts.add_engine_request(42, [1, 2, 3])
        facts.note_termination("engine_error", "RequestError: dead")
        return {"ok": True}

    with TestClient(app) as client:
        assert client.post(CHAT, json={"a": 1}).status_code == 200
    (request_line,) = lines(tmp_path, "requests")
    (response_line,) = lines(tmp_path, "responses")
    assert not FACT_KEYS & set(request_line), "the facts are response-line fields"
    assert response_line["engine_requests"] == [
        {"request_id": 42, "prompt_tokens": prompt_token_digest([1, 2, 3])}
    ]
    assert response_line["served_model"] == "served-m"
    assert response_line["termination"] == {"cause": "engine_error", "detail": "RequestError: dead"}
    assert "disaggregated" not in response_line


def test_a_handler_that_sets_nothing_leaves_the_response_line_as_before(tmp_path):
    trace = make_trace(tmp_path)
    app = traced_app(trace)

    @app.post(CHAT)
    async def chat(request: Request):
        assert isinstance(request.state.request_trace, RequestTraceFacts)
        return {"ok": True}

    with TestClient(app) as client:
        client.post(CHAT, json={"a": 1})
    (response_line,) = lines(tmp_path, "responses")
    assert not FACT_KEYS & set(response_line)
    assert set(response_line) <= BASE_RESPONSE_KEYS


def test_no_facts_object_exists_for_a_request_the_trace_does_not_record(tmp_path):
    trace = make_trace(tmp_path)
    app = traced_app(trace)
    seen = []

    @app.post(CHAT)
    async def chat(request: Request):
        seen.append(getattr(request.state, "request_trace", None))
        return {}

    @app.get("/health")
    async def health(request: Request):
        seen.append(getattr(request.state, "request_trace", None))
        return {}

    with TestClient(app) as client:
        client.post(CHAT, json={"disaggregated_params": {"request_type": "context_only"}})
        client.get("/health")
    assert seen == [None, None], "an internal hop and a GET give a handler nothing to fill"
    assert lines(tmp_path, "responses") == []


def test_without_the_trace_a_handler_finds_no_facts():
    app = FastAPI()
    register_middlewares(
        app, collect_perf_metrics=False, expose_perf_metrics=False, perf_metrics_writer=None
    )
    seen = []

    @app.post(CHAT)
    async def chat(request: Request):
        seen.append(getattr(request.state, "request_trace", None))
        return {}

    with TestClient(app) as client:
        client.post(CHAT, json={"a": 1})
    assert seen == [None]


def test_the_handlers_lookup_finds_no_facts_without_the_trace():
    """The untraced path must not invent an object: it would then digest every prompt."""
    assert _trace_facts(None) is None
    scope = {"type": "http", "method": "POST", "path": CHAT, "headers": [], "query_string": b""}
    assert _trace_facts(Request({**scope, "state": {"server_arrival_time": 1.0}})) is None
    facts = RequestTraceFacts()
    assert _trace_facts(Request({**scope, "state": {"request_trace": facts}})) is facts


def test_the_prompt_digest_is_computed_on_the_writer_thread_not_in_the_handler(
    tmp_path, monkeypatch
):
    digest_threads = []

    def recording_digest(ids=None, *, b64=None):
        digest_threads.append(threading.get_ident())
        return prompt_token_digest(ids, b64=b64)

    monkeypatch.setattr(request_trace_module, "prompt_token_digest", recording_digest)
    trace = make_trace(tmp_path)
    app = traced_app(trace)
    handler_thread = []
    digests_when_the_handler_returned = []

    @app.post(CHAT)
    async def chat(request: Request):
        request.state.request_trace.add_engine_request(1, list(range(1000)))
        handler_thread.append(threading.get_ident())
        digests_when_the_handler_returned.append(len(digest_threads))
        return {}

    with TestClient(app) as client:
        client.post(CHAT, json={"a": 1})
    (response_line,) = lines(tmp_path, "responses")
    assert response_line["engine_requests"][0]["prompt_tokens"]["count"] == 1000
    assert digests_when_the_handler_returned == [0], "nothing is hashed while the handler runs"
    assert digest_threads and set(digest_threads).isdisjoint(handler_thread), (
        "the digest is computed on the writer's thread, not the one that ran the handler"
    )


# -- why a stream stopped -------------------------------------------------------------


async def _collect(stream):
    out = []
    async for chunk in stream:
        out.append(chunk)
    return out


@pytest.mark.asyncio
async def test_note_stream_termination_reports_a_producer_failure_once_and_reraises():
    reports = []

    async def failing():
        yield b"a"
        yield b"b"
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await _collect(note_stream_termination(failing(), lambda c, d: reports.append((c, d))))
    assert reports == [(STREAM_TERMINATION_INTERNAL_ERROR, "RuntimeError")], (
        "the cause is classified; only engine errors carry their message"
    )


@pytest.mark.asyncio
async def test_note_stream_termination_leaves_a_completed_stream_and_its_frames_alone():
    reports = []

    async def fine():
        yield b"a"
        yield b"b"

    assert await _collect(note_stream_termination(fine(), lambda c, d: reports.append(c))) == [
        b"a",
        b"b",
    ]
    assert reports == []


@pytest.mark.asyncio
async def test_an_upstream_failure_is_classified_as_such():
    reports = []

    async def relay():
        yield b"a"
        raise aiohttp.ClientPayloadError("worker went away")

    with pytest.raises(aiohttp.ClientPayloadError):
        await _collect(note_stream_termination(relay(), lambda c, d: reports.append((c, d))))
    assert reports == [(STREAM_TERMINATION_UPSTREAM_ERROR, "ClientPayloadError")]


@pytest.mark.asyncio
async def test_a_client_hangup_is_not_a_producer_failure():
    reports = []

    async def cancelled():
        yield b"a"
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await _collect(note_stream_termination(cancelled(), lambda c, d: reports.append(c)))
    assert reports == []


@pytest.mark.asyncio
async def test_the_responses_guard_reports_the_cause_and_still_emits_its_terminal_events():
    reports = []

    async def cut():
        yield "event: response.created\ndata: {}\n\n"
        raise RuntimeError("renderer bug")

    def terminal_events(cause, detail, events_sent):
        return [
            f"event: response.failed\ndata: {json.dumps({'cause': cause, 'detail': detail})}\n\n"
        ]

    frames = []
    with pytest.raises(RuntimeError):
        async for frame in guard_responses_stream(
            cut(), terminal_events, on_termination=lambda c, d: reports.append((c, d))
        ):
            frames.append(frame)
    assert reports == [(STREAM_TERMINATION_INTERNAL_ERROR, "RuntimeError")]
    assert frames[-1].startswith("event: response.failed"), (
        "the client still gets the terminal event"
    )


@pytest.mark.asyncio
async def test_the_responses_guard_reports_nothing_for_a_completed_stream():
    reports = []

    async def fine():
        yield "event: response.created\ndata: {}\n\n"
        yield 'event: response.completed\ndata: {"response": {}}\n\n'

    await _collect(
        guard_responses_stream(fine(), lambda *a: [], on_termination=lambda c, d: reports.append(c))
    )
    assert reports == []


def test_a_stream_that_fails_mid_way_ends_with_its_cause_on_the_response_line(tmp_path):
    trace = make_trace(tmp_path)
    app = traced_app(trace)

    @app.post(CHAT)
    async def chat(request: Request):
        facts = request.state.request_trace

        async def frames():
            yield b"data: one\n\n"
            raise RuntimeError("generation died")

        return StreamingResponse(
            note_stream_termination(frames(), facts.note_termination),
            media_type="text/event-stream",
        )

    with TestClient(app, raise_server_exceptions=False) as client:
        client.post(CHAT, json={"stream": True})
    (response_line,) = lines(tmp_path, "responses")
    assert response_line["status"] == "error"
    assert response_line["termination"] == {
        "cause": STREAM_TERMINATION_INTERNAL_ERROR,
        "detail": "RuntimeError",
    }
    assert response_line["response"]["body"] == "data: one\n\n"


# -- the disaggregated server's hooks ------------------------------------------------


def _hooks(facts):
    scope = {
        "type": "http",
        "method": "POST",
        "path": CHAT,
        "headers": [],
        "query_string": b"",
        "state": {"server_arrival_time": 1.0, **({"request_trace": facts} if facts else {})},
    }
    return RawRequestResponseHooks(
        Request(scope), SimpleNamespace(observe=lambda value: None), collect_perf_metrics=False
    )


def test_the_disaggregated_hooks_record_the_ids_the_servers_and_the_prompt_digest():
    facts = RequestTraceFacts()
    hooks = _hooks(facts)
    hooks.on_req_begin(
        SimpleNamespace(
            disaggregated_params=SimpleNamespace(disagg_request_id=None, ctx_request_id=3)
        )
    )
    hooks.on_disagg_request_id(5)
    ids = [7, 8, 9, 10]
    hooks.on_ctx_resp(
        "ctx-a",
        SimpleNamespace(
            prompt_token_ids=None, prompt_token_ids_b64=base64.b64encode(int32le(ids)).decode()
        ),
    )
    hooks.on_first_token("gen-b", SimpleNamespace(), None)
    assert facts.as_fields() == {
        "disaggregated": {
            "ctx_request_id": 3,
            "disagg_request_id": 5,
            "ctx_server": "ctx-a",
            "prompt_tokens": prompt_token_digest(ids),
            "gen_server": "gen-b",
        }
    }


def test_a_context_reply_that_carries_ids_in_the_clear_is_digested_the_same_way():
    facts = RequestTraceFacts()
    hooks = _hooks(facts)
    hooks.on_ctx_resp("ctx-a", SimpleNamespace(prompt_token_ids=[1, 2], prompt_token_ids_b64=None))
    assert facts.as_fields()["disaggregated"]["prompt_tokens"] == prompt_token_digest([1, 2])
    hooks.on_resp_done("", SimpleNamespace(), None)  # a context-only reply: no generation server
    assert "gen_server" not in facts.as_fields()["disaggregated"]


def test_the_disaggregated_hooks_do_nothing_without_the_trace():
    hooks = _hooks(None)
    assert hooks.trace_facts is None
    hooks.on_req_begin(
        SimpleNamespace(disaggregated_params=SimpleNamespace(disagg_request_id=1, ctx_request_id=2))
    )
    hooks.on_disagg_request_id(5)
    hooks.on_ctx_resp("ctx-a", SimpleNamespace(prompt_token_ids=[1], prompt_token_ids_b64=None))
    hooks.on_first_token("gen-b", SimpleNamespace(), None)


# -- the aggregated server, through the real handlers --------------------------------

# threadleak is off for these: they create TestClients and run requests inside the
# test body, so the client's portal thread and the fake server's worker-pool threads
# are still alive when the leak check runs; the serve_env fixture closes them in its
# teardown, after that check (precedent: tests/unittest/serve/test_serve_test_helpers.py).
through_the_server = pytest.mark.threadleak(enabled=False)

_CHAT_BODY = {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}]}


def _wait_for_lines(directory, kind, count, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        records = lines(directory, kind)
        if len(records) >= count:
            return records
        time.sleep(0.02)
    raise AssertionError(f"{kind}: expected {count} lines, found {len(lines(directory, kind))}")


def _server_with_trace(serve_env, model_type="llama"):
    trace_dir = serve_env.tmp_dir / "trace"
    server = serve_env.build_openai_server(
        AggConfig(model_type=model_type, env=(("TRTLLM_REQUEST_TRACE_DIR", str(trace_dir)),))
    )
    return server, trace_dir


def _harmony_reply():
    """Two scripted steps whose token ids the Harmony parser accepts as a final message."""
    harmony = pytest.importorskip("openai_harmony")
    encoding = harmony.load_harmony_encoding(harmony.HarmonyEncodingName.HARMONY_GPT_OSS)
    tokens = encoding.encode("<|channel|>final<|message|>hi there<|return|>", allowed_special="all")
    return ScriptedOutput(
        steps=(
            ScriptedStep(text="<|channel|>final<|message|>hi", token_ids=tuple(tokens[:-2])),
            ScriptedStep(text=" there", token_ids=tuple(tokens[-2:]), finish_reason="stop"),
        )
    )


def _two_tokens(finish_reason="stop"):
    return ScriptedOutput(
        steps=(
            ScriptedStep(text="he", token_ids=(1,)),
            ScriptedStep(text="llo", token_ids=(2,), finish_reason=finish_reason),
        )
    )


@through_the_server
@pytest.mark.parametrize(
    ("path", "body"),
    [
        (CHAT, _CHAT_BODY),
        (CHAT, {**_CHAT_BODY, "stream": True}),
        ("/v1/completions", {"model": "fake-model", "prompt": "hi"}),
        ("/v1/responses", {"model": "fake-model", "input": "hi"}),
        ("/v1/messages", {**_CHAT_BODY, "max_tokens": 8}),
    ],
    ids=["chat", "chat-stream", "completions", "responses", "messages"],
)
def test_every_aggregated_handler_records_its_engine_request(serve_env, path, body):
    server, trace_dir = _server_with_trace(serve_env)
    server.generator.script(_two_tokens())
    client = serve_env.client(server.app)
    response = client.post(path, json=body)
    assert response.status_code == 200, response.text

    (response_line,) = _wait_for_lines(trace_dir, "responses", 1)
    promise = server.generator.generate_calls[-1].result
    assert response_line["engine_requests"] == [
        {
            "request_id": promise.request_id,
            "prompt_tokens": prompt_token_digest(promise.prompt_token_ids),
        }
    ], "the digest is over the ids the engine was given, taken from the promise"
    assert (
        response_line["engine_requests"][0]["prompt_tokens"]["count"]
        == len(promise.prompt_token_ids)
        > 0
    )
    assert response_line["served_model"] == "fake-model"
    assert "termination" not in response_line, "a stream that completes has no termination cause"
    assert response_line["status"] == "completed"


@through_the_server
@pytest.mark.parametrize(
    ("path", "body"),
    [
        (CHAT, {**_CHAT_BODY, "stream": True}),
        ("/v1/completions", {"model": "fake-model", "prompt": "hi", "stream": True}),
        ("/v1/responses", {"model": "fake-model", "input": "hi", "stream": True}),
    ],
    ids=["chat", "completions", "responses"],
)
def test_a_stream_the_engine_fails_records_why_it_stopped(serve_env, path, body):
    server, trace_dir = _server_with_trace(serve_env)
    # hold_after_steps keeps the first token's output unfinished, as the engine's own
    # output is when it fails mid-stream; the scripted error then comes in place of
    # the next one.
    server.generator.script(
        ScriptedOutput(
            steps=(ScriptedStep(text="he", token_ids=(1,)),),
            raise_after_steps=RequestError("engine said no"),
            hold_after_steps=True,
        )
    )
    client = serve_env.client(server.app)
    frames = []
    if path == "/v1/completions":
        # A completions stream answers the failure with an SSE error event.
        with client.stream("POST", path, json=body) as response:
            frames = [line for line in response.iter_lines() if line]
        assert any('"server_error"' in frame for frame in frames) and frames[-1] == "data: [DONE]"
    else:
        with pytest.raises(RequestError, match="engine said no"):
            with client.stream("POST", path, json=body) as response:
                for _ in response.iter_lines():
                    pass

    (response_line,) = _wait_for_lines(trace_dir, "responses", 1)
    assert response_line["termination"] == {
        "cause": "engine_error",
        "detail": "RequestError: engine said no",
    }, "an engine error is named with its message; see classify_stream_termination"
    promise = server.generator.generate_calls[-1].result
    assert response_line["engine_requests"][0]["request_id"] == promise.request_id
    if path == "/v1/completions":
        assert response_line["status"] == "completed", "the stream ended with its [DONE]"
    else:
        assert response_line["status"] == "error"
    if path == "/v1/responses":
        assert response_line["response_status"] == "failed", (
            "the guard's terminal event reaches the client and the trace's summary"
        )


@through_the_server
@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
def test_the_harmony_chat_path_records_its_engine_request(serve_env, stream):
    """gpt-oss chat goes through chat_harmony, a handler of its own."""
    reply = _harmony_reply()
    server, trace_dir = _server_with_trace(serve_env, model_type="gpt_oss")
    assert server.use_harmony
    server.generator.script(reply)
    client = serve_env.client(server.app)
    response = client.post(CHAT, json={**_CHAT_BODY, "stream": stream})
    assert response.status_code == 200, response.text

    (response_line,) = _wait_for_lines(trace_dir, "responses", 1)
    promise = server.generator.generate_calls[-1].result
    assert response_line["engine_requests"] == [
        {
            "request_id": promise.request_id,
            "prompt_tokens": prompt_token_digest(promise.prompt_token_ids),
        }
    ]
    assert response_line["engine_requests"][0]["prompt_tokens"]["count"] > 0
    assert response_line["served_model"] == "fake-model"
    assert response_line["status"] == "completed" and "termination" not in response_line


@through_the_server
def test_a_harmony_chat_stream_the_engine_fails_records_why_it_stopped(serve_env):
    pytest.importorskip("openai_harmony")
    server, trace_dir = _server_with_trace(serve_env, model_type="gpt_oss")
    server.generator.script(
        ScriptedOutput(
            steps=(ScriptedStep(text="<|channel|>final<|message|>he", token_ids=(1,)),),
            raise_after_steps=RequestError("engine said no"),
            hold_after_steps=True,
        )
    )
    client = serve_env.client(server.app)
    with pytest.raises(RequestError, match="engine said no"):
        with client.stream("POST", CHAT, json={**_CHAT_BODY, "stream": True}) as response:
            for _ in response.iter_lines():
                pass
    (response_line,) = _wait_for_lines(trace_dir, "responses", 1)
    assert response_line["termination"] == {
        "cause": "engine_error",
        "detail": "RequestError: engine said no",
    }
    assert response_line["status"] == "error"


@through_the_server
def test_a_handler_records_no_facts_when_the_trace_is_off(serve_env):
    server = serve_env.build_openai_server(AggConfig())
    server.generator.script(_two_tokens())
    client = serve_env.client(server.app)
    assert client.post(CHAT, json=_CHAT_BODY).status_code == 200
    assert server._request_trace is None
    assert not list(serve_env.tmp_dir.glob("**/*.jsonl"))
