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
"""A silent Responses stream repeats its opening ``response.in_progress``.

A tool call is sent only once it is complete. While a long one is generated
the client receives nothing, and agent clients drop a stream that stays silent
for long (seen in production agent traffic: a client-side idle timeout of
about two minutes cut every long tool call). The keepalive is the egress loop
of the server (``keep_responses_stream_alive``), so these tests feed it the
frames the streaming processor makes, with the pauses a generation has, and
read what a client would.

The interval is patched where the stream reads it. A keepalive can never come
early, so the gaps between frames are asserted exactly; counts are asserted
with the slack a loaded machine needs.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from tensorrt_llm.serve._base import responses_stream
from tensorrt_llm.serve._base.responses_stream import keep_responses_stream_alive
from tensorrt_llm.serve.openai_protocol import ResponsesRequest
from tensorrt_llm.serve.responses_utils import ResponsesStreamingProcessor

pytestmark = pytest.mark.cpu_only

# Short enough to keep the tests quick, long enough that the event loop's own
# latency does not look like a silence.
_INTERVAL = 0.1
# A keepalive fires on a loop timer, which never runs early; the slack covers
# the clock's resolution only.
_EARLY_SLACK = 0.005


def _event_type(frame):
    return frame.split("\n", 1)[0][len("event: ") :]


def _event_data(frame):
    return json.loads(frame.split("data: ", 1)[1])


def _without_number(frame):
    data = _event_data(frame)
    del data["sequence_number"]
    return _event_type(frame), data


def _processor():
    request = ResponsesRequest(model="test-model", input="hi", stream=True)
    return ResponsesStreamingProcessor(
        request=request,
        sampling_params=request.to_sampling_params(),
        model_name="test-model",
        use_harmony=False,
    )


def _step(text, diff, done):
    """One generation step, as the streaming post-processor receives it."""
    output = SimpleNamespace(
        index=0,
        text=text,
        text_diff=diff,
        finish_reason="stop" if done else None,
        token_ids=[1] * len(text),
        disaggregated_params=None,
    )
    return SimpleNamespace(
        outputs=[output], _done=done, prompt_token_ids=[1], cached_tokens=0, finished=done
    )


async def _generation(steps, produced, opening=None):
    """The frames a generation streams: the opening pair, then each step's events.

    ``steps`` are (pause, text) pairs; the last one finishes the response. A
    step that adds no text sends no event, as while a tool call is held.
    ``opening`` replaces the processor's opening pair. Every frame is also
    appended to ``produced``.
    """
    processor = _processor()
    for frame in processor.get_initial_responses() if opening is None else opening:
        produced.append(frame)
        yield frame
    text = ""
    for i, (pause, diff) in enumerate(steps):
        await asyncio.sleep(pause)
        text += diff
        result = _step(text, diff, done=i == len(steps) - 1)
        frames = processor.process_single_output(result)
        if result._done:
            frames.append(processor.get_final_response_non_store(result))
        for frame in frames:
            produced.append(frame)
            yield frame


async def _send(monkeypatch, steps, interval=_INTERVAL, opening=None):
    """(produced, sent): the frames made, and (arrival time, frame) of what a client gets."""
    monkeypatch.setattr(responses_stream, "RESPONSES_STREAM_KEEPALIVE_INTERVAL_S", interval)
    produced, sent = [], []
    loop = asyncio.get_running_loop()

    async def drain():
        async for frame in keep_responses_stream_alive(_generation(steps, produced, opening)):
            sent.append((loop.time(), frame))
            assert len(sent) < 500, "the stream does not end"

    await asyncio.wait_for(drain(), timeout=10)
    return produced, sent


def _repeats(sent):
    """Positions of the response.in_progress frames after the opening one."""
    positions = [i for i, (_, f) in enumerate(sent) if _event_type(f) == "response.in_progress"]
    return positions[1:]


def _assert_never_early(sent, interval):
    """Each repeated in_progress comes a whole interval after the frame before it."""
    for position in _repeats(sent):
        gap = sent[position][0] - sent[position - 1][0]
        assert gap >= interval - _EARLY_SLACK, (position, gap)


def _first(sent, event_type):
    return [_event_type(f) for _, f in sent].index(event_type)


@pytest.mark.asyncio
async def test_the_repeated_frame_is_the_opening_in_progress(monkeypatch):
    produced, sent = await _send(monkeypatch, [(0.25, "Hello")])

    repeats = _repeats(sent)
    assert repeats
    opening = _without_number(produced[1])
    assert produced[1].startswith("event: response.in_progress\n")
    for position in repeats:
        assert _without_number(sent[position][1]) == opening
        assert _event_data(sent[position][1])["sequence_number"] != 1


@pytest.mark.asyncio
async def test_nothing_before_the_interval_then_the_frame(monkeypatch):
    """A pause shorter than the interval sends nothing; a longer one sends the frame."""
    _, sent = await _send(monkeypatch, [(0.03, "Hello"), (0.25, " world")])

    repeats = _repeats(sent)
    first_delta = _first(sent, "response.output_text.delta")
    assert [i for i in repeats if i < first_delta] == []
    assert 1 <= len([i for i in repeats if i > first_delta]) <= 2
    _assert_never_early(sent, _INTERVAL)


@pytest.mark.asyncio
async def test_each_silent_interval_gets_one_keepalive(monkeypatch):
    """A silence of several intervals draws one frame per interval, each a whole interval apart."""
    _, sent = await _send(monkeypatch, [(0.55, "Hello")])

    repeats = _repeats(sent)
    # Five intervals fit the silence; a loaded loop may be late with the last ones.
    assert 3 <= len(repeats) <= 5, repeats
    assert repeats[0] == 2
    _assert_never_early(sent, _INTERVAL)


@pytest.mark.asyncio
async def test_frames_reset_the_silence(monkeypatch):
    """A stream longer than the interval that is never silent for it gets no keepalive."""
    steps = [(0.03, f"w{i} ") for i in range(8)] + [(0.25, "done")]
    _, sent = await _send(monkeypatch, steps)

    repeats = _repeats(sent)
    deltas = [i for i, (_, f) in enumerate(sent) if _event_type(f) == "response.output_text.delta"]
    assert [i for i in repeats if i < deltas[-2]] == []
    assert [i for i in repeats if i > deltas[-2]]
    _assert_never_early(sent, _INTERVAL)


@pytest.mark.asyncio
async def test_a_stream_that_opens_without_in_progress_is_left_alone(monkeypatch):
    """With no in_progress to repeat there is no keepalive, however long the silence."""
    created = _processor().get_initial_responses()[0]
    produced, sent = await _send(monkeypatch, [(0.25, "Hello")], opening=[created])

    assert [frame for _, frame in sent] == produced
    assert "response.in_progress" not in [_event_type(f) for f in produced]


@pytest.mark.asyncio
async def test_the_egress_stream_stays_numbered_and_says_nothing_new(monkeypatch):
    """A delta, a long silence while a call is generated, then the call's events.

    Sequence numbers run from 0 without a gap, keepalives included; a keepalive
    carries the opening response and nothing else; every other frame leaves as
    it was made, except for its place in the stream.
    """
    steps = [(0, "Let me check."), (0.25, ""), (0, " Done.")]
    produced, sent = await _send(monkeypatch, steps)

    frames = [frame for _, frame in sent]
    assert [_event_data(f)["sequence_number"] for f in frames] == list(range(len(frames)))
    repeats = _repeats(sent)
    assert len(repeats) >= 2
    opening = _without_number(produced[1])
    for position in repeats:
        assert _without_number(frames[position]) == opening
    others = [frame for i, frame in enumerate(frames) if i not in repeats]
    assert [_without_number(f) for f in others] == [_without_number(f) for f in produced]
    assert _event_type(frames[-1]) == "response.completed"


# ---------------------------------------------------------------------------
# The route: the server's stream, from a request to the frames the client gets
# ---------------------------------------------------------------------------


async def _post_responses(app, body, on_send):
    """POST ``body`` to ``/v1/responses`` and return the ASGI messages sent back.

    ``on_send`` sees each message as it is sent, while the request runs. The
    client stays connected until the response is complete, then disconnects.
    """
    requested = False
    completed = asyncio.Event()

    async def receive():
        nonlocal requested
        if not requested:
            requested = True
            return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}
        await completed.wait()
        return {"type": "http.disconnect"}

    sent = []

    async def send(message):
        sent.append(message)
        if message["type"] == "http.response.body" and not message.get("more_body", False):
            completed.set()
        on_send(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/responses",
        "raw_path": b"/v1/responses",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 40000),
        "server": ("127.0.0.1", 8000),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=10)
    return sent


@pytest.mark.asyncio
async def test_the_responses_route_keeps_a_silent_stream_alive(monkeypatch, tmp_path):
    """The served stream: one delta, then the engine holds the request as a long call would.

    The scripted engine keeps the request open until released, which the test
    does once two keepalives have reached the client; released, it ends the
    generation there. The client sees the opening, the delta, keepalives that
    repeat the opening, one sequence of numbers, and the end of the stream.
    """
    from test_common.serve.app_factory import AggConfig, isolated_serve_env
    from test_common.serve.fakes import ScriptedOutput, ScriptedStep

    monkeypatch.setattr(responses_stream, "RESPONSES_STREAM_KEEPALIVE_INTERVAL_S", 0.05)
    with isolated_serve_env(tmp_path) as env:
        server = env.build_openai_server(AggConfig())
        server.generator.script(
            ScriptedOutput(
                steps=(ScriptedStep(text="Hello", token_ids=(6,)),), hold_after_steps=True
            )
        )
        in_progress = 0

        def release_on_the_second_keepalive(message):
            nonlocal in_progress
            if message.get("body", b"").startswith(b"event: response.in_progress\n"):
                in_progress += 1
                if in_progress == 3:
                    server.generator.generate_calls[-1].result.release()

        body = {"model": "fake-model", "input": "hi", "stream": True, "store": False}
        messages = await _post_responses(server.app, body, release_on_the_second_keepalive)

    start = next(m for m in messages if m["type"] == "http.response.start")
    assert start["status"] == 200
    assert dict(start["headers"])[b"content-type"].startswith(b"text/event-stream")
    frames = [
        m["body"].decode() for m in messages if m["type"] == "http.response.body" and m["body"]
    ]
    assert messages[-1].get("more_body") is False

    types = [_event_type(f) for f in frames]
    assert types[:2] == ["response.created", "response.in_progress"]
    repeats = [i for i, t in enumerate(types) if t == "response.in_progress"][1:]
    assert len(repeats) >= 2
    assert types.index("response.output_text.delta") < repeats[0]
    opening = _without_number(frames[1])
    assert all(_without_number(frames[i]) == opening for i in repeats)
    assert [_event_data(f)["sequence_number"] for f in frames] == list(range(len(frames)))
