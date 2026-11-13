# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace

import pytest

from tensorrt_llm.serve.scripts import backend_request_func as benchmark

pytestmark = pytest.mark.cpu_only


class _Content:
    def __init__(self, events):
        self.events = events

    async def iter_any(self):
        for event in self.events:
            yield ("data: " + json.dumps(event) + "\n\n").encode()
        yield b"data: [DONE]\n\n"


class _Response:
    status = 200

    def __init__(self, events):
        self.content = _Content(events)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _Session:
    def __init__(self, events):
        self.events = events

    def post(self, **kwargs):
        return _Response(self.events)


def _request(monkeypatch, timed_events):
    ticks = iter([100.0] + [timestamp for timestamp, _ in timed_events])
    monkeypatch.setattr(benchmark, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    request = benchmark.RequestFuncInput(
        prompt="hello",
        api_url="http://localhost/v1/chat/completions",
        prompt_len=1,
        output_len=2,
        model="example",
    )
    return asyncio.run(
        benchmark.async_request_openai_chat_completions(
            request, session=_Session([event for _, event in timed_events])
        )
    )


def _choice(delta, **kwargs):
    return {"choices": [{"delta": delta, **kwargs}]}


def test_metadata_events_do_not_count_as_generated_output(monkeypatch):
    out = _request(
        monkeypatch,
        [
            (101.0, _choice({"role": "assistant"})),
            (110.0, _choice({"content": "H"})),
            (110.5, _choice({"content": ""})),
            (111.0, _choice({"content": "i"})),
            (112.0, _choice({}, finish_reason="stop")),
            (113.0, {"choices": [], "usage": {"completion_tokens": 2}}),
        ],
    )
    assert out.success
    assert out.generated_text == "Hi"
    assert out.output_tokens == 2
    assert out.ttft == 10.0
    assert out.itl == [1.0]
    assert out.latency == 13.0


@pytest.mark.parametrize(
    "delta",
    [
        {"content": "hello"},
        {"reasoning": "thinking"},
        {"reasoning_content": "thinking"},
        {"tool_calls": [{"index": 0, "function": {"name": "weather"}}]},
        {"function_call": {"arguments": "{}"}},
        {"refusal": "cannot comply"},
        {"audio": {"id": "audio_0", "data": "UklGRg=="}},
    ],
)
def test_nonempty_output_types_set_ttft(monkeypatch, delta):
    out = _request(
        monkeypatch,
        [
            (101.0, _choice({"role": "assistant"})),
            (110.0, _choice(delta)),
            (111.0, _choice({}, finish_reason="stop")),
        ],
    )
    assert out.success
    assert out.ttft == 10.0
    assert out.itl == []
    assert out.latency == 11.0


def test_finish_event_can_also_carry_output(monkeypatch):
    out = _request(
        monkeypatch,
        [
            (103.0, _choice({"content": "A"})),
            (105.0, _choice({"content": "B"}, finish_reason="stop")),
        ],
    )
    assert out.success
    assert out.generated_text == "AB"
    assert out.ttft == 3.0
    assert out.itl == [2.0]
    assert out.latency == 5.0


def test_no_output_has_no_first_token_or_intervals(monkeypatch):
    out = _request(
        monkeypatch,
        [(101.0, _choice({"role": "assistant"})), (102.0, _choice({}, finish_reason="stop"))],
    )
    assert not out.success
    assert "output-bearing chunk" in out.error
    assert out.ttft == 0.0
    assert out.itl == []
    assert out.latency == 2.0


def test_no_output_with_usage_is_excluded_from_metrics(monkeypatch):
    """Usage alone must not make a stream without output count as successful.

    ``calculate_metrics`` only aggregates successful requests, so marking this
    response failed keeps its default TTFT of 0.0 out of the TTFT and TPOT
    statistics.
    """
    out = _request(
        monkeypatch,
        [
            (101.0, _choice({"role": "assistant"})),
            (150.0, _choice({}, finish_reason="stop")),
            (151.0, {"choices": [], "usage": {"completion_tokens": 5}}),
        ],
    )
    assert not out.success
    assert out.output_tokens == 5
    assert out.itl == []
    assert out.latency == 51.0
