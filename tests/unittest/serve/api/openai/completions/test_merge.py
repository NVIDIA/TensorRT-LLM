# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The functions lifted out of ``CompletionServing.openai_completion``, one by one.

As closures they read the serving object and the request from the enclosing
method and could only be exercised through a whole request; as module-level
functions with keyword-only parameters each one runs here with a small
``serving`` stand-in and scripted promises.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from tensorrt_llm.serve.api.openai.completions import _merge
from tensorrt_llm.serve.openai_protocol import (
    CompletionResponse,
    CompletionResponseChoice,
    PromptTokensDetails,
    UsageInfo,
)

pytestmark = pytest.mark.cpu_only


def _response(index, prompt_tokens, completion_tokens, cached_tokens, prompt_token_ids=None):
    return CompletionResponse(
        model="ignored",
        choices=[CompletionResponseChoice(index=index, text=f"text-{index}", finish_reason="stop")],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=cached_tokens),
        ),
        prompt_token_ids=prompt_token_ids,
    )


async def _resolved(value):
    return value


async def _items(*items, events=None, label="item"):
    for i, item in enumerate(items):
        if events is not None:
            events.append(f"{label}{i}")
        yield item


class _Recorder:
    """A ``serving`` stand-in: records what the lifted functions ask of it."""

    def __init__(self, pp_result=None):
        self.model = "m"
        self.pp_result = pp_result
        self.applied = []
        self.extracted = []
        self._postproc = SimpleNamespace(apply=self._apply)
        self._adjusted_steady_clock = SimpleNamespace(now=lambda: 123.5)

    def _apply(self, output, params):
        self.applied.append((output, params))
        return self.pp_result

    async def _extract_metrics(self, output, raw_request):
        self.extracted.append((output, raw_request))


def test_merge_completion_responses_sums_usage_and_prompt_ids():
    first = _response(
        0, prompt_tokens=3, completion_tokens=2, cached_tokens=1, prompt_token_ids=[1, 2, 3]
    )
    second = _response(
        1, prompt_tokens=4, completion_tokens=5, cached_tokens=2, prompt_token_ids=[4, 5]
    )

    merged = _merge.merge_completion_responses([first, second], serving=SimpleNamespace(model="m"))

    assert merged.model == "m"
    assert [choice.index for choice in merged.choices] == [0, 1]
    assert merged.usage.prompt_tokens == 7
    assert merged.usage.completion_tokens == 7
    assert merged.usage.total_tokens == 14
    assert merged.usage.prompt_tokens_details.cached_tokens == 3
    assert merged.prompt_token_ids == [[1, 2, 3], [4, 5]]


def test_merge_completion_responses_skips_absent_prompt_ids():
    merged = _merge.merge_completion_responses(
        [_response(0, 1, 1, 0), _response(1, 1, 1, 0, prompt_token_ids=[9])],
        serving=SimpleNamespace(model="m"),
    )
    assert merged.prompt_token_ids == [[9]]


def test_merge_generators_yields_everything_and_finishes():
    async def run():
        merged = _merge.merge_generators([_items("a", "b", "c"), _items("d")])
        items = [item async for item in merged]
        # merge_generators awaits its producer tasks before finishing, so none
        # is left behind in the loop.
        pending = {task for task in asyncio.all_tasks() if task is not asyncio.current_task()}
        return items, pending

    items, pending = asyncio.run(run())
    assert sorted(items) == ["a", "b", "c", "d"]
    assert pending == set()


def test_completion_generator_streams_post_processed_chunks_and_records_metrics():
    serving = _Recorder(pp_result=["data: chunk\n\n"])
    raw_request = object()

    async def run():
        outputs = ["out-0", "out-1"]
        return [
            chunk
            async for chunk in _merge.completion_generator(
                _items(*outputs), "params", serving=serving, raw_request=raw_request
            )
        ]

    chunks = asyncio.run(run())
    assert chunks == ["data: chunk\n\n", "data: chunk\n\n"]
    assert [params for _, params in serving.applied] == ["params", "params"]
    # Metrics are recorded once, from the last output, with the raw request.
    assert serving.extracted == [("out-1", raw_request)]


def test_completion_generator_turns_an_error_into_an_sse_frame():
    serving = _Recorder()

    async def failing():
        raise RuntimeError("engine said no")
        yield  # pragma: no cover - makes this an async generator

    async def run():
        return [
            chunk
            async for chunk in _merge.completion_generator(
                failing(), None, serving=serving, raw_request=None
            )
        ]

    chunks = asyncio.run(run())  # raises nothing: the stream ends cleanly
    assert len(chunks) == 2
    assert chunks[0].startswith("data: ") and chunks[0].endswith("\n\n")
    error = json.loads(chunks[0][len("data: ") : -2])["error"]
    assert error == {
        "message": "engine said no",
        "type": "server_error",
        "code": None,
        "param": None,
    }
    assert chunks[1] == "data: [DONE]\n\n"
    assert serving.extracted == []


def test_generator_wrapper_stamps_first_token_time():
    events = []
    serving = _Recorder()
    serving._adjusted_steady_clock = SimpleNamespace(now=lambda: events.append("stamp") or 123.5)
    raw_request = SimpleNamespace(state=SimpleNamespace())

    async def run():
        return [
            chunk
            async for chunk in _merge.generator_wrapper(
                _items("first", "second", events=events), serving=serving, raw_request=raw_request
            )
        ]

    chunks = asyncio.run(run())
    assert chunks == ["first", "second", "data: [DONE]\n\n"]
    assert raw_request.state.server_first_token_time == 123.5
    # The stamp is taken once the first item exists, before the second is pulled.
    assert events == ["item0", "stamp", "item1"]


@pytest.mark.parametrize(
    "request_type,carries_prompt_ids",
    [("context_only", True), ("generation_only", False), (None, False)],
)
def test_completion_response_context_only_carries_prompt_ids(request_type, carries_prompt_ids):
    pp_result = SimpleNamespace(prompt_token_ids=None)
    serving = _Recorder(pp_result=pp_result)
    response = SimpleNamespace(error=None, prompt_token_ids=[7, 8, 9])
    raw_request = object()
    disaggregated_params = (
        SimpleNamespace(request_type=request_type) if request_type is not None else None
    )

    result = asyncio.run(
        _merge.completion_response(
            _resolved(response),
            "params",
            serving=serving,
            disaggregated_params=disaggregated_params,
            raw_request=raw_request,
        )
    )

    assert result is pp_result
    assert serving.applied == [(response, "params")]
    assert result.prompt_token_ids == ([7, 8, 9] if carries_prompt_ids else None)
    assert serving.extracted == [(response, raw_request)]


def test_completion_response_raises_on_error():
    serving = _Recorder(pp_result=SimpleNamespace())

    with pytest.raises(RuntimeError, match="Generation failed: engine said no"):
        asyncio.run(
            _merge.completion_response(
                _resolved(SimpleNamespace(error="engine said no")),
                None,
                serving=serving,
                disaggregated_params=None,
                raw_request=None,
            )
        )
    assert serving.applied == [] and serving.extracted == []
