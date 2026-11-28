# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from contextlib import contextmanager

import pytest
from test_common.serve.app_factory import AggConfig, isolated_serve_env

from tensorrt_llm._utils import AdjustedSteadyClock
from tensorrt_llm.executor import GenerationResultBase
from tensorrt_llm.llmapi.llm import BaseLLM
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.openai_protocol import CompletionRequest, StreamOptions
from tensorrt_llm.serve.openai_server import OpenAIServer, merge_generators
from tensorrt_llm.serve.postprocess_handlers import ChatPostprocArgs, chat_stream_post_processor

pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]


def _stream_payloads_from_chunks(chunks):
    payloads = []
    for chunk in chunks:
        if isinstance(chunk, bytes):
            chunk = chunk.decode()
        for line in chunk.splitlines():
            if line.startswith("data: "):
                data = line[len("data: ") :].strip()
                if data != "[DONE]":
                    payloads.append(json.loads(data))
    return payloads


def test_chat_stream_post_processor_reuses_stream_metadata() -> None:
    result = GenerationResultBase(123, SamplingParams())
    output = result._outputs[0]
    output.text = "x"
    output.token_ids = [1]

    args = ChatPostprocArgs(role="assistant", model="test-model")
    chunks = chat_stream_post_processor(result, args)

    output._last_text_len = len(output.text)
    output._last_token_ids_len = len(output.token_ids)
    output.text = "xy"
    output.token_ids.append(2)
    chunks += chat_stream_post_processor(result, args)

    payloads = _stream_payloads_from_chunks(chunks)

    assert {payload["id"] for payload in payloads} == {"chatcmpl-123"}
    assert len({payload["created"] for payload in payloads}) == 1
    assert payloads[0]["choices"][0]["delta"]["role"] == "assistant"
    assert payloads[-1]["choices"][0]["delta"]["content"] == "y"


def test_chat_stream_post_processor_usage_applies_prompt_token_offset() -> None:
    result = GenerationResultBase(123, SamplingParams())
    output = result._outputs[0]
    output.text = "x"
    output.token_ids = [1]
    output.finish_reason = "stop"
    result._done = True

    args = ChatPostprocArgs(
        role="assistant",
        model="test-model",
        num_prompt_tokens=5,
        num_prompt_tokens_offset=3,
        stream_options=StreamOptions(include_usage=True),
    )
    payloads = _stream_payloads_from_chunks(chat_stream_post_processor(result, args))

    final_chunk = payloads[-1]
    assert final_chunk["choices"] == []
    assert final_chunk["usage"]["prompt_tokens"] == 2
    assert final_chunk["usage"]["completion_tokens"] == 1
    assert final_chunk["usage"]["total_tokens"] == 3


def test_chat_stream_post_processor_usage_requires_prompt_token_count() -> None:
    # Usage arithmetic needs a concrete count: a missing one must surface as a
    # clear error, never as None leaking into UsageInfo or a TypeError.
    result = GenerationResultBase(123, SamplingParams())
    args = ChatPostprocArgs(
        role="assistant", model="test-model", stream_options=StreamOptions(include_usage=True)
    )

    with pytest.raises(ValueError, match="num_prompt_tokens"):
        chat_stream_post_processor(result, args)


class _FakeCompletionGeneratorArgs:
    backend = "pytorch"
    gather_generation_logits = False
    num_postprocess_workers = 0
    return_perf_metrics = False
    reasoning_parser = None


class _FakeModelConfig:
    vocab_size = 32000


class _FakeCompletionStreamResult(GenerationResultBase):
    @property
    def finished(self):
        return self._done

    @property
    def request_id(self):
        return self.id


class _FakeCompletionPromise:
    def __init__(self, result, prompt_token_ids):
        self._result = result
        self._yielded = False
        self.prompt_token_ids = prompt_token_ids
        self.aborted = False

    @property
    def finished(self):
        return True

    @property
    def request_id(self):
        return self._result.id

    def abort(self):
        self.aborted = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._yielded:
            raise StopAsyncIteration
        self._yielded = True
        return self._result


class _FakeCompletionGenerator:
    def __init__(self):
        self.args = _FakeCompletionGeneratorArgs()
        self.postproc_args = []

    def input_processor(self, prompt, _sampling_params):
        token_id = ord(prompt["prompt"])
        return [token_id], {}

    def generate_async(self, inputs, sampling_params, _postproc_params, streaming, **_kwargs):
        assert streaming
        result_id = 100 + len(self.postproc_args)
        result = _FakeCompletionStreamResult(result_id, sampling_params)
        result._done = True

        output = result._outputs[0]
        output.text = f"text-{result_id}"
        output.token_ids = [result_id]
        output.finish_reason = "stop"

        self.postproc_args.append(_postproc_params.postproc_args)
        return _FakeCompletionPromise(result, inputs["prompt_token_ids"])


class _FakeRawRequestState:
    pass


class _FakeRawRequest:
    def __init__(self):
        self.headers = {}
        self.state = _FakeRawRequestState()
        self.client = "test-client"

    async def is_disconnected(self):
        return True


@contextmanager
def _completion_server(tmp_path, generator):
    # Build the server through the shared factory instead of
    # object.__new__ (the ratchet in test_no_new_object_new_sites only
    # lets the baseline sites shrink), then swap in the fake generator
    # the test drives and pin the attributes the request path reads.
    with isolated_serve_env(tmp_path) as env:
        server = env.build_openai_server(AggConfig(model_name="test-model"))
        server.generator = generator
        server.model_config = _FakeModelConfig()
        server.tokenizer = None
        server.metrics_collector = None
        server._collect_perf_metrics = False
        server._input_proc_executor = None
        server._adjusted_steady_clock = AdjustedSteadyClock()
        yield server


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "checkpoint_io_policy",
    [
        None,
        {
            "requested": "auto",
            "selected": "rank_striped_read_ahead",
            "activated": True,
            "effective": "native",
            "fallback_reason": "reader failed after activation",
        },
    ],
)
def test_startup_metrics_by_server_info(
    checkpoint_io_policy: dict[str, str | bool | None] | None,
) -> None:
    class FakeExecutor:
        def __init__(self):
            self.calls = 0

        def get_startup_metrics(self):
            self.calls += 1  # keep track of # calls to ensure the cache in the server works
            if self.calls == 1:
                return None
            return {
                "model_loader": {
                    "total_model_loading_seconds": 1.5,
                    "checkpoint_io_policy": checkpoint_io_policy,
                }
            }

    executor = FakeExecutor()
    generator = object.__new__(BaseLLM)
    generator._executor = executor
    generator._startup_metrics = None
    generator._disaggregated_params = {}

    server = object.__new__(OpenAIServer)
    server.generator = generator
    first_response = asyncio.run(server.get_server_info())
    second_response = asyncio.run(server.get_server_info())
    third_response = asyncio.run(server.get_server_info())

    first, second, third = (
        json.loads(response.body) for response in (first_response, second_response, third_response)
    )
    # The render fingerprint is covered by its own tests; here it only has to be
    # present and the same on every call.
    fingerprints = [body.pop("render_fingerprint") for body in (first, second, third)]
    assert fingerprints[0]["digest"]
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]

    assert first == {
        "disaggregated_params": {},
        "startup_metrics": {},
    }
    expected_success = {
        "disaggregated_params": {},
        "startup_metrics": {
            "model_loader": {
                "total_model_loading_seconds": 1.5,
                "checkpoint_io_policy": checkpoint_io_policy,
            }
        },
    }
    assert second == expected_success
    assert third == expected_success
    assert executor.calls == 2


def test_openai_completion_list_prompt_stream_reuses_stream_metadata() -> None:
    async def run_request():
        generator = _FakeCompletionGenerator()
        server = object.__new__(OpenAIServer)
        server.generator = generator
        server.model = "test-model"
        server.model_config = _FakeModelConfig()
        server.tokenizer = None
        server.metrics_collector = None
        server._collect_perf_metrics = False
        server._input_proc_executor = None
        server._adjusted_steady_clock = AdjustedSteadyClock()

        request = CompletionRequest(model="test-model", prompt=["A", "B"], stream=True)
        response = await server.openai_completion(request, _FakeRawRequest())
        chunks = [chunk async for chunk in response.body_iterator]
        return generator, _stream_payloads_from_chunks(chunks)

    generator, payloads = asyncio.run(run_request())

    ids = {payload["id"] for payload in payloads}
    created = {payload["created"] for payload in payloads}
    choice_indexes = {payload["choices"][0]["index"] for payload in payloads if payload["choices"]}

    assert len(payloads) == 2
    assert len(ids) == 1
    assert ids.isdisjoint({"cmpl-100", "cmpl-101"})
    assert len(created) == 1
    assert choice_indexes == {0, 1}
    assert {args.stream_response_id for args in generator.postproc_args} == ids
    assert {args.stream_created for args in generator.postproc_args} == created


def test_openai_completion_multi_prompt_stream_sums_usage(tmp_path) -> None:
    # A streaming /v1/completions request with several prompts and
    # include_usage must end with a single usage chunk for the whole
    # request, not one per prompt (issue #22740). The fake generator gives
    # each prompt one prompt token and one generated token.
    async def run_request():
        generator = _FakeCompletionGenerator()
        with _completion_server(tmp_path, generator) as server:
            request = CompletionRequest(
                model="test-model",
                prompt=["A", "B"],
                stream=True,
                stream_options=StreamOptions(include_usage=True),
            )
            response = await server.openai_completion(request, _FakeRawRequest())
            chunks = [chunk async for chunk in response.body_iterator]
            return _stream_payloads_from_chunks(chunks)

    payloads = asyncio.run(run_request())

    # Both prompts still emit their choice chunk, indexes 0 and 1 exactly
    # once each, before the aggregate.
    choice_indexes = [
        payload["choices"][0]["index"]
        for payload in payloads
        if payload["choices"]
    ]
    assert sorted(choice_indexes) == [0, 1]

    usage_chunks = [payload for payload in payloads if payload["choices"] == []]
    assert len(usage_chunks) == 1
    assert usage_chunks[0] is payloads[-1]
    usage = usage_chunks[0]["usage"]
    assert usage["prompt_tokens"] == 2
    assert usage["completion_tokens"] == 2
    assert usage["total_tokens"] == 4
    assert usage["prompt_tokens_details"]["cached_tokens"] == 0
    assert {payload["id"] for payload in payloads} == {usage_chunks[0]["id"]}


def test_openai_completion_multi_prompt_stream_continuous_usage(
        tmp_path) -> None:
    # With continuous_usage_stats, every choice chunk carries its own
    # prompt's usage. Those chunks are not final usage chunks: the merger
    # must pass them through untouched, and only the trailing usage-only
    # chunk holds the request total.
    async def run_request():
        generator = _FakeCompletionGenerator()
        with _completion_server(tmp_path, generator) as server:
            request = CompletionRequest(
                model="test-model",
                prompt=["A", "B"],
                stream=True,
                stream_options=StreamOptions(
                    include_usage=True, continuous_usage_stats=True),
            )
            response = await server.openai_completion(request, _FakeRawRequest())
            chunks = [chunk async for chunk in response.body_iterator]
            return _stream_payloads_from_chunks(chunks)

    payloads = asyncio.run(run_request())

    choice_payloads = [payload for payload in payloads if payload["choices"]]
    assert sorted(
        payload["choices"][0]["index"] for payload in choice_payloads) == [0, 1]
    for payload in choice_payloads:
        # Each prompt generated one token from one prompt token, and its
        # choice chunk keeps exactly that per-prompt usage.
        assert payload["usage"]["prompt_tokens"] == 1
        assert payload["usage"]["completion_tokens"] == 1
        assert payload["usage"]["total_tokens"] == 2

    usage_chunks = [payload for payload in payloads if payload["choices"] == []]
    assert len(usage_chunks) == 1
    assert usage_chunks[0] is payloads[-1]
    usage = usage_chunks[0]["usage"]
    assert usage["prompt_tokens"] == 2
    assert usage["completion_tokens"] == 2
    assert usage["total_tokens"] == 4


async def _fake_prompt_usage_stream(prompt_tokens: int,
                                    completion_tokens: int,
                                    cached_tokens: int = 0):
    chunk = {
        "id": "cmpl-1",
        "object": "text_completion",
        "created": 1,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "text": "x",
            "finish_reason": "stop",
        }],
        "usage": None,
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    chunk["choices"] = []
    chunk["usage"] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {
            "cached_tokens": cached_tokens
        },
    }
    yield f"data: {json.dumps(chunk)}\n\n"


def test_merge_generators_sums_per_prompt_usage_chunks() -> None:
    async def collect():
        merged = merge_generators([
            _fake_prompt_usage_stream(3, 2, cached_tokens=1),
            _fake_prompt_usage_stream(5, 4, cached_tokens=2),
        ])
        return [chunk async for chunk in merged]

    payloads = _stream_payloads_from_chunks(asyncio.run(collect()))

    usage_chunks = [payload for payload in payloads if payload["choices"] == []]
    assert len(usage_chunks) == 1
    assert usage_chunks[0] is payloads[-1]
    assert usage_chunks[0]["usage"] == {
        "prompt_tokens": 8,
        "completion_tokens": 6,
        "total_tokens": 14,
        "prompt_tokens_details": {
            "cached_tokens": 3
        },
    }


def test_merge_generators_forwards_malformed_usage_chunks() -> None:
    # A usage chunk whose fields are malformed must be forwarded
    # untouched: it must not abort the merged stream, and it must not
    # change the summed total. One frame has prompt_tokens_details as a
    # list instead of an object, the other a string token count.
    def usage_frame(usage):
        return "data: %s\n\n" % json.dumps({
            "id": "cmpl-1",
            "object": "text_completion",
            "created": 1,
            "model": "test-model",
            "choices": [],
            "usage": usage,
        })

    bad_details = usage_frame({
        "prompt_tokens": 100,
        "completion_tokens": 100,
        "total_tokens": 200,
        "prompt_tokens_details": [1],
    })
    bad_count = usage_frame({
        "prompt_tokens": "3",
        "completion_tokens": 2,
        "total_tokens": 5,
        "prompt_tokens_details": {
            "cached_tokens": 0
        },
    })

    async def malformed_stream():
        yield bad_details
        yield bad_count

    async def collect():
        merged = merge_generators([
            malformed_stream(),
            _fake_prompt_usage_stream(3, 2),
            _fake_prompt_usage_stream(5, 4),
        ])
        return [chunk async for chunk in merged]

    chunks = asyncio.run(collect())

    assert bad_details in chunks
    assert bad_count in chunks
    payloads = _stream_payloads_from_chunks(chunks)
    totals = sorted(
        payload["usage"]["total_tokens"]
        for payload in payloads
        if payload["choices"] == [] and isinstance(
            payload["usage"]["total_tokens"], int))
    # The two malformed frames keep their own totals, and the only
    # aggregate is the sum of the two valid frames.
    assert totals == [5, 14, 200]


def test_merge_generators_skips_aggregate_when_a_prompt_fails() -> None:
    # One prompt completes with usage, the other fails: its generator
    # yields an error chunk and [DONE], like completion_generator does.
    # The error goes out unchanged and no aggregate follows it, since a
    # summed total here would be partial and would land after [DONE].
    error_chunk = ('data: {"error": {"message": "boom", "type": '
                   '"server_error", "code": null, "param": null}}\n\n')

    async def failing_stream():
        yield error_chunk
        yield "data: [DONE]\n\n"

    async def collect():
        merged = merge_generators([
            _fake_prompt_usage_stream(3, 2),
            failing_stream(),
        ])
        return [chunk async for chunk in merged]

    chunks = asyncio.run(collect())

    assert error_chunk in chunks
    done_index = chunks.index("data: [DONE]\n\n")
    assert _stream_payloads_from_chunks(chunks[done_index + 1:]) == []
    usage_payloads = [
        payload for payload in _stream_payloads_from_chunks(chunks)
        if payload.get("choices") == [] and payload.get("usage")
    ]
    assert usage_payloads == []
