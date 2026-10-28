# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Streaming post-processors read completion through ``finished``.

``finished`` lives on ``GenerationResultBase`` so that the
``DetokenizedGenerationResultBase`` records the post-processing workers hand
to the post-processors (``num_postprocess_workers > 0``) expose it exactly
like the ``RequestOutput`` objects the in-process path uses.
"""

import json

import pytest

from tensorrt_llm.executor import DetokenizedGenerationResultBase, GenerationResultBase
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve.openai_protocol import ResponsesRequest, StreamOptions
from tensorrt_llm.serve.postprocess_handlers import (
    ChatPostprocArgs,
    CompletionPostprocArgs,
    ResponsesAPIPostprocArgs,
    chat_stream_post_processor,
    completion_stream_post_processor,
    responses_api_streaming_post_processor,
)

pytestmark = pytest.mark.cpu_only

RESULT_CLASSES = [GenerationResultBase, DetokenizedGenerationResultBase]


def _make_result(result_cls):
    """A real result object mid-stream: one token out, not finished."""
    result = result_cls(1, SamplingParams())
    output = result._outputs[0]
    output.text = "x"
    output.token_ids = [11]
    return result


def _finish(result):
    """Advance the result by one token and mark the request finished."""
    output = result._outputs[0]
    output._last_text_len = len(output.text)
    output._last_token_ids_len = len(output.token_ids)
    output.text += "y"
    output.token_ids.append(12)
    output.finish_reason = "stop"
    result._done = True


def _payloads(chunks):
    payloads = []
    for chunk in chunks:
        for line in chunk.splitlines():
            if line.startswith("data: ") and not line.endswith("[DONE]"):
                payloads.append(json.loads(line[len("data: ") :]))
    return payloads


@pytest.mark.parametrize("result_cls", RESULT_CLASSES)
def test_finished_is_a_read_only_view_of_the_completion_flag(result_cls):
    result = result_cls(1, SamplingParams())
    assert result.finished is False
    result._done = True
    assert result.finished is True
    with pytest.raises(AttributeError):
        result.finished = False


@pytest.mark.parametrize("result_cls", RESULT_CLASSES)
def test_chat_stream_emits_final_usage_only_once_finished(result_cls):
    result = _make_result(result_cls)
    args = ChatPostprocArgs(
        role="assistant",
        model="test-model",
        num_prompt_tokens=3,
        stream_options=StreamOptions(include_usage=True),
    )

    mid_payloads = _payloads(chat_stream_post_processor(result, args))
    assert mid_payloads, "the mid-stream call must emit delta chunks"
    assert all(payload["choices"] for payload in mid_payloads), (
        "no usage-only chunk before the request finished"
    )

    _finish(result)
    final_payloads = _payloads(chat_stream_post_processor(result, args))
    usage_chunk = final_payloads[-1]
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"]["prompt_tokens"] == 3
    assert usage_chunk["usage"]["completion_tokens"] == 2
    assert usage_chunk["usage"]["total_tokens"] == 5


@pytest.mark.parametrize("result_cls", RESULT_CLASSES)
def test_completion_stream_emits_final_usage_only_once_finished(result_cls):
    result = _make_result(result_cls)
    args = CompletionPostprocArgs(
        model="test-model", num_prompt_tokens=3, stream_options=StreamOptions(include_usage=True)
    )

    mid_payloads = _payloads(completion_stream_post_processor(result, args))
    assert mid_payloads, "the mid-stream call must emit delta chunks"
    assert all(payload["choices"] for payload in mid_payloads), (
        "no usage-only chunk before the request finished"
    )

    _finish(result)
    final_payloads = _payloads(completion_stream_post_processor(result, args))
    usage_chunk = final_payloads[-1]
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"]["prompt_tokens"] == 3
    assert usage_chunk["usage"]["completion_tokens"] == 2
    assert usage_chunk["usage"]["total_tokens"] == 5


class _RecordingStreamingProcessor:
    """Stands in for the Responses streaming processor; records the calls."""

    def __init__(self):
        self.single_output_calls = []
        self.final_response_calls = []

    def process_single_output(self, rsp):
        self.single_output_calls.append(rsp)
        return ["data: step\n\n"]

    def get_final_response_non_store(self, rsp, num_prompt_tokens, tokenizer=None):
        self.final_response_calls.append((rsp, num_prompt_tokens))
        return "data: final\n\n"


@pytest.mark.parametrize("result_cls", RESULT_CLASSES)
def test_responses_stream_requests_final_response_only_once_finished(result_cls):
    processor = _RecordingStreamingProcessor()
    args = ResponsesAPIPostprocArgs(
        model="test-model",
        request=ResponsesRequest(input="hi", model="test-model"),
        sampling_params=SamplingParams(),
        use_harmony=False,
        streaming_processor=processor,
        num_prompt_tokens=3,
    )
    result = _make_result(result_cls)

    outputs = responses_api_streaming_post_processor(result, args)
    assert outputs == ["data: step\n\n"]
    assert processor.final_response_calls == []

    _finish(result)
    outputs = responses_api_streaming_post_processor(result, args)
    assert outputs == ["data: step\n\n", "data: final\n\n"]
    assert len(processor.final_response_calls) == 1
    assert processor.final_response_calls[0] == (result, 3)
