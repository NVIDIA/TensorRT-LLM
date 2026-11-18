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
"""Offline tests for the Responses API usage block.

The usage block was never populated, so every response reported no token
consumption at all. Agentic clients use it to track how much of the context
window a conversation has spent and to decide when to compact; without it a
long session keeps appending turns until it overflows the context window.
"""

import base64
import json
import struct
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tensorrt_llm.executor.result import CompletionOutput
from tensorrt_llm.inputs.tokenization import tokenize_prompt
from tensorrt_llm.serve.openai_disagg_service import _ctx_usage_info
from tensorrt_llm.serve.openai_protocol import (
    PromptTokensDetails,
    ResponsesRequest,
    ResponseUsage,
    StreamOptions,
    UsageInfo,
)
from tensorrt_llm.serve.postprocess_handlers import (
    ChatPostprocArgs,
    ResponsesAPIPostprocArgs,
    chat_response_post_processor,
    chat_stream_post_processor,
    responses_api_post_processor,
    responses_api_streaming_post_processor,
)
from tensorrt_llm.serve.responses_utils import (
    ResponsesStreamingProcessor,
    _count_reasoning_tokens,
    _create_output_content,
    _create_usage,
)

from .openai_server_stub import make_stub_openai_server

# The CPU-* CI stages run pytest with -m 'cpu_only'. Without this marker every
# test in the file is deselected, which pytest reports as exit code 5 and the
# stage reports as a failure.
pytestmark = pytest.mark.cpu_only


def _generation(prompt_tokens=7, completion_tokens=3, cached_tokens=0):
    return SimpleNamespace(
        prompt_token_ids=list(range(prompt_tokens)),
        outputs=[SimpleNamespace(token_ids=list(range(completion_tokens)))],
        cached_tokens=cached_tokens,
    )


def test_usage_counts_prompt_and_generated_tokens():
    usage = _create_usage(_generation(prompt_tokens=7, completion_tokens=3))
    assert usage.input_tokens == 7
    assert usage.output_tokens == 3
    assert usage.total_tokens == 10


def test_usage_reports_cached_tokens():
    usage = _create_usage(_generation(cached_tokens=5))
    assert usage.input_tokens_details.cached_tokens == 5


def test_usage_defaults_cached_tokens_when_backend_omits_them():
    """cached_tokens is absent on some result types; it must not raise."""
    result = _generation()
    del result.cached_tokens
    assert _create_usage(result).input_tokens_details.cached_tokens == 0


def test_usage_sums_every_output_sequence():
    result = _generation(completion_tokens=3)
    result.outputs.append(SimpleNamespace(token_ids=[1, 2]))
    assert _create_usage(result).output_tokens == 5


def test_usage_is_omitted_without_prompt_tokens():
    """Reporting zero would look like a real count of no tokens."""
    result = _generation()
    result.prompt_token_ids = None
    assert _create_usage(result) is None


# ---------------------------------------------------------------------------
# Results handed to a postprocessing worker
# ---------------------------------------------------------------------------


def test_usage_uses_the_prompt_token_count_supplied_by_the_caller():
    """Regression: usage was null for every request on the served path.

    Postprocessing runs in a separate worker, and the result it receives has
    no link back to the request that produced it, so its prompt tokens are
    unreachable. The executor records the count on the postprocessing
    arguments instead, and that is what has to be used.
    """
    result = _generation(prompt_tokens=7, completion_tokens=3)
    del result.prompt_token_ids
    usage = _create_usage(result, num_prompt_tokens=11)
    assert usage.input_tokens == 11
    assert usage.total_tokens == 14


def test_supplied_prompt_token_count_wins_over_the_result():
    result = _generation(prompt_tokens=7)
    assert _create_usage(result, num_prompt_tokens=11).input_tokens == 11


# ---------------------------------------------------------------------------
# The usage block has to survive being streamed
# ---------------------------------------------------------------------------


def test_usage_validates_inside_a_streamed_completion_event():
    """Regression: a usage block the SDK rejects silently truncates the stream.

    response.completed embeds the response, and the SDK model re-validates it.
    A missing field raises while the response is already being streamed, so
    the client gets deltas and then nothing - no terminating event, no error -
    and waits indefinitely. Building the event is what catches this; checking
    our own model would not, since ours is what was wrong.
    """
    from openai.types.responses import ResponseCompletedEvent

    usage = _create_usage(_generation(prompt_tokens=7, completion_tokens=3, cached_tokens=6))
    event = ResponseCompletedEvent(
        type="response.completed",
        sequence_number=0,
        response={
            "id": "resp_1",
            "created_at": 0.0,
            "model": "m",
            "object": "response",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": usage.model_dump(),
        },
    )
    assert event.response.usage.input_tokens == 7
    assert event.response.usage.input_tokens_details.cached_tokens == 6


# ---------------------------------------------------------------------------
# Disaggregated serving: the context phase's usage comes with the handoff
# ---------------------------------------------------------------------------


def _ctx_usage(prompt_tokens, cached_tokens):
    return UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=1,
        total_tokens=prompt_tokens + 1,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=cached_tokens),
    )


def _handed_off(ctx_usage, prompt_tokens=100, completion_tokens=5):
    """A generation-worker result: the whole prompt arrived as transferred KV."""
    result = _generation(prompt_tokens, completion_tokens, cached_tokens=prompt_tokens)
    result.outputs[0].disaggregated_params = SimpleNamespace(ctx_usage=ctx_usage)
    return result


@pytest.mark.parametrize(
    "ctx_usage, input_tokens, cached_tokens",
    [
        (_ctx_usage(96, 32), 96, 32),
        (_ctx_usage(100, 0), 100, 0),
        (_ctx_usage(100, 40).model_dump(), 100, 40),
        (UsageInfo(prompt_tokens=100, completion_tokens=1, total_tokens=101), 100, 0),
    ],
    ids=["warm", "cold", "as_dict", "no_details"],
)
def test_the_context_phase_decides_prompt_and_cached_tokens(ctx_usage, input_tokens, cached_tokens):
    usage = _create_usage(_handed_off(ctx_usage), num_prompt_tokens=100)
    assert usage.input_tokens == input_tokens
    assert usage.input_tokens_details.cached_tokens == cached_tokens
    assert usage.output_tokens == 5


# ---------------------------------------------------------------------------
# reasoning_tokens: re-encoded from the text the reasoning parser claimed
# ---------------------------------------------------------------------------


class _WordTokenizer:
    """One token per whitespace-separated word."""

    def encode(self, text, add_special_tokens=False):
        return text.split()


@pytest.mark.parametrize(
    "text, parser, tokenizer, expected",
    [
        ("one two three</think>four five", "glm", _WordTokenizer(), 3),
        ("one two</think>three four</think><tool_call>x</tool_call>", "glm", _WordTokenizer(), 2),
        ("planning the call<tool_call>{}</tool_call>", "glm", _WordTokenizer(), 3),
        ("still thinking about it", "glm", _WordTokenizer(), 4),
        ("one two three", None, _WordTokenizer(), 0),
        ("one two three", "glm", None, 0),
    ],
    ids=["closed", "first_close_counts", "implicit_end", "never_closed", "no_parser", "no_tok"],
)
def test_reasoning_tokens_count_what_the_parser_called_reasoning(text, parser, tokenizer, expected):
    _items, _messages, reasoning_texts = _create_output_content(
        SimpleNamespace(outputs=[SimpleNamespace(index=0, text=text)]), reasoning_parser=parser
    )
    assert _count_reasoning_tokens(tokenizer, reasoning_texts, 10_000) == expected


def test_reasoning_tokens_never_exceed_generated_tokens():
    assert _count_reasoning_tokens(_WordTokenizer(), ["a b c d e"], 3) == 3


def test_usage_reports_reasoning_tokens_within_the_output_tokens():
    usage = _create_usage(
        _generation(prompt_tokens=7, completion_tokens=9),
        tokenizer=_WordTokenizer(),
        reasoning_texts=["one two three four"],
    )
    assert usage.output_tokens_details.reasoning_tokens == 4
    assert (usage.output_tokens, usage.total_tokens) == (9, 16)
    assert _create_usage(_generation()).output_tokens_details.reasoning_tokens == 0


def test_a_streamed_response_counts_reasoning_tokens():
    request = ResponsesRequest(model="m", input="hi", stream=True)
    processor = ResponsesStreamingProcessor(
        request=request,
        sampling_params=request.to_sampling_params(),
        model_name="m",
        use_harmony=False,
        reasoning_parser="glm",
    )
    output = SimpleNamespace(
        index=0,
        text="one two</think>answer",
        text_diff="one two</think>answer",
        finish_reason="stop",
        token_ids=[1, 2, 3],
        disaggregated_params=None,
    )
    result = SimpleNamespace(outputs=[output], _done=True, prompt_token_ids=[1], cached_tokens=0)
    processor.process_single_output(result)
    frame = processor.get_final_response_non_store(result, tokenizer=_WordTokenizer())
    usage = json.loads(frame.split("data: ", 1)[1])["response"]["usage"]
    assert usage["output_tokens_details"]["reasoning_tokens"] == 2


# ---------------------------------------------------------------------------
# Prompt usage is counted as on the chat route
# ---------------------------------------------------------------------------

_QUESTION = "What is two plus two?"
_ANSWER_TOKENS = [41, 42, 43]
_CACHED_TOKENS = 4
# Trailing rendered prompt tokens left out of prompt usage: kimi_k3's serving
# extension leaves out its three-token generation channel opener, and a model
# without one reports the whole prompt.
_EXCLUDED = {"kimi_k3": 3, "qwen3": 0}


class _PromptTokenizer:
    """One token per whitespace-separated word."""

    vocab_size = 1024

    def __init__(self):
        self.tokenizer = self

    def encode(self, text, add_special_tokens=False, **kwargs):
        return [len(word) for word in text.split()]


async def _render(*, conversation, add_generation_prompt, **kwargs):
    """A chat template: the messages, then a three-token generation prompt."""
    text = " ".join(f"{message['role']}: {message['content']}" for message in conversation)
    if add_generation_prompt:
        text += " <|open|> think <|sep|>"
    return text


class _FinishedGeneration:
    """What the engine hands back for a request: finished, one output, one step."""

    def __init__(self, prompt_token_ids, disaggregated_params=None, cached_tokens=_CACHED_TOKENS):
        self.id = self.request_id = 1
        self.prompt_token_ids = prompt_token_ids
        self.cached_tokens = cached_tokens
        self.outputs = [
            CompletionOutput(
                index=0,
                text="Four.",
                token_ids=list(_ANSWER_TOKENS),
                finish_reason="stop",
                disaggregated_params=disaggregated_params,
            )
        ]
        self.finished = self._done = True
        self.error = None
        self._steps = [self]

    async def aresult(self):
        return self

    def __await__(self):
        return self.aresult().__await__()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._steps:
            raise StopAsyncIteration
        return self._steps.pop()


def _serve(
    tmp_path,
    model_type,
    *,
    tokenizer=None,
    chat_template=None,
    cached_tokens=_CACHED_TOKENS,
    allow_request_chat_template=False,
):
    """Chat, Messages and Responses routes of one server.

    Returns (client, prompts sent to the engine). ``cached_tokens`` is the engine's count of
    reused prompt tokens, or a function of the prompt token ids that gives it.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    if tokenizer is None:
        tokenizer = _PromptTokenizer()
    prompts = []

    def generate_async(*, inputs, disaggregated_params=None, **kwargs):
        if isinstance(inputs, dict):
            ids = inputs.get("prompt_token_ids") or tokenize_prompt(
                tokenizer, inputs["prompt"], add_special_tokens=False
            )
        else:
            ids = list(inputs)
        prompts.append(ids)
        cached = cached_tokens(ids) if callable(cached_tokens) else cached_tokens
        return _FinishedGeneration(ids, disaggregated_params, cached_tokens=cached)

    model_config = type("ModelConfig", (), {"model_type": model_type, "vocab_size": 1024})()
    server = make_stub_openai_server(
        tmp_path, "test-model", model_config, generate_async, tokenizer=tokenizer
    )
    server.chat_template = chat_template
    server.allow_request_chat_template = allow_request_chat_template
    server.await_disconnected = AsyncMock()
    server._extract_metrics = AsyncMock()
    app = FastAPI()
    app.add_api_route("/v1/chat/completions", server.openai_chat, methods=["POST"])
    app.add_api_route("/v1/messages", server.anthropic_messages, methods=["POST"])
    app.add_api_route("/v1/responses", server.openai_responses, methods=["POST"])
    return TestClient(app), prompts


@pytest.fixture
def text_only(monkeypatch):
    """Text-only conversations: the multimodal placeholder registry has nothing to add."""
    from tensorrt_llm.serve import chat_utils

    monkeypatch.setattr(
        chat_utils.MULTIMODAL_PLACEHOLDER_REGISTRY, "get_content_format", lambda _: None
    )


@pytest.fixture
def serve(tmp_path, monkeypatch, text_only):
    """``_serve`` with ``_render`` as the chat template."""
    from tensorrt_llm.serve.render import chat as render_chat_module

    monkeypatch.setattr(render_chat_module, "async_apply_chat_template", _render)
    return lambda model_type, **kwargs: _serve(tmp_path, model_type, **kwargs)


def _sse_data(text):
    return [
        json.loads(line[len("data: ") :])
        for line in text.splitlines()
        if line.startswith("data: {")
    ]


def _chat_usage(client, stream, **fields):
    """(prompt_tokens, total_tokens, cached_tokens) of a chat completion of ``_QUESTION``."""
    body = {"model": "test-model", "messages": [{"role": "user", "content": _QUESTION}], **fields}
    if stream:
        body.update(stream=True, stream_options={"include_usage": True})
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text
    if stream:
        (usage,) = [chunk["usage"] for chunk in _sse_data(response.text) if not chunk["choices"]]
    else:
        usage = response.json()["usage"]
    return (
        usage["prompt_tokens"],
        usage["total_tokens"],
        usage["prompt_tokens_details"]["cached_tokens"],
    )


def _responses_usage(client, stream, **fields):
    """(input_tokens, total_tokens, cached_tokens) of a Responses request for ``_QUESTION``."""
    body = {"model": "test-model", "input": _QUESTION, "stream": stream, **fields}
    response = client.post("/v1/responses", json=body)
    assert response.status_code == 200, response.text
    if stream:
        (usage,) = [
            event["response"]["usage"]
            for event in _sse_data(response.text)
            if event["type"] == "response.completed"
        ]
    else:
        usage = response.json()["usage"]
    return (
        usage["input_tokens"],
        usage["total_tokens"],
        usage["input_tokens_details"]["cached_tokens"],
    )


@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
@pytest.mark.parametrize("model_type", ["kimi_k3", "qwen3"])
def test_responses_counts_the_prompt_as_chat_completions_does(serve, model_type, stream):
    """One conversation, one prompt count: the serving extension decides it on both routes.

    Cached tokens are the engine's count on both routes, and the total is the
    reported prompt plus the generated tokens.
    """
    client, prompts = serve(model_type)

    chat = _chat_usage(client, stream)
    responses = _responses_usage(client, stream)

    chat_prompt, responses_prompt = prompts
    assert responses_prompt == chat_prompt
    prompt_tokens = len(chat_prompt) - _EXCLUDED[model_type]
    assert chat == (prompt_tokens, prompt_tokens + len(_ANSWER_TOKENS), _CACHED_TOKENS)
    assert responses == chat


@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
def test_a_server_chat_template_is_counted_whole_on_both_routes(serve, stream):
    """A server-level template need not end in the opener, so neither route leaves it out."""
    client, prompts = serve("kimi_k3", chat_template="{{ messages }}")

    chat = _chat_usage(client, stream)
    responses = _responses_usage(client, stream)

    assert chat[0] == responses[0] == len(prompts[0])


@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
def test_a_disaggregated_response_counts_the_prompt_as_chat_completions_does(serve, stream):
    """The context worker leaves the opener out; the generation worker reports its count as is."""
    client, prompts = serve("kimi_k3")
    context = client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": _QUESTION,
            "store": False,
            "disaggregated_params": {"request_type": "context_only"},
        },
    )
    assert context.status_code == 200, context.text
    context = context.json()
    # What the orchestrator relays to the generation worker.
    ctx_usage = _ctx_usage_info(
        SimpleNamespace(usage=ResponseUsage.model_validate(context["usage"]))
    )

    generation = _responses_usage(
        client,
        stream,
        store=False,
        prompt_token_ids=context["prompt_token_ids"],
        disaggregated_params={
            "request_type": "generation_only",
            "ctx_usage": ctx_usage.model_dump(),
        },
    )

    rendered, relayed = prompts
    assert relayed == rendered
    assert generation[0] == len(rendered) - 3 == _chat_usage(client, stream)[0]


def test_a_relayed_prompt_without_context_usage_is_counted_as_chat_counts_it(serve):
    """A generation worker that renders nothing has no opener to leave out, on either route."""
    client, _ = serve("kimi_k3")
    relayed = {
        "prompt_token_ids": [5, 4, 3, 2, 1],
        "disaggregated_params": {"request_type": "generation_only"},
    }

    responses = _responses_usage(client, False, store=False, **relayed)
    chat = _chat_usage(client, False, **relayed)

    assert responses[0] == chat[0] == 5


def test_an_empty_relayed_prompt_is_rendered_and_leaves_out_the_opener(serve):
    """An empty ``prompt_token_ids_b64`` relays no prompt, so the input is rendered as usual."""
    client, prompts = serve("kimi_k3")

    plain = _responses_usage(client, False, store=False)
    empty_relay = _responses_usage(
        client,
        False,
        store=False,
        prompt_token_ids_b64="",
        disaggregated_params={"request_type": "generation_only"},
    )

    rendered, also_rendered = prompts
    assert also_rendered == rendered
    assert empty_relay[0] == plain[0] == len(rendered) - 3


_RECORDING = "usage-facts-model"


@pytest.fixture
def asked(monkeypatch):
    """Rendering facts the serving extension of ``_RECORDING`` is asked about, in order."""
    from tensorrt_llm.serve import serving_extensions

    recorded = []

    class RecordingExtension(serving_extensions.ServingExtension):
        def prompt_tokens_excluded_from_usage(self, **facts):
            recorded.append(facts)
            return 3

    serving_extensions.load_builtin_extensions()
    monkeypatch.setitem(serving_extensions._BY_MODEL_TYPE, _RECORDING, RecordingExtension())
    return recorded


@pytest.mark.parametrize(
    "relayed",
    [
        {},
        {"prompt_token_ids": [5, 4, 3, 2, 1]},
        {"prompt_token_ids_b64": base64.b64encode(struct.pack("<5i", 5, 4, 3, 2, 1)).decode()},
    ],
    ids=["rendered", "relayed_ids", "relayed_b64"],
)
def test_both_routes_ask_the_extension_the_same_question(serve, asked, relayed):
    """Both routes hand the extension how the prompt is rendered, and the same facts for it."""
    client, _ = serve(_RECORDING)
    fields = {}
    if relayed:
        fields = {**relayed, "disaggregated_params": {"request_type": "generation_only"}}

    _chat_usage(client, False, **fields)
    _responses_usage(client, False, store=False, **fields)

    facts = {
        "add_generation_prompt": True,
        "pre_tokenized": bool(relayed),
        "request_chat_template": None,
    }
    assert asked == [facts, facts]


@pytest.mark.parametrize(
    "fields",
    [{"chat_template": "{{ messages }}"}, {"add_generation_prompt": False}],
    ids=["request_chat_template", "no_generation_prompt"],
)
def test_chat_puts_how_its_request_renders_to_the_extension(serve, asked, fields):
    client, _ = serve(_RECORDING, allow_request_chat_template=True)

    _chat_usage(client, False, **fields)

    assert asked == [
        {
            "add_generation_prompt": fields.get("add_generation_prompt", True),
            "pre_tokenized": False,
            "request_chat_template": fields.get("chat_template"),
        }
    ]


def test_neither_route_asks_the_extension_under_a_server_chat_template(serve, asked):
    client, _ = serve(_RECORDING, chat_template="{{ messages }}")

    _chat_usage(client, False)
    _responses_usage(client, False)

    assert asked == []


def _messages_usage(client):
    """(input_tokens, cache_read_input_tokens) of an Anthropic Messages request for ``_QUESTION``."""
    body = {
        "model": "test-model",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": _QUESTION}],
    }
    response = client.post("/v1/messages", json=body)
    assert response.status_code == 200, response.text
    usage = response.json()["usage"]
    return usage["input_tokens"], usage.get("cache_read_input_tokens") or 0


@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
def test_cached_tokens_never_exceed_the_reported_prompt(serve, stream):
    """An identical retry reuses all but the last prompt token: more than the reported prompt.

    Clients take the uncached input as prompt minus cached tokens, so every route caps the
    cached count at the prompt count it reports.
    """
    client, prompts = serve("kimi_k3", cached_tokens=lambda prompt: len(prompt) - 1)

    chat = _chat_usage(client, stream)
    responses = _responses_usage(client, stream)

    prompt_tokens = len(prompts[0]) - 3
    assert chat == responses == (prompt_tokens, prompt_tokens + len(_ANSWER_TOKENS), prompt_tokens)


def test_messages_counts_no_more_input_than_chat_completions(serve):
    """Messages input is the uncached plus the cached tokens: chat's prompt count, not more."""
    client, prompts = serve("kimi_k3", cached_tokens=lambda prompt: len(prompt) - 1)

    input_tokens, cache_read_input_tokens = _messages_usage(client)

    assert (input_tokens, cache_read_input_tokens) == (0, len(prompts[0]) - 3)


def test_every_continuous_chat_usage_caps_the_cached_tokens(serve):
    client, prompts = serve("kimi_k3", cached_tokens=lambda prompt: len(prompt) - 1)
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": _QUESTION}],
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }

    response = client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200, response.text
    usages = [chunk["usage"] for chunk in _sse_data(response.text) if chunk.get("usage")]
    prompt_tokens = len(prompts[0]) - 3
    # The role chunk, the content chunk, the finish chunk and the final usage chunk.
    assert len(usages) == 4
    for usage in usages:
        assert usage["prompt_tokens"] == prompt_tokens
        assert usage["prompt_tokens_details"]["cached_tokens"] == prompt_tokens


@pytest.mark.parametrize(
    ("cached_tokens", "reported_cached_tokens"),
    [(_CACHED_TOKENS, _CACHED_TOKENS), (19, 17)],
    ids=["cached_within_the_reported_prompt", "cached_beyond_the_reported_prompt"],
)
@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
def test_a_postprocessing_worker_counts_the_prompt_as_chat_completions_does(
    stream, cached_tokens, reported_cached_tokens
):
    """On a postprocessing worker the executor records the prompt length; the offset applies there."""
    request = ResponsesRequest(model="m", input=_QUESTION, stream=stream)
    sampling_params = request.to_sampling_params()
    counts = {"num_prompt_tokens": 20, "num_prompt_tokens_offset": 3}
    responses_args = ResponsesAPIPostprocArgs(
        model="m",
        request=request,
        sampling_params=sampling_params,
        use_harmony=False,
        streaming_processor=ResponsesStreamingProcessor(
            request=request, sampling_params=sampling_params, model_name="m", use_harmony=False
        ),
        **counts,
    )
    chat_args = ChatPostprocArgs(
        role="assistant", model="m", stream_options=StreamOptions(include_usage=True), **counts
    )

    def result():
        return _FinishedGeneration([], cached_tokens=cached_tokens)

    if stream:
        (completed,) = [
            event
            for event in _sse_data(
                "".join(responses_api_streaming_post_processor(result(), responses_args))
            )
            if event["type"] == "response.completed"
        ]
        usage = completed["response"]["usage"]
        chat_usage = _sse_data("".join(chat_stream_post_processor(result(), chat_args)))[-1][
            "usage"
        ]
    else:
        usage = responses_api_post_processor(result(), responses_args).usage
        usage = usage.model_dump()
        chat_usage = chat_response_post_processor(result(), chat_args).usage
        chat_usage = chat_usage.model_dump()

    assert usage["input_tokens"] == chat_usage["prompt_tokens"] == 17
    assert usage["total_tokens"] == chat_usage["total_tokens"] == 17 + len(_ANSWER_TOKENS)
    assert (
        usage["input_tokens_details"]["cached_tokens"]
        == chat_usage["prompt_tokens_details"]["cached_tokens"]
        == reported_cached_tokens
    )


def test_kimi_k3_leaves_out_its_channel_opener_on_both_routes(tmp_path, text_only):
    """With the checkpoint's own tokenizer and template (skipped when the checkpoint is absent)."""
    from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer

    from .render_helpers import KIMI_K3, make_tokenizer

    hf_tokenizer = make_tokenizer(name=KIMI_K3)
    client, prompts = _serve(tmp_path, "kimi_k3", tokenizer=TransformersTokenizer(hf_tokenizer))

    chat = _chat_usage(client, stream=False)
    responses = _responses_usage(client, stream=False)

    chat_prompt, responses_prompt = prompts
    assert responses_prompt == chat_prompt
    assert hf_tokenizer.decode(chat_prompt[-3:]) in (
        "<|open|>think<|sep|>",
        "<|open|>response<|sep|>",
    )
    assert responses[0] == chat[0] == len(chat_prompt) - 3
