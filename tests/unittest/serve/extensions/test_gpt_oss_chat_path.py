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
"""gpt-oss chat completions through the shared chat handler and the gpt-oss serving extension.

Every server here is built through the real constructor on a fake gpt-oss checkpoint,
so ``use_harmony`` comes from ``OpenAIServer._init_llm``. The legacy handler
``chat_harmony`` is mounted on a second route so that the same request body can be
sent through both handlers and their answers compared: every rejection must be the
same, the engine must receive the same prompt tokens and sampling parameters, and
only the capabilities listed in the chat handler may differ.

The harmony vocabulary is read from ``$LLM_MODELS_ROOT/datasets/tiktoken_vocab``;
tests that render skip when it is missing.
"""

import base64
import json
import logging
import signal
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from llmapi.apps.render_helpers import use_harmony_vocab
from test_common.serve.app_factory import AggConfig
from test_common.serve.fakes import ScriptedOutput, ScriptedStep
from test_common.serve.routes import effective_routes

from tensorrt_llm.commands.serve import main as serve_main
from tensorrt_llm.executor import CppExecutorError
from tensorrt_llm.executor.utils import RequestError, context_length_exceeded_message
from tensorrt_llm.logger import logger
from tensorrt_llm.serve import openai_server
from tensorrt_llm.serve.chat_tokenization import tokenize_harmony_chat_request
from tensorrt_llm.serve.harmony_adapter import get_harmony_adapter
from tensorrt_llm.serve.openai_protocol.chat import ChatCompletionRequest
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.postprocess_handlers import (
    chat_harmony_post_processor,
    chat_harmony_streaming_post_processor,
    chat_response_post_processor,
)
from tensorrt_llm.serve.serving_extensions import (
    UNSUPPORTED_FEATURE_MESSAGES,
    OutputMode,
    UnsupportedFeature,
)

# threadleak is off: the TestClients and the fake server's worker pools are
# closed by the serve_env fixture's teardown, which runs after the leak check.
pytestmark = [pytest.mark.cpu_only, pytest.mark.threadleak(enabled=False)]

MODEL = "fake-model"
CHAT_ROUTE = "/v1/chat/completions"
# The legacy handler, mounted by _gpt_oss_server for side-by-side comparison.
LEGACY_ROUTE = "/legacy/v1/chat/completions"
SWITCH = "TRTLLM_GPT_OSS_LEGACY_CHAT"
FINAL_CHANNEL_START = "<|start|>assistant<|channel|>final<|message|>"

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather of a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
_JSON_SCHEMA_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
    },
}
GENERIC_NAMED_TOOL_MESSAGE = (
    "tool_choice with a named function requires the server to be started with a "
    "guided decoding backend (set `guided_decoding_backend` to 'xgrammar' or "
    "'llguidance')."
)


@pytest.fixture
def harmony_vocab(monkeypatch):
    use_harmony_vocab(monkeypatch)


@pytest.fixture
def warning_records(monkeypatch):
    """Messages the TRT-LLM logger emits at WARNING or above during the test.

    The library logger does not propagate, so ``caplog`` sees nothing; a handler
    is attached instead, and the logger is lowered to ``warning`` for the test
    because its default level drops warnings before any handler.
    """
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Capture(level=logging.WARNING)
    previous_level = logger.level
    monkeypatch.setattr(logger, "_set_from_env", False)
    logger.set_level("warning")
    logger._logger.addHandler(handler)
    try:
        yield records
    finally:
        logger._logger.removeHandler(handler)
        logger.set_level(previous_level)


def _final_message_tokens(text: str = "ok") -> tuple:
    """Token ids of a harmony final-channel assistant message ending in <|return|>."""
    encoding = get_harmony_adapter().encoding
    return tuple(encoding.encode(f"{FINAL_CHANNEL_START}{text}<|return|>", allowed_special="all"))


def _harmony_output(text: str = "ok") -> ScriptedOutput:
    return ScriptedOutput(
        steps=(ScriptedStep(token_ids=_final_message_tokens(text), finish_reason="stop"),)
    )


def _gpt_oss_server(test_env, **config) -> OpenAIServer:
    server = test_env.build_openai_server(AggConfig(model_type="gpt_oss", **config))
    server.app.add_api_route(LEGACY_ROUTE, server.chat_harmony, methods=["POST"])
    return server


def _chat_body(**overrides) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "max_completion_tokens": 8,
    }
    body.update(overrides)
    return body


def _chat_endpoint(app):
    for path, route in effective_routes(app):
        if path == CHAT_ROUTE and "POST" in getattr(route, "methods", ()):
            return route.endpoint
    raise AssertionError(f"{CHAT_ROUTE} is not registered")


def _answer(response) -> tuple:
    return response.status_code, response.json()


def _prompt_token_ids_of(server, call_index: int = -1) -> list:
    """The prompt token ids the shared chat handler handed to the engine.

    The handler runs the engine's ``preprocess`` on the prompt first, so the
    recorded ``generate_async`` input is the preprocessed object; the token ids
    are what ``preprocess`` received.
    """
    call = server.generator.preprocess_calls[call_index]
    inputs = call.inputs
    return list(inputs["prompt_token_ids"] if isinstance(inputs, dict) else inputs)


# ---------------------------------------------------------------------------
# Route selection and the switch
# ---------------------------------------------------------------------------


def test_default_route_is_openai_chat(serve_env, warning_records):
    server = _gpt_oss_server(serve_env)

    assert server.use_harmony is True
    assert _chat_endpoint(server.app).__name__ == "openai_chat", (
        "gpt-oss chat must route to openai_chat unless TRTLLM_GPT_OSS_LEGACY_CHAT=1"
    )
    assert not [m for m in warning_records if SWITCH in m], (
        "no switch warning is logged when the switch is not set"
    )
    # Positive control: the capture sees a WARNING when one is logged, so the
    # "not logged" assertion above cannot pass vacuously.
    logger.warning("log-capture control")
    assert any("log-capture control" in m for m in warning_records), (
        "the log-capture handler saw no WARNING; lower the TRT-LLM logger level for "
        "the test (its default level, error, drops warnings)"
    )


def test_switch_restores_chat_harmony_and_warns_once(serve_env, warning_records):
    server = _gpt_oss_server(serve_env, env=((SWITCH, "1"),))

    assert _chat_endpoint(server.app).__name__ == "chat_harmony", (
        "TRTLLM_GPT_OSS_LEGACY_CHAT=1 must route gpt-oss chat to chat_harmony"
    )
    switch_warnings = [m for m in warning_records if SWITCH in m]
    assert len(switch_warnings) == 1, f"the switch must be logged exactly once: {switch_warnings}"
    assert "chat_harmony" in switch_warnings[0] and "removed" in switch_warnings[0]


def test_switch_is_ignored_on_a_generic_model(serve_env, warning_records):
    server = serve_env.build_openai_server(AggConfig(model_type="llama", env=((SWITCH, "1"),)))

    assert server.use_harmony is False
    assert _chat_endpoint(server.app).__name__ == "openai_chat"
    assert not [m for m in warning_records if SWITCH in m]


# ---------------------------------------------------------------------------
# Rejection parity with chat_harmony
# ---------------------------------------------------------------------------

_NAMED = {"type": "function", "function": {"name": "get_weather"}}
# An id far beyond any vocabulary, so to_sampling_params rejects the logit_bias.
_OUT_OF_RANGE_LOGIT_BIAS = {str(10**9): 1.0}

REJECTIONS = {
    "request_chat_template": _chat_body(chat_template="{{ messages }}"),
    "named_tool_choice": _chat_body(tools=[_WEATHER_TOOL], tool_choice=_NAMED),
    "required_tool_choice": _chat_body(tools=[_WEATHER_TOOL], tool_choice="required"),
    "thinking_token_budget": _chat_body(thinking_token_budget=16),
    "top_logprobs": _chat_body(logprobs=True, top_logprobs=2),
    # Pairs: both handlers must report the same one of the two problems.
    "named_tool_choice+budget": _chat_body(
        tools=[_WEATHER_TOOL], tool_choice=_NAMED, thinking_token_budget=16
    ),
    # The two features checked before rendering, in the order chat_harmony checks them.
    "named_tool_choice+top_logprobs": _chat_body(
        tools=[_WEATHER_TOOL], tool_choice=_NAMED, logprobs=True, top_logprobs=2
    ),
    "top_logprobs+budget": _chat_body(logprobs=True, top_logprobs=2, thinking_token_budget=16),
    "logit_bias+budget": _chat_body(logit_bias=_OUT_OF_RANGE_LOGIT_BIAS, thinking_token_budget=16),
}


@pytest.mark.parametrize("feature", sorted(REJECTIONS), ids=sorted(REJECTIONS))
def test_rejections_match_chat_harmony(serve_env, harmony_vocab, feature):
    server = _gpt_oss_server(serve_env)
    client = serve_env.client(server.app)
    body = REJECTIONS[feature]

    new = _answer(client.post(CHAT_ROUTE, json=body))
    legacy = _answer(client.post(LEGACY_ROUTE, json=body))

    assert new == legacy, (
        f"openai_chat must reject {feature} for gpt-oss exactly as chat_harmony does: "
        f"openai_chat answered {new}, chat_harmony answered {legacy}"
    )
    assert new[0] == 400
    assert not server.generator.generate_calls, "a rejected request must not reach the engine"


def test_rejection_messages_are_chat_harmony_texts(serve_env, harmony_vocab):
    """The declared messages are byte for byte what chat_harmony answers."""
    server = _gpt_oss_server(serve_env)
    client = serve_env.client(server.app)
    for feature, body in (
        (UnsupportedFeature.NAMED_TOOL_CHOICE, REJECTIONS["named_tool_choice"]),
        (UnsupportedFeature.THINKING_TOKEN_BUDGET, REJECTIONS["thinking_token_budget"]),
        (UnsupportedFeature.TOP_LOGPROBS, REJECTIONS["top_logprobs"]),
    ):
        legacy_status, legacy_body = _answer(client.post(LEGACY_ROUTE, json=body))
        assert (legacy_status, legacy_body["message"]) == (
            400,
            UNSUPPORTED_FEATURE_MESSAGES[feature],
        ), f"UNSUPPORTED_FEATURE_MESSAGES[{feature}] must equal chat_harmony's message"


# ---------------------------------------------------------------------------
# Generation inputs
# ---------------------------------------------------------------------------

_GENERATION_BODY = _chat_body(
    messages=[
        {"role": "developer", "content": "Use tools when useful."},
        {"role": "user", "content": "weather in Paris?"},
    ],
    tools=[_WEATHER_TOOL],
    tool_choice="auto",
    reasoning_effort="medium",
    stop_token_ids=[7],
    response_format=_JSON_SCHEMA_FORMAT,
    temperature=0.5,
    top_p=0.9,
    n=1,
)
_COMPARED_SAMPLING_FIELDS = (
    "stop_token_ids",
    "detokenize",
    "logprobs",
    "max_tokens",
    "temperature",
    "top_p",
    "n",
)


def _guided(sampling_params):
    guided = sampling_params.guided_decoding
    return None if guided is None else json.loads(guided.structural_tag)


def _sse_frames(text: str) -> list:
    """The JSON frames of an SSE body without the per-response id and timestamp."""
    frames = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            frame = json.loads(line[len("data: ") :])
            frame.pop("id", None)
            frame.pop("created", None)
            frames.append(frame)
    return frames


@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
@pytest.mark.parametrize("workers", [0, 1], ids=["in-process", "postproc-workers"])
def test_generation_inputs_match_chat_harmony(serve_env, harmony_vocab, workers, stream):
    server = _gpt_oss_server(serve_env, num_postprocess_workers=workers)
    server.generator.script(_harmony_output(), _harmony_output())
    client = serve_env.client(server.app)
    body = {**_GENERATION_BODY, "stream": stream}

    new = client.post(CHAT_ROUTE, json=body)
    legacy = client.post(LEGACY_ROUTE, json=body)
    assert new.status_code == legacy.status_code == 200, (new.text, legacy.text)
    if stream:
        assert new.text.endswith("data: [DONE]\n\n") and legacy.text.endswith("data: [DONE]\n\n")
        assert _sse_frames(new.text) == _sse_frames(legacy.text)
    else:
        assert new.json()["choices"][0]["message"] == legacy.json()["choices"][0]["message"]

    new_call, legacy_call = server.generator.generate_calls
    expected_tokens = tokenize_harmony_chat_request(ChatCompletionRequest(**body))
    assert _prompt_token_ids_of(server) == list(legacy_call.inputs) == expected_tokens, (
        "gpt-oss generation inputs differ between openai_chat and chat_harmony beyond "
        "the listed capabilities: the harmony prompt token ids"
    )
    new_sp, legacy_sp = new_call.sampling_params, legacy_call.sampling_params
    for field in _COMPARED_SAMPLING_FIELDS:
        new_value, legacy_value = getattr(new_sp, field), getattr(legacy_sp, field)
        if field == "stop_token_ids":
            # Both handlers append what the adapter returns, and the adapter
            # returns its two stop ids in no fixed order.
            new_value, legacy_value = sorted(new_value), sorted(legacy_value)
        assert new_value == legacy_value, (
            f"gpt-oss generation inputs differ between openai_chat and chat_harmony beyond "
            f"the listed capabilities: sampling_params.{field}: {new_value!r} vs {legacy_value!r}"
        )
    assert new_sp.detokenize is False and new_sp.logprobs is None
    assert set(get_harmony_adapter().get_stop_tokens()) <= set(new_sp.stop_token_ids)
    assert 7 in new_sp.stop_token_ids, "the client's stop ids are kept"
    assert _guided(new_sp) == _guided(legacy_sp)
    assert _guided(new_sp)["format"]["type"] == "triggered_tags"
    assert _guided(new_sp)["format"]["triggers"] == [FINAL_CHANNEL_START]
    # The only keyword argument gained is cache_salt.
    assert set(new_call.kwargs) - set(legacy_call.kwargs) == {"cache_salt"}
    assert not (set(legacy_call.kwargs) - set(new_call.kwargs))
    for key in ("streaming", "lora_request", "disaggregated_params", "kv_hints", "priority"):
        assert new_call.kwargs[key] == legacy_call.kwargs[key], key
    assert new_call.kwargs["streaming"] is stream
    expected_post_processor = (
        chat_harmony_streaming_post_processor if stream else chat_harmony_post_processor
    )
    if workers:
        assert new_call.kwargs["_postproc_params"].post_processor is expected_post_processor
        assert legacy_call.kwargs["_postproc_params"].post_processor is expected_post_processor
    else:
        assert new_call.kwargs["_postproc_params"] is None


def test_stream_null_is_a_non_streaming_request(serve_env, harmony_vocab):
    """``"stream": null`` validates to None; every handler hands the engine a bool."""
    gpt_oss = _gpt_oss_server(serve_env)
    gpt_oss.generator.script(_harmony_output(), _harmony_output())
    client = serve_env.client(gpt_oss.app)
    body = _chat_body(stream=None)

    new = client.post(CHAT_ROUTE, json=body)
    legacy = client.post(LEGACY_ROUTE, json=body)
    assert new.status_code == legacy.status_code == 200, (new.text, legacy.text)
    assert new.json()["choices"][0]["message"] == legacy.json()["choices"][0]["message"]
    new_call, legacy_call = gpt_oss.generator.generate_calls
    assert new_call.kwargs["streaming"] is False and legacy_call.kwargs["streaming"] is False, (
        "a null stream field must reach the engine as streaming=False on both handlers"
    )

    generic = serve_env.build_openai_server(AggConfig(model_type="llama"))
    assert serve_env.client(generic.app).post(CHAT_ROUTE, json=body).status_code == 200
    assert generic.generator.generate_calls[0].kwargs["streaming"] is False, (
        "the generic chat path must hand the engine a bool: the executor's request binding "
        "rejects streaming=None"
    )


def test_structured_output_uses_the_extension_format(serve_env, harmony_vocab):
    gpt_oss = _gpt_oss_server(serve_env)
    gpt_oss.generator.script(_harmony_output())
    serve_env.client(gpt_oss.app).post(
        CHAT_ROUTE, json=_chat_body(response_format=_JSON_SCHEMA_FORMAT)
    )
    guided = _guided(gpt_oss.generator.generate_calls[0].sampling_params)
    assert guided is not None and guided["format"]["type"] == "triggered_tags", (
        "gpt-oss structured output must be placed on the final channel"
    )
    assert guided["format"]["triggers"] == [FINAL_CHANNEL_START]
    assert (
        guided["format"]["tags"][0]["content"]["json_schema"]
        == (_JSON_SCHEMA_FORMAT["json_schema"]["schema"])
    )


def test_generic_structured_output_keeps_the_raw_grammar(serve_env):
    generic = serve_env.build_openai_server(AggConfig(model_type="llama"))
    serve_env.client(generic.app).post(
        CHAT_ROUTE, json=_chat_body(response_format=_JSON_SCHEMA_FORMAT)
    )
    guided = generic.generator.generate_calls[0].sampling_params.guided_decoding
    assert guided is not None and guided.structural_tag is None, (
        "generic models with no reasoning parser keep the raw grammar"
    )
    assert guided.json == _JSON_SCHEMA_FORMAT["json_schema"]["schema"]


def test_logprobs_without_top_logprobs_is_accepted_and_not_requested(serve_env, harmony_vocab):
    server = _gpt_oss_server(serve_env)
    server.generator.script(_harmony_output())
    status, _ = _answer(
        serve_env.client(server.app).post(CHAT_ROUTE, json=_chat_body(logprobs=True))
    )
    assert status == 200
    assert server.generator.generate_calls[0].sampling_params.logprobs is None, (
        "gpt-oss must not ask the engine for logprobs the harmony output never reports"
    )


def test_history_with_non_json_tool_arguments_is_accepted(serve_env, harmony_vocab):
    server = _gpt_oss_server(serve_env)
    server.generator.script(_harmony_output(), _harmony_output())
    client = serve_env.client(server.app)
    body = _chat_body(
        messages=[
            {"role": "user", "content": "weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "not json"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            {"role": "user", "content": "and tomorrow?"},
        ],
        tools=[_WEATHER_TOOL],
    )

    new = _answer(client.post(CHAT_ROUTE, json=body))
    legacy = _answer(client.post(LEGACY_ROUTE, json=body))

    assert new[0] == 200, (
        f"gpt-oss must accept non-JSON historical tool-call arguments as chat_harmony did: {new}"
    )
    assert legacy[0] == 200
    new_tokens, legacy_tokens = (
        _prompt_token_ids_of(server),
        list(server.generator.generate_calls[1].inputs),
    )
    assert new_tokens == legacy_tokens


# ---------------------------------------------------------------------------
# What gpt-oss gains from the shared handler
# ---------------------------------------------------------------------------


def _b64_int32(ids) -> str:
    return base64.b64encode(np.asarray(ids, dtype=np.int32).tobytes()).decode("ascii")


def test_gained_capabilities(serve_env, harmony_vocab, monkeypatch):
    server = _gpt_oss_server(serve_env)
    client = serve_env.client(server.app)
    gained = "behavior that gpt-oss gains through openai_chat no longer holds"

    # cache_salt reaches the engine; the engine's preprocess hook runs.
    server.generator.script(_harmony_output())
    status, _ = _answer(client.post(CHAT_ROUTE, json=_chat_body(cache_salt="salt-1")))
    assert status == 200
    call = server.generator.generate_calls[-1]
    assert call.kwargs["cache_salt"] == "salt-1", f"{gained}: cache_salt"
    assert len(server.generator.preprocess_calls) == 1, f"{gained}: the preprocess hook"

    # prompt_token_ids_b64 is decoded for any client, also without disaggregated_params.
    server.generator.script(_harmony_output())
    status, _ = _answer(
        client.post(CHAT_ROUTE, json=_chat_body(prompt_token_ids_b64=_b64_int32([11, 12, 13])))
    )
    assert status == 200
    assert _prompt_token_ids_of(server) == [11, 12, 13], f"{gained}: prompt_token_ids_b64"

    # A malformed value answers 400 BadRequestError: incorrect padding, then a
    # byte count that is not a multiple of four.
    for malformed in ("abc", "YWJj"):
        status, body = _answer(
            client.post(CHAT_ROUTE, json=_chat_body(prompt_token_ids_b64=malformed))
        )
        assert (status, body["type"]) == (400, "BadRequestError"), (
            f"{gained}: malformed prompt_token_ids_b64 {malformed!r} answered {status} {body}"
        )

    # An engine failure sends SIGINT and is answered HTTP 200 with body null.
    signals = []
    monkeypatch.setattr(signal, "raise_signal", signals.append)
    monkeypatch.setattr(openai_server, "_record_generator_termination", lambda generator: None)
    server.generator.script(ScriptedOutput(raise_on_call=CppExecutorError("engine down")))
    response = client.post(CHAT_ROUTE, json=_chat_body())
    assert (response.status_code, response.json()) == (200, None), f"{gained}: engine failure"
    assert signals == [signal.SIGINT]
    # The same failure through chat_harmony: 400 internal_error, no SIGINT.
    server.generator.script(ScriptedOutput(raise_on_call=CppExecutorError("engine down")))
    status, body = _answer(client.post(LEGACY_ROUTE, json=_chat_body()))
    assert (status, body["type"]) == (400, "internal_error")
    assert signals == [signal.SIGINT]


class _FailingRenderAdapter:
    """The process-wide adapter, except that rendering fails."""

    def __init__(self, error: Exception) -> None:
        self._real = get_harmony_adapter()
        self._error = error

    def openai_to_harmony_tokens(self, *args, **kwargs):
        raise self._error

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_unexpected_error_type_follows_openai_chat(serve_env, harmony_vocab):
    server = _gpt_oss_server(serve_env)
    client = serve_env.client(server.app)
    # Both handlers render with the server's adapter when it is set.
    server.harmony_adapter = _FailingRenderAdapter(RuntimeError("render boom"))

    new = _answer(client.post(CHAT_ROUTE, json=_chat_body()))
    legacy = _answer(client.post(LEGACY_ROUTE, json=_chat_body()))
    assert (new[0], new[1]["type"], new[1]["message"]) == (400, "BadRequestError", "render boom")
    assert (legacy[0], legacy[1]["type"], legacy[1]["message"]) == (
        400,
        "internal_error",
        "render boom",
    )

    # An over-length prompt: 400 with code context_length_exceeded on both; the
    # type follows each handler.
    server.harmony_adapter = None
    message = context_length_exceeded_message(2048, 4096)
    for route, err_type in ((CHAT_ROUTE, "BadRequestError"), (LEGACY_ROUTE, "internal_error")):
        server.generator.script(ScriptedOutput(raise_on_call=RequestError(message)))
        status, body = _answer(client.post(route, json=_chat_body()))
        assert (status, body["code"], body["type"], body["message"]) == (
            400,
            "context_length_exceeded",
            err_type,
            message,
        ), f"{route}: {status} {body}"


# ---------------------------------------------------------------------------
# DISABLE_HARMONY_ADAPTER, the startup check, the hook guard
# ---------------------------------------------------------------------------


def test_disable_harmony_adapter_keeps_the_text_path(serve_env):
    server = _gpt_oss_server(serve_env, env=(("DISABLE_HARMONY_ADAPTER", "1"),))
    client = serve_env.client(server.app)
    keep = "DISABLE_HARMONY_ADAPTER=1 must keep gpt-oss on the text path"

    assert server.use_harmony is False, keep
    status, _ = _answer(client.post(CHAT_ROUTE, json=_chat_body()))
    assert status == 200
    call = server.generator.generate_calls[0]
    # The request model's default list, with no harmony ids added.
    assert not call.sampling_params.stop_token_ids, f"{keep}: no extra stop ids"
    assert call.sampling_params.detokenize is True, keep
    rendered = server.generator.preprocess_calls[0].inputs
    assert isinstance(rendered, dict) and rendered.get("prompt") == "<|user|>hi<|assistant|>", (
        f"{keep}: the chat template renders the prompt, got {rendered!r}"
    )

    status, body = _answer(client.post(CHAT_ROUTE, json=REJECTIONS["named_tool_choice"]))
    assert (status, body["message"]) == (400, GENERIC_NAMED_TOOL_MESSAGE), (
        f"{keep}: a named tool_choice gets openai_chat's generic message"
    )


STARTUP_CHECK_CASES = {
    "gpt-oss+qwen3-tool-parser": (
        dict(
            use_harmony=True,
            model_type="gpt_oss",
            tool_parser="qwen3",
            reasoning_parser=None,
            parser_aliases=(),
        ),
        "--tool_parser 'qwen3'",
    ),
    "gpt-oss+qwen3-reasoning-parser": (
        dict(
            use_harmony=True,
            model_type="gpt_oss",
            tool_parser=None,
            reasoning_parser="qwen3",
            parser_aliases=(),
        ),
        "--reasoning_parser 'qwen3'",
    ),
    "gpt-oss+alias": (
        dict(
            use_harmony=True,
            model_type="gpt_oss",
            tool_parser=None,
            reasoning_parser=None,
            parser_aliases=("gpt_oss",),
        ),
        None,
    ),
    "gpt-oss+yaml-reasoning-parser-gpt_oss": (
        dict(
            use_harmony=True,
            model_type="gpt_oss",
            tool_parser=None,
            reasoning_parser="gpt_oss",
            parser_aliases=(),
        ),
        None,
    ),
    "generic+alias": (
        dict(
            use_harmony=False,
            model_type="llama",
            tool_parser=None,
            reasoning_parser=None,
            parser_aliases=("gpt_oss",),
        ),
        "not one the alias is registered for",
    ),
    "gpt-oss+alias+DISABLE_HARMONY_ADAPTER": (
        dict(
            use_harmony=False,
            model_type="gpt_oss",
            tool_parser=None,
            reasoning_parser=None,
            parser_aliases=("gpt_oss",),
        ),
        "DISABLE_HARMONY_ADAPTER=1",
    ),
    "generic+qwen3": (
        dict(
            use_harmony=False,
            model_type="llama",
            tool_parser="qwen3",
            reasoning_parser="qwen3",
            parser_aliases=(),
        ),
        None,
    ),
    "gpt-oss+DISABLE_HARMONY_ADAPTER+qwen3": (
        dict(
            use_harmony=False,
            model_type="gpt_oss",
            tool_parser="qwen3",
            reasoning_parser=None,
            parser_aliases=(),
        ),
        None,
    ),
}


@pytest.mark.parametrize("case", sorted(STARTUP_CHECK_CASES), ids=sorted(STARTUP_CHECK_CASES))
def test_parser_options_startup_check(case):
    kwargs, expected_error = STARTUP_CHECK_CASES[case]
    check = OpenAIServer._ensure_harmony_parser_options_supported
    if expected_error is None:
        check(**kwargs)
        return
    with pytest.raises(ValueError, match=".*") as info:
        check(**kwargs)
    assert expected_error in str(info.value), (
        f"startup check: {case} should fail naming {expected_error!r}, got {info.value}"
    )


def test_startup_check_runs_at_construction(serve_env):
    with pytest.raises(ValueError, match="--tool_parser 'qwen3'"):
        serve_env.build_openai_server(AggConfig(model_type="gpt_oss", tool_parser="qwen3"))
    with pytest.raises(ValueError, match="not one the alias is registered for"):
        serve_env.build_openai_server(AggConfig(model_type="llama", parser_aliases=("gpt_oss",)))
    # The alias and a config-file reasoning parser equal to it start.
    server = serve_env.build_openai_server(
        AggConfig(model_type="gpt_oss", reasoning_parser="gpt_oss", parser_aliases=("gpt_oss",))
    )
    assert server.use_harmony is True
    # Another model keeps its text parsers; no startup check applies.
    generic = serve_env.build_openai_server(
        AggConfig(model_type="llama", tool_parser="qwen3", reasoning_parser="qwen3")
    )
    assert generic.tool_parser == "qwen3"


def test_post_processor_hook_guard_follows_output_mode():
    guard = OpenAIServer._ensure_post_processor_hook_supported
    with pytest.raises(ValueError, match="not supported with harmony"):
        guard(OutputMode.HARMONY_TOKENS, "my_pkg.guardrail.Hook")
    guard(OutputMode.TEXT, "my_pkg.guardrail.Hook")
    guard(OutputMode.HARMONY_TOKENS, None)
    guard(OutputMode.TEXT, None)


# ---------------------------------------------------------------------------
# In-process callers: /health_generate; the KV-cache time output
# ---------------------------------------------------------------------------


def _health_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        model=MODEL,
        max_completion_tokens=1,
        stream=False,
        temperature=0.0,
    )


def test_health_generate_renders_with_harmony_by_default(serve_env, harmony_vocab):
    server = _gpt_oss_server(serve_env, num_postprocess_workers=1)
    server.generator.script(_harmony_output())
    response = serve_env.client(server.app).get("/health_generate")
    expect = "the default /health_generate on gpt-oss must render with harmony and use the harmony post-processor"

    assert (response.status_code, response.text) == (200, "Generation health check OK"), expect
    call = server.generator.generate_calls[0]
    assert _prompt_token_ids_of(server) == tokenize_harmony_chat_request(_health_request()), expect
    assert call.sampling_params.detokenize is False, expect
    assert call.sampling_params.max_tokens == 1
    assert set(get_harmony_adapter().get_stop_tokens()) <= set(call.sampling_params.stop_token_ids)
    assert call.kwargs["_postproc_params"].post_processor is chat_harmony_post_processor, expect


def test_health_generate_under_the_switch_matches_the_text_path(serve_env):
    server = _gpt_oss_server(serve_env, env=((SWITCH, "1"),), num_postprocess_workers=1)
    response = serve_env.client(server.app).get("/health_generate")
    expect = "TRTLLM_GPT_OSS_LEGACY_CHAT=1 must restore /health_generate too"

    assert (response.status_code, response.text) == (200, "Generation health check OK"), expect
    call = server.generator.generate_calls[0]
    rendered = server.generator.preprocess_calls[0].inputs
    assert isinstance(rendered, dict) and rendered.get("prompt") == "<|user|>hi<|assistant|>", (
        f"{expect}: the chat template renders the prompt, got {rendered!r}"
    )
    assert call.sampling_params.detokenize is True, expect
    assert call.kwargs["_postproc_params"].post_processor is chat_response_post_processor, expect


def test_kvcache_time_output_requests_metrics_for_gpt_oss(serve_env, harmony_vocab, tmp_path):
    server = _gpt_oss_server(
        serve_env, env=(("TRTLLM_KVCACHE_TIME_OUTPUT_PATH", str(tmp_path / "kv")),)
    )
    server.generator.script(_harmony_output())
    status, _ = _answer(serve_env.client(server.app).post(CHAT_ROUTE, json=_chat_body()))
    assert status == 200
    assert server.generator.generate_calls[0].sampling_params.return_perf_metrics is True, (
        "TRTLLM_KVCACHE_TIME_OUTPUT_PATH is set but gpt-oss chat did not request per-request "
        "perf metrics; the chat handler must apply the check to every model"
    )


def test_kvcache_time_output_unset_requests_no_metrics_for_gpt_oss(serve_env, harmony_vocab):
    server = _gpt_oss_server(serve_env)
    server.generator.script(_harmony_output())
    status, _ = _answer(serve_env.client(server.app).post(CHAT_ROUTE, json=_chat_body()))
    assert status == 200
    assert server.generator.generate_calls[0].sampling_params.return_perf_metrics is False


# ---------------------------------------------------------------------------
# The hooks seen from the chat handler
# ---------------------------------------------------------------------------


def test_harmony_hooks_follow_the_switch(serve_env):
    from tensorrt_llm.serve.extensions.gpt_oss import GptOssServingExtension
    from tensorrt_llm.serve.render import RenderResources

    with_harmony = RenderResources.from_server(_gpt_oss_server(serve_env))
    assert isinstance(with_harmony.harmony_hooks, GptOssServingExtension)
    assert with_harmony.harmony_hooks.output_mode() is OutputMode.HARMONY_TOKENS

    # The same model type without the switch: the extension lookup is unchanged
    # (the fingerprint keeps the class), the hooks answer as the generic path.
    off = RenderResources.from_server(
        SimpleNamespace(model_config=with_harmony.hf_config, use_harmony=False)
    )
    assert isinstance(off.extension, GptOssServingExtension)
    assert off.harmony_hooks.output_mode() is OutputMode.TEXT
    assert off.harmony_hooks.extra_stop_token_ids() == []
    assert off.harmony_hooks.unsupported_features() == frozenset()


# ---------------------------------------------------------------------------
# The CLI: the gpt_oss alias is split off before the parsers and the engine see it
# ---------------------------------------------------------------------------


class TestCliAliasWiring:
    """``--tool_parser gpt_oss`` / ``--reasoning_parser gpt_oss`` reach the server as aliases."""

    @staticmethod
    def _launch_server_call(args):
        # As the serve CLI wiring tests do: no model, no GPU, no server; the
        # arguments launch_server receives are what the CLI resolved.
        with (
            mock.patch(
                "tensorrt_llm.commands.serve.get_is_diffusion_only_model", return_value=False
            ),
            mock.patch("tensorrt_llm.commands.serve.device_count", return_value=1),
            mock.patch("tensorrt_llm.commands.serve.launch_server") as launch_server,
        ):
            serve_main(args=["dummy/model", *args], standalone_mode=False)
        assert launch_server.call_count == 1
        call = launch_server.call_args
        # launch_server(host, port, llm_args, tool_parser, ...); llm_args carries
        # reasoning_parser only when one is set.
        return call.args[2].get("reasoning_parser"), call.args[3], call.kwargs["parser_aliases"]

    @pytest.mark.parametrize(
        ("args", "expected"),
        [
            pytest.param([], (None, None, ()), id="no-parser-options"),
            pytest.param(["--tool_parser", "gpt_oss"], (None, None, ("gpt_oss",)), id="tool-alias"),
            pytest.param(
                ["--reasoning_parser", "gpt_oss"], (None, None, ("gpt_oss",)), id="reasoning-alias"
            ),
            pytest.param(
                ["--tool_parser", "gpt_oss", "--reasoning_parser", "gpt_oss"],
                (None, None, ("gpt_oss",)),
                id="both-aliases-deduplicated",
            ),
            pytest.param(
                ["--tool_parser", "qwen3", "--reasoning_parser", "qwen3"],
                ("qwen3", "qwen3", ()),
                id="parser-names-unchanged",
            ),
        ],
    )
    def test_alias_split(self, args, expected):
        reasoning_parser, tool_parser, parser_aliases = self._launch_server_call(args)
        assert (reasoning_parser, tool_parser, parser_aliases) == expected, (
            "the alias must not reach LlmArgs.reasoning_parser or the tool_parser argument; "
            "it reaches the server as parser_aliases"
        )

    @pytest.mark.parametrize("option", ["--tool_parser", "--reasoning_parser"])
    def test_grpc_rejects_the_alias_like_the_other_http_only_options(self, option):
        with (
            mock.patch(
                "tensorrt_llm.commands.serve.get_is_diffusion_only_model", return_value=False
            ),
            mock.patch("tensorrt_llm.commands.serve.device_count", return_value=1),
            mock.patch("tensorrt_llm.commands.serve.launch_server") as launch_server,
            pytest.raises(
                ValueError,
                match="Argument 'parser alias' is not supported when running in gRPC mode",
            ),
        ):
            serve_main(args=["dummy/model", "--grpc", option, "gpt_oss"], standalone_mode=False)
        launch_server.assert_not_called()
