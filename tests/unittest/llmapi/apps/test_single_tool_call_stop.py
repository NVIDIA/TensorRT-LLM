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
"""A request for at most one tool call per turn stops generation after the first call.

``parallel_tool_calls=false`` (Chat Completions, Responses) and Anthropic's
``disable_parallel_tool_use`` are honoured with a stop on the text that ends one
call (``BaseToolParser.single_call_stops``), kept in the output so the tool
parser still reads that call. These tests cover where the stop is configured,
what a client sees of it, and that it matches the tokens the model writes.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from openai.types.responses.tool import FunctionTool
from utils.llm_data import llm_models_root

from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.serve import openai_server as openai_server_module
from tensorrt_llm.serve.anthropic_protocol import AnthropicMessagesRequest
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionToolsParam,
    ChatMessage,
    FunctionDefinition,
    ResponsesRequest,
    UsageInfo,
)
from tensorrt_llm.serve.openai_server import (
    OpenAIServer,
    _single_tool_call_stops,
    _stop_after_first_tool_call,
)
from tensorrt_llm.serve.postprocess_handlers import (
    ChatPostprocArgs,
    chat_response_post_processor,
    chat_stream_post_processor,
)
from tensorrt_llm.serve.render import chat as render_chat_module
from tensorrt_llm.serve.responses_utils import ConversationHistoryStore

pytestmark = pytest.mark.cpu_only

MODEL = "test-model"
V41_STOPS = ("</｜DSML｜ invoke>\n", "</｜DSML｜ invoke>")
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
    },
}
WEATHER_INVOKE = (
    '<｜DSML｜ invoke name="get_weather">\n'
    '<｜DSML｜ parameter name="location" string="true">NYC</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>"
)


# ---------------------------------------------------------------------------
# The stop and the request's own stops
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool_parser, stops",
    [
        ("deepseek_v41", V41_STOPS),
        ("DeepSeek_V41", V41_STOPS),
        ("deepseek_v4", ("</｜DSML｜invoke>\n", "</｜DSML｜invoke>")),
        ("deepseek_v32", ("</｜DSML｜invoke>\n", "</｜DSML｜invoke>")),
        ("qwen3", ()),
        ("not_a_parser", ()),
        (None, ()),
    ],
)
def test_the_stops_come_from_the_tool_parser(tool_parser, stops):
    assert _single_tool_call_stops(tool_parser) == stops


def test_the_stops_are_added_and_kept_in_the_output():
    sampling_params = SamplingParams()
    assert _stop_after_first_tool_call(sampling_params, V41_STOPS)
    assert sampling_params.stop == list(V41_STOPS)
    assert sampling_params.include_stop_str_in_output is True


@pytest.mark.parametrize(
    "fields",
    [{"stop": "END"}, {"stop": ["END"]}, {"stop_token_ids": [7]}],
    ids=["stop_string", "stop_list", "stop_id"],
)
def test_nothing_changes_when_the_request_drops_its_own_stops_from_the_output(fields):
    """Keeping stop text applies to every stop, so the client's would show."""
    sampling_params = SamplingParams(**fields)
    assert not _stop_after_first_tool_call(sampling_params, V41_STOPS)
    assert sampling_params.stop == fields.get("stop")
    assert sampling_params.include_stop_str_in_output is False


def test_the_stops_follow_the_request_stops_it_keeps_in_the_output():
    sampling_params = SamplingParams(stop=["END", V41_STOPS[0]], include_stop_str_in_output=True)
    assert _stop_after_first_tool_call(sampling_params, V41_STOPS)
    assert sampling_params.stop == ["END", *V41_STOPS]


@pytest.mark.parametrize(
    "model_name, tool_parser",
    [
        ("DeepSeek-V3.2-hf", "deepseek_v32"),
        ("DeepSeek-V4-Flash", "deepseek_v4"),
        ("DeepSeek-V4-Pro", "deepseek_v4"),
        ("DeepSeek-V4.1-Flash", "deepseek_v41"),
    ],
)
def test_the_stop_matches_the_tokens_the_model_writes(model_name, tool_parser):
    """Stops match token ids, and '>' plus the newline after it are one token.

    So in each encoder's layout of DeepSeek tool calls the stop with the
    newline matches once per call, and the bare closing tag never does.
    """
    root = llm_models_root()
    tokenizer_file = None if root is None else root / model_name / "tokenizer.json"
    if tokenizer_file is None or not tokenizer_file.is_file():
        pytest.skip(f"{model_name} is not under LLM_MODELS_ROOT")
    from transformers import PreTrainedTokenizerFast

    from tensorrt_llm.tokenizer.deepseek_v4 import DeepseekV4Tokenizer
    from tensorrt_llm.tokenizer.deepseek_v32 import DeepseekV32Tokenizer
    from tensorrt_llm.tokenizer.deepseek_v41 import DeepseekV41Tokenizer

    tokenizer_class = {
        "deepseek_v32": DeepseekV32Tokenizer,
        "deepseek_v4": DeepseekV4Tokenizer,
        "deepseek_v41": DeepseekV41Tokenizer,
    }[tool_parser]

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
    # The ids the engine matches: SamplingParams tokenizes its stop strings.
    stops = _single_tool_call_stops(tool_parser)
    sampling_params = SamplingParams(stop=list(stops))
    sampling_params._setup(tokenizer, None, None)
    with_newline, bare = sampling_params._get_stop_words()

    calls = [
        {
            "id": f"call_{index}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }
        for index, (name, arguments) in enumerate([("exec", {"cmd": "ls"}), ("wait", {"s": 2})])
    ]
    prompt = tokenizer_class(tokenizer).apply_chat_template(
        [
            {"role": "user", "content": "List the files, then wait."},
            {"role": "assistant", "content": "", "tool_calls": calls},
        ],
        thinking=False,
        tokenize=False,
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)

    def matches(word):
        return sum(ids[i : i + len(word)] == word for i in range(len(ids)))

    assert prompt.count(stops[0]) == 2
    assert matches(with_newline) == 2
    assert matches(bare) == 0


# ---------------------------------------------------------------------------
# Chat Completions and Anthropic Messages
# ---------------------------------------------------------------------------


class _ModelConfig:
    model_type = "llama"
    vocab_size = 1024


def _chat_server(monkeypatch, tool_parser):
    """A server whose real ``openai_chat`` records what reaches the engine."""

    async def render(**kwargs):
        return "rendered prompt"

    monkeypatch.setattr(render_chat_module, "async_apply_chat_template", render)
    seen = {}

    def generate_async(*, inputs, **kwargs):
        seen["sampling_params"] = kwargs["sampling_params"]
        return SimpleNamespace(prompt_token_ids=[1, 2, 3], finished=True)

    async def create_chat_response(promise, postproc_params, raw_request, disaggregated_params):
        seen["postproc_args"] = postproc_params.postproc_args
        return ChatCompletionResponse(
            model=MODEL,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="ok"),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=3, completion_tokens=1, total_tokens=4),
        )

    server = object.__new__(OpenAIServer)
    server.model = MODEL
    server.allow_request_chat_template = False
    server.model_config = _ModelConfig()
    server.processor = None
    server.tokenizer = SimpleNamespace(tokenizer=SimpleNamespace(vocab_size=1024))
    server.chat_template = None
    server.tool_parser = tool_parser
    server.tool_call_id_type = "random"
    server.multimodal_server_config = None
    server._input_proc_executor = None
    server._is_visual_gen = False
    server.generator = SimpleNamespace(
        args=SimpleNamespace(
            gather_generation_logits=False,
            reasoning_parser=None,
            backend="pytorch",
            guided_decoding_backend=None,
            num_postprocess_workers=0,
        ),
        generate_async=generate_async,
    )
    server.await_disconnected = AsyncMock()
    server._create_chat_response = create_chat_response
    return server, seen


def _raw_request():
    return SimpleNamespace(state=SimpleNamespace(), headers={})


@pytest.fixture
def server_logger(monkeypatch):
    log = Mock()
    monkeypatch.setattr(openai_server_module, "logger", log)
    return log


def _warned(log, key):
    return key in [call.kwargs.get("key") for call in log.warning_once.call_args_list]


@pytest.mark.parametrize(
    "fields, tool_parser, stop, enforced",
    [
        ({"parallel_tool_calls": False}, "deepseek_v41", list(V41_STOPS), True),
        ({}, "deepseek_v41", [], False),
        ({"parallel_tool_calls": True}, "deepseek_v41", [], False),
        ({"parallel_tool_calls": False, "tool_choice": "none"}, "deepseek_v41", [], False),
        ({"parallel_tool_calls": False, "tools": None}, "deepseek_v41", [], False),
        ({"parallel_tool_calls": False}, "qwen3", [], False),
        ({"parallel_tool_calls": False, "stop": ["END"]}, "deepseek_v41", ["END"], False),
        (
            {"parallel_tool_calls": False, "stop": ["END"], "include_stop_str_in_output": True},
            "deepseek_v41",
            ["END", *V41_STOPS],
            True,
        ),
    ],
    ids=[
        "asked",
        "omitted",
        "parallel_allowed",
        "tool_choice_none",
        "no_tools",
        "format_without_stop",
        "own_stops_dropped",
        "own_stops_kept",
    ],
)
def test_chat_configures_the_stop_only_when_asked_and_possible(
    monkeypatch, server_logger, fields, tool_parser, stop, enforced
):
    server, seen = _chat_server(monkeypatch, tool_parser)
    request = ChatCompletionRequest(
        **{
            "model": MODEL,
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [WEATHER_TOOL],
            **fields,
        }
    )
    response = asyncio.run(server.openai_chat(request, _raw_request()))

    assert response.status_code == 200, response.body
    sampling_params = seen["sampling_params"]
    assert sampling_params.stop == stop
    assert sampling_params.include_stop_str_in_output is (
        "include_stop_str_in_output" in fields or enforced
    )
    assert seen["postproc_args"].single_call_stops == (V41_STOPS if enforced else ())
    asked = fields.get("parallel_tool_calls") is False and fields.get("tools", [WEATHER_TOOL])
    asked = asked and fields.get("tool_choice") != "none"
    assert _warned(server_logger, "chat_parallel_tool_calls_unenforced") is bool(
        asked and not enforced
    )


def test_a_stop_the_request_lists_itself_keeps_its_stop_reason(monkeypatch, server_logger):
    """Only the stops the server adds are hidden from the reply's stop reason."""
    server, seen = _chat_server(monkeypatch, "deepseek_v41")
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": "weather?"}],
        tools=[WEATHER_TOOL],
        parallel_tool_calls=False,
        stop=[V41_STOPS[0]],
        include_stop_str_in_output=True,
    )
    response = asyncio.run(server.openai_chat(request, _raw_request()))

    assert response.status_code == 200, response.body
    assert seen["sampling_params"].stop == list(V41_STOPS)
    assert seen["postproc_args"].single_call_stops == V41_STOPS[1:]


@pytest.mark.parametrize(
    "tool_parser, stop_sequences, enforced",
    [
        ("deepseek_v41", [], True),
        ("deepseek_v41", ["END"], False),
        ("qwen3", [], False),
        (None, [], False),
    ],
    ids=["supported", "own_stops_dropped", "format_without_stop", "no_tool_parser"],
)
def test_anthropic_disable_parallel_tool_use_configures_the_stop_when_possible(
    monkeypatch, server_logger, tool_parser, stop_sequences, enforced
):
    server, seen = _chat_server(monkeypatch, tool_parser)
    request = AnthropicMessagesRequest(
        model=MODEL,
        max_tokens=64,
        messages=[{"role": "user", "content": "weather?"}],
        tools=[{"name": "get_weather", "input_schema": {"type": "object"}}],
        tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        stop_sequences=stop_sequences,
    )
    response = asyncio.run(server.anthropic_messages(request, _raw_request()))

    assert response.status_code == 200, response.body
    assert seen["sampling_params"].stop == (list(V41_STOPS) if enforced else stop_sequences)
    assert seen["sampling_params"].include_stop_str_in_output is enforced
    assert seen["postproc_args"].single_call_stops == (V41_STOPS if enforced else ())
    assert _warned(server_logger, "chat_parallel_tool_calls_unenforced") is not enforced


# ---------------------------------------------------------------------------
# What a chat client receives
# ---------------------------------------------------------------------------


def _chat_args(single_call_stops=V41_STOPS):
    args = ChatPostprocArgs(
        role="assistant",
        model=MODEL,
        tools=[ChatCompletionToolsParam(function=FunctionDefinition(**WEATHER_TOOL["function"]))],
        tool_choice="auto",
        single_call_stops=single_call_stops,
    )
    args.tool_parser = "deepseek_v41"
    args.num_prompt_tokens = 3
    return args


def _output(text, finish_reason, stop_reason):
    return SimpleNamespace(
        index=0,
        text=text,
        text_diff=text,
        token_ids=[1, 2],
        token_ids_diff=[1, 2],
        length=2,
        finish_reason=finish_reason,
        stop_reason=stop_reason,
        logprobs=None,
        logprobs_diff=None,
        disaggregated_params=None,
    )


# The output a generation stopped after its first call leaves: the calls block
# is open and the kept stop ends it.
STOPPED_REPLY = "Checking.\n\n<｜DSML｜ calls>\n" + WEATHER_INVOKE + "\n"


@pytest.mark.parametrize(
    "stop_reason, reported", [(V41_STOPS[0], None), (V41_STOPS[1], None), ("END", "END")]
)
def test_a_chat_reply_reports_its_call_and_no_stop_the_server_added(stop_reason, reported):
    rsp = SimpleNamespace(
        outputs=[_output(STOPPED_REPLY, "stop", stop_reason)], prompt_token_ids=[1], cached_tokens=0
    )
    choice = chat_response_post_processor(rsp, _chat_args()).choices[0]

    assert choice.finish_reason == "tool_calls"
    assert choice.stop_reason == reported
    assert choice.message.content == "Checking."
    calls = [(c.function.name, json.loads(c.function.arguments)) for c in choice.message.tool_calls]
    assert calls == [("get_weather", {"location": "NYC"})]


def test_a_streamed_chat_reply_reports_its_call_and_no_stop_the_server_added():
    args = _chat_args()
    head, tail = STOPPED_REPLY.split("<｜DSML｜ invoke", 1)
    frames = []
    for diff, done in [(head, False), ("<｜DSML｜ invoke" + tail, True)]:
        output = _output(diff, "stop" if done else None, V41_STOPS[0] if done else None)
        rsp = SimpleNamespace(id=1, outputs=[output], cached_tokens=0, _done=done)
        frames += chat_stream_post_processor(rsp, args)

    choices = [json.loads(frame[len("data: ") :])["choices"][0] for frame in frames]
    deltas = [call for choice in choices for call in choice["delta"].get("tool_calls") or []]
    assert [d["function"]["name"] for d in deltas if d["function"].get("name")] == ["get_weather"]
    assert json.loads("".join(d["function"].get("arguments") or "" for d in deltas)) == {
        "location": "NYC"
    }
    assert choices[-1]["finish_reason"] == "tool_calls"
    assert all(choice.get("stop_reason") is None for choice in choices)
    assert "｜DSML｜" not in "".join(choice["delta"].get("content") or "" for choice in choices)


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class _EngineReached(Exception):
    pass


def _responses_server(monkeypatch, tool_parser):
    """A server whose real ``openai_responses`` records what reaches the engine."""
    seen = {}

    async def preprocess(*, request, **kwargs):
        return [1, 2, 3], request.to_sampling_params(default_sampling_params={"stop_token_ids": []})

    def generate_async(**kwargs):
        seen["sampling_params"] = kwargs["sampling_params"]
        raise _EngineReached()

    monkeypatch.setattr(openai_server_module, "responses_api_request_preprocess", preprocess)
    server = object.__new__(OpenAIServer)
    server.model = MODEL
    server.enable_store = False
    server._is_visual_gen = False
    server.use_harmony = False
    server.tokenizer = None
    server.model_config = None
    server.processor = None
    server.multimodal_server_config = None
    server.chat_template = None
    server.tool_parser = tool_parser
    server.metrics_collector = None
    server.conversation_store = ConversationHistoryStore()
    server.generator = SimpleNamespace(
        args=SimpleNamespace(reasoning_parser=None, num_postprocess_workers=0),
        generate_async=generate_async,
    )
    return server, seen


@pytest.mark.parametrize(
    "fields, tool_parser, enforced, warned",
    [
        ({"parallel_tool_calls": False}, "deepseek_v41", True, False),
        ({}, "deepseek_v41", False, False),
        ({"parallel_tool_calls": True}, "deepseek_v41", False, False),
        ({"parallel_tool_calls": False, "tool_choice": "none"}, "deepseek_v41", False, False),
        ({"parallel_tool_calls": False, "tools": []}, "deepseek_v41", False, False),
        ({"parallel_tool_calls": False}, "qwen3", False, True),
        ({"parallel_tool_calls": False}, None, False, True),
    ],
    ids=[
        "asked",
        "omitted",
        "parallel_allowed",
        "tool_choice_none",
        "no_tools",
        "format_without_stop",
        "no_tool_parser",
    ],
)
def test_responses_configures_the_stop_only_when_asked_and_possible(
    monkeypatch, server_logger, fields, tool_parser, enforced, warned
):
    server, seen = _responses_server(monkeypatch, tool_parser)
    tool = FunctionTool(
        name="get_weather",
        type="function",
        strict=False,
        parameters=WEATHER_TOOL["function"]["parameters"],
    )
    request = ResponsesRequest(**{"model": MODEL, "input": "weather?", "tools": [tool], **fields})
    raw_request = SimpleNamespace(
        state=SimpleNamespace(),
        headers={},
        url=SimpleNamespace(path="/v1/responses"),
        json=AsyncMock(return_value={}),
    )
    asyncio.run(server.openai_responses(request, raw_request))

    sampling_params = seen["sampling_params"]
    assert sampling_params.stop == (list(V41_STOPS) if enforced else None)
    assert sampling_params.include_stop_str_in_output is enforced
    assert _warned(server_logger, "responses_parallel_tool_calls_unenforced") is warned
