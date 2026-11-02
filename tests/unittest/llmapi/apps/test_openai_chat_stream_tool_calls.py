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
"""Tool calls in a streamed chat completion: one entry per index per chunk.

OpenAI never repeats a tool call's ``index`` within one chunk's
``delta.tool_calls``, and clients rely on it: the openai-python stream helper
behind ``client.chat.completions.stream()`` takes the first ``tool_calls`` list
it receives as the calls themselves (releases before 3.14 do). Several tool
parsers report a call's name and its arguments as separate items of one
streaming step; the stream still has to send each call as one entry.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest, ChatCompletionToolsParam
from tensorrt_llm.serve.postprocess_handlers import ChatPostprocArgs, chat_stream_post_processor
from tensorrt_llm.serve.tool_parser.tool_parser_factory import ToolParserFactory

from .openai_server_stub import make_stub_openai_server

pytestmark = pytest.mark.cpu_only

MODEL = "test-model"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    },
]
WEATHER = ("get_weather", {"location": "NYC"})
SEARCH = ("search_web", {"query": "AI"})


def _dsml_calls(prefix: str, block: str, *calls: tuple[str, dict]) -> str:
    """A DeepSeek DSML tool-call block with string parameters."""
    invokes = [
        f'<{prefix}invoke name="{name}">\n'
        + "".join(
            f'<{prefix}parameter name="{key}" string="true">{value}</{prefix}parameter>\n'
            for key, value in arguments.items()
        )
        + f"</{prefix}invoke>"
        for name, arguments in calls
    ]
    return f"<{prefix}{block}>\n" + "\n".join(invokes) + f"\n</{prefix}{block}>"


def _dsml_v41_calls(*calls: tuple[str, dict]) -> str:
    return _dsml_calls("｜DSML｜ ", "calls", *calls)


def _generation_step(text: str, step: int, last: bool) -> SimpleNamespace:
    """One streamed engine result, its single output carrying ``text``."""
    output = SimpleNamespace(
        index=0,
        text_diff=text,
        token_ids_diff=[step],
        logprobs_diff=[],
        finish_reason="stop" if last else None,
        stop_reason=None,
        length=step + 1,
        disaggregated_params=None,
    )
    return SimpleNamespace(
        outputs=[output], cached_tokens=0, id="request", _done=last, finished=last
    )


class _GenerationStream:
    """What ``generate_async`` returns: one engine result per text increment."""

    def __init__(self, increments: list[str]) -> None:
        self.prompt_token_ids = [1, 2, 3]
        last = len(increments) - 1
        self._steps = iter(
            [_generation_step(text, step, step == last) for step, text in enumerate(increments)]
        )

    def __aiter__(self) -> "_GenerationStream":
        return self

    async def __anext__(self) -> SimpleNamespace:
        try:
            return next(self._steps)
        except StopIteration:
            raise StopAsyncIteration from None


class _StubModelConfig:
    model_type = "llama"
    vocab_size = 1024


class _StubTokenizer:
    """Renders every conversation to one fixed prompt."""

    def __init__(self) -> None:
        self.tokenizer = self
        self.vocab_size = _StubModelConfig.vocab_size
        self.name_or_path = None

    def apply_chat_template(self, *args, **kwargs) -> str:
        return "<prompt>"

    def encode(self, text: str, **kwargs) -> list[int]:
        return [1, 2, 3]


def _stream_chat_route(tmp_path, increments: list[str]) -> list[dict]:
    """POST a streaming request to a ``deepseek_v41`` server; return its chunks."""
    server = make_stub_openai_server(
        tmp_path,
        MODEL,
        _StubModelConfig(),
        lambda **kwargs: _GenerationStream(increments),
        tokenizer=_StubTokenizer(),
        tool_parser="deepseek_v41",
    )
    server.await_disconnected = AsyncMock()
    app = FastAPI()
    app.add_api_route("/v1/chat/completions", server.openai_chat, methods=["POST"])
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "What is the weather in NYC?"}],
        "tools": TOOLS,
        "stream": True,
    }
    response = TestClient(app).post("/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text
    events = [
        line[len("data: ") :] for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert events[-1] == "[DONE]"
    return [json.loads(event) for event in events[:-1]]


def _stream_post_processor(tool_parser: str, increments: list[str]) -> list[dict]:
    """Feed text increments through ``chat_stream_post_processor``; return its chunks."""
    request = ChatCompletionRequest(
        model=MODEL, messages=[{"role": "user", "content": "q"}], tools=TOOLS, stream=True
    )
    args = ChatPostprocArgs.from_request(request)
    args.tool_parser = tool_parser
    args.num_prompt_tokens = 3
    chunks = []
    last = len(increments) - 1
    for step, text in enumerate(increments):
        for line in chat_stream_post_processor(_generation_step(text, step, step == last), args):
            chunks.append(json.loads(line[len("data: ") :]))
    return chunks


def _tool_call_entries(chunks: list[dict]) -> list[dict]:
    """All ``delta.tool_calls`` entries in order; fails if a chunk repeats an index."""
    entries = []
    for chunk in chunks:
        for choice in chunk["choices"]:
            tool_calls = choice["delta"].get("tool_calls") or []
            indices = [entry["index"] for entry in tool_calls]
            assert len(indices) == len(set(indices)), f"one chunk repeats an index: {indices}"
            entries += tool_calls
    return entries


@pytest.mark.parametrize("calls", [[WEATHER], [WEATHER, SEARCH]], ids=["one_call", "two_calls"])
def test_dsml_block_in_one_step_streams_one_entry_per_call(tmp_path, calls):
    """A whole DeepSeek-V4.1 call block in one step: each call is one complete entry."""
    chunks = _stream_chat_route(tmp_path, ["Let me check.", _dsml_v41_calls(*calls)])

    entries = _tool_call_entries(chunks)
    assert [
        (entry["index"], entry["function"]["name"], json.loads(entry["function"]["arguments"]))
        for entry in entries
    ] == [(index, name, arguments) for index, (name, arguments) in enumerate(calls)]
    assert all(entry["id"] and entry["type"] == "function" for entry in entries)
    content = "".join(
        choice["delta"].get("content") or "" for chunk in chunks for choice in chunk["choices"]
    )
    assert content == "Let me check."
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.parametrize("calls", [[WEATHER], [WEATHER, SEARCH]], ids=["one_call", "two_calls"])
def test_openai_sdk_stream_helper_assembles_the_calls(tmp_path, calls):
    """The openai-python stream state ends with exactly the calls the model made."""
    streaming = pytest.importorskip("openai.lib.streaming.chat")
    from openai.types.chat import ChatCompletionChunk

    state = streaming.ChatCompletionStreamState()
    for chunk in _stream_chat_route(tmp_path, ["Let me check.", _dsml_v41_calls(*calls)]):
        state.handle_chunk(ChatCompletionChunk.model_validate(chunk))

    message = state.get_final_completion().choices[0].message
    assert message.content == "Let me check."
    assert len(message.tool_calls) == len(calls)
    assert [
        (tool_call.function.name, json.loads(tool_call.function.arguments))
        for tool_call in message.tool_calls
    ] == calls


@pytest.mark.parametrize(
    "tool_parser, text",
    [
        ("deepseek_v32", _dsml_calls("｜DSML｜", "function_calls", WEATHER)),
        ("deepseek_v4", _dsml_calls("｜DSML｜", "tool_calls", WEATHER)),
        (
            "poolside_v1",
            "<tool_call>get_weather\n<arg_key>location</arg_key>\n"
            "<arg_value>NYC</arg_value>\n</tool_call>",
        ),
        ("qwen3", json.dumps({"name": "get_weather", "arguments": {"location": "NYC"}})),
        (
            "qwen3_coder",
            "<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>\n"
            "</function>\n</tool_call>",
        ),
    ],
    ids=["deepseek_v32", "deepseek_v4", "poolside_v1", "qwen3_bare_json", "qwen3_coder"],
)
def test_parsers_reporting_name_and_arguments_in_one_step(tool_parser, text):
    """These parsers report one call as several items of a step; it streams as one entry."""
    entries = _tool_call_entries(_stream_post_processor(tool_parser, [text]))

    assert [
        (entry["index"], entry["function"]["name"], json.loads(entry["function"]["arguments"]))
        for entry in entries
    ] == [(0, *WEATHER)]


@pytest.mark.parametrize(
    "tool_parser, text",
    [
        (
            "qwen3",
            '<tool_call>\n{"name": "get_weather", "arguments": {"location": "NYC"}}\n</tool_call>',
        ),
        (
            "deepseek_v31",
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>"
            '{"location": "NYC"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
        ),
    ],
    ids=["qwen3", "deepseek_v31"],
)
def test_name_and_arguments_in_separate_steps_stream_as_parsed(tool_parser, text):
    """When no step repeats an index, every chunk carries the parser's items as they are."""
    increments = [text[start : start + 8] for start in range(0, len(text), 8)]
    parser = ToolParserFactory.create_tool_parser(tool_parser)
    tools = [ChatCompletionToolsParam.model_validate(tool) for tool in TOOLS]
    expected = []
    for step, increment in enumerate(increments):
        items = parser.parse_streaming_increment(increment, tools).calls
        if step == len(increments) - 1:
            items += parser.finish(tools).calls
        expected.append([(item.tool_index, item.name, item.parameters) for item in items])
    # The parser names the call in one step and streams its arguments in later ones.
    named_steps = [step for step, items in enumerate(expected) if any(item[1] for item in items)]
    argument_steps = [
        step for step, items in enumerate(expected) if any(not item[1] for item in items)
    ]
    assert named_steps and argument_steps and max(named_steps) < min(argument_steps)

    chunks = _stream_post_processor(tool_parser, increments)

    # The first chunk announces the role; each step then sends one chunk.
    assert len(chunks) == 1 + len(increments)
    assert [
        [
            (entry["index"], entry["function"].get("name"), entry["function"]["arguments"])
            for entry in chunk["choices"][0]["delta"].get("tool_calls") or []
        ]
        for chunk in chunks[1:]
    ] == expected
