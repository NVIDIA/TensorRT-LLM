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
"""ChatCompletionRequest accepts every OpenAI reasoning effort, and harmony maps each one.

OpenAI's reasoning efforts are none, minimal, low, medium, high, xhigh and max. Harmony
(gpt-oss) has three levels, so "minimal" takes its lowest and "xhigh" and "max" its highest,
and "none" takes the harmony adapter's default level, on the chat completions and the
responses routes alike.
"""

import typing
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from openai_harmony import ReasoningEffort, Role
from pydantic import ValidationError

from tensorrt_llm.serve.chat_tokenization import tokenize_harmony_chat_request
from tensorrt_llm.serve.harmony_adapter import get_harmony_adapter, maybe_transform_reasoning_effort
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest, ResponsesRequest
from tensorrt_llm.serve.responses_utils import _construct_harmony_messages, _render_for_completion

from .openai_server_stub import make_stub_openai_server

pytestmark = pytest.mark.cpu_only

OPENAI_EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]


def _request(**fields) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], **fields)


def _schema_effort_names() -> list[str]:
    """The effort strings the request schema accepts, read from the field's annotation."""
    names: list[str] = []
    pending = [ChatCompletionRequest.model_fields["reasoning_effort"].annotation]
    while pending:
        annotation = pending.pop()
        if typing.get_origin(annotation) is typing.Literal:
            names.extend(typing.get_args(annotation))
        else:
            pending.extend(typing.get_args(annotation))
    return names


@pytest.mark.parametrize("effort", OPENAI_EFFORTS)
def test_every_openai_reasoning_effort_validates(effort):
    assert _request(reasoning_effort=effort).reasoning_effort == effort


@pytest.mark.parametrize(
    "name, level",
    [
        ("Low", ReasoningEffort.LOW),
        ("Medium", ReasoningEffort.MEDIUM),
        ("High", ReasoningEffort.HIGH),
    ],
)
def test_harmony_level_names_still_validate(name, level):
    assert _request(reasoning_effort=name).reasoning_effort is level


@pytest.mark.parametrize("effort", ["turbo", "extra-high"])
def test_an_unknown_reasoning_effort_is_rejected(effort):
    with pytest.raises(ValidationError) as caught:
        _request(reasoning_effort=effort)
    assert {error["loc"][0] for error in caught.value.errors()} == {"reasoning_effort"}


class _StubModelConfig:
    model_type = "llama"
    vocab_size = 64


class _StubTokenizer:
    """Renders every conversation to one fixed prompt."""

    def __init__(self):
        self.tokenizer = self
        self.vocab_size = _StubModelConfig.vocab_size
        self.name_or_path = None

    def apply_chat_template(self, *args, **kwargs) -> str:
        return "prompt"

    def encode(self, text, **kwargs):
        return [1, 2, 3]


@pytest.mark.parametrize("effort, status", [("minimal", 200), ("xhigh", 200), ("turbo", 400)])
def test_chat_route_rejects_only_an_unknown_effort(tmp_path, effort, status):
    submitted = []

    def generate_async(*, inputs, **kwargs):
        submitted.append(inputs)
        return SimpleNamespace(prompt_token_ids=[1, 2, 3], request_id=1, finished=True)

    server = make_stub_openai_server(
        tmp_path, "m", _StubModelConfig(), generate_async, tokenizer=_StubTokenizer()
    )
    server.await_disconnected = AsyncMock()
    server._create_chat_response = AsyncMock(
        return_value=SimpleNamespace(model_dump=lambda: {"ok": True})
    )
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": effort,
    }

    response = TestClient(server.app).post("/v1/chat/completions", json=body)

    assert response.status_code == status, response.text
    if status == 400:
        assert "reasoning_effort" in response.json()["error"]
        assert submitted == []
    else:
        assert len(submitted) == 1, "an accepted effort must reach the engine"


@pytest.mark.parametrize(
    "effort, level",
    [
        ("minimal", ReasoningEffort.LOW),
        ("low", ReasoningEffort.LOW),
        ("medium", ReasoningEffort.MEDIUM),
        ("high", ReasoningEffort.HIGH),
        ("xhigh", ReasoningEffort.HIGH),
        ("max", ReasoningEffort.HIGH),
        ("none", None),
        (ReasoningEffort.MEDIUM, ReasoningEffort.MEDIUM),
        (None, None),
    ],
)
def test_harmony_maps_each_effort_to_a_harmony_level(effort, level):
    assert maybe_transform_reasoning_effort(effort) is level


def test_every_effort_the_schema_accepts_has_a_harmony_level():
    names = _schema_effort_names()
    assert names, "the reasoning_effort annotation lists its accepted strings"
    for name in names:
        level = maybe_transform_reasoning_effort(name)
        assert level is None or isinstance(level, ReasoningEffort), name


class _RecordingHarmonyAdapter:
    """Records the effort the chat request hands to harmony prompt building."""

    def __init__(self):
        self.reasoning_effort = "not called"

    def openai_to_harmony_tokens(self, messages, tools, reasoning_effort=None, tool_choice=None):
        self.reasoning_effort = reasoning_effort
        return [1, 2, 3]


@pytest.mark.parametrize(
    "effort, level", [("minimal", ReasoningEffort.LOW), ("xhigh", ReasoningEffort.HIGH)]
)
def test_harmony_chat_prompt_gets_the_mapped_level(effort, level):
    adapter = _RecordingHarmonyAdapter()

    tokenize_harmony_chat_request(_request(reasoning_effort=effort), harmony_adapter=adapter)

    assert adapter.reasoning_effort is level


def _responses_request(effort) -> ResponsesRequest:
    return ResponsesRequest.model_validate(
        {"model": "m", "input": "hi", "reasoning": {"effort": effort}}
    )


def _responses_system_effort(effort) -> ReasoningEffort | None:
    system_message = _construct_harmony_messages(_responses_request(effort), prev_response=None)[0]
    assert system_message.author.role == Role.SYSTEM
    return system_message.content[0].reasoning_effort


@pytest.mark.parametrize(
    "effort, level",
    [
        ("minimal", ReasoningEffort.LOW),
        ("xhigh", ReasoningEffort.HIGH),
        ("max", ReasoningEffort.HIGH),
    ],
)
def test_responses_harmony_system_message_gets_the_mapped_level(effort, level):
    assert _responses_system_effort(effort) is level


def test_responses_none_effort_takes_the_harmony_adapter_default(harmony_vocab):
    assert _responses_system_effort("none") is get_harmony_adapter().default_reasoning_effort


def _reasoning_lines(token_ids: list[int]) -> list[str]:
    prompt = get_harmony_adapter().encoding.decode(token_ids)
    return [line for line in prompt.split("\n") if line.startswith("Reasoning:")]


@pytest.mark.parametrize("effort", OPENAI_EFFORTS)
def test_chat_and_responses_give_harmony_the_same_effort(harmony_vocab, effort):
    chat_lines = _reasoning_lines(
        tokenize_harmony_chat_request(
            _request(reasoning_effort=effort), harmony_adapter=get_harmony_adapter()
        )
    )
    responses_lines = _reasoning_lines(
        _render_for_completion(
            _construct_harmony_messages(_responses_request(effort), prev_response=None)
        )
    )

    assert len(chat_lines) == 1, chat_lines
    assert responses_lines == chat_lines
