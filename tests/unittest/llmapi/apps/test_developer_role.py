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
"""A developer message reaches a prompt that the tokenizer renders in Python.

Clients such as Codex CLI send their instructions as a ``developer`` message. A tokenizer that
renders the conversation in its own Python code, with no Jinja template (the Kimi-K3 checkpoint
tokenizer), may have no branch for that role and leave the message out of the prompt. Such a
renderer is probed once, and is handed developer messages as system messages when it drops them.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensorrt_llm.inputs import utils as inputs_utils
from tensorrt_llm.serve import openai_server as openai_server_module
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)
from tensorrt_llm.serve.render import render_chat

from .openai_server_stub import make_stub_openai_server
from .render_helpers import KIMI_K3, chat_request, make_tokenizer, resources

pytestmark = pytest.mark.cpu_only

MODEL = "test-model"
INSTRUCTIONS = "Answer briefly."
BRIEF = "You are a coding agent. Run the tests before you finish."
QUESTION = "Fix the failing test."
USAGE = UsageInfo(prompt_tokens=3, completion_tokens=1, total_tokens=4)
PROBE = "trtllm-chat-renderer-role-probe"


class _ModelConfig:
    model_type = "llama"
    vocab_size = 1024


class _NativeRenderer:
    """A tokenizer that renders the conversation in Python and has no Jinja template.

    It renders the roles in ``ROLES`` and leaves any other message out, as the Kimi-K3 checkpoint
    tokenizer does. Tokenizing encodes one character per id. ``rendered`` records the role and
    content of every message of every conversation it renders.
    """

    ROLES = ("system", "user", "assistant", "tool")
    chat_template = None

    def __init__(self):
        self.tokenizer = self
        self.vocab_size = _ModelConfig.vocab_size
        self.name_or_path = None
        self.rendered = []
        self.template_lookups = 0

    def get_chat_template(self, chat_template=None, tools=None):
        self.template_lookups += 1
        raise ValueError("this tokenizer has no chat template")

    def _render(self, conversation):
        return "".join(
            f"<{message['role']}>{message['content']}"
            for message in conversation
            if message["role"] in self.ROLES
        )

    def apply_chat_template(
        self, conversation, tools=None, tokenize=False, add_generation_prompt=True, **kwargs
    ):
        self.rendered.append([(message["role"], message["content"]) for message in conversation])
        text = self._render(conversation) + ("<assistant>" if add_generation_prompt else "")
        return self.encode(text) if tokenize else text

    def encode(self, text, **kwargs):
        return [ord(character) for character in text]

    def probes(self):
        """The conversations rendered to probe a role."""
        return [conversation for conversation in self.rendered if conversation[0][1] == PROBE]

    def prompts(self):
        """The conversations rendered for requests."""
        return [conversation for conversation in self.rendered if conversation[0][1] != PROBE]


class _RendersDeveloper(_NativeRenderer):
    ROLES = ("system", "developer", "user", "assistant", "tool")


class _RejectsDeveloper(_NativeRenderer):
    """Raises for a role it does not know, as some renderers do."""

    def _render(self, conversation):
        for message in conversation:
            if message["role"] not in self.ROLES:
                raise ValueError(f"Unsupported role: {message['role']}")
        return super()._render(conversation)


class _NeedsATemplateKwarg(_NativeRenderer):
    """Renders only with a template kwarg the request carries, so a probe tells nothing."""

    def apply_chat_template(self, conversation, **kwargs):
        if "persona" not in kwargs:
            raise KeyError("persona")
        return super().apply_chat_template(conversation, **kwargs)


class _JinjaTokenizer(_RendersDeveloper):
    """A tokenizer with a Jinja chat template that has no developer branch."""

    chat_template = (
        "{% for m in messages %}{% if m.role != 'developer' %}<{{ m.role }}>{{ m.content }}"
        "{% endif %}{% endfor %}"
    )

    def get_chat_template(self, chat_template=None, tools=None):
        self.template_lookups += 1
        return self.chat_template


class _Generation:
    """A finished generation, as far as the routes read one before post-processing."""

    prompt_token_ids = [1, 2, 3]
    finished = True
    error = None

    async def aresult(self):
        return self

    def __await__(self):
        return self.aresult().__await__()


def _serve(tmp_path, monkeypatch, tokenizer):
    """A stub-engine server with ``tokenizer``; ``prompts`` holds what reached the engine.

    Prompt rendering is real; each route's post-processing is stubbed.
    """
    responses_body = SimpleNamespace(
        prompt_token_ids=None, model_dump=lambda **kwargs: {"object": "response"}
    )
    monkeypatch.setattr(
        openai_server_module,
        "responses_api_create_response",
        AsyncMock(return_value=responses_body),
    )
    prompts = []

    def generate_async(*, inputs, sampling_params, **kwargs):
        prompts.append(inputs)
        return _Generation()

    server = make_stub_openai_server(
        tmp_path, MODEL, _ModelConfig(), generate_async, tokenizer=tokenizer
    )
    server.await_disconnected = AsyncMock()
    server._extract_metrics = AsyncMock()
    server._create_chat_response = AsyncMock(
        return_value=ChatCompletionResponse(
            model=MODEL,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Done."),
                    finish_reason="stop",
                )
            ],
            usage=USAGE,
        )
    )
    app = FastAPI()
    app.add_api_route("/v1/chat/completions", server.openai_chat, methods=["POST"])
    app.add_api_route("/v1/responses", server.openai_responses, methods=["POST"])
    return SimpleNamespace(client=TestClient(app), prompts=prompts)


def _prompt_text(inputs):
    """The prompt text a route handed to the engine (text, or the stub's one id per character)."""
    if isinstance(inputs, list):
        return "".join(chr(token) for token in inputs)
    if "prompt_token_ids" in inputs:
        return "".join(chr(token) for token in inputs["prompt_token_ids"])
    return inputs["prompt"]


def _post_responses(served, **fields):
    """A Responses request shaped as Codex CLI sends one: instructions, a brief, the question."""
    body = {
        "model": MODEL,
        "instructions": INSTRUCTIONS,
        "input": [
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": BRIEF}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": QUESTION}],
            },
        ],
        **fields,
    }
    response = served.client.post("/v1/responses", json=body)
    assert response.status_code == 200, response.text
    return _prompt_text(served.prompts[-1])


def _post_chat(served, messages=None, **fields):
    body = {
        "model": MODEL,
        "messages": messages
        or [{"role": "developer", "content": BRIEF}, {"role": "user", "content": QUESTION}],
        **fields,
    }
    response = served.client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text
    return _prompt_text(served.prompts[-1])


def test_responses_developer_message_renders_as_system_after_the_instructions(
    tmp_path, monkeypatch
):
    tokenizer = _NativeRenderer()
    served = _serve(tmp_path, monkeypatch, tokenizer)

    prompt = _post_responses(served)

    assert prompt == f"<system>{INSTRUCTIONS}<system>{BRIEF}<user>{QUESTION}<assistant>"
    # No Jinja template is looked up for a tokenizer that renders in Python.
    assert tokenizer.template_lookups == 0


def test_chat_developer_message_renders_as_system(tmp_path, monkeypatch):
    served = _serve(tmp_path, monkeypatch, _NativeRenderer())

    prompt = _post_chat(served)

    assert prompt == f"<system>{BRIEF}<user>{QUESTION}<assistant>"


@pytest.mark.parametrize("route", ["chat", "responses"])
def test_a_renderer_that_rejects_developer_gets_it_as_system(tmp_path, monkeypatch, route):
    served = _serve(tmp_path, monkeypatch, _RejectsDeveloper())

    prompt = _post_chat(served) if route == "chat" else _post_responses(served)

    assert f"<system>{BRIEF}<user>{QUESTION}" in prompt


@pytest.mark.parametrize("route", ["chat", "responses"])
def test_a_renderer_that_renders_developer_gets_it_unchanged(tmp_path, monkeypatch, route):
    tokenizer = _RendersDeveloper()
    served = _serve(tmp_path, monkeypatch, tokenizer)

    prompt = _post_chat(served) if route == "chat" else _post_responses(served)

    assert f"<developer>{BRIEF}<user>{QUESTION}" in prompt
    assert tokenizer.prompts()[0][-2] == ("developer", BRIEF)


def test_the_renderer_is_probed_once(tmp_path, monkeypatch):
    tokenizer = _NativeRenderer()
    served = _serve(tmp_path, monkeypatch, tokenizer)

    for _ in range(2):
        _post_chat(served)
        _post_responses(served)

    # One developer probe, then one system probe because the developer message was left out.
    assert [conversation[0][0] for conversation in tokenizer.probes()] == ["developer", "system"]
    assert len(tokenizer.prompts()) == 4


def test_a_request_without_a_developer_message_is_not_probed(tmp_path, monkeypatch):
    tokenizer = _NativeRenderer()
    served = _serve(tmp_path, monkeypatch, tokenizer)

    prompt = _post_chat(
        served,
        messages=[{"role": "system", "content": BRIEF}, {"role": "user", "content": QUESTION}],
    )

    assert prompt == f"<system>{BRIEF}<user>{QUESTION}<assistant>"
    assert tokenizer.probes() == []


def test_a_probe_that_tells_nothing_keeps_the_roles_and_warns_once(tmp_path, monkeypatch):
    tokenizer = _NeedsATemplateKwarg()
    served = _serve(tmp_path, monkeypatch, tokenizer)
    # Recorded on the logger itself, so the count does not depend on how logging is configured.
    warnings = []
    monkeypatch.setattr(
        inputs_utils.logger, "warning", lambda message, *args: warnings.append(message % args)
    )

    for _ in range(2):
        prompt = _post_chat(served, chat_template_kwargs={"persona": "agent"})

    # As before the probe: the developer message reaches the renderer as it was sent.
    assert prompt == f"<user>{QUESTION}<assistant>"
    assert [conversation[0] for conversation in tokenizer.prompts()] == [("developer", BRIEF)] * 2
    assert len(warnings) == 1
    assert "_NeedsATemplateKwarg" in warnings[0] and "developer messages" in warnings[0]


def test_a_jinja_template_is_rendered_as_before(tmp_path, monkeypatch):
    tokenizer = _JinjaTokenizer()
    served = _serve(tmp_path, monkeypatch, tokenizer)

    _post_chat(served)
    _post_responses(served)

    chat, responses = tokenizer.prompts()
    # The chat route hands the developer message to a Jinja template as it was sent; the
    # Responses route maps it to system for a template that has no developer branch.
    assert [role for role, _ in chat] == ["developer", "user"]
    assert [role for role, _ in responses] == ["system", "system", "user"]
    assert tokenizer.probes() == []


def test_kimi_k3_renders_a_developer_message_as_system():
    """The Kimi-K3 checkpoint tokenizer, from LLM_MODELS_ROOT; skipped when it is not there."""
    request = chat_request(
        messages=[
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "developer", "content": BRIEF},
            {"role": "user", "content": QUESTION},
        ]
    )
    res = resources(make_tokenizer(name=KIMI_K3), model_type="kimi_k3")

    text = render_chat(request, res, tokenize=False).text

    system = '<|open|>message role="system"<|sep|>'
    assert system + INSTRUCTIONS in text
    assert system + BRIEF in text
    assert text.index(INSTRUCTIONS) < text.index(BRIEF) < text.index(QUESTION)
