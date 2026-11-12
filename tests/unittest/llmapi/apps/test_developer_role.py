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
"""A developer message reaches the prompt when the chat renderer has no branch for the role.

Clients such as Codex CLI (Responses) and pi (Chat Completions) send their instructions as a
``developer`` message. A Jinja chat template written before that role existed (GLM-5.x's), or a
tokenizer that renders the conversation in its own Python code (the Kimi-K3 checkpoint tokenizer),
may have no branch for it and leave the message out of the prompt. Such a renderer is probed once,
and is handed developer messages as system messages when it drops them.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from transformers.utils.chat_template_utils import render_jinja_template

from tensorrt_llm.inputs import utils as inputs_utils
from tensorrt_llm.inputs.utils import apply_chat_template
from tensorrt_llm.serve import openai_server as openai_server_module
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)
from tensorrt_llm.serve.render import render_chat

from .openai_server_stub import make_stub_openai_server
from .render_helpers import KIMI_K3, chat_request, make_tokenizer, model_dir, resources

pytestmark = pytest.mark.cpu_only

MODEL = "test-model"
INSTRUCTIONS = "Answer briefly."
BRIEF = "You are a coding agent. Run the tests before you finish."
QUESTION = "Fix the failing test."
USAGE = UsageInfo(prompt_tokens=3, completion_tokens=1, total_tokens=4)
PROBE = "trtllm-chat-template-role-probe"


def _glm_shaped(*, developer_branch=False, preamble=""):
    """A Jinja chat template shaped like GLM-5.x's.

    It renders system, user, assistant and tool messages and leaves any other role out, unless
    ``developer_branch`` adds a branch for developer messages. ``preamble`` comes first.
    """
    developer = "{%- elif m.role == 'developer' -%}<|developer|>{{ m.content }}"
    return (
        preamble + "[gMASK]<sop>{%- for m in messages -%}"
        "{%- if m.role == 'system' -%}<|system|>{{ m.content }}"
        "{%- elif m.role == 'user' -%}<|user|>{{ m.content }}"
        "{%- elif m.role == 'assistant' -%}<|assistant|>{{ m.content }}"
        "{%- elif m.role == 'tool' -%}<|observation|>{{ m.content }}"
        + (developer if developer_branch else "")
        + "{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
    )


class _ModelConfig:
    model_type = "llama"
    vocab_size = 1024


class _StubTokenizer:
    """What the server reads from a tokenizer. Tokenizing encodes one character per id.

    ``rendered`` records the role and content of every message of every conversation the
    tokenizer renders.
    """

    def __init__(self):
        self.tokenizer = self
        self.vocab_size = _ModelConfig.vocab_size
        self.name_or_path = None
        self.rendered = []
        self.template_lookups = 0

    def encode(self, text, **kwargs):
        return [ord(character) for character in text]

    def _record(self, conversation):
        self.rendered.append([(message["role"], message["content"]) for message in conversation])

    def probes(self):
        """The conversations rendered to probe a role."""
        return [conversation for conversation in self.rendered if conversation[0][1] == PROBE]

    def prompts(self):
        """The conversations rendered for requests."""
        return [conversation for conversation in self.rendered if conversation[0][1] != PROBE]


class _NativeRenderer(_StubTokenizer):
    """A tokenizer that renders the conversation in Python and has no Jinja template.

    It renders the roles in ``ROLES`` and leaves any other message out, as the Kimi-K3 checkpoint
    tokenizer does.
    """

    ROLES = ("system", "user", "assistant", "tool")
    chat_template = None

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
        self._record(conversation)
        text = self._render(conversation) + ("<assistant>" if add_generation_prompt else "")
        return self.encode(text) if tokenize else text


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


class _JinjaTokenizer(_StubTokenizer):
    """A tokenizer with a Jinja chat template, rendered as a Hugging Face tokenizer renders it."""

    def __init__(self, template):
        super().__init__()
        self.chat_template = template

    def get_chat_template(self, chat_template=None, tools=None):
        self.template_lookups += 1
        return chat_template or self.chat_template

    def apply_chat_template(
        self,
        conversation,
        chat_template=None,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        documents=None,
        return_dict=False,
        **kwargs,
    ):
        self._record(conversation)
        rendered, _ = render_jinja_template(
            conversations=[conversation],
            chat_template=chat_template or self.chat_template,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            documents=documents,
            **kwargs,
        )
        return self.encode(rendered[0]) if tokenize else rendered[0]


class _Generation:
    """A finished generation, as far as the routes read one before post-processing."""

    prompt_token_ids = [1, 2, 3]
    request_id = 1
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


@pytest.fixture
def template_probes(monkeypatch):
    """The roles the role probe renders a Jinja chat template with, in order."""
    probes = []
    render = inputs_utils.render_jinja_template

    def recording(*, conversations, **kwargs):
        if conversations[0][0]["content"] == PROBE:
            probes.append(conversations[0][0]["role"])
        return render(conversations=conversations, **kwargs)

    monkeypatch.setattr(inputs_utils, "render_jinja_template", recording)
    return probes


@pytest.fixture
def logged_warnings(monkeypatch):
    """The warnings the renderer logs, recorded on its logger whatever the logging config."""
    logged = []
    monkeypatch.setattr(
        inputs_utils.logger, "warning", lambda message, *args: logged.append(message % args)
    )
    return logged


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
    """A Chat Completions request shaped as pi sends one: its system prompt as a developer message."""
    body = {
        "model": MODEL,
        "messages": messages
        or [{"role": "developer", "content": BRIEF}, {"role": "user", "content": QUESTION}],
        **fields,
    }
    response = served.client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text
    return _prompt_text(served.prompts[-1])


def _post(served, route):
    return _post_chat(served) if route == "chat" else _post_responses(served)


# ---------------------------------------------------------------------------
# A Jinja chat template
# ---------------------------------------------------------------------------


def test_chat_developer_message_renders_as_system_for_a_template_without_the_role(
    tmp_path, monkeypatch
):
    served = _serve(tmp_path, monkeypatch, _JinjaTokenizer(_glm_shaped()))

    prompt = _post_chat(served)

    assert prompt == f"[gMASK]<sop><|system|>{BRIEF}<|user|>{QUESTION}<|assistant|>"


def test_responses_developer_message_renders_as_system_for_a_template_without_the_role(
    tmp_path, monkeypatch
):
    served = _serve(tmp_path, monkeypatch, _JinjaTokenizer(_glm_shaped()))

    prompt = _post_responses(served)

    assert prompt == (
        f"[gMASK]<sop><|system|>{INSTRUCTIONS}<|system|>{BRIEF}<|user|>{QUESTION}<|assistant|>"
    )


@pytest.mark.parametrize("route", ["chat", "responses"])
def test_a_template_with_a_developer_branch_keeps_the_role(tmp_path, monkeypatch, route):
    tokenizer = _JinjaTokenizer(_glm_shaped(developer_branch=True))
    served = _serve(tmp_path, monkeypatch, tokenizer)

    prompt = _post(served, route)

    assert f"<|developer|>{BRIEF}<|user|>{QUESTION}" in prompt
    assert tokenizer.prompts()[0][-2] == ("developer", BRIEF)


def test_a_template_is_probed_once(tmp_path, monkeypatch, template_probes):
    served = _serve(
        tmp_path, monkeypatch, _JinjaTokenizer(_glm_shaped(preamble="{#- probed once -#}"))
    )

    for _ in range(2):
        _post_chat(served)
        _post_responses(served)

    # One developer probe, then one system probe because the developer message was left out.
    assert template_probes == ["developer", "system"]


def test_a_request_without_a_developer_message_does_not_probe_the_template(
    tmp_path, monkeypatch, template_probes
):
    served = _serve(
        tmp_path, monkeypatch, _JinjaTokenizer(_glm_shaped(preamble="{#- no developer -#}"))
    )

    prompt = _post_chat(
        served,
        messages=[{"role": "system", "content": BRIEF}, {"role": "user", "content": QUESTION}],
    )

    assert prompt == f"[gMASK]<sop><|system|>{BRIEF}<|user|>{QUESTION}<|assistant|>"
    assert template_probes == []


def test_a_template_probe_that_tells_nothing_keeps_the_roles_and_warns_once(
    tmp_path, monkeypatch, logged_warnings
):
    # Needs a template kwarg the request carries and the probe cannot know.
    template = _glm_shaped(
        preamble="{%- if persona is not defined -%}{{ raise_exception('no persona') }}{%- endif -%}"
    )
    tokenizer = _JinjaTokenizer(template)
    served = _serve(tmp_path, monkeypatch, tokenizer)

    for _ in range(2):
        prompt = _post_chat(served, chat_template_kwargs={"persona": "agent"})

    # As before the probe: the developer message reaches the template as it was sent.
    assert prompt == f"[gMASK]<sop><|user|>{QUESTION}<|assistant|>"
    assert [conversation[0] for conversation in tokenizer.prompts()] == [("developer", BRIEF)] * 2
    assert len(logged_warnings) == 1
    assert "the chat template" in logged_warnings[0] and "developer messages" in logged_warnings[0]


class _RecordingTemplateTokenizer:
    """A tokenizer with a Jinja template that records the roles it is asked to render."""

    def __init__(self, template):
        self.chat_template = template
        self.roles = []

    def get_chat_template(self, chat_template=None, tools=None):
        return self.chat_template

    def apply_chat_template(self, conversation, **kwargs):
        self.roles.append([message["role"] for message in conversation])
        return "rendered"


def _loop(body):
    return "{% for m in messages %}" + body + "{% endfor %}"


@pytest.mark.parametrize(
    "template, role",
    [
        (_loop("{% if m.role in ('system', 'user') %}{{ m.content }}{% endif %}"), "system"),
        (
            _loop("{% if m.role in ('system', 'developer', 'user') %}{{ m.content }}{% endif %}"),
            "developer",
        ),
        # Naming the role is not handling it.
        (
            _loop(
                "{% if m.role == 'developer' %}"
                "{{ raise_exception('Unsupported role: developer') }}{% endif %}{{ m.content }}"
            ),
            "system",
        ),
        (
            "{# developer messages are not supported #}"
            + _loop("{% if m.role != 'developer' %}{{ m.content }}{% endif %}"),
            "system",
        ),
        # A template that renders neither role tells nothing; the roles stay.
        (_loop("{{ m.content.missing.attribute }}"), "developer"),
    ],
    ids=[
        "no_developer_branch",
        "developer_branch",
        "rejects_developer",
        "developer_only_in_a_comment",
        "renders_neither",
    ],
)
def test_a_template_gets_developer_as_system_only_where_it_lacks_the_role(template, role):
    tokenizer = _RecordingTemplateTokenizer(template)
    conversation = [{"role": "developer", "content": BRIEF}, {"role": "user", "content": QUESTION}]

    apply_chat_template(
        model_type="developer-role-test-model",
        tokenizer=tokenizer,
        processor=None,
        conversation=conversation,
        add_generation_prompt=True,
        mm_placeholder_counts=[{}, {}],
    )

    assert tokenizer.roles == [[role, "user"]]
    # The caller's messages are left as they are.
    assert conversation[0]["role"] == "developer"


@pytest.mark.parametrize("selection", ["processor", "explicit", "named"])
@pytest.mark.parametrize("supports_developer", [False, True])
def test_developer_fallback_uses_the_same_source_as_dispatch_analysis(
    selection, supports_developer
):
    active = _glm_shaped(developer_branch=supports_developer)
    tokenizer = _JinjaTokenizer(_glm_shaped(developer_branch=not supports_developer))
    processor = None
    chat_template = None
    if selection == "processor":
        processor = SimpleNamespace(chat_template=active)
    elif selection == "explicit":
        chat_template = active
    else:
        tokenizer.chat_template = {"default": tokenizer.chat_template, "selected": active}
        chat_template = "selected"
    render_args = dict(
        tokenizer=tokenizer,
        processor=processor,
        chat_template=chat_template,
        model_type="developer-role-test-model",
    )
    assert inputs_utils.rendered_hf_chat_template_source(**render_args) == active
    conversation = [{"role": "developer", "content": BRIEF}, {"role": "user", "content": QUESTION}]

    prompt = apply_chat_template(
        **render_args,
        conversation=conversation,
        add_generation_prompt=True,
        mm_placeholder_counts=[{}, {}],
    )

    role = "developer" if supports_developer else "system"
    assert tokenizer.prompts() == [[(role, BRIEF), ("user", QUESTION)]]
    assert prompt == f"[gMASK]<sop><|{role}|>{BRIEF}<|user|>{QUESTION}<|assistant|>"
    assert conversation[0]["role"] == "developer"


@pytest.mark.parametrize(
    "name, role",
    [("GLM-5.3", "system"), ("gemma/gemma-4-E2B-it", "developer")],
    ids=["glm_without_developer", "gemma_with_developer"],
)
def test_a_real_chat_template_gets_developer_as_its_branches_allow(name, role):
    """Chat templates of real checkpoints from LLM_MODELS_ROOT; skipped when they are not there."""
    path = model_dir(name) / "chat_template.jinja"
    if not path.is_file():
        pytest.skip(f"{name} has no chat_template.jinja")
    tokenizer = _JinjaTokenizer(path.read_text())

    text = apply_chat_template(
        model_type="developer-role-test-model",
        tokenizer=tokenizer,
        processor=None,
        conversation=[
            {"role": "developer", "content": BRIEF},
            {"role": "user", "content": QUESTION},
        ],
        add_generation_prompt=True,
        mm_placeholder_counts=[{}, {}],
    )

    assert tokenizer.prompts()[0][0] == (role, BRIEF)
    assert BRIEF in text and text.index(BRIEF) < text.index(QUESTION)


# ---------------------------------------------------------------------------
# A tokenizer that renders in Python, with no Jinja template
# ---------------------------------------------------------------------------


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

    prompt = _post(served, route)

    assert f"<system>{BRIEF}<user>{QUESTION}" in prompt


@pytest.mark.parametrize("route", ["chat", "responses"])
def test_a_renderer_that_renders_developer_gets_it_unchanged(tmp_path, monkeypatch, route):
    tokenizer = _RendersDeveloper()
    served = _serve(tmp_path, monkeypatch, tokenizer)

    prompt = _post(served, route)

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


def test_a_probe_that_tells_nothing_keeps_the_roles_and_warns_once(
    tmp_path, monkeypatch, logged_warnings
):
    tokenizer = _NeedsATemplateKwarg()
    served = _serve(tmp_path, monkeypatch, tokenizer)

    for _ in range(2):
        prompt = _post_chat(served, chat_template_kwargs={"persona": "agent"})

    # As before the probe: the developer message reaches the renderer as it was sent.
    assert prompt == f"<user>{QUESTION}<assistant>"
    assert [conversation[0] for conversation in tokenizer.prompts()] == [("developer", BRIEF)] * 2
    assert len(logged_warnings) == 1
    assert (
        "_NeedsATemplateKwarg" in logged_warnings[0] and "developer messages" in logged_warnings[0]
    )


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
