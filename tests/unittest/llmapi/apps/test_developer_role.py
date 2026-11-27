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
_PROBE_MARKERS = (
    PROBE,
    inputs_utils._SECOND_ROLE_PROBE,
)


def _content_contains_probe(content):
    if isinstance(content, str):
        return inputs_utils._PROBE_BASE in content
    if isinstance(content, list):
        return any(
            isinstance(part, dict)
            and isinstance(part.get("text"), str)
            and inputs_utils._PROBE_BASE in part["text"]
            for part in content
        )
    return False


def _is_probe_conversation(conversation):
    return any(_content_contains_probe(content) for _role, content in conversation)


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
        return [
            conversation for conversation in self.rendered if _is_probe_conversation(conversation)
        ]

    def prompts(self):
        """The conversations rendered for requests."""
        return [
            conversation
            for conversation in self.rendered
            if not _is_probe_conversation(conversation)
        ]


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


class _RendersNeitherRole(_NativeRenderer):
    """Renders only user turns, so probing developer and system roles tells nothing."""

    ROLES = ("user",)


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
        first_conv = conversations[0]
        probed_roles = [
            msg["role"] for msg in first_conv if _content_contains_probe(msg.get("content"))
        ]
        probes.append(probed_roles[0] if probed_roles else first_conv[0]["role"])
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
    before = len(served.prompts)
    response = served.client.post("/v1/responses", json=body)
    assert response.status_code == 200, response.text
    assert len(served.prompts) == before + 1
    return _prompt_text(served.prompts[-1])


def _post_chat(served, messages=None, **fields):
    """A Chat Completions request shaped as pi sends one: its system prompt as a developer message."""
    body = {
        "model": MODEL,
        "messages": messages
        or [{"role": "developer", "content": BRIEF}, {"role": "user", "content": QUESTION}],
        **fields,
    }
    before = len(served.prompts)
    response = served.client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text
    assert len(served.prompts) == before + 1
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
    # One developer probe, then one system probe for Chat (tools=None); cached on repeat.
    assert template_probes == ["developer", "system"]

    for _ in range(2):
        _post_responses(served)
    # Responses passes tools=[]; probed once for that distinct context and cached on repeat.
    assert template_probes == ["developer", "system", "developer", "system"]


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
    # Renders neither developer nor system, so the probe tells nothing.
    template = (
        "{#- warns once -#}[gMASK]<sop>{%- for m in messages -%}"
        "{%- if m.role == 'user' -%}<|user|>{{ m.content }}"
        "{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
    )
    tokenizer = _JinjaTokenizer(template)
    served = _serve(tmp_path, monkeypatch, tokenizer)

    for _ in range(2):
        prompt = _post_chat(served)

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
    # One developer probe, then one system probe for Chat (tools=None); cached on repeat.
    assert [conversation[0][0] for conversation in tokenizer.probes()] == ["developer", "system"]

    for _ in range(2):
        _post_responses(served)
    # Responses passes tools=[]; probed once for that distinct context and cached on repeat.
    assert [conversation[0][0] for conversation in tokenizer.probes()] == [
        "developer",
        "system",
        "developer",
        "system",
    ]
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
    tokenizer = _RendersNeitherRole()
    served = _serve(tmp_path, monkeypatch, tokenizer)

    for _ in range(2):
        prompt = _post_chat(served)

    # As before the probe: the developer message reaches the renderer as it was sent.
    assert prompt == f"<user>{QUESTION}<assistant>"
    assert [conversation[0] for conversation in tokenizer.prompts()] == [("developer", BRIEF)] * 2
    assert len(logged_warnings) == 1
    assert (
        "_RendersNeitherRole" in logged_warnings[0] and "developer messages" in logged_warnings[0]
    )


def test_strict_single_system_template_merges_developer_into_leading_system(tmp_path, monkeypatch):
    """Strict templates rejecting a second or non-leading system merge developers into index 0."""
    strict_template = (
        "{%- if messages[0].role == 'system' -%}<|system|>{{ messages[0].content }}"
        "{%- set rest = messages[1:] -%}{%- else -%}{%- set rest = messages -%}{%- endif -%}"
        "{%- for m in rest -%}"
        "{%- if m.role == 'system' -%}{{ raise_exception('system role only allowed at index 0') }}"
        "{%- elif m.role == 'user' -%}<|user|>{{ m.content }}"
        "{%- elif m.role == 'assistant' -%}<|assistant|>{{ m.content }}"
        "{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
    )
    served = _serve(tmp_path, monkeypatch, _JinjaTokenizer(strict_template))

    # Responses API with instructions (leading system) + developer brief
    resp_prompt = _post_responses(served)
    assert resp_prompt == f"<|system|>{INSTRUCTIONS}\n\n{BRIEF}<|user|>{QUESTION}<|assistant|>"

    # Chat Completions with system -> user -> late developer -> user
    late_prompt = _post_chat(
        served,
        messages=[
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": "First."},
            {"role": "developer", "content": BRIEF},
            {"role": "user", "content": QUESTION},
        ],
    )
    assert (
        late_prompt
        == f"<|system|>{INSTRUCTIONS}\n\n{BRIEF}<|user|>First.\n\n{QUESTION}<|assistant|>"
    )

    # Chat Completions with no leading system and a late developer
    lifted_prompt = _post_chat(
        served,
        messages=[
            {"role": "user", "content": "First."},
            {"role": "developer", "content": BRIEF},
            {"role": "user", "content": QUESTION},
        ],
    )
    assert lifted_prompt == f"<|system|>{BRIEF}<|user|>First.\n\n{QUESTION}<|assistant|>"

    # Developer before system merges into a single leading system preserving input order
    dev_first_prompt = _post_chat(
        served,
        messages=[
            {"role": "developer", "content": BRIEF},
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": QUESTION},
        ],
    )
    assert dev_first_prompt == f"<|system|>{BRIEF}\n\n{INSTRUCTIONS}<|user|>{QUESTION}<|assistant|>"


def test_alternating_turn_template_accepting_single_late_system_keeps_developer_in_place(
    tmp_path, monkeypatch
):
    """A template allowing only ONE system anywhere + alternating turns keeps late developer in place."""
    single_anywhere_template = (
        "{%- set ns = namespace(last='', sys_count=0) -%}"
        "{%- for m in messages -%}"
        "{%- if m.role == 'system' -%}"
        "{%- set ns.sys_count = ns.sys_count + 1 -%}"
        "{%- if ns.sys_count > 1 -%}{{ raise_exception('at most one system') }}{%- endif -%}"
        "<|system|>{{ m.content }}"
        "{%- elif m.role == 'user' -%}"
        "{%- if ns.last == 'user' -%}{{ raise_exception('turns must alternate') }}{%- endif -%}"
        "{%- set ns.last = 'user' -%}<|user|>{{ m.content }}"
        "{%- elif m.role == 'assistant' -%}"
        "{%- if ns.last == 'assistant' -%}{{ raise_exception('turns must alternate') }}{%- endif -%}"
        "{%- set ns.last = 'assistant' -%}<|assistant|>{{ m.content }}"
        "{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
    )
    served = _serve(tmp_path, monkeypatch, _JinjaTokenizer(single_anywhere_template))

    prompt = _post_chat(
        served,
        messages=[
            {"role": "user", "content": "First."},
            {"role": "assistant", "content": "Reply."},
            {"role": "developer", "content": BRIEF},
            {"role": "user", "content": QUESTION},
        ],
    )
    assert (
        prompt
        == f"<|user|>First.<|assistant|>Reply.<|system|>{BRIEF}<|user|>{QUESTION}<|assistant|>"
    )


def test_template_accepting_one_late_system_but_rejecting_two_late_systems_merges(
    tmp_path, monkeypatch
):
    """A template allowing two leading systems and one late system merges two late developers."""
    one_late_max_template = (
        "{%- set ns = namespace(seen_turn=false, late_sys=0) -%}"
        "{%- for m in messages -%}"
        "{%- if m.role == 'system' -%}"
        "{%- if ns.seen_turn -%}"
        "{%- set ns.late_sys = ns.late_sys + 1 -%}"
        "{%- if ns.late_sys > 1 -%}{{ raise_exception('at most one late system') }}{%- endif -%}"
        "{%- endif -%}"
        "<|system|>{{ m.content }}"
        "{%- elif m.role == 'user' -%}{%- set ns.seen_turn = true -%}<|user|>{{ m.content }}"
        "{%- elif m.role == 'assistant' -%}{%- set ns.seen_turn = true -%}<|assistant|>{{ m.content }}"
        "{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
    )
    served = _serve(tmp_path, monkeypatch, _JinjaTokenizer(one_late_max_template))

    two_late_prompt = _post_chat(
        served,
        messages=[
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"},
            {"role": "developer", "content": INSTRUCTIONS},
            {"role": "user", "content": "U2"},
            {"role": "assistant", "content": "A2"},
            {"role": "developer", "content": BRIEF},
            {"role": "user", "content": QUESTION},
        ],
    )
    assert two_late_prompt == (
        f"<|system|>{INSTRUCTIONS}\n\n{BRIEF}"
        f"<|user|>U1<|assistant|>A1<|user|>U2<|assistant|>A2<|user|>{QUESTION}<|assistant|>"
    )

    # Three late developers on a renderer that drops systems beyond the first two must also merge,
    # and two adjacent late developers on a renderer that rejects consecutive late systems must merge.
    max_two_sys_template = (
        "{%- set ns = namespace(sc=0) -%}"
        "{%- for m in messages -%}"
        "{%- if m.role == 'system' -%}"
        "{%- set ns.sc = ns.sc + 1 -%}"
        "{%- if ns.sc <= 2 -%}<|system|>{{ m.content }}{%- endif -%}"
        "{%- elif m.role == 'user' -%}<|user|>{{ m.content }}"
        "{%- elif m.role == 'assistant' -%}<|assistant|>{{ m.content }}"
        "{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
    )
    served_two = _serve(tmp_path, monkeypatch, _JinjaTokenizer(max_two_sys_template))
    three_late_prompt = _post_chat(
        served_two,
        messages=[
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"},
            {"role": "developer", "content": "D1"},
            {"role": "user", "content": "U2"},
            {"role": "assistant", "content": "A2"},
            {"role": "developer", "content": "D2"},
            {"role": "user", "content": "U3"},
            {"role": "assistant", "content": "A3"},
            {"role": "developer", "content": "D3"},
            {"role": "user", "content": QUESTION},
        ],
    )
    assert three_late_prompt == (
        "<|system|>D1\n\nD2\n\nD3"
        "<|user|>U1<|assistant|>A1<|user|>U2<|assistant|>A2"
        f"<|user|>U3<|assistant|>A3<|user|>{QUESTION}<|assistant|>"
    )

    no_adj_late_sys_template = (
        "{%- set ns = namespace(seen_turn=false, prev='') -%}"
        "{%- for m in messages -%}"
        "{%- if m.role == 'system' -%}"
        "{%- if ns.seen_turn and ns.prev == 'system' -%}{{ raise_exception('no adjacent late system') }}{%- endif -%}"
        "<|system|>{{ m.content }}"
        "{%- elif m.role == 'user' -%}{%- set ns.seen_turn = true -%}<|user|>{{ m.content }}"
        "{%- elif m.role == 'assistant' -%}{%- set ns.seen_turn = true -%}<|assistant|>{{ m.content }}"
        "{%- endif -%}{%- set ns.prev = m.role -%}{%- endfor -%}"
        "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
    )
    served_adj = _serve(tmp_path, monkeypatch, _JinjaTokenizer(no_adj_late_sys_template))
    adj_late_prompt = _post_chat(
        served_adj,
        messages=[
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"},
            {"role": "developer", "content": INSTRUCTIONS},
            {"role": "developer", "content": BRIEF},
            {"role": "user", "content": QUESTION},
        ],
    )
    assert (
        adj_late_prompt
        == f"<|system|>{INSTRUCTIONS}\n\n{BRIEF}<|user|>U1<|assistant|>A1<|user|>{QUESTION}<|assistant|>"
    )

    # Dropping the FIRST of three consecutive system messages must not be masked by substring
    # overlap between probe markers.
    drop_first_of_three_template = (
        "{%- set sys_msgs = messages | selectattr('role', 'equalto', 'system') | list -%}"
        "{%- set ns = namespace(sc=0) -%}"
        "{%- for m in messages -%}"
        "{%- if m.role == 'system' -%}"
        "{%- set ns.sc = ns.sc + 1 -%}"
        "{%- if (sys_msgs | length) < 3 or ns.sc > 1 -%}<|system|>{{ m.content }}{%- endif -%}"
        "{%- elif m.role == 'user' -%}<|user|>{{ m.content }}"
        "{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
    )
    served_drop_first = _serve(tmp_path, monkeypatch, _JinjaTokenizer(drop_first_of_three_template))
    three_leading_prompt = _post_chat(
        served_drop_first,
        messages=[
            {"role": "developer", "content": "D1"},
            {"role": "developer", "content": "D2"},
            {"role": "developer", "content": "D3"},
            {"role": "user", "content": QUESTION},
        ],
    )
    assert three_leading_prompt == f"<|system|>D1\n\nD2\n\nD3<|user|>{QUESTION}<|assistant|>"

    # A template that drops a system message placed right after a tool turn must merge developer
    # into the leading system rather than falsely passing a user-substituted layout probe.
    drops_system_after_tool = (
        "{%- set ns = namespace(prev='') -%}"
        "{%- for m in messages -%}"
        "{%- if m.role == 'system' -%}"
        "{%- if ns.prev != 'tool' -%}<|system|>{{ m.content }}{%- endif -%}"
        "{%- elif m.role == 'user' -%}<|user|>{{ m.content }}"
        "{%- elif m.role == 'assistant' -%}"
        "<|assistant|>{{ m.content }}{%- if m.tool_calls -%}[call]{%- endif -%}"
        "{%- elif m.role == 'tool' -%}<|tool|>{{ m.content }}"
        "{%- endif -%}{%- set ns.prev = m.role -%}{%- endfor -%}"
    )
    tool_tok = _JinjaTokenizer(drops_system_after_tool)
    tool_prompt = apply_chat_template(
        model_type="llama",
        tokenizer=tool_tok,
        processor=None,
        conversation=[
            {"role": "user", "content": "U1"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "f", "content": "res"},
            {"role": "developer", "content": BRIEF},
            {"role": "user", "content": QUESTION},
        ],
        add_generation_prompt=False,
        mm_placeholder_counts=[{}, {}, {}, {}, {}],
        chat_template=drops_system_after_tool,
    )
    assert (
        tool_prompt
        == f"<|system|>{BRIEF}<|user|>U1<|assistant|>[call]<|tool|>res<|user|>{QUESTION}"
    )

    # A template that forbids consecutive system messages but allows non-adjacent system messages
    # across turns must keep a late developer in place as system instead of hoisting it to index 0.
    no_consecutive_system_template = (
        "{%- set ns = namespace(prev='') -%}"
        "{%- for m in messages -%}"
        "{%- if m.role == 'system' -%}"
        "{%- if ns.prev == 'system' -%}{{ raise_exception('consecutive system forbidden') }}{%- endif -%}"
        "<|system|>{{ m.content }}"
        "{%- elif m.role == 'user' -%}<|user|>{{ m.content }}"
        "{%- elif m.role == 'assistant' -%}<|assistant|>{{ m.content }}"
        "{%- endif -%}{%- set ns.prev = m.role -%}{%- endfor -%}"
        "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
    )
    served_non_adj = _serve(tmp_path, monkeypatch, _JinjaTokenizer(no_consecutive_system_template))
    non_adj_prompt = _post_chat(
        served_non_adj,
        messages=[
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": "U1"},
            {"role": "assistant", "content": "A1"},
            {"role": "developer", "content": BRIEF},
            {"role": "user", "content": QUESTION},
        ],
    )
    assert (
        non_adj_prompt
        == f"<|system|>{INSTRUCTIONS}<|user|>U1<|assistant|>A1<|system|>{BRIEF}<|user|>{QUESTION}<|assistant|>"
    )

    # Chat Completions decodes tool_calls function.arguments into a dict; layout probing must
    # preserve that dict representation so templates calling arguments.items() do not fail the
    # probe and hoist a post-tool developer message to index 0.
    dict_tool_args_template = (
        "{%- for m in messages -%}"
        "{%- if m.role == 'system' -%}<|system|>{{ m.content }}"
        "{%- elif m.role == 'user' -%}<|user|>{{ m.content }}"
        "{%- elif m.role == 'assistant' -%}<|assistant|>{{ m.content }}"
        "{%- if m.tool_calls -%}"
        "{%- for tc in m.tool_calls -%}"
        "[{{ tc.function.name }}:{%- for k, v in tc.function.arguments.items() -%}{{ k }}={{ v }}{%- endfor -%}]"
        "{%- endfor -%}"
        "{%- endif -%}"
        "{%- elif m.role == 'tool' -%}<|tool|>{{ m.content }}"
        "{%- endif -%}{%- endfor -%}"
        "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
    )
    served_dict_args = _serve(tmp_path, monkeypatch, _JinjaTokenizer(dict_tool_args_template))
    dict_tool_prompt = _post_chat(
        served_dict_args,
        messages=[
            {"role": "user", "content": "U1"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"k": "v"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "name": "f", "content": "res"},
            {"role": "developer", "content": BRIEF},
            {"role": "user", "content": QUESTION},
        ],
    )
    assert (
        dict_tool_prompt
        == f"<|user|>U1<|assistant|>[f:k=v]<|tool|>res<|system|>{BRIEF}<|user|>{QUESTION}<|assistant|>"
    )

    # Parallel tool calls with matching tool results must preserve call count and 1-to-1 call/result
    # id mapping during layout probing so strict templates keep a post-tool developer in place.
    strict_parallel_tool_template = (
        "{%- set ns = namespace(open_ids=[]) -%}"
        "{%- for m in messages -%}"
        "{%- if m.role == 'system' -%}"
        "{%- if ns.open_ids -%}{{ raise_exception('unclosed tool calls') }}{%- endif -%}"
        "<|system|>{{ m.content }}"
        "{%- elif m.role == 'user' -%}<|user|>{{ m.content }}"
        "{%- elif m.role == 'assistant' -%}"
        "<|assistant|>{{ m.content }}"
        "{%- if m.tool_calls -%}"
        "{%- for tc in m.tool_calls -%}"
        "{%- set ns.open_ids = ns.open_ids + [tc.id] -%}"
        "[call:{{ tc.id }}]"
        "{%- endfor -%}"
        "{%- endif -%}"
        "{%- elif m.role == 'tool' -%}"
        "{%- if not ns.open_ids or ns.open_ids[0] != m.tool_call_id -%}"
        "{{ raise_exception('tool_call_id mismatch') }}"
        "{%- endif -%}"
        "{%- set ns.open_ids = ns.open_ids[1:] -%}"
        "<|tool:{{ m.tool_call_id }}|>{{ m.content }}"
        "{%- endif -%}{%- endfor -%}"
        "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
    )
    served_parallel = _serve(tmp_path, monkeypatch, _JinjaTokenizer(strict_parallel_tool_template))
    parallel_prompt = _post_chat(
        served_parallel,
        messages=[
            {"role": "user", "content": "U1"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "f1", "arguments": '{"x": 1}'},
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "f2", "arguments": '{"y": 2}'},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "name": "f1", "content": "r1"},
            {"role": "tool", "tool_call_id": "call_b", "name": "f2", "content": "r2"},
            {"role": "developer", "content": BRIEF},
            {"role": "user", "content": QUESTION},
        ],
    )
    assert parallel_prompt == (
        "<|user|>U1<|assistant|>[call:call_a][call:call_b]"
        f"<|tool:call_a|>r1<|tool:call_b|>r2<|system|>{BRIEF}<|user|>{QUESTION}<|assistant|>"
    )


def test_template_accepting_leading_multi_system_but_rejecting_late_system(tmp_path, monkeypatch):
    """A template allowing leading multi-system but rejecting late system merges only late runs."""
    leading_only_template = (
        "{%- set ns = namespace(seen_non_sys=false) -%}"
        "{%- for m in messages -%}"
        "{%- if m.role == 'system' -%}"
        "{%- if ns.seen_non_sys -%}{{ raise_exception('late system forbidden') }}{%- endif -%}"
        "<|system|>{{ m.content }}"
        "{%- elif m.role == 'user' -%}{%- set ns.seen_non_sys = true -%}<|user|>{{ m.content }}"
        "{%- elif m.role == 'assistant' -%}{%- set ns.seen_non_sys = true -%}<|assistant|>{{ m.content }}"
        "{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
    )
    served = _serve(tmp_path, monkeypatch, _JinjaTokenizer(leading_only_template))

    # Consecutive leading [system, developer, user] stays two separate leading system blocks
    leading_prompt = _post_responses(served)
    assert (
        leading_prompt
        == f"<|system|>{INSTRUCTIONS}<|system|>{BRIEF}<|user|>{QUESTION}<|assistant|>"
    )

    # Late developer after user merges into leading system instead of raising
    late_prompt = _post_chat(
        served,
        messages=[
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": "First."},
            {"role": "developer", "content": BRIEF},
            {"role": "user", "content": QUESTION},
        ],
    )
    assert (
        late_prompt
        == f"<|system|>{INSTRUCTIONS}\n\n{BRIEF}<|user|>First.\n\n{QUESTION}<|assistant|>"
    )

    # Developer at index 0 + pre-existing late system also merges on leading-only templates
    dev_plus_late_sys = _post_chat(
        served,
        messages=[
            {"role": "developer", "content": BRIEF},
            {"role": "user", "content": "First."},
            {"role": "assistant", "content": "Reply."},
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": QUESTION},
        ],
    )
    assert (
        dev_plus_late_sys
        == f"<|system|>{BRIEF}\n\n{INSTRUCTIONS}<|user|>First.<|assistant|>Reply.<|user|>{QUESTION}<|assistant|>"
    )


def test_merge_developers_preserves_mm_placeholder_counts_alignment():
    """Merging developer into leading system keeps mm_placeholder_counts aligned with user turns."""
    conv = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "developer", "content": BRIEF},
        {"role": "user", "content": QUESTION},
    ]
    counts = [{}, {}, {"image": 2}]
    merged_conv, merged_counts = inputs_utils._merge_developers_into_leading_system(conv, counts)
    assert len(merged_conv) == len(merged_counts) == 2
    assert merged_conv[0] == {"role": "system", "content": f"{INSTRUCTIONS}\n\n{BRIEF}"}
    assert merged_conv[1] == {"role": "user", "content": QUESTION}
    assert merged_counts == [{}, {"image": 2}]

    # Late media-bearing developer after earlier user media stays in place coalesced with the
    # adjacent user turn so both media order (A then B) and strict user/assistant alternation hold.
    strict_alt_template = (
        "{%- set ns = namespace(last='') -%}"
        "{%- for m in messages -%}"
        "{%- if m.role == 'system' -%}"
        "{%- if not loop.first -%}{{ raise_exception('system only at 0') }}{%- endif -%}"
        "<|system|>{{ m.content }}"
        "{%- elif m.role in ['user', 'assistant'] -%}"
        "{%- if ns.last == m.role -%}{{ raise_exception('turns must alternate') }}{%- endif -%}"
        "{%- set ns.last = m.role -%}<|{{ m.role }}|>{{ m.content }}"
        "{%- endif -%}{%- endfor -%}"
    )
    mm_conv = [
        {"role": "user", "content": "<imgA>", "media": ["imgA"]},
        {"role": "assistant", "content": "Reply."},
        {"role": "developer", "content": "<imgB>", "media": ["imgB"]},
        {"role": "user", "content": QUESTION},
    ]
    mm_counts = [{"image": 1}, {}, {"image": 1}, {}]
    kept_conv, kept_counts = inputs_utils._developer_as_system(
        mm_conv,
        inputs_utils._template_developer_policy,
        inputs_utils._template_accepts_late_system,
        strict_alt_template,
        mm_placeholder_counts=mm_counts,
    )
    assert [m["role"] for m in kept_conv] == ["user", "assistant", "user"]
    assert kept_conv[2]["content"] == f"<imgB>\n\n{QUESTION}"
    assert kept_counts == [{"image": 1}, {}, {"image": 1}]
    rendered, _ = render_jinja_template(
        conversations=[kept_conv], chat_template=strict_alt_template, add_generation_prompt=False
    )
    assert rendered[0] == f"<|user|><imgA><|assistant|>Reply.<|user|><imgB>\n\n{QUESTION}"

    # Late media-bearing developer sandwiched between two users collapses into one user turn
    sandwich_conv = [
        {"role": "user", "content": "<imgA>", "media": ["imgA"]},
        {"role": "developer", "content": "<imgB>", "media": ["imgB"]},
        {"role": "user", "content": QUESTION},
    ]
    sandwich_counts = [{"image": 1}, {"image": 1}, {}]
    s_conv, s_counts = inputs_utils._developer_as_system(
        sandwich_conv,
        inputs_utils._template_developer_policy,
        inputs_utils._template_accepts_late_system,
        strict_alt_template,
        mm_placeholder_counts=sandwich_counts,
    )
    assert [m["role"] for m in s_conv] == ["user"]
    assert s_conv[0]["content"] == f"<imgA>\n\n<imgB>\n\n{QUESTION}"
    assert s_counts == [{"image": 2}]
    s_rendered, _ = render_jinja_template(
        conversations=[s_conv], chat_template=strict_alt_template, add_generation_prompt=False
    )
    assert s_rendered[0] == f"<|user|><imgA>\n\n<imgB>\n\n{QUESTION}"

    # Text-only developer lifted from between two users also coalesces adjacent users for strict_alt
    text_sandwich = [
        {"role": "user", "content": "A"},
        {"role": "developer", "content": BRIEF},
        {"role": "user", "content": QUESTION},
    ]
    ts_conv, _ = inputs_utils._developer_as_system(
        text_sandwich,
        inputs_utils._template_developer_policy,
        inputs_utils._template_accepts_late_system,
        strict_alt_template,
    )
    assert [m["role"] for m in ts_conv] == ["system", "user"]
    ts_rendered, _ = render_jinja_template(
        conversations=[ts_conv], chat_template=strict_alt_template, add_generation_prompt=False
    )
    assert ts_rendered[0] == f"<|system|>{BRIEF}<|user|>A\n\n{QUESTION}"

    # Merging one message with counts only and one with content_parts keeps both placeholders in order
    counts_first_conv = [
        {"role": "system", "content": "First text", "media": ["imgA"]},
        {
            "role": "developer",
            "content": "Second text",
            "media": ["imgB"],
            "content_parts": [{"type": "image", "media_index": 0}, "Second text"],
        },
        {"role": "user", "content": QUESTION},
    ]
    counts_first_mm = [{"<image>": 1}, {"<image>": 1}, {}]
    cf_conv, cf_counts = inputs_utils._merge_developers_into_leading_system(
        counts_first_conv, counts_first_mm
    )
    assert cf_counts == [{"<image>": 2}, {}]
    assert inputs_utils._build_openai_content(cf_conv[0], cf_counts[0]) == [
        {"type": "text", "text": "First text"},
        {"type": "image"},
        {"type": "text", "text": "\n\n"},
        {"type": "image"},
        {"type": "text", "text": "Second text"},
    ]
    assert cf_conv[0]["content_parts"] == [
        "First text",
        {"type": "image", "media_index": 0},
        "\n\n",
        {"type": "image", "media_index": 1},
        "Second text",
    ]

    parts_first_conv = [
        {
            "role": "system",
            "content": "First text",
            "media": ["imgA"],
            "content_parts": [{"type": "image", "media_index": 0}, "First text"],
        },
        {"role": "developer", "content": "Second text", "media": ["imgB"]},
        {"role": "user", "content": QUESTION},
    ]
    pf_conv, pf_counts = inputs_utils._merge_developers_into_leading_system(
        parts_first_conv, counts_first_mm
    )
    assert pf_counts == [{"<image>": 2}, {}]
    assert inputs_utils._build_openai_content(pf_conv[0], pf_counts[0]) == [
        {"type": "image"},
        {"type": "text", "text": "First text"},
        {"type": "text", "text": "\n\n"},
        {"type": "text", "text": "Second text"},
        {"type": "image"},
    ]
    assert pf_conv[0]["content_parts"] == [
        {"type": "image", "media_index": 0},
        "First text",
        "\n\n",
        "Second text",
        {"type": "image", "media_index": 1},
    ]


@pytest.mark.parametrize(
    "system_blocks,user_blocks",
    [(True, True), (True, False), (False, True)],
    ids=["both_blocks", "system_blocks_user_str", "system_str_user_blocks"],
)
def test_openai_block_content_template_probes_as_blocks(
    tmp_path, monkeypatch, system_blocks, user_blocks
):
    """Templates iterating over block content per role render plain-text chat developer and user."""
    sys_body = (
        "<|system|>{%- for p in m.content -%}{%- if p.type == 'text' -%}{{ p.text }}{%- endif -%}{%- endfor -%}"
        if system_blocks
        else "{{ '<|system|>' + m.content }}"
    )
    user_body = (
        "<|user|>{%- for p in m.content -%}{%- if p.type == 'text' -%}{{ p.text }}{%- endif -%}{%- endfor -%}"
        if user_blocks
        else "{{ '<|user|>' + m.content }}"
    )
    block_template = (
        "{%- for m in messages -%}"
        f"{{%- if m.role == 'system' -%}}{sys_body}"
        f"{{%- elif m.role == 'user' -%}}{user_body}"
        "{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
    )
    served = _serve(tmp_path, monkeypatch, _JinjaTokenizer(block_template))
    prompt = _post_chat(served)
    assert prompt == f"<|system|>{BRIEF}<|user|>{QUESTION}<|assistant|>"


def test_no_developer_block_template_skips_probes_and_preserves_caller_conversation(
    template_probes,
):
    """No-developer requests on block-format templates skip role probes and never mutate caller dicts."""
    block_template = (
        "{#- no-dev-block -#}{%- for m in messages -%}"
        "{%- if m.role == 'user' -%}"
        "<|user|>{%- if m.content is string -%}{{ m.content }}{%- else -%}"
        "{%- for p in m.content -%}{%- if p.type == 'text' -%}{{ p.text }}{%- endif -%}{%- endfor -%}"
        "{%- endif -%}{%- endif -%}{%- endfor -%}"
    )
    tokenizer = _JinjaTokenizer(block_template)
    conversation = [{"role": "user", "content": QUESTION}]

    rendered = apply_chat_template(
        model_type="llama",
        tokenizer=tokenizer,
        processor=None,
        conversation=conversation,
        add_generation_prompt=False,
        mm_placeholder_counts=[{}],
        chat_template=block_template,
    )

    assert rendered == f"<|user|>{QUESTION}"
    assert template_probes == []
    assert tokenizer.prompts() == [[("user", QUESTION)]]
    assert conversation == [{"role": "user", "content": QUESTION}]
    assert isinstance(conversation[0]["content"], str)


def test_openai_block_assistant_requires_leading_user_and_keeps_late_system_in_place(
    tmp_path, monkeypatch
):
    """Block-only assistant requiring first non-system=user preserves assistant text and late system."""
    template = (
        "{%- set ns = namespace(seen_user=false) -%}"
        "{%- for m in messages -%}"
        "{%- if m.role == 'system' -%}{{ '<|system|>' + m.content }}"
        "{%- elif m.role == 'user' -%}{%- set ns.seen_user = true -%}{{ '<|user|>' + m.content }}"
        "{%- elif m.role == 'assistant' -%}"
        "{%- if not ns.seen_user -%}{{ raise_exception('first non-system must be user') }}{%- endif -%}"
        "<|assistant|>{%- for p in m.content -%}{%- if p.type == 'text' -%}{{ p.text }}{%- endif -%}{%- endfor -%}"
        "{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
    )
    served = _serve(tmp_path, monkeypatch, _JinjaTokenizer(template))
    prompt = _post_chat(
        served,
        messages=[
            {"role": "user", "content": "First."},
            {"role": "assistant", "content": "Reply."},
            {"role": "developer", "content": BRIEF},
            {"role": "user", "content": QUESTION},
        ],
    )
    assert (
        prompt
        == f"<|user|>First.<|assistant|>Reply.<|system|>{BRIEF}<|user|>{QUESTION}<|assistant|>"
    )


def test_tools_context_separates_developer_probe_cache(tmp_path, monkeypatch):
    """A template with distinct behavior for tools=None, tools=[], and non-empty tools keeps all 3."""
    tools_dependent = (
        "{%- set ns = namespace(sc=0) -%}"
        "{%- for m in messages -%}"
        "{%- if tools and m.role == 'developer' -%}<|developer|>{{ m.content }}"
        "{%- elif m.role == 'system' -%}"
        "{%- set ns.sc = ns.sc + 1 -%}"
        "{%- if tools is none and ns.sc > 1 -%}{{ raise_exception('single system when tools is none') }}{%- endif -%}"
        "<|system|>{{ m.content }}"
        "{%- elif m.role == 'user' -%}<|user|>{{ m.content }}"
        "{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
    )
    tokenizer = _JinjaTokenizer(tools_dependent)
    served = _serve(tmp_path, monkeypatch, tokenizer)
    msgs = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "developer", "content": BRIEF},
        {"role": "user", "content": QUESTION},
    ]
    # tools=None: strict single system -> merges developer into system
    assert (
        _post_chat(served, messages=msgs)
        == f"<|system|>{INSTRUCTIONS}\n\n{BRIEF}<|user|>{QUESTION}<|assistant|>"
    )
    # tools=[]: allows multiple systems, drops developer -> two separate system blocks
    assert (
        apply_chat_template(
            model_type="llama",
            tokenizer=tokenizer,
            processor=None,
            conversation=[dict(m) for m in msgs],
            add_generation_prompt=True,
            mm_placeholder_counts=[{}, {}, {}],
            tools=[],
            chat_template=tools_dependent,
        )
        == f"<|system|>{INSTRUCTIONS}<|system|>{BRIEF}<|user|>{QUESTION}<|assistant|>"
    )
    # non-empty tools: renders developer directly -> keeps developer role
    assert (
        _post_chat(
            served,
            messages=msgs,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "ping",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )
        == f"<|system|>{INSTRUCTIONS}<|developer|>{BRIEF}<|user|>{QUESTION}<|assistant|>"
    )


def test_native_renderer_tools_context_separates_developer_probe_cache(tmp_path, monkeypatch):
    """A Python native renderer with distinct behavior for None, [], and non-empty tools keeps all 3."""

    class _ToolsDependentNative(_NativeRenderer):
        def apply_chat_template(
            self, conversation, tools=None, tokenize=False, add_generation_prompt=True, **kwargs
        ):
            self._record(conversation)
            if tools is None:
                if sum(1 for m in conversation if m["role"] == "system") > 1:
                    raise ValueError("single system when tools is None")
                roles = ("system", "user", "assistant")
            elif len(tools) == 0:
                roles = ("system", "user", "assistant")
            else:
                roles = ("system", "developer", "user", "assistant")
            text = "".join(
                f"<{m['role']}>{m['content']}" for m in conversation if m["role"] in roles
            )
            if add_generation_prompt:
                text += "<assistant>"
            return self.encode(text) if tokenize else text

    tokenizer = _ToolsDependentNative()
    served = _serve(tmp_path, monkeypatch, tokenizer)
    msgs = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "developer", "content": BRIEF},
        {"role": "user", "content": QUESTION},
    ]
    assert (
        _post_chat(served, messages=msgs)
        == f"<system>{INSTRUCTIONS}\n\n{BRIEF}<user>{QUESTION}<assistant>"
    )
    assert (
        apply_chat_template(
            model_type="llama",
            tokenizer=tokenizer,
            processor=None,
            conversation=[dict(m) for m in msgs],
            add_generation_prompt=True,
            mm_placeholder_counts=[{}, {}, {}],
            tools=[],
        )
        == f"<system>{INSTRUCTIONS}<system>{BRIEF}<user>{QUESTION}<assistant>"
    )
    assert (
        _post_chat(
            served,
            messages=msgs,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "ping",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )
        == f"<system>{INSTRUCTIONS}<developer>{BRIEF}<user>{QUESTION}<assistant>"
    )


def test_add_generation_prompt_flag_separates_probe_cache():
    """Renderers differing by add_generation_prompt preserve developer content for both True/False."""

    class _GenPromptDependentNative(_NativeRenderer):
        def apply_chat_template(
            self, conversation, tools=None, tokenize=False, add_generation_prompt=True, **kwargs
        ):
            self._record(conversation)
            roles = ("system", "developer", "user") if add_generation_prompt else ("system", "user")
            return "".join(
                f"<{m['role']}>{m['content']}" for m in conversation if m["role"] in roles
            )

    msgs = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "developer", "content": BRIEF},
        {"role": "user", "content": QUESTION},
    ]
    native = _GenPromptDependentNative()
    assert (
        apply_chat_template(
            model_type="llama",
            tokenizer=native,
            processor=None,
            conversation=[dict(m) for m in msgs],
            add_generation_prompt=True,
            mm_placeholder_counts=[{}, {}, {}],
        )
        == f"<system>{INSTRUCTIONS}<developer>{BRIEF}<user>{QUESTION}"
    )
    assert (
        apply_chat_template(
            model_type="llama",
            tokenizer=native,
            processor=None,
            conversation=[dict(m) for m in msgs],
            add_generation_prompt=False,
            mm_placeholder_counts=[{}, {}, {}],
        )
        == f"<system>{INSTRUCTIONS}<system>{BRIEF}<user>{QUESTION}"
    )

    jinja_tmpl = (
        "{%- for m in messages -%}"
        "{%- if m.role == 'system' -%}<|system|>{{ m.content }}"
        "{%- elif add_generation_prompt and m.role == 'developer' -%}<|developer|>{{ m.content }}"
        "{%- elif m.role == 'user' -%}<|user|>{{ m.content }}"
        "{%- endif -%}{%- endfor -%}"
    )
    jinja_tok = _JinjaTokenizer(jinja_tmpl)
    assert (
        apply_chat_template(
            model_type="llama",
            tokenizer=jinja_tok,
            processor=None,
            conversation=[dict(m) for m in msgs],
            add_generation_prompt=True,
            mm_placeholder_counts=[{}, {}, {}],
            chat_template=jinja_tmpl,
        )
        == f"<|system|>{INSTRUCTIONS}<|developer|>{BRIEF}<|user|>{QUESTION}"
    )
    assert (
        apply_chat_template(
            model_type="llama",
            tokenizer=jinja_tok,
            processor=None,
            conversation=[dict(m) for m in msgs],
            add_generation_prompt=False,
            mm_placeholder_counts=[{}, {}, {}],
            chat_template=jinja_tmpl,
        )
        == f"<|system|>{INSTRUCTIONS}<|system|>{BRIEF}<|user|>{QUESTION}"
    )


@pytest.mark.parametrize(
    "agent_first", [False, True], ids=["default_then_agent", "agent_then_default"]
)
def test_effective_rendering_options_drive_probe_and_cache_in_either_order(
    tmp_path, monkeypatch, agent_first
):
    """Probe and cache use effective chat_template_kwargs and tools for both Jinja and native."""
    persona_template = (
        "{%- for m in messages -%}"
        "{%- if persona is defined and persona == 'agent' -%}"
        "{%- if m.role == 'system' -%}{{ raise_exception('system rejected for agent') }}"
        "{%- elif m.role == 'developer' -%}<|developer|>{{ m.content }}"
        "{%- elif m.role == 'user' -%}<|user|>{{ m.content }}"
        "{%- endif -%}"
        "{%- else -%}"
        "{%- if m.role == 'system' -%}<|system|>{{ m.content }}"
        "{%- elif m.role == 'user' -%}<|user|>{{ m.content }}"
        "{%- endif -%}"
        "{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
    )
    served_jinja = _serve(
        tmp_path,
        monkeypatch,
        _JinjaTokenizer(f"{{#- persona-{agent_first} -#}}" + persona_template),
    )

    class _PersonaNative(_NativeRenderer):
        def apply_chat_template(
            self, conversation, tools=None, tokenize=False, add_generation_prompt=True, **kwargs
        ):
            self._record(conversation)
            if kwargs.get("persona") == "agent":
                for m in conversation:
                    if m["role"] == "system":
                        raise ValueError("system rejected for agent")
                roles = ("developer", "user")
            else:
                roles = ("system", "user")
            text = "".join(
                f"<{m['role']}>{m['content']}" for m in conversation if m["role"] in roles
            )
            if add_generation_prompt:
                text += "<assistant>"
            return self.encode(text) if tokenize else text

    served_native = _serve(tmp_path, monkeypatch, _PersonaNative())
    order = (True, False) if agent_first else (False, True)
    for use_agent in order:
        extra = {"chat_template_kwargs": {"persona": "agent"}} if use_agent else {}
        expected_j = (
            f"<|developer|>{BRIEF}<|user|>{QUESTION}<|assistant|>"
            if use_agent
            else f"<|system|>{BRIEF}<|user|>{QUESTION}<|assistant|>"
        )
        expected_n = (
            f"<developer>{BRIEF}<user>{QUESTION}<assistant>"
            if use_agent
            else f"<system>{BRIEF}<user>{QUESTION}<assistant>"
        )
        assert _post_chat(served_jinja, **extra) == expected_j
        assert _post_chat(served_native, **extra) == expected_n

    # Tool-name-dependent template must distinguish non-empty tool lists with different names.
    tool_name_template = (
        f"{{#- tool-name-{agent_first} -#}}"
        "{%- set special = tools and tools[0].function.name == 'special_tool' -%}"
        "{%- for m in messages -%}"
        "{%- if special and m.role == 'developer' -%}<|developer|>{{ m.content }}"
        "{%- elif special and m.role == 'system' -%}{{ raise_exception('system rejected for special') }}"
        "{%- elif (not special) and m.role == 'system' -%}<|system|>{{ m.content }}"
        "{%- elif m.role == 'user' -%}<|user|>{{ m.content }}"
        "{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
    )
    served_tools = _serve(tmp_path, monkeypatch, _JinjaTokenizer(tool_name_template))
    tool_variants = [
        ("special_tool", f"<|developer|>{BRIEF}<|user|>{QUESTION}<|assistant|>"),
        ("other_tool", f"<|system|>{BRIEF}<|user|>{QUESTION}<|assistant|>"),
    ]
    if not agent_first:
        tool_variants.reverse()
    for tool_name, expected in tool_variants:
        tool = {
            "type": "function",
            "function": {
                "name": tool_name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        assert _post_chat(served_tools, tools=[tool]) == expected

    # Dict insertion order in chat_template_kwargs must not collide in the probe cache.
    order_template = (
        f"{{#- dict-order-{agent_first} -#}}"
        "{%- set first_key = options | first -%}"
        "{%- for m in messages -%}"
        "{%- if first_key == 'dev' and m.role == 'developer' -%}<|developer|>{{ m.content }}"
        "{%- elif first_key == 'dev' and m.role == 'system' -%}{{ raise_exception('sys rejected') }}"
        "{%- elif first_key != 'dev' and m.role == 'system' -%}<|system|>{{ m.content }}"
        "{%- elif m.role == 'user' -%}<|user|>{{ m.content }}"
        "{%- endif -%}{%- endfor -%}"
    )
    order_tok = _JinjaTokenizer(order_template)
    order_variants = [
        ({"dev": 1, "sys": 2}, f"<|developer|>{BRIEF}<|user|>{QUESTION}"),
        ({"sys": 2, "dev": 1}, f"<|system|>{BRIEF}<|user|>{QUESTION}"),
    ]
    if not agent_first:
        order_variants.reverse()
    for opts, expected in order_variants:
        assert (
            apply_chat_template(
                model_type="llama",
                tokenizer=order_tok,
                processor=None,
                conversation=[
                    {"role": "developer", "content": BRIEF},
                    {"role": "user", "content": QUESTION},
                ],
                add_generation_prompt=False,
                mm_placeholder_counts=[{}, {}],
                chat_template=order_template,
                chat_template_kwargs={"options": opts},
            )
            == expected
        )


def test_probe_marker_in_documents_does_not_cause_false_positive():
    """Real documents containing the default probe marker must not mask dropped developer messages."""
    doc_template = (
        "{#- doc-collision -#}"
        "{%- if documents -%}{%- for d in documents -%}<|doc|>{{ d.text }}{%- endfor -%}{%- endif -%}"
        "{%- for m in messages -%}"
        "{%- if m.role == 'system' -%}<|system|>{{ m.content }}"
        "{%- elif m.role == 'user' -%}<|user|>{{ m.content }}"
        "{%- endif -%}{%- endfor -%}"
    )
    tokenizer = _JinjaTokenizer(doc_template)
    rendered = apply_chat_template(
        model_type="llama",
        tokenizer=tokenizer,
        processor=None,
        conversation=[
            {"role": "developer", "content": BRIEF},
            {"role": "user", "content": QUESTION},
        ],
        add_generation_prompt=False,
        mm_placeholder_counts=[{}, {}],
        documents=[{"text": PROBE}],
        chat_template=doc_template,
    )
    assert rendered == f"<|doc|>{PROBE}<|system|>{BRIEF}<|user|>{QUESTION}"


def test_unhashable_native_tokenizer_renders_without_type_error():
    """A custom native tokenizer defining __eq__ without __hash__ bypasses weak-key caching safely."""

    class _UnhashableNative(_NativeRenderer):
        def __eq__(self, other):
            return isinstance(other, _UnhashableNative)

        __hash__ = None

    tokenizer = _UnhashableNative()
    conv = [{"role": "developer", "content": BRIEF}, {"role": "user", "content": QUESTION}]
    rendered = apply_chat_template(
        model_type="llama",
        tokenizer=tokenizer,
        processor=None,
        conversation=conv,
        add_generation_prompt=True,
        mm_placeholder_counts=[{}, {}],
    )
    assert rendered == f"<system>{BRIEF}<user>{QUESTION}<assistant>"


def test_single_system_merge_preserves_message_level_tools():
    """Merging [developer, system-with-tools, user] preserves and combines message-level tools."""

    class _StrictSingleSystemWithMsgTools(_NativeRenderer):
        def apply_chat_template(
            self, conversation, tools=None, tokenize=False, add_generation_prompt=True, **kwargs
        ):
            self._record(conversation)
            if sum(1 for m in conversation if m["role"] == "system") > 1:
                raise ValueError("single system only")
            parts = []
            for m in conversation:
                if m["role"] not in self.ROLES:
                    continue
                tool_names = ",".join(t["name"] for t in m.get("tools", ()))
                suffix = f"[tools={tool_names}]" if tool_names else ""
                parts.append(f"<{m['role']}{suffix}>{m['content']}")
            return "".join(parts)

    tokenizer = _StrictSingleSystemWithMsgTools()
    t1 = {"name": "search"}
    t2 = {"name": "calc"}

    # Developer first, system with tools second -> source tools preserved
    rendered = apply_chat_template(
        model_type="llama",
        tokenizer=tokenizer,
        processor=None,
        conversation=[
            {"role": "developer", "content": BRIEF},
            {"role": "system", "content": INSTRUCTIONS, "tools": [t1]},
            {"role": "user", "content": QUESTION},
        ],
        add_generation_prompt=False,
        mm_placeholder_counts=[{}, {}, {}],
    )
    assert rendered == f"<system[tools=search]>{BRIEF}\n\n{INSTRUCTIONS}<user>{QUESTION}"

    # Both messages carry tools -> combined in order without duplicating identical entries
    rendered_both = apply_chat_template(
        model_type="llama",
        tokenizer=tokenizer,
        processor=None,
        conversation=[
            {"role": "developer", "content": BRIEF, "tools": [t1]},
            {"role": "system", "content": INSTRUCTIONS, "tools": [t1, t2]},
            {"role": "user", "content": QUESTION},
        ],
        add_generation_prompt=False,
        mm_placeholder_counts=[{}, {}, {}],
    )
    assert rendered_both == f"<system[tools=search,calc]>{BRIEF}\n\n{INSTRUCTIONS}<user>{QUESTION}"


def test_template_rendering_developer_only_at_index_zero_rewrites_nonleading_developer(
    tmp_path, monkeypatch
):
    """A template rendering developer only at loop.first rewrites non-leading developer as system."""
    first_only_dev = (
        "{%- for m in messages -%}"
        "{%- if m.role == 'system' -%}<|system|>{{ m.content }}"
        "{%- elif loop.first and m.role == 'developer' -%}<|developer|>{{ m.content }}"
        "{%- elif m.role == 'user' -%}<|user|>{{ m.content }}"
        "{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
    )
    served = _serve(tmp_path, monkeypatch, _JinjaTokenizer(first_only_dev))

    # Leading developer at index 0 stays developer unchanged
    assert _post_chat(served) == f"<|developer|>{BRIEF}<|user|>{QUESTION}<|assistant|>"

    # Responses [system(INSTRUCTIONS), developer(BRIEF), user(QUESTION)] rewrites non-leading dev
    resp_prompt = _post_responses(served)
    assert (
        resp_prompt == f"<|system|>{INSTRUCTIONS}<|system|>{BRIEF}<|user|>{QUESTION}<|assistant|>"
    )

    # A template accepting developer before the first user but dropping late developer rewrites late
    pre_user_dev = (
        "{%- set ns = namespace(seen_user=false) -%}"
        "{%- for m in messages -%}"
        "{%- if m.role == 'system' -%}<|system|>{{ m.content }}"
        "{%- elif not ns.seen_user and m.role == 'developer' -%}<|developer|>{{ m.content }}"
        "{%- elif m.role == 'user' -%}{%- set ns.seen_user = true -%}<|user|>{{ m.content }}"
        "{%- elif m.role == 'assistant' -%}<|assistant|>{{ m.content }}"
        "{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
    )
    served_pre = _serve(tmp_path, monkeypatch, _JinjaTokenizer(pre_user_dev))
    late_dev_prompt = _post_chat(
        served_pre,
        messages=[
            {"role": "user", "content": "First."},
            {"role": "assistant", "content": "Reply."},
            {"role": "developer", "content": BRIEF},
            {"role": "user", "content": QUESTION},
        ],
    )
    assert (
        late_dev_prompt
        == f"<|user|>First.<|assistant|>Reply.<|system|>{BRIEF}<|user|>{QUESTION}<|assistant|>"
    )

    # Leading developer + late developer in one request: only the late one becomes system
    mixed_dev_prompt = _post_chat(
        served_pre,
        messages=[
            {"role": "developer", "content": INSTRUCTIONS},
            {"role": "user", "content": "First."},
            {"role": "assistant", "content": "Reply."},
            {"role": "developer", "content": BRIEF},
            {"role": "user", "content": QUESTION},
        ],
    )
    assert mixed_dev_prompt == (
        f"<|developer|>{INSTRUCTIONS}<|user|>First.<|assistant|>Reply."
        f"<|system|>{BRIEF}<|user|>{QUESTION}<|assistant|>"
    )

    # When late developer forces a leading-system merge, an early developer supported by the
    # template keeps its developer role and position rather than being absorbed into system.
    single_sys_pre_user_dev = (
        "{%- set ns = namespace(seen_user=false) -%}"
        "{%- for m in messages -%}"
        "{%- if m.role == 'system' -%}"
        "{%- if not loop.first -%}{{ raise_exception('system only at 0') }}{%- endif -%}"
        "<|system|>{{ m.content }}"
        "{%- elif not ns.seen_user and m.role == 'developer' -%}<|developer|>{{ m.content }}"
        "{%- elif m.role == 'user' -%}{%- set ns.seen_user = true -%}<|user|>{{ m.content }}"
        "{%- elif m.role == 'assistant' -%}<|assistant|>{{ m.content }}"
        "{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
    )
    served_mixed_merge = _serve(tmp_path, monkeypatch, _JinjaTokenizer(single_sys_pre_user_dev))
    mixed_merge_prompt = _post_chat(
        served_mixed_merge,
        messages=[
            {"role": "system", "content": "SYS0"},
            {"role": "developer", "content": INSTRUCTIONS},
            {"role": "user", "content": "First."},
            {"role": "assistant", "content": "Reply."},
            {"role": "developer", "content": BRIEF},
            {"role": "user", "content": QUESTION},
        ],
    )
    assert mixed_merge_prompt == (
        f"<|system|>SYS0\n\n{BRIEF}<|developer|>{INSTRUCTIONS}"
        f"<|user|>First.<|assistant|>Reply.<|user|>{QUESTION}<|assistant|>"
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
