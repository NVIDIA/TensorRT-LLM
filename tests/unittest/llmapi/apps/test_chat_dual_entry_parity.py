# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Both chat-rendering entry points must produce the same token stream.

A chat request is rendered in two places: ``serve/chat_tokenization.py``
(KV-aware routing and P-D disaggregation tokenize on the router) and
``openai_server.py`` via ``inputs/utils.py:apply_chat_template`` (the serving
path). The server normalizes messages through
``chat_utils.parse_chat_message_content`` first -- parsing string-typed
tool-call ``function.arguments`` into dicts, among other things -- while the
router path used to hand the raw request messages to the template. The same
tool-call request then rendered fine on the server and raised on the router.
These tests pin the two paths to identical output for the same request.

Also here: a rendered-token-delta test per supported chat template kwarg. A
kwarg the template never reads is silently accepted and silently does nothing;
one such no-op invalidated every banked reasoning-off measurement before an
eval anomaly exposed it. Two different values of a supported kwarg must render
differently, which costs no GPU to check.

Runs against the real checkpoint tokenizer (like
``test_musespark_open_tokenizer.py``); not wired into any L0 list.
"""

import json
import os
from types import SimpleNamespace

import pytest

from tensorrt_llm.inputs.utils import apply_chat_template
from tensorrt_llm.serve.chat_tokenization import render_chat_request_for_tokenizer
from tensorrt_llm.serve.chat_utils import parse_chat_messages_coroutines
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
from tensorrt_llm.tokenizer import load_custom_tokenizer

pytestmark = pytest.mark.cpu_only

# Full path to the NVFP4 reference checkpoint; unset means skip.
CHECKPOINT = os.environ.get("MUSESPARK_REFERENCE_CKPT", "")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

# The OpenAI wire format sends tool-call arguments as a JSON STRING; templates
# consume the dict form. Normalization is the difference under test.
TOOL_CALL_MESSAGES = [
    {"role": "user", "content": "What is the weather in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Paris"}',
                },
            }
        ],
    },
    {"role": "tool", "content": "sunny, 21 C", "tool_call_id": "call_1"},
    {"role": "user", "content": "And tomorrow?"},
]


@pytest.fixture(scope="module")
def tokenizer():
    if not CHECKPOINT or not os.path.isdir(CHECKPOINT):
        pytest.skip("checkpoint not present; set MUSESPARK_OPEN_ROOT")
    return load_custom_tokenizer("musespark_open", CHECKPOINT)


@pytest.fixture(scope="module")
def model_type():
    with open(os.path.join(CHECKPOINT, "config.json")) as fp:
        return json.load(fp)["model_type"]


def _render_via_router(request, tokenizer, model_type):
    """Entry point 1: serve/chat_tokenization.py (KV-aware routing path)."""
    return render_chat_request_for_tokenizer(request, tokenizer, model_type=model_type)


def _render_via_server(request, tokenizer, model_type):
    """Entry point 2: openai_server.py's parse + apply_chat_template path."""
    model_config = SimpleNamespace(model_type=model_type)
    (conversation, mm_coroutines, mm_placeholder_counts, _) = parse_chat_messages_coroutines(
        request.messages, model_config
    )
    mm_coroutines.close()  # text-only request; nothing to await
    tool_dicts = (
        [tool.model_dump() for tool in request.tools] if request.tools is not None else None
    )
    return apply_chat_template(
        model_type=model_type,
        tokenizer=tokenizer,
        processor=None,
        conversation=conversation,
        add_generation_prompt=request.add_generation_prompt,
        mm_placeholder_counts=mm_placeholder_counts,
        tools=tool_dicts,
        documents=None,
        chat_template=None,
        chat_template_kwargs=request.chat_template_kwargs or {},
    )


def _request(**overrides):
    payload = {"model": "musespark_open", "messages": TOOL_CALL_MESSAGES, "tools": TOOLS}
    payload.update(overrides)
    return ChatCompletionRequest(**payload)


def test_dual_entry_points_render_identical_tokens(tokenizer, model_type):
    """String tool-call arguments must render the same through both paths."""
    request = _request()
    via_router = _render_via_router(_request(), tokenizer, model_type)
    via_server = _render_via_server(request, tokenizer, model_type)
    assert isinstance(via_router, str) and isinstance(via_server, str)
    assert via_router == via_server
    router_ids = tokenizer.encode(via_router, add_special_tokens=False)
    server_ids = tokenizer.encode(via_server, add_special_tokens=False)
    assert router_ids == server_ids
    assert len(router_ids) > 0


def test_dual_entry_points_agree_with_kwargs(tokenizer, model_type):
    """Parity must hold with template kwargs applied, not only for defaults."""
    kwargs = {"disable_reasoning": True}
    via_router = _render_via_router(_request(chat_template_kwargs=kwargs), tokenizer, model_type)
    via_server = _render_via_server(_request(chat_template_kwargs=kwargs), tokenizer, model_type)
    assert via_router == via_server


@pytest.mark.parametrize(
    "kwarg,value_a,value_b",
    [
        ("disable_reasoning", False, True),
        ("reasoning_effort", "low", "high"),
    ],
)
def test_supported_template_kwargs_change_the_rendered_tokens(
    tokenizer, model_type, kwarg, value_a, value_b
):
    """A supported kwarg must be live: two values, two renderings.

    A template that silently ignores a kwarg turns every request that sets it
    into a no-op -- accepted, unrendered and invisible until an eval anomaly.
    Rendering with two different values and asserting a token delta proves the
    template actually reads it, for the cost of two CPU renders.
    """
    render_a = _render_via_router(
        _request(chat_template_kwargs={kwarg: value_a}), tokenizer, model_type
    )
    render_b = _render_via_router(
        _request(chat_template_kwargs={kwarg: value_b}), tokenizer, model_type
    )
    assert render_a != render_b, (
        f"chat_template_kwargs[{kwarg!r}] = {value_a!r} and {value_b!r} render "
        "identically; the template never reads it and every request setting it "
        "is a silent no-op"
    )
    ids_a = tokenizer.encode(render_a, add_special_tokens=False)
    ids_b = tokenizer.encode(render_b, add_special_tokens=False)
    assert ids_a != ids_b
