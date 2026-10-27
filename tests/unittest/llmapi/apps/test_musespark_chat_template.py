# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Muse Spark chat template through the serve's render pipeline.

A kwarg the template never reads is accepted and does nothing, so two
different efforts must render differently, and the request-level
``reasoning_effort`` must reach the template the same way an explicit kwarg
does. Runs against the checkpoint's own tokenizer and chat template under
``LLM_MODELS_ROOT``; skips when the checkpoint is absent.
"""

import os

import pytest
from utils.llm_data import llm_models_root

from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
from tensorrt_llm.serve.render import RenderResources, render_chat
from tensorrt_llm.tokenizer import load_hf_tokenizer

pytestmark = pytest.mark.cpu_only

CHECKPOINT = "Muse-Spark-Open-816B-A42B-NVFP4"

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

# The OpenAI wire format sends tool-call arguments as a JSON string; the
# template consumes the parsed form.
TOOL_CALL_MESSAGES = [
    {"role": "user", "content": "What is the weather in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ],
    },
    {"role": "tool", "content": "sunny, 21 C", "tool_call_id": "call_1"},
    {"role": "user", "content": "And tomorrow?"},
]


@pytest.fixture(scope="module")
def resources():
    root = llm_models_root()
    if root is None:
        pytest.skip("LLM_MODELS_ROOT is not available")
    path = os.path.join(root, CHECKPOINT)
    if not os.path.isdir(path) or not os.listdir(path):
        pytest.skip(f"{CHECKPOINT} is not under LLM_MODELS_ROOT")
    tokenizer = load_hf_tokenizer(path)
    assert tokenizer is not None, f"could not load a tokenizer from {path}"
    return RenderResources.from_tokenizer(tokenizer, model_type="muse_spark")


def _render(resources, **overrides) -> str:
    payload = {"model": "muse_spark", "messages": [{"role": "user", "content": "hi"}]}
    payload.update(overrides)
    rendered = render_chat(ChatCompletionRequest(**payload), resources, tokenize=False)
    assert isinstance(rendered.text, str) and rendered.text
    return rendered.text


def test_reasoning_effort_kwarg_changes_the_rendering(resources):
    low = _render(resources, chat_template_kwargs={"reasoning_effort": "low"})
    high = _render(resources, chat_template_kwargs={"reasoning_effort": "high"})
    assert low != high, "the template never reads chat_template_kwargs['reasoning_effort']"


@pytest.mark.parametrize("effort", ["low", "high", "xhigh"])
def test_request_level_effort_renders_like_the_kwarg(resources, effort):
    assert _render(resources, reasoning_effort=effort) == _render(
        resources, chat_template_kwargs={"reasoning_effort": effort}
    )


def test_protocol_effort_without_a_template_level_is_normalized(resources):
    # The template has no `max`; the extension maps it to its top level.
    assert _render(resources, reasoning_effort="max") == _render(
        resources, chat_template_kwargs={"reasoning_effort": "xhigh"}
    )


def test_tool_calls_with_string_arguments_render(resources):
    text = _render(resources, messages=TOOL_CALL_MESSAGES, tools=TOOLS)
    assert "get_weather" in text
    assert "Paris" in text
