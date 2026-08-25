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
"""End-to-end checks for resolving reasoning mode from the rendered prompt.

`openai_server` reads the mode off the rendered prompt and writes it into
`ChatPostprocArgs.chat_template_kwargs`; `postprocess_handlers` builds the
parser from that field. The two halves live in different modules, so this
pins the contract between them against a template that prefills the marker.
"""

import jinja2
import pytest

from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams
from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory
from tensorrt_llm.serve.postprocess_handlers import (
    ChatPostprocArgs,
    apply_reasoning_parser,
    chat_response_post_processor,
)

pytestmark = pytest.mark.cpu_only

# Mirrors the shape of the Laguna templates: the marker goes into the prompt,
# so it never appears in the output and the request kwargs cannot reveal it.
PREFILLING_TEMPLATE = (
    "{%- set enable_thinking = enable_thinking | default(true) -%}"
    "{%- for message in messages -%}"
    "{{- '<|' + message['role'] + '|>' + message['content'] -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<|assistant|>' -}}"
    "{%- if enable_thinking -%}{{- '<think>' -}}"
    "{%- else -%}{{- '</think>' -}}{%- endif -%}"
    "{%- endif -%}"
)

MESSAGES = [{"role": "user", "content": "hi"}]


def render(chat_template_kwargs: dict | None, add_generation_prompt: bool = True) -> str:
    template = jinja2.Environment().from_string(PREFILLING_TEMPLATE)
    return template.render(
        messages=MESSAGES,
        add_generation_prompt=add_generation_prompt,
        **(chat_template_kwargs or {}),
    )


def build_args(
    rendered_prompt: str | None, request_kwargs: dict | None, add_generation_prompt: bool = True
) -> ChatPostprocArgs:
    """Mirror what `openai_server` does after rendering the prompt."""
    args = ChatPostprocArgs(
        role="assistant", model="test-model", chat_template_kwargs=request_kwargs
    )
    args.reasoning_parser = "poolside_v1"
    args.num_prompt_tokens = 3
    if args.reasoning_parser and rendered_prompt and add_generation_prompt:
        thinking = ReasoningParserFactory.resolve_prefilled_thinking(
            args.reasoning_parser, rendered_prompt
        )
        if thinking is not None:
            args.chat_template_kwargs = {
                **(request_kwargs or {}),
                "thinking": thinking,
                "enable_thinking": thinking,
            }
    return args


@pytest.mark.parametrize(
    ("request_kwargs", "reasoning", "content"),
    [
        (None, "hidden", "visible"),
        ({"enable_thinking": True}, "hidden", "visible"),
    ],
)
def test_thinking_template_splits_reasoning(
    request_kwargs: dict[str, bool] | None, reasoning: str, content: str
) -> None:
    """A bare request must still land in the mode the template rendered."""
    prompt = render(request_kwargs)
    assert prompt.endswith("<think>")

    args = build_args(prompt, request_kwargs)
    got_content, got_reasoning = apply_reasoning_parser(
        args, 0, "hidden</think>visible", streaming=False
    )

    assert got_content == content
    assert got_reasoning == reasoning


def test_non_thinking_template_keeps_everything_as_content() -> None:
    prompt = render({"enable_thinking": False})
    assert prompt.endswith("</think>")

    args = build_args(prompt, {"enable_thinking": False})
    content, reasoning = apply_reasoning_parser(args, 0, "visible", streaming=False)

    assert content == "visible"
    assert not reasoning


def test_streaming_path_uses_the_same_resolved_mode() -> None:
    prompt = render(None)
    args = build_args(prompt, None)

    deltas = [
        apply_reasoning_parser(args, 0, chunk, streaming=True)
        for chunk in ("hid", "den</think>vis", "ible")
    ]

    assert "".join(c for c, _ in deltas if c) == "visible"
    assert "".join(r for _, r in deltas if r) == "hidden"


class _FakeOutput:
    """Minimal stand-in for one entry of `GenerationResultBase.outputs`."""

    def __init__(self, text: str) -> None:
        self.index = 0
        self.text = text
        self.token_ids = []
        self.length = 0
        self.finish_reason = "stop"
        self.stop_reason = None
        self.logprobs = None
        self.disaggregated_params = LlmDisaggregatedParams(request_type="context_only")


class _FakeResult:
    def __init__(self, text: str) -> None:
        self.outputs = [_FakeOutput(text)]
        self.prompt_token_ids = [1, 2, 3]
        self.cached_tokens = 0


@pytest.mark.parametrize(
    ("template_kwargs", "expected"),
    [(None, True), ({"enable_thinking": False}, False)],
)
def test_context_worker_stamps_the_resolved_mode(
    template_kwargs: dict[str, bool] | None, expected: bool
) -> None:
    """Calls the real handler, so dropping the stamp fails this test."""
    args = build_args(render(template_kwargs), template_kwargs)

    response = chat_response_post_processor(_FakeResult("hidden</think>visible"), args)

    assert response.choices[0].disaggregated_params.resolved_thinking is expected


def test_context_worker_does_not_stamp_for_other_parsers() -> None:
    """Only parsers that resolve from the prompt should relay a mode."""
    args = build_args(render(None), None)
    args.reasoning_parser = "deepseek_v4"

    response = chat_response_post_processor(_FakeResult("visible"), args)

    assert response.choices[0].disaggregated_params.resolved_thinking is None


@pytest.mark.parametrize("relayed", [True, False])
def test_relayed_mode_drives_the_parser(relayed: bool) -> None:
    """Contract the generation worker relies on, given a relayed value.

    The generation-side block lives in `openai_server` and needs a live
    server, so this pins what it feeds the parser rather than the wiring.
    """
    args = ChatPostprocArgs(
        role="assistant",
        model="test-model",
        chat_template_kwargs={"thinking": relayed, "enable_thinking": relayed},
    )
    args.reasoning_parser = "poolside_v1"
    content, reasoning = apply_reasoning_parser(args, 0, "hidden</think>visible", streaming=False)

    if relayed:
        assert (content, reasoning) == ("visible", "hidden")
    else:
        assert content == "hidden</think>visible"
        assert not reasoning


def test_unrendered_prompt_falls_back_without_crashing() -> None:
    """Disagg generation servers get `prompt_token_ids` and never render."""
    args = build_args(None, None)

    content, reasoning = apply_reasoning_parser(args, 0, "a<think>b</think>c", streaming=False)

    # Unresolved, so the pre-existing split on an emitted `<think>` applies.
    assert content == "c"
    assert reasoning == "b"
