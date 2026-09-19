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
"""CPU-only unit tests for the MuseSpark serving extensions.

Covers the request-schema acceptance of the extended reasoning-effort scale
(`minimal`/`xhigh`), the routing of a request-level `reasoning_effort`
into the chat-template kwargs for musespark_open_omni deployments, whose chat
template reads the effort from its kwargs only, and the Responses-API
reasoning routing, which must not inject template kwargs the active
template never reads. No GPU or checkpoint required.
"""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tensorrt_llm.inputs.chat_template_guard import validate_chat_template_kwargs
from tensorrt_llm.serve.harmony_adapter import maybe_transform_reasoning_effort
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest, ResponsesRequest
from tensorrt_llm.serve.openai_server import _apply_musespark_chat_extensions
from tensorrt_llm.serve.responses_utils import reasoning_chat_template_kwargs

# The L0 CPU stage invokes registered files with `-m cpu_only`; without this
# marker every test is deselected and pytest exits with code 5.
pytestmark = pytest.mark.cpu_only

USER_MSG = {"role": "user", "content": "hello"}

MUSESPARK_EFFORTS = ["minimal", "low", "medium", "high", "xhigh"]


def make_request(**kwargs) -> ChatCompletionRequest:
    kwargs.setdefault("model", "musespark_open")
    kwargs.setdefault("messages", [USER_MSG])
    return ChatCompletionRequest(**kwargs)


class TestReasoningEffortSchema:
    """The request schema accepts every effort the templates support."""

    @pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh", "max", "none"])
    def test_all_supported_efforts_accepted(self, effort: str) -> None:
        req = make_request(reasoning_effort=effort)
        assert req.reasoning_effort == effort

    def test_unknown_effort_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_request(reasoning_effort="xxhigh")


class TestMuseSparkEffortRouting:
    def apply(self, model_type: str = "musespark_open_omni", **kwargs) -> ChatCompletionRequest:
        req = make_request(**kwargs)
        _apply_musespark_chat_extensions(req, model_type)
        return req

    def kwargs_of(self, req: ChatCompletionRequest) -> dict:
        return req.chat_template_kwargs or {}

    @pytest.mark.parametrize("effort", MUSESPARK_EFFORTS)
    def test_top_level_effort_routed_to_template_kwargs(self, effort: str) -> None:
        req = self.apply(reasoning_effort=effort)
        assert self.kwargs_of(req)["reasoning_effort"] == effort

    def test_non_musespark_untouched(self) -> None:
        req = self.apply(model_type="llama", reasoning_effort="xhigh")
        assert req.chat_template_kwargs is None

    def test_kimi_model_untouched(self) -> None:
        # The kimi path has its own derivation; this one must not add a
        # MuseSpark-style kwarg there.
        req = self.apply(model_type="kimi_k3", reasoning_effort="high")
        assert req.chat_template_kwargs is None

    def test_default_effort_not_derived(self) -> None:
        # The field defaults to LOW via pydantic; a client that never sent
        # reasoning_effort must not have an effort injected.
        req = self.apply()
        assert req.chat_template_kwargs is None

    def test_explicit_null_effort_not_derived(self) -> None:
        req = self.apply(reasoning_effort=None)
        assert req.chat_template_kwargs is None

    @pytest.mark.parametrize("effort", ["max", "none"])
    def test_efforts_without_musespark_equivalent_left_to_template(self, effort: str) -> None:
        req = self.apply(reasoning_effort=effort)
        assert req.chat_template_kwargs is None

    def test_client_chat_template_kwargs_win(self) -> None:
        req = self.apply(reasoning_effort="xhigh", chat_template_kwargs={"reasoning_effort": "low"})
        assert self.kwargs_of(req)["reasoning_effort"] == "low"

    def test_other_client_kwargs_preserved(self) -> None:
        req = self.apply(reasoning_effort="xhigh", chat_template_kwargs={"foo": "bar"})
        assert self.kwargs_of(req) == {"reasoning_effort": "xhigh", "foo": "bar"}


# A template in the MuseSpark style: it reads the effort level from its
# kwargs but implements no thinking toggle.
TEMPLATE_EFFORT_ONLY = (
    "{%- if reasoning_effort %}Reasoning: {{ reasoning_effort }}.{%- endif %}"
    "{%- for message in messages %}{{ message['content'] }}{%- endfor %}"
)

# A template in the DeepSeek-V4 style: it reads both the thinking toggle and
# the effort level.
TEMPLATE_EFFORT_AND_THINKING = (
    "{%- if thinking %}<think>{%- endif %}"
    "{%- if reasoning_effort %}Reasoning: {{ reasoning_effort }}.{%- endif %}"
    "{%- for message in messages %}{{ message['content'] }}{%- endfor %}"
)


def make_responses_request(**kwargs) -> ResponsesRequest:
    kwargs.setdefault("model", "musespark_open")
    kwargs.setdefault("input", "hello")
    return ResponsesRequest(**kwargs)


class TestResponsesReasoningKwargsInjection:
    """/v1/responses reasoning routing is template-aware.

    The routing maps `reasoning.effort` into chat_template_kwargs. It must
    only inject keys the active template references: the render-site guard
    rejects unreferenced kwargs with a 400, and a client cannot fix a 400
    caused by a kwarg the serve itself constructed. A `/v1/responses` request
    carrying `reasoning` against an effort-only template used to fail exactly
    this way.
    """

    def test_effort_only_template_gets_no_thinking_key(self):
        req = make_responses_request(reasoning={"effort": "high"})
        kwargs = reasoning_chat_template_kwargs(req, template_source=TEMPLATE_EFFORT_ONLY)
        assert kwargs == {"reasoning_effort": "high"}

    def test_effort_only_template_injection_passes_the_guard(self):
        # The previously failing path end to end: what the routing constructs
        # for a reasoning-bearing Responses request must pass the same guard
        # check the render site applies (a raise here is the 400).
        req = make_responses_request(reasoning={"effort": "high"})
        kwargs = reasoning_chat_template_kwargs(req, template_source=TEMPLATE_EFFORT_ONLY)
        validate_chat_template_kwargs(TEMPLATE_EFFORT_ONLY, kwargs)

    def test_thinking_template_keeps_both_keys(self):
        req = make_responses_request(reasoning={"effort": "high"})
        kwargs = reasoning_chat_template_kwargs(req, template_source=TEMPLATE_EFFORT_AND_THINKING)
        assert kwargs == {"reasoning_effort": "high", "thinking": True}
        validate_chat_template_kwargs(TEMPLATE_EFFORT_AND_THINKING, kwargs)

    def test_no_template_source_keeps_full_injection(self):
        # The reasoning-parser call sites and the custom non-Jinja tokenizers
        # pass no template source; they must keep the full derivation.
        req = make_responses_request(reasoning={"effort": "high"})
        assert reasoning_chat_template_kwargs(req) == {
            "reasoning_effort": "high",
            "thinking": True,
        }

    def test_client_sent_kwargs_are_not_filtered(self):
        # Filtering covers only the serve's own injection. A kwarg the client
        # sent is forwarded untouched, and the guard still rejects it when
        # the template never reads it - that rejection is correct and stays.
        req = SimpleNamespace(
            chat_template_kwargs={"thinking": True},
            reasoning=SimpleNamespace(effort="high"),
        )
        kwargs = reasoning_chat_template_kwargs(req, template_source=TEMPLATE_EFFORT_ONLY)
        assert kwargs == {"thinking": True, "reasoning_effort": "high"}
        with pytest.raises(ValueError, match="thinking"):
            validate_chat_template_kwargs(TEMPLATE_EFFORT_ONLY, kwargs)

    def test_request_without_reasoning_injects_nothing(self):
        req = make_responses_request()
        assert reasoning_chat_template_kwargs(req, template_source=TEMPLATE_EFFORT_ONLY) == {}


class TestHarmonyTransformCoversExtendedScale:
    """No KeyError from `maybe_transform_reasoning_effort` on accepted efforts.

    It runs on harmony-only paths, where the extended-scale values map to the
    nearest harmony level.
    """

    @pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh", "max", "none"])
    def test_no_keyerror_for_any_accepted_effort(self, effort: str) -> None:
        maybe_transform_reasoning_effort(effort)

    def test_extended_efforts_map_to_nearest_harmony_level(self) -> None:
        from openai_harmony import ReasoningEffort

        assert maybe_transform_reasoning_effort("minimal") == ReasoningEffort.LOW
        assert maybe_transform_reasoning_effort("xhigh") == ReasoningEffort.HIGH
