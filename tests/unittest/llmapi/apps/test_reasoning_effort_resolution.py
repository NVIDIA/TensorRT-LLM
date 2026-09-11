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
"""Forwarding `reasoning_effort` to the chat template.

Only `chat_template_kwargs` reaches the renderer, so a template that gates
reasoning on `reasoning_effort` never sees the top-level field. The value is
forwarded unchanged: a template that does not accept a level rejects it
itself, so no substitute is guessed on the caller's behalf.
"""

import pytest

from tensorrt_llm.serve.harmony_adapter import maybe_transform_reasoning_effort
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
from tensorrt_llm.serve.openai_server import _chat_template_kwargs_with_effort

pytestmark = pytest.mark.cpu_only

LEVELS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]


def make_request(**fields) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], **fields)


@pytest.mark.parametrize("effort", LEVELS)
def test_a_sent_level_reaches_the_template_unchanged(effort):
    kwargs = _chat_template_kwargs_with_effort(make_request(reasoning_effort=effort))
    assert kwargs["reasoning_effort"] == effort


def test_an_unsent_effort_is_not_forwarded():
    """The field defaults to LOW for the harmony path.

    Forwarding that default would hand every request a level its template may
    not accept, so absence has to be detected rather than read off the value.
    """
    request = make_request()
    assert "reasoning_effort" not in request.model_fields_set
    assert _chat_template_kwargs_with_effort(request) == {}


def test_a_sent_level_overrides_chat_template_kwargs():
    """Matches vLLM's merge order, where the field wins."""
    kwargs = _chat_template_kwargs_with_effort(
        make_request(reasoning_effort="max", chat_template_kwargs={"reasoning_effort": "high"})
    )
    assert kwargs["reasoning_effort"] == "max"


def test_chat_template_kwargs_still_work_on_their_own():
    """How the plugin passes it today; must keep working untouched."""
    kwargs = _chat_template_kwargs_with_effort(
        make_request(chat_template_kwargs={"reasoning_effort": "high"})
    )
    assert kwargs["reasoning_effort"] == "high"


def test_other_template_kwargs_are_preserved():
    kwargs = _chat_template_kwargs_with_effort(
        make_request(
            reasoning_effort="high", chat_template_kwargs={"enable_thinking": True, "custom": 1}
        )
    )
    assert kwargs["enable_thinking"] is True
    assert kwargs["custom"] == 1
    assert kwargs["reasoning_effort"] == "high"


def test_the_request_kwargs_are_not_mutated():
    sent = {"custom": 1}
    _chat_template_kwargs_with_effort(
        make_request(reasoning_effort="none", chat_template_kwargs=sent)
    )
    assert sent == {"custom": 1}


@pytest.mark.parametrize("effort", LEVELS)
def test_harmony_tolerates_every_level_the_field_accepts(effort):
    """Widening the field must not 500 a GPT-OSS deployment.

    The harmony transform indexed a dict, so a level it did not know raised
    KeyError instead of falling through to unspecified.
    """
    maybe_transform_reasoning_effort(effort)


def test_harmony_still_maps_the_levels_it_owns():
    assert maybe_transform_reasoning_effort("none") is None
    assert maybe_transform_reasoning_effort("xhigh") is maybe_transform_reasoning_effort("max")
    for known in ("low", "medium", "high"):
        assert maybe_transform_reasoning_effort(known) is not None
