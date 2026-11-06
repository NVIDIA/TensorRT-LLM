# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ChatCompletionRequest's `continue_final_message`.

The flag asks the renderer to continue the final (assistant) message rather
than close it. It reaches the chat template through `chat_template_kwargs`,
where a caller may also set it. It cannot be combined with
`add_generation_prompt`: a renderer that does not read the flag (the native
DeepSeek renderers, for one) would silently start a new assistant turn.
"""

import pytest
from pydantic import ValidationError

from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
from tensorrt_llm.serve.render import render_chat

from .render_helpers import make_tokenizer, resources

pytestmark = pytest.mark.cpu_only

_PARTIAL_ANSWER = "The capital of France is"
_MESSAGES = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": _PARTIAL_ANSWER},
]


def _request(**fields) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="m", messages=_MESSAGES, **fields)


def test_unset_leaves_the_template_kwargs_alone():
    request = _request()
    assert request.continue_final_message is False
    assert request.chat_template_kwargs is None


def test_field_reaches_the_template_kwargs():
    request = _request(continue_final_message=True, add_generation_prompt=False)
    assert request.chat_template_kwargs == {"continue_final_message": True}


def test_field_takes_precedence_over_the_template_kwargs():
    request = _request(
        continue_final_message=False,
        chat_template_kwargs={"continue_final_message": True, "enable_thinking": False},
    )
    assert request.chat_template_kwargs == {
        "continue_final_message": False,
        "enable_thinking": False,
    }


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"continue_final_message": True}, id="field"),
        pytest.param({"chat_template_kwargs": {"continue_final_message": True}}, id="kwargs"),
    ],
)
def test_conflict_with_the_generation_prompt_is_rejected(fields):
    # add_generation_prompt defaults to True.
    with pytest.raises(ValidationError, match="Cannot set both"):
        _request(**fields)
    with pytest.raises(ValidationError, match="Cannot set both"):
        _request(add_generation_prompt=True, **fields)


def test_template_kwargs_without_the_generation_prompt_are_accepted():
    kwargs = {"continue_final_message": True}
    request = _request(add_generation_prompt=False, chat_template_kwargs=kwargs)
    assert request.chat_template_kwargs == kwargs


def test_forwarded_request_validates_the_same():
    """A disaggregated router forwards the set fields; the worker re-validates them."""
    request = _request(continue_final_message=True, add_generation_prompt=False)
    forwarded = ChatCompletionRequest(**request.model_dump(mode="json", exclude_unset=True))
    assert forwarded.continue_final_message is True
    assert forwarded.chat_template_kwargs == {"continue_final_message": True}


@pytest.mark.parametrize("continue_final_message", [False, True])
def test_hf_template_continues_the_final_message(continue_final_message):
    rendered = render_chat(
        _request(continue_final_message=continue_final_message, add_generation_prompt=False),
        resources(make_tokenizer()),
        tokenize=False,
    ).text
    assert rendered.endswith(_PARTIAL_ANSWER) == continue_final_message, rendered
    assert _PARTIAL_ANSWER in rendered
