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
"""ChatCompletionRequest must accept the standard `prompt_cache_key` field.

The request schema is extra="forbid", so the field must exist for requests
from OpenAI clients that set it (for example agent frameworks that pass a
per-session key) to validate. The value is not used.
"""

import pytest
from pydantic import ValidationError

from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

pytestmark = pytest.mark.cpu_only


def _request(**extra) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        **extra,
    )


@pytest.mark.parametrize("value", ["session-1234", ""])
def test_prompt_cache_key_accepted(value):
    req = _request(prompt_cache_key=value)
    assert req.prompt_cache_key == value


def test_prompt_cache_key_defaults_to_none():
    assert _request().prompt_cache_key is None


def test_prompt_cache_key_accepted_from_json():
    req = ChatCompletionRequest.model_validate_json(
        '{"model": "test-model", "messages": [{"role": "user", "content": "hi"}],'
        ' "prompt_cache_key": "session-1234"}'
    )
    assert req.prompt_cache_key == "session-1234"


@pytest.mark.parametrize("value", [1, ["session"], {"key": "session"}])
def test_prompt_cache_key_must_be_a_string(value):
    with pytest.raises(ValidationError):
        _request(prompt_cache_key=value)


def test_prompt_cache_key_leaves_other_fields_unchanged():
    plain = _request(cache_salt="tenant")
    keyed = _request(cache_salt="tenant", prompt_cache_key="session-1234")
    assert keyed.cache_salt == "tenant"
    assert keyed.model_dump(exclude={"prompt_cache_key"}) == plain.model_dump(
        exclude={"prompt_cache_key"}
    )


def test_unknown_fields_remain_rejected():
    with pytest.raises(ValidationError):
        _request(prompt_cache_keys="typo")
