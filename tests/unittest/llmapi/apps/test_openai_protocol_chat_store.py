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
"""ChatCompletionRequest must accept the standard OpenAI `store` field.

The request schema is extra="forbid", so a request from an OpenAI client that
sets `store` (the pi coding agent sends `store: false` on every Chat
Completions request by default) failed validation with HTTP 400. The field is
accepted and not used: the server keeps no copy of a chat completion.
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


@pytest.mark.parametrize("value", [False, True, None])
def test_store_accepted(value):
    assert _request(store=value).store is value


def test_store_defaults_to_none():
    assert _request().store is None


def test_store_accepted_from_json():
    req = ChatCompletionRequest.model_validate_json(
        '{"model": "test-model", "messages": [{"role": "user", "content": "hi"}],'
        ' "store": false, "stream": true}'
    )
    assert req.store is False
    assert req.stream is True


@pytest.mark.parametrize(
    "value, expected", [("false", False), ("true", True), (0, False), (1, True)]
)
def test_store_coerces_boolean_like_values(value, expected):
    # The request schema is not strict, so a boolean-like string or integer is
    # coerced, as for every other boolean field of the request.
    assert _request(store=value).store is expected


def test_store_rejects_a_value_that_is_not_boolean_like():
    with pytest.raises(ValidationError):
        _request(store={"keep": True})


def test_store_leaves_other_fields_unchanged():
    plain = _request(temperature=0.5)
    stored = _request(temperature=0.5, store=False)
    assert stored.model_dump(exclude={"store"}) == plain.model_dump(exclude={"store"})


def test_unknown_fields_remain_rejected():
    with pytest.raises(ValidationError):
        _request(stores=False)
