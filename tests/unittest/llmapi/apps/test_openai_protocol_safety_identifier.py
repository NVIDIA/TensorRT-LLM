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
"""ChatCompletionRequest must accept the standard `safety_identifier` field.

The request schema is extra="forbid", so the field must exist for requests
from OpenAI clients that set it (it replaces the deprecated `user` field) to
validate. The value is not used.
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


@pytest.mark.parametrize("value", ["user-7f3a9c", ""])
def test_safety_identifier_accepted(value):
    req = _request(safety_identifier=value)
    assert req.safety_identifier == value


def test_safety_identifier_defaults_to_none():
    assert _request().safety_identifier is None


def test_safety_identifier_accepted_from_json():
    req = ChatCompletionRequest.model_validate_json(
        '{"model": "test-model", "messages": [{"role": "user", "content": "hi"}],'
        ' "safety_identifier": "user-7f3a9c"}'
    )
    assert req.safety_identifier == "user-7f3a9c"


@pytest.mark.parametrize("value", [1, ["user"], {"id": "user"}])
def test_safety_identifier_must_be_a_string(value):
    with pytest.raises(ValidationError):
        _request(safety_identifier=value)


def test_safety_identifier_leaves_other_fields_unchanged():
    plain = _request(user="legacy-user")
    tagged = _request(user="legacy-user", safety_identifier="user-7f3a9c")
    assert tagged.user == "legacy-user"
    assert tagged.model_dump(exclude={"safety_identifier"}) == plain.model_dump(
        exclude={"safety_identifier"}
    )


def test_unknown_fields_remain_rejected():
    with pytest.raises(ValidationError):
        _request(safety_identifiers="typo")
