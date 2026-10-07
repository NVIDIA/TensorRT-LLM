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
"""ChatCompletionRequest must accept the standard `parallel_tool_calls` field.

The request schema is extra="forbid", so the field must exist for requests
from stock OpenAI clients (which send it by default alongside `tools`) to
validate.
"""

import pytest

from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

pytestmark = pytest.mark.cpu_only

_TOOL = {
    "type": "function",
    "function": {
        "name": "record_value",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    },
}


def _request(**extra) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        **extra,
    )


@pytest.mark.parametrize("value", [True, False])
def test_parallel_tool_calls_accepted(value):
    req = _request(tools=[_TOOL], tool_choice="auto", parallel_tool_calls=value)
    assert req.parallel_tool_calls is value


def test_parallel_tool_calls_defaults_to_none():
    req = _request(tools=[_TOOL])
    assert req.parallel_tool_calls is None


def test_parallel_tool_calls_without_tools_accepted():
    # OpenAI clients may send the field even on tool-less requests.
    req = _request(parallel_tool_calls=True)
    assert req.parallel_tool_calls is True
