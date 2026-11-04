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
"""Where DeepSeek-V3.2's chat template puts the ``tools`` argument."""

import pytest

from tensorrt_llm.tokenizer.deepseek_v32 import DeepseekV32Tokenizer, encode_messages

pytestmark = pytest.mark.cpu_only

_TOOLS = [
    {
        "type": "function",
        "function": {"name": "search", "description": "Search", "parameters": {"type": "object"}},
    }
]


class _DummyTokenizer:
    all_special_tokens = []


@pytest.mark.parametrize("role", ["system", "developer"])
def test_tools_argument_joins_the_leading_system_message(role):
    """The encoder renders tools from the leading system or developer message."""
    conversation = [{"role": role, "content": "Be brief."}, {"role": "user", "content": "hi"}]

    prompt = DeepseekV32Tokenizer(_DummyTokenizer()).apply_chat_template(conversation, tools=_TOOLS)

    expected = [dict(conversation[0], tools=_TOOLS), conversation[1]]
    assert prompt == encode_messages(expected, thinking_mode="chat", drop_thinking=False)
    assert prompt.count("## Tools") == 1
    assert "tools" not in conversation[0]


def test_tools_argument_follows_the_system_prompt():
    conversation = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]

    prompt = DeepseekV32Tokenizer(_DummyTokenizer()).apply_chat_template(conversation, tools=_TOOLS)

    assert prompt.index("Be brief.") < prompt.index("## Tools")


def test_tools_argument_without_a_system_message_gets_one():
    conversation = [{"role": "user", "content": "hi"}]

    prompt = DeepseekV32Tokenizer(_DummyTokenizer()).apply_chat_template(conversation, tools=_TOOLS)

    expected = [{"role": "system", "tools": _TOOLS}, conversation[0]]
    assert prompt == encode_messages(expected, thinking_mode="chat", drop_thinking=False)
