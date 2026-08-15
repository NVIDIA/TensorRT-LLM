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

from tensorrt_llm.inputs.utils import apply_chat_template
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.tokenizer.deepseek_v4 import DeepseekV4Tokenizer


class _DummyTokenizer:
    all_special_tokens = []
    eos_token_id = 1
    pad_token_id = 0
    name_or_path = "dummy"

    def encode(self, text, *args, **kwargs):
        self.last_encoded_text = text
        return [1, 2, 3]


def test_deepseek_v4_chat_template_matches_reference_single_user_prompt():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Question: 1+1?\nAnswer:",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert prompt == (
        "<｜begin▁of▁sentence｜><｜User｜>Question: 1+1?\nAnswer:<｜Assistant｜></think>"
    )


def test_deepseek_v4_chat_template_tokenize_uses_rendered_prompt():
    dummy = _DummyTokenizer()
    tokenizer = DeepseekV4Tokenizer(dummy)

    token_ids = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "hello",
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
    )

    assert token_ids == [1, 2, 3]
    assert dummy.last_encoded_text == (
        "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>"
    )


def test_deepseek_v4_chat_template_supports_thinking_mode():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "hello",
            }
        ],
        tokenize=False,
        enable_thinking=True,
    )

    assert prompt == ("<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜><think>")


def test_deepseek_v4_chat_template_supports_thinking_alias():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "hello",
            }
        ],
        tokenize=False,
        thinking=True,
    )

    assert prompt == ("<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜><think>")


def test_deepseek_v4_chat_template_matches_vllm_add_generation_prompt_behavior():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "hello",
            }
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

    assert prompt == ("<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>")


def test_deepseek_v4_chat_template_accepts_openai_reasoning_effort_values():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    for reasoning_effort in ("none", "low", "medium", "high"):
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "hello",
                }
            ],
            tokenize=False,
            enable_thinking=True,
            reasoning_effort=reasoning_effort,
        )

        assert prompt.endswith("<｜Assistant｜><think>")
        assert "Reasoning Effort: Absolute maximum" not in prompt


def test_deepseek_v4_chat_template_preserves_reference_max_reasoning_effort():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "hello",
            }
        ],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="max",
    )

    assert prompt.startswith("<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum")
    assert prompt.endswith("<｜User｜>hello<｜Assistant｜><think>")


def test_deepseek_v4_chat_template_drops_historical_thinking_without_tools():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "first",
            },
            {
                "role": "assistant",
                "reasoning": "hidden chain",
                "content": "answer",
            },
            {
                "role": "user",
                "content": "second",
            },
        ],
        tokenize=False,
        enable_thinking=True,
    )

    assert "hidden chain" not in prompt
    assert "answer<｜end▁of▁sentence｜>" in prompt
    assert prompt.endswith("<｜User｜>second<｜Assistant｜><think>")


def test_deepseek_v4_chat_template_keeps_historical_thinking_with_tools():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search",
                "parameters": {"type": "object"},
            },
        }
    ]

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "first",
            },
            {
                "role": "assistant",
                "reasoning": "kept chain",
                "content": "answer",
            },
            {
                "role": "user",
                "content": "second",
            },
        ],
        tools=tools,
        tokenize=False,
        enable_thinking=True,
    )

    assert "kept chain</think>answer<｜end▁of▁sentence｜>" in prompt
    assert prompt.endswith("<｜User｜>second<｜Assistant｜><think>")


def test_deepseek_v4_chat_template_renders_developer_tools_and_latest_reminder():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search",
                "parameters": {"type": "object"},
            },
        }
    ]
    messages = [
        {
            "role": "system",
            "content": "sys",
        },
        {
            "role": "latest_reminder",
            "content": "today",
        },
        {
            "role": "developer",
            "content": "dev",
            "tools": tools,
        },
        {
            "role": "assistant",
            "reasoning": "need search",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "x"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "[0]",
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        enable_thinking=True,
    )

    assert prompt.startswith("<｜begin▁of▁sentence｜>sys<｜latest_reminder｜>today<｜User｜>dev")
    assert "## Tools" in prompt
    assert '<｜DSML｜invoke name="search">' in prompt
    assert "need search</think>" in prompt
    assert "<｜User｜><tool_result>[0]</tool_result><｜Assistant｜><think>" in prompt


def test_deepseek_v4_chat_template_renders_action_task_token():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "sys",
            },
            {
                "role": "latest_reminder",
                "content": "today",
            },
            {
                "role": "user",
                "content": "search this",
                "task": "action",
            },
            {
                "role": "assistant",
                "content": "Search",
            },
        ],
        tokenize=False,
    )

    assert prompt == (
        "<｜begin▁of▁sentence｜>sys<｜latest_reminder｜>today"
        "<｜User｜>search this<｜Assistant｜></think><｜action｜>"
        "Search<｜end▁of▁sentence｜>"
    )


def test_deepseek_v4_chat_template_uses_v4_tool_prompt_from_request_tools():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Weather?",
            }
        ],
        tools=tools,
        tokenize=False,
    )

    assert "## Tools" in prompt
    assert "<｜DSML｜tool_calls>" in prompt
    assert "</｜DSML｜tool_calls>" in prompt
    assert "function_calls" not in prompt
    assert '"name": "get_weather"' in prompt
    assert prompt.endswith("<｜User｜>Weather?<｜Assistant｜></think>")


def test_deepseek_v4_chat_template_renders_tool_call_history():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())
    messages = [
        {
            "role": "user",
            "content": "List the repo",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "str_replace_editor",
                        "arguments": '{"command": "view", "path": "/testbed"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "file list",
        },
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    assert '<｜DSML｜invoke name="str_replace_editor">' in prompt
    assert '<｜DSML｜parameter name="command" string="true">view' in prompt
    assert '<｜DSML｜parameter name="path" string="true">/testbed' in prompt
    assert "<｜User｜><tool_result>file list</tool_result><｜Assistant｜></think>" in prompt
    assert 'parameter name="arguments"' not in prompt


def test_deepseek_v4_custom_tokenizer_reuses_loaded_wrapper():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    args = TorchLlmArgs(model="dummy", tokenizer=tokenizer, custom_tokenizer="deepseek_v4")

    assert args.tokenizer is tokenizer


def test_deepseek_v4_server_chat_template_path_uses_custom_tokenizer():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())

    prompt = apply_chat_template(
        model_type="deepseek_v4",
        tokenizer=tokenizer,
        processor=None,
        conversation=[
            {
                "role": "user",
                "content": "hello",
            }
        ],
        add_generation_prompt=True,
        mm_placeholder_counts=[{}],
    )

    assert prompt == ("<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜></think>")


def test_deepseek_v4_server_chat_template_path_forwards_tools():
    tokenizer = DeepseekV4Tokenizer(_DummyTokenizer())
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search",
                "parameters": {"type": "object"},
            },
        }
    ]

    prompt = apply_chat_template(
        model_type="deepseek_v4",
        tokenizer=tokenizer,
        processor=None,
        conversation=[
            {
                "role": "user",
                "content": "hello",
            }
        ],
        add_generation_prompt=True,
        mm_placeholder_counts=[{}],
        tools=tools,
    )

    assert "<｜DSML｜tool_calls>" in prompt
    assert '"name": "search"' in prompt
