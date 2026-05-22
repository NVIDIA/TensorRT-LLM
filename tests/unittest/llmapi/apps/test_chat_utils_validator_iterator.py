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

import copy
import json
from typing import Any, List
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from tensorrt_llm.serve.chat_utils import parse_chat_message_content, parse_chat_messages_coroutines


class SingleUseIterator:
    """Mimics Pydantic v2 ValidatorIterator: yields items once, then empty."""

    def __init__(self, items: List[Any]):
        self._items = list(items)
        self._exhausted = False

    def __iter__(self):
        if self._exhausted:
            return iter([])
        self._exhausted = True
        return iter(self._items)


class FailingValidatorIterator:
    """Mimics ValidatorIterator that rejects extra fields on iteration."""

    def __iter__(self):
        raise ValidationError.from_exception_data(
            title="ChatCompletionMessageToolCallParam",
            line_errors=[
                {
                    "type": "extra_forbidden",
                    "loc": ("name",),
                    "msg": "Extra inputs are not permitted",
                    "input": "get_weather",
                }
            ],
        )


TOOL_CALL = {
    "id": "call_1",
    "type": "function",
    "function": {
        "name": "get_weather",
        "arguments": '{"location": "SF"}',
    },
}

PARSED_ARGS = {"location": "SF"}


@pytest.fixture
def mm_tracker():
    return MagicMock()


class TestValidatorIteratorHandling:
    def test_single_use_iterator(self, mm_tracker):
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": SingleUseIterator([TOOL_CALL]),
        }
        result = parse_chat_message_content(msg, mm_tracker)
        assert result["tool_calls"][0]["function"]["arguments"] == PARSED_ARGS

    def test_single_use_iterator_multiple(self, mm_tracker):
        second = {
            **TOOL_CALL,
            "id": "call_2",
            "function": {"name": "get_time", "arguments": '{"tz": "EST"}'},
        }
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": SingleUseIterator([TOOL_CALL, second]),
        }
        result = parse_chat_message_content(msg, mm_tracker)
        assert len(result["tool_calls"]) == 2

    def test_regular_list(self, mm_tracker):
        msg = {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]}
        result = parse_chat_message_content(msg, mm_tracker)
        assert result["tool_calls"][0]["function"]["arguments"] == PARSED_ARGS

    def test_failing_iterator_raises(self, mm_tracker):
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": FailingValidatorIterator(),
        }
        with pytest.raises(ValidationError):
            parse_chat_message_content(msg, mm_tracker)


class TestToolCallsExtraFields:
    def test_extra_name_field(self, mm_tracker):
        tc = {**TOOL_CALL, "name": "get_weather"}
        msg = {"role": "assistant", "content": None, "tool_calls": [tc]}
        result = parse_chat_message_content(msg, mm_tracker)
        assert result["tool_calls"][0]["name"] == "get_weather"
        assert result["tool_calls"][0]["function"]["arguments"] == PARSED_ARGS


class TestDefensiveCopy:
    def test_original_message_unchanged(self, mm_tracker):
        msg = {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]}
        before = copy.deepcopy(msg)
        result = parse_chat_message_content(msg, mm_tracker)
        assert isinstance(result["tool_calls"][0]["function"]["arguments"], dict)
        assert msg == before

    def test_function_dict_unchanged(self, mm_tracker):
        func = {"name": "get_weather", "arguments": '{"a": 1}'}
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": func}],
        }
        parse_chat_message_content(msg, mm_tracker)
        assert func["arguments"] == '{"a": 1}'


class TestParseChatMessagesCoroutines:
    def _mock_config(self):
        cfg = MagicMock(spec=None)
        cfg.model_type = "dummy"
        return cfg

    def test_list_tool_calls(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]},
            {"role": "tool", "content": "72F", "tool_call_id": "call_1"},
        ]
        conv, _, _ = parse_chat_messages_coroutines(messages, self._mock_config(), None)
        assert len(conv) == 3
        assert conv[1]["tool_calls"][0]["function"]["arguments"] == PARSED_ARGS

    def test_iterator_tool_calls(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": SingleUseIterator([TOOL_CALL])},
        ]
        conv, _, _ = parse_chat_messages_coroutines(messages, self._mock_config(), None)
        assert conv[1]["tool_calls"][0]["function"]["arguments"] == PARSED_ARGS

    def test_extra_fields_raw_dict(self):
        tc = {**TOOL_CALL, "name": "get_weather"}
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [tc]},
        ]
        conv, _, _ = parse_chat_messages_coroutines(messages, self._mock_config(), None)
        assert conv[1]["tool_calls"][0]["name"] == "get_weather"


class TestPydanticV2ValidatorIteratorEndToEnd:
    """End-to-end test with real ChatCompletionRequest validation.

    OpenAI SDK types tool_calls as Iterable[...], so Pydantic v2 wraps it
    in a ValidatorIterator (lazy, single-use). Extra fields cause
    ValidationError during iteration, not during request validation.
    """

    def test_full_request_extra_field_raises_on_iteration(self):
        """Extra 'name' field passes request validation but fails on iteration."""
        from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest

        raw_request = {
            "model": "test",
            "messages": [
                {"role": "user", "content": "What is the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "name": "get_weather",
                            "function": {
                                "name": "get_weather",
                                "arguments": json.dumps({"location": "SF"}),
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "72F", "tool_call_id": "call_1"},
            ],
        }
        req = ChatCompletionRequest(**raw_request)
        tc = req.messages[1].get("tool_calls")
        assert type(tc).__name__ == "ValidatorIterator"
        with pytest.raises(ValidationError, match="extra_forbidden"):
            list(tc)
