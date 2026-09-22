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
"""Unit tests for tool-schema validation on ChatCompletionRequest."""

import pytest
from pydantic import ValidationError

from tensorrt_llm.serve.openai_protocol import (
    TOOL_PARAM_MAX_ENUM_VALUES,
    ChatCompletionRequest,
    FunctionDefinition,
)

pytestmark = pytest.mark.cpu_only


def _enum_tool(enum_count: int, name: str = "pick") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {
                    "choice": {
                        "type": "string",
                        "enum": [f"v{i}" for i in range(enum_count)],
                    }
                },
            },
        },
    }


def _request(**extra) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        **extra,
    )


class TestToolEnumValueCap:
    def test_over_cap_enum_rejected(self):
        with pytest.raises(ValidationError) as exc:
            _request(tools=[_enum_tool(TOOL_PARAM_MAX_ENUM_VALUES + 1)])
        msg = str(exc.value)
        assert "enum values" in msg and "maximum" in msg

    def test_under_cap_enum_accepted(self):
        req = _request(tools=[_enum_tool(5)])
        assert req.tools is not None and len(req.tools) == 1

    def test_at_cap_enum_accepted(self):
        req = _request(tools=[_enum_tool(TOOL_PARAM_MAX_ENUM_VALUES)])
        assert req.tools is not None

    def test_nested_enums_counted_together(self):
        half = TOOL_PARAM_MAX_ENUM_VALUES // 2 + 1
        parameters = {
            "type": "object",
            "properties": {
                "a": {"type": "string", "enum": [f"a{i}" for i in range(half)]},
                "b": {
                    "type": "object",
                    "properties": {
                        "c": {"type": "string", "enum": [f"c{i}" for i in range(half)]},
                    },
                },
            },
        }
        with pytest.raises(ValidationError, match="enum values"):
            FunctionDefinition(name="nested", parameters=parameters)

    def test_no_parameters_accepted(self):
        FunctionDefinition(name="bare")  # must not raise

    def test_cap_is_per_function(self):
        # Two tools each just under the cap are fine; the cap is per function,
        # not per request.
        under = TOOL_PARAM_MAX_ENUM_VALUES - 1
        req = _request(tools=[_enum_tool(under, "one"), _enum_tool(under, "two")])
        assert len(req.tools) == 2

    def test_enum_in_instance_data_not_counted(self):
        # An `enum` key that is instance data -- here the `default` value of a
        # property happens to be a dict with an `enum` list -- is not an enum
        # constraint and must not count against the cap.
        over = TOOL_PARAM_MAX_ENUM_VALUES + 1
        parameters = {
            "type": "object",
            "properties": {
                "cfg": {
                    "type": "object",
                    "default": {"enum": [f"v{i}" for i in range(over)]},
                },
            },
        }
        FunctionDefinition(name="instance_data", parameters=parameters)

    def test_property_named_like_keyword_still_counted(self):
        # A property literally named `default` still carries a real enum
        # constraint, so it must count -- the instance-data skip must not open a
        # bypass through the property name.
        over = TOOL_PARAM_MAX_ENUM_VALUES + 1
        parameters = {
            "type": "object",
            "properties": {
                "default": {
                    "type": "string",
                    "enum": [f"v{i}" for i in range(over)],
                },
            },
        }
        with pytest.raises(ValidationError, match="enum values"):
            FunctionDefinition(name="named_default", parameters=parameters)
