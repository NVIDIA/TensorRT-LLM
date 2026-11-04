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

# Adapted from
# https://github.com/vllm-project/vllm/blob/4db5176d9758b720b05460c50ace3c01026eb158/vllm/entrypoints/openai/protocol.py

import uuid
from typing import Any, Dict, Literal, Optional

from pydantic import Field, model_validator

from tensorrt_llm.serve.openai_protocol.common import OpenAIBaseModel


class FunctionCall(OpenAIBaseModel):
    name: str
    arguments: str


class DeltaFunctionCall(OpenAIBaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ToolCall(OpenAIBaseModel):
    id: str = Field(
        default_factory=lambda: f"chatcmpl-tool-{str(uuid.uuid4().hex)}")
    type: Literal["function"] = "function"
    function: FunctionCall


class DeltaToolCall(OpenAIBaseModel):
    id: Optional[str] = None
    type: Literal["function"] = "function"
    index: int
    function: Optional[DeltaFunctionCall] = None


# Maximum total number of `enum` values across all properties of one tool
# function's parameters schema. Mirrors the cap reference OpenAI-compatible
# platforms enforce; unbounded enums also inflate the guided-decoding
# grammar compiled for strict tool calls.
TOOL_PARAM_MAX_ENUM_VALUES = 1000

# JSON Schema keywords whose value is instance data rather than a subschema.
# An `enum` key nested inside these is a value (e.g. a default that happens to
# be `{"enum": [...]}`), not an enum constraint, so it must not count.
_SCHEMA_INSTANCE_KEYWORDS = frozenset({"default", "const", "examples"})

# JSON Schema keywords whose value is a map of {name: subschema}. The names are
# user-controlled (a property can be literally named `default` or `enum`), so
# recurse into the values as subschemas without treating the names as keywords.
_SCHEMA_MAP_KEYWORDS = frozenset({
    "properties", "patternProperties", "$defs", "definitions",
    "dependentSchemas"
})


def _count_schema_enum_values(schema: Any) -> int:
    """Recursively count `enum` *constraint* entries in a JSON-schema fragment.

    Only `enum` keywords in schema positions are counted. `enum` keys that are
    instance data -- nested under `default`/`const`/`examples`, or the name of a
    property -- are ignored, so a valid schema is not rejected for values that
    are not enum constraints.
    """
    count = 0
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "enum" and isinstance(value, list):
                count += len(value)
            elif key in _SCHEMA_INSTANCE_KEYWORDS:
                continue
            elif key in _SCHEMA_MAP_KEYWORDS and isinstance(value, dict):
                for subschema in value.values():
                    count += _count_schema_enum_values(subschema)
            else:
                count += _count_schema_enum_values(value)
    elif isinstance(schema, list):
        for item in schema:
            count += _count_schema_enum_values(item)
    return count


class FunctionDefinition(OpenAIBaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = None

    @model_validator(mode="after")
    def check_enum_value_cap(self):
        if self.parameters is not None:
            num_enum_values = _count_schema_enum_values(self.parameters)
            if num_enum_values > TOOL_PARAM_MAX_ENUM_VALUES:
                raise ValueError(
                    f"tool function {self.name!r} declares {num_enum_values} "
                    f"enum values across its parameters schema; the maximum "
                    f"is {TOOL_PARAM_MAX_ENUM_VALUES}.")
        return self


class ChatCompletionToolsParam(OpenAIBaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class NamespacedFunctionToolParam(ChatCompletionToolsParam):
    """A tool that a Responses request declared inside a namespace.

    ``function`` is what a chat template without namespaces is given: the
    tool as a function named ``<namespace>.<name>``. That is all a dump
    carries. The declaration itself is kept beside it, out of every dump, for
    a model whose renderer reads namespaces (``ServingExtension.serialize_tool``).
    """

    namespace: str = Field(exclude=True)
    namespace_description: Optional[str] = Field(default=None, exclude=True)
    declared_name: str = Field(exclude=True)
    declared_description: Optional[str] = Field(default=None, exclude=True)


class ChatCompletionNamedFunction(OpenAIBaseModel):
    name: str


class ChatCompletionNamedToolChoiceParam(OpenAIBaseModel):
    function: ChatCompletionNamedFunction
    type: Literal["function"] = "function"
