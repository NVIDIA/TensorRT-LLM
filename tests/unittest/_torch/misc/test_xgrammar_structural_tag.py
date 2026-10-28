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
"""XGrammar regression cases for structural tags, regexes, and JSON schemas.

The structural tags mirror the tool-call grammars that serving frontends build
for XML-style tool parsers. The regex and schema-pattern cases pin matching
behavior that regressed in XGrammar 0.2.7. A one-character-per-token vocabulary
keeps the tests model-free while still going through the XGrammar matcher
factory.
"""

import json

import pytest

from tensorrt_llm._torch.pyexecutor.grammar_matcher import XGrammarMatcherFactory
from tensorrt_llm.bindings.executor import GuidedDecodingParams
from tensorrt_llm.llmapi.llm_args import GuidedDecodingConfig

pytestmark = pytest.mark.cpu_only

_EOS = 128
_VOCAB = [chr(i) for i in range(128)] + ["</s>"]
_VOCAB_SIZE_PADDED = 160

_MIXED_SCALARS_SCHEMA = {
    "type": "object",
    "properties": {
        "destination": {"type": "string"},
        "passengers": {"type": "integer"},
        "first_class": {"type": "boolean"},
    },
    "required": ["destination", "passengers", "first_class"],
    "additionalProperties": False,
}


@pytest.fixture(scope="module")
def factory() -> XGrammarMatcherFactory:
    return XGrammarMatcherFactory(
        GuidedDecodingConfig(encoded_vocab=_VOCAB, stop_token_ids=[_EOS]),
        vocab_size_padded=_VOCAB_SIZE_PADDED,
    )


def _tool_call_tag(name: str, schema) -> str:
    return json.dumps(
        {
            "type": "structural_tag",
            "format": {
                "type": "triggered_tags",
                "triggers": ["<tool_call>\n<function="],
                "tags": [
                    {
                        "type": "tag",
                        "begin": f"<tool_call>\n<function={name}>\n",
                        "content": {
                            "type": "json_schema",
                            "json_schema": schema,
                            "style": "qwen_xml",
                        },
                        "end": "\n</function>\n</tool_call>",
                    }
                ],
                "at_least_one": True,
                "stop_after_first": True,
            },
        }
    )


def _tool_call(name: str, body: str) -> str:
    return f"<tool_call>\n<function={name}>\n{body}\n</function>\n</tool_call>"


def _parameter(key: str, value: str) -> str:
    return f"<parameter={key}>\n{value}\n</parameter>"


def _assert_completes(factory: XGrammarMatcherFactory, name: str, schema, text: str) -> None:
    """Compile the tool-call tag, feed `text`, and require a clean termination."""
    matcher = factory.create(
        GuidedDecodingParams(
            GuidedDecodingParams.GuideType.STRUCTURAL_TAG, _tool_call_tag(name, schema)
        )
    )
    for position, char in enumerate(text):
        assert matcher.accept_token(ord(char)), f"rejected {char!r} at {position} of {text!r}"
    assert matcher.accept_token(_EOS)
    assert matcher.is_terminated()


def _accepts(factory: XGrammarMatcherFactory, guide_type, guide: str, text: str) -> bool:
    """Return whether the guide accepts exactly `text` followed by EOS."""
    matcher = factory.create(GuidedDecodingParams(guide_type, guide))
    return (
        all(matcher.accept_token(ord(char)) for char in text)
        and matcher.accept_token(_EOS)
        and matcher.is_terminated()
    )


@pytest.mark.parametrize("additional_properties", [False, None], ids=["closed", "open"])
def test_empty_object_schema(factory, additional_properties):
    schema = {"type": "object", "properties": {}, "required": []}
    if additional_properties is not None:
        schema["additionalProperties"] = additional_properties
    _assert_completes(factory, "get_server_time", schema, _tool_call("get_server_time", ""))


def test_bare_empty_schema(factory):
    _assert_completes(factory, "get_server_time", {}, _tool_call("get_server_time", ""))


@pytest.mark.parametrize(
    "body", ["", _parameter("path", "/tmp")], ids=["no_arguments", "one_argument"]
)
def test_boolean_true_schema(factory, body):
    _assert_completes(factory, "collect_items", True, _tool_call("collect_items", body))


@pytest.mark.parametrize("separator", ["\n", ""], ids=["newline", "adjacent"])
def test_separator_between_parameters(factory, separator):
    body = separator.join(
        [
            _parameter("destination", "Lisbon"),
            _parameter("passengers", "2"),
            _parameter("first_class", "false"),
        ]
    )
    _assert_completes(
        factory, "book_flight", _MIXED_SCALARS_SCHEMA, _tool_call("book_flight", body)
    )


@pytest.mark.parametrize("text, expected", [("aa", True), ("aaaa", True), ("a", False)])
def test_regex_nested_repetition(factory, text, expected):
    assert _accepts(factory, GuidedDecodingParams.GuideType.REGEX, "(a{2,}){1,2}", text) == expected


@pytest.mark.parametrize("text, expected", [('"acd"', True), ('"aacd"', True), ('"cd"', False)])
def test_json_schema_string_pattern(factory, text, expected):
    schema = json.dumps({"type": "string", "pattern": "^a*acd$"})
    assert _accepts(factory, GuidedDecodingParams.GuideType.JSON_SCHEMA, schema, text) == expected
