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
"""Tests for ``OpenAIServer._stream_lead_in_token_count``."""

from types import SimpleNamespace

import pytest

from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory
from tensorrt_llm.serve.openai_server import OpenAIServer

pytestmark = pytest.mark.cpu_only

K3_LEAD_IN = "<|close|>think<|sep|><|open|>response<|sep|>"


class _CountingTokenizer:
    """Encodes the K3 lead-in as six tokens and counts the calls."""

    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=True):
        self.calls.append((text, add_special_tokens))
        return list(range(6 if text == K3_LEAD_IN else len(text)))


def _server(tokenizer):
    return SimpleNamespace(tokenizer=tokenizer, _stream_lead_in_token_counts={})


@pytest.fixture
def parsers_built(monkeypatch):
    """Records the (parser, chat_template_kwargs) of every reasoning parser the server builds."""
    built = []
    create = ReasoningParserFactory.create_reasoning_parser

    def counting_create(reasoning_parser, chat_template_kwargs=None):
        built.append((reasoning_parser, chat_template_kwargs))
        return create(reasoning_parser, chat_template_kwargs)

    monkeypatch.setattr(ReasoningParserFactory, "create_reasoning_parser", counting_create)
    return built


def test_kimi_k3_thinking_mode_counts_the_lead_in_tokens(parsers_built):
    tokenizer = _CountingTokenizer()
    server = _server(tokenizer)
    assert OpenAIServer._stream_lead_in_token_count(server, "kimi_k3", None) == 6
    assert tokenizer.calls == [(K3_LEAD_IN, False)]
    assert len(parsers_built) == 1
    # the same mode again: neither a parser nor an encode
    assert OpenAIServer._stream_lead_in_token_count(server, "kimi_k3", {"max_depth": 3}) == 6
    assert len(parsers_built) == 1 and len(tokenizer.calls) == 1
    # resolved thinking kwargs, as the server passes them, are a mode of their own with the same lead-in
    kwargs = {"thinking": True, "enable_thinking": True}
    assert OpenAIServer._stream_lead_in_token_count(server, "kimi_k3", kwargs) == 6
    assert OpenAIServer._stream_lead_in_token_count(server, "kimi_k3", dict(kwargs)) == 6
    assert len(parsers_built) == 2


def test_cached_per_parser_and_mode(parsers_built):
    tokenizer = _CountingTokenizer()
    server = _server(tokenizer)
    off = {"thinking": False, "enable_thinking": False}
    for _ in range(3):
        assert OpenAIServer._stream_lead_in_token_count(server, "kimi_k3", None) == 6
        assert OpenAIServer._stream_lead_in_token_count(server, "kimi_k3", off) == 0
        assert OpenAIServer._stream_lead_in_token_count(server, "deepseek-r1", None) == 0
    assert [p for p, _ in parsers_built] == ["kimi_k3", "kimi_k3", "deepseek-r1"]
    assert server._stream_lead_in_token_counts == {
        ("kimi_k3", None, None): 6,
        ("kimi_k3", False, False): 0,
        ("deepseek-r1", None, None): 0,
    }


def test_non_boolean_mode_is_counted_without_caching(parsers_built):
    server = _server(_CountingTokenizer())
    for _ in range(2):
        assert OpenAIServer._stream_lead_in_token_count(server, "kimi_k3", {"thinking": "yes"}) == 6
    assert len(parsers_built) == 2
    assert server._stream_lead_in_token_counts == {}


@pytest.mark.parametrize(
    ("reasoning_parser", "kwargs"),
    [
        (None, None),
        ("kimi_k3", {"thinking": False}),
        ("deepseek-r1", None),
    ],
)
def test_no_lead_in_is_zero(reasoning_parser, kwargs):
    tokenizer = _CountingTokenizer()
    server = _server(tokenizer)
    assert OpenAIServer._stream_lead_in_token_count(server, reasoning_parser, kwargs) == 0
    assert tokenizer.calls == []


def test_without_a_tokenizer_is_zero(parsers_built):
    assert OpenAIServer._stream_lead_in_token_count(_server(None), "kimi_k3", None) == 0
    assert parsers_built == []
