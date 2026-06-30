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
import sys
import types
from types import SimpleNamespace

from tensorrt_llm.evaluate.interface import extract_final_content_from_generation
from tensorrt_llm.llmapi.reasoning_parser import HARMONY_REASONING_PARSER


def _request_output(text: str, token_ids: list[int] | None = None):
    completion = SimpleNamespace(text=text, token_ids=token_ids or [])
    return SimpleNamespace(outputs=[completion])


def test_extract_final_content_does_not_guess_reasoning_parser():
    text = '<think>scratch</think>{"answer": "final"}'
    output = _request_output(text)

    assert extract_final_content_from_generation(output) == text


def test_extract_final_content_uses_explicit_reasoning_parser():
    output = _request_output('<think>scratch</think>{"answer": "final"}')

    assert (
        extract_final_content_from_generation(output, reasoning_parser="qwen3")
        == '{"answer": "final"}'
    )


def test_extract_final_content_non_harmony_parser_ignores_harmony_tokens(monkeypatch):
    harmony_module = types.ModuleType("tensorrt_llm.serve.harmony_adapter")

    def _unexpected_adapter_lookup():
        raise AssertionError("Harmony adapter should not be used for qwen3")

    harmony_module.get_harmony_adapter = _unexpected_adapter_lookup
    monkeypatch.setitem(sys.modules, "tensorrt_llm.serve.harmony_adapter", harmony_module)
    output = _request_output('<think>scratch</think>{"answer": "final"}', token_ids=[1, 2, 3])

    assert (
        extract_final_content_from_generation(output, reasoning_parser="qwen3")
        == '{"answer": "final"}'
    )


def test_extract_final_content_does_not_guess_harmony_from_tokens(monkeypatch):
    harmony_module = types.ModuleType("tensorrt_llm.serve.harmony_adapter")

    def _unexpected_adapter_lookup():
        raise AssertionError("Harmony must be selected from model context")

    harmony_module.get_harmony_adapter = _unexpected_adapter_lookup
    monkeypatch.setitem(sys.modules, "tensorrt_llm.serve.harmony_adapter", harmony_module)
    output = _request_output("not json", token_ids=[1, 2, 3])

    assert extract_final_content_from_generation(output) == "not json"


def test_extract_final_content_uses_harmony_tokens(monkeypatch):
    harmony_module = types.ModuleType("tensorrt_llm.serve.harmony_adapter")

    class _FakeHarmonyAdapter:
        def harmony_output_to_openai(self, token_ids):
            assert token_ids == [1, 2, 3]
            return {"content": '{"answer": "harmony"}'}

    harmony_module.get_harmony_adapter = lambda: _FakeHarmonyAdapter()
    monkeypatch.setitem(sys.modules, "tensorrt_llm.serve.harmony_adapter", harmony_module)
    output = _request_output("raw harmony transcript", token_ids=[1, 2, 3])

    assert (
        extract_final_content_from_generation(output, reasoning_parser=HARMONY_REASONING_PARSER)
        == '{"answer": "harmony"}'
    )
