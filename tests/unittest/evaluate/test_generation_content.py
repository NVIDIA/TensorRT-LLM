# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
import types
from types import SimpleNamespace

from tensorrt_llm.evaluate.interface import extract_final_content_from_generation
from tensorrt_llm.llmapi.reasoning_parser import HARMONY_REASONING_PARSER


def _request_output(text: str, token_ids: list[int] | None = None):
    completion = SimpleNamespace(text=text, token_ids=token_ids or [])
    return SimpleNamespace(outputs=[completion])


def test_extract_final_content_returns_raw_plain_text_by_default():
    output = _request_output('{"answer": "plain"}')

    assert extract_final_content_from_generation(output) == '{"answer": "plain"}'


def test_extract_final_content_can_decline_raw_fallback():
    output = _request_output('{"answer": "plain"}')

    assert extract_final_content_from_generation(
        output, fallback_to_raw=False) is None


def test_extract_final_content_uses_explicit_reasoning_parser():
    output = _request_output('<think>scratch</think>{"answer": "final"}')

    assert extract_final_content_from_generation(
        output, reasoning_parser="qwen3") == '{"answer": "final"}'


def test_extract_final_content_detects_visible_reasoning_tags():
    output = _request_output('<think>scratch</think>{"answer": "final"}')

    assert extract_final_content_from_generation(
        output, fallback_to_raw=False) == '{"answer": "final"}'


def test_extract_final_content_uses_harmony_tokens(monkeypatch):
    harmony_module = types.ModuleType("tensorrt_llm.serve.harmony_adapter")

    class _FakeHarmonyAdapter:

        def harmony_output_to_openai(self, token_ids):
            assert token_ids == [1, 2, 3]
            return {"content": '{"answer": "harmony"}'}

    harmony_module.get_harmony_adapter = lambda: _FakeHarmonyAdapter()
    monkeypatch.setitem(sys.modules, "tensorrt_llm.serve.harmony_adapter",
                        harmony_module)
    output = _request_output("raw harmony transcript", token_ids=[1, 2, 3])

    assert extract_final_content_from_generation(
        output, reasoning_parser=HARMONY_REASONING_PARSER
    ) == '{"answer": "harmony"}'
