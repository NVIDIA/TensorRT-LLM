# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for interleaved thinking support in trtllm-serve.

Tests cover:
- ThinkingParams protocol model
- Content block types (ThinkingContentBlock, TextContentBlock)
- ChatMessage with content blocks
- DeltaMessage with content blocks
- ChatCompletionRequest with thinking parameter
"""

import json

import pytest

from tensorrt_llm.llmapi.reasoning_parser import ContentBlock, ReasoningParserFactory
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatMessage,
    DeltaMessage,
    TextContentBlock,
    ThinkingContentBlock,
    ThinkingParams,
)

# ── ThinkingParams model tests ─────────────────────────────────────


class TestThinkingParams:
    def test_default_type_is_enabled(self):
        params = ThinkingParams()
        assert params.type == "enabled"
        assert params.budget_tokens is None

    def test_enabled_with_budget(self):
        params = ThinkingParams(type="enabled", budget_tokens=1024)
        assert params.type == "enabled"
        assert params.budget_tokens == 1024

    def test_disabled(self):
        params = ThinkingParams(type="disabled")
        assert params.type == "disabled"

    def test_invalid_type_rejected(self):
        with pytest.raises(Exception):
            ThinkingParams(type="invalid")

    def test_serialization(self):
        params = ThinkingParams(type="enabled", budget_tokens=2048)
        data = params.model_dump()
        assert data["type"] == "enabled"
        assert data["budget_tokens"] == 2048

    def test_deserialization(self):
        params = ThinkingParams.model_validate({"type": "enabled", "budget_tokens": 512})
        assert params.type == "enabled"
        assert params.budget_tokens == 512


# ── Content block type tests ──────────────────────────────────────


class TestContentBlocks:
    def test_thinking_block(self):
        block = ThinkingContentBlock(thinking="reasoning text")
        assert block.type == "thinking"
        assert block.thinking == "reasoning text"

    def test_text_block(self):
        block = TextContentBlock(text="response text")
        assert block.type == "text"
        assert block.text == "response text"

    def test_thinking_block_serialization(self):
        block = ThinkingContentBlock(thinking="let me think...")
        data = block.model_dump()
        assert data == {"type": "thinking", "thinking": "let me think..."}

    def test_text_block_serialization(self):
        block = TextContentBlock(text="the answer is 42")
        data = block.model_dump()
        assert data == {"type": "text", "text": "the answer is 42"}


# ── ChatMessage with content blocks ──────────────────────────────


class TestChatMessageContentBlocks:
    def test_string_content_backward_compat(self):
        msg = ChatMessage(role="assistant", content="hello")
        assert msg.content == "hello"
        assert msg.reasoning_content is None

    def test_content_blocks(self):
        blocks = [
            ThinkingContentBlock(thinking="reasoning"),
            TextContentBlock(text="answer"),
        ]
        msg = ChatMessage(role="assistant", content=blocks)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert msg.content[0].type == "thinking"
        assert msg.content[1].type == "text"

    def test_content_blocks_serialization(self):
        blocks = [
            ThinkingContentBlock(thinking="reasoning"),
            TextContentBlock(text="answer"),
        ]
        msg = ChatMessage(role="assistant", content=blocks)
        data = msg.model_dump(exclude_none=True)
        assert data["content"] == [
            {"type": "thinking", "thinking": "reasoning"},
            {"type": "text", "text": "answer"},
        ]

    def test_interleaved_content_blocks(self):
        blocks = [
            ThinkingContentBlock(thinking="think1"),
            TextContentBlock(text="text1"),
            ThinkingContentBlock(thinking="think2"),
            TextContentBlock(text="text2"),
        ]
        msg = ChatMessage(role="assistant", content=blocks)
        data = msg.model_dump(exclude_none=True)
        assert len(data["content"]) == 4
        assert data["content"][0]["type"] == "thinking"
        assert data["content"][1]["type"] == "text"
        assert data["content"][2]["type"] == "thinking"
        assert data["content"][3]["type"] == "text"

    def test_json_serialization_roundtrip(self):
        blocks = [
            ThinkingContentBlock(thinking="reasoning"),
            TextContentBlock(text="answer"),
        ]
        msg = ChatMessage(role="assistant", content=blocks)
        json_str = msg.model_dump_json(exclude_none=True)
        data = json.loads(json_str)
        assert data["role"] == "assistant"
        assert len(data["content"]) == 2


# ── DeltaMessage with content blocks ─────────────────────────────


class TestDeltaMessageContentBlocks:
    def test_string_content_backward_compat(self):
        delta = DeltaMessage(content="hello")
        assert delta.content == "hello"

    def test_thinking_content_block_in_delta(self):
        delta = DeltaMessage(content=[ThinkingContentBlock(thinking="reasoning")])
        assert isinstance(delta.content, list)
        assert delta.content[0].type == "thinking"

    def test_text_content_block_in_delta(self):
        delta = DeltaMessage(content=[TextContentBlock(text="answer")])
        assert isinstance(delta.content, list)
        assert delta.content[0].type == "text"

    def test_delta_serialization_with_blocks(self):
        delta = DeltaMessage(content=[ThinkingContentBlock(thinking="let me think")])
        data = delta.model_dump(exclude_none=True)
        assert data["content"] == [{"type": "thinking", "thinking": "let me think"}]


# ── ChatCompletionRequest with thinking param ─────────────────────


class TestChatCompletionRequestThinking:
    def test_thinking_param_default_none(self):
        req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert req.thinking is None

    def test_thinking_param_enabled(self):
        req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            thinking=ThinkingParams(type="enabled", budget_tokens=1024),
        )
        assert req.thinking is not None
        assert req.thinking.type == "enabled"
        assert req.thinking.budget_tokens == 1024

    def test_thinking_param_disabled(self):
        req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            thinking=ThinkingParams(type="disabled"),
        )
        assert req.thinking.type == "disabled"

    def test_thinking_param_from_dict(self):
        req = ChatCompletionRequest.model_validate(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 2048,
                },
            }
        )
        assert req.thinking.type == "enabled"
        assert req.thinking.budget_tokens == 2048


# ── Reasoning parser interleaved blocks ──────────────────────────


class TestReasoningParserInterleavedBlocks:
    def test_deepseek_r1_single_block(self):
        parser = ReasoningParserFactory.create_reasoning_parser("deepseek-r1")
        blocks = parser.parse_to_blocks("reasoning</think>content")
        assert len(blocks) == 2
        assert blocks[0] == ContentBlock("thinking", "reasoning")
        assert blocks[1] == ContentBlock("text", "content")

    def test_deepseek_r1_multiple_blocks(self):
        parser = ReasoningParserFactory.create_reasoning_parser("deepseek-r1")
        text = "t1</think>c1<think>t2</think>c2<think>t3</think>c3"
        blocks = parser.parse_to_blocks(text)
        assert len(blocks) == 6
        assert blocks[0] == ContentBlock("thinking", "t1")
        assert blocks[1] == ContentBlock("text", "c1")
        assert blocks[2] == ContentBlock("thinking", "t2")
        assert blocks[3] == ContentBlock("text", "c2")
        assert blocks[4] == ContentBlock("thinking", "t3")
        assert blocks[5] == ContentBlock("text", "c3")

    def test_qwen3_no_thinking(self):
        parser = ReasoningParserFactory.create_reasoning_parser("qwen3")
        blocks = parser.parse_to_blocks("just plain text")
        assert len(blocks) == 1
        assert blocks[0] == ContentBlock("text", "just plain text")

    def test_qwen3_interleaved(self):
        parser = ReasoningParserFactory.create_reasoning_parser("qwen3")
        text = "<think>t1</think>c1<think>t2</think>c2"
        blocks = parser.parse_to_blocks(text)
        assert len(blocks) == 4
        assert blocks[0] == ContentBlock("thinking", "t1")
        assert blocks[1] == ContentBlock("text", "c1")
        assert blocks[2] == ContentBlock("thinking", "t2")
        assert blocks[3] == ContentBlock("text", "c2")

    def test_streaming_block_type_tracking(self):
        parser = ReasoningParserFactory.create_reasoning_parser("deepseek-r1")
        # First delta: reasoning content
        result = parser.parse_delta("thinking")
        assert result.current_block_type == "thinking"
        # Second delta: end of reasoning, start of content
        result = parser.parse_delta("</think>content")
        assert result.current_block_type == "text"

    def test_streaming_block_type_text_only(self):
        parser = ReasoningParserFactory.create_reasoning_parser("qwen3")
        result = parser.parse_delta("hello")
        assert result.current_block_type == "text"
