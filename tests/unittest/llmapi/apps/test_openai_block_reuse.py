# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from tensorrt_llm.inputs.utils import apply_chat_template
from tensorrt_llm.llmapi import SchedulingParams
from tensorrt_llm.llmapi.block_reuse import (
    BLOCK_REUSE_STABLE_TOKEN_COUNT_TRACE_KEY,
    get_block_reuse_stable_token_count,
    set_block_reuse_stable_token_count,
)
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.serve.openai_server import OpenAIServer, _count_text_prompt_tokens


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def encode(self, prompt: str, add_special_tokens: bool = False) -> list[int]:
        self.calls.append((prompt, add_special_tokens))
        return [1, 2, 3] + ([4] if add_special_tokens else [])


class _FakeChatTokenizer:
    def get_chat_template(self, chat_template=None, tools=None) -> str:
        return chat_template or "fake-qwen-chat-template"

    def apply_chat_template(
        self,
        conversation,
        tokenize=False,
        return_dict=False,
        add_generation_prompt=False,
        tools=None,
        documents=None,
        chat_template=None,
        **kwargs,
    ) -> str:
        assert tokenize is False
        assert return_dict is False
        prompt = "".join(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
            for message in conversation
        )
        if add_generation_prompt:
            prompt += "<|im_start|>assistant\n<think>\n"
        return prompt

    def encode(self, prompt: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(prompt) + (1 if add_special_tokens else 0)))


class _FakePromise:
    def __init__(self) -> None:
        self.awaited = False

    async def aresult(self) -> None:
        self.awaited = True


class _FakeGenerator:
    def __init__(self) -> None:
        self.args = SimpleNamespace(
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=True,
                mamba_save_last_snapshot=True,
            )
        )
        self.calls = []
        self.promise = _FakePromise()

    def generate_async(self, **kwargs):
        self.calls.append(kwargs)
        return self.promise


def test_block_reuse_stable_token_count_trace_header_round_trip() -> None:
    trace_headers = set_block_reuse_stable_token_count({"traceparent": "p"}, 123)

    assert trace_headers == {
        "traceparent": "p",
        BLOCK_REUSE_STABLE_TOKEN_COUNT_TRACE_KEY: "123",
    }
    assert get_block_reuse_stable_token_count(trace_headers) == 123


def test_block_reuse_stable_token_count_ignores_invalid_values() -> None:
    assert (
        get_block_reuse_stable_token_count({BLOCK_REUSE_STABLE_TOKEN_COUNT_TRACE_KEY: "0"}) is None
    )
    assert (
        get_block_reuse_stable_token_count({BLOCK_REUSE_STABLE_TOKEN_COUNT_TRACE_KEY: "not-an-int"})
        is None
    )


def test_count_text_prompt_tokens_matches_request_special_token_mode() -> None:
    tokenizer = _FakeTokenizer()

    assert _count_text_prompt_tokens(tokenizer, "prompt", add_special_tokens=False) == 3
    assert _count_text_prompt_tokens(tokenizer, "prompt", add_special_tokens=True) == 4
    assert tokenizer.calls == [("prompt", False), ("prompt", True)]


def test_stable_token_count_handles_agentx_last_assistant_prompt() -> None:
    tokenizer = _FakeChatTokenizer()
    conversation = [
        {"role": "user", "content": "large shared prefix"},
        {"role": "assistant", "content": "previous answer"},
    ]
    mm_placeholder_counts = [{}, {}]

    stable_prompt = apply_chat_template(
        model_type="qwen3",
        tokenizer=tokenizer,
        processor=None,
        conversation=conversation,
        add_generation_prompt=False,
        mm_placeholder_counts=mm_placeholder_counts,
    )
    full_prompt = apply_chat_template(
        model_type="qwen3",
        tokenizer=tokenizer,
        processor=None,
        conversation=conversation,
        add_generation_prompt=True,
        mm_placeholder_counts=mm_placeholder_counts,
    )
    stable_token_count = _count_text_prompt_tokens(
        tokenizer, stable_prompt, add_special_tokens=False
    )
    trace_headers = set_block_reuse_stable_token_count(None, stable_token_count)

    assert full_prompt.startswith(stable_prompt)
    assert full_prompt != stable_prompt
    assert full_prompt.endswith("<|im_start|>assistant\n<think>\n")
    assert stable_token_count == len(stable_prompt)
    assert get_block_reuse_stable_token_count(trace_headers) == stable_token_count


def test_stable_prompt_prewarm_requires_mamba_save_last_snapshot() -> None:
    server = object.__new__(OpenAIServer)
    server.generator = SimpleNamespace(
        args=SimpleNamespace(
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=True,
                mamba_save_last_snapshot=True,
            )
        )
    )

    assert server._should_prewarm_block_reuse_stable_prompt("stable", "stable<gen>")
    assert not server._should_prewarm_block_reuse_stable_prompt("stable", "stable")
    assert not server._should_prewarm_block_reuse_stable_prompt("stable", "other")

    server.generator.args.kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        mamba_save_last_snapshot=False,
    )
    assert not server._should_prewarm_block_reuse_stable_prompt("stable", "stable<gen>")


@pytest.mark.asyncio
async def test_stable_prompt_prewarm_generates_internal_stable_request() -> None:
    server = object.__new__(OpenAIServer)
    generator = _FakeGenerator()
    server.generator = generator
    trace_headers = {"traceparent": "parent"}
    scheduling_params = SchedulingParams(agent_hierarchy=None)

    await server._prewarm_block_reuse_stable_prompt(
        "stable",
        add_special_tokens=True,
        lora_request="lora",
        disaggregated_params=None,
        cache_salt="salt",
        trace_headers=trace_headers,
        scheduling_params=scheduling_params,
    )

    assert generator.promise.awaited
    assert len(generator.calls) == 1
    call = generator.calls[0]
    assert call["inputs"] == {"prompt": "stable"}
    assert call["sampling_params"].max_tokens == 1
    assert call["sampling_params"].temperature == 0
    assert call["sampling_params"].add_special_tokens
    assert call["streaming"] is False
    assert call["lora_request"] == "lora"
    assert call["cache_salt"] == "salt"
    assert call["trace_headers"] is trace_headers
    assert call["scheduling_params"] is scheduling_params
