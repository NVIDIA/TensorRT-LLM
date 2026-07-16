# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

from jinja2.exceptions import TemplateError

from tensorrt_llm.inputs.utils import apply_chat_template
from tensorrt_llm.serve.openai_server import (
    _encode_text_prompt,
    _get_reusable_prompt_len,
    _mamba_save_last_snapshot_enabled,
    _maybe_apply_stable_chat_template,
)


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


def test_encode_text_prompt_matches_request_special_token_mode() -> None:
    tokenizer = _FakeTokenizer()

    assert _encode_text_prompt(tokenizer, "prompt", add_special_tokens=False) == [1, 2, 3]
    assert _encode_text_prompt(tokenizer, "prompt", add_special_tokens=True) == [1, 2, 3, 4]
    assert tokenizer.calls == [("prompt", False), ("prompt", True)]


def test_reusable_prompt_len_handles_agentx_last_assistant_prompt() -> None:
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
    reusable_prompt_len = _get_reusable_prompt_len(
        tokenizer,
        stable_prompt,
        full_prompt,
        add_special_tokens=False,
    )

    assert full_prompt.startswith(stable_prompt)
    assert full_prompt != stable_prompt
    assert full_prompt.endswith("<|im_start|>assistant\n<think>\n")
    assert reusable_prompt_len == len(stable_prompt)


def test_router_prompt_token_ids_preserve_stable_boundary() -> None:
    tokenizer = _FakeChatTokenizer()
    stable_prompt = "stable"
    full_prompt = "stable-generation-suffix"
    router_token_ids = tokenizer.encode(full_prompt, add_special_tokens=False)

    assert _get_reusable_prompt_len(
        tokenizer,
        stable_prompt,
        router_token_ids,
        add_special_tokens=False,
    ) == len(stable_prompt)


def test_router_prompt_token_ids_ignore_request_special_token_mode() -> None:
    tokenizer = _FakeChatTokenizer()
    stable_prompt = "stable"
    full_prompt = "stable-generation-suffix"
    router_token_ids = tokenizer.encode(full_prompt, add_special_tokens=False)

    assert _get_reusable_prompt_len(
        tokenizer,
        stable_prompt,
        router_token_ids,
        add_special_tokens=True,
    ) == len(stable_prompt)


def test_reusable_prompt_len_rejects_non_prefix_token_ids() -> None:
    tokenizer = _FakeTokenizer()

    assert (
        _get_reusable_prompt_len(
            tokenizer,
            "stable",
            [9, 9, 9, 9],
            add_special_tokens=False,
        )
        is None
    )


def test_stable_prompt_render_is_skipped_when_feature_is_disabled(monkeypatch) -> None:
    calls = []

    async def fake_apply_chat_template(**kwargs):
        calls.append(kwargs)
        return "stable"

    monkeypatch.setattr(
        "tensorrt_llm.serve.openai_server.async_apply_chat_template",
        fake_apply_chat_template,
    )

    generator = SimpleNamespace(
        args=SimpleNamespace(
            kv_cache_config=SimpleNamespace(
                enable_block_reuse=True,
                mamba_save_last_snapshot=False,
            )
        )
    )
    enabled = _mamba_save_last_snapshot_enabled(generator)
    stable_prompt = asyncio.run(_maybe_apply_stable_chat_template(enabled, {"unused": True}))

    assert stable_prompt is None
    assert calls == []


def test_unsupported_stable_prompt_render_is_best_effort(monkeypatch) -> None:
    async def unsupported_apply_chat_template(**kwargs):
        raise TemplateError("add_generation_prompt=False is unsupported")

    monkeypatch.setattr(
        "tensorrt_llm.serve.openai_server.async_apply_chat_template",
        unsupported_apply_chat_template,
    )

    stable_prompt = asyncio.run(_maybe_apply_stable_chat_template(True, {"unused": True}))

    assert stable_prompt is None
