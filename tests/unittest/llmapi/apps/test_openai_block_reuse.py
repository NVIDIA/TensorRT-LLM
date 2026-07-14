# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_llm.inputs.utils import apply_chat_template
from tensorrt_llm.serve.openai_server import _count_text_prompt_tokens


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


def test_count_text_prompt_tokens_matches_request_special_token_mode() -> None:
    tokenizer = _FakeTokenizer()

    assert _count_text_prompt_tokens(tokenizer, "prompt", add_special_tokens=False) == 3
    assert _count_text_prompt_tokens(tokenizer, "prompt", add_special_tokens=True) == 4
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
    reusable_prompt_len = _count_text_prompt_tokens(
        tokenizer, stable_prompt, add_special_tokens=False
    )

    assert full_prompt.startswith(stable_prompt)
    assert full_prompt != stable_prompt
    assert full_prompt.endswith("<|im_start|>assistant\n<think>\n")
    assert reusable_prompt_len == len(stable_prompt)
