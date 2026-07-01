# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_llm.llmapi.block_reuse import (
    BLOCK_REUSE_STABLE_TOKEN_COUNT_TRACE_KEY,
    get_block_reuse_stable_token_count,
    set_block_reuse_stable_token_count,
)
from tensorrt_llm.serve.openai_server import _count_text_prompt_tokens


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def encode(self, prompt: str, add_special_tokens: bool = False) -> list[int]:
        self.calls.append((prompt, add_special_tokens))
        return [1, 2, 3] + ([4] if add_special_tokens else [])


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
