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
"""The shared text-tokenization step and its use by ``DefaultInputProcessor``."""

from __future__ import annotations

import threading
from typing import Any, Optional

import pytest
from llmapi.apps.render_helpers import LLAMA, real_tokenizer

from tensorrt_llm.inputs.registry import DefaultInputProcessor
from tensorrt_llm.inputs.tokenization import TIKTOKEN_ALLOWED_SPECIAL_TOKENS, tokenize_prompt
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


@pytest.fixture(scope="module")
def tokenizer():
    """The real Llama-3.1 tokenizer: it adds BOS by default and truncates on the left."""
    return real_tokenizer(LLAMA)


PROMPT = "hello world again hello world"


class TestTokenizePrompt:
    def test_special_tokens_follow_the_flag(self, tokenizer) -> None:
        without = tokenize_prompt(tokenizer, PROMPT, add_special_tokens=False)
        with_specials = tokenize_prompt(tokenizer, PROMPT, add_special_tokens=True)

        assert without == tokenizer.encode(PROMPT, add_special_tokens=False)
        assert with_specials == [tokenizer.bos_token_id, *without]

    def test_truncation_keeps_the_end_for_a_left_truncating_tokenizer(self, tokenizer) -> None:
        full = tokenize_prompt(tokenizer, PROMPT, add_special_tokens=False)
        ids = tokenize_prompt(tokenizer, PROMPT, add_special_tokens=False, truncate_prompt_tokens=3)

        assert tokenizer.truncation_side == "left"
        assert ids == full[-3:]

    def test_truncating_and_plain_encodes_can_share_a_tokenizer_across_threads(
        self, tokenizer
    ) -> None:
        # A fast tokenizer truncates by changing its own state; a plain encode that overlaps
        # fails with "Already borrowed". Truncation runs on a private copy instead.
        errors: list = []
        plain = tokenize_prompt(tokenizer, PROMPT, add_special_tokens=False)
        truncated = plain[-3:]

        def run(truncate: Optional[int], expected: list) -> None:
            try:
                for _ in range(300):
                    ids = tokenize_prompt(
                        tokenizer, PROMPT, add_special_tokens=False, truncate_prompt_tokens=truncate
                    )
                    assert ids == expected
            except Exception as error:  # noqa: BLE001
                errors.append(error)

        threads = [threading.Thread(target=run, args=(3, truncated)) for _ in range(4)]
        threads += [threading.Thread(target=run, args=(None, plain)) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors, errors[:1]

    def test_missing_tokenizer_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="tokenizer is required"):
            tokenize_prompt(None, PROMPT, add_special_tokens=False)

    def test_a_tiktoken_encoding_gets_the_allowed_special_set(self) -> None:
        # A tiktoken `Encoding` has no `add_special_tokens` argument, so the first call raises
        # TypeError and the second passes the special tokens the Harmony prompt contains.
        tiktoken = pytest.importorskip("tiktoken")
        try:
            encoding = tiktoken.get_encoding("o200k_base")
        except Exception:  # noqa: BLE001 - the vocabulary is downloaded or read from a cache
            pytest.skip("the o200k_base vocabulary is not available")
        text = "<|start|>user<|message|>hi<|end|>"

        ids = tokenize_prompt(encoding, text, add_special_tokens=True)

        assert ids == encoding.encode(text, allowed_special=TIKTOKEN_ALLOWED_SPECIAL_TOKENS)

    def test_prefix_cache_is_used_only_when_the_tokenizer_is_called_as_the_cache_does(
        self, tokenizer
    ) -> None:
        class RecordingCache:
            def __init__(self) -> None:
                self.calls = 0

            def encode(self, tok: Any, text: str) -> list[int]:
                self.calls += 1
                return [99]

        cache = RecordingCache()

        assert tokenize_prompt(
            tokenizer, PROMPT, add_special_tokens=False, prefix_token_cache=cache
        ) == [99]
        # Special tokens or truncation mean a different tokenizer call: bypass.
        tokenize_prompt(tokenizer, PROMPT, add_special_tokens=True, prefix_token_cache=cache)
        tokenize_prompt(
            tokenizer,
            PROMPT,
            add_special_tokens=False,
            truncate_prompt_tokens=2,
            prefix_token_cache=cache,
        )
        assert cache.calls == 1


class TestDefaultInputProcessorUsesTheSharedStep:
    """``DefaultInputProcessor`` returns exactly what ``tokenize_prompt`` returns."""

    @pytest.mark.parametrize("add_special_tokens", [False, True])
    @pytest.mark.parametrize("truncate", [None, 2])
    def test_matches_the_function(self, tokenizer, add_special_tokens, truncate) -> None:
        processor = DefaultInputProcessor(None, None, tokenizer)
        params = SamplingParams(
            add_special_tokens=add_special_tokens, truncate_prompt_tokens=truncate
        )

        ids, extra = processor({"prompt": PROMPT}, params)

        assert extra is None
        assert ids == tokenize_prompt(
            tokenizer,
            PROMPT,
            add_special_tokens=add_special_tokens,
            truncate_prompt_tokens=truncate,
        )

    def test_missing_tokenizer_is_still_rejected(self) -> None:
        processor = DefaultInputProcessor(None, None, None)
        with pytest.raises(ValueError, match="tokenizer is required"):
            processor({"prompt": PROMPT}, SamplingParams())
