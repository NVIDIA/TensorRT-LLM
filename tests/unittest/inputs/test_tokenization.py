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

from typing import Any

import pytest

from tensorrt_llm.inputs.registry import DefaultInputProcessor
from tensorrt_llm.inputs.tokenization import TIKTOKEN_ALLOWED_SPECIAL_TOKENS, tokenize_prompt
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


@pytest.fixture(scope="module")
def tokenizer():
    """A fast word-level tokenizer whose post-processor adds BOS/EOS."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "<s>": 1, "</s>": 2, "hello": 3, "world": 4, "again": 5}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 1), ("</s>", 2)]
    )
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")


PROMPT = "hello world again hello world"


class TestTokenizePrompt:
    def test_special_tokens_follow_the_flag(self, tokenizer) -> None:
        without = tokenize_prompt(tokenizer, PROMPT, add_special_tokens=False)
        with_specials = tokenize_prompt(tokenizer, PROMPT, add_special_tokens=True)

        assert without == [3, 4, 5, 3, 4]
        assert with_specials == [1, 3, 4, 5, 3, 4, 2]

    def test_truncation_keeps_the_leading_tokens(self, tokenizer) -> None:
        ids = tokenize_prompt(tokenizer, PROMPT, add_special_tokens=False, truncate_prompt_tokens=3)
        assert ids == [3, 4, 5]

    def test_missing_tokenizer_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="tokenizer is required"):
            tokenize_prompt(None, PROMPT, add_special_tokens=False)

    def test_tiktoken_style_tokenizers_get_the_allowed_special_set(self) -> None:
        calls: list = []

        class TiktokenLike:
            def encode(self, text: str, **kwargs: Any) -> list[int]:
                calls.append(kwargs)
                if "add_special_tokens" in kwargs:
                    raise TypeError("unexpected keyword argument 'add_special_tokens'")
                return [11, 12]

        assert tokenize_prompt(TiktokenLike(), "x", add_special_tokens=True) == [11, 12]
        assert calls[0] == {"add_special_tokens": True}
        assert calls[1] == {"allowed_special": TIKTOKEN_ALLOWED_SPECIAL_TOKENS}

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
