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
"""The single text-tokenization step used wherever a rendered prompt becomes ids.

``DefaultInputProcessor`` (the engine side) and the serving-layer prompt
preparation (``tensorrt_llm.serve.render``) both call :func:`tokenize_prompt`, so
a prompt tokenizes to the same ids no matter which of them produced it.
"""

from __future__ import annotations

from typing import Any, List, Optional

from .._utils import nvtx_range_debug
from .prefix_token_cache import PrefixTokenCache

__all__ = ["TIKTOKEN_ALLOWED_SPECIAL_TOKENS", "tokenize_prompt"]

# Special tokens a tiktoken-backed tokenizer must be told to allow in text.
TIKTOKEN_ALLOWED_SPECIAL_TOKENS = {
    "<|startoftext|>",
    "<|endoftext|>",
    "<|reserved_200000|>",
    "<|reserved_200001|>",
    "<|return|>",
    "<|constrain|>",
    "<|reserved_200004|>",
    "<|channel|>",
    "<|start|>",
    "<|end|>",
    "<|message|>",
    "<|reserved_200009|>",
    "<|reserved_200010|>",
    "<|reserved_200011|>",
    "<|call|>",
    "<|reserved_200013|>",
}


def tokenize_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    add_special_tokens: bool,
    truncate_prompt_tokens: Optional[int] = None,
    prefix_token_cache: Optional[PrefixTokenCache] = None,
) -> List[int]:
    """Tokenize ``prompt`` the way the engine does for a text prompt.

    Args:
        tokenizer: The model tokenizer.
        prompt: The rendered prompt text.
        add_special_tokens: Whether the tokenizer adds its own special tokens
            (BOS/EOS and the like) on top of what the text already contains.
        truncate_prompt_tokens: Keep at most this many tokens, or ``None``.
        prefix_token_cache: Optional prefix cache. It is consulted only when the
            tokenizer would be called exactly as the cache calls it: no special
            tokens added and no truncation.
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required to tokenize string prompt")
    if prefix_token_cache is not None and not add_special_tokens and truncate_prompt_tokens is None:
        with nvtx_range_debug("tokenize prompt"), nvtx_range_debug("prefix cache"):
            return prefix_token_cache.encode(tokenizer, prompt)
    kwargs = {}
    if truncate_prompt_tokens is not None:
        kwargs = dict(truncation=True, max_length=truncate_prompt_tokens)
    with nvtx_range_debug("tokenize prompt"):
        try:
            return tokenizer.encode(prompt, add_special_tokens=add_special_tokens, **kwargs)
        except Exception:
            # Tiktoken path
            return tokenizer.encode(prompt, allowed_special=TIKTOKEN_ALLOWED_SPECIAL_TOKENS)
