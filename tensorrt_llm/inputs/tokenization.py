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

import copy
import threading
import weakref
from typing import Any, List, Optional, Tuple

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


# A fast tokenizer changes its own state to truncate (``enable_truncation``) and a later plain
# call resets it, which fails with "Already borrowed" when another thread is encoding. Truncating
# encodes therefore run on a private copy of the tokenizer, one at a time, so the shared
# tokenizer is never modified.
_TRUNCATING_COPIES: "weakref.WeakKeyDictionary[Any, Tuple[Any, threading.Lock]]" = (
    weakref.WeakKeyDictionary()
)
_TRUNCATING_COPIES_LOCK = threading.Lock()


def _truncating_copy(tokenizer: Any) -> Optional[Tuple[Any, threading.Lock]]:
    """The private copy and its lock for a fast tokenizer, ``None`` for any other kind."""
    if not hasattr(getattr(tokenizer, "tokenizer", tokenizer), "backend_tokenizer"):
        return None
    with _TRUNCATING_COPIES_LOCK:
        private = _TRUNCATING_COPIES.get(tokenizer)
        if private is None:
            private = _TRUNCATING_COPIES[tokenizer] = (copy.deepcopy(tokenizer), threading.Lock())
    return private


def _encode(tokenizer: Any, prompt: str, add_special_tokens: bool, kwargs: dict) -> List[int]:
    try:
        return tokenizer.encode(prompt, add_special_tokens=add_special_tokens, **kwargs)
    except TypeError as error:
        # A tiktoken ``Encoding`` rejects ``add_special_tokens``; it neither adds special tokens
        # nor truncates, and only needs the special tokens the prompt contains to be allowed.
        # Any other ``TypeError`` is an error inside an ordinary tokenizer's encode (a bad
        # option, say) and is reported as it is, not retried with the options dropped.
        if "add_special_tokens" not in str(error):
            raise
        return tokenizer.encode(prompt, allowed_special=TIKTOKEN_ALLOWED_SPECIAL_TOKENS)


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
    private = None
    if truncate_prompt_tokens is not None:
        kwargs = dict(truncation=True, max_length=truncate_prompt_tokens)
        private = _truncating_copy(tokenizer)
    with nvtx_range_debug("tokenize prompt"):
        if private is not None:
            private_tokenizer, lock = private
            with lock:
                return _encode(private_tokenizer, prompt, add_special_tokens, kwargs)
        return _encode(tokenizer, prompt, add_special_tokens, kwargs)
