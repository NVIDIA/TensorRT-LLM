# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for tensorrt_llm.inputs.prefix_token_cache.

The cache splices a cached prefix's token ids with a freshly tokenized tail, so
the property that matters is that its output is *identical* to tokenizing the
whole prompt. Splitting a string and tokenizing the tail in isolation is not
generally equal to tokenizing the whole -- BPE merges can straddle the seam --
so these tests use tokenizers whose boundaries are genuinely context-sensitive
rather than a char-level stub that could never expose the bug.

Covers:
- spliced ids are identical to whole-prompt ids across multi-turn growth
- interleaved unrelated prompts stay correct and miss the cache
- a tokenizer that never re-synchronizes falls back to a full tokenization
- LRU eviction bounds the entry count
- concurrent encode() from several threads stays correct
- the feature is off unless TLLM_PREFIX_TOKEN_CACHE == "1"
"""

import os
import threading
from unittest.mock import patch

import pytest

from tensorrt_llm.inputs.prefix_token_cache import (
    PrefixTokenCache,
    get_prefix_token_cache,
    prefix_cache_enabled,
)


def _tok_id(token: str) -> int:
    """Stable across processes, unlike hash() on str."""
    return int.from_bytes(token.encode(), "big") % 100_000 + 1


class MergeTokenizer:
    """Char-level tokenizer with two-char merges.

    The merges make tokenization context-sensitive at a split point: a tail
    beginning mid-merge tokenizes differently than the same characters do
    inside the whole string. That is the BPE property the cache must survive.
    """

    is_fast = True
    MERGES = frozenset({"ab", "cd", "th", "he", "in", "er"})

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
        ids, offsets, i = [], [], 0
        while i < len(text):
            pair = text[i : i + 2]
            token, width = (pair, 2) if pair in self.MERGES else (text[i], 1)
            ids.append(_tok_id(token))
            offsets.append((i, i + width))
            i += width
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


class UnsyncableTokenizer:
    """Pathological: every id encodes the token's absolute position.

    A tail tokenized in isolation restarts its position counter, so its ids can
    never equal the cached prefix's ids over the resync span. This is the case
    the cache must detect and answer by tokenizing the prompt in full -- it
    stands in for a tokenizer whose state genuinely depends on the whole input.
    """

    is_fast = True

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **kwargs):
        ids, offsets = [], []
        for i in range(0, len(text), 2):
            token = text[i : i + 2]
            ids.append(_tok_id(token) + 1000 * len(ids))
            offsets.append((i, i + len(token)))
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


# A shared opening longer than bucket_chars, so growing prompts land in the
# same bucket and the prefix lookup can actually find them.
PREAMBLE = "the cabinet had inner thread. " * 4


def _cache(**kwargs):
    defaults = dict(max_entries=64, overlap=4, resync=2, min_chars=0, bucket_chars=16)
    defaults.update(kwargs)
    return PrefixTokenCache(**defaults)


def _turns(n=12):
    """Prompts that grow by a small delta, as multi-turn serving does."""
    text = PREAMBLE
    out = []
    for i in range(n):
        text += f"turn {i}: theابcd inner answer aber {i} thereabouts. "
        out.append(text)
    return out


def test_multi_turn_growth_is_byte_identical():
    tokenizer, cache = MergeTokenizer(), _cache()
    for prompt in _turns():
        assert cache.encode(tokenizer, prompt) == tokenizer(prompt)["input_ids"]
    # The whole point: later turns must actually reuse a cached prefix.
    assert cache.hits > 0
    assert cache.resync_failures == 0


def test_interleaved_unrelated_prompts_are_correct():
    tokenizer, cache = MergeTokenizer(), _cache()
    unrelated = "a wholly different opening that shares no prefix at all. " * 3
    for prompt in _turns(6):
        assert cache.encode(tokenizer, prompt) == tokenizer(prompt)["input_ids"]
        assert cache.encode(tokenizer, unrelated) == tokenizer(unrelated)["input_ids"]


def test_resync_failure_falls_back_to_full_tokenization():
    tokenizer, cache = UnsyncableTokenizer(), _cache()
    results = [cache.encode(tokenizer, p) for p in _turns()]
    for prompt, ids in zip(_turns(), results):
        assert ids == tokenizer(prompt)["input_ids"]
    # The fallback must have been exercised, else this proves nothing.
    assert cache.resync_failures > 0


def test_eviction_bounds_entry_count():
    tokenizer, cache = MergeTokenizer(), _cache(max_entries=8)
    for prompt in _turns(40):
        cache.encode(tokenizer, prompt)
    assert len(cache._entries) <= 8
    assert len(cache._order) <= 8


def test_concurrent_encode_is_correct():
    tokenizer, cache = MergeTokenizer(), _cache()
    prompts = _turns(16)
    expected = {p: tokenizer(p)["input_ids"] for p in prompts}
    errors = []

    def worker():
        try:
            for prompt in prompts:
                assert cache.encode(tokenizer, prompt) == expected[prompt]
        except Exception as exc:  # surface in the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


@pytest.mark.parametrize(
    "value,expected", [(None, False), ("0", False), ("", False), ("true", False), ("1", True)]
)
def test_enabled_only_for_exactly_one(value, expected):
    env = dict(os.environ)
    env.pop("TLLM_PREFIX_TOKEN_CACHE", None)
    if value is not None:
        env["TLLM_PREFIX_TOKEN_CACHE"] = value
    with patch.dict(os.environ, env, clear=True):
        assert prefix_cache_enabled() is expected


def test_get_prefix_token_cache_is_a_singleton():
    assert get_prefix_token_cache() is get_prefix_token_cache()
