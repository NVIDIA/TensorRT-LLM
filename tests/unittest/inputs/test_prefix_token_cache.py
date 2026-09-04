# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for tensorrt_llm.inputs.prefix_token_cache.

The cache splices a cached prefix's token ids with a freshly tokenized tail, so
the property that matters is that its output is identical to tokenizing the
whole prompt. The stub tokenizers below have context-sensitive boundaries, so
the seam-straddling case a char-level stub could never expose is exercised.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import patch

import pytest

from tensorrt_llm.inputs.prefix_token_cache import (
    ENABLE_ENV_VAR,
    PrefixTokenCache,
    PrefixTokenCacheConfig,
    create_prefix_token_cache,
    prefix_cache_enabled,
)
from tensorrt_llm.inputs.registry import DefaultInputProcessor
from tensorrt_llm.sampling_params import SamplingParams


def _tok_id(token: str) -> int:
    """Stable across processes, unlike hash() on str."""
    return int.from_bytes(token.encode(), "big") % 100_000 + 1


class _MergeTokenizer:
    """Char-level tokenizer with two-char merges.

    Merges make tokenization context-sensitive at a split point: a tail
    beginning mid-merge tokenizes differently than the same characters do
    inside the whole string. That is the BPE property the cache must survive.
    """

    is_fast = True
    MERGES = frozenset({"ab", "cd", "th", "he", "in", "er"})

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        ids, offsets, i = [], [], 0
        while i < len(text):
            pair = text[i : i + 2]
            token, width = (pair, 2) if pair in self.MERGES else (text[i], 1)
            ids.append(_tok_id(token))
            offsets.append((i, i + width))
            i += width
        return {"input_ids": ids, "offset_mapping": offsets}

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return self(text)["input_ids"]


class _UnsyncableTokenizer:
    """Pathological: every id encodes the token's absolute position.

    A tail tokenized in isolation restarts its position counter, so its ids can
    never equal the cached prefix's ids over the resync span. The cache must
    detect this and tokenize the prompt in full.
    """

    is_fast = True

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        ids, offsets = [], []
        for i in range(0, len(text), 2):
            token = text[i : i + 2]
            ids.append(_tok_id(token) + 1000 * len(ids))
            offsets.append((i, i + len(token)))
        return {"input_ids": ids, "offset_mapping": offsets}


_MIN_CHARS = 16
# A shared opening longer than min_chars, so growing prompts land in the same
# bucket and the prefix lookup can find them.
_PREAMBLE = "the cabinet had inner thread. " * 4


def _cache(**overrides: int) -> PrefixTokenCache:
    config = dict(max_entries=64, overlap=4, resync=2, min_chars=_MIN_CHARS)
    return PrefixTokenCache(PrefixTokenCacheConfig(**{**config, **overrides}))


def _turns(n: int = 12, preamble: str = _PREAMBLE) -> list[str]:
    """Prompts that grow by a small delta, as multi-turn serving does."""
    text = preamble
    out = []
    for i in range(n):
        # Includes non-ASCII so character offsets are exercised too.
        text += f"turn {i}: thé cd inner answer aber {i} thereabouts. "
        out.append(text)
    return out


def _conversations(n: int) -> list[str]:
    """First turns of ``n`` unrelated conversations."""
    return [_turns(1, preamble=f"conversation {i:03d}: {_PREAMBLE}")[0] for i in range(n)]


def test_multi_turn_growth_is_identical_to_full_tokenization() -> None:
    tokenizer, cache = _MergeTokenizer(), _cache()
    for prompt in _turns():
        assert cache.encode(tokenizer, prompt) == tokenizer(prompt)["input_ids"]
    # The whole point: later turns must actually reuse a cached prefix.
    assert cache.hits > 0
    assert cache.resync_failures == 0
    # Each turn replaces the entry it extended, so the conversation is one entry.
    assert len(cache._entries) == 1


def test_interleaved_unrelated_prompts_are_correct() -> None:
    tokenizer, cache = _MergeTokenizer(), _cache()
    unrelated = "a wholly different opening that shares no prefix at all. " * 3
    for prompt in _turns(6):
        assert cache.encode(tokenizer, prompt) == tokenizer(prompt)["input_ids"]
        assert cache.encode(tokenizer, unrelated) == tokenizer(unrelated)["input_ids"]


def test_resync_failure_falls_back_to_full_tokenization() -> None:
    tokenizer, cache = _UnsyncableTokenizer(), _cache()
    for prompt in _turns():
        assert cache.encode(tokenizer, prompt) == tokenizer(prompt)["input_ids"]
    # The fallback must have been exercised, else this proves nothing.
    assert cache.resync_failures > 0


def test_short_prompts_are_tokenized_but_not_cached() -> None:
    tokenizer, cache = _MergeTokenizer(), _cache()
    prompt = "the cab"
    assert len(prompt) < _MIN_CHARS
    assert cache.encode(tokenizer, prompt) == tokenizer(prompt)["input_ids"]
    assert not cache._entries
    assert cache.hits == cache.misses == 0


def test_repeated_prompt_is_not_stored_twice() -> None:
    tokenizer, cache = _MergeTokenizer(), _cache()
    prompt = _turns(1)[0]
    cache.encode(tokenizer, prompt)
    cache.encode(tokenizer, prompt)
    assert len(cache._entries) == 1
    assert cache.hits == 1


def test_eviction_bounds_entry_count() -> None:
    tokenizer, cache = _MergeTokenizer(), _cache(max_entries=8)
    for prompt in _conversations(40):
        cache.encode(tokenizer, prompt)
    assert len(cache._entries) == 8
    assert sum(len(b) for b in cache._buckets.values()) == 8


def test_eviction_bounds_total_chars() -> None:
    prompts = _conversations(40)
    budget = len(prompts[0]) * 3
    tokenizer, cache = _MergeTokenizer(), _cache(max_total_chars=budget)
    for prompt in prompts:
        cache.encode(tokenizer, prompt)
        assert cache._total_chars <= budget
    assert len(cache._entries) == 3


def test_eviction_is_least_recently_used() -> None:
    tokenizer, cache = _MergeTokenizer(), _cache(max_entries=2)
    hot, cold, unrelated = _conversations(3)
    cache.encode(tokenizer, hot)
    cache.encode(tokenizer, cold)
    # Touch the older entry, then insert a third: the untouched one must go.
    assert cache.encode(tokenizer, hot) == tokenizer(hot)["input_ids"]
    cache.encode(tokenizer, unrelated)
    texts = {entry.text for entry in cache._entries.values()}
    assert hot in texts
    assert cold not in texts


def test_concurrent_encode_is_correct() -> None:
    tokenizer, cache = _MergeTokenizer(), _cache()
    prompts = _turns(16)
    expected = {p: tokenizer(p)["input_ids"] for p in prompts}
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for prompt in prompts:
                assert cache.encode(tokenizer, prompt) == expected[prompt]
        except BaseException as exc:  # surface in the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_error_disables_cache_and_still_returns_ids() -> None:
    tokenizer, cache = _MergeTokenizer(), _cache()
    prompt = _turns(1)[0]
    with patch.object(PrefixTokenCache, "_encode", side_effect=RuntimeError("boom")):
        assert cache.encode(tokenizer, prompt) == tokenizer(prompt)["input_ids"]
    assert cache.disabled
    assert cache.encode(tokenizer, prompt) == tokenizer(prompt)["input_ids"]
    assert not cache._entries


@pytest.mark.parametrize(
    "value,expected", [(None, False), ("0", False), ("", False), ("true", False), ("1", True)]
)
def test_enabled_only_for_exactly_one(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool
) -> None:
    if value is None:
        monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(ENABLE_ENV_VAR, value)
    assert prefix_cache_enabled() is expected


def test_config_from_env_reads_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TLLM_PREFIX_TOKEN_CACHE_ENTRIES", "7")
    monkeypatch.setenv("TLLM_PREFIX_TOKEN_CACHE_MIN_CHARS", "100")
    config = PrefixTokenCacheConfig.from_env()
    assert config.max_entries == 7
    assert config.min_chars == 100
    assert config.overlap == PrefixTokenCacheConfig.overlap


@pytest.mark.parametrize("value", ["abc", "-1", "0", ""])
def test_config_from_env_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("TLLM_PREFIX_TOKEN_CACHE_ENTRIES", value)
    with pytest.raises(ValueError, match="TLLM_PREFIX_TOKEN_CACHE_ENTRIES"):
        PrefixTokenCacheConfig.from_env()


def test_create_returns_none_for_slow_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    class Slow:
        is_fast = False

    monkeypatch.setenv(ENABLE_ENV_VAR, "1")
    assert create_prefix_token_cache(Slow()) is None
    assert create_prefix_token_cache(None) is None


def test_create_raises_on_invalid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")
    monkeypatch.setenv("TLLM_PREFIX_TOKEN_CACHE_ENTRIES", "many")
    with pytest.raises(ValueError):
        create_prefix_token_cache(_MergeTokenizer())


# --- DefaultInputProcessor integration --------------------------------------


def _processor(monkeypatch: pytest.MonkeyPatch, enabled: bool = True) -> DefaultInputProcessor:
    if enabled:
        monkeypatch.setenv(ENABLE_ENV_VAR, "1")
        monkeypatch.setenv("TLLM_PREFIX_TOKEN_CACHE_MIN_CHARS", str(_MIN_CHARS))
    else:
        monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
    return DefaultInputProcessor(None, None, _MergeTokenizer())


def test_processor_uses_cache_for_eligible_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = _processor(monkeypatch)
    cache = processor._prefix_token_cache
    assert cache is not None
    params = SamplingParams(add_special_tokens=False)
    for prompt in _turns(4):
        ids, extra = processor({"prompt": prompt}, params)
        assert ids == _MergeTokenizer()(prompt)["input_ids"]
        assert extra is None
    assert cache.hits > 0


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(add_special_tokens=True),
        SamplingParams(add_special_tokens=False, truncate_prompt_tokens=8),
    ],
)
def test_processor_bypasses_cache_when_arguments_alter_tokenization(
    monkeypatch: pytest.MonkeyPatch, params: SamplingParams
) -> None:
    processor = _processor(monkeypatch)
    cache = processor._prefix_token_cache
    prompt = _turns(1)[0]
    ids, _ = processor({"prompt": prompt}, params)
    assert ids == _MergeTokenizer().encode(prompt)
    assert cache.hits == cache.misses == 0


def test_processor_bypasses_cache_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = _processor(monkeypatch, enabled=False)
    assert processor._prefix_token_cache is None
    prompt = _turns(1)[0]
    ids, _ = processor({"prompt": prompt}, SamplingParams(add_special_tokens=False))
    assert ids == _MergeTokenizer().encode(prompt)
